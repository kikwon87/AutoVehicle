"""Key performance indicators, with thresholds and an explicit pass/fail.

A number without a threshold is not a test.  Every KPI here carries the value,
the unit, the bound it is judged against and the direction of that bound, so a
scenario report says *whether the run was acceptable* rather than leaving the
reader to decide from a wall of statistics.

Six groups:

``safety``      collisions, clearance, TTC, red-light compliance, corridor
                departure, tire friction usage, rollover margin
``tracking``    cross-track, heading and speed error against the reference
``comfort``     accelerations, jerk and steering rate
``progress``    distance covered, mean speed, whether the goal was reached
``compute``     MPC solve time, real-time factor, solver success, fallback use
``perception``  recall against what was actually visible, and track continuity

The perception group is measured against the sensor's **visibility**, not
against all ground truth.  Counting an occluded vehicle as a perception miss
conflates "the sensor could not see it" with "the tracker dropped it", and only
the second is a bug in this code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from ..autonomy.stack import Telemetry
from ..core.geometry import polygon_distance, time_to_collision
from ..models.params import VehicleParams
from ..models.suspension import rollover_index
from ..world.path import ReferencePath
from ..world.traffic_light import SignalState, TrafficLightController
from ..world.world import WorldSnapshot


@dataclass
class KPI:
    name: str
    value: float
    unit: str
    threshold: float | None
    direction: str          #: "max" (value <= threshold) or "min" (value >= threshold)
    group: str
    description: str = ""

    @property
    def passed(self) -> bool:
        if self.threshold is None or not np.isfinite(self.value):
            return True
        return self.value <= self.threshold if self.direction == "max" else self.value >= self.threshold

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        bound = "" if self.threshold is None else f"  ({self.direction} {self.threshold:g})"
        return f"[{mark}] {self.name:34s} {self.value:12.4g} {self.unit:8s}{bound}"


@dataclass
class KPIThresholds:
    """Acceptance bounds. Defaults target ordinary urban driving, not racing."""

    max_collisions: float = 0.0
    min_clearance: float = 0.30            # m
    min_ttc: float = 1.2                   # s
    max_red_light_violations: float = 0.0
    max_corridor_exit_time: float = 0.5    # s
    max_friction_usage: float = 0.95       # of the friction ellipse
    max_rollover_index: float = 0.7
    max_cross_track_rms: float = 0.35      # m, whole run including transients
    #: RMS after the first ``settle_time`` seconds.  A whole-run RMS made of one
    #: initial transient and a whole-run RMS made of a permanent bias are the
    #: same number and call for different fixes; reporting both separates them.
    max_cross_track_rms_settled: float = 0.15  # m
    settle_time: float = 5.0               # s
    max_cross_track_peak: float = 0.80     # m
    max_heading_rms: float = 0.08          # rad
    max_speed_rms: float = 1.5             # m/s
    max_lat_accel: float = 4.5             # m/s^2
    max_lon_accel: float = 3.5             # m/s^2
    max_jerk_rms: float = 3.0              # m/s^3
    max_steer_rate: float = 0.75           # rad/s
    min_mean_speed: float = 0.0            # m/s
    max_solve_time_p95: float = 0.07       # s, against a 100 ms control period
    max_real_time_factor: float = 1.0
    min_solver_success: float = 0.90       # fraction
    max_fallback_fraction: float = 0.10
    min_detection_recall: float = 0.75


@dataclass
class KPIReport:
    scenario: str
    kpis: list[KPI] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(k.passed for k in self.kpis)

    @property
    def failures(self) -> list[KPI]:
        return [k for k in self.kpis if not k.passed]

    def get(self, name: str) -> KPI | None:
        return next((k for k in self.kpis if k.name == name), None)

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "passed": self.passed,
            "kpis": [asdict(k) | {"passed": k.passed} for k in self.kpis],
            "extra": self.extra,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=float)

    def to_table(self, groups: Sequence[str] | None = None) -> str:
        lines = [f"=== {self.scenario} === {'PASS' if self.passed else 'FAIL'}"]
        for g in groups or ["safety", "tracking", "comfort", "progress", "compute", "perception"]:
            rows = [k for k in self.kpis if k.group == g]
            if not rows:
                continue
            lines.append(f"  -- {g} --")
            lines.extend("    " + str(k) for k in rows)
        return "\n".join(lines)


def compute_kpis(
    scenario: str,
    history: Sequence[WorldSnapshot],
    telemetry: Sequence[Telemetry],
    route: ReferencePath,
    params: VehicleParams,
    thresholds: KPIThresholds | None = None,
    dt: float = 0.02,
    wall_time: float | None = None,
    goal_s: float | None = None,
    stop_line_s: float | None = None,
    signal_group: str | None = None,
    lights: TrafficLightController | None = None,
    corridor_half_width: float = 1.75,
    visible_counts: tuple[int, int] | None = None,
) -> KPIReport:
    """Reduce a run to a report.

    ``visible_counts`` is ``(detected, visible)`` accumulated by the runner from
    the sensor's own visibility test; without it the perception group is
    omitted rather than guessed at.
    """
    th = thresholds or KPIThresholds()
    rep = KPIReport(scenario=scenario)
    add = rep.kpis.append

    if not history:
        add(KPI("no_data", 1.0, "", 0.0, "max", "safety", "the run produced no samples"))
        return rep

    t = np.array([h.t for h in history])
    a_x = np.array([h.diagnostics["a_x"] for h in history])
    a_y = np.array([h.diagnostics["a_y"] for h in history])
    usage = np.array(
        [max(h.diagnostics["usage_front"], h.diagnostics["usage_rear"]) for h in history]
    )
    delta = np.array([h.diagnostics["delta_actual"] for h in history])
    speed = np.array([h.ego_state.v for h in history])

    # --- safety --------------------------------------------------------------
    collisions = 0
    min_clear = np.inf
    min_ttc = np.inf
    for h in history:
        ego = h.ego_state
        for a in h.actors:
            d = polygon_distance(ego.corners(), a.corners())
            min_clear = min(min_clear, d)
            if d <= 0.0:
                collisions += 1
            r = 0.25 * (params.length + params.width + a.length + a.width)
            min_ttc = min(min_ttc, time_to_collision(ego.position, ego.velocity, a.position, a.velocity, r))

    add(KPI("collision_samples", float(collisions), "steps", th.max_collisions, "max", "safety",
            "simulation steps in which the ego box overlapped another actor"))
    add(KPI("min_clearance", float(min_clear), "m", th.min_clearance, "min", "safety",
            "smallest box-to-box distance to any actor"))
    add(KPI("min_ttc", float(min_ttc), "s", th.min_ttc, "min", "safety",
            "smallest constant-velocity time to collision"))

    # Frenet corridor departure
    s_hist, e_hist = [], []
    s_guess = None
    for h in history:
        z = h.ego
        xr = z[0] - params.l_r * np.cos(z[2])
        yr = z[1] - params.l_r * np.sin(z[2])
        s_guess = route.project(xr, yr, s_guess)
        s_hist.append(s_guess)
        e_hist.append(route.lateral_offset(xr, yr, s_guess))
    s_hist = np.array(s_hist)
    e_hist = np.array(e_hist)
    outside = np.abs(e_hist) > corridor_half_width
    add(KPI("corridor_exit_time", float(outside.sum() * dt), "s", th.max_corridor_exit_time, "max",
            "safety", "time spent outside the lane corridor"))
    add(KPI("max_friction_usage", float(usage.max()), "-", th.max_friction_usage, "max", "safety",
            "peak point on the friction ellipse, over both axles"))
    add(KPI("max_rollover_index", float(max(rollover_index(v, params) for v in a_y)), "-",
            th.max_rollover_index, "max", "safety", "|a_y| / (g * static stability factor)"))

    if stop_line_s is not None and lights is not None and signal_group is not None:
        crossed_on_red = 0
        for i in range(1, len(s_hist)):
            if s_hist[i - 1] <= stop_line_s < s_hist[i]:
                if lights.state(signal_group, float(t[i])) is SignalState.RED:
                    crossed_on_red += 1
        add(KPI("red_light_violations", float(crossed_on_red), "count",
                th.max_red_light_violations, "max", "safety", "stop line crossed while red"))

    # --- tracking ------------------------------------------------------------
    add(KPI("cross_track_rms", float(np.sqrt(np.mean(e_hist**2))), "m", th.max_cross_track_rms,
            "max", "tracking", "RMS lateral offset from the route centerline"))
    add(KPI("cross_track_peak", float(np.abs(e_hist).max()), "m", th.max_cross_track_peak,
            "max", "tracking", "peak lateral offset"))
    settled = t >= th.settle_time
    if settled.sum() > 10:
        add(KPI("cross_track_rms_settled", float(np.sqrt(np.mean(e_hist[settled] ** 2))), "m",
                th.max_cross_track_rms_settled, "max", "tracking",
                f"RMS lateral offset after the first {th.settle_time:g} s"))
    psi_err = np.array(
        [float(route.heading_error(h.ego[2], s)) for h, s in zip(history, s_hist)]
    )
    add(KPI("heading_error_rms", float(np.sqrt(np.mean(psi_err**2))), "rad", th.max_heading_rms,
            "max", "tracking", "RMS heading error against the route tangent"))
    if telemetry:
        v_ref = np.array([r.ref_speed for r in telemetry])
        v_act = np.array([r.v for r in telemetry])
        add(KPI("speed_error_rms", float(np.sqrt(np.mean((v_act - v_ref) ** 2))), "m/s",
                th.max_speed_rms, "max", "tracking", "RMS speed error against the behaviour target"))

    # --- comfort -------------------------------------------------------------
    jerk = np.gradient(a_x, t) if len(t) > 2 else np.zeros_like(a_x)
    steer_rate = np.gradient(delta, t) if len(t) > 2 else np.zeros_like(delta)
    add(KPI("max_lat_accel", float(np.abs(a_y).max()), "m/s^2", th.max_lat_accel, "max", "comfort",
            "peak lateral acceleration at the CG"))
    add(KPI("max_lon_accel", float(np.abs(a_x).max()), "m/s^2", th.max_lon_accel, "max", "comfort",
            "peak longitudinal acceleration"))
    add(KPI("jerk_rms", float(np.sqrt(np.mean(jerk**2))), "m/s^3", th.max_jerk_rms, "max", "comfort",
            "RMS longitudinal jerk"))
    add(KPI("max_steer_rate", float(np.abs(steer_rate).max()), "rad/s", th.max_steer_rate, "max",
            "comfort", "peak road-wheel steering rate"))

    # --- progress ------------------------------------------------------------
    progress = float(s_hist[-1] - s_hist[0])
    add(KPI("distance_travelled", progress, "m", None, "min", "progress", "arc length covered"))
    add(KPI("mean_speed", float(speed.mean()), "m/s", th.min_mean_speed, "min", "progress",
            "mean speed over the run"))
    if goal_s is not None:
        add(KPI("goal_reached", 1.0 if s_hist[-1] >= goal_s else 0.0, "bool", 1.0, "min",
                "progress", f"route arc length {goal_s:.1f} m reached"))
        reach = np.where(s_hist >= goal_s)[0]
        add(KPI("time_to_goal", float(t[reach[0]]) if len(reach) else float("inf"), "s", None,
                "max", "progress", "first time the goal arc length was reached"))

    # --- compute -------------------------------------------------------------
    if telemetry:
        st = np.array([r.mpc_time for r in telemetry])
        add(KPI("solve_time_mean", float(st.mean()), "s", None, "max", "compute",
                "mean MPC solve time"))
        add(KPI("solve_time_p95", float(np.percentile(st, 95)), "s", th.max_solve_time_p95, "max",
                "compute", "95th-percentile MPC solve time"))
        add(KPI("solve_time_max", float(st.max()), "s", None, "max", "compute", "worst MPC solve"))
        control_dt = float(np.median(np.diff([r.t for r in telemetry]))) if len(telemetry) > 1 else 0.1
        add(KPI("real_time_factor", float(st.mean() / max(control_dt, 1e-9)), "-",
                th.max_real_time_factor, "max", "compute",
                "mean solve time divided by the control period; must be below 1"))
        # Ticks below walking pace are excluded.  At rest the prediction model
        # is degenerate -- the steering column of B vanishes, and the augmented
        # Lagrangian fights the v >= 0 bound -- so a "failure" there is a known
        # model limitation rather than a solver defect, and counting it hides
        # the solver's behaviour on the problems it is meant to solve.
        skipped = ("skipped_at_standstill", "launch_below_min_speed")
        moving = [r for r in telemetry if r.v >= 0.5 and r.mpc_status not in skipped]
        if moving:
            ok = np.mean([r.mpc_status == "converged" and r.mpc_violation < 5e-2 for r in moving])
            add(KPI("solver_success_rate", float(ok), "-", th.min_solver_success, "min",
                    "compute",
                    "fraction of *moving* ticks where the MPC converged within its constraints"))
        add(KPI("standstill_fraction", 1.0 - len(moving) / len(telemetry), "-", None, "max",
                "compute", "fraction of ticks below 0.5 m/s, where the model is degenerate"))
        add(KPI("fallback_plan_fraction", float(np.mean([r.used_fallback_plan for r in telemetry])),
                "-", th.max_fallback_fraction, "max", "compute",
                "ticks where the lattice found no feasible candidate"))
        add(KPI("fallback_control_fraction",
                float(np.mean([r.used_fallback_control for r in telemetry])), "-",
                th.max_fallback_fraction, "max", "compute",
                "ticks where the MPC was replaced by the geometric controller"))
        if wall_time is not None:
            add(KPI("sim_wall_ratio", float(wall_time / max(t[-1], 1e-9)), "-", None, "max",
                    "compute", "wall-clock seconds per simulated second, whole stack"))

    # --- perception ----------------------------------------------------------
    if visible_counts is not None and visible_counts[1] > 0:
        detected, visible = visible_counts
        add(KPI("detection_recall", detected / visible, "-", th.min_detection_recall, "min",
                "perception", "tracked fraction of the objects the sensor could actually see"))
    if telemetry:
        add(KPI("mean_tracks", float(np.mean([r.n_tracks for r in telemetry])), "count", None,
                "min", "perception", "mean number of confirmed tracks"))

    rep.extra = {
        "s": s_hist.tolist(),
        "e_y": e_hist.tolist(),
        "t": t.tolist(),
        "behaviors": [r.behavior for r in telemetry],
    }
    return rep


def summarize(reports: Sequence[KPIReport]) -> str:
    """One line per scenario, plus the failing KPIs underneath."""
    lines = []
    n_pass = sum(r.passed for r in reports)
    lines.append(f"{n_pass}/{len(reports)} scenarios passed")
    for r in reports:
        lines.append(f"  {'PASS' if r.passed else 'FAIL'}  {r.scenario}")
        for k in r.failures:
            bound = f"{k.direction} {k.threshold:g}"
            lines.append(f"          {k.name} = {k.value:.4g} {k.unit} (needs {bound})")
    return "\n".join(lines)
