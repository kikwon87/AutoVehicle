"""Frenet-frame trajectory generation: a lattice of candidates, then a choice.

This is the Werling-style planner: sample a set of lateral and longitudinal
polynomials in the road frame, convert them to world coordinates, discard the
infeasible and colliding ones, and rank the rest.

Why a lattice at all, when an MPC follows?  Lecture 2/3, *Constraints Linearize
Too*:

    The nominal trajectory chooses the local homotopy class -- which side of the
    obstacle to pass.  The convex subproblem does not explore every side.
    Initialization therefore selects the manoeuvre, and a bad initial guess
    produces a locally optimal but strategically wrong plan.

The lattice is that initialization.  It searches *discretely* over manoeuvres --
pass left, pass right, slow down and stay -- which a gradient-based solver
cannot do, and hands the winner to the MPC as a warm start and a reference.

Both polynomial families are minimum-jerk in their own coordinate, which is why
quintics appear: a quintic is the minimum-jerk connection between two fully
specified endpoint states, and a quartic is the minimum-jerk connection when the
endpoint *position* is free (velocity keeping).

Collision checking is done at the sample times only.  That is the exact
weakness the lecture names -- "constraints hold at the nodes, only there" -- so
the check inflates each obstacle by half the distance it can travel between
samples, which is the cheap version of the "tightening" remedy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..core.conventions import wrap_to_pi
from ..core.geometry import circle_centres, ellipse_clearance, multi_circle_cover
from ..models.params import VehicleParams
from ..world.path import ReferencePath
from .prediction import Prediction


def quintic(x0: float, dx0: float, ddx0: float, x1: float, dx1: float, ddx1: float, T: float):
    """Coefficients of the minimum-jerk quintic between two full endpoint states."""
    a0, a1, a2 = x0, dx0, 0.5 * ddx0
    T2, T3, T4, T5 = T * T, T**3, T**4, T**5
    A = np.array([[T3, T4, T5], [3 * T2, 4 * T3, 5 * T4], [6 * T, 12 * T2, 20 * T3]])
    b = np.array([x1 - (a0 + a1 * T + a2 * T2), dx1 - (a1 + 2 * a2 * T), ddx1 - 2 * a2])
    a3, a4, a5 = np.linalg.solve(A, b)
    return np.array([a0, a1, a2, a3, a4, a5])


def quartic(x0: float, dx0: float, ddx0: float, dx1: float, ddx1: float, T: float):
    """Minimum-jerk quartic with a free endpoint position (velocity keeping)."""
    a0, a1, a2 = x0, dx0, 0.5 * ddx0
    A = np.array([[3 * T * T, 4 * T**3], [6 * T, 12 * T * T]])
    b = np.array([dx1 - (a1 + 2 * a2 * T), ddx1 - 2 * a2])
    a3, a4 = np.linalg.solve(A, b)
    return np.array([a0, a1, a2, a3, a4])


def _poly(c: np.ndarray, t: np.ndarray, order: int = 0) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    for _ in range(order):
        c = c[1:] * np.arange(1, len(c))
        if len(c) == 0:
            return np.zeros_like(t)
    return sum(ci * t**i for i, ci in enumerate(c))


@dataclass
class Trajectory:
    """A candidate trajectory, in both frames, with its cost breakdown."""

    t: np.ndarray
    s: np.ndarray
    e_y: np.ndarray
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    v: np.ndarray
    a: np.ndarray
    kappa: np.ndarray
    cost: float = np.inf
    cost_terms: dict = field(default_factory=dict)
    feasible: bool = True
    reject_reason: str = ""
    target_offset: float = 0.0
    horizon: float = 0.0

    def states(self) -> np.ndarray:
        """``(N, 4)`` array of ``[X, Y, psi, v]`` -- the MPC's reference."""
        return np.column_stack([self.x, self.y, self.psi, self.v])

    def as_reference(self, times: np.ndarray) -> np.ndarray:
        """Resample onto another time grid, extrapolating past the horizon.

        ``np.interp`` clamps beyond the last sample, which would place every
        reference point past the horizon at the *same* position.  An MPC whose
        horizon outlasts the trajectory then reads that frozen point as "come
        to a stop here" and brakes -- a decelerating vehicle with no reason for
        it anywhere in the logs.  Past the end the reference therefore
        continues in a straight line at the final speed and heading.
        """
        times = np.asarray(times, dtype=float)
        psi_un = np.unwrap(self.psi)
        x = np.interp(times, self.t, self.x)
        y = np.interp(times, self.t, self.y)
        psi = np.interp(times, self.t, psi_un)
        v = np.interp(times, self.t, self.v)

        over = times > self.t[-1]
        if over.any():
            dt = times[over] - self.t[-1]
            x[over] = self.x[-1] + self.v[-1] * np.cos(psi_un[-1]) * dt
            y[over] = self.y[-1] + self.v[-1] * np.sin(psi_un[-1]) * dt
            psi[over] = psi_un[-1]
            v[over] = self.v[-1]
        return np.column_stack([x, y, wrap_to_pi(psi), v])


@dataclass
class LatticeConfig:
    """Sampling and cost weights for the candidate set."""

    #: Lateral offsets sampled *relative to* the behaviour layer's target.  The
    #: range must span a full lane change, otherwise the lattice can never
    #: represent "go around" and will report every blocked lane as a hard stop.
    #: Candidates leaving the corridor passed to :meth:`FrenetLatticePlanner.plan`
    #: are rejected, so widening the sampling does not widen what is allowed.
    lateral_offsets: tuple[float, ...] = (-3.5, -1.75, -0.9, -0.4, 0.0, 0.4, 0.9, 1.75, 3.5)
    #: Manoeuvre durations sampled during ordinary driving.
    #:
    #: The short end is deliberately absent here.  Risk is evaluated only over a
    #: candidate's own horizon, so a 1.5 s candidate at 15 m/s stops 15 m short
    #: of an obstacle, never sees it, and wins on the time term -- the planner
    #: becomes myopic exactly where it must not be.  Short horizons are supplied
    #: by the caller (see ``AutonomyConfig.manoeuvre_min_horizon``) only while a
    #: *commanded* manoeuvre has a deadline to meet.
    horizons: tuple[float, ...] = (2.5, 3.5, 4.5)
    speed_samples: tuple[float, ...] = (-2.0, -1.0, 0.0)
    dt: float = 0.1

    w_jerk_lat: float = 0.6
    w_jerk_lon: float = 0.2
    w_offset: float = 4.0
    w_speed: float = 2.5
    w_time: float = 0.4
    w_risk: float = 60.0
    #: Penalty on changing the chosen lateral offset between replans.
    #:
    #: Candidates a few tenths of a metre apart routinely score within noise of
    #: each other, so without this the planner picks a different one every tick.
    #: The reference then jerks sideways at the replan rate, the MPC chases it,
    #: and a manoeuvre that is comfortable in any single plan violates the
    #: lateral-acceleration constraint in the sequence of them.
    #:
    #: It must stay **below** ``w_offset``: its job is to break ties between
    #: candidates a few tenths apart, not to argue with a commanded manoeuvre.
    #: At 25 against an offset weight of 4 it vetoes every lane change, and the
    #: vehicle drives into the obstacle it was told to go around.
    w_consistency: float = 2.0

    a_lat_limit: float = 4.4
    a_lon_max: float = 3.0
    a_lon_min: float = -6.0
    kappa_max: float = 0.25
    #: how many sigma of prediction uncertainty to clear
    inflate_sigma: float = 1.0


class FrenetLatticePlanner:
    """Sample, filter and rank Frenet trajectories against predictions."""

    def __init__(self, params: VehicleParams, config: LatticeConfig | None = None, n_circles: int = 3):
        self.p = params
        self.cfg = config or LatticeConfig()
        self.cfg.kappa_max = min(self.cfg.kappa_max, np.tan(params.actuator.delta_max) / params.L)
        self.n_circles = int(n_circles)
        self._ego_offsets, self._ego_radius = multi_circle_cover(
            params.length, params.width, self.n_circles
        )
        #: the ego box is centred ahead of the rear axle, not on it
        self._ego_body_offset = 0.5 * params.length - params.rear_overhang
        self.last_candidates: list[Trajectory] = []
        self.previous_offset: float | None = None

    # --- generation ----------------------------------------------------------

    def _to_cartesian(self, path: ReferencePath, s: np.ndarray, e_y: np.ndarray, de_y: np.ndarray, ds: np.ndarray):
        """Frenet to world, vectorized over the whole candidate.

        The heading offset comes from ``de_y/ds`` and the speed from both
        components of the Frenet velocity, with the denominator
        ``1 - kappa e_y`` floored -- the Frenet map is singular where it
        vanishes, and a candidate that reaches there is not merely inaccurate,
        it is meaningless.
        """
        px, py, th, kap = path.frames(s)
        sin_th, cos_th = np.sin(th), np.cos(th)
        x = px - e_y * sin_th
        y = py + e_y * cos_th
        den = np.maximum(1.0 - kap * e_y, 1e-3)
        de_ds = de_y / np.maximum(ds, 1e-3)
        psi = wrap_to_pi(th + np.arctan2(de_ds, den))
        v = np.hypot(ds * den, de_y)
        return x, y, psi, v

    def generate(
        self,
        path: ReferencePath,
        s0: float,
        e_y0: float,
        de_y0: float,
        dde_y0: float,
        v0: float,
        target_speed: float,
        target_offset: float = 0.0,
        stop_s: float | None = None,
        corridor: tuple[float, float] | None = None,
        a0: float = 0.0,
        horizons: Sequence[float] | None = None,
    ) -> list[Trajectory]:
        """Build the candidate set for the current situation.

        ``corridor`` is the ``(e_y_min, e_y_max)`` box the trajectory must stay
        inside -- the road boundary expressed in Frenet coordinates, which is
        the whole reason the Frenet frame was adopted.

        ``a0`` is the longitudinal acceleration the plan starts from.  Leaving
        it at zero -- as a naive minimum-jerk formulation does -- means every
        replan begins with no deceleration, and since only the first step of
        each plan is ever executed, the vehicle can never brake harder than the
        first derivative of a curve that starts flat.  Against a leader braking
        at 4 m/s^2 that is the difference between stopping and not.
        """
        cfg = self.cfg
        out: list[Trajectory] = []
        for T in (horizons if horizons is not None else cfg.horizons):
            t = np.arange(0.0, T + 1e-9, cfg.dt)
            for off in cfg.lateral_offsets:
                e_target = target_offset + off
                cl = quintic(e_y0, de_y0, dde_y0, e_target, 0.0, 0.0, T)
                e_y = _poly(cl, t)
                de_y = _poly(cl, t, 1)
                jerk_lat = float(np.mean(_poly(cl, t, 3) ** 2))

                lon_specs = []
                if stop_s is not None:
                    s_end = max(stop_s, s0)
                    lon_specs.append(("stop", quintic(s0, v0, a0, s_end, 0.0, 0.0, T)))
                # Clamp each sampled target speed to what is reachable in T at
                # the acceleration limits.  Without this, a vehicle pulling away
                # from a stop samples only targets it cannot reach, every
                # candidate is rejected as over-accelerating, and the planner
                # reports "no plan" on an empty road.
                # A minimum-jerk speed change peaks at 1.5x its average:
                # v(tau) = v0 + dv (3 tau^2 - 2 tau^3) has |v'|_max = 1.5 dv / T.
                # Clamping to the average alone still samples targets whose peak
                # acceleration violates the limit, so every candidate is
                # rejected -- which is what makes a stopped vehicle unable to
                # find any plan and never pull away.
                peak = 1.5
                v_hi = v0 + cfg.a_lon_max * T / peak
                v_lo = max(v0 + cfg.a_lon_min * T / peak, 0.0)
                if a0 < 0.0:  # already braking: that speed is reachable too
                    v_lo = max(v0 + (cfg.a_lon_min + a0) * T / (2 * peak), 0.0)
                for dv in cfg.speed_samples:
                    v_t = float(np.clip(target_speed + dv, v_lo, v_hi))
                    lon_specs.append(("keep", quartic(s0, v0, a0, max(v_t, 0.0), 0.0, T)))

                for kind, cs in lon_specs:
                    s = _poly(cs, t)
                    ds = _poly(cs, t, 1)
                    dds = _poly(cs, t, 2)
                    jerk_lon = float(np.mean(_poly(cs, t, 3) ** 2))
                    if np.any(ds < -0.5):
                        continue  # no reversing in these manoeuvres
                    ds = np.maximum(ds, 0.0)

                    x, y, psi, v = self._to_cartesian(path, s, e_y, de_y, ds)
                    kappa = self._path_curvature(x, y, psi, v, t)
                    a = np.gradient(v, t)

                    traj = Trajectory(
                        t=t, s=s, e_y=e_y, x=x, y=y, psi=psi, v=v, a=a, kappa=kappa,
                        target_offset=e_target, horizon=T,
                    )
                    traj.cost_terms = {
                        "jerk_lat": cfg.w_jerk_lat * jerk_lat,
                        "jerk_lon": cfg.w_jerk_lon * jerk_lon,
                        "offset": cfg.w_offset * (e_target - target_offset) ** 2,
                        "speed": cfg.w_speed * float(np.mean((v - target_speed) ** 2)),
                        "time": cfg.w_time * T,
                        "consistency": (
                            0.0 if self.previous_offset is None
                            else cfg.w_consistency * (e_target - self.previous_offset) ** 2
                        ),
                    }
                    traj.reject_reason = self._feasibility(traj, corridor)
                    traj.feasible = traj.reject_reason == ""
                    out.append(traj)
        return out

    @staticmethod
    def _path_curvature(x, y, psi, v, t) -> np.ndarray:
        """``kappa = dpsi/ds``, computed from the unwrapped heading."""
        dpsi = np.gradient(np.unwrap(psi), t)
        return dpsi / np.maximum(v, 0.3)

    #: Below this speed the path-curvature of a trajectory is not a meaningful
    #: quantity -- ``kappa = dpsi/ds`` divides by a vanishing ``ds`` -- so the
    #: geometric checks are skipped.  A stopped vehicle has no curvature, and
    #: rejecting a stop because its numerical curvature is large is a bug, not
    #: a safety feature.
    CURVATURE_SPEED_FLOOR = 0.8

    def _feasibility(self, traj: Trajectory, corridor: tuple[float, float] | None = None) -> str:
        cfg = self.cfg
        if corridor is not None:
            lo, hi = corridor
            if traj.e_y.min() < lo - 1e-6 or traj.e_y.max() > hi + 1e-6:
                return (
                    f"leaves the corridor: e_y in [{traj.e_y.min():.2f}, "
                    f"{traj.e_y.max():.2f}] vs [{lo:.2f}, {hi:.2f}]"
                )
        moving = traj.v > self.CURVATURE_SPEED_FLOOR
        if moving.any():
            kap = np.abs(traj.kappa[moving])
            if kap.max() > cfg.kappa_max:
                return f"curvature {kap.max():.3f} > {cfg.kappa_max:.3f}"
            a_lat = traj.v[moving] ** 2 * kap
            if a_lat.max() > cfg.a_lat_limit:
                return f"a_y {a_lat.max():.2f} > {cfg.a_lat_limit:.2f}"
        if np.max(traj.a) > cfg.a_lon_max + 1e-6 or np.min(traj.a) < cfg.a_lon_min - 1e-6:
            return f"a_x range [{np.min(traj.a):.2f}, {np.max(traj.a):.2f}] outside limits"
        return ""

    # --- evaluation ----------------------------------------------------------

    def risk(self, traj: Trajectory, predictions: Sequence[Prediction]) -> tuple[float, bool]:
        """Probability-weighted proximity cost, and whether the path is blocked.

        Both vehicles are covered by :func:`multi_circle_cover` circles, and the
        required separation is an **oriented ellipse** whose semi-axes are the
        summed circle radii plus ``inflate_sigma`` of the prediction's
        longitudinal and lateral uncertainty.

        Collision is checked at the sample times only -- the exact weakness the
        lecture names, "constraints hold at the nodes, only there".  The remedy
        applied here is tightening: each obstacle is grown by half the distance
        it travels between samples, which bounds the gap a node-only check
        would otherwise miss.
        """
        if not predictions:
            return 0.0, False
        total, blocked = 0.0, False
        for pred in predictions:
            obj_off, obj_r = multi_circle_cover(pred.length, pred.width, self.n_circles)
            step = float(np.mean(np.diff(pred.times))) if len(pred.times) > 1 else 0.1
            speed = float(np.linalg.norm(pred.positions[1] - pred.positions[0]) / max(step, 1e-6))
            inter_sample = 0.5 * speed * self.cfg.dt
            sig_lon, sig_lat = pred.ellipse_axes(self.cfg.inflate_sigma)

            for i, ti in enumerate(traj.t):
                if ti > pred.times[-1]:
                    break
                obj_c = pred.position_at(float(ti))
                obj_h = float(np.interp(ti, pred.times, np.unwrap(pred.headings)))
                a = self._ego_radius + obj_r + inter_sample + float(np.interp(ti, pred.times, sig_lon))
                b = self._ego_radius + obj_r + inter_sample + float(np.interp(ti, pred.times, sig_lat))

                ego_pts = circle_centres(
                    traj.x[i], traj.y[i], traj.psi[i], self._ego_offsets, self._ego_body_offset
                )
                obj_pts = circle_centres(obj_c[0], obj_c[1], obj_h, obj_off)
                worst = np.inf
                for ep in ego_pts:
                    for op in obj_pts:
                        worst = min(worst, ellipse_clearance(ep - op, obj_h, a, b))
                if worst < 0.0:
                    blocked = True
                    total += pred.probability * (1.0 - worst) ** 2 * 100.0
                elif worst < 0.5:
                    total += pred.probability * (0.5 - worst) ** 2 * 10.0
        return total, blocked

    def plan(
        self,
        path: ReferencePath,
        s0: float,
        e_y0: float,
        de_y0: float,
        dde_y0: float,
        v0: float,
        target_speed: float,
        predictions: Sequence[Prediction] = (),
        target_offset: float = 0.0,
        stop_s: float | None = None,
        corridor: tuple[float, float] | None = None,
        a0: float = 0.0,
        horizons: Sequence[float] | None = None,
    ) -> Trajectory | None:
        """Return the lowest-cost feasible, unblocked candidate, or ``None``.

        ``horizons`` overrides the sampled durations.  A *commanded* manoeuvre --
        a lane change, not a lane-keeping correction -- must finish by a fixed
        time, not by a horizon that recedes with every replan.  Re-planning a
        3 s convergence every 0.1 s converges geometrically, not in 3 s, and a
        lane change ordered 70 m before an obstacle is then still half-done when
        the vehicle arrives.

        ``None`` is a real answer: it means every sampled manoeuvre is either
        dynamically infeasible or collides.  The caller must then fall back --
        typically to a maximum-braking stop -- rather than being handed a
        trajectory that quietly violates something.
        """
        cands = self.generate(
            path, s0, e_y0, de_y0, dde_y0, v0, target_speed, target_offset, stop_s, corridor,
            a0, horizons
        )
        best, best_cost = None, np.inf
        for c in cands:
            risk, blocked = self.risk(c, predictions)
            c.cost_terms["risk"] = self.cfg.w_risk * risk
            c.cost = float(sum(c.cost_terms.values()))
            if blocked:
                c.feasible = False
                if not c.reject_reason:
                    c.reject_reason = "collision with a predicted occupancy"
            if c.feasible and c.cost < best_cost:
                best, best_cost = c, c.cost
        self.last_candidates = cands
        self.previous_offset = None if best is None else float(best.target_offset)
        return best

    def reset(self) -> None:
        """Forget the previous manoeuvre, e.g. between scenario runs."""
        self.previous_offset = None
        self.last_candidates = []

    def emergency_stop(
        self,
        path: ReferencePath,
        s0: float,
        e_y0: float,
        de_y0: float,
        v0: float,
        decel: float | None = None,
        horizon: float = 4.0,
    ) -> Trajectory:
        """The fallback when :meth:`plan` finds nothing: brake in a straight line.

        Returned unconditionally and without a feasibility check, because it is
        what the vehicle does when no plan exists.  It is still a *plan* -- the
        MPC tracks it -- rather than an open-loop actuator command, so the
        controller keeps its constraints and the KPI layer keeps its record.
        """
        decel = abs(self.cfg.a_lon_min if decel is None else decel)
        t = np.arange(0.0, horizon + 1e-9, self.cfg.dt)
        t_stop = v0 / max(decel, 1e-3)
        v = np.maximum(v0 - decel * t, 0.0)
        s = s0 + np.where(t < t_stop, v0 * t - 0.5 * decel * t**2, 0.5 * v0 * t_stop)
        e_y = e_y0 + de_y0 * np.minimum(t, t_stop) * 0.0  # hold the current offset
        e_y = np.full_like(t, e_y0)
        ds = v.copy()
        x, y, psi, vv = self._to_cartesian(path, s, e_y, np.zeros_like(t), ds)
        kappa = self._path_curvature(x, y, psi, np.maximum(vv, 0.3), t)
        return Trajectory(
            t=t, s=s, e_y=e_y, x=x, y=y, psi=psi, v=vv, a=np.gradient(vv, t), kappa=kappa,
            cost=np.inf, feasible=True, reject_reason="emergency fallback",
            target_offset=e_y0, horizon=horizon,
        )
