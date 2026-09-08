"""One run: build the world, drive it, and reduce it to numbers and frames.

A session owns everything that must be identical across the controllers being
compared -- the world, the traffic seed, **and the perception** -- and hands the
controller only an :class:`~avsim.platform.controller_api.Observation`.  Running
a separate sensor per controller would make a comparison a comparison of two
perception draws.

The loop is two-rate, as everywhere else in this package: the plant integrates
at ``plant_dt`` while the controller runs at ``control_dt`` and its command is
held in between.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from ..autonomy.stack import AutonomyConfig
from ..control.mpc import MPCConfig, VehicleMPC
from ..core.conventions import wrap_to_pi
from ..core.geometry import polygon_distance, time_to_collision
from ..models.params import VehicleParams
from ..perception.sensor import VisionSensor
from ..perception.tracker import MultiObjectTracker
from ..world.actors import ActorState
from ..world.grid import GridLayout
from ..world.grid_traffic import GridTrafficSource, TrafficConfig
from ..world.traffic_light import SignalState
from ..world.world import World
from .controller_api import (
    ControlCommand,
    Controller,
    DetectedObject,
    EgoState,
    Observation,
    RoutePoint,
    ScenarioContext,
    SignalInfo,
    VehicleInfo,
    check_command,
    load_controller,
)
from .controllers import BUILTIN_CONTROLLERS, MPCController
from .parameters import build_vehicle_params, clamp, merge
from .presets import PRESETS, Preset
from .scoring import ScoreBreakdown, ScoreConfig, score_run


@dataclass
class RunConfig:
    """Everything that defines a run, and therefore everything a rerun needs."""

    preset: str = "grid_random"
    controller: str = "mpc"           #: a builtin key, or a path to a ``.py``
    controller_options: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    seed: int = 0
    n_vehicles: int | None = None
    duration: float | None = None
    #: how far ahead the observation's route samples reach [m]
    route_horizon: float = 120.0
    route_sample_ds: float = 4.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Frame:
    """One rendered instant, small enough to send to a browser every tick."""

    t: float
    ego: dict
    actors: list[dict]
    signals: dict[str, str]
    command: dict
    readout: dict
    diagnostics: dict = field(default_factory=dict)


class RunSession:
    """Build, step and score a single run."""

    def __init__(self, config: RunConfig, score_config: ScoreConfig | None = None):
        self.config = config
        self.score_config = score_config or ScoreConfig()
        self.preset: Preset = PRESETS[config.preset]
        self.values = clamp(merge(config.parameters))
        self.params: VehicleParams = build_vehicle_params(self.values)

        self.duration = float(config.duration or self.preset.duration)
        self.control_dt = float(self.values["control_dt"])
        self.plant_dt = float(self.values["plant_dt"])
        self.speed_limit = float(self.values["speed_limit"])

        self.layout: GridLayout = self.preset.build_layout()
        self.lane_ids = self.layout.route_lanes(
            self.preset.ego.node, self.preset.ego.direction,
            self.preset.ego.plan, self.preset.ego.lane,
        )
        self.route = self.layout.route_path(self.lane_ids)
        self.route_signals = [(s, g) for s, g, _ in self.layout.route_signals(self.lane_ids)]
        self.goal_s = max(self.route.length - self.preset.ego.goal_margin, 10.0)

        traffic_cfg = TrafficConfig(**{
            **asdict(self.preset.traffic),
            **({"n_vehicles": int(config.n_vehicles)} if config.n_vehicles is not None else {}),
        })
        self.traffic = GridTrafficSource(
            self.layout, traffic_cfg, seed=config.seed, forced=self.preset.forced
        )

        self.world = World(
            self.layout.network, self.params, sync_source=self.traffic,
            lights=self.layout.signals, dt=self.plant_dt,
            tire_model=str(self.values["tire_model"]),
        )
        self.world.reset()
        self.world.place_ego(self.route, s=0.5, v=float(self.preset.ego.start_speed))

        self.sensor = VisionSensor()
        self.tracker = MultiObjectTracker(self.control_dt)
        self.rng = np.random.default_rng(config.seed + 977)

        self.controller = self._build_controller()
        self.controller.reset(ScenarioContext(
            scenario=self.preset.key,
            vehicle=self.vehicle_info(),
            control_dt=self.control_dt,
            goal_s=self.goal_s,
            route_length=self.route.length,
            speed_limit=self.speed_limit,
            seed=config.seed,
            options=dict(config.controller_options),
        ))

        # --- run state ---------------------------------------------------------
        self.t = 0.0
        self._s = 0.5
        self._cmd = ControlCommand()
        self._physical = (0.0, 0.0)
        self._delta_prev = 0.0
        self._steps = 0
        self.finished = False
        self.finish_reason = ""
        self.frames: list[Frame] = []
        self.log: list[dict] = []
        self._acc = _Accumulator(ttc_threshold=self.score_config.ttc_threshold)

    # --- construction helpers -------------------------------------------------

    def _build_controller(self) -> Controller:
        name = self.config.controller
        opts = dict(self.config.controller_options)
        if name == "mpc":
            return MPCController(
                self.params, self.layout.network, self.route,
                AutonomyConfig(
                    control_dt=self.control_dt,
                    perception_dt=self.control_dt,
                    speed_limit=self.speed_limit,
                    corridor_half_width=0.5 * self.layout.lane_width - 0.2,
                    seed=self.config.seed,
                ),
                route_signals=self.route_signals,
                lights=self.layout.signals,
                crossing_conflict=True,
            )
        if name in BUILTIN_CONTROLLERS:
            return BUILTIN_CONTROLLERS[name](**opts)
        return load_controller(name, **opts)

    def vehicle_info(self) -> VehicleInfo:
        p = self.params
        return VehicleInfo(
            wheelbase=p.L, l_f=p.l_f, l_r=p.l_r, mass=p.m, yaw_inertia=p.I_z,
            length=p.length, width=p.width,
            delta_max=p.actuator.delta_max, delta_rate_max=p.actuator.delta_rate_max,
            a_max=p.actuator.a_max, a_min=p.actuator.a_min, mu=p.mu,
            understeer_gradient=p.understeer_gradient,
        )

    # --- observation ------------------------------------------------------------

    def _route_points(self, s: float) -> tuple[RoutePoint, ...]:
        n = max(int(self.config.route_horizon / self.config.route_sample_ds) + 1, 2)
        ss = np.clip(s + np.arange(n) * self.config.route_sample_ds, 0.0, self.route.length)
        x, y, th, ka = self.route.frames(ss)
        return tuple(
            RoutePoint(s=float(ss[i]), x=float(x[i]), y=float(y[i]),
                       heading=float(th[i]), curvature=float(ka[i]),
                       speed_limit=self.speed_limit)
            for i in range(n)
        )

    def _signal_info(self, s: float) -> SignalInfo | None:
        for stop_s, group in self.route_signals:
            if stop_s > s - 1.0:
                return SignalInfo(
                    group=group,
                    colour=self.layout.signals.state(group, self.t).value,
                    distance=float(stop_s - s),
                    time_to_change=float(self.layout.signals.time_to_change(group, self.t)),
                )
        return None

    def _observation(self, tracks, actors: list[ActorState]) -> Observation:
        z = self.world.ego
        x_rear = self.world.ego_rear_axle()
        s = self.route.project(x_rear[0], x_rear[1], s_guess=self._s)
        self._s = s
        e_y = self.route.lateral_offset(x_rear[0], x_rear[1], s)
        e_psi = self.route.heading_error(float(z[2]), s)
        d = self.world.history[-1].diagnostics if self.world.history else {}

        ego = EgoState(
            x=float(x_rear[0]), y=float(x_rear[1]), psi=float(x_rear[2]), v=float(x_rear[3]),
            v_x=float(z[3]), v_y=float(z[4]), yaw_rate=float(z[5]), delta=float(z[10]),
            a_x=float(d.get("a_x", 0.0)), a_y=float(d.get("a_y", 0.0)),
            beta=float(d.get("beta", 0.0)),
        )

        objects = []
        for tr in tracks:
            dx, dy = tr.position[0] - ego.x, tr.position[1] - ego.y
            objects.append(DetectedObject(
                id=int(tr.id), x=float(tr.position[0]), y=float(tr.position[1]),
                psi=float(tr.heading), v=float(tr.speed),
                v_x=float(tr.velocity[0]), v_y=float(tr.velocity[1]),
                length=float(tr.length), width=float(tr.width),
                range=float(np.hypot(dx, dy)),
                bearing=float(wrap_to_pi(np.arctan2(dy, dx) - ego.psi)),
                age=int(tr.age),
                position_variance=float(np.trace(tr.P[:2, :2])),
            ))

        half = 0.5 * self.layout.lane_width - 0.2
        return Observation(
            t=self.t, dt=self.control_dt, ego=ego, objects=tuple(objects),
            route=self._route_points(s), s=float(s), e_y=float(e_y), e_psi=float(e_psi),
            lane_bounds=(-half, half), speed_limit=self.speed_limit, goal_s=self.goal_s,
            signal=self._signal_info(s), vehicle=self.vehicle_info(),
            extras={"ego_actor": self.world.ego_actor(), "actors": actors, "tracks": tracks},
        )

    # --- stepping ---------------------------------------------------------------

    def step(self) -> Frame:
        """One control tick plus the plant steps it is held over."""
        actors = self.world.current_actors()
        ego_actor = self.world.ego_actor()

        detections = self.sensor.observe(ego_actor, actors, self.rng)
        tracks = self.tracker.update(detections)
        obs = self._observation(tracks, actors)

        t0 = time.perf_counter()
        try:
            raw = self.controller.control(obs)
            cmd = check_command(raw, who=getattr(self.controller, "name", "controller"))
            error = ""
        except Exception as exc:  # noqa: BLE001 - a plug-in's failure is a result
            # A controller that raises has failed the test, not crashed the
            # platform: hold the last command, record it, and keep the run
            # comparable with the ones that did not raise.
            cmd, error = self._cmd, f"{type(exc).__name__}: {exc}"
        compute = time.perf_counter() - t0

        diagnostics = {}
        try:
            diagnostics = dict(self.controller.diagnostics() or {})
        except Exception:  # noqa: BLE001 - diagnostics must never break a run
            diagnostics = {}
        if error:
            diagnostics["controller_error"] = error

        self._cmd = cmd
        a_cmd, delta_cmd = cmd.to_physical(obs.vehicle)
        self._physical = (a_cmd, delta_cmd)

        n_plant = max(int(round(self.control_dt / self.plant_dt)), 1)
        for _ in range(n_plant):
            self.world.step(np.array([a_cmd, delta_cmd]))
        self.t = self.world.t

        self._acc.update(self, obs, cmd, a_cmd, compute, n_plant)
        self._steps += 1

        frame = self._frame(obs, cmd, compute, diagnostics)
        self.frames.append(frame)
        self.log.append({
            "t": self.t, "s": obs.s, "e_y": obs.e_y, "v": obs.ego.v,
            "steer": cmd.steer, "throttle": cmd.throttle, "brake": cmd.brake,
            "a_cmd": a_cmd, "delta_cmd": delta_cmd, "compute": compute,
            **{k: v for k, v in diagnostics.items() if isinstance(v, (int, float, str, bool))},
        })
        self._check_finished(obs)
        return frame

    def _check_finished(self, obs: Observation) -> None:
        if self.world.collisions():
            self.finished, self.finish_reason = True, "collision"
        elif obs.s >= self.goal_s:
            self.finished, self.finish_reason = True, "goal reached"
        elif self.t >= self.duration:
            self.finished, self.finish_reason = True, "time limit"

    def run(self, on_frame: Callable[[Frame], None] | None = None) -> "RunResult":
        while not self.finished:
            frame = self.step()
            if on_frame is not None:
                on_frame(frame)
        return self.result()

    # --- output -------------------------------------------------------------------

    def _frame(self, obs: Observation, cmd: ControlCommand, compute: float,
               diagnostics: dict) -> Frame:
        e = self.world.ego_actor()
        d = self.world.history[-1].diagnostics if self.world.history else {}
        return Frame(
            t=round(self.t, 3),
            ego={
                "x": round(e.x, 3), "y": round(e.y, 3), "psi": round(e.psi, 4),
                "v": round(e.v, 3), "length": e.length, "width": e.width,
                "delta": round(float(self.world.ego[10]), 4),
            },
            actors=[
                {"id": a.id, "x": round(a.x, 2), "y": round(a.y, 2),
                 "psi": round(a.psi, 3), "v": round(a.v, 2),
                 "length": a.length, "width": a.width}
                for a in self.world.current_actors()
            ],
            signals=self.world.signal_states(),
            command={"steer": round(cmd.steer, 4), "throttle": round(cmd.throttle, 4),
                     "brake": round(cmd.brake, 4)},
            readout={
                "s": round(obs.s, 1), "goal_s": round(self.goal_s, 1),
                "progress": round(obs.s / max(self.goal_s, 1e-6), 4),
                "e_y": round(obs.e_y, 3), "speed": round(obs.ego.v, 2),
                "a_y": round(float(d.get("a_y", 0.0)), 2),
                "friction": round(
                    max(float(d.get("usage_front", 0.0)), float(d.get("usage_rear", 0.0))), 3
                ),
                "tracks": len(obs.objects),
                "compute_ms": round(compute * 1e3, 2),
                "signal": (obs.signal.colour if obs.signal else "-"),
                "signal_distance": round(obs.signal.distance, 1) if obs.signal else None,
                "min_clearance": round(self._acc.min_clearance, 2),
                "min_ttc": round(min(self._acc.min_ttc, 99.0), 2),
            },
            diagnostics=diagnostics,
        )

    def metrics(self) -> dict:
        return self._acc.finalize(self)

    def score(self, config: ScoreConfig | None = None) -> ScoreBreakdown:
        return score_run(self.metrics(), config or self.score_config)

    def static_scene(self) -> dict:
        """Geometry that does not change, sent to the client once."""
        lanes = []
        for lid, lane in self.layout.network.lanes.items():
            pts = lane.sample(ds=6.0 if lane.kind.startswith("connector") else 12.0)
            lanes.append({
                "id": lid, "kind": lane.kind, "width": lane.width,
                "pts": [[round(float(x), 1), round(float(y), 1)] for x, y in pts],
            })
        route = self.route.sample(ds=4.0)
        return {
            "lanes": lanes,
            "nodes": [
                {"id": n, "x": float(c[0]), "y": float(c[1]), "half": self.layout.box_half}
                for n, c in self.layout.centres.items()
            ],
            "route": [[round(float(x), 1), round(float(y), 1)] for x, y in route],
            "box_half": self.layout.box_half,
            "lane_width": self.layout.lane_width,
            "goal": [
                round(float(v), 1) for v in self.route.position(self.goal_s)
            ],
            "bounds": self._bounds(),
        }

    def _bounds(self) -> dict:
        xs, ys = [], []
        for lane in self.layout.network.lanes.values():
            p = lane.sample(ds=40.0)
            xs.extend(p[:, 0])
            ys.extend(p[:, 1])
        pad = 20.0
        return {"xmin": float(min(xs)) - pad, "xmax": float(max(xs)) + pad,
                "ymin": float(min(ys)) - pad, "ymax": float(max(ys)) + pad}

    def result(self) -> "RunResult":
        m = self.metrics()
        return RunResult(
            config=self.config,
            preset=self.preset.key,
            controller=getattr(self.controller, "name", str(self.config.controller)),
            metrics=m,
            score=score_run(m, self.score_config),
            finish_reason=self.finish_reason,
            duration=self.t,
            log=self.log,
        )


@dataclass
class RunResult:
    config: RunConfig
    preset: str
    controller: str
    metrics: dict
    score: ScoreBreakdown
    finish_reason: str
    duration: float
    log: list[dict] = field(default_factory=list)

    def to_dict(self, include_log: bool = False) -> dict:
        out = {
            "config": self.config.to_dict(),
            "preset": self.preset,
            "controller": self.controller,
            "metrics": self.metrics,
            "score": self.score.to_dict(),
            "finish_reason": self.finish_reason,
            "duration": self.duration,
        }
        if include_log:
            out["log"] = self.log
        return out


class _Accumulator:
    """Running measurements, updated once per control tick."""

    def __init__(self, ttc_threshold: float = 2.0):
        self.ttc_threshold = float(ttc_threshold)
        self.min_clearance = float("inf")
        self.min_ttc = float("inf")
        self.time_below_ttc = 0.0
        self.collisions = 0
        self.max_friction = 0.0
        self.max_lat_accel = 0.0
        self.max_lon_accel = 0.0
        self.steering_effort = 0.0
        self.accel_effort = 0.0
        self.tractive_energy = 0.0
        self.corridor_exit_time = 0.0
        self.red_light_violations = 0
        self.compute_times: list[float] = []
        self.speeds: list[float] = []
        self.cross_track: list[float] = []
        self.jerks: list[float] = []
        self.time_to_goal = float("inf")
        self.max_s = 0.0
        self._prev_delta = 0.0
        self._prev_a = 0.0
        self._prev_s = 0.0
        self._prev_front = 0.0
        self._signal_state: dict[str, float] = {}

    def update(self, session: "RunSession", obs: Observation, cmd: ControlCommand,
               a_cmd: float, compute: float, n_plant: int) -> None:
        dt = session.control_dt
        ego = session.world.ego_actor()
        actors = session.world.current_actors()

        for a in actors:
            d = polygon_distance(ego.corners(), a.corners())
            self.min_clearance = min(self.min_clearance, d)
            if d <= 0.0:
                self.collisions += 1
            r = 0.25 * (ego.length + ego.width + a.length + a.width)
            self.min_ttc = min(
                self.min_ttc,
                time_to_collision(ego.position, ego.velocity, a.position, a.velocity, r),
            )
        ttc_now = min(
            (time_to_collision(ego.position, ego.velocity, a.position, a.velocity,
                               0.25 * (ego.length + ego.width + a.length + a.width))
             for a in actors), default=float("inf"),
        )
        if ttc_now < self.ttc_threshold:
            self.time_below_ttc += dt

        for snap in session.world.history[-n_plant:]:
            d = snap.diagnostics
            self.max_friction = max(
                self.max_friction, float(d.get("usage_front", 0.0)), float(d.get("usage_rear", 0.0))
            )
            self.max_lat_accel = max(self.max_lat_accel, abs(float(d.get("a_y", 0.0))))
            self.max_lon_accel = max(self.max_lon_accel, abs(float(d.get("a_x", 0.0))))

        delta_now = float(session.world.ego[10])
        self.steering_effort += abs(delta_now - self._prev_delta)
        self._prev_delta = delta_now
        self.accel_effort += abs(a_cmd) * dt
        self.jerks.append((a_cmd - self._prev_a) / max(dt, 1e-6))
        self._prev_a = a_cmd
        if a_cmd > 0.0:
            self.tractive_energy += a_cmd * session.params.m * max(obs.ego.v, 0.0) * dt / 1000.0

        self.cross_track.append(obs.e_y)
        if abs(obs.e_y) > abs(obs.lane_bounds[1]):
            self.corridor_exit_time += dt
        self.speeds.append(obs.ego.v)
        self.compute_times.append(compute)
        self.max_s = max(self.max_s, obs.s)
        if not np.isfinite(self.time_to_goal) and obs.s >= session.goal_s:
            self.time_to_goal = session.t

        # A stop line is crossed when the **front bumper** passes it, not the
        # rear axle the arc length is measured at: a vehicle has entered the
        # intersection when its nose is in it.
        front = obs.s + (session.params.length - session.params.rear_overhang)
        for stop_s, group in session.route_signals:
            if self._prev_front <= stop_s < front:
                if session.layout.signals.state(group, session.t) is SignalState.RED:
                    self.red_light_violations += 1
        self._prev_s = obs.s
        self._prev_front = front

    def finalize(self, session: "RunSession") -> dict:
        goal_reached = self.max_s >= session.goal_s
        n = max(len(self.compute_times), 1)
        return {
            "collisions": self.collisions,
            "goal_reached": bool(goal_reached),
            "time_to_goal": self.time_to_goal if goal_reached else float("inf"),
            "progress_ratio": float(min(self.max_s / max(session.goal_s, 1e-6), 1.0)),
            "distance": float(self.max_s),
            "mean_speed": float(np.mean(self.speeds)) if self.speeds else 0.0,
            "min_clearance": self.min_clearance,
            "min_ttc": self.min_ttc,
            "time_below_ttc": self.time_below_ttc,
            "max_friction_usage": self.max_friction,
            "corridor_exit_time": self.corridor_exit_time,
            "red_light_violations": float(self.red_light_violations),
            "steering_effort": self.steering_effort,
            "accel_effort": self.accel_effort,
            "tractive_energy": self.tractive_energy,
            "max_lat_accel": self.max_lat_accel,
            "max_lon_accel": self.max_lon_accel,
            "jerk_rms": float(np.sqrt(np.mean(np.square(self.jerks)))) if self.jerks else 0.0,
            "cross_track_rms": (
                float(np.sqrt(np.mean(np.square(self.cross_track)))) if self.cross_track else 0.0
            ),
            "cross_track_peak": float(np.max(np.abs(self.cross_track))) if self.cross_track else 0.0,
            "compute_mean": float(np.mean(self.compute_times)) if self.compute_times else 0.0,
            "compute_p95": (
                float(np.percentile(self.compute_times, 95)) if self.compute_times else 0.0
            ),
            "real_time_factor": (
                float(np.mean(self.compute_times) / session.control_dt) if self.compute_times else 0.0
            ),
            "sim_time": session.t,
            "control_ticks": n,
        }
