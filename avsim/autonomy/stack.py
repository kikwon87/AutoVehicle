"""The integrated autonomy stack: sense, track, predict, decide, plan, control.

One tick of :meth:`AutonomyStack.step` is the whole pipeline:

===  ==========================  ===========================================
1    :mod:`avsim.perception`     ground truth -> detections -> tracks
2    :mod:`avsim.planning`       tracks -> multi-modal predictions
3    localization                ego pose -> ``(s, e_y)`` on the route
4    :mod:`avsim.planning`       behaviour decision, with a reason
5    :mod:`avsim.planning`       velocity profile + Frenet lattice -> reference
6    :mod:`avsim.control`        NMPC -> ``(a, delta)``
===  ==========================  ===========================================

Two rates.  The stack runs at ``control_dt`` while the world integrates at its
own, smaller step; between control updates the command is **held**, which is
the zero-order hold the whole modelling chapter is about.  Perception may run
slower still.

Two fallbacks, both explicit and both logged:

* the lattice returns ``None`` -- every sampled manoeuvre collides or is
  infeasible -- so the stack tracks a maximum-braking trajectory instead;
* the MPC reports an infeasible or timed-out solve, so the stack falls back to
  the geometric controller (pure pursuit plus the feedforward) on the same
  reference.

A fallback that is not recorded is indistinguishable from a controller that
worked, which is why every tick returns a :class:`Telemetry` record and the KPI
layer counts them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..control.mpc import CorridorStage, MPCConfig, StopLineConstraint, VehicleMPC
from ..core.conventions import wrap_to_pi
from ..models.params import VehicleParams
from ..perception.sensor import VisionSensor
from ..perception.tracker import MultiObjectTracker
from ..planning.behavior import BehaviorConfig, BehaviorDecision, BehaviorPlanner, BehaviorState
from ..planning.prediction import ConstantVelocityPredictor, LaneFollowingPredictor
from ..planning.trajectory import FrenetLatticePlanner, LatticeConfig, Trajectory
from ..planning.velocity_profile import SpeedConstraint, build_velocity_profile
from ..world.actors import ActorState
from ..world.network import RoadNetwork
from ..world.path import ReferencePath
from ..world.traffic_light import TrafficLightController


@dataclass
class Telemetry:
    """Everything one tick decided, and why."""

    t: float
    s: float
    e_y: float
    v: float
    behavior: str
    reason: str
    target_speed: float
    #: Speed the *plan* asks for right now, which is what the controller is
    #: actually tracking.  The behaviour target can be 0 during a stop while the
    #: plan is still decelerating through 12 m/s; comparing the measured speed
    #: against the behaviour target would score that as a 12 m/s error.
    ref_speed: float
    n_detections: int
    n_tracks: int
    n_predictions: int
    mpc_status: str
    mpc_time: float
    mpc_violation: float
    mpc_iterations: int
    a_cmd: float
    delta_cmd: float
    used_fallback_plan: bool
    used_fallback_control: bool
    trajectory_offset: float = 0.0
    stop_s: float | None = None
    detail: dict = field(default_factory=dict)


@dataclass
class AutonomyConfig:
    control_dt: float = 0.1
    #: Perception runs on the control tick.  The tracker's ``dt`` must equal the
    #: interval at which ``update()`` is actually called: a Kalman filter given
    #: half the real interval propagates half the motion each step, and its
    #: velocity estimate lags a decelerating leader by seconds -- long enough to
    #: turn a routine brake into a rear-end collision.
    perception_dt: float = 0.1
    #: horizon of the prediction/behaviour time grid [s]
    prediction_horizon: float = 5.0
    prediction_steps: int = 11
    speed_limit: float = 13.9
    #: lateral corridor for the MPC, half-width [m]; used when
    #: ``corridor_bounds`` is not given
    corridor_half_width: float = 1.55
    #: Explicit ``(e_y_min, e_y_max)`` corridor.  Roads are not symmetric about
    #: the ego's lane: the adjacent lane may be on one side only, and a
    #: symmetric corridor either forbids a legal pass or permits driving off the
    #: far edge.
    corridor_bounds: tuple[float, float] | None = None
    lateral_use: float = 0.5
    longitudinal_use: float = 0.6
    #: replan the velocity profile every this many control ticks
    profile_every: int = 5
    use_lane_predictor: bool = True
    #: Re-plan the lattice from the **previous plan's** state rather than from
    #: the measurement, letting the MPC close the tracking error.
    #:
    #: Seeding the lattice with the measured state each tick defers the
    #: correction forever: every plan promises to return to the centerline over
    #: its horizon, only the first 0.1 s of it is executed, and the next plan
    #: makes the same promise from a slightly worse state.  On a 300 m radius at
    #: 16 m/s this is the difference between a 1.6 m excursion and 0.13 m.
    plan_from_nominal: bool = True
    #: Re-seed from the measurement when the plan and reality disagree by more
    #: than this, which is what makes the loop recover from a real disturbance
    #: instead of tracking a nominal that no longer describes the vehicle.
    replan_reinit_lateral: float = 0.6
    replan_reinit_speed: float = 2.5
    #: Time allowed to complete a *commanded* lateral manoeuvre [s].  While one
    #: is in progress the lattice horizons are clipped to the remaining time, so
    #: the manoeuvre finishes on schedule instead of receding with each replan.
    #: Below this speed the steering command is **held** rather than updated.
    #:
    #: Lecture 2/3, *At Standstill: What Vanishes and What Does Not*: the
    #: steering column of ``B`` scales with ``v``, so the linearization loses
    #: rank at rest. The optimizer, seeing almost no steering authority, asks
    #: for very large angles to achieve very little -- and at 0.3 m/s the plant
    #: obliges and yaws in place. Holding is the honest response: the model does
    #: not describe what steering does here, so do not act on it.
    standstill_speed: float = 0.5
    manoeuvre_time: float = 3.5
    #: The manoeuvre horizon is biased *shorter* while one is in progress, but
    #: never below this: a 3.5 m lane change squeezed into 1.2 s demands 14
    #: m/s^2 of lateral acceleration, every candidate is rejected as infeasible,
    #: and the vehicle does not change lane at all.
    manoeuvre_min_horizon: float = 1.5
    #: Stop this far before the end of the route.  A route is a finite path;
    #: without an explicit stop the planner keeps extrapolating past its end,
    #: where the clamped arc length makes every candidate look infeasible.
    route_end_margin: float = 6.0
    seed: int = 0


class AutonomyStack:
    """Wires the modules together and holds their state between ticks."""

    def __init__(
        self,
        params: VehicleParams,
        network: RoadNetwork,
        route: ReferencePath,
        config: AutonomyConfig | None = None,
        sensor: VisionSensor | None = None,
        tracker: MultiObjectTracker | None = None,
        behavior: BehaviorPlanner | None = None,
        lattice: FrenetLatticePlanner | None = None,
        mpc: VehicleMPC | None = None,
        lights: TrafficLightController | None = None,
        signal_group: str | None = None,
        stop_line_s: float | None = None,
        crossing_conflict: bool = False,
    ):
        self.p = params
        self.network = network
        self.route = route
        self.cfg = config or AutonomyConfig()

        self.sensor = sensor or VisionSensor()
        if tracker is not None and abs(tracker.dt - self.cfg.control_dt) > 1e-9:
            raise ValueError(
                f"tracker dt {tracker.dt} does not match the control period "
                f"{self.cfg.control_dt}; the filter would propagate the wrong "
                "interval on every step"
            )
        self.tracker = tracker or MultiObjectTracker(self.cfg.control_dt)
        self.predictor = (
            LaneFollowingPredictor(network) if self.cfg.use_lane_predictor
            else ConstantVelocityPredictor()
        )
        self.behavior = behavior or BehaviorPlanner(params, BehaviorConfig())
        lat_cfg = LatticeConfig(
            a_lat_limit=params.max_lateral_accel(self.cfg.lateral_use),
            a_lon_max=min(params.actuator.a_max, self.cfg.longitudinal_use * params.mu * 9.80665),
            a_lon_min=max(params.actuator.a_min, -self.cfg.longitudinal_use * params.mu * 9.80665),
        )
        self.lattice = lattice or FrenetLatticePlanner(params, lat_cfg)
        self.mpc = mpc or VehicleMPC(
            params, MPCConfig(a_y_max=params.max_lateral_accel(self.cfg.lateral_use))
        )

        self.lights = lights
        self.signal_group = signal_group
        self.stop_line_s = stop_line_s
        self.crossing_conflict = crossing_conflict

        self.rng = np.random.default_rng(self.cfg.seed)
        self.times = np.linspace(0.0, self.cfg.prediction_horizon, self.cfg.prediction_steps)

        self.reset()

    # --- lifecycle -----------------------------------------------------------

    def reset(self) -> None:
        self.sensor.reset()
        self.tracker.reset()
        self.mpc.reset()
        self.behavior.reset()
        self.lattice.reset()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._s = 0.0
        self._e_y_prev = 0.0
        self._de_y = 0.0
        self._profile = None
        self._tick = 0
        self._last_cmd = np.zeros(2)
        self._last_traj: Trajectory | None = None
        self._nominal: Trajectory | None = None
        self.reinit_count = 0
        self._stop_constraint_dropped = False
        self._offset_target = 0.0
        self._manoeuvre_deadline = -np.inf
        self.telemetry: list[Telemetry] = []
        self.tracks = []
        self.predictions = []

    # --- pipeline ------------------------------------------------------------

    def _localize(self, x_rear: np.ndarray) -> tuple[float, float]:
        s = self.route.project(x_rear[0], x_rear[1], s_guess=self._s)
        e_y = self.route.lateral_offset(x_rear[0], x_rear[1], s)
        self._s = s
        return s, e_y

    def _rebuild_profile(self, v_now: float, decision: BehaviorDecision, s: float, stop_s: float | None = None):
        constraints = []
        stop_s = decision.stop_s if stop_s is None else stop_s
        for bound, kind in ((stop_s, "stop"), (decision.safety_bound_s, "safety")):
            if bound is not None:
                # Both become a zero-speed point in the profile; the backward
                # pass then turns them into the speed ceiling
                # v(s) <= sqrt(2 b (bound - s)).  Only the hard one also becomes
                # a position constraint for the lattice and the MPC.
                constraints.append(
                    SpeedConstraint(s=float(max(bound, s)), v_max=0.0,
                                    label=f"{kind}: {decision.reason}")
                )
        self._profile = build_velocity_profile(
            self.route,
            self.p,
            v_limit=min(decision.target_speed, self.cfg.speed_limit),
            constraints=constraints,
            v_start=v_now,
            lateral_use=self.cfg.lateral_use,
            longitudinal_use=self.cfg.longitudinal_use,
            curvature_lookahead=15.0,
        )

    def _plan_seed(self, e_y_meas: float, v_meas: float) -> tuple[float, float, float, float, float]:
        """Lateral, speed and acceleration state the lattice should plan from.

        Returns ``(e_y, de_y, dde_y, v, a)``: the previous plan evaluated one
        control step in, or the measurement when the two have diverged.  The
        acceleration is carried so a replan continues the braking already under
        way instead of restarting from zero.
        """
        cfg = self.cfg
        a_meas = float(self._last_cmd[0])
        if not cfg.plan_from_nominal or self._nominal is None:
            return e_y_meas, self._de_y, 0.0, v_meas, a_meas

        tr = self._nominal
        tq = cfg.control_dt
        if tq > tr.t[-1]:
            return e_y_meas, self._de_y, 0.0, v_meas, a_meas

        e_y = float(np.interp(tq, tr.t, tr.e_y))
        de = np.gradient(tr.e_y, tr.t)
        dde = np.gradient(de, tr.t)
        de_y = float(np.interp(tq, tr.t, de))
        dde_y = float(np.interp(tq, tr.t, dde))
        v = float(np.interp(tq, tr.t, tr.v))
        a = float(np.interp(tq, tr.t, tr.a))

        if abs(e_y - e_y_meas) > cfg.replan_reinit_lateral or abs(v - v_meas) > cfg.replan_reinit_speed:
            self.reinit_count += 1
            return e_y_meas, self._de_y, 0.0, v_meas, a_meas
        return e_y, de_y, dde_y, v, a

    def _pursue_trajectory(self, x_rear: np.ndarray, v: float, traj: Trajectory) -> float:
        """Pure pursuit **on the planned trajectory**, not on the lane centre.

        The fallback exists because the MPC failed, not because the plan did.
        Steering to the route centreline instead undoes whatever manoeuvre was
        in progress -- which, during an avoidance, means steering back towards
        the obstacle. The geometry is the standard rear-axle law
        ``delta = arctan(2 L sin(alpha) / l_d)``.
        """
        l_d = float(np.clip(0.6 * abs(v) + 4.0, 3.0, 20.0))
        pts = np.column_stack([traj.x, traj.y])
        d = np.linalg.norm(pts - x_rear[:2], axis=1)
        idx = int(np.argmax(d >= l_d)) if np.any(d >= l_d) else len(pts) - 1
        target = pts[idx]

        dx, dy = target[0] - x_rear[0], target[1] - x_rear[1]
        dist = max(float(np.hypot(dx, dy)), 1e-3)
        alpha = wrap_to_pi(np.arctan2(dy, dx) - x_rear[2])
        kappa = 2.0 * np.sin(alpha) / dist
        return float(np.clip(np.arctan(kappa * self.p.L),
                             -self.p.actuator.delta_max, self.p.actuator.delta_max))

    @property
    def corridor(self) -> tuple[float, float]:
        cfg = self.cfg
        if cfg.corridor_bounds is not None:
            return cfg.corridor_bounds
        return (-cfg.corridor_half_width, cfg.corridor_half_width)

    def _corridor(self, ss: np.ndarray) -> list[CorridorStage]:
        lo, hi = self.corridor
        out = []
        for si in ss:
            si = float(np.clip(si, 0.0, self.route.length))
            out.append(
                CorridorStage(
                    p_ref=self.route.position(si),
                    normal=self.route.normal(si),
                    lo=lo,
                    hi=hi,
                )
            )
        return out

    def step(
        self,
        t: float,
        ego: ActorState,
        x_rear: np.ndarray,
        delta_actual: float,
        actors: list[ActorState],
    ) -> np.ndarray:
        """Advance the stack one control tick; returns ``[a_cmd, delta_cmd]``."""
        cfg = self.cfg

        # 1-2. perception
        detections = self.sensor.observe(ego, actors, self.rng)
        self.tracks = self.tracker.update(detections)
        self.predictions = self.predictor(self.tracks, self.times)

        # 3. localization
        s, e_y = self._localize(x_rear)
        v = float(x_rear[3])
        self._de_y = (e_y - self._e_y_prev) / max(cfg.control_dt, 1e-3)
        self._e_y_prev = e_y

        # 4. behaviour -- needs a nominal s(t) to test conflicts against
        if self._profile is None:
            self._rebuild_profile(v, BehaviorDecision(BehaviorState.CRUISE, cfg.speed_limit), s)
        s_of_t = self._profile.sample_time_grid(s, self.times)
        # Where the ego is *planned* to be, so the conflict check sees the
        # manoeuvre rather than the centerline.
        if self._nominal is not None and self._nominal.t[-1] >= self.times[-1] * 0.5:
            ego_path = np.column_stack(
                [
                    np.interp(self.times, self._nominal.t, self._nominal.x),
                    np.interp(self.times, self._nominal.t, self._nominal.y),
                ]
            )
        else:
            ego_path = np.array([self.route.to_cartesian(si, e_y) for si in s_of_t])
        decision = self.behavior.decide(
            route=self.route,
            s_ego=s,
            v_ego=v,
            times=self.times,
            s_of_t=s_of_t,
            ego_path=ego_path,
            predictions=self.predictions,
            speed_limit=cfg.speed_limit,
            stop_line_s=self.stop_line_s,
            signal_group=self.signal_group,
            lights=self.lights,
            t_now=t,
            box_length=2.0 * (self.network.box_half or 11.0),
            crossing_conflict=self.crossing_conflict,
            corridor=self.corridor,
            e_y_ego=e_y,
        )

        # 5. reference trajectory
        stop_abs = decision.stop_s  # behaviour reports an absolute arc length
        route_end = self.route.length - cfg.route_end_margin
        if s > route_end - 60.0 and (stop_abs is None or route_end < stop_abs):
            # The route is finite. Without an explicit stop at its end the
            # planner extrapolates past it, where the clamped arc length makes
            # every candidate look infeasible.
            stop_abs = max(route_end, s)
        if self._tick % cfg.profile_every == 0 or stop_abs is not None:
            self._rebuild_profile(v, decision, s, stop_abs)
        # A change of commanded offset starts a new manoeuvre with its own
        # deadline, and invalidates the nominal that was heading elsewhere.
        if abs(decision.lateral_offset - self._offset_target) > 0.5:
            self._offset_target = float(decision.lateral_offset)
            self._manoeuvre_deadline = t + cfg.manoeuvre_time
            self._nominal = None
            # The previous manoeuvre's offset is no longer the thing to be
            # consistent with.
            self.lattice.previous_offset = None
        horizons = None
        if t < self._manoeuvre_deadline:
            remaining = max(self._manoeuvre_deadline - t, cfg.manoeuvre_min_horizon)
            horizons = tuple(h for h in self.lattice.cfg.horizons if h <= remaining + 1e-9)
            # As the deadline closes, the shortest sampled horizon is the
            # remaining time itself -- the manoeuvre must finish, not recede.
            horizons = horizons or (remaining,)

        e_y0, de_y0, dde_y0, v0, a0 = self._plan_seed(e_y, v)
        traj = self.lattice.plan(
            path=self.route,
            s0=s,
            e_y0=e_y0,
            de_y0=de_y0,
            dde_y0=dde_y0,
            v0=v0,
            target_speed=min(decision.target_speed, self._profile.speed_at(s + 5.0)),
            predictions=self.predictions,
            target_offset=decision.lateral_offset,
            stop_s=stop_abs,
            corridor=self.corridor,
            a0=a0,
            horizons=horizons,
        )
        used_fallback_plan = traj is None
        if traj is None:
            # No candidate survived: brake from the *measured* state, because
            # the nominal is exactly what has just been shown to be unachievable.
            traj = self.lattice.emergency_stop(self.route, s, e_y, self._de_y, v)
            self._nominal = None
        else:
            self._nominal = traj
        self._last_traj = traj

        # 6. MPC
        N, h = self.mpc.cfg.horizon, self.mpc.cfg.dt
        mpc_times = np.arange(N + 1) * h
        reference = traj.as_reference(mpc_times)
        ss = np.array([self.route.project(px, py, s_guess=s) for px, py in reference[:, :2]])
        corridor = self._corridor(ss)
        x0 = np.array([x_rear[0], x_rear[1], x_rear[2], v, delta_actual])
        stop_constraint = None
        # An already-violated hard constraint is poison for an augmented
        # Lagrangian: the multiplier grows without bound, the solve never
        # converges, and the commanded steering goes wherever the broken
        # subproblem points.  If the front bumper is already past the line the
        # constraint is dropped and the cost is left to handle it.
        front_s = s + (self.p.length - self.p.rear_overhang)
        self._stop_constraint_dropped = (
            stop_abs is not None and stop_abs <= front_s + 0.5
        )
        if (stop_abs is not None and stop_abs < self.route.length - 1e-6
                and not self._stop_constraint_dropped):
            th_stop = self.route.heading(stop_abs)
            stop_constraint = StopLineConstraint(
                point=self.route.position(stop_abs),
                tangent=np.array([np.cos(th_stop), np.sin(th_stop)]),
            )
        res = self.mpc.solve(x0, reference, corridor, self.predictions, stop_constraint)

        used_fallback_control = not res.feasible or res.status in ("regularization_limit", "line_search_failed")
        if used_fallback_control:
            # Geometric steering plus a *proportional* speed hold.  Dividing the
            # speed error by the MPC step, as an inverse-dynamics command would,
            # turns a 1 m/s error into 8 m/s^2 of braking and makes the fallback
            # more dangerous than the failure it is covering.
            delta = self._pursue_trajectory(x_rear, v, traj)
            # Feedforward the trajectory's own acceleration and correct the
            # speed error proportionally.  A pure proportional term with a
            # half-second lookahead under-brakes an emergency stop by roughly a
            # factor of two, which turns a stop into a collision.
            idx = min(int(0.5 / max(traj.t[1] - traj.t[0], 1e-3)), len(traj.v) - 1)
            a_ff = float(traj.a[0])
            a = float(np.clip(a_ff + 1.5 * (float(traj.v[idx]) - v),
                              self.p.actuator.a_min, self.p.actuator.a_max))
            cmd = np.array([a, delta])
        else:
            cmd = np.array([res.a_cmd, res.delta_cmd])

        if v < cfg.standstill_speed and self._tick > 0:
            cmd = np.array([cmd[0], float(self._last_cmd[1])])

        self._last_cmd = cmd
        self._tick += 1
        self.telemetry.append(
            Telemetry(
                t=t, s=s, e_y=e_y, v=v,
                behavior=decision.state.value, reason=decision.reason,
                target_speed=decision.target_speed,
                ref_speed=float(np.interp(0.6, traj.t, traj.v)),
                n_detections=len(detections), n_tracks=len(self.tracks),
                n_predictions=len(self.predictions),
                mpc_status=res.status, mpc_time=res.solve_time,
                mpc_violation=res.violation, mpc_iterations=res.iterations,
                a_cmd=float(cmd[0]), delta_cmd=float(cmd[1]),
                used_fallback_plan=used_fallback_plan,
                used_fallback_control=used_fallback_control,
                trajectory_offset=float(traj.target_offset),
                stop_s=stop_abs,
                detail=dict(decision.detail, stop_constraint_dropped=self._stop_constraint_dropped),
            )
        )
        return cmd

    def last_command(self) -> np.ndarray:
        return self._last_cmd

    def last_trajectory(self) -> Trajectory | None:
        return self._last_traj
