"""The behaviour layer: what to do, expressed as constraints for the planner.

This module decides *whether* to go, and hands the trajectory planner a small
set of scalars -- a target speed, a stop point, a lead vehicle -- rather than a
trajectory.  Keeping the decision separate from the optimization is what makes
"why did it brake?" answerable: every decision carries a
:class:`BehaviorDecision.reason` string, and the KPI layer logs it.

Decisions, in strict priority order:

1. **Emergency** -- a predicted collision inside ``t_emergency`` that braking
   alone can still avoid.
2. **Signal** -- red, or yellow classified as "stop" by the dilemma-zone rule,
   or *green but not green long enough to clear the box*.  The third case is
   the one a controller that only reads the current colour cannot make.
3. **Intersection conflict** -- for an unprotected turn, a space-time check
   against every prediction **mode**, not just the most likely one.
4. **Car following** -- the nearest prediction whose current position lies
   ahead on the ego's own route, within a lane width.
5. **Cruise**.

The layer never *commands* anything.  It returns constraints; the trajectory
planner and the MPC decide how to satisfy them, and can refuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np

from ..core.geometry import circle_centres, ellipse_clearance, multi_circle_cover
from ..models.params import VehicleParams
from ..world.path import ReferencePath
from ..world.traffic_light import DilemmaZone, SignalState, TrafficLightController
from ..world.actors import IDMParams, idm_acceleration
from .prediction import Prediction
from .velocity_profile import stopping_distance


class BehaviorState(Enum):
    CRUISE = "cruise"
    FOLLOW = "follow"
    STOP_FOR_SIGNAL = "stop_for_signal"
    YIELD = "yield"
    CLEAR_INTERSECTION = "clear_intersection"
    EMERGENCY_STOP = "emergency_stop"
    OVERTAKE = "overtake"


@dataclass
class BehaviorDecision:
    """Constraints handed to the trajectory planner."""

    state: BehaviorState
    target_speed: float
    #: **Absolute** arc length along the route to stop at, or ``None``.
    #: Always absolute -- a mix of absolute and relative stop points is the kind
    #: of convention mismatch that produces a vehicle stopping in the middle of
    #: the intersection and no error anywhere.
    stop_s: float | None = None
    #: A **soft** longitudinal bound: the ego must be *able* to stop before this
    #: arc length, which is a speed ceiling ``v <= sqrt(2 b (s_bound - s))``, not
    #: an instruction to stop there.  Treating a following distance as a hard
    #: stop line makes the vehicle brake to rest fifty metres behind a car that
    #: is still moving at 14 m/s.
    safety_bound_s: float | None = None
    lead_gap: float | None = None        #: bumper gap to the lead vehicle [m]
    lead_speed: float = 0.0
    lateral_offset: float = 0.0          #: commanded ``e_y`` target [m]
    reason: str = ""
    #: diagnostics that made the decision, logged verbatim
    detail: dict = field(default_factory=dict)


@dataclass
class BehaviorConfig:
    t_emergency: float = 1.6           #: TTC below which emergency braking triggers
    follow_time_gap: float = 1.6       #: desired time headway [s]
    follow_min_gap: float = 4.0        #: minimum bumper gap [m]
    lane_half_width: float = 1.9       #: lateral window for "on my route" [m]
    conflict_margin: float = 1.2       #: extra clearance in the space-time check [m]
    #: Deceleration the *leader* is assumed capable of [m/s^2].  The follow
    #: state converts it into a hard positional bound: if the leader can stop
    #: within ``v^2 / 2b``, the ego must be able to stop short of that point.
    #: A speed target alone cannot express this -- a target the planner cannot
    #: reach within its horizon is silently clamped, and the gap closes anyway.
    lead_max_decel: float = 6.0
    conflict_horizon: float = 6.0      #: how far ahead conflicts are checked [s]
    #: Clearance the **front bumper** keeps from the stop line [m].
    #:
    #: ``stop_s`` is where the *nose* must not pass, because that is what a stop
    #: line means and what the MPC's half-space constrains.  The arc length the
    #: planner works in is measured at the rear axle, so the velocity profile and
    #: the lattice are given ``stop_s - (length - rear_overhang)`` instead --
    #: 3.7 m further back on the reference car.  Handing the same number to both
    #: is a bug in one direction or the other: the nose ends up 1.7 m past the
    #: line, or the car stops 3.7 m short of where it was asked to.
    stop_offset: float = 2.0
    #: Once an overtake is chosen, hold it for at least this long.  A manoeuvre
    #: re-decided every tick is never executed: the offset target flickers, the
    #: lattice re-plans from a different homotopy each time, and the vehicle
    #: arrives at the obstacle still in its own lane.
    overtake_hold: float = 4.0
    #: Consecutive ticks a predicted contact must persist before emergency
    #: braking.  A single frame of it is usually a perception artefact.
    emergency_confirm: int = 2
    #: Consecutive clear ticks before emergency braking is released.  Without a
    #: latch the state alternates with whatever ran before it, and the two
    #: issue opposite lateral targets on alternate ticks.
    emergency_release: int = 6
    creep_speed: float = 1.5           #: speed while inching for visibility [m/s]
    #: Speed to clear an intersection the ego is already committed inside [m/s]
    clear_speed: float = 4.0
    #: Acceleration assumed when estimating how long a *standing* vehicle needs
    #: to clear the box [m/s^2].
    launch_accel: float = 1.5
    comfortable_decel: float = 3.0
    #: A lead slower than this is an obstacle to go around, not a car to follow.
    overtake_speed: float = 1.0
    #: Lateral room needed on one side before an overtake is proposed [m]
    overtake_room: float = 3.0
    #: Offset commanded when overtaking [m]
    overtake_offset: float = 3.5


class BehaviorPlanner:
    """Priority-ordered behaviour selection with explicit reasons."""

    def __init__(self, params: VehicleParams, config: BehaviorConfig | None = None):
        self.p = params
        self.cfg = config or BehaviorConfig()
        self.dilemma = DilemmaZone(comfortable_decel=self.cfg.comfortable_decel)
        self.state = BehaviorState.CRUISE
        self._overtake_offset = 0.0
        self._overtake_side = 0.0
        self._overtake_until = -np.inf
        self._emergency_streak = 0
        self._clear_streak = 0
        self._ego_off, self._ego_r = multi_circle_cover(params.length, params.width, 3)

    def reset(self) -> None:
        self.state = BehaviorState.CRUISE
        self._overtake_offset = 0.0
        self._overtake_until = -np.inf
        self._emergency_streak = 0

    # --- helpers -------------------------------------------------------------

    def _project_onto_route(
        self, route: ReferencePath, s_ego: float, point: np.ndarray
    ) -> tuple[float, float] | None:
        """``(s, e_y)`` of a world point on the route, or ``None`` if far off it."""
        s = route.project(point[0], point[1], s_guess=s_ego + 20.0, window=90.0)
        e_y = route.lateral_offset(point[0], point[1], s)
        if abs(e_y) > self.cfg.lane_half_width:
            return None
        return s, e_y

    def _lead_vehicle(
        self, route: ReferencePath, s_ego: float, predictions: Sequence[Prediction]
    ) -> tuple[float, float, Prediction, float] | None:
        """Nearest object ahead on the ego's own route.

        Returns ``(gap, speed, prediction, s_object)``.
        """
        best = None
        seen: set[int] = set()
        for pred in predictions:
            if pred.track_id in seen:
                continue
            hit = self._project_onto_route(route, s_ego, pred.positions[0])
            if hit is None:
                continue
            s_obj, _ = hit
            gap = s_obj - s_ego - 0.5 * (self.p.length + pred.length)
            if gap < -0.5:
                continue
            seen.add(pred.track_id)
            speed = float(np.linalg.norm(pred.positions[1] - pred.positions[0]) /
                          max(pred.times[1] - pred.times[0], 1e-6))
            if best is None or gap < best[0]:
                best = (gap, speed, pred, s_obj)
        return best

    def conflict_time(
        self,
        ego_path: np.ndarray,
        times: np.ndarray,
        predictions: Sequence[Prediction],
        inflate: float = 1.0,
    ) -> tuple[float | None, Prediction | None, float]:
        """Earliest time the ego's swept disc meets any prediction's.

        ``ego_path`` is ``(T, 2)``: where the ego is **planned to be** at each
        time, not where the centerline is.  The distinction is not cosmetic --
        during a lane change the two differ by a full lane, and a check that
        assumes the centerline reports a collision with the very obstacle the
        manoeuvre is avoiding, then cancels the manoeuvre, then reports it
        again.  That is the loop that produces behaviour chatter.

        Every mode of every prediction is tested; a conflict with a
        low-probability mode is still a conflict, and the returned probability
        lets the caller decide how much to weigh it.
        """
        margin = self.cfg.conflict_margin
        heads = _path_headings(ego_path)
        for i, t in enumerate(times):
            if t > self.cfg.conflict_horizon:
                break
            ego_pts = circle_centres(ego_path[i][0], ego_path[i][1], heads[i], self._ego_off)
            for pred in predictions:
                obj_off, obj_r = multi_circle_cover(pred.length, pred.width, 3)
                obj_c = pred.position_at(float(t))
                obj_h = float(np.interp(t, pred.times, np.unwrap(pred.headings)))
                sig_lon, sig_lat = pred.ellipse_axes(inflate)
                a = self._ego_r + obj_r + margin + float(np.interp(t, pred.times, sig_lon))
                b = self._ego_r + obj_r + margin + float(np.interp(t, pred.times, sig_lat))
                obj_pts = circle_centres(obj_c[0], obj_c[1], obj_h, obj_off)
                for ep in ego_pts:
                    for op in obj_pts:
                        if ellipse_clearance(ep - op, obj_h, a, b) < 0.0:
                            return float(t), pred, pred.probability
        return None, None, 0.0

    # --- main entry point ----------------------------------------------------

    def decide(
        self,
        route: ReferencePath,
        s_ego: float,
        v_ego: float,
        times: np.ndarray,
        s_of_t: np.ndarray,
        ego_path: np.ndarray,
        predictions: Sequence[Prediction],
        speed_limit: float,
        stop_line_s: float | None = None,
        signal_group: str | None = None,
        lights: TrafficLightController | None = None,
        t_now: float = 0.0,
        box_length: float = 22.0,
        crossing_conflict: bool = False,
        corridor: tuple[float, float] = (-1.75, 1.75),
        e_y_ego: float = 0.0,
    ) -> BehaviorDecision:
        cfg = self.cfg
        # "Committed" means the front of the vehicle is past the stop line, so
        # the box can only be left by going forward.
        committed = stop_line_s is not None and s_ego > stop_line_s - 1.0

        # --- 1. emergency ----------------------------------------------------
        t_conf, pred, prob = self.conflict_time(ego_path, times, predictions)
        if t_conf is not None and t_conf < cfg.t_emergency:
            self._emergency_streak += 1
            self._clear_streak = 0
        else:
            self._clear_streak += 1
            if self._clear_streak >= cfg.emergency_release:
                self._emergency_streak = 0
        latched = (
            self.state is BehaviorState.EMERGENCY_STOP
            and self._clear_streak < cfg.emergency_release
        )
        if (self._emergency_streak >= cfg.emergency_confirm or latched) and not (
            committed and crossing_conflict
        ):
            self.state = BehaviorState.EMERGENCY_STOP
            return BehaviorDecision(
                state=self.state,
                target_speed=0.0,
                stop_s=s_ego + max(stopping_distance(v_ego, self.p.actuator.a_min), 0.5),  # absolute
                lateral_offset=self._overtake_offset if t_now < self._overtake_until else 0.0,
                reason=(
                    f"predicted contact in {t_conf:.2f} s with track "
                    f"{pred.track_id if pred else '?'}"
                    if t_conf is not None else "holding the emergency stop"
                ),
                detail={"t_conflict": t_conf, "mode": pred.mode if pred else None},
            )

        # --- 2. signal -------------------------------------------------------
        if stop_line_s is not None and lights is not None and signal_group is not None:
            dist = stop_line_s - s_ego
            if dist > -1.0:  # not yet committed past the line
                colour = lights.state(signal_group, t_now)
                decision = self._signal_decision(
                    colour, dist, v_ego, lights, signal_group, t_now, box_length, stop_line_s
                )
                if decision is not None:
                    self.state = decision.state
                    return decision

        # --- 3. intersection conflict ---------------------------------------
        if crossing_conflict and t_conf is not None and committed:
            # Already inside the box: stopping here is the one thing that is
            # certainly wrong.  A vehicle with right of way is arriving and the
            # ego is parked in its path, which is exactly how the unprotected
            # left ends in a collision.  Clear.
            self.state = BehaviorState.CLEAR_INTERSECTION
            return BehaviorDecision(
                state=self.state,
                target_speed=max(v_ego, cfg.clear_speed),
                reason=f"committed inside the box, conflict in {t_conf:.2f} s: clearing",
                detail={"t_conflict": t_conf, "mode": pred.mode if pred else None},
            )
        if crossing_conflict and t_conf is not None:
            # Stop short of the conflict rather than at the stop line: on an
            # unprotected turn the ego may legally be inside the box already.
            s_conf = float(np.interp(t_conf, times, s_of_t))
            self.state = BehaviorState.YIELD
            return BehaviorDecision(
                state=self.state,
                target_speed=0.0,
                stop_s=max(s_conf - cfg.conflict_margin - 0.5 * self.p.length, s_ego),
                reason=f"yielding: conflict in {t_conf:.2f} s (mode p={prob:.2f})",
                detail={"t_conflict": t_conf, "mode": pred.mode if pred else None, "p": prob},
            )

        # --- 4. an overtake already in progress stays in progress -------------
        if t_now < self._overtake_until:
            self.state = BehaviorState.OVERTAKE
            return BehaviorDecision(
                state=self.state,
                target_speed=min(speed_limit, max(v_ego, 4.0)),
                lateral_offset=self._overtake_offset,
                reason="completing the overtake",
                detail={"until": self._overtake_until},
            )

        # --- 5. car following -------------------------------------------------
        lead = self._lead_vehicle(route, s_ego, predictions)
        if lead is not None:
            gap, lead_v, lead_pred, s_lead = lead

            # A stationary obstacle is not a car to follow.  Following it means
            # arriving at it slowly and then discovering there is nowhere to go;
            # the decision to go around has to be made while there is still
            # room to make it, so it is made here rather than left to the
            # lattice's cost to discover.
            if lead_v < cfg.overtake_speed and gap < 6.0 * max(v_ego, 1.0):
                # Room is measured from the *lane centre*, not from where the
                # vehicle currently is: half way through the manoeuvre the two
                # sides look equally roomy, and re-deciding then flips the
                # vehicle back across the obstacle it was passing.
                room_left = corridor[1]
                room_right = -corridor[0]
                if max(room_left, room_right) >= cfg.overtake_room:
                    if self._overtake_side == 0.0:
                        self._overtake_side = 1.0 if room_left >= room_right else -1.0
                    side = self._overtake_side
                    room = room_left if side > 0 else room_right
                    offset = side * min(cfg.overtake_offset, room - 0.4)
                    self._overtake_offset = float(offset)
                    # Re-arm every tick the object is still ahead, so the
                    # commitment outlasts the manoeuvre rather than expiring in
                    # the middle of it.
                    self._overtake_until = t_now + cfg.overtake_hold
                    self.state = BehaviorState.OVERTAKE
                    return BehaviorDecision(
                        state=self.state,
                        target_speed=min(speed_limit, max(v_ego, 4.0)),
                        lateral_offset=float(offset),
                        lead_gap=gap,
                        lead_speed=lead_v,
                        reason=f"going around a stopped object at {gap:.1f} m",
                        detail={"room_left": room_left, "room_right": room_right},
                    )

            desired = cfg.follow_min_gap + cfg.follow_time_gap * v_ego
            if gap < desired * 2.2:
                # Intelligent Driver Model on the *measured* gap and closing
                # rate.  A law that reacts only to the gap error is far too
                # weak when the leader brakes: the gap is still comfortable at
                # the moment the leader starts, and by the time it is not, the
                # closing rate is what matters.  The prediction is
                # constant-velocity and cannot see the braking at all, so this
                # term is the only thing that does.
                idm = IDMParams(
                    v0=speed_limit,
                    T=cfg.follow_time_gap,
                    s0=cfg.follow_min_gap,
                    a_max=min(2.0, self.p.actuator.a_max),
                    b=cfg.comfortable_decel,
                    b_emergency=abs(self.p.actuator.a_min),
                )
                a_des = idm_acceleration(v_ego, gap, v_ego - lead_v, idm)
                v_target = float(np.clip(v_ego + a_des * 1.0, 0.0, speed_limit))
                # Worst case the leader stops as hard as it can; the ego must be
                # able to stop short of where that leaves it.
                buffer = cfg.follow_min_gap + 0.5 * (self.p.length + lead_pred.length)
                stop_bound = s_lead + lead_v**2 / (2.0 * cfg.lead_max_decel) - buffer
                self.state = BehaviorState.FOLLOW
                return BehaviorDecision(
                    state=self.state,
                    target_speed=v_target,
                    safety_bound_s=float(max(stop_bound, s_ego)),
                    lead_gap=gap,
                    lead_speed=lead_v,
                    reason=f"following track {lead_pred.track_id} at {gap:.1f} m",
                    detail={"desired_gap": desired, "stop_bound": stop_bound},
                )

        # --- 6. cruise --------------------------------------------------------
        self._overtake_side = 0.0   # nothing ahead: the next overtake is free to choose
        self.state = BehaviorState.CRUISE
        return BehaviorDecision(state=self.state, target_speed=speed_limit, reason="clear")

    # --- signal logic --------------------------------------------------------

    def _time_to_clear(self, dist: float, v_ego: float, box_length: float) -> float:
        """Seconds to put the whole vehicle past the far side of the box.

        Dividing the distance by the *current* speed is wrong for a stopped
        vehicle: it returns something near infinity, no green is ever long
        enough, and a car waiting at a red light can never decide to depart when
        it turns green. From rest the estimate is the launch profile
        ``t = sqrt(2 d / a)`` instead.
        """
        d = dist + box_length
        if v_ego < 1.0:
            return float(np.sqrt(2.0 * d / max(self.cfg.launch_accel, 1e-3)))
        # Moving: distance over speed, but never slower than the launch profile
        # would manage from here.
        return float(min(d / v_ego, np.sqrt(2.0 * d / self.cfg.launch_accel) + v_ego * 0.0))

    def _signal_decision(
        self,
        colour: SignalState,
        dist: float,
        v_ego: float,
        lights: TrafficLightController,
        group: str,
        t_now: float,
        box_length: float,
        stop_line_s: float,
    ) -> BehaviorDecision | None:
        cfg = self.cfg
        stop_target = stop_line_s - cfg.stop_offset

        if colour is SignalState.RED:
            return BehaviorDecision(
                state=BehaviorState.STOP_FOR_SIGNAL,
                target_speed=0.0,
                stop_s=stop_target,
                reason="red light",
                detail={"distance": dist, "until_green": lights.time_until_green(group, t_now)},
            )

        if colour is SignalState.YELLOW:
            yellow_left = lights.time_to_change(group, t_now)
            verdict = self.dilemma.classify(dist, v_ego, yellow_left, box_length)
            if verdict in ("stop", "either"):
                return BehaviorDecision(
                    state=BehaviorState.STOP_FOR_SIGNAL,
                    target_speed=0.0,
                    stop_s=stop_target,
                    reason=f"yellow, dilemma verdict '{verdict}'",
                    detail={"distance": dist, "yellow_left": yellow_left},
                )
            return BehaviorDecision(
                state=BehaviorState.CLEAR_INTERSECTION,
                target_speed=max(v_ego, 1.0),
                reason=f"yellow, dilemma verdict '{verdict}': clearing",
                detail={"distance": dist, "yellow_left": yellow_left},
            )

        # Green -- but is it green long enough to get across?
        if dist > 0.5:
            time_to_clear = self._time_to_clear(dist, v_ego, box_length)
            if not lights.will_be_green_at(group, t_now, min(time_to_clear, 12.0)):
                if self.dilemma.can_stop(dist, v_ego):
                    return BehaviorDecision(
                        state=BehaviorState.STOP_FOR_SIGNAL,
                        target_speed=0.0,
                        stop_s=stop_target,
                        reason="green will end before the box is cleared",
                        detail={"time_to_clear": time_to_clear,
                                "green_left": lights.time_to_change(group, t_now)},
                    )
        return None


def _path_headings(path: np.ndarray) -> np.ndarray:
    """Headings along a sampled path, by forward difference with a held last value."""
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return np.zeros(len(path))
    d = np.diff(path, axis=0)
    th = np.arctan2(d[:, 1], d[:, 0])
    return np.concatenate([th, th[-1:]])
