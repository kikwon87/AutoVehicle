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

from ..models.params import VehicleParams
from ..world.path import ReferencePath
from ..world.traffic_light import DilemmaZone, SignalState, TrafficLightController
from .prediction import Prediction
from .velocity_profile import stopping_distance


class BehaviorState(Enum):
    CRUISE = "cruise"
    FOLLOW = "follow"
    STOP_FOR_SIGNAL = "stop_for_signal"
    YIELD = "yield"
    CLEAR_INTERSECTION = "clear_intersection"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class BehaviorDecision:
    """Constraints handed to the trajectory planner."""

    state: BehaviorState
    target_speed: float
    stop_s: float | None = None          #: arc length to stop at, if any
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
    conflict_horizon: float = 6.0      #: how far ahead conflicts are checked [s]
    stop_offset: float = 2.0           #: stop this far before the line [m]
    creep_speed: float = 1.5           #: speed while inching for visibility [m/s]
    comfortable_decel: float = 3.0


class BehaviorPlanner:
    """Priority-ordered behaviour selection with explicit reasons."""

    def __init__(self, params: VehicleParams, config: BehaviorConfig | None = None):
        self.p = params
        self.cfg = config or BehaviorConfig()
        self.dilemma = DilemmaZone(comfortable_decel=self.cfg.comfortable_decel)
        self.state = BehaviorState.CRUISE

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
    ) -> tuple[float, float, Prediction] | None:
        """Nearest object ahead on the ego's own route: ``(gap, speed, prediction)``."""
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
                best = (gap, speed, pred)
        return best

    def conflict_time(
        self,
        route: ReferencePath,
        s_of_t: np.ndarray,
        times: np.ndarray,
        predictions: Sequence[Prediction],
        inflate: float = 1.0,
    ) -> tuple[float | None, Prediction | None, float]:
        """Earliest time the ego's swept disc meets any prediction's.

        ``s_of_t`` is where the ego would be at each time under its nominal
        speed profile.  Every mode of every prediction is tested; a conflict
        with a low-probability mode is still a conflict, and the returned
        probability lets the caller decide how much to weigh it.
        """
        ego_r = 0.5 * float(np.hypot(self.p.length, self.p.width)) + self.cfg.conflict_margin
        for i, t in enumerate(times):
            if t > self.cfg.conflict_horizon:
                break
            ego_p = route.position(float(s_of_t[i]))
            for pred in predictions:
                obj_p = pred.position_at(float(t))
                r = ego_r + float(np.interp(t, pred.times, pred.radius(inflate)))
                if float(np.linalg.norm(ego_p - obj_p)) < r:
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
        predictions: Sequence[Prediction],
        speed_limit: float,
        stop_line_s: float | None = None,
        signal_group: str | None = None,
        lights: TrafficLightController | None = None,
        t_now: float = 0.0,
        box_length: float = 22.0,
        crossing_conflict: bool = False,
    ) -> BehaviorDecision:
        cfg = self.cfg

        # --- 1. emergency ----------------------------------------------------
        t_conf, pred, prob = self.conflict_time(route, s_of_t, times, predictions)
        if t_conf is not None and t_conf < cfg.t_emergency:
            self.state = BehaviorState.EMERGENCY_STOP
            return BehaviorDecision(
                state=self.state,
                target_speed=0.0,
                stop_s=s_ego + max(stopping_distance(v_ego, self.p.actuator.a_min), 0.5),
                reason=f"predicted contact in {t_conf:.2f} s with track {pred.track_id if pred else '?'}",
                detail={"t_conflict": t_conf, "mode": pred.mode if pred else None},
            )

        # --- 2. signal -------------------------------------------------------
        if stop_line_s is not None and lights is not None and signal_group is not None:
            dist = stop_line_s - s_ego
            if dist > -1.0:  # not yet committed past the line
                colour = lights.state(signal_group, t_now)
                decision = self._signal_decision(
                    colour, dist, v_ego, lights, signal_group, t_now, box_length
                )
                if decision is not None:
                    self.state = decision.state
                    return decision

        # --- 3. intersection conflict ---------------------------------------
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

        # --- 4. car following -------------------------------------------------
        lead = self._lead_vehicle(route, s_ego, predictions)
        if lead is not None:
            gap, lead_v, lead_pred = lead
            desired = cfg.follow_min_gap + cfg.follow_time_gap * v_ego
            if gap < desired * 1.8:
                # Track the leader's speed, corrected by the gap error.  Not a
                # full IDM: the MPC does the smoothing, this only sets the
                # target it converges to.
                v_target = float(
                    np.clip(lead_v + 0.6 * (gap - desired), 0.0, speed_limit)
                )
                self.state = BehaviorState.FOLLOW
                return BehaviorDecision(
                    state=self.state,
                    target_speed=v_target,
                    lead_gap=gap,
                    lead_speed=lead_v,
                    reason=f"following track {lead_pred.track_id} at {gap:.1f} m",
                    detail={"desired_gap": desired},
                )

        # --- 5. cruise --------------------------------------------------------
        self.state = BehaviorState.CRUISE
        return BehaviorDecision(state=self.state, target_speed=speed_limit, reason="clear")

    # --- signal logic --------------------------------------------------------

    def _signal_decision(
        self,
        colour: SignalState,
        dist: float,
        v_ego: float,
        lights: TrafficLightController,
        group: str,
        t_now: float,
        box_length: float,
    ) -> BehaviorDecision | None:
        cfg = self.cfg
        stop_s_rel = dist - cfg.stop_offset

        if colour is SignalState.RED:
            return BehaviorDecision(
                state=BehaviorState.STOP_FOR_SIGNAL,
                target_speed=0.0,
                stop_s=stop_s_rel,
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
                    stop_s=stop_s_rel,
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
            time_to_clear = (dist + box_length) / max(v_ego, 0.5)
            if not lights.will_be_green_at(group, t_now, min(time_to_clear, 12.0)):
                if self.dilemma.can_stop(dist, v_ego):
                    return BehaviorDecision(
                        state=BehaviorState.STOP_FOR_SIGNAL,
                        target_speed=0.0,
                        stop_s=stop_s_rel,
                        reason="green will end before the box is cleared",
                        detail={"time_to_clear": time_to_clear,
                                "green_left": lights.time_to_change(group, t_now)},
                    )
        return None
