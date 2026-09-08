"""Signal groups for the intersection, as an explicit function of time.

The controller is deliberately **analytic**: ``state(group, t)`` is a pure
function of the clock rather than an integrated state machine.  Two things
follow, and both matter downstream:

* the simulation is exactly reproducible and can be restarted at any ``t``;
* a planner can ask ``state(group, t + tau)`` for any horizon, which is what
  makes "will it still be green when I get there?" answerable instead of
  guessed.  A behaviour layer that can only see the *current* colour has to
  treat a stale green as safe.

Phase order per cycle: ``NS green -> NS yellow -> all red -> EW green ->
EW yellow -> all red``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SignalState(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"

    @property
    def may_proceed(self) -> bool:
        return self is SignalState.GREEN

    @property
    def must_stop_if_able(self) -> bool:
        return self in (SignalState.RED, SignalState.YELLOW)


@dataclass(frozen=True)
class TrafficLightController:
    """A two-group fixed-time signal controller.

    ``offset`` shifts the whole cycle, which is how a scenario places the ego
    vehicle at a chosen point in the cycle without having to simulate its way
    there.
    """

    green: float = 20.0
    yellow: float = 3.0
    all_red: float = 2.0
    offset: float = 0.0
    first_group: str = "NS"
    second_group: str = "EW"

    @property
    def half_cycle(self) -> float:
        return self.green + self.yellow + self.all_red

    @property
    def cycle(self) -> float:
        return 2.0 * self.half_cycle

    def _phase_time(self, t: float) -> float:
        return float((t + self.offset) % self.cycle)

    def state(self, group: str, t: float) -> SignalState:
        """Colour shown to ``group`` at time ``t``."""
        tau = self._phase_time(t)
        first = tau < self.half_cycle
        local = tau if first else tau - self.half_cycle
        active = self.first_group if first else self.second_group
        if group != active:
            return SignalState.RED
        if local < self.green:
            return SignalState.GREEN
        if local < self.green + self.yellow:
            return SignalState.YELLOW
        return SignalState.RED

    def time_to_change(self, group: str, t: float) -> float:
        """Seconds until ``group``'s colour next changes."""
        tau = self._phase_time(t)
        first = tau < self.half_cycle
        local = tau if first else tau - self.half_cycle
        active = self.first_group if first else self.second_group
        if group == active:
            if local < self.green:
                return self.green - local
            if local < self.green + self.yellow:
                return self.green + self.yellow - local
            return self.half_cycle - local
        # Red for this group: it turns green at the start of the other half.
        return (self.half_cycle - local) if not first else (self.half_cycle - local)

    def time_until_green(self, group: str, t: float) -> float:
        """Seconds until ``group`` next shows green (0 if green now)."""
        if self.state(group, t) is SignalState.GREEN:
            return 0.0
        step = 0.25
        for k in range(1, int(self.cycle / step) + 2):
            if self.state(group, t + k * step) is SignalState.GREEN:
                return k * step
        return float("inf")  # pragma: no cover - a group always goes green

    def will_be_green_at(self, group: str, t: float, horizon: float) -> bool:
        """Whether the group is green throughout ``[t, t + horizon]``.

        The conservative test a "can I clear the box?" decision needs: it is not
        enough that the light is green now.
        """
        step = 0.1
        n = max(int(horizon / step), 1)
        return all(self.state(group, t + k * step) is SignalState.GREEN for k in range(n + 1))


@dataclass(frozen=True)
class DilemmaZone:
    """The classical yellow-light dilemma-zone test.

    Given the distance to the stop line, the speed and the yellow duration, a
    driver either *can stop* comfortably or *can clear* the intersection --
    and there is a zone where neither is true.  The behaviour layer reports
    which of the three it is in, rather than picking a rule and hiding it.
    """

    comfortable_decel: float = 3.0
    reaction_time: float = 0.3

    def can_stop(self, distance: float, v: float) -> bool:
        stop_distance = v * self.reaction_time + v**2 / (2.0 * self.comfortable_decel)
        return bool(stop_distance <= distance)

    def can_clear(self, distance: float, v: float, yellow_left: float, box_length: float) -> bool:
        return bool(v * yellow_left >= distance + box_length)

    def classify(self, distance: float, v: float, yellow_left: float, box_length: float) -> str:
        stop_ok = self.can_stop(distance, v)
        clear_ok = self.can_clear(distance, v, yellow_left, box_length)
        if stop_ok and clear_ok:
            return "either"
        if stop_ok:
            return "stop"
        if clear_ok:
            return "clear"
        return "dilemma"
