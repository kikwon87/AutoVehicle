"""Other traffic participants and the sync sources that supply them.

A **sync source** is the seam between the ego's world model and wherever the
other agents actually come from.  The ego stack never constructs traffic; it
asks a :class:`SyncSource` for the current set of :class:`ActorState` objects.
That indirection is what lets the same scenario run against

* :class:`SimulatedTrafficSource` -- reactive vehicles driving routes with IDM
  longitudinal behaviour and pure-pursuit steering on the kinematic bicycle;
* :class:`ScriptedSyncSource` -- open-loop actors whose motion is a function of
  time, for deterministic regression tests (a cut-in that must happen at
  exactly ``t = 4.0 s``);
* :class:`ReplaySyncSource` -- recorded logs played back;
* :class:`CompositeSyncSource` -- any mixture of the above.

The reactive and the scripted sources answer different questions.  A reactive
source tells you whether the plan survives *traffic that reacts to it*; a
scripted one tells you whether it survives a *specific* adversarial event, the
same way every run.  A test suite needs both, and confusing them is how a
scenario silently stops testing what it was written to test.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from ..core.conventions import wrap_to_pi
from ..models.params import VehicleParams
from .path import ReferencePath
from .traffic_light import SignalState, TrafficLightController


@dataclass
class ActorState:
    """The pose and extent of one traffic participant, in world coordinates.

    This is the *ground truth*.  The perception module in
    :mod:`avsim.perception` degrades it into measurements; the planner is never
    handed this object directly, which is the only way to be sure the planner
    is not cheating.
    """

    id: str
    x: float
    y: float
    psi: float
    v: float
    a: float = 0.0
    length: float = 4.6
    width: float = 1.85
    kind: str = "vehicle"
    lane_id: str | None = None
    #: rear-axle-to-centre offset, so the box is drawn about the geometric centre
    centre_offset: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def velocity(self) -> np.ndarray:
        return self.v * np.array([np.cos(self.psi), np.sin(self.psi)])

    def corners(self) -> np.ndarray:
        """The four corners of the bounding box, counter-clockwise ``(4, 2)``."""
        c, s = np.cos(self.psi), np.sin(self.psi)
        R = np.array([[c, -s], [s, c]])
        hl, hw = 0.5 * self.length, 0.5 * self.width
        local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]]) + np.array(
            [self.centre_offset, 0.0]
        )
        return (R @ local.T).T + self.position

    def predict_constant_velocity(self, dt: float) -> "ActorState":
        """Straight-line extrapolation -- the honest baseline prediction.

        It is wrong for any turning vehicle, and it is stated as the baseline
        so that a scenario failure can be attributed to the *prediction* rather
        than blamed on the planner.
        """
        return replace(self, x=self.x + self.v * np.cos(self.psi) * dt,
                       y=self.y + self.v * np.sin(self.psi) * dt)


# --- longitudinal behaviour ---------------------------------------------------

@dataclass(frozen=True)
class IDMParams:
    """Intelligent Driver Model parameters (Treiber, Hennecke & Helbing, 2000)."""

    v0: float = 13.9      #: desired speed [m/s]
    T: float = 1.4        #: desired time headway [s]
    s0: float = 2.5       #: minimum bumper gap [m]
    a_max: float = 1.5    #: maximum acceleration [m/s^2]
    b: float = 2.0        #: comfortable deceleration [m/s^2]
    delta: float = 4.0    #: acceleration exponent
    b_emergency: float = 6.0


def idm_acceleration(v: float, gap: float | None, dv: float, p: IDMParams) -> float:
    """``a = a_max [1 - (v/v0)^delta - (s*/s)^2]``, ``s* = s0 + vT + v dv / (2 sqrt(a b))``.

    ``gap`` is the bumper-to-bumper distance to the leader and ``dv = v - v_lead``
    the *approach* rate.  ``gap = None`` means free flow.  The gap is floored at
    a small positive number: at zero gap the interaction term is infinite, and
    an unfloored IDM emits ``-inf`` the first time two actors touch.
    """
    free = 1.0 - (max(v, 0.0) / p.v0) ** p.delta
    if gap is None:
        return float(p.a_max * free)
    s = max(gap, 0.1)
    s_star = p.s0 + max(v * p.T + v * dv / (2.0 * np.sqrt(p.a_max * p.b)), 0.0)
    a = p.a_max * (free - (s_star / s) ** 2)
    return float(np.clip(a, -p.b_emergency, p.a_max))


# --- reactive actors ----------------------------------------------------------

@dataclass
class TrafficActor:
    """A background vehicle following a route with IDM and pure pursuit.

    Motion uses the **kinematic bicycle at the rear axle** -- the same model the
    planner uses -- integrated with RK4.  Background traffic does not need tire
    forces, but it does need to be *dynamically consistent*: a rail-following
    actor teleports its heading around a corner, and the perception and
    prediction modules then see headings no real vehicle produces.
    """

    id: str
    path: ReferencePath
    s: float = 0.0
    v: float = 0.0
    idm: IDMParams = field(default_factory=IDMParams)
    length: float = 4.6
    width: float = 1.85
    #: signal group this actor obeys, if it crosses a signalized stop line
    signal_group: str | None = None
    #: arc length of the stop line along ``path``
    stop_line_s: float | None = None
    #: Every signalized stop line along the route, ``(arc length, group)``.
    #: A route across a grid crosses several; an actor that can hold only one
    #: obeys the first light and runs every one after it.
    route_signals: list[tuple[float, str]] = field(default_factory=list)
    #: lane ids the route is made of, and the arc length each begins at
    lane_sequence: list[str] = field(default_factory=list)
    lane_starts: list[float] = field(default_factory=list)
    wheelbase: float = 2.7
    lane_id: str | None = None
    kind: str = "vehicle"
    #: Actors sharing a ``route_id`` follow one another.  Identified by name
    #: rather than by the identity of the ``path`` object, so that resetting or
    #: copying a source cannot silently sever the car-following relation.
    route_id: str = "default"

    # internal kinematic state, seeded from the path on first step
    _x: float | None = None
    _y: float | None = None
    _psi: float | None = None
    _a: float = 0.0
    _lookahead: float = 6.0
    done: bool = False

    def __post_init__(self) -> None:
        if not self.route_signals and self.stop_line_s is not None and self.signal_group:
            self.route_signals = [(float(self.stop_line_s), str(self.signal_group))]
        self.route_signals = sorted(self.route_signals)
        if self._x is None:
            p = self.path.position(self.s)
            self._x, self._y, self._psi = float(p[0]), float(p[1]), self.path.heading(self.s)
        self._initial = (self.s, self.v, self._x, self._y, self._psi)

    def reset(self) -> None:
        """Restore the mutable state, keeping the (shared, immutable) path."""
        self.s, self.v, self._x, self._y, self._psi = self._initial
        self._a = 0.0
        self.done = False

    def state(self) -> ActorState:
        return ActorState(
            id=self.id,
            x=float(self._x),
            y=float(self._y),
            psi=float(self._psi),
            v=float(self.v),
            a=float(self._a),
            length=self.length,
            width=self.width,
            kind=self.kind,
            lane_id=self.lane_id,
            centre_offset=0.5 * self.length - 0.9,
        )

    def next_signal(self) -> tuple[float | None, str | None]:
        """The next stop line at or ahead of the actor, or ``(None, None)``."""
        for stop_s, group in self.route_signals:
            if stop_s > self.s - 1.0:
                return stop_s, group
        return None, None

    def current_lane(self) -> str | None:
        """Lane id the actor is on, from its arc length along the route."""
        if not self.lane_sequence:
            return self.lane_id
        i = int(np.searchsorted(self.lane_starts, self.s, side="right") - 1)
        return self.lane_sequence[min(max(i, 0), len(self.lane_sequence) - 1)]

    def _virtual_leader_gap(
        self, t: float, lights: TrafficLightController | None
    ) -> tuple[float | None, float]:
        """Gap and approach rate to a red or yellow light treated as an obstacle.

        A signal is modelled as a stationary vehicle parked on the stop line.
        That reuses the car-following law instead of adding a second
        deceleration rule, so a queue behind a red light forms for free.
        """
        stop_line_s, group = self.next_signal()
        if lights is None or group is None or stop_line_s is None:
            return None, 0.0
        if self.s > stop_line_s:  # already committed into the box
            return None, 0.0
        colour = lights.state(group, t)
        if colour is SignalState.GREEN:
            return None, 0.0
        gap = stop_line_s - self.s - 0.5 * self.length
        if colour is SignalState.YELLOW and gap < 0.5 * self.v**2 / self.idm.b:
            return None, 0.0  # cannot stop comfortably; clear the box
        return max(gap, 0.05), self.v

    def step(
        self,
        dt: float,
        t: float,
        leader_gap: float | None = None,
        leader_dv: float = 0.0,
        lights: TrafficLightController | None = None,
    ) -> ActorState:
        light_gap, light_dv = self._virtual_leader_gap(t, lights)
        # Whichever constraint binds first wins; both are car-following terms.
        candidates = [(leader_gap, leader_dv), (light_gap, light_dv)]
        a = min(
            idm_acceleration(self.v, g, dv, self.idm) for g, dv in candidates
        )
        self._a = a

        # Pure pursuit on the route, then a kinematic bicycle step.
        s_now = self.path.project(self._x, self._y, self.s)
        l_d = float(np.clip(0.6 * self.v + 4.0, 3.0, 20.0))
        target = self.path.position(min(s_now + l_d, self.path.length))
        dx, dy = target[0] - self._x, target[1] - self._y
        alpha = wrap_to_pi(np.arctan2(dy, dx) - self._psi)
        dist = max(float(np.hypot(dx, dy)), 1e-3)
        kappa = 2.0 * np.sin(alpha) / dist
        delta = float(np.arctan(kappa * self.wheelbase))

        def f(x: np.ndarray, u: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    x[3] * np.cos(x[2]),
                    x[3] * np.sin(x[2]),
                    x[3] / self.wheelbase * np.tan(u[1]),
                    u[0],
                ]
            )

        x = np.array([self._x, self._y, self._psi, self.v])
        u = np.array([a, delta])
        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        self._x, self._y, self._psi = float(x[0]), float(x[1]), wrap_to_pi(x[2])
        self.v = max(float(x[3]), 0.0)
        self.s = self.path.project(self._x, self._y, s_now)
        if self.s >= self.path.length - 1e-3:
            self.done = True
        return self.state()


# --- sync sources -------------------------------------------------------------

class SyncSource(ABC):
    """Supplies the set of other actors at each simulation tick."""

    @abstractmethod
    def reset(self, seed: int = 0) -> None: ...

    @abstractmethod
    def step(self, t: float, dt: float, ego: ActorState | None = None) -> list[ActorState]:
        """Advance to time ``t + dt`` and return the actor states at that time."""

    def actors(self) -> list[ActorState]:  # pragma: no cover - convenience
        return []


class SimulatedTrafficSource(SyncSource):
    """Reactive traffic: IDM car-following plus signal compliance.

    Leaders are resolved **per path**: an actor follows the nearest actor ahead
    of it on the same route.  Cross-traffic on a conflicting route is *not*
    treated as a leader, because a driver does not brake for a car that is
    merely near -- which is precisely why an intersection needs a right-of-way
    rule rather than car-following alone.
    """

    def __init__(
        self,
        actors: Sequence[TrafficActor],
        lights: TrafficLightController | None = None,
        remove_when_done: bool = True,
    ):
        self._template = list(actors)
        self.lights = lights
        self.remove_when_done = remove_when_done
        self._actors: list[TrafficActor] = []
        self.reset()

    def reset(self, seed: int = 0) -> None:
        for a in self._template:
            a.reset()
        self._actors = list(self._template)

    def _leader_for(self, actor: TrafficActor) -> tuple[float | None, float]:
        best_gap, best_dv = None, 0.0
        for other in self._actors:
            if other is actor or other.route_id != actor.route_id:
                continue
            ds = other.s - actor.s
            if ds <= 0:
                continue
            gap = ds - 0.5 * (actor.length + other.length)
            if best_gap is None or gap < best_gap:
                best_gap, best_dv = gap, actor.v - other.v
        return best_gap, best_dv

    def step(self, t: float, dt: float, ego: ActorState | None = None) -> list[ActorState]:
        out = []
        for actor in self._actors:
            gap, dv = self._leader_for(actor)
            out.append(actor.step(dt, t, gap, dv, self.lights))
        if self.remove_when_done:
            keep = [a for a in self._actors if not a.done]
            if len(keep) != len(self._actors):
                self._actors = keep
                out = [a.state() for a in self._actors]
        return out

    def actors(self) -> list[ActorState]:
        return [a.state() for a in self._actors]


class ScriptedSyncSource(SyncSource):
    """Open-loop actors whose state is an explicit function of time.

    Use for adversarial events that must be *identical* on every run: a cut-in
    at a fixed time, a lead vehicle braking at a fixed deceleration, a
    pedestrian stepping out at a fixed distance.  Nothing the ego does changes
    what happens, which is the point -- the scenario tests the ego's response,
    not its ability to deter the other driver.
    """

    def __init__(self, scripts: dict[str, Callable[[float], ActorState]], t_window: dict[str, tuple[float, float]] | None = None):
        self.scripts = scripts
        self.t_window = t_window or {}
        self._latest: list[ActorState] = []

    def reset(self, seed: int = 0) -> None:
        self._latest = []

    def step(self, t: float, dt: float, ego: ActorState | None = None) -> list[ActorState]:
        now = t + dt
        out = []
        for name, fn in self.scripts.items():
            lo, hi = self.t_window.get(name, (-np.inf, np.inf))
            if lo <= now <= hi:
                out.append(fn(now))
        self._latest = out
        return out

    def actors(self) -> list[ActorState]:
        return list(self._latest)


class ReplaySyncSource(SyncSource):
    """Plays back recorded trajectories, interpolating between samples.

    ``tracks`` maps an actor id to an ``(N, 5)`` array of
    ``[t, x, y, psi, v]``.  Heading is interpolated on the unwrapped angle so a
    log that crosses ``+-pi`` does not produce a spurious full rotation.
    """

    def __init__(self, tracks: dict[str, np.ndarray], meta: dict[str, dict] | None = None):
        self.tracks = {k: np.asarray(v, dtype=float) for k, v in tracks.items()}
        for k, v in self.tracks.items():
            if v.ndim != 2 or v.shape[1] != 5:
                raise ValueError(f"track {k!r} must be an (N, 5) array of [t, x, y, psi, v]")
        self.meta = meta or {}
        self._latest: list[ActorState] = []

    def reset(self, seed: int = 0) -> None:
        self._latest = []

    def step(self, t: float, dt: float, ego: ActorState | None = None) -> list[ActorState]:
        now = t + dt
        out = []
        for name, tr in self.tracks.items():
            if now < tr[0, 0] or now > tr[-1, 0]:
                continue
            x = float(np.interp(now, tr[:, 0], tr[:, 1]))
            y = float(np.interp(now, tr[:, 0], tr[:, 2]))
            psi = float(wrap_to_pi(np.interp(now, tr[:, 0], np.unwrap(tr[:, 3]))))
            v = float(np.interp(now, tr[:, 0], tr[:, 4]))
            out.append(ActorState(id=name, x=x, y=y, psi=psi, v=v, **self.meta.get(name, {})))
        self._latest = out
        return out

    def actors(self) -> list[ActorState]:
        return list(self._latest)


class CompositeSyncSource(SyncSource):
    """Merge several sources; ids must be unique across them."""

    def __init__(self, sources: Sequence[SyncSource]):
        self.sources = list(sources)

    def reset(self, seed: int = 0) -> None:
        for s in self.sources:
            s.reset(seed)

    def step(self, t: float, dt: float, ego: ActorState | None = None) -> list[ActorState]:
        out: list[ActorState] = []
        seen: set[str] = set()
        for s in self.sources:
            for a in s.step(t, dt, ego):
                if a.id in seen:
                    raise ValueError(f"duplicate actor id {a.id!r} across sync sources")
                seen.add(a.id)
                out.append(a)
        return out

    def actors(self) -> list[ActorState]:
        return [a for s in self.sources for a in s.actors()]
