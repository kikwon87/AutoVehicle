"""Random background traffic on a grid, with intersection right-of-way.

The request is specific about what this traffic is and is not: the vehicles
drive **on their own** -- no autonomy stack, no coordination with the ego --
but they do not deliberately collide, and they may stop and set off again.

That is three rules, and each is implemented as a *virtual leader* so the same
IDM car-following law produces all of them:

1. **Car following** -- the nearest vehicle ahead on the same lane, or on the
   next lane of the route when the current one runs out.  Lane-indexed, so it
   costs O(n) rather than an O(n^2) sweep of path projections.
2. **Signals** -- a red or a non-clearable yellow is a vehicle parked on the
   stop line (:class:`avsim.world.actors.TrafficActor` already does this, now
   over a whole route's worth of lights).
3. **Right of way** -- a vehicle may not enter an intersection box while a
   vehicle from a *different* approach is inside it, and a left-turner also
   yields to oncoming traffic close enough to the box to cross in front of it.
   Both are the same signal phase, so the light cannot separate them; without
   this rule the grid produces broadside collisions between vehicles that both
   have a green.

Vehicles are removed when their route ends and replaced immediately, so the
population stays at the requested count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .actors import ActorState, IDMParams, SyncSource, TrafficActor, idm_acceleration
from .grid import OPPOSITE, GridLayout, GridSignals


@dataclass
class ForcedVehicle:
    """A vehicle placed exactly where a scenario needs it."""

    node: str
    direction: str
    plan: tuple[str, ...]
    s0: float
    v: float
    v0: float = 12.0
    lane: int = 0
    headway: float = 1.4
    id: str | None = None


@dataclass
class TrafficConfig:
    """Population and driver-behaviour spread for the background traffic."""

    n_vehicles: int = 12
    #: desired speed drawn from ``U(v0_min, v0_max)`` per driver [m/s]
    v0_min: float = 8.0
    v0_max: float = 14.0
    #: time headway spread [s]
    headway_min: float = 1.1
    headway_max: float = 1.9
    a_max: float = 1.6
    b_comfort: float = 2.2
    #: route length in intersections
    route_min_nodes: int = 2
    route_max_nodes: int = 5
    #: seconds of margin a left-turner demands from oncoming traffic
    left_turn_gap: float = 4.0
    #: how far back from a stop line a vehicle starts checking the box [m]
    yield_lookahead: float = 30.0
    #: minimum spacing required to spawn a vehicle at an entry [m]
    spawn_clearance: float = 18.0
    length: float = 4.6
    width: float = 1.85
    wheelbase: float = 2.7


class GridTrafficSource(SyncSource):
    """``n`` self-driving-in-the-colloquial-sense vehicles on a grid."""

    def __init__(
        self,
        layout: GridLayout,
        config: TrafficConfig | None = None,
        seed: int = 0,
        signals: GridSignals | None = None,
        forced: Sequence["ForcedVehicle"] = (),
    ):
        self.layout = layout
        self.cfg = config or TrafficConfig()
        #: Vehicles placed deterministically before the random fill.  A scenario
        #: that is *about* an interaction -- an oncoming stream for a left turn,
        #: a slow car to overtake -- cannot leave that interaction to a random
        #: draw and still be the same test twice.
        self.forced = list(forced)
        self.signals = signals if signals is not None else layout.signals
        self.seed = int(seed)
        self._actors: list[TrafficActor] = []
        self._next_id = 0
        self.reset(seed)

    # --- lifecycle -------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._actors = []
        self._next_id = 0
        for spec in self.forced:
            self._spawn_forced(spec)
        for _ in range(max(self.cfg.n_vehicles - len(self._actors), 0)):
            self._spawn(initial=True)

    # --- spawning ---------------------------------------------------------------

    def _make_route(self):
        lane_ids, node, direction = self.layout.random_route(
            self.rng, self.cfg.route_min_nodes, self.cfg.route_max_nodes
        )
        return lane_ids, self.layout.route_path(lane_ids)

    def _spawn(self, initial: bool = False) -> TrafficActor | None:
        cfg = self.cfg
        for _ in range(12):  # a few tries before giving up this tick
            lane_ids, path = self._make_route()
            # An initial fill is scattered along the route; a replacement enters
            # at the boundary, which is where traffic comes from.
            s0 = float(self.rng.uniform(0.0, path.length * 0.75)) if initial else 0.0
            p = path.position(s0)
            if any(
                float(np.hypot(a._x - p[0], a._y - p[1])) < cfg.spawn_clearance
                for a in self._actors
            ):
                continue

            self._next_id += 1
            v0 = float(self.rng.uniform(cfg.v0_min, cfg.v0_max))
            starts = np.cumsum([0.0] + [self.layout.network.lanes[i].length for i in lane_ids[:-1]])
            actor = TrafficActor(
                id=f"t{self._next_id}",
                path=path,
                s=s0,
                v=v0 * float(self.rng.uniform(0.6, 1.0)),
                idm=IDMParams(
                    v0=v0,
                    T=float(self.rng.uniform(cfg.headway_min, cfg.headway_max)),
                    a_max=cfg.a_max,
                    b=cfg.b_comfort,
                ),
                length=cfg.length,
                width=cfg.width,
                wheelbase=cfg.wheelbase,
                route_id=f"r{self._next_id}",
                route_signals=[(s, g) for s, g, _ in self.layout.route_signals(lane_ids)],
                lane_sequence=list(lane_ids),
                lane_starts=list(starts),
            )
            self._actors.append(actor)
            return actor
        return None

    def _spawn_forced(self, spec: "ForcedVehicle") -> TrafficActor:
        lane_ids = self.layout.route_lanes(spec.node, spec.direction, spec.plan, spec.lane)
        path = self.layout.route_path(lane_ids)
        self._next_id += 1
        starts = np.cumsum([0.0] + [self.layout.network.lanes[i].length for i in lane_ids[:-1]])
        actor = TrafficActor(
            id=spec.id or f"f{self._next_id}",
            path=path,
            s=float(np.clip(spec.s0, 0.0, path.length - 1.0)),
            v=float(spec.v),
            idm=IDMParams(v0=float(spec.v0), T=spec.headway, a_max=self.cfg.a_max,
                          b=self.cfg.b_comfort),
            length=self.cfg.length, width=self.cfg.width, wheelbase=self.cfg.wheelbase,
            route_id=f"forced{self._next_id}",
            route_signals=[(s, g) for s, g, _ in self.layout.route_signals(lane_ids)],
            lane_sequence=list(lane_ids),
            lane_starts=list(starts),
        )
        self._actors.append(actor)
        return actor

    # --- interaction rules --------------------------------------------------------

    def _lane_occupancy(self) -> dict[str, list[tuple[TrafficActor, float]]]:
        """lane id -> [(actor, arc length within that lane)], for O(n) lookups."""
        out: dict[str, list[tuple[TrafficActor, float]]] = {}
        for a in self._actors:
            lane = a.current_lane()
            if lane is None:
                continue
            i = a.lane_sequence.index(lane) if lane in a.lane_sequence else 0
            s_local = a.s - (a.lane_starts[i] if a.lane_starts else 0.0)
            out.setdefault(lane, []).append((a, s_local))
        return out

    def _ego_gap(
        self, actor: TrafficActor, ego: ActorState | None
    ) -> tuple[float, float] | None:
        """Gap and closing rate to the **ego**, if it is in this actor's way.

        Traffic knows nothing about the ego's route, so this is geometric rather
        than lane-based: the ego's position in the actor's own frame, against a
        corridor wide enough to cover it at any orientation.  The leader speed is
        the ego's velocity *along the actor's heading*, which makes a crossing
        vehicle read as a slow obstacle and a stopped one as a wall.

        Without it the ego is invisible to the traffic model and gets rear-ended
        while stopped at a red light -- which is not a failure of the ego's
        driving, and scoring it as one makes every safety number meaningless.
        """
        if ego is None:
            return None
        c, s_ = np.cos(actor._psi), np.sin(actor._psi)
        dx, dy = ego.x - actor._x, ego.y - actor._y
        forward = dx * c + dy * s_
        lateral = -dx * s_ + dy * c
        if forward <= 0.0:
            return None
        # Half-width that covers the ego whatever way it is pointing.
        half = 0.5 * actor.width + 0.5 * float(np.hypot(ego.length, ego.width))
        if abs(lateral) > half:
            return None
        gap = forward - 0.5 * (actor.length + ego.length)
        if gap > self.cfg.yield_lookahead * 1.5:
            return None
        v_along = float(ego.v) * float(np.cos(ego.psi - actor._psi))
        return max(gap, 0.05), actor.v - max(v_along, 0.0)

    def _leader_gap(
        self, actor: TrafficActor, occupancy: dict[str, list[tuple[TrafficActor, float]]]
    ) -> tuple[float | None, float]:
        """Gap and closing rate to the nearest vehicle ahead on the route."""
        lane = actor.current_lane()
        if lane is None or lane not in actor.lane_sequence:
            return None, 0.0
        idx = actor.lane_sequence.index(lane)
        s_local = actor.s - actor.lane_starts[idx]
        lane_len = self.layout.network.lanes[lane].length

        best_gap, best_dv = None, 0.0
        # Same lane, ahead of us.
        for other, s_other in occupancy.get(lane, ()):
            if other is actor or s_other <= s_local:
                continue
            gap = s_other - s_local - 0.5 * (actor.length + other.length)
            if best_gap is None or gap < best_gap:
                best_gap, best_dv = gap, actor.v - other.v
        if best_gap is not None:
            return max(best_gap, 0.05), best_dv

        # Nothing ahead here: look into the next lane of our own route.
        if idx + 1 < len(actor.lane_sequence):
            nxt = actor.lane_sequence[idx + 1]
            remaining = lane_len - s_local
            for other, s_other in occupancy.get(nxt, ()):
                if other is actor:
                    continue
                gap = remaining + s_other - 0.5 * (actor.length + other.length)
                if best_gap is None or gap < best_gap:
                    best_gap, best_dv = gap, actor.v - other.v
        return (max(best_gap, 0.05), best_dv) if best_gap is not None else (None, 0.0)

    @staticmethod
    def _connector_info(lane_id: str) -> tuple[str, str, str] | None:
        """``(node, approach, manoeuvre)`` for a connector lane, else ``None``."""
        parts = lane_id.split("_")
        # "<node c>_<node r>_<dir>_<man>[_i]" with node = "n{c}_{r}"
        if len(parts) < 4 or not parts[0].startswith("n"):
            return None
        node = f"{parts[0]}_{parts[1]}"
        direction, man = parts[2], parts[3]
        if man not in ("straight", "left", "right"):
            return None
        return node, direction, man

    def _right_of_way_gap(self, actor: TrafficActor) -> float | None:
        """Distance to the stop line if the actor must yield, else ``None``.

        Yield when another vehicle from a different approach is inside the box,
        and -- for a left turn -- when oncoming traffic is close enough to cross
        in front.  Both cases share a green phase, so the signal cannot
        separate them.
        """
        cfg = self.cfg
        lane = actor.current_lane()
        if lane is None or lane not in actor.lane_sequence:
            return None
        idx = actor.lane_sequence.index(lane)
        if idx + 1 >= len(actor.lane_sequence):
            return None
        nxt = actor.lane_sequence[idx + 1]
        info = self._connector_info(nxt)
        if info is None:
            return None  # the next lane is not a connector: not entering a box
        node, direction, manoeuvre = info

        stop_s = actor.lane_starts[idx] + self.layout.network.lanes[lane].length
        distance = stop_s - actor.s
        if distance > cfg.yield_lookahead or distance < 0.0:
            return None

        for other in self._actors:
            if other is actor:
                continue
            other_lane = other.current_lane()
            if other_lane is None:
                continue
            other_info = self._connector_info(other_lane)
            if other_info is not None and other_info[0] == node and other_info[1] != direction:
                # Someone from another approach is in the box.
                return max(distance - 0.5 * actor.length, 0.05)

            if manoeuvre == "left" and other_info is None:
                # Oncoming traffic that will reach the box before we clear it.
                o_stop, _ = self._next_box_arrival(other, node, OPPOSITE[direction])
                if o_stop is not None and o_stop < cfg.left_turn_gap * max(other.v, 1.0):
                    return max(distance - 0.5 * actor.length, 0.05)
        return None

    def _next_box_arrival(
        self, actor: TrafficActor, node: str, direction: str
    ) -> tuple[float | None, str | None]:
        """Distance from ``actor`` to ``node``'s box on approach ``direction``."""
        lane = actor.current_lane()
        if lane is None or lane not in actor.lane_sequence:
            return None, None
        idx = actor.lane_sequence.index(lane)
        if idx + 1 >= len(actor.lane_sequence):
            return None, None
        info = self._connector_info(actor.lane_sequence[idx + 1])
        if info is None or info[0] != node or info[1] != direction:
            return None, None
        stop_s = actor.lane_starts[idx] + self.layout.network.lanes[lane].length
        return max(stop_s - actor.s, 0.0), info[2]

    # --- stepping -------------------------------------------------------------------

    def step(self, t: float, dt: float, ego: ActorState | None = None) -> list[ActorState]:
        occupancy = self._lane_occupancy()
        out: list[ActorState] = []
        for actor in self._actors:
            gap, dv = self._leader_gap(actor, occupancy)
            yield_gap = self._right_of_way_gap(actor)
            if yield_gap is not None and (gap is None or yield_gap < gap):
                gap, dv = yield_gap, actor.v
            ego_gap = self._ego_gap(actor, ego)
            if ego_gap is not None and (gap is None or ego_gap[0] < gap):
                gap, dv = ego_gap
            out.append(actor.step(dt, t, gap, dv, self.signals))

        finished = [a for a in self._actors if a.done]
        if finished:
            self._actors = [a for a in self._actors if not a.done]
            for _ in finished:
                self._spawn()
            out = [a.state() for a in self._actors]
        return out

    def actors(self) -> list[ActorState]:
        return [a.state() for a in self._actors]

    def count(self) -> int:
        return len(self._actors)
