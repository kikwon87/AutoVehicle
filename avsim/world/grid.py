"""A grid road network -- the platform's default world.

The layout is a rectangular grid of roads: ``cols`` running east-west and
``rows`` running north-south, meeting at ``rows * cols`` signalized crossroads.
The literal 井 of the request is ``rows = cols = 2``; the default is ``3 x 3``,
which has a crossroads exactly at the centre of the map, and the shape is a
parameter rather than a constant so either reading is available.

Construction reuses the geometry of :func:`avsim.world.network.four_way_intersection`
at every node:

* each node owns a box of half-size ``box_half = n_lanes * lane_width + flare``;
* inside it, one straight connector per lane plus a left and a right turn, each
  the **unique arc** the lane offsets admit;
* between two adjacent nodes, the upstream exit stub and the downstream approach
  stub are each half the free span, so they meet exactly and a route through
  them is ``C^0`` in position and heading by construction;
* at the boundary the outward arms are longer, and they are where traffic is
  spawned and removed.

Each node carries its own two-phase signal, offset by a node-dependent amount so
the grid is not synchronized -- a synchronized grid makes every vehicle arrive
at every intersection in the same phase, which is a much easier world than the
one it is meant to represent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .network import (
    DIRECTION_HEADING,
    LEFT_OF,
    RIGHT_OF,
    SIGNAL_GROUP,
    RoadNetwork,
    _arc_between,
    _left_normal,
    _tangent,
)
from .path import ConcatPath, PrimitivePath, Straight
from .road import Lane, StopLine
from .traffic_light import SignalState, TrafficLightController

#: The direction a vehicle travels to move from one node to the next.
DELTA = {"E": (1, 0), "W": (-1, 0), "N": (0, 1), "S": (0, -1)}
OPPOSITE = {"E": "W", "W": "E", "N": "S", "S": "N"}
MANOEUVRES = ("straight", "left", "right")


def node_id(col: int, row: int) -> str:
    return f"n{col}_{row}"


@dataclass
class GridSignals:
    """One two-phase controller per node, addressed as ``"<node>:<group>"``.

    The :class:`avsim.world.world.World` and :class:`avsim.autonomy.stack.AutonomyStack`
    both accept anything with ``state(group, t)``; this adds ``groups`` so a
    renderer can enumerate the lights without knowing the layout.
    """

    controllers: dict[str, TrafficLightController] = field(default_factory=dict)

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(f"{n}:{g}" for n in self.controllers for g in ("NS", "EW"))

    def _split(self, group: str) -> tuple[TrafficLightController, str]:
        node, _, phase = group.partition(":")
        return self.controllers[node], phase

    def state(self, group: str, t: float) -> SignalState:
        ctl, phase = self._split(group)
        return ctl.state(phase, t)

    def time_to_change(self, group: str, t: float) -> float:
        ctl, phase = self._split(group)
        return ctl.time_to_change(phase, t)

    def time_until_green(self, group: str, t: float) -> float:
        ctl, phase = self._split(group)
        return ctl.time_until_green(phase, t)

    def will_be_green_at(self, group: str, t: float, horizon: float) -> bool:
        ctl, phase = self._split(group)
        return ctl.will_be_green_at(phase, t, horizon)


@dataclass
class GridLayout:
    """The network, its node geometry and its signals."""

    network: RoadNetwork
    signals: GridSignals
    rows: int
    cols: int
    spacing: float
    box_half: float
    arm_length: float
    n_lanes: int
    lane_width: float
    #: node id -> centre position
    centres: dict[str, np.ndarray] = field(default_factory=dict)

    # --- addressing ----------------------------------------------------------

    def node(self, col: int, row: int) -> str:
        return node_id(col, row)

    def coords(self, node: str) -> tuple[int, int]:
        c, _, r = node[1:].partition("_")
        return int(c), int(r)

    def neighbour(self, node: str, direction: str) -> str | None:
        """The node reached by travelling ``direction`` from ``node``."""
        c, r = self.coords(node)
        dc, dr = DELTA[direction]
        c2, r2 = c + dc, r + dr
        if 0 <= c2 < self.cols and 0 <= r2 < self.rows:
            return node_id(c2, r2)
        return None

    def signal_group(self, node: str, approach: str) -> str:
        return f"{node}:{SIGNAL_GROUP[approach]}"

    def entry_lanes(self) -> list[tuple[str, str, int]]:
        """``(node, direction, lane)`` for every boundary entry into the grid.

        These are where traffic can be spawned: the long outward arms whose
        upstream end is off the map.
        """
        out = []
        for node in self.centres:
            for d in DIRECTION_HEADING:
                if self.neighbour(node, OPPOSITE[d]) is None:
                    for i in range(self.n_lanes):
                        out.append((node, d, i))
        return out

    # --- routes ---------------------------------------------------------------

    def allowed_manoeuvres(self, lane: int) -> tuple[str, ...]:
        """Manoeuvres a vehicle in ``lane`` may make without changing lane.

        Left turns leave from the innermost lane and right turns from the curb
        lane, because that is where the connectors start.  A route that turns
        right from an inner lane is not merely unusual -- there is no such lane
        in the network, and building it produces a path with a lane-width jump
        in the middle and no other symptom.  With one lane per direction the two
        roles coincide and everything is allowed.
        """
        out = ["straight"]
        if lane == 0:
            out.append("left")
        if lane == self.n_lanes - 1:
            out.append("right")
        return tuple(out)

    def _lane_index_for(self, manoeuvre: str, lane: int) -> int:
        if manoeuvre == "left":
            return 0
        if manoeuvre == "right":
            return self.n_lanes - 1
        return lane

    def leg(self, node: str, direction: str, manoeuvre: str, lane: int = 0) -> list[str]:
        """Lane ids for crossing ``node`` while travelling ``direction``.

        Returns the approach lane, the connector and the exit lane.  Left turns
        are made from the innermost lane and right turns from the curb lane,
        because that is what the connector geometry was built for.
        """
        i = self._lane_index_for(manoeuvre, lane)
        exit_dir = {
            "straight": direction, "left": LEFT_OF[direction], "right": RIGHT_OF[direction]
        }[manoeuvre]
        connector = f"{node}_{direction}_{manoeuvre}" if manoeuvre != "straight" else (
            f"{node}_{direction}_straight" if i == 0 else f"{node}_{direction}_straight_{i}"
        )
        return [f"{node}_{direction}_in_{i}", connector, f"{node}_{exit_dir}_out_{i}"]

    def route_lanes(
        self, start_node: str, start_dir: str, plan: Sequence[str], lane: int = 0
    ) -> list[str]:
        """Lane ids for a sequence of manoeuvres starting at a boundary entry.

        ``plan`` is one manoeuvre per node visited.  The walk stops early if it
        would leave the grid, so a plan longer than the grid is simply truncated
        rather than raising -- which is what a random walker needs.
        """
        node, direction = start_node, start_dir
        lanes: list[str] = []
        for man in plan:
            if man not in self.allowed_manoeuvres(lane):
                raise ValueError(
                    f"a {man} turn is not available from lane {lane} of "
                    f"{self.n_lanes}; allowed here: {self.allowed_manoeuvres(lane)}"
                )
            i = self._lane_index_for(man, lane)
            leg = self.leg(node, direction, man, lane)
            if lanes:
                # The previous exit lane and this approach lane are the same
                # physical stub only when they connect; drop the duplicate.
                lanes = lanes[:-1] if lanes[-1] == leg[0] else lanes
            lanes.extend(leg if not lanes or lanes[-1] != leg[0] else leg[1:])
            exit_dir = {
                "straight": direction, "left": LEFT_OF[direction], "right": RIGHT_OF[direction]
            }[man]
            nxt = self.neighbour(node, exit_dir)
            if nxt is None:
                break
            # Hand over to the next node: its approach lane *is* our exit lane's
            # continuation, so replace the exit stub with the through pair.
            lanes = lanes[:-1]
            lanes.append(f"{node}_{exit_dir}_out_{i}")
            node, direction, lane = nxt, exit_dir, i
        return lanes

    def route_path(self, lane_ids: Sequence[str]) -> ConcatPath:
        return self.network.route_path(lane_ids)

    def route_signals(self, lane_ids: Sequence[str]) -> list[tuple[float, str, str]]:
        """``(arc length of the stop line, signal group, node)`` along a route.

        The autonomy stack uses this to know which light applies next; without
        it a vehicle crossing a grid can only ever obey one intersection.
        """
        out: list[tuple[float, str, str]] = []
        s0 = 0.0
        for lid in lane_ids:
            lane = self.network.lanes[lid]
            stop = self.network.stop_lines.get(lid)
            if stop is not None and stop.controlled_by:
                out.append((s0 + stop.s, stop.controlled_by, lane.approach or ""))
            s0 += lane.length
        return out

    def random_route(
        self, rng: np.random.Generator, min_nodes: int = 2, max_nodes: int = 4,
        start: tuple[str, str, int] | None = None,
    ) -> tuple[list[str], str, str]:
        """A random legal route from a boundary entry, as ``(lane ids, node, dir)``."""
        entries = self.entry_lanes()
        node, direction, lane = start if start is not None else entries[rng.integers(len(entries))]
        n = int(rng.integers(min_nodes, max_nodes + 1))
        # The plan has to be built lane by lane: which manoeuvres are available
        # depends on the lane the previous one left us in.
        plan: list[str] = []
        cur_node, cur_dir, cur_lane = node, direction, lane
        for _ in range(n):
            options = self.allowed_manoeuvres(cur_lane)
            man = options[int(rng.integers(len(options)))]
            plan.append(man)
            exit_dir = {
                "straight": cur_dir, "left": LEFT_OF[cur_dir], "right": RIGHT_OF[cur_dir]
            }[man]
            cur_lane = self._lane_index_for(man, cur_lane)
            nxt = self.neighbour(cur_node, exit_dir)
            if nxt is None:
                break
            cur_node, cur_dir = nxt, exit_dir
        return self.route_lanes(node, direction, plan, lane), node, direction


def grid_network(
    rows: int = 3,
    cols: int = 3,
    spacing: float = 150.0,
    n_lanes: int = 1,
    lane_width: float = 3.5,
    flare: float = 4.0,
    arm_length: float = 80.0,
    speed_limit: float = 13.9,
    green: float = 20.0,
    yellow: float = 3.0,
    all_red: float = 2.0,
    signal_stagger: float = 13.0,
    min_turn_radius: float = 3.9,
) -> GridLayout:
    """Build the grid.

    ``spacing`` is the centre-to-centre distance between adjacent crossroads.
    It must leave a positive free span between the boxes, so
    ``spacing > 2 * box_half``; the check is explicit because a grid whose boxes
    overlap produces routes with negative-length lanes and no obvious symptom.
    """
    box_half = n_lanes * lane_width + flare
    link = spacing - 2.0 * box_half
    if link <= 10.0:
        raise ValueError(
            f"spacing {spacing:.1f} m leaves only {link:.1f} m between "
            f"intersection boxes of half-size {box_half:.1f} m; increase spacing "
            f"or reduce n_lanes/flare"
        )

    net = RoadNetwork(box_half=box_half, lane_width=lane_width, n_lanes=n_lanes)
    signals = GridSignals()
    layout = GridLayout(
        network=net, signals=signals, rows=rows, cols=cols, spacing=spacing,
        box_half=box_half, arm_length=arm_length, n_lanes=n_lanes, lane_width=lane_width,
    )

    for c in range(cols):
        for r in range(rows):
            centre = np.array([
                (c - (cols - 1) / 2.0) * spacing,
                (r - (rows - 1) / 2.0) * spacing,
            ])
            layout.centres[node_id(c, r)] = centre

    def lane_offset(direction: str, i: int) -> np.ndarray:
        return -(0.5 + i) * lane_width * _left_normal(direction)

    # --- stubs ----------------------------------------------------------------
    for node, centre in layout.centres.items():
        for d, th in DIRECTION_HEADING.items():
            t = _tangent(d)
            has_upstream = layout.neighbour(node, OPPOSITE[d]) is not None
            has_downstream = layout.neighbour(node, d) is not None
            len_in = link / 2.0 if has_upstream else arm_length
            len_out = link / 2.0 if has_downstream else arm_length

            for i in range(n_lanes):
                o = lane_offset(d, i)
                start = centre + t * (-(box_half + len_in)) + o
                approach = Lane(
                    id=f"{node}_{d}_in_{i}",
                    centerline=PrimitivePath(
                        [Straight(float(start[0]), float(start[1]), th, len_in)]
                    ),
                    width=lane_width, speed_limit=speed_limit,
                    kind="approach", approach=d,
                )
                net.add_lane(approach)
                net.stop_lines[approach.id] = StopLine(
                    lane_id=approach.id, s=len_in,
                    controlled_by=f"{node}:{SIGNAL_GROUP[d]}",
                )

                estart = centre + t * box_half + o
                net.add_lane(Lane(
                    id=f"{node}_{d}_out_{i}",
                    centerline=PrimitivePath(
                        [Straight(float(estart[0]), float(estart[1]), th, len_out)]
                    ),
                    width=lane_width, speed_limit=speed_limit,
                    kind="exit", approach=d,
                ))

    # --- connectors -----------------------------------------------------------
    for node, centre in layout.centres.items():
        for d in DIRECTION_HEADING:
            t = _tangent(d)

            for i in range(n_lanes):
                o = lane_offset(d, i)
                p0 = centre + t * (-box_half) + o
                lane = Lane(
                    id=f"{node}_{d}_straight" if i == 0 else f"{node}_{d}_straight_{i}",
                    centerline=PrimitivePath(
                        [Straight(float(p0[0]), float(p0[1]), DIRECTION_HEADING[d], 2 * box_half)]
                    ),
                    width=lane_width, speed_limit=speed_limit,
                    kind="connector_straight", approach=d,
                )
                net.add_lane(lane)
                net.connect(f"{node}_{d}_in_{i}", lane.id)
                net.connect(lane.id, f"{node}_{d}_out_{i}")

            for turn, exit_dir, idx in (
                ("left", LEFT_OF[d], 0),
                ("right", RIGHT_OF[d], n_lanes - 1),
            ):
                p0 = centre + t * (-box_half) + lane_offset(d, idx)
                p1 = centre + _tangent(exit_dir) * box_half + lane_offset(exit_dir, idx)
                arc = _arc_between(p0 - centre, d, p1 - centre, exit_dir, turn)
                # _arc_between works in the node's own frame; shift it back.
                arc = type(arc)(
                    x=float(arc.x + centre[0]), y=float(arc.y + centre[1]),
                    theta=arc.theta, length=arc.length, kappa=arc.kappa,
                )
                lane = Lane(
                    id=f"{node}_{d}_{turn}",
                    centerline=PrimitivePath([arc]),
                    width=lane_width, speed_limit=speed_limit,
                    kind=f"connector_{turn}", approach=d,
                )
                net.add_lane(lane)
                net.connect(f"{node}_{d}_in_{idx}", lane.id)
                net.connect(lane.id, f"{node}_{exit_dir}_out_{idx}")

                R = 1.0 / abs(arc.kappa)
                if R < min_turn_radius:
                    raise ValueError(
                        f"connector {lane.id} has radius {R:.2f} m, below the "
                        f"vehicle's minimum {min_turn_radius:.2f} m -- increase `flare`"
                    )

    # --- inter-node links ------------------------------------------------------
    for node in layout.centres:
        for d in DIRECTION_HEADING:
            nxt = layout.neighbour(node, d)
            if nxt is None:
                continue
            for i in range(n_lanes):
                net.connect(f"{node}_{d}_out_{i}", f"{nxt}_{d}_in_{i}")

    # --- signals ----------------------------------------------------------------
    for node in layout.centres:
        c, r = layout.coords(node)
        signals.controllers[node] = TrafficLightController(
            green=green, yellow=yellow, all_red=all_red,
            # Stagger by node so the grid is not one synchronized light.
            offset=((c * 2 + r * 3) * signal_stagger) % (2 * (green + yellow + all_red)),
        )
    return layout
