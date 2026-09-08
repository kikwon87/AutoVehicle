"""Road network: lanes, connectivity, and a four-way signalized intersection.

Geometry of the crossroads (네거리)
-----------------------------------
Right-hand traffic.  Four arms E, W, N, S meet at the origin.  Each arm carries
``n_lanes`` lanes per direction of width ``lane_width``; lane ``0`` is the
innermost (next to the centerline) and lane ``n-1`` hugs the curb.

The intersection box is a square of half-size

    ``box_half = n_lanes * lane_width + flare``

The ``flare`` term is not decoration.  With a square box and in-lane turning,
a right turn's radius is forced to be exactly half a lane width -- about 1.75 m
here -- which is **below the vehicle's minimum turning radius**
``L / tan(delta_max) = 3.86 m``.  Real intersections solve this by rounding the
curb; ``flare`` is that rounding, and it makes the right-turn connector a
clean 5.75 m arc that the reference car can actually drive.

Every connector is a **single circular arc whose radius is determined by the
lane geometry**, not chosen: given the entry and exit poses, ``R`` follows from
``R (n_in -+ n_out) = p_exit - p_entry``.  The builder solves that equation and
raises if the two components disagree, so a mis-specified layout fails loudly
instead of producing a path with a kink in it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ..core.conventions import wrap_to_pi
from .path import Arc, ConcatPath, PrimitivePath, ReferencePath, Straight
from .road import Lane, Road, StopLine

#: heading of each approach, in radians
DIRECTION_HEADING = {"E": 0.0, "N": math.pi / 2, "W": math.pi, "S": -math.pi / 2}
#: the direction reached by turning left / right from each approach
LEFT_OF = {"E": "N", "N": "W", "W": "S", "S": "E"}
RIGHT_OF = {"E": "S", "S": "W", "W": "N", "N": "E"}
#: which signal group controls each approach
SIGNAL_GROUP = {"E": "EW", "W": "EW", "N": "NS", "S": "NS"}


def _tangent(direction: str) -> np.ndarray:
    th = DIRECTION_HEADING[direction]
    return np.array([math.cos(th), math.sin(th)])


def _left_normal(direction: str) -> np.ndarray:
    th = DIRECTION_HEADING[direction]
    return np.array([-math.sin(th), math.cos(th)])


@dataclass
class RoadNetwork:
    """A set of lanes plus the connectivity and stop lines that tie them together."""

    lanes: dict[str, Lane] = field(default_factory=dict)
    roads: dict[str, Road] = field(default_factory=dict)
    stop_lines: dict[str, StopLine] = field(default_factory=dict)
    #: half-size of the intersection conflict box, or ``None`` for an open road
    box_half: float | None = None
    lane_width: float = 3.5
    n_lanes: int = 2

    def add_lane(self, lane: Lane) -> Lane:
        if lane.id in self.lanes:
            raise ValueError(f"duplicate lane id {lane.id!r}")
        self.lanes[lane.id] = lane
        return lane

    def connect(self, upstream: str, downstream: str) -> None:
        self.lanes[upstream].successors.append(downstream)
        self.lanes[downstream].predecessors.append(upstream)

    def route_path(self, lane_ids: Sequence[str]) -> ConcatPath:
        """Concatenate lanes into one reference path, validating continuity."""
        return ConcatPath([self.lanes[i].centerline for i in lane_ids])

    def route(self, approach: str, manoeuvre: str = "straight", lane_index: int | None = None) -> list[str]:
        """Lane ids for a manoeuvre through the intersection.

        ``manoeuvre`` is ``"straight"``, ``"left"`` or ``"right"``.  Left turns
        are made from the innermost lane and right turns from the curb lane,
        which is what the connector geometry was built for; asking for another
        lane raises rather than silently returning a route with a gap.
        """
        if manoeuvre == "straight":
            idx = 0 if lane_index is None else lane_index
            exit_dir = approach
        elif manoeuvre == "left":
            idx = 0
            exit_dir = LEFT_OF[approach]
        elif manoeuvre == "right":
            idx = self.n_lanes - 1
            exit_dir = RIGHT_OF[approach]
        else:
            raise ValueError(f"unknown manoeuvre {manoeuvre!r}")
        if lane_index is not None and lane_index != idx and manoeuvre != "straight":
            raise ValueError(
                f"a {manoeuvre} turn from {approach} uses lane {idx}, not {lane_index}"
            )
        connector = f"{approach}_{manoeuvre}"
        return [f"{approach}_in_{idx}", connector, f"{exit_dir}_out_{idx}"]

    def stop_line_for(self, lane_id: str) -> StopLine | None:
        return self.stop_lines.get(lane_id)

    def all_lane_polylines(self, ds: float = 1.0) -> dict[str, np.ndarray]:
        return {lid: lane.sample(ds) for lid, lane in self.lanes.items()}


def _arc_between(p0: np.ndarray, d_in: str, p1: np.ndarray, d_out: str, turn: str) -> Arc:
    """The unique circular arc joining two poses 90 degrees apart.

    Left turn: ``C = p0 + R n_in = p1 + R n_out``  =>  ``R (n_in - n_out) = p1 - p0``
    Right turn: ``C = p0 - R n_in = p1 - R n_out`` =>  ``R (n_out - n_in) = p1 - p0``

    Both components must give the same ``R``; if they do not, the lane offsets
    and the box size are inconsistent and the layout is wrong.
    """
    n_in, n_out = _left_normal(d_in), _left_normal(d_out)
    lhs = (n_in - n_out) if turn == "left" else (n_out - n_in)
    rhs = p1 - p0
    candidates = [rhs[k] / lhs[k] for k in range(2) if abs(lhs[k]) > 1e-9]
    if not candidates or max(candidates) - min(candidates) > 1e-6:
        raise ValueError(
            f"no single arc joins {d_in}->{d_out} ({turn}): radius candidates {candidates}; "
            "check lane offsets against box_half"
        )
    R = float(np.mean(candidates))
    if R <= 0:
        raise ValueError(f"non-positive turn radius {R:.3f} for {d_in}->{d_out}")
    kappa = (1.0 / R) if turn == "left" else (-1.0 / R)
    return Arc(float(p0[0]), float(p0[1]), DIRECTION_HEADING[d_in], R * math.pi / 2.0, kappa)


def four_way_intersection(
    n_lanes: int = 2,
    lane_width: float = 3.5,
    arm_length: float = 120.0,
    flare: float = 4.0,
    speed_limit: float = 13.9,
    min_turn_radius: float = 3.9,
) -> RoadNetwork:
    """Build a signalized four-way intersection.

    ``min_turn_radius`` is checked against every connector: a layout that
    demands a tighter turn than the vehicle can execute is rejected here rather
    than discovered as an unexplained tracking failure three modules later.
    """
    net = RoadNetwork(box_half=n_lanes * lane_width + flare, lane_width=lane_width, n_lanes=n_lanes)
    box = net.box_half
    assert box is not None

    def lane_offset(direction: str, i: int) -> np.ndarray:
        """Centre of lane ``i`` relative to the arm centerline (right of travel)."""
        return -(0.5 + i) * lane_width * _left_normal(direction)

    # --- approach and exit lanes ---------------------------------------------
    for d in DIRECTION_HEADING:
        t = _tangent(d)
        th = DIRECTION_HEADING[d]
        in_lanes, out_lanes = [], []
        for i in range(n_lanes):
            o = lane_offset(d, i)

            start = t * (-(box + arm_length)) + o
            approach = Lane(
                id=f"{d}_in_{i}",
                centerline=PrimitivePath([Straight(float(start[0]), float(start[1]), th, arm_length)]),
                width=lane_width,
                speed_limit=speed_limit,
                kind="approach",
                approach=d,
            )
            net.add_lane(approach)
            in_lanes.append(approach)
            net.stop_lines[approach.id] = StopLine(
                lane_id=approach.id, s=arm_length, controlled_by=f"signal_{SIGNAL_GROUP[d]}"
            )

            estart = t * box + o
            exit_lane = Lane(
                id=f"{d}_out_{i}",
                centerline=PrimitivePath([Straight(float(estart[0]), float(estart[1]), th, arm_length)]),
                width=lane_width,
                speed_limit=speed_limit,
                kind="exit",
                approach=d,
            )
            net.add_lane(exit_lane)
            out_lanes.append(exit_lane)

        net.roads[f"{d}_in"] = Road(id=f"{d}_in", lanes=in_lanes, direction=d)
        net.roads[f"{d}_out"] = Road(id=f"{d}_out", lanes=out_lanes, direction=d)

    # --- connectors -----------------------------------------------------------
    for d in DIRECTION_HEADING:
        t = _tangent(d)

        # straight: one connector per lane index
        for i in range(n_lanes):
            o = lane_offset(d, i)
            p0 = t * (-box) + o
            lane = Lane(
                id=f"{d}_straight" if i == 0 else f"{d}_straight_{i}",
                centerline=PrimitivePath(
                    [Straight(float(p0[0]), float(p0[1]), DIRECTION_HEADING[d], 2 * box)]
                ),
                width=lane_width,
                speed_limit=speed_limit,
                kind="connector_straight",
                approach=d,
            )
            net.add_lane(lane)
            net.connect(f"{d}_in_{i}", lane.id)
            net.connect(lane.id, f"{d}_out_{i}")

        # left turn: innermost lane to innermost lane
        d_left = LEFT_OF[d]
        p0 = t * (-box) + lane_offset(d, 0)
        p1 = _tangent(d_left) * box + lane_offset(d_left, 0)
        arc = _arc_between(p0, d, p1, d_left, "left")
        left = Lane(
            id=f"{d}_left",
            centerline=PrimitivePath([arc]),
            width=lane_width,
            speed_limit=speed_limit,
            kind="connector_left",
            approach=d,
        )
        net.add_lane(left)
        net.connect(f"{d}_in_0", left.id)
        net.connect(left.id, f"{d_left}_out_0")

        # right turn: curb lane to curb lane
        d_right = RIGHT_OF[d]
        j = n_lanes - 1
        p0 = t * (-box) + lane_offset(d, j)
        p1 = _tangent(d_right) * box + lane_offset(d_right, j)
        arc = _arc_between(p0, d, p1, d_right, "right")
        right = Lane(
            id=f"{d}_right",
            centerline=PrimitivePath([arc]),
            width=lane_width,
            speed_limit=speed_limit,
            kind="connector_right",
            approach=d,
        )
        net.add_lane(right)
        net.connect(f"{d}_in_{j}", right.id)
        net.connect(right.id, f"{d_right}_out_{j}")

    for lid, lane in net.lanes.items():
        if lane.kind.startswith("connector") and lane.kind != "connector_straight":
            R = 1.0 / abs(lane.centerline.curvature(lane.length / 2))
            if R < min_turn_radius:
                raise ValueError(
                    f"connector {lid} has radius {R:.2f} m, below the vehicle's "
                    f"minimum {min_turn_radius:.2f} m -- increase `flare`"
                )
    return net


def straight_road(
    length: float = 400.0,
    n_lanes: int = 2,
    lane_width: float = 3.5,
    speed_limit: float = 22.2,
    curvature: float = 0.0,
) -> RoadNetwork:
    """A single multi-lane road, straight or of constant curvature.

    Used by the lane-keeping, lane-change and lead-vehicle scenarios, where an
    intersection would only add noise to the measurement.
    """
    net = RoadNetwork(box_half=None, lane_width=lane_width, n_lanes=n_lanes)
    lanes = []
    for i in range(n_lanes):
        y = -(0.5 + i) * lane_width
        if abs(curvature) < 1e-9:
            path: ReferencePath = PrimitivePath([Straight(0.0, y, 0.0, length)])
        else:
            # Offset a constant-curvature arc: an inner lane is shorter, so the
            # arc length must be scaled by (1 - kappa * e_y) for the lanes to
            # stay parallel rather than diverge.
            scale = 1.0 - curvature * y
            path = PrimitivePath([Arc(0.0, y, 0.0, length * scale, curvature / scale)])
        lanes.append(
            Lane(
                id=f"lane_{i}",
                centerline=path,
                width=lane_width,
                speed_limit=speed_limit,
                kind="lane",
            )
        )
        net.add_lane(lanes[-1])
    net.roads["main"] = Road(id="main", lanes=lanes, direction="E")
    return net
