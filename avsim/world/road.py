"""Lanes, roads and the lateral bounds a planner turns into a box constraint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .path import ReferencePath


@dataclass
class Lane:
    """One lane: a centerline, a width, a speed limit and its connectivity.

    The ``successors`` list is what makes a route search possible; a lane with
    no successors is a sink and a vehicle reaching its end must be removed
    rather than allowed to drive off the end of its own path.
    """

    id: str
    centerline: ReferencePath
    width: float = 3.5
    speed_limit: float = 13.9  #: 50 km/h
    successors: list[str] = field(default_factory=list)
    predecessors: list[str] = field(default_factory=list)
    #: free text: "approach", "exit", "connector_straight", "connector_left", ...
    kind: str = "lane"
    #: for connectors: which approach this lane belongs to ("E", "W", "N", "S")
    approach: str | None = None

    @property
    def length(self) -> float:
        return self.centerline.length

    def bounds(self, s: float, margin: float = 0.0) -> tuple[float, float]:
        """Lateral box constraint ``(e_y_min, e_y_max)`` at arc length ``s``.

        This is the benefit the Frenet frame was adopted for: a road boundary
        becomes two numbers instead of a distance to an arbitrary polygon.
        ``margin`` shrinks the corridor by the vehicle half-width plus whatever
        safety buffer the planner wants.
        """
        half = 0.5 * self.width - margin
        return (-half, half)

    def contains(self, s: float, e_y: float, margin: float = 0.0) -> bool:
        lo, hi = self.bounds(s, margin)
        return bool(lo <= e_y <= hi and 0.0 <= s <= self.length)

    def sample(self, ds: float = 1.0) -> np.ndarray:
        return self.centerline.sample(ds)

    def edges(self, ds: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Left and right lane-edge polylines, for rendering and collision tests."""
        n = max(int(self.length / ds) + 1, 2)
        ss = np.linspace(0.0, self.length, n)
        left = np.array([self.centerline.to_cartesian(s, 0.5 * self.width) for s in ss])
        right = np.array([self.centerline.to_cartesian(s, -0.5 * self.width) for s in ss])
        return left, right


@dataclass
class Road:
    """A bundle of parallel lanes carrying traffic in the same direction."""

    id: str
    lanes: list[Lane]
    direction: str = ""  #: "E", "W", "N", "S" for the crossroads layout

    def lane_by_index(self, i: int) -> Lane:
        return self.lanes[i]

    @property
    def width(self) -> float:
        return float(sum(l.width for l in self.lanes))


@dataclass
class StopLine:
    """A stop line: where to halt for a red light or a right-of-way rule."""

    lane_id: str
    s: float                #: arc length along that lane
    controlled_by: str | None = None  #: traffic-light id, or None for a stop sign

    def distance_from(self, s_ego: float) -> float:
        """Signed distance to the line; negative once it has been crossed."""
        return float(self.s - s_ego)
