"""Test presets: a named situation, reproducible from a seed.

Every preset is built on the **same grid world**, because a platform whose
scenarios live in different worlds cannot compare the numbers they produce.
A preset therefore says only three things: how the grid is shaped, where the
ego starts and where it is going, and what traffic is placed in its way.

Traffic comes in two parts and the split is deliberate:

* **forced** vehicles are the situation -- the oncoming stream a left turn must
  yield to, the slow car an overtake is about.  They are placed exactly, so the
  scenario is the same test on every run.
* **random** vehicles are the background, drawn from the seed.  They make the
  situation happen inside traffic rather than in a vacuum, and the seed is
  reported so a surprising run can be reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..world.grid import GridLayout, grid_network, node_id
from ..world.grid_traffic import ForcedVehicle, TrafficConfig


@dataclass
class GridSpec:
    rows: int = 3
    cols: int = 3
    spacing: float = 150.0
    n_lanes: int = 1
    lane_width: float = 3.5
    arm_length: float = 90.0
    green: float = 20.0
    signal_stagger: float = 13.0


@dataclass
class EgoSpec:
    """Where the ego enters the grid and what it does at each intersection."""

    node: str
    direction: str
    plan: tuple[str, ...]
    lane: int = 0
    start_speed: float = 10.0
    #: metres of the route left un-driven at the end, so the run finishes on the
    #: exit arm rather than at the very end of a finite path
    goal_margin: float = 25.0


@dataclass
class Preset:
    key: str
    name: str
    description: str
    grid: GridSpec
    ego: EgoSpec
    duration: float
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    forced: tuple[ForcedVehicle, ...] = ()
    #: what the preset is testing, shown next to the KPI table
    focus: str = ""

    def build_layout(self) -> GridLayout:
        g = self.grid
        return grid_network(
            rows=g.rows, cols=g.cols, spacing=g.spacing, n_lanes=g.n_lanes,
            lane_width=g.lane_width, arm_length=g.arm_length, green=g.green,
            signal_stagger=g.signal_stagger,
        )


CENTRE = node_id(1, 1)
WEST_MID = node_id(0, 1)
EAST_MID = node_id(2, 1)
SOUTH_MID = node_id(1, 0)


def _presets() -> dict[str, Preset]:
    out: dict[str, Preset] = {}

    out["grid_random"] = Preset(
        key="grid_random",
        name="Grid cruise 격자 주행",
        description=(
            "Cross the grid west to east through three signalized intersections, "
            "in n vehicles of random traffic."
        ),
        focus="General driving: signals, car following, and traffic that does not care about you.",
        grid=GridSpec(),
        ego=EgoSpec(node=WEST_MID, direction="E", plan=("straight", "straight", "straight"),
                    start_speed=11.0),
        duration=150.0,
        traffic=TrafficConfig(n_vehicles=14),
    )

    out["unprotected_left"] = Preset(
        key="unprotected_left",
        name="Unprotected left 비보호 좌회전",
        description=(
            "Turn left at the centre crossroads across an oncoming stream that "
            "shares the same green."
        ),
        focus="Yielding: the signal cannot separate a left turn from oncoming traffic.",
        grid=GridSpec(green=35.0, signal_stagger=0.0),
        ego=EgoSpec(node=WEST_MID, direction="E", plan=("straight", "left", "straight"),
                    start_speed=10.0),
        duration=120.0,
        traffic=TrafficConfig(n_vehicles=6, route_min_nodes=2, route_max_nodes=3),
        forced=(
            # Oncoming, westbound through the centre: the vehicles to yield to.
            ForcedVehicle(node=EAST_MID, direction="W", plan=("straight", "straight"),
                          s0=95.0, v=11.0, v0=11.0, id="onc1"),
            ForcedVehicle(node=EAST_MID, direction="W", plan=("straight", "straight"),
                          s0=55.0, v=11.0, v0=11.0, id="onc2"),
        ),
    )

    out["right_turn"] = Preset(
        key="right_turn",
        name="Right turn 우회전",
        description=(
            "Turn right at the centre crossroads. The connector radius is 5.75 m, "
            "so the speed profile must slow to about 5 m/s."
        ),
        focus="Speed profile: a tight radius is drivable only at the right speed.",
        grid=GridSpec(),
        ego=EgoSpec(node=WEST_MID, direction="E", plan=("straight", "right", "straight"),
                    start_speed=11.0),
        duration=120.0,
        traffic=TrafficConfig(n_vehicles=8),
    )

    out["overtake_straight"] = Preset(
        key="overtake_straight",
        name="Overtake 직선도로 추월",
        description=(
            "A two-lane arm with a slow vehicle ahead in the same lane: pass it, "
            "or follow it to the next intersection."
        ),
        focus="Manoeuvre choice: the lattice must find the go-around, not just brake.",
        grid=GridSpec(n_lanes=2, spacing=190.0, arm_length=160.0),
        ego=EgoSpec(node=WEST_MID, direction="E", plan=("straight", "straight"),
                    start_speed=15.0),
        duration=110.0,
        traffic=TrafficConfig(n_vehicles=6, v0_min=6.0, v0_max=10.0),
        forced=(
            ForcedVehicle(node=WEST_MID, direction="E", plan=("straight", "straight"),
                          s0=95.0, v=5.0, v0=5.0, id="slow1"),
        ),
    )

    out["cross_traffic"] = Preset(
        key="cross_traffic",
        name="Cross traffic 교차 통행",
        description=(
            "Straight through the centre while a vehicle enters from the south "
            "against its phase."
        ),
        focus="Conflict detection with a vehicle that is not obeying its light.",
        grid=GridSpec(green=25.0, signal_stagger=0.0),
        ego=EgoSpec(node=WEST_MID, direction="E", plan=("straight", "straight", "straight"),
                    start_speed=12.0),
        duration=110.0,
        traffic=TrafficConfig(n_vehicles=6),
        forced=(
            ForcedVehicle(node=SOUTH_MID, direction="N", plan=("straight", "straight"),
                          s0=118.0, v=11.0, v0=11.0, id="runner"),
        ),
    )

    out["free_drive"] = Preset(
        key="free_drive",
        name="Free drive 무교통 주행",
        description="The same grid crossing with no other traffic: the tracking baseline.",
        focus="Lane keeping and signal compliance, with nothing else in the way.",
        grid=GridSpec(),
        ego=EgoSpec(node=WEST_MID, direction="E", plan=("straight", "straight", "straight"),
                    start_speed=11.0),
        duration=140.0,
        traffic=TrafficConfig(n_vehicles=0),
    )

    return out


PRESETS: dict[str, Preset] = _presets()


def preset_summaries() -> list[dict]:
    return [
        {
            "key": p.key,
            "name": p.name,
            "description": p.description,
            "focus": p.focus,
            "duration": p.duration,
            "n_vehicles": p.traffic.n_vehicles,
            "rows": p.grid.rows,
            "cols": p.grid.cols,
            "n_lanes": p.grid.n_lanes,
        }
        for p in PRESETS.values()
    ]
