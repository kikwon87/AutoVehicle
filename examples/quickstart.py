"""A minimal end-to-end example: build a world, drive it, read the KPIs.

Run with ``python examples/quickstart.py``.
"""

from __future__ import annotations

import numpy as np

from avsim.autonomy.stack import AutonomyConfig, AutonomyStack
from avsim.eval.kpi import compute_kpis
from avsim.models.params import REFERENCE_VEHICLE
from avsim.world.network import four_way_intersection
from avsim.world.traffic_light import TrafficLightController
from avsim.world.world import World


def main() -> None:
    params = REFERENCE_VEHICLE

    # 1. A four-way signalized intersection, and a route straight through it.
    net = four_way_intersection()
    lights = TrafficLightController(offset=-45.0)      # the ego arrives on red
    route = net.route_path(net.route("E", "straight"))
    stop_line_s = net.lanes["E_in_0"].length

    # 2. The world owns ground truth: the plant, the signals, the other traffic.
    world = World(net, params, sync_source=None, lights=lights, dt=0.02)
    world.reset()
    world.place_ego(route, s=stop_line_s - 80.0, v=13.0)

    # 3. The stack sees the world only through its sensor.
    stack = AutonomyStack(
        params, net, route,
        AutonomyConfig(speed_limit=13.9),
        lights=lights, signal_group="EW", stop_line_s=stop_line_s,
    )

    # 4. Two rates: the plant integrates at 20 ms, the controller runs at 100 ms
    #    and its command is held in between.
    steps_per_control = int(round(stack.cfg.control_dt / world.dt))
    command = np.zeros(2)
    for k in range(int(40.0 / world.dt)):
        if k % steps_per_control == 0:
            command = stack.step(
                world.t,
                world.ego_actor(),
                world.ego_rear_axle(),
                float(world.ego[10]),
                world.current_actors(),
            )
        world.step(command)

    # 5. Reduce the run to a report.
    report = compute_kpis(
        "quickstart", world.history, stack.telemetry, route, params,
        dt=world.dt, stop_line_s=stop_line_s, signal_group="EW", lights=lights,
    )
    print(report.to_table())

    print("\nbehaviour timeline:")
    previous = None
    for rec in stack.telemetry:
        if rec.behavior != previous:
            print(f"  t = {rec.t:5.1f} s   {rec.behavior:18s} {rec.reason}")
            previous = rec.behavior


if __name__ == "__main__":
    main()
