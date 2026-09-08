"""Execute scenarios and reduce them to KPI reports.

The runner owns the **two-rate loop**: the world integrates at ``world.dt``
while the autonomy stack updates at ``stack.cfg.control_dt`` and its command is
held in between.  That hold is the zero-order hold the modelling chapter is
about, and running the controller at the plant's rate would quietly hide every
consequence of it.

It also accumulates the perception ground truth needed for a recall KPI:
at each perception tick it asks the sensor which objects were *actually
visible* and compares that with which ones the tracker was holding.  Comparing
against all ground truth instead would score an occluded vehicle as a tracker
failure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..autonomy.stack import Telemetry
from ..world.world import World, WorldSnapshot
from .kpi import KPIReport, compute_kpis
from .scenarios import SCENARIOS, ScenarioSetup


@dataclass
class RunResult:
    setup: ScenarioSetup
    report: KPIReport
    history: list[WorldSnapshot]
    telemetry: list[Telemetry]
    wall_time: float
    terminated_early: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.report.passed

    def __str__(self) -> str:
        return self.report.to_table()


def run_setup(setup: ScenarioSetup, verbose: bool = False) -> RunResult:
    """Run one already-built scenario to completion."""
    world: World = setup.world
    stack = setup.stack
    steps_per_control = max(int(round(stack.cfg.control_dt / world.dt)), 1)

    cmd = np.zeros(2)
    detected = visible = 0
    n_steps = int(round(setup.duration / world.dt))
    t0 = time.perf_counter()
    terminated = ""

    for k in range(n_steps):
        if k % steps_per_control == 0:
            ego = world.ego_actor()
            actors = world.current_actors()
            cmd = stack.step(world.t, ego, world.ego_rear_axle(), float(world.ego[10]), actors)

            vis = stack.sensor.visible_ids(ego, actors)
            visible += len(vis)
            held = {t.truth_id for t in stack.tracks if t.truth_id is not None}
            detected += len(vis & held)

        world.step(cmd)

        # A run ends at its goal, or when the route is exhausted.
        #
        # Stopping at the goal matters for more than tidiness: past it the
        # planner brakes for the end of a finite route, and that deceleration
        # lands in the comfort KPIs of a scenario that was never about braking.
        s_now = stack._s
        if setup.goal_s is not None and s_now >= setup.goal_s:
            terminated = "goal reached"
            break
        if s_now >= setup.route.length - 2.0:
            terminated = "route complete"
            break
        if world.collisions():
            terminated = "collision"
            if verbose:
                print(f"  collision at t = {world.t:.2f}s: {world.collisions()}")
            break

    wall = time.perf_counter() - t0
    report = compute_kpis(
        scenario=setup.name,
        history=world.history,
        telemetry=stack.telemetry,
        route=setup.route,
        params=world.params,
        thresholds=setup.thresholds,
        dt=world.dt,
        wall_time=wall,
        goal_s=setup.goal_s,
        stop_line_s=setup.stop_line_s,
        signal_group=setup.signal_group,
        lights=world.lights,
        corridor_half_width=stack.cfg.corridor_half_width,
        visible_counts=(detected, visible),
    )
    report.extra["description"] = setup.description
    report.extra["notes"] = setup.notes
    report.extra["terminated"] = terminated
    return RunResult(
        setup=setup,
        report=report,
        history=list(world.history),
        telemetry=list(stack.telemetry),
        wall_time=wall,
        terminated_early=terminated,
    )


def run_scenario(name: str, seed: int = 0, verbose: bool = False) -> RunResult:
    """Build and run a scenario by name."""
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; available: {sorted(SCENARIOS)}")
    return run_setup(SCENARIOS[name](seed), verbose=verbose)


def run_all(
    names: Sequence[str] | None = None, seed: int = 0, verbose: bool = True
) -> list[RunResult]:
    """Run the whole suite (or a subset) and return the results in order."""
    out = []
    for name in names or list(SCENARIOS):
        if verbose:
            print(f"running {name} ...", flush=True)
        res = run_scenario(name, seed=seed, verbose=verbose)
        if verbose:
            status = "PASS" if res.passed else "FAIL"
            print(f"  {status}  ({res.wall_time:.1f}s wall)")
            for k in res.report.failures:
                print(f"        {k.name} = {k.value:.4g} {k.unit} "
                      f"(needs {k.direction} {k.threshold:g})")
        out.append(res)
    return out
