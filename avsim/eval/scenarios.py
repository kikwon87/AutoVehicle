"""The scenario suite.

Each scenario is a *question*, and the KPI thresholds are the answer it demands:

===========================  ==============================================
scenario                     the question it asks
===========================  ==============================================
``lane_keeping``             does the closed loop hold a lane at speed?
``curved_lane_keeping``      does it hold a lane where feedforward matters?
``static_obstacle``          does the lattice find the go-around homotopy?
``blocked_single_lane``      does it stop when there is no way around?
``lead_braking``             does it keep a gap when the leader brakes hard?
``cut_in``                   does it survive a scripted lateral intrusion?
``signal_green``             does it proceed when the light allows?
``signal_red``               does it stop before the line?
``signal_yellow_dilemma``    does it make the stop/clear decision correctly?
``unprotected_left``         the lecture's running example, with oncoming
                             traffic and a multi-modal prediction
``tight_right_turn``         does the speed profile respect a 5.75 m radius?
``low_mu_curve``             what happens when friction drops mid-corner?
``cross_traffic``            does it yield to a vehicle running the cross
                             phase?
===========================  ==============================================

Scenarios come in two flavours, and the difference is deliberate: the
signalized and turning ones use **reactive** traffic (:class:`SimulatedTrafficSource`),
which tests whether the plan survives traffic that responds to it, while the
cut-in and lead-braking ones use **scripted** traffic
(:class:`ScriptedSyncSource`), which reproduces one adversarial event
identically on every run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..autonomy.stack import AutonomyConfig, AutonomyStack
from ..models.params import REFERENCE_VEHICLE, VehicleParams
from ..world.actors import (
    ActorState,
    IDMParams,
    ScriptedSyncSource,
    SimulatedTrafficSource,
    SyncSource,
    TrafficActor,
)
from ..world.network import SIGNAL_GROUP, four_way_intersection, straight_road
from ..world.path import ReferencePath
from ..world.traffic_light import TrafficLightController
from ..world.world import FrictionPatch, World
from .kpi import KPIThresholds


@dataclass
class ScenarioSetup:
    """Everything the runner needs to execute one scenario."""

    name: str
    description: str
    world: World
    stack: AutonomyStack
    route: ReferencePath
    duration: float
    goal_s: float | None = None
    thresholds: KPIThresholds = field(default_factory=KPIThresholds)
    stop_line_s: float | None = None
    signal_group: str | None = None
    notes: str = ""


Builder = Callable[[int], ScenarioSetup]


def _stack(
    params: VehicleParams,
    net,
    route,
    speed_limit: float,
    seed: int,
    **kwargs,
) -> AutonomyStack:
    cfg = AutonomyConfig(speed_limit=speed_limit, seed=seed)
    return AutonomyStack(params, net, route, cfg, **kwargs)


# --- open-road scenarios ------------------------------------------------------

def lane_keeping(seed: int = 0) -> ScenarioSetup:
    p = REFERENCE_VEHICLE
    net = straight_road(length=500, n_lanes=2, speed_limit=16.0)
    route = net.lanes["lane_0"].centerline
    world = World(net, p, dt=0.02)
    world.reset()
    world.place_ego(route, s=10.0, v=16.0, e_y=1.0)
    return ScenarioSetup(
        name="lane_keeping",
        description="Hold a straight lane at 16 m/s from a 1.0 m initial offset.",
        world=world,
        stack=_stack(p, net, route, 16.0, seed),
        route=route,
        duration=25.0,
        goal_s=420.0,
        thresholds=KPIThresholds(max_cross_track_rms=0.25, max_cross_track_peak=1.05),
    )


def curved_lane_keeping(seed: int = 0) -> ScenarioSetup:
    """A 300 m radius at 16 m/s: ``a_y = 0.85 m/s^2``, and the force-dependent
    feedforward term is a third of the geometric one."""
    p = REFERENCE_VEHICLE
    net = straight_road(length=500, n_lanes=2, speed_limit=16.0, curvature=1.0 / 300.0)
    route = net.lanes["lane_0"].centerline
    world = World(net, p, dt=0.02)
    world.reset()
    world.place_ego(route, s=10.0, v=16.0)
    return ScenarioSetup(
        name="curved_lane_keeping",
        description="Hold a 300 m radius curve at 16 m/s.",
        world=world,
        stack=_stack(p, net, route, 16.0, seed),
        route=route,
        duration=25.0,
        goal_s=380.0,
        thresholds=KPIThresholds(max_cross_track_rms=0.30, max_cross_track_peak=1.0),
    )


def _parked_vehicle(route: ReferencePath, s: float, e_y: float = 0.0, name: str = "blocker") -> SyncSource:
    p = route.to_cartesian(s, e_y)
    th = route.heading(s)
    state = ActorState(id=name, x=float(p[0]), y=float(p[1]), psi=float(th), v=0.0)
    return ScriptedSyncSource({name: lambda t, st=state: st})


def static_obstacle(seed: int = 0) -> ScenarioSetup:
    p = REFERENCE_VEHICLE
    net = straight_road(length=500, n_lanes=2, speed_limit=14.0)
    route = net.lanes["lane_0"].centerline
    world = World(net, p, sync_source=_parked_vehicle(route, 150.0), dt=0.02)
    world.reset()
    world.place_ego(route, s=20.0, v=14.0)
    stack = _stack(p, net, route, 14.0, seed)
    # The road spans both lanes, and it is *not* symmetric about lane 0: the
    # second lane lies at e_y = -3.5, the far kerb at -5.4, the near kerb at
    # +1.6.  A symmetric corridor would either forbid the legal pass or allow
    # driving off the near edge.
    stack.cfg.corridor_bounds = (-5.4, 1.6)
    stack.mpc.cfg.v_max = 16.0
    return ScenarioSetup(
        name="static_obstacle",
        description="A stalled vehicle blocks the lane; a second lane is open.",
        world=world,
        stack=stack,
        route=route,
        duration=30.0,
        goal_s=400.0,
        thresholds=KPIThresholds(min_clearance=0.5, max_cross_track_rms=2.5,
                                 max_cross_track_peak=4.8, max_corridor_exit_time=30.0),
        notes="corridor widened to two lanes; cross-track is measured against lane 0",
    )


def blocked_single_lane(seed: int = 0) -> ScenarioSetup:
    p = REFERENCE_VEHICLE
    net = straight_road(length=400, n_lanes=1, speed_limit=14.0)
    route = net.lanes["lane_0"].centerline
    world = World(net, p, sync_source=_parked_vehicle(route, 150.0), dt=0.02)
    world.reset()
    world.place_ego(route, s=20.0, v=14.0)
    return ScenarioSetup(
        name="blocked_single_lane",
        description="The same obstacle with no room to pass: the answer is to stop.",
        world=world,
        stack=_stack(p, net, route, 14.0, seed),
        route=route,
        duration=25.0,
        goal_s=None,
        # An emergency stop spends the whole friction budget; that is the
        # correct behaviour here, not a failure, so the usage bound is relaxed
        # to 1.0 rather than the 0.95 of ordinary driving.
        thresholds=KPIThresholds(min_clearance=0.5, min_mean_speed=0.0,
                                 max_fallback_fraction=1.0, max_solve_time_p95=0.06,
                                 max_friction_usage=1.0, max_lon_accel=8.0,
                                 max_jerk_rms=9.0, min_solver_success=0.25),
        notes="no goal: success is stopping short, not making progress",
    )


def lead_braking(seed: int = 0) -> ScenarioSetup:
    """Leader cruises at 14 m/s then brakes at 4 m/s^2 from ``t = 6 s``."""
    p = REFERENCE_VEHICLE
    net = straight_road(length=600, n_lanes=1, speed_limit=16.0)
    route = net.lanes["lane_0"].centerline

    def leader(t: float) -> ActorState:
        v0, s0, t_brake, a = 14.0, 70.0, 6.0, 4.0
        if t < t_brake:
            s, v = s0 + v0 * t, v0
        else:
            dt = min(t - t_brake, v0 / a)
            s = s0 + v0 * t_brake + v0 * dt - 0.5 * a * dt**2
            v = max(v0 - a * dt, 0.0)
        pos = route.to_cartesian(min(s, route.length), 0.0)
        return ActorState(id="lead", x=float(pos[0]), y=float(pos[1]),
                          psi=float(route.heading(min(s, route.length))), v=v)

    world = World(net, p, sync_source=ScriptedSyncSource({"lead": leader}), dt=0.02)
    world.reset()
    world.place_ego(route, s=20.0, v=16.0)
    return ScenarioSetup(
        name="lead_braking",
        description="A leader 50 m ahead brakes hard at 4 m/s^2.",
        world=world,
        stack=_stack(p, net, route, 16.0, seed),
        route=route,
        duration=25.0,
        goal_s=None,
        thresholds=KPIThresholds(min_clearance=0.5, min_ttc=0.8, max_lon_accel=8.0,
                                 max_jerk_rms=9.0, max_friction_usage=1.0),
    )


def cut_in(seed: int = 0) -> ScenarioSetup:
    """A vehicle in the adjacent lane merges in over 2 s starting at ``t = 5 s``."""
    p = REFERENCE_VEHICLE
    net = straight_road(length=600, n_lanes=2, speed_limit=16.0)
    route = net.lanes["lane_0"].centerline

    def merger(t: float) -> ActorState:
        v = 12.0
        s = 55.0 + v * t
        t0, T = 5.0, 2.0
        tau = float(np.clip((t - t0) / T, 0.0, 1.0))
        e_y = -3.5 * (1.0 - (3 * tau**2 - 2 * tau**3))  # from lane 1 to lane 0
        sc = min(s, route.length)
        pos = route.to_cartesian(sc, e_y)
        de = -3.5 * (-(6 * tau - 6 * tau**2) / T) if 0 < tau < 1 else 0.0
        return ActorState(id="cutter", x=float(pos[0]), y=float(pos[1]),
                          psi=float(route.heading(sc) + np.arctan2(de, v)), v=v)

    world = World(net, p, sync_source=ScriptedSyncSource({"cutter": merger}), dt=0.02)
    world.reset()
    world.place_ego(route, s=20.0, v=16.0)
    return ScenarioSetup(
        name="cut_in",
        description="An adjacent-lane vehicle cuts in 35 m ahead at t = 5 s.",
        world=world,
        stack=_stack(p, net, route, 16.0, seed),
        route=route,
        duration=25.0,
        goal_s=None,
        thresholds=KPIThresholds(min_clearance=0.3, min_ttc=0.6, max_lon_accel=8.0,
                                 max_jerk_rms=9.0, max_friction_usage=1.0),
    )


def low_mu_curve(seed: int = 0) -> ScenarioSetup:
    """Friction drops to 0.35 inside a corner the plan already committed to."""
    p = REFERENCE_VEHICLE
    net = straight_road(length=500, n_lanes=1, speed_limit=18.0, curvature=1.0 / 120.0)
    route = net.lanes["lane_0"].centerline
    centre = route.position(180.0)
    patch = FrictionPatch(
        x_min=float(centre[0] - 40), x_max=float(centre[0] + 40),
        y_min=float(centre[1] - 40), y_max=float(centre[1] + 40),
        mu=0.35, label="wet patch",
    )
    world = World(net, p, friction_patches=[patch], dt=0.02)
    world.reset()
    world.place_ego(route, s=20.0, v=18.0)
    return ScenarioSetup(
        name="low_mu_curve",
        description="A 120 m radius at 18 m/s with mu dropping to 0.35 mid-corner.",
        world=world,
        stack=_stack(p, net, route, 18.0, seed),
        route=route,
        duration=25.0,
        goal_s=None,
        thresholds=KPIThresholds(max_cross_track_rms=1.2, max_cross_track_peak=3.0,
                                 max_corridor_exit_time=8.0, max_friction_usage=1.05),
        notes="the planner is not told about the patch: this measures the failure, "
              "not the recovery",
    )


# --- intersection scenarios ---------------------------------------------------

def _intersection_stack(net, route, approach, lights, seed, speed_limit=13.9, crossing=False):
    stop_s = net.lanes[f"{approach}_in_0"].length
    stack = _stack(
        REFERENCE_VEHICLE, net, route, speed_limit, seed,
        lights=lights, signal_group=SIGNAL_GROUP[approach], stop_line_s=stop_s,
        crossing_conflict=crossing,
    )
    return stack, stop_s


def _signal_scenario(name: str, offset: float, description: str, seed: int, duration: float = 40.0):
    p = REFERENCE_VEHICLE
    net = four_way_intersection()
    lights = TrafficLightController(offset=offset)
    route = net.route_path(net.route("E", "straight"))
    stack, stop_s = _intersection_stack(net, route, "E", lights, seed)
    world = World(net, p, None, lights, dt=0.02)
    world.reset()
    world.place_ego(route, s=stop_s - 80.0, v=13.0)
    return ScenarioSetup(
        name=name, description=description, world=world, stack=stack, route=route,
        duration=duration, goal_s=None, stop_line_s=stop_s, signal_group="EW",
        thresholds=KPIThresholds(max_cross_track_rms=0.25, max_cross_track_peak=0.8),
    )


def signal_green(seed: int = 0) -> ScenarioSetup:
    return _signal_scenario("signal_green", -15.0,
                            "Approach a green light with time to clear the box.", seed, 30.0)


def signal_red(seed: int = 0) -> ScenarioSetup:
    s = _signal_scenario("signal_red", -45.0,
                         "Approach a red light and stop before the line.", seed, 40.0)
    s.thresholds.max_red_light_violations = 0.0
    return s


def signal_yellow_dilemma(seed: int = 0) -> ScenarioSetup:
    return _signal_scenario("signal_yellow_dilemma", -8.0,
                            "Arrive as the green ends: stop or clear, but decide.", seed, 45.0)


def tight_right_turn(seed: int = 0) -> ScenarioSetup:
    p = REFERENCE_VEHICLE
    net = four_way_intersection()
    lights = TrafficLightController(offset=-15.0)
    route = net.route_path(net.route("E", "right"))
    stack, stop_s = _intersection_stack(net, route, "E", lights, seed)
    world = World(net, p, None, lights, dt=0.02)
    world.reset()
    world.place_ego(route, s=stop_s - 60.0, v=12.0)
    return ScenarioSetup(
        name="tight_right_turn",
        description="A 5.75 m radius right turn: the speed profile must slow to ~5 m/s.",
        world=world, stack=stack, route=route, duration=35.0,
        goal_s=stop_s + 60.0, stop_line_s=stop_s, signal_group="EW",
        thresholds=KPIThresholds(max_cross_track_rms=0.45, max_cross_track_peak=1.3,
                                 max_lat_accel=5.5),
    )


def unprotected_left(seed: int = 0) -> ScenarioSetup:
    """The lecture's running example: brake, low-speed turn, accelerate --
    across oncoming traffic that has the same green."""
    p = REFERENCE_VEHICLE
    net = four_way_intersection()
    lights = TrafficLightController(green=30.0, offset=-25.0)
    route = net.route_path(net.route("E", "left"))
    oncoming_route = net.route_path(net.route("W", "straight"))
    stop_s_w = net.lanes["W_in_0"].length
    actors = [
        TrafficActor(id=f"onc{i}", path=oncoming_route, s=stop_s_w - 150.0 - 40.0 * i,
                     v=12.0, idm=IDMParams(v0=12.0), signal_group="EW",
                     stop_line_s=stop_s_w, route_id="W_straight")
        for i in range(2)
    ]
    src = SimulatedTrafficSource(actors, lights)
    stack, stop_s = _intersection_stack(net, route, "E", lights, seed, crossing=True)
    world = World(net, p, src, lights, dt=0.02)
    world.reset()
    world.place_ego(route, s=stop_s - 70.0, v=12.0)
    return ScenarioSetup(
        name="unprotected_left",
        description="Unprotected left across two oncoming vehicles on the same green.",
        world=world, stack=stack, route=route, duration=45.0,
        goal_s=stop_s + 40.0, stop_line_s=stop_s, signal_group="EW",
        thresholds=KPIThresholds(min_clearance=0.6, min_ttc=1.0,
                                 max_cross_track_rms=0.5, max_cross_track_peak=1.4,
                                 max_lat_accel=5.0),
    )


def cross_traffic(seed: int = 0) -> ScenarioSetup:
    """A vehicle enters from the north against its red -- a red-light runner."""
    p = REFERENCE_VEHICLE
    net = four_way_intersection()
    lights = TrafficLightController(offset=-15.0)
    route = net.route_path(net.route("E", "straight"))
    cross_route = net.route_path(net.route("N", "straight"))

    def runner(t: float) -> ActorState:
        v = 11.0
        s = float(np.clip(net.lanes["N_in_0"].length - 60.0 + v * t, 0.0, cross_route.length))
        pos = cross_route.position(s)
        return ActorState(id="runner", x=float(pos[0]), y=float(pos[1]),
                          psi=float(cross_route.heading(s)), v=v)

    stack, stop_s = _intersection_stack(net, route, "E", lights, seed, crossing=True)
    world = World(net, p, ScriptedSyncSource({"runner": runner}), lights, dt=0.02)
    world.reset()
    world.place_ego(route, s=stop_s - 75.0, v=13.0)
    return ScenarioSetup(
        name="cross_traffic",
        description="A red-light runner crosses from the north as the ego enters on green.",
        world=world, stack=stack, route=route, duration=30.0,
        goal_s=None, stop_line_s=stop_s, signal_group="EW",
        thresholds=KPIThresholds(min_clearance=0.4, min_ttc=0.7),
    )


SCENARIOS: dict[str, Builder] = {
    "lane_keeping": lane_keeping,
    "curved_lane_keeping": curved_lane_keeping,
    "static_obstacle": static_obstacle,
    "blocked_single_lane": blocked_single_lane,
    "lead_braking": lead_braking,
    "cut_in": cut_in,
    "low_mu_curve": low_mu_curve,
    "signal_green": signal_green,
    "signal_red": signal_red,
    "signal_yellow_dilemma": signal_yellow_dilemma,
    "tight_right_turn": tight_right_turn,
    "unprotected_left": unprotected_left,
    "cross_traffic": cross_traffic,
}
