"""Paths, the intersection layout, signals, actors and the world container."""

import numpy as np
import pytest

from avsim.core.geometry import (
    multi_circle_cover,
    obb_penetration,
    polygon_distance,
    rect_corners,
    time_to_collision,
)
from avsim.models.params import REFERENCE_VEHICLE as P
from avsim.world.actors import (
    ActorState,
    IDMParams,
    ScriptedSyncSource,
    SimulatedTrafficSource,
    TrafficActor,
    idm_acceleration,
)
from avsim.world.network import four_way_intersection, straight_road
from avsim.world.path import ConcatPath, PrimitivePath, SplinePath
from avsim.world.traffic_light import DilemmaZone, SignalState, TrafficLightController
from avsim.world.world import FrictionPatch, World


# --- paths ---------------------------------------------------------------------

def test_primitive_path_geometry_is_exact():
    R = 20.0
    p = PrimitivePath.chain(0, 0, 0, [("straight", 30, 0), ("arc", R * np.pi / 2, 1 / R),
                                      ("straight", 30, 0)])
    assert p.length == pytest.approx(60 + R * np.pi / 2)
    assert p.position(p.length) == pytest.approx(np.array([50.0, 50.0]))
    assert p.heading(p.length) == pytest.approx(np.pi / 2)
    assert p.curvature(40.0) == pytest.approx(1 / R)


def test_projection_round_trip():
    p = PrimitivePath.chain(0, 0, 0, [("straight", 30, 0), ("arc", 20 * np.pi / 2, 1 / 20)])
    rng = np.random.default_rng(0)
    for s in np.linspace(0.5, p.length - 0.5, 40):
        e_y = rng.uniform(-3, 3)
        q = p.to_cartesian(s, e_y)
        s_hat = p.project(q[0], q[1], s_guess=s)
        assert s_hat == pytest.approx(s, abs=1e-6)
        assert p.lateral_offset(q[0], q[1], s_hat) == pytest.approx(e_y, abs=1e-6)


def test_spline_path_recovers_a_circle():
    th = np.linspace(0, 2 * np.pi, 60)
    sp = SplinePath(np.column_stack([25 * np.cos(th), 25 * np.sin(th)]), ds=0.2)
    assert sp.length == pytest.approx(2 * np.pi * 25, rel=1e-3)
    assert sp.kappa[5:-5].mean() == pytest.approx(1 / 25, rel=1e-2)


def test_path_table_is_accurate_and_cached():
    p = PrimitivePath.chain(0, 0, 0, [("arc", 30.0, 1 / 40.0)])
    ss = np.linspace(0, p.length, 200)
    x, y, th, ka = p.frames(ss)
    exact = np.array([p.position(s) for s in ss])
    assert np.abs(np.column_stack([x, y]) - exact).max() < 5e-3
    assert p.table() is p.table()


def test_concat_rejects_a_disconnected_route():
    a = PrimitivePath.chain(0, 0, 0, [("straight", 10, 0)])
    bad = PrimitivePath.chain(11, 0, 0, [("straight", 5, 0)])
    with pytest.raises(ValueError, match="do not connect"):
        ConcatPath([a, bad])


# --- network --------------------------------------------------------------------

def test_intersection_turn_radii_are_solved_not_assumed():
    net = four_way_intersection()
    assert net.box_half == pytest.approx(11.0)
    left = net.lanes["E_left"]
    right = net.lanes["E_right"]
    assert 1 / abs(left.centerline.curvature(left.length / 2)) == pytest.approx(12.75, abs=1e-6)
    assert 1 / abs(right.centerline.curvature(right.length / 2)) == pytest.approx(5.75, abs=1e-6)


def test_every_connector_is_drivable_by_the_reference_car():
    net = four_way_intersection()
    min_radius = P.L / np.tan(P.actuator.delta_max)
    for lane in net.lanes.values():
        if lane.kind in ("connector_left", "connector_right"):
            R = 1 / abs(lane.centerline.curvature(lane.length / 2))
            assert R > min_radius


def test_a_square_box_without_flare_is_rejected():
    """The right turn would need 1.75 m, below the car's 3.86 m minimum."""
    with pytest.raises(ValueError, match="increase `flare`"):
        four_way_intersection(flare=0.0)


def test_routes_are_continuous():
    net = four_way_intersection()
    for approach in ("E", "N", "W", "S"):
        for man in ("straight", "left", "right"):
            path = net.route_path(net.route(approach, man))  # raises if disconnected
            assert path.length > 200.0


def test_curved_road_lanes_stay_parallel():
    net = straight_road(curvature=1 / 300.0)
    a = net.lanes["lane_0"].centerline
    b = net.lanes["lane_1"].centerline
    d = [np.linalg.norm(a.position(u * a.length) - b.position(u * b.length))
         for u in np.linspace(0, 1, 20)]
    assert max(d) - min(d) < 1e-6
    assert d[0] == pytest.approx(3.5)


# --- signals ---------------------------------------------------------------------

def test_signal_cycle():
    c = TrafficLightController()
    assert c.cycle == pytest.approx(50.0)
    assert c.state("NS", 0.0) is SignalState.GREEN
    assert c.state("EW", 0.0) is SignalState.RED
    assert c.state("NS", 21.0) is SignalState.YELLOW
    assert c.state("NS", 24.0) is SignalState.RED
    assert c.state("EW", 24.0) is SignalState.RED     # the all-red overlap
    assert c.state("EW", 26.0) is SignalState.GREEN


def test_green_now_is_not_green_later():
    c = TrafficLightController()
    assert c.state("NS", 18.0) is SignalState.GREEN
    assert not c.will_be_green_at("NS", 18.0, 5.0)
    assert c.will_be_green_at("NS", 1.0, 5.0)


def test_dilemma_zone_has_three_outcomes():
    d = DilemmaZone()
    assert d.classify(60, 14, 3.0, 22.0) == "stop"
    assert d.classify(20, 14, 3.0, 22.0) == "clear"
    assert d.classify(30, 14, 3.0, 22.0) == "dilemma"


# --- geometry ---------------------------------------------------------------------

def test_oriented_box_collision_and_clearance():
    a = rect_corners(0, 0, 0, 4.6, 1.85)
    assert polygon_distance(a, rect_corners(5, 0, 0, 4.6, 1.85)) == pytest.approx(0.4)
    assert obb_penetration(a, rect_corners(4, 0, 0, 4.6, 1.85)) == pytest.approx(0.6)


def test_multi_circle_cover_is_much_tighter_than_one_circle():
    _, r3 = multi_circle_cover(4.6, 1.85, 3)
    assert r3 == pytest.approx(1.20, abs=0.01)
    assert r3 < 0.5 * np.hypot(4.6, 1.85) / 2


def test_ttc():
    assert time_to_collision(np.zeros(2), np.array([10.0, 0]), np.array([100.0, 0]),
                             np.array([-10.0, 0]), 4.0) == pytest.approx(4.8)
    assert not np.isfinite(time_to_collision(np.zeros(2), np.array([10.0, 0]),
                                             np.array([100.0, 0]), np.array([10.0, 0]), 4.0))


# --- actors ------------------------------------------------------------------------

def test_idm_is_finite_at_zero_gap():
    a = idm_acceleration(10.0, 0.0, 10.0, IDMParams())
    assert np.isfinite(a) and a < 0


def test_queue_forms_behind_a_red_light():
    net = four_way_intersection()
    lights = TrafficLightController()
    route = net.route_path(net.route("W", "straight"))
    stop_s = net.lanes["W_in_0"].length
    actors = [
        TrafficActor(id=f"v{i}", path=route, s=stop_s - 20 - 18 * i, v=12.0,
                     signal_group="EW", stop_line_s=stop_s, route_id="W")
        for i in range(3)
    ]
    src = SimulatedTrafficSource(actors, lights)
    t = 0.0
    for _ in range(int(20 / 0.05)):
        src.step(t, 0.05)
        t += 0.05
    s = sorted((a.s for a in src._actors), reverse=True)
    assert all(v < 0.2 for v in (a.v for a in src._actors))     # stopped
    assert s[0] < stop_s                                        # behind the line
    gaps = -np.diff(s)
    assert all(6.0 < g < 9.0 for g in gaps)                     # s0 = 2.5 m bumper gap


def test_sync_sources_reject_duplicate_ids():
    from avsim.world.actors import CompositeSyncSource

    mk = lambda: ScriptedSyncSource({"a": lambda t: ActorState("a", 0, 0, 0, 0)})  # noqa: E731
    with pytest.raises(ValueError, match="duplicate actor id"):
        CompositeSyncSource([mk(), mk()]).step(0.0, 0.1)


# --- world -------------------------------------------------------------------------

def test_friction_patch_is_spatial():
    net = straight_road()
    w = World(net, P, friction_patches=[FrictionPatch(50, 100, -10, 10, 0.3)])
    assert w.mu_at(10, 0) == pytest.approx(P.mu)
    assert w.mu_at(75, 0) == pytest.approx(0.3)


def test_place_ego_applies_the_lever_arm():
    net = straight_road()
    route = net.lanes["lane_0"].centerline
    w = World(net, P)
    w.reset()
    w.place_ego(route, s=50.0, v=10.0)
    xr = w.ego_rear_axle()
    assert route.project(xr[0], xr[1]) == pytest.approx(50.0, abs=1e-6)
    assert route.lateral_offset(xr[0], xr[1], 50.0) == pytest.approx(0.0, abs=1e-9)
