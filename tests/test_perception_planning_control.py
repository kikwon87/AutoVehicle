"""Sensor, tracker, prediction, velocity profile, lattice, iLQR and MPC."""

import numpy as np
import pytest

from avsim.control.ilqr import ILQR, ILQROptions
from avsim.control.mpc import MPCConfig, StopLineConstraint, VehicleMPC
from avsim.models.params import REFERENCE_VEHICLE as P
from avsim.perception.sensor import VisionSensor
from avsim.perception.tracker import MultiObjectTracker
from avsim.planning.prediction import ConstantVelocityPredictor, LaneFollowingPredictor
from avsim.planning.trajectory import FrenetLatticePlanner
from avsim.planning.velocity_profile import SpeedConstraint, build_velocity_profile
from avsim.world.actors import ActorState
from avsim.world.network import four_way_intersection, straight_road


# --- perception -----------------------------------------------------------------

def test_occlusion_and_field_of_view():
    s = VisionSensor()
    ego = ActorState("ego", 0, 0, 0, 0)
    near, far = ActorState("near", 20, 0, 0, 0), ActorState("far", 30, 0, 0, 0)
    assert s.visible_ids(ego, [near, far]) == {"near"}
    s.occlusion = False
    assert s.visible_ids(ego, [near, far]) == {"near", "far"}
    s.occlusion = True
    assert s.visible_ids(ego, [ActorState("behind", -20, 0, 0, 0)]) == set()
    assert s.visible_ids(ego, [ActorState("distant", 200, 0, 0, 0)]) == set()


def test_depth_error_grows_faster_than_bearing_error():
    s = VisionSensor()
    rng = np.random.default_rng(0)
    ego = ActorState("ego", 0, 0, 0, 0)

    def spread(distance):
        errs = []
        for _ in range(400):
            d = s._measure(ego, [ActorState("a", distance, 0, 0, 0)], rng)
            if d:
                errs.append([d[0].x - distance, d[0].y])
        return np.std(errs, axis=0)

    near, far = spread(10.0), spread(70.0)
    assert far[0] / near[0] > far[1] / near[1]     # depth degrades faster
    assert far[0] > far[1]                          # and is worse in absolute terms


def test_tracker_converges_and_keeps_identity():
    dt = 0.1
    sensor, trk = VisionSensor(), MultiObjectTracker(dt)
    rng = np.random.default_rng(0)
    truth_v = 11.0
    ids, errs = set(), []
    for k in range(120):
        t = k * dt
        ego = ActorState("ego", 10.0 * t, 0, 0, 10.0)
        obj = ActorState("a", 30.0 + truth_v * t, 0.0, 0.0, truth_v)
        for tr in trk.update(sensor.observe(ego, [obj], rng)):
            ids.add(tr.id)
            if k > 40:
                errs.append((np.linalg.norm(tr.position - obj.position), abs(tr.speed - truth_v)))
    e = np.array(errs)
    assert np.sqrt((e[:, 0] ** 2).mean()) < 1.5
    assert np.sqrt((e[:, 1] ** 2).mean()) < 1.5
    assert len(ids) <= 3


def test_tracker_dt_must_match_the_update_rate():
    from avsim.autonomy.stack import AutonomyConfig, AutonomyStack

    net = straight_road()
    route = net.lanes["lane_0"].centerline
    with pytest.raises(ValueError, match="does not match the control period"):
        AutonomyStack(P, net, route, AutonomyConfig(control_dt=0.1),
                      tracker=MultiObjectTracker(0.05))


# --- prediction -------------------------------------------------------------------

def test_prediction_uncertainty_is_anisotropic():
    from avsim.perception.tracker import Track

    tr = Track(id=1, x=np.array([0.0, 0.0, 12.0, 0.0]), P=np.eye(4) * 0.16)
    times = np.linspace(0, 4, 9)
    pred = ConstantVelocityPredictor()([tr], times)[0]
    # Longitudinal uncertainty integrates an unknown acceleration twice; lateral
    # only accumulates lane drift, so the ratio grows with the horizon.
    ratio = pred.sigma_lon / pred.sigma_lat
    # Over a short horizon lane drift dominates; the quadratic longitudinal term
    # overtakes it and keeps growing.
    assert ratio[-1] > 2.5
    assert np.all(np.diff(ratio[1:]) > 0)

    still = Track(id=2, x=np.array([0.0, 0.0, 0.0, 0.0]), P=np.eye(4) * 0.16)
    parked = ConstantVelocityPredictor()([still], times)[0]
    # A stationary object does not drift sideways at all.
    assert np.allclose(parked.sigma_lat, parked.sigma_lat[0])


def test_lane_predictor_covers_the_turn_constant_velocity_misses():
    from avsim.perception.tracker import Track

    net = four_way_intersection()
    path = net.route_path(net.route("E", "left"))
    times = np.linspace(0, 4, 9)
    s0, v = 115.0, 8.0
    p, th = path.position(s0), path.heading(s0)
    tr = Track(id=1, x=np.array([p[0], p[1], v * np.cos(th), v * np.sin(th)]),
               P=np.eye(4) * 0.25, psi=th)
    truth = path.position(min(s0 + v * times[-1], path.length))

    cv = ConstantVelocityPredictor()([tr], times)[0]
    modes = LaneFollowingPredictor(net)([tr], times)
    best = min(np.linalg.norm(m.positions[-1] - truth) for m in modes)

    assert np.linalg.norm(cv.positions[-1] - truth) > 10.0
    assert best < 1.0
    assert sum(m.probability for m in modes) == pytest.approx(1.0)


# --- velocity profile ---------------------------------------------------------------

def test_profile_respects_curvature_and_stops_exactly():
    net = four_way_intersection()
    for man, expected in [("left", 7.5), ("right", 5.0)]:
        path = net.route_path(net.route("E", man))
        vp = build_velocity_profile(path, P, 13.9, curvature_lookahead=15.0, v_start=13.9)
        assert vp.v.min() == pytest.approx(expected, abs=0.2)
        a_y = vp.v**2 * np.array([abs(path.curvature(s)) for s in vp.s])
        assert a_y.max() <= vp.a_lat_limit * 1.02

    path = net.route_path(net.route("E", "straight"))
    stop_s = net.lanes["E_in_0"].length
    vp = build_velocity_profile(path, P, 13.9, [SpeedConstraint(stop_s, 0.0)],
                                v_start=13.9, s_end=stop_s + 5)
    assert vp.speed_at(stop_s) == pytest.approx(0.0, abs=1e-6)
    assert vp.speed_at(stop_s - 40) > 10.0


# --- lattice ----------------------------------------------------------------------

def test_lattice_can_pull_away_from_a_standstill():
    """A minimum-jerk profile peaks at 1.5x its average; sampling the average
    alone rejects every candidate and the vehicle never moves."""
    path = straight_road(length=400).lanes["lane_0"].centerline
    pl = FrenetLatticePlanner(P)
    for v0 in (0.0, 0.5, 2.0, 13.0):
        traj = pl.plan(path, 50.0, 0.0, 0.0, 0.0, v0, 13.9, corridor=(-1.55, 1.55))
        assert traj is not None, v0
        assert traj.a.max() <= pl.cfg.a_lon_max + 1e-6


def test_lattice_goes_around_when_there_is_room_and_stops_when_there_is_not():
    from avsim.perception.tracker import Track

    path = straight_road(length=400).lanes["lane_0"].centerline
    pl = FrenetLatticePlanner(P)
    p = path.position(90.0)
    blocker = Track(id=1, x=np.array([p[0], p[1], 0.0, 0.0]), P=np.eye(4) * 0.16)
    preds = ConstantVelocityPredictor()([blocker], np.linspace(0, 6, 25))

    assert pl.plan(path, 50, 0, 0, 0, 15.0, 20.0, preds, corridor=(-1.75, 1.75)) is None
    wide = pl.plan(path, 50, 0, 0, 0, 15.0, 20.0, preds, corridor=(-1.75, 5.25))
    assert wide is not None and abs(wide.target_offset) > 2.5


def test_lattice_rejects_candidates_that_leave_the_corridor():
    path = straight_road(length=400).lanes["lane_0"].centerline
    pl = FrenetLatticePlanner(P)
    traj = pl.plan(path, 50, 0, 0, 0, 12.0, 12.0, corridor=(-0.5, 0.5))
    assert traj is not None
    assert np.abs(traj.e_y).max() <= 0.5 + 1e-6
    assert any("corridor" in c.reject_reason for c in pl.last_candidates)


def test_emergency_stop_actually_stops():
    path = straight_road(length=400).lanes["lane_0"].centerline
    pl = FrenetLatticePlanner(P)
    traj = pl.emergency_stop(path, 50.0, 0.3, 0.0, 15.0)
    assert traj.v[-1] == pytest.approx(0.0, abs=1e-9)
    assert traj.s[-1] - 50.0 < 15.0 ** 2 / (2 * abs(pl.cfg.a_lon_min)) + 1.0


# --- iLQR ---------------------------------------------------------------------------

def _double_integrator(dt=0.1):
    A = np.array([[1, dt], [0, 1]], dtype=float)
    B = np.array([[0.5 * dt * dt], [dt]])
    Q, R = np.diag([1.0, 0.1]), np.array([[0.05]])
    return A, B, Q, R


def test_ilqr_matches_the_analytic_finite_horizon_lqr():
    A, B, Q, R = _double_integrator()
    N = 40
    solver = ILQR(lambda x, u, k: A @ x + B @ u,
                  lambda x, u, k: float(x @ Q @ x + u @ R @ u),
                  lambda x: float(10 * x @ Q @ x), 2, 1, N)
    res = solver.solve(np.array([2.0, 0.0]))

    P_k = 10 * Q
    Ps = [P_k]
    for _ in range(N):
        P_k = Q + A.T @ P_k @ A - A.T @ P_k @ B @ np.linalg.inv(R + B.T @ P_k @ B) @ B.T @ P_k @ A
        Ps.append(P_k)
    x, cost = np.array([2.0, 0.0]), 0.0
    for k in range(N):
        Pk = Ps[N - 1 - k]
        u = -np.linalg.solve(R + B.T @ Pk @ B, B.T @ Pk @ A) @ x
        cost += float(x @ Q @ x + u @ R @ u)
        x = A @ x + B @ u
    cost += float(10 * x @ Q @ x)
    assert res.cost == pytest.approx(cost, rel=1e-9)


def test_ilqr_respects_input_box_and_state_constraints():
    A, B, Q, R = _double_integrator()
    solver = ILQR(lambda x, u, k: A @ x + B @ u,
                  lambda x, u, k: float(x @ Q @ x + u @ R @ u),
                  lambda x: float(10 * x @ Q @ x), 2, 1, 40,
                  u_lo=np.array([-0.5]), u_hi=np.array([0.5]),
                  constraints=lambda x, u, k: np.array([-0.35 - x[1]]), n_constraints=1)
    res = solver.solve(np.array([2.0, 0.0]))
    assert np.abs(res.U).max() <= 0.5 + 1e-12
    assert res.X[:, 1].min() > -0.36
    assert res.constraint_violation < 1e-3


def test_time_budget_is_reported_not_hidden():
    A, B, Q, R = _double_integrator()
    solver = ILQR(lambda x, u, k: A @ x + B @ u,
                  lambda x, u, k: float(x @ Q @ x + u @ R @ u),
                  lambda x: float(10 * x @ Q @ x), 2, 1, 60,
                  options=ILQROptions(time_budget=1e-6))
    assert solver.solve(np.array([5.0, 0.0])).status == "time_budget"


# --- MPC -------------------------------------------------------------------------------

def _fd_constraint_jacobian(mpc, x, u, k, eps=1e-6):
    return np.column_stack([
        (mpc._c_padded(x + np.eye(5)[i] * eps, u, k) - mpc._c_padded(x - np.eye(5)[i] * eps, u, k))
        / (2 * eps)
        for i in range(5)
    ])


def test_mpc_constraint_jacobians_are_analytic_and_correct():
    from avsim.perception.tracker import Track

    path = straight_road(length=400).lanes["lane_0"].centerline
    mpc = VehicleMPC(P)
    N = mpc.cfg.horizon
    ref = np.array([[*path.position(50 + 13 * i * mpc.cfg.dt), 0.0, 13.0] for i in range(N + 1)])
    p = path.position(95.0)
    preds = ConstantVelocityPredictor()(
        [Track(id=1, x=np.array([p[0], p[1], 0.0, 0.0]), P=np.eye(4) * 0.16)],
        np.linspace(0, 4, 9),
    )
    stop = StopLineConstraint(point=path.position(120.0), tangent=np.array([1.0, 0.0]))
    mpc._set_context(ref, None, preds, stop)
    x, u = np.array([60.0, -1.0, 0.05, 13.0, 0.03]), np.array([0.4, 0.1])
    cx, _ = mpc._cj_padded(x, u, 3)
    assert np.abs(cx - _fd_constraint_jacobian(mpc, x, u, 3)).max() < 1e-6


def test_mpc_cost_derivatives_are_analytic_and_correct():
    mpc = VehicleMPC(P)
    N = mpc.cfg.horizon
    mpc._set_context(np.array([[i * 1.5, 0.3 * i, 0.4 + 0.01 * i, 15.0] for i in range(N + 1)]),
                     None, [])
    x, u, eps = np.array([2.0, 1.0, 0.5, 14.0, 0.03]), np.array([0.4, 0.05]), 1e-5
    lx, lu, _, _, _ = mpc._stage_derivs(x, u, 3)
    g = np.array([(mpc._stage_cost(x + np.eye(5)[i] * eps, u, 3)
                   - mpc._stage_cost(x - np.eye(5)[i] * eps, u, 3)) / (2 * eps) for i in range(5)])
    assert np.abs(lx - g).max() < 1e-6


def test_understeer_correction_reproduces_the_steady_state_relation():
    mpc = VehicleMPC(P)
    kappa, V = 1 / 300.0, 16.0
    assert np.arctan(kappa * mpc._effective_wheelbase(V)) == pytest.approx(
        P.steady_state_steer(kappa, V), rel=1e-3
    )
    plain = VehicleMPC(P, MPCConfig(understeer_correction=False))
    assert np.arctan(kappa * plain._effective_wheelbase(V)) < P.steady_state_steer(kappa, V)


def test_mpc_holds_a_curve_against_the_nonlinear_plant():
    from avsim.control.mpc import CorridorStage
    from avsim.world.world import World

    net = straight_road(length=500, n_lanes=2, speed_limit=16.0, curvature=1 / 300.0)
    route = net.lanes["lane_0"].centerline
    w = World(net, P, dt=0.02)
    w.reset()
    w.place_ego(route, s=10.0, v=16.0)
    mpc = VehicleMPC(P)
    N, h, V = mpc.cfg.horizon, mpc.cfg.dt, 16.0
    cmd, s, errs = np.zeros(2), 10.0, []
    for k in range(int(15 / w.dt)):
        if k % 5 == 0:
            xr = w.ego_rear_axle()
            s = route.project(xr[0], xr[1], s)
            ss = s + V * np.arange(N + 1) * h
            ref = np.array([[*route.position(min(a, route.length)),
                             route.heading(min(a, route.length)), V] for a in ss])
            cor = [CorridorStage(route.position(min(a, route.length)),
                                 route.normal(min(a, route.length)), -1.55, 1.55) for a in ss]
            r = mpc.solve(np.array([xr[0], xr[1], xr[2], xr[3], w.ego[10]]), ref, cor)
            cmd = np.array([r.a_cmd, r.delta_cmd])
        w.step(cmd)
        xr = w.ego_rear_axle()
        errs.append(route.lateral_offset(xr[0], xr[1], route.project(xr[0], xr[1], s)))
    e = np.array(errs)
    assert np.abs(e).max() < 0.35
    assert np.sqrt((e[200:] ** 2).mean()) < 0.12
    assert w.ego[3] == pytest.approx(V, abs=0.3)
