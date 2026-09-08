"""Vehicle parameters, tires, suspension, and the two bicycle models."""

import numpy as np
import pytest

from avsim.core.integrators import rollout, step
from avsim.models import suspension as susp
from avsim.models.dynamic_bicycle import DynamicBicycle, linear_single_track
from avsim.models.embedded import BlendParams, blend_weight, embedded_field
from avsim.models.frenet import frenet_domain_margin, max_offset_for_curvature
from avsim.models.kinematic_bicycle import (
    cg_field,
    curvature_from_steer,
    lateral_acceleration,
    rear_axle_field,
    rear_axle_jacobians,
    rear_axle_to_cg,
    steer_from_curvature,
)
from avsim.models.linearization import (
    expm,
    jacobian_route_mismatch,
    numeric_jacobians,
    rk4_jacobians,
    van_loan,
)
from avsim.models.params import REFERENCE_VEHICLE as P
from avsim.models.params import VehicleParams
from avsim.models.tire import PacejkaTire, friction_ellipse_usage


# --- parameters ---------------------------------------------------------------

def test_reference_vehicle_matches_the_lecture():
    assert P.L == pytest.approx(2.7)
    assert P.understeer_gradient == pytest.approx(3.75e-3, rel=1e-6)
    assert P.characteristic_speed == pytest.approx(26.8, abs=0.1)
    assert P.characteristic_speed * 3.6 == pytest.approx(96.6, abs=0.5)


@pytest.mark.parametrize(
    "V,kappa,geom,force",
    [(15.0, 0.01, 0.0270, 0.0084), (30.0, 0.005, 0.0135, 0.0169)],
)
def test_feedforward_decomposition(V, kappa, geom, force):
    """The lecture's table: at 30 m/s the force term is the larger of the two."""
    assert P.L * kappa == pytest.approx(geom, abs=5e-5)
    assert P.understeer_gradient * V**2 * kappa == pytest.approx(force, abs=5e-5)
    assert P.steady_state_steer(kappa, V) == pytest.approx(geom + force, abs=1e-4)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        VehicleParams(C_f=-1.0)
    with pytest.raises(ValueError):
        VehicleParams(brake_bias_front=1.4)


# --- tires ---------------------------------------------------------------------

def test_pacejka_matches_requested_slope_and_peak():
    tire = PacejkaTire.for_axle(P.C_f, P.F_z_front_static, P.tire, np.deg2rad(8.0))
    assert tire.cornering_stiffness(0.0) == pytest.approx(P.C_f, rel=1e-5)
    assert np.rad2deg(tire.peak_slip_angle()) == pytest.approx(8.0, abs=0.2)
    peak = max(tire.lateral_force(a, P.F_z_front_static) for a in np.linspace(0, 0.5, 400))
    assert peak == pytest.approx(P.mu * P.F_z_front_static, rel=1e-3)


def test_linear_regime_is_only_local():
    """The lecture's 'valid for |alpha| <~ 4 deg' is a statement about the slope."""
    tire = PacejkaTire.for_axle(P.C_f, P.F_z_front_static, P.tire)
    assert tire.cornering_stiffness(np.deg2rad(1.0)) > 0.9 * P.C_f
    assert tire.cornering_stiffness(np.deg2rad(8.0)) < 0.3 * P.C_f


def test_friction_ellipse():
    cap = 0.9 * 5000.0
    assert friction_ellipse_usage(cap, 0.0, 5000.0, 0.9) == pytest.approx(1.0)
    assert friction_ellipse_usage(0.0, 0.0, 5000.0, 0.9) == 0.0


# --- suspension -----------------------------------------------------------------

def test_load_transfer_conserves_total_load():
    for ax, ay in [(0, 0), (-5, 0), (0, 4), (-5, 4), (3, -3)]:
        loads = susp.quasi_static_load_transfer(ax, ay, P)
        assert loads.sum() == pytest.approx(P.m * 9.80665, rel=1e-9)
        assert (loads >= 0).all()


def test_braking_loads_the_front_and_cornering_loads_the_outside():
    fl, fr, rl, rr = susp.quasi_static_load_transfer(-5.0, 0.0, P)
    assert fl + fr > rl + rr
    # a_y > 0 is leftward acceleration, which loads the right-hand wheels
    fl, fr, rl, rr = susp.quasi_static_load_transfer(0.0, 4.0, P)
    assert fr > fl and rr > rl


# --- kinematic bicycle -----------------------------------------------------------

def test_rear_axle_and_cg_forms_describe_the_same_path():
    delta, u = 0.1, np.array([0.0, 0.1])
    x0r = np.array([0.0, 0.0, 0.0, 10.0])
    Xr = rollout(rear_axle_field(P), x0r, np.tile(u, (400, 1)), 0.01, "rk4")
    Xc = rollout(cg_field(P), rear_axle_to_cg(x0r, delta, P), np.tile(u, (400, 1)), 0.01, "rk4")
    mapped = np.array([rear_axle_to_cg(x, delta, P) for x in Xr])
    assert np.abs(mapped - Xc).max() < 1e-10


def test_analytic_jacobians_match_finite_differences():
    x, u = np.array([1.0, 2.0, 0.3, 12.0]), np.array([0.5, 0.08])
    A, B = rear_axle_jacobians(x, u, P)
    An, Bn = numeric_jacobians(rear_axle_field(P), x, u)
    assert np.abs(A - An).max() < 1e-7
    assert np.abs(B - Bn).max() < 1e-7


def test_steering_and_curvature_are_inverses():
    for kappa in (-0.2, 0.0, 0.05, 0.2):
        assert curvature_from_steer(steer_from_curvature(kappa, P), P) == pytest.approx(kappa)


def test_urban_turn_lateral_acceleration_from_the_lecture():
    """8 m/s on a 30 m radius gives 2.13 m/s^2 -- inside the kinematic envelope."""
    delta = steer_from_curvature(1 / 30.0, P)
    assert lateral_acceleration(8.0, delta, P) == pytest.approx(2.13, abs=0.01)


# --- dynamic bicycle -------------------------------------------------------------

@pytest.mark.parametrize("V", [8.0, 15.0, 20.0])
def test_linear_tires_reproduce_the_understeer_gradient(V):
    """With linear tires the plant follows delta_ss = (L + K_us V^2) kappa."""
    model = DynamicBicycle(P, tire_model="linear")
    f = model.field()
    R, h = 100.0, 0.005
    z = model.initial_state(v=V)
    delta = P.steady_state_steer(1 / R, V)
    for _ in range(int(12.0 / h)):
        z = model.sanitize(step(f, z, np.array([(V - z[3]) * 1.5, delta]), h, "rk4"))
    R_actual = z[3] / z[5]
    assert abs(R_actual - R) / R < 0.03


def test_pacejka_saturation_shows_up_as_a_balance_shift():
    """The same manoeuvre with a saturating tire deviates, and the diagnostic says why."""
    model = DynamicBicycle(P)
    balance = model.limit_balance()
    assert balance["linear_behaviour"] == "understeer"
    # These parameters have a shorter rear reach, so the rear gives up first.
    assert balance["linear_reach_rear_deg"] < balance["linear_reach_front_deg"]
    assert balance["terminal_behaviour"] == "oversteer"


def test_linear_single_track_is_undefined_at_standstill():
    with pytest.raises(ValueError):
        linear_single_track(0.0, P)
    A, _ = linear_single_track(1.0, P)
    assert np.linalg.norm(A) > np.linalg.norm(linear_single_track(20.0, P)[0])


def test_plant_stops_rather_than_reversing_under_braking():
    model = DynamicBicycle(P)
    f = model.field()
    z = model.initial_state(v=5.0)
    for _ in range(int(6.0 / 0.01)):
        z = model.sanitize(step(f, z, np.array([-6.0, 0.0]), 0.01, "rk4"))
    assert z[3] == pytest.approx(0.0, abs=1e-6)


# --- blended model ---------------------------------------------------------------

def test_blend_weight_transitions_where_stated():
    bp = BlendParams()
    assert blend_weight(0.0, bp) < 1e-3
    assert blend_weight(bp.V_star, bp) == pytest.approx(0.5)
    assert blend_weight(20.0, bp) > 0.999


def test_embedded_model_reduces_to_the_kinematic_one_at_low_speed():
    bp = BlendParams()
    f, fk = embedded_field(P, bp), cg_field(P)
    u = np.array([0.0, 0.15])
    beta = np.arctan(P.l_r / P.L * np.tan(u[1]))
    V = 1.0
    z0 = np.array([0, 0, 0, V, V * np.tan(beta), V / P.l_r * np.tan(beta)])
    Z = rollout(f, z0, np.tile(u, (200, 1)), 0.02, "rk4")
    Xk = rollout(fk, np.array([0, 0, 0, V / np.cos(beta)]), np.tile(u, (200, 1)), 0.02, "rk4")
    assert np.abs(Z[:, :2] - Xk[:, :2]).max() < 1e-4


def test_embedded_model_is_finite_from_standstill():
    Z = rollout(embedded_field(P), np.zeros(6), np.tile(np.array([1.0, 0.3]), (200, 1)), 0.02, "rk4")
    assert np.isfinite(Z).all()


# --- Frenet ----------------------------------------------------------------------

def test_frenet_domain_is_one_sided():
    """|e_y| < 1/|kappa| is NOT equivalent to 1 - kappa e_y > 0."""
    kappa = 0.05
    lo, hi = max_offset_for_curvature(kappa)
    assert lo == -np.inf and hi == pytest.approx(0.9 / kappa)
    assert frenet_domain_margin(kappa, -100.0) > 0     # far to the right is fine
    assert frenet_domain_margin(kappa, 1 / kappa) < 0  # the centre of curvature is not


# --- linearization ----------------------------------------------------------------

def test_expm_against_a_known_closed_form():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    assert np.allclose(expm(A * 0.1), np.array([[1.0, 0.1], [0.0, 1.0]]))
    lam = -3.0
    assert expm(np.array([[lam]]) * 0.7)[0, 0] == pytest.approx(np.exp(lam * 0.7))


def test_van_loan_needs_no_invertible_A():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])  # singular
    B = np.array([[0.0], [1.0]])
    Ad, Bd = van_loan(A, B, 0.1)
    assert np.allclose(Ad, [[1.0, 0.1], [0.0, 1.0]])
    assert np.allclose(Bd.ravel(), [0.005, 0.1])


def test_rk4_jacobians_match_finite_differences():
    L = P.L

    def f(x, u):
        _, _, psi, v, d = x
        return np.array([v * np.cos(psi), v * np.sin(psi), v / L * np.tan(d), u[0], u[1]])

    def A_c(x, u):
        _, _, psi, v, d = x
        A = np.zeros((5, 5))
        A[0, 2], A[0, 3] = -v * np.sin(psi), np.cos(psi)
        A[1, 2], A[1, 3] = v * np.cos(psi), np.sin(psi)
        A[2, 3], A[2, 4] = np.tan(d) / L, v / (L * np.cos(d) ** 2)
        return A

    def B_c(x, u):
        B = np.zeros((5, 2))
        B[3, 0] = B[4, 1] = 1.0
        return B

    x, u = np.array([3.0, -1.0, 0.4, 14.0, 0.06]), np.array([0.8, 0.15])
    from avsim.models.linearization import discrete_jacobians

    for h in (0.2, 0.1, 0.05):
        Aa, Ba = rk4_jacobians(A_c, B_c, f, x, u, h)
        An, Bn = discrete_jacobians(f, x, u, h, "rk4", eps=1e-7)
        assert np.abs(Aa - An).max() < 1e-7
        assert np.abs(Ba - Bn).max() < 1e-7


def test_the_two_linearization_routes_differ_at_second_order():
    x, u = np.array([0.0, 0.0, 0.2, 15.0]), np.array([0.5, 0.05])
    gaps = [jacobian_route_mismatch(rear_axle_field(P), x, u, h)["A_gap"] for h in (0.2, 0.1, 0.05)]
    ratios = [gaps[i] / gaps[i + 1] for i in range(2)]
    assert all(3.0 < r < 5.0 for r in ratios), ratios  # O(h^2)
