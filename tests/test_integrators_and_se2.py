"""Study Tasks 1 and 2, plus the stability thresholds of Study Task 3."""

import numpy as np
import pytest

from avsim.core import se2
from avsim.core.integrators import (
    EULER,
    METHODS,
    MIDPOINT,
    RK4,
    convergence_study,
    max_stable_step,
    rollout,
)
from avsim.models.dynamic_bicycle import linear_single_track
from avsim.models.params import REFERENCE_VEHICLE as P

L, V, DELTA, T = 2.7, 10.0, 0.1, 4.0


def kinematic_field(x, u):
    _, _, psi, v = x
    a, delta = u
    return np.array([v * np.cos(psi), v * np.sin(psi), v / L * np.tan(delta), a])


@pytest.mark.parametrize("name,order", [("euler", 1), ("midpoint", 2), ("heun", 2), ("rk4", 4)])
def test_observed_convergence_order(name, order):
    """Study Task 1: reproduce the observed orders 1, 2, 2 and 4."""
    steps = {"rk4": [0.4, 0.2, 0.1, 0.05]}.get(name, [0.2, 0.1, 0.05, 0.025, 0.0125])
    r = convergence_study(
        kinematic_field, np.array([0.0, 0.0, 0.0, V]), np.array([0.0, DELTA]), T, steps, name
    )
    assert abs(r["observed_order"] - order) < 0.2, r


def test_lecture_global_error_figures():
    """Euler ~0.7 m and midpoint ~2 mm after 4 s at h = 0.1 s."""
    x0, u = np.array([0.0, 0.0, 0.0, V]), np.array([0.0, DELTA])
    ref = convergence_study(kinematic_field, x0, u, T, [0.1], "euler")["reference"]

    def err(method):
        x = rollout(kinematic_field, x0, np.tile(u, (40, 1)), 0.1, method)[-1]
        return float(np.linalg.norm(x[:2] - ref[:2]))

    assert 0.6 < err("euler") < 0.8
    assert 1e-3 < err("midpoint") < 4e-3
    assert err("rk4") < 1e-6


def test_stability_function_matches_exponential_to_order():
    for method in METHODS.values():
        z = -0.05
        exact = np.exp(z)
        approx = complex(method.stability_function(np.array(z)))
        assert abs(approx - exact) < abs(z) ** (method.order + 1)


def test_explicit_stability_limits_on_the_real_axis():
    """Euler is stable to h|lambda| = 2, RK4 to 2.7853."""
    assert max_stable_step([-100 + 0j], "euler") == pytest.approx(0.02, rel=1e-3)
    assert max_stable_step([-100 + 0j], "rk4") == pytest.approx(2.7853 / 100, rel=1e-3)


def test_low_speed_is_where_the_step_runs_out():
    """Study Task 3: at h = 0.1 s the linear model is unintegrable at low speed."""
    h = 0.1
    eig_slow = np.linalg.eigvals(linear_single_track(2.0, P)[0])
    eig_fast = np.linalg.eigvals(linear_single_track(20.0, P)[0])
    assert max_stable_step(eig_slow, "euler") < h
    assert max_stable_step(eig_fast, "euler") > h
    assert max_stable_step(eig_slow, "rk4") < h


# --- SE(2) --------------------------------------------------------------------

def test_group_identities():
    g1, g2 = se2.pose(1.0, 2.0, 0.4), se2.pose(-0.5, 0.3, -1.1)
    assert np.allclose(se2.inverse(g1) @ g1, np.eye(3), atol=1e-14)
    xi = np.array([0.3, -0.2, 0.7])
    assert np.allclose(se2.log(se2.exp(xi)), xi, atol=1e-12)
    assert np.allclose(se2.boxplus(g2, se2.boxminus(g1, g2)), g1, atol=1e-12)


@pytest.mark.parametrize("xi", [
    np.array([0.7, -0.3, 0.9]),
    np.array([1.0, 0.0, 1e-9]),      # the small-angle branch
    np.array([2.0, 0.5, -1.7]),
])
def test_right_jacobian_matches_finite_difference(xi):
    eps = 1e-7
    J = np.zeros((3, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        J[:, i] = (
            se2.log(se2.inverse(se2.exp(xi)) @ se2.exp(xi + d))
            - se2.log(se2.inverse(se2.exp(xi)) @ se2.exp(xi - d))
        ) / (2 * eps)
    assert np.allclose(J, se2.right_jacobian(xi), atol=1e-6)


def test_constant_twist_drift_reproduces_the_lecture():
    """Study Task 2: v = 10, omega = 0.5, h = 0.1, T = 4 -> 0.84 m Euler drift."""
    v, w, h, T = 10.0, 0.5, 0.1, 4.0
    xi = np.array([v, 0.0, w])
    exact = se2.unpose(se2.exp(T * xi))

    g = np.eye(3)
    for _ in range(int(T / h)):
        g = g @ se2.exp(h * xi)
    assert np.linalg.norm(se2.unpose(g)[:2] - exact[:2]) < 1e-12

    x = np.zeros(3)
    for _ in range(int(T / h)):
        x = x + h * np.array([v * np.cos(x[2]), v * np.sin(x[2]), w])
    assert np.linalg.norm(x[:2] - exact[:2]) == pytest.approx(0.84, abs=0.01)

    one_step = np.linalg.norm((h * xi)[:2] - se2.unpose(se2.exp(h * xi))[:2])
    assert one_step == pytest.approx(0.025, abs=0.001)


def test_sinc_branch_is_accurate_at_tiny_angles():
    for theta in (1e-12, 1e-9, 1e-6, 1e-3):
        g = se2.exp(np.array([1.0, 0.0, theta]))
        assert np.isfinite(g).all()
        assert se2.unpose(g)[0] == pytest.approx(np.sin(theta) / theta if theta > 1e-8 else 1.0, rel=1e-9)
