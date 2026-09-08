"""The kinematic bicycle, in both reference points, plus its Jacobians.

Lecture 2/3, *The Kinematic Bicycle*:

    Every wheel velocity is tangent to a circle about the same instantaneous
    centre of rotation -- that is, no tire slips laterally.

    The model has no mass, no inertia, no friction coefficient and no tire
    force.  It therefore cannot know when the demanded lateral force is
    impossible.  It is a **geometry** model, and it is exact under its own
    assumption.

Two forms live here and they are *not* interchangeable term by term:

``rear_axle_field``  ``x = [X_r, Y_r, psi, v]``, ``u = [a, delta]``
    The common planning model.  ``v`` is the speed at the rear-axle centre and
    the geometric sideslip is zero by construction.

``cg_field``  ``x = [X_c, Y_c, psi, v]``, ``u = [a, delta]``
    ``v`` is the speed of the CG **along its own velocity direction**, which is
    rotated from the body x-axis by the geometric sideslip
    ``beta = arctan((l_r / L) tan delta)``.

:func:`rear_axle_to_cg` and :func:`cg_to_rear_axle` perform the lever-arm
transform.  Never mix a state equation written at one reference point with
constraints measured at another.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..core.conventions import KIN_NX, NU, wrap_to_pi
from .params import VehicleParams


def geometric_sideslip(delta: float, p: VehicleParams) -> float:
    """``beta = arctan((l_r / L) tan delta)``.

    This is a *geometric* sideslip angle, not the dynamic sideslip produced by
    a tire-force balance.  Same symbol, different object.
    """
    return float(np.arctan(p.l_r / p.L * np.tan(delta)))


def rear_axle_field(p: VehicleParams) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Continuous field ``f_c`` for ``x = [X_r, Y_r, psi, v]``, ``u = [a, delta]``.

    ``X_r' = v cos psi``,  ``Y_r' = v sin psi``,
    ``psi' = (v / L) tan delta``,  ``v' = a``.
    """
    L = p.L

    def f(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        psi, v = x[2], x[3]
        a, delta = u[0], u[1]
        return np.array([v * np.cos(psi), v * np.sin(psi), v / L * np.tan(delta), a])

    return f


def cg_field(p: VehicleParams) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Continuous field for the CG form ``x = [X_c, Y_c, psi, v]``.

    ``X_c' = v cos(psi + beta)``,  ``Y_c' = v sin(psi + beta)``,
    ``psi' = (v / l_r) sin beta``,  ``v' = a``.
    """
    l_r = p.l_r

    def f(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        psi, v = x[2], x[3]
        a, delta = u[0], u[1]
        beta = geometric_sideslip(delta, p)
        return np.array(
            [v * np.cos(psi + beta), v * np.sin(psi + beta), v / l_r * np.sin(beta), a]
        )

    return f


def rear_axle_to_cg(x_r: np.ndarray, delta: float, p: VehicleParams) -> np.ndarray:
    """Lever-arm transform ``[X_r, Y_r, psi, v_r] -> [X_c, Y_c, psi, v_c]``.

    ``p_c = p_r + l_r [cos psi, sin psi]`` and, since the rear-axle speed is
    the body-x speed, ``v_c = v_r / cos beta``.
    """
    X, Y, psi, v = x_r
    beta = geometric_sideslip(delta, p)
    return np.array(
        [
            X + p.l_r * np.cos(psi),
            Y + p.l_r * np.sin(psi),
            psi,
            v / max(np.cos(beta), 1e-6),
        ]
    )


def cg_to_rear_axle(x_c: np.ndarray, delta: float, p: VehicleParams) -> np.ndarray:
    """Inverse of :func:`rear_axle_to_cg`."""
    X, Y, psi, v = x_c
    beta = geometric_sideslip(delta, p)
    return np.array(
        [X - p.l_r * np.cos(psi), Y - p.l_r * np.sin(psi), psi, v * np.cos(beta)]
    )


def rear_axle_jacobians(x: np.ndarray, u: np.ndarray, p: VehicleParams) -> tuple[np.ndarray, np.ndarray]:
    """Analytic ``(A_c, B_c) = (df/dx, df/du)`` of :func:`rear_axle_field`.

    From the *World-Chart Jacobians* slide::

        A = [[0, 0, -v sin psi, cos psi],
             [0, 0,  v cos psi, sin psi],
             [0, 0,      0,     tan(delta)/L],
             [0, 0,      0,          0]]

        B = [[0, 0], [0, 0], [0, v / (L cos^2 delta)], [1, 0]]

    Read the structure: heading changes position sensitivity in proportion to
    speed; steering affects yaw rate in proportion to ``v``; and ``sec^2 delta``
    grows near 90 deg.  At standstill the whole steering column of ``B``
    vanishes -- the linearization loses the manoeuvre the nonlinear car can
    still perform by reversing, steering and advancing again.
    """
    psi, v = x[2], x[3]
    delta = u[1]
    A = np.zeros((KIN_NX, KIN_NX))
    A[0, 2] = -v * np.sin(psi)
    A[0, 3] = np.cos(psi)
    A[1, 2] = v * np.cos(psi)
    A[1, 3] = np.sin(psi)
    A[2, 3] = np.tan(delta) / p.L

    B = np.zeros((KIN_NX, NU))
    B[2, 1] = v / (p.L * np.cos(delta) ** 2)
    B[3, 0] = 1.0
    return A, B


def curvature_from_steer(delta: float, p: VehicleParams) -> float:
    """Path curvature ``kappa = tan(delta) / L`` of the rear-axle reference point."""
    return float(np.tan(delta) / p.L)


def steer_from_curvature(kappa: float, p: VehicleParams) -> float:
    """Inverse: the steering angle that produces a given curvature."""
    return float(np.arctan(kappa * p.L))


def lateral_acceleration(v: float, delta: float, p: VehicleParams) -> float:
    """``a_y = v^2 tan(delta) / L`` -- the kinematic model's own force demand.

    The kinematic model cannot refuse this, which is exactly why it must be
    checked externally against ``mu g``.
    """
    return float(v**2 * np.tan(delta) / p.L)


def validity_envelope(v: float, delta: float, p: VehicleParams, use_fraction: float = 0.5) -> dict:
    """Check the lecture's practical validity envelope for the kinematic model.

    ``|a_y| <= use_fraction * mu * g`` (the conservative planning rule is
    ``0.5 g ~ 4.4 m/s^2`` at ``mu = 0.9``).  Returns the demand, the limit and
    whether the approximation is being used inside its stated domain.
    """
    a_y = lateral_acceleration(v, delta, p)
    limit = p.max_lateral_accel(use_fraction)
    return {
        "a_y": a_y,
        "limit": limit,
        "usage": abs(a_y) / max(limit, 1e-9),
        "valid": bool(abs(a_y) <= limit),
    }
