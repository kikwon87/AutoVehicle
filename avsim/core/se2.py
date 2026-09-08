"""Planar rigid-body motion on ``SE(2)``.

Lecture 2/3, *Planar Rigid-Body Motion on SE(2)*:

    Stacking position and heading into a three-vector is not a vector space
    operation -- a pose composes, it does not add.

A pose ``g = (R(psi), p)`` is stored as a 3x3 homogeneous matrix.  A *twist*
``xi = [v_x, v_y, omega]`` lives in the Lie algebra ``se(2)``; the exponential
map turns a **constant** twist held for ``h`` seconds into the exact finite
motion, which is what an Euler step on ``(X, Y, psi)`` fails to reproduce.

All small-angle branches use the ``sinc``-style helpers in
:mod:`avsim.core.conventions` so the maps stay accurate to full double
precision as ``omega -> 0``.
"""

from __future__ import annotations

import numpy as np

from .conventions import one_minus_cos_over_x, sinc_taylor, wrap_to_pi

__all__ = [
    "rot",
    "pose",
    "unpose",
    "hat",
    "vee",
    "exp",
    "log",
    "adjoint",
    "right_jacobian",
    "compose",
    "inverse",
    "boxplus",
    "boxminus",
]


def rot(psi: float) -> np.ndarray:
    """2x2 rotation by ``psi`` (counter-clockwise positive)."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


def pose(x: float, y: float, psi: float) -> np.ndarray:
    """Homogeneous 3x3 pose from ``(x, y, psi)``."""
    g = np.eye(3)
    g[:2, :2] = rot(psi)
    g[0, 2] = x
    g[1, 2] = y
    return g


def unpose(g: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pose`: extract ``(x, y, psi)`` from a 3x3 pose."""
    return np.array([g[0, 2], g[1, 2], np.arctan2(g[1, 0], g[0, 0])])


def hat(xi: np.ndarray) -> np.ndarray:
    """Wedge operator ``se(2) -> R^{3x3}``.

    ``xi = [v_x, v_y, omega]`` is expressed in the **body** frame.
    """
    vx, vy, w = np.asarray(xi, dtype=float)
    return np.array([[0.0, -w, vx], [w, 0.0, vy], [0.0, 0.0, 0.0]])


def vee(X: np.ndarray) -> np.ndarray:
    """Inverse of :func:`hat`."""
    return np.array([X[0, 2], X[1, 2], X[1, 0]])


def _V(theta: float) -> np.ndarray:
    """Left Jacobian of ``SE(2)``: the matrix mapping ``v`` to the translation.

    ``V(theta) = [[sin t / t, -(1-cos t)/t], [(1-cos t)/t, sin t / t]]``
    """
    a = float(sinc_taylor(theta))
    b = float(one_minus_cos_over_x(theta))
    return np.array([[a, -b], [b, a]])


def exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map: constant body twist ``xi`` applied for unit time.

    To move for ``h`` seconds at constant twist, pass ``h * xi``.  This is the
    *exact* flow of the constant-twist field, whereas an Euler step on the
    ``(X, Y, psi)`` chart integrates the heading and the position with the same
    frozen slope and therefore cuts the corner.
    """
    xi = np.asarray(xi, dtype=float)
    theta = xi[2]
    g = np.eye(3)
    g[:2, :2] = rot(theta)
    g[:2, 2] = _V(theta) @ xi[:2]
    return g


def log(g: np.ndarray) -> np.ndarray:
    """Logarithm map ``SE(2) -> se(2)``; inverse of :func:`exp`.

    Used to turn a *relative pose* (reference vs. actual) into a local tangent
    vector -- the "Lie-log error" of the path-tracking slide.
    """
    theta = wrap_to_pi(np.arctan2(g[1, 0], g[0, 0]))
    Vinv = np.linalg.inv(_V(theta))
    v = Vinv @ g[:2, 2]
    return np.array([v[0], v[1], float(theta)])


def adjoint(g: np.ndarray) -> np.ndarray:
    """``Ad_g``: changes the frame in which a twist is expressed."""
    R = g[:2, :2]
    p = g[:2, 2]
    Ad = np.eye(3)
    Ad[:2, :2] = R
    Ad[0, 2] = p[1]
    Ad[1, 2] = -p[0]
    return Ad


def _S(theta: float) -> np.ndarray:
    """``S(theta) = sum_{m>=0} (-theta J)^m / (m+2)!`` in closed form.

    Appears in the top-right block of the right Jacobian.  Both entries are
    removable singularities at ``theta = 0`` and are expanded there.
    """
    if abs(theta) < 1e-5:
        a = 0.5 - theta**2 / 24.0          # (1 - cos t) / t^2
        b = theta / 6.0 - theta**3 / 120.0  # (t - sin t) / t^2
    else:
        a = (1.0 - np.cos(theta)) / theta**2
        b = (theta - np.sin(theta)) / theta**2
    return np.array([[a, b], [-b, a]])


def right_jacobian(xi: np.ndarray) -> np.ndarray:
    """Right Jacobian ``J_r(xi)`` of the ``SE(2)`` exponential.

    Defined by ``exp(xi + dxi) ~= exp(xi) exp(J_r(xi) dxi)``; it corrects the
    false assumption that ``exp(a + b) = exp(a) exp(b)``.  It is therefore the
    **input Jacobian** of a constant-twist group step, ``B_k = J_r(h xi_k) h``.

    Closed form (Lecture 2/3), with ``rho`` the translation part of the twist
    and ``theta`` its rotation part::

        J_r(rho, theta) = [[ V(-theta),  S(theta) J rho ],
                           [     0     ,        1       ]]

    where ``V`` is the left Jacobian block already used by :func:`exp` and
    ``S`` is defined in :func:`_S`.  The series branch is used near
    ``theta = 0``; the apparent divisions are removable singularities.
    """
    xi = np.asarray(xi, dtype=float)
    rho = xi[:2]
    theta = float(xi[2])
    Jmat = np.array([[0.0, -1.0], [1.0, 0.0]])

    Jr = np.eye(3)
    Jr[:2, :2] = _V(-theta)
    Jr[:2, 2] = _S(theta) @ (Jmat @ rho)
    return Jr


def compose(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """Group product ``g1 * g2``."""
    return g1 @ g2


def inverse(g: np.ndarray) -> np.ndarray:
    """Group inverse, computed in closed form (never with ``np.linalg.inv``)."""
    R = g[:2, :2]
    p = g[:2, 2]
    out = np.eye(3)
    out[:2, :2] = R.T
    out[:2, 2] = -R.T @ p
    return out


def boxplus(g: np.ndarray, dxi: np.ndarray) -> np.ndarray:
    """Right retraction ``g [+] dxi = g exp(dxi)``.

    This is how an optimizer updates a pose decision variable: the increment
    lives in the tangent space, the iterate stays on the group.
    """
    return g @ exp(dxi)


def boxminus(g_a: np.ndarray, g_b: np.ndarray) -> np.ndarray:
    """Right difference ``g_a [-] g_b = log(g_b^{-1} g_a)``."""
    return log(inverse(g_b) @ g_a)
