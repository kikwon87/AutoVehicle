"""Linearization, exact LTI discretization, and the two routes between them.

Lecture 2/3, *Linearize Then Discretize, or Discretize Then Linearize?*

    The two routes agree to first order in ``h`` but generally differ at higher
    order.  Inside a nonlinear optimizer, use derivatives of the **discrete
    constraint you actually evaluate**.

That rule is why :func:`discrete_jacobians` differentiates *through* the very
Runge-Kutta step the simulator and the MPC use, rather than exponentiating a
continuous Jacobian and hoping the two agree.

Also here:

* :func:`expm` -- scaling-and-squaring Pade matrix exponential (numpy only);
* :func:`van_loan` -- ``A_d`` and ``B_d`` from a single block exponential, with
  no assumption that ``A_c`` is invertible;
* :func:`trust_region_radius` -- how far a linearization may be trusted before
  the neglected second-order term exceeds a tolerance.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..core.integrators import get_method

Field = Callable[[np.ndarray, np.ndarray], np.ndarray]


# --- matrix exponential -------------------------------------------------------

def _pade_coeffs(p: int = 6) -> np.ndarray:
    """Numerator coefficients of the diagonal Pade(p, p) approximant of ``exp``.

    ``c_k = (2p - k)! p! / ((2p)! k! (p - k)!)``.  The denominator is the same
    polynomial evaluated at ``-x``, which is why only one coefficient array is
    needed.
    """
    from math import factorial as fac

    return np.array(
        [fac(2 * p - k) * fac(p) / (fac(2 * p) * fac(k) * fac(p - k)) for k in range(p + 1)],
        dtype=float,
    )


_PADE_COEFFS = _pade_coeffs(6)


def expm(A: np.ndarray) -> np.ndarray:
    """Matrix exponential by scaling-and-squaring with a Pade(6,6) core.

    ``exp(A) = (exp(A / 2^s))^{2^s}`` with ``s`` chosen so that
    ``||A / 2^s||_inf <= 1/2``, where the diagonal Pade approximant is accurate
    to machine precision.  Kept dependency-free so the core numerics of this
    package need nothing beyond numpy.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expm expects a square matrix")
    n = A.shape[0]
    norm = float(np.abs(A).sum(axis=1).max())
    s = max(0, int(np.ceil(np.log2(max(norm, 1e-300) / 0.5)))) if norm > 0.5 else 0
    As = A / (2.0**s)

    # Diagonal Pade: N(As) / D(As) with D(x) = N(-x).
    U = np.zeros((n, n))
    V = np.zeros((n, n))
    Apow = np.eye(n)
    for k, c in enumerate(_PADE_COEFFS):
        term = c * Apow
        if k % 2 == 0:
            U += term
            V += term
        else:
            U += term
            V -= term
        Apow = Apow @ As
    E = np.linalg.solve(V, U)
    for _ in range(s):
        E = E @ E
    return E


def van_loan(A_c: np.ndarray, B_c: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretization via a single block exponential.

    ``exp([[A_c, B_c], [0, 0]] h) = [[A_d, B_d], [0, I]]``

    No assumption that ``A_c`` is invertible is needed -- which is exactly why
    this form is preferred over ``B_d = A_c^{-1}(A_d - I) B_c`` for vehicle
    models, whose ``A_c`` is routinely singular (position states integrate
    velocity and nothing feeds back into them).
    """
    A_c = np.atleast_2d(np.asarray(A_c, dtype=float))
    B_c = np.atleast_2d(np.asarray(B_c, dtype=float))
    n, m = A_c.shape[0], B_c.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A_c
    M[:n, n:] = B_c
    E = expm(M * h)
    return E[:n, :n], E[:n, n:]


# --- Jacobians ----------------------------------------------------------------

def numeric_jacobians(
    F: Field, x: np.ndarray, u: np.ndarray, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference ``(dF/dx, dF/du)`` of any map, discrete or continuous.

    Central differences (not forward) because the ``O(eps)`` bias of a forward
    difference shows up directly as a systematic model error in an SQP step,
    and is easily mistaken for a badly conditioned problem.
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    f0 = np.asarray(F(x, u), dtype=float)
    nx, nu, nf = x.size, u.size, f0.size

    A = np.empty((nf, nx))
    for i in range(nx):
        dx = np.zeros(nx)
        dx[i] = eps * max(1.0, abs(x[i]))
        A[:, i] = (F(x + dx, u) - F(x - dx, u)) / (2 * dx[i])

    B = np.empty((nf, nu))
    for j in range(nu):
        du = np.zeros(nu)
        du[j] = eps * max(1.0, abs(u[j]))
        B[:, j] = (F(x, u + du) - F(x, u - du)) / (2 * du[j])
    return A, B


def discrete_jacobians(
    f: Field,
    x: np.ndarray,
    u: np.ndarray,
    h: float,
    method: str = "rk4",
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """**Discretize, then linearize**: derivatives of the actual discrete map.

    Differentiates ``F_h = RK(f, h)`` itself, so the Jacobians are consistent
    with the equality constraint the solver evaluates.  This is the route to
    use inside a nonlinear optimizer.
    """
    tab = get_method(method)

    def F(xx: np.ndarray, uu: np.ndarray) -> np.ndarray:
        return tab.step(f, xx, uu, h)

    return numeric_jacobians(F, x, u, eps)


def continuous_then_discrete_jacobians(
    f: Field,
    x: np.ndarray,
    u: np.ndarray,
    h: float,
    eps: float = 1e-6,
    A_c: np.ndarray | None = None,
    B_c: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """**Linearize, then discretize**: ``A_c, B_c`` then an exact LTI ZOH step.

    Exact for the frozen linear model and standard for LTI analysis and
    controller design.  It is *not* the derivative of the nonlinear discrete
    map, and using it as such inside an SQP introduces an ``O(h^2)``
    inconsistency between the constraint and its gradient.
    """
    if A_c is None or B_c is None:
        A_c, B_c = numeric_jacobians(f, x, u, eps)
    return van_loan(A_c, B_c, h)


def jacobian_route_mismatch(
    f: Field, x: np.ndarray, u: np.ndarray, h: float, method: str = "rk4"
) -> dict:
    """Quantify the gap between the two linearization routes.

    Returns the max-norm difference of ``A_d`` and ``B_d``.  The gap is
    ``O(h^2)`` for a first-order-consistent pair and shrinks with ``h``; if it
    does *not* shrink, one of the two routes is wrong.
    """
    A1, B1 = discrete_jacobians(f, x, u, h, method)
    A2, B2 = continuous_then_discrete_jacobians(f, x, u, h)
    return {
        "A_gap": float(np.abs(A1 - A2).max()),
        "B_gap": float(np.abs(B1 - B2).max()),
        "A_discretize_then_linearize": A1,
        "A_linearize_then_discretize": A2,
    }


# --- trust region -------------------------------------------------------------

def trust_region_radius(
    f: Field,
    x: np.ndarray,
    u: np.ndarray,
    tol: float = 0.05,
    directions: int = 16,
    r_max: float = 10.0,
    seed: int = 0,
) -> float:
    """Largest ``||dx||`` for which the linear model's relative error stays under ``tol``.

    Lecture 2/3, *Linearization Has a Trust Region*: a Jacobian is informative
    only together with the state, the input order, the reference point and the
    operating point.  This turns "small enough" into a number, by sampling
    random directions and bisecting on the residual
    ``||f(x + dx, u) - f(x, u) - A dx|| / ||f(x + dx, u) - f(x, u)||``.
    """
    rng = np.random.default_rng(seed)
    A, _ = numeric_jacobians(f, x, u)
    f0 = np.asarray(f(x, u), dtype=float)

    def rel_err(r: float) -> float:
        worst = 0.0
        for _ in range(directions):
            d = rng.normal(size=x.size)
            d /= np.linalg.norm(d)
            df_true = np.asarray(f(x + r * d, u), dtype=float) - f0
            df_lin = A @ (r * d)
            den = max(np.linalg.norm(df_true), 1e-12)
            worst = max(worst, float(np.linalg.norm(df_true - df_lin) / den))
        return worst

    lo, hi = 1e-6, r_max
    if rel_err(hi) <= tol:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if rel_err(mid) <= tol:
            lo = mid
        else:
            hi = mid
    return lo
