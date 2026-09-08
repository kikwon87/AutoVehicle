"""Explicit Runge-Kutta integrators and the machinery to reason about them.

Lecture 2/3, *From Differential Equation to Difference Equation*:

    An integrator is an approximation of the flow map ``phi_h``, not an
    approximation of the differential equation.

Every integrator here turns a continuous field ``f_c(x, u)`` into the discrete
map ``F_h(x_k, u_k)`` the solver actually holds, under a **zero-order hold** on
``u``.  The hold itself creates no state error; the approximate integration
does.

The module also exposes the *stability function* ``R(z)`` of each method, so
the maximum affordable step for a given linear model can be computed rather
than guessed (Study Task 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

import numpy as np

Field = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ButcherTableau:
    """An explicit Runge-Kutta method in Butcher form.

    ``c`` are the stage times, ``A`` the (strictly lower triangular) stage
    coefficients and ``b`` the quadrature weights.  ``order`` is the classical
    order of consistency, i.e. the local error is ``O(h^{order+1})`` and the
    global error is ``O(h^{order})``.
    """

    name: str
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    order: int

    @property
    def stages(self) -> int:
        return len(self.b)

    def step(self, f: Field, x: np.ndarray, u: np.ndarray, h: float) -> np.ndarray:
        """One zero-order-hold step: ``x_{k+1} = F_h(x_k, u_k)``."""
        x = np.asarray(x, dtype=float)
        k = np.empty((self.stages, x.size), dtype=float)
        for i in range(self.stages):
            xi = x.copy()
            for j in range(i):
                aij = self.A[i, j]
                if aij != 0.0:
                    xi = xi + h * aij * k[j]
            k[i] = np.asarray(f(xi, u), dtype=float)
        return x + h * (self.b @ k)

    def stability_function(self, z: np.ndarray | complex) -> np.ndarray | complex:
        """``R(z)`` for the scalar test equation ``x' = lambda x``, ``z = h*lambda``.

        For an explicit method with ``s`` stages this is a polynomial in ``z``.
        The method is absolutely stable exactly where ``|R(z)| <= 1``.
        """
        z = np.asarray(z)
        s = self.stages
        # Stage values of the test equation: g_i = 1 + z * sum_j A_ij g_j
        g = np.zeros((s,) + z.shape, dtype=complex) if z.ndim else np.zeros(s, dtype=complex)
        for i in range(s):
            acc = np.ones_like(z, dtype=complex)
            for j in range(i):
                if self.A[i, j] != 0.0:
                    acc = acc + z * self.A[i, j] * g[j]
            g[i] = acc
        out = np.ones_like(z, dtype=complex)
        for i in range(s):
            out = out + z * self.b[i] * g[i]
        return out


EULER = ButcherTableau(
    name="euler",
    c=np.array([0.0]),
    A=np.zeros((1, 1)),
    b=np.array([1.0]),
    order=1,
)

MIDPOINT = ButcherTableau(
    name="midpoint",
    c=np.array([0.0, 0.5]),
    A=np.array([[0.0, 0.0], [0.5, 0.0]]),
    b=np.array([0.0, 1.0]),
    order=2,
)

HEUN = ButcherTableau(
    name="heun",
    c=np.array([0.0, 1.0]),
    A=np.array([[0.0, 0.0], [1.0, 0.0]]),
    b=np.array([0.5, 0.5]),
    order=2,
)

RK4 = ButcherTableau(
    name="rk4",
    c=np.array([0.0, 0.5, 0.5, 1.0]),
    A=np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    ),
    b=np.array([1.0, 2.0, 2.0, 1.0]) / 6.0,
    order=4,
)

METHODS: Dict[str, ButcherTableau] = {
    m.name: m for m in (EULER, MIDPOINT, HEUN, RK4)
}


def get_method(method: str | ButcherTableau) -> ButcherTableau:
    if isinstance(method, ButcherTableau):
        return method
    try:
        return METHODS[str(method).lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"unknown integrator {method!r}; available: {sorted(METHODS)}"
        ) from exc


def step(f: Field, x: np.ndarray, u: np.ndarray, h: float, method: str | ButcherTableau = "rk4") -> np.ndarray:
    """Single discrete step ``F_h(x, u)`` of the field ``f`` under a ZOH input."""
    return get_method(method).step(f, x, u, h)


def make_discrete(f: Field, h: float, method: str | ButcherTableau = "rk4") -> Field:
    """Return the discrete map ``F_h`` as a callable ``(x, u) -> x_next``.

    Handing the *same* callable to the simulator, to the MPC prediction and to
    the Jacobian routine is what makes "the derivatives match the discrete
    map" true by construction rather than by review.
    """
    tab = get_method(method)

    def F(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return tab.step(f, x, u, h)

    F.__name__ = f"F_{tab.name}_h{h:g}"  # type: ignore[attr-defined]
    return F


def rollout(
    f: Field,
    x0: np.ndarray,
    u_seq: Sequence[np.ndarray] | np.ndarray,
    h: float,
    method: str | ButcherTableau = "rk4",
) -> np.ndarray:
    """Integrate a whole input sequence, returning ``N+1`` states."""
    tab = get_method(method)
    u_seq = np.atleast_2d(np.asarray(u_seq, dtype=float))
    x = np.asarray(x0, dtype=float)
    out = np.empty((len(u_seq) + 1, x.size))
    out[0] = x
    for k, u in enumerate(u_seq):
        x = tab.step(f, x, u, h)
        out[k + 1] = x
    return out


def substep(
    f: Field,
    x: np.ndarray,
    u: np.ndarray,
    h: float,
    n_sub: int,
    method: str | ButcherTableau = "rk4",
) -> np.ndarray:
    """Advance by ``h`` using ``n_sub`` equal internal steps.

    The input is held constant over the whole interval, so this refines the
    *integration* without changing the zero-order hold -- which is exactly the
    knob you want when the plant must be more accurate than the controller's
    prediction model.
    """
    tab = get_method(method)
    hs = h / float(n_sub)
    for _ in range(n_sub):
        x = tab.step(f, x, u, hs)
    return x


def convergence_study(
    f: Field,
    x0: np.ndarray,
    u: np.ndarray,
    t_final: float,
    steps: Sequence[float],
    method: str | ButcherTableau,
    reference: np.ndarray | None = None,
    ref_substeps: int = 64,
) -> dict:
    """Measure the observed global order of a method on a constant-input task.

    Reproduces Study Task 1 ("reproduce the observed orders 1, 2, 2 and 4").
    The reference is either supplied or produced by RK4 at ``h/ref_substeps``.

    Returns a dict with ``h``, ``error`` and ``observed_order`` (the slope of
    ``log error`` against ``log h``, fitted by least squares).
    """
    tab = get_method(method)
    steps = np.asarray(sorted(steps, reverse=True), dtype=float)

    if reference is None:
        h_ref = steps.min() / ref_substeps
        n_ref = int(round(t_final / h_ref))
        xr = np.asarray(x0, dtype=float)
        for _ in range(n_ref):
            xr = RK4.step(f, xr, u, h_ref)
        reference = xr
    reference = np.asarray(reference, dtype=float)

    errors = []
    for h in steps:
        n = int(round(t_final / h))
        x = np.asarray(x0, dtype=float)
        for _ in range(n):
            x = tab.step(f, x, u, h)
        errors.append(float(np.linalg.norm(x - reference)))
    errors = np.asarray(errors)

    # Ignore points that have already hit the round-off floor of the
    # reference; including them biases the fitted slope downwards.
    floor = 1e3 * np.finfo(float).eps * max(1.0, float(np.linalg.norm(reference)))
    good = errors > floor
    if good.sum() >= 2:
        slope = float(np.polyfit(np.log(steps[good]), np.log(errors[good]), 1)[0])
    else:  # pragma: no cover - only when the method is exact for this field
        slope = float("nan")

    return {
        "method": tab.name,
        "nominal_order": tab.order,
        "h": steps,
        "error": errors,
        "observed_order": slope,
        "reference": reference,
    }


def max_stable_step(eigenvalues: np.ndarray, method: str | ButcherTableau, h_hi: float = 10.0) -> float:
    """Largest ``h`` for which every ``h*lambda_i`` lies in the stability region.

    Bisection on ``max_i |R(h lambda_i)| <= 1``.  This is the quantitative
    version of "which speeds can I still integrate at ``h = 0.1 s``" from
    Study Task 3.
    """
    tab = get_method(method)
    lam = np.asarray(eigenvalues, dtype=complex).ravel()

    def stable(h: float) -> bool:
        return bool(np.max(np.abs(tab.stability_function(h * lam))) <= 1.0)

    if not stable(1e-12):
        return 0.0
    lo, hi = 1e-12, h_hi
    if stable(hi):
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
    return lo
