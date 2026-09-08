"""Frenet-frame kinematics and the singularity that comes with them.

Lecture 2/3, *The Frenet Frame Follows the Road*:

    Benefit: road boundaries become a box constraint
    ``e_y^min(s) <= e_y <= e_y^max(s)`` instead of a distance to an arbitrary
    polygon.
    Price: ``kappa(s)`` enters the dynamics, and the coordinate map has a
    curvature-dependent singularity.

The kinematics, derived by differentiating the offset curve
``p = p_r(s) + e_y n(s)``::

    s'    = v cos(e_psi + beta) / (1 - kappa(s) e_y)
    e_y'  = v sin(e_psi + beta)
    e_psi' = psi' - kappa(s) s'

The denominator comes from differentiating the *offset curve*, not from vehicle
dynamics -- it would appear for a point mass too.

Domain (stated correctly).  The map is regular iff ``1 - kappa e_y > 0``.  The
symmetric shorthand ``|e_y| < 1/|kappa|`` is **not** equivalent: for
``kappa > 0`` the exact one-sided condition is ``e_y < 1/kappa``, with no lower
bound from this term.  :func:`frenet_domain_margin` implements the correct one.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..core.conventions import wrap_to_pi
from .params import VehicleParams

#: Default safety margin on the Frenet denominator: ``1 - kappa e_y >= EPS_DOMAIN``.
EPS_DOMAIN = 0.10


def frenet_denominator(kappa: float, e_y: float) -> float:
    """``1 - kappa e_y``.  Positive means the coordinate map is regular here."""
    return float(1.0 - kappa * e_y)


def frenet_domain_margin(kappa: float, e_y: float, eps: float = EPS_DOMAIN) -> float:
    """Slack in ``1 - kappa e_y >= eps``; negative means the map has broken.

    Bound this explicitly whenever Frenet states are decision variables --
    otherwise the solver is free to walk into the singularity and will happily
    report a "better" cost on the far side of it.
    """
    return frenet_denominator(kappa, e_y) - eps


def max_offset_for_curvature(kappa: float, eps: float = EPS_DOMAIN) -> tuple[float, float]:
    """The exact one-sided bounds on ``e_y`` implied by the domain condition.

    Returns ``(e_y_min, e_y_max)``.  For ``kappa > 0`` only the upper bound is
    finite; for ``kappa < 0`` only the lower one; for ``kappa = 0`` neither.
    """
    if abs(kappa) < 1e-12:
        return (-np.inf, np.inf)
    bound = (1.0 - eps) / kappa
    return (-np.inf, float(bound)) if kappa > 0 else (float(bound), np.inf)


def frenet_field(
    p: VehicleParams,
    curvature: Callable[[float], float],
    eps: float = EPS_DOMAIN,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Field for the Frenet-frame kinematic bicycle.

    State ``x = [s, e_y, e_psi, v]``, input ``u = [a, delta]``.  ``curvature``
    supplies ``kappa(s)`` for the reference path.

    The denominator is clamped at ``eps`` rather than allowed to cross zero:
    an integrator that steps through the singularity produces a finite,
    plausible and completely wrong trajectory, which is worse than a raised
    error.
    """
    l_r, L = p.l_r, p.L

    def f(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        s, e_y, e_psi, v = x
        a, delta = u
        beta = np.arctan(l_r / L * np.tan(delta))
        kappa = float(curvature(s))
        den = max(frenet_denominator(kappa, e_y), eps)
        s_dot = v * np.cos(e_psi + beta) / den
        e_y_dot = v * np.sin(e_psi + beta)
        psi_dot = v / l_r * np.sin(beta)
        return np.array([s_dot, e_y_dot, psi_dot - kappa * s_dot, a])

    return f


def cartesian_to_frenet(
    x: float,
    y: float,
    psi: float,
    v: float,
    s_ref: float,
    path,
) -> np.ndarray:
    """Project a world pose onto a reference path, returning ``[s, e_y, e_psi, v]``.

    ``path`` must expose ``project(x, y, s_guess)`` returning ``s``, plus
    ``position(s)`` and ``heading(s)``; :class:`avsim.world.path.ReferencePath`
    does.  ``s_ref`` seeds the projection so the branch stays continuous along
    a trajectory -- re-projecting globally each step is what makes a controller
    jump between two lanes of a hairpin.
    """
    s = path.project(x, y, s_guess=s_ref)
    px, py = path.position(s)
    theta = path.heading(s)
    # Signed offset along the path normal n = [-sin theta, cos theta].
    e_y = -(x - px) * np.sin(theta) + (y - py) * np.cos(theta)
    e_psi = wrap_to_pi(psi - theta)
    return np.array([s, e_y, e_psi, v])


def frenet_to_cartesian(state: np.ndarray, path) -> np.ndarray:
    """Inverse of :func:`cartesian_to_frenet`: ``[s, e_y, e_psi, v] -> [X, Y, psi, v]``."""
    s, e_y, e_psi, v = state
    px, py = path.position(s)
    theta = path.heading(s)
    return np.array(
        [px - e_y * np.sin(theta), py + e_y * np.cos(theta), wrap_to_pi(theta + e_psi), v]
    )
