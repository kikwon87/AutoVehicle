"""A single state space valid from standstill to highway speed.

Lecture 2/3, *Switching and Blending: The State Spaces Must Match*:

    The shorthand ``f = lambda f_kin + (1 - lambda) f_dyn`` is **undefined** if
    ``f_kin`` is four-dimensional and ``f_dyn`` is six-dimensional, or if the
    two use different reference points.

The construction implemented here is the deck's *common-state embedding*: both
regimes are lifted to ``z = [X, Y, psi, v_x, v_y, r]`` and the kinematic
constraints are made to **relax** rather than disappear::

    v_x' = a_x
    v_y' = (v_y* - v_y) / tau,   v_y* = v_x tan(beta)
    r'   = (r*   - r  ) / tau,   r*   = (v_x / l_r) tan(beta)

    f(z, u) = (1 - w(V)) f_kin_emb(z, u) + w(V) f_dyn(z, u)
    w(V)    = 0.5 (1 + tanh((|V| - V*) / dV))

``tau``, ``V*`` and ``dV`` are **modelling parameters and must be validated**.
This is one optimization-friendly construction, not the uniquely correct one.

Why bother: the linear dynamic model's coefficients scale as ``1/V`` and blow
up at standstill.  That is a failure of the slip-angle description, not
evidence that the physical car becomes infinitely fast.  The remedy is to
change the model near zero speed -- not to shrink the integration step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .kinematic_bicycle import geometric_sideslip
from .params import VehicleParams
from .tire import SLIP_EPS


@dataclass(frozen=True)
class BlendParams:
    """Parameters of the kinematic/dynamic blend.

    ``V_star`` is the speed at which the two fields are weighted equally and
    ``dV`` sets the width of the transition.  The lecture's running example
    uses "the embedded kinematic field dominates below 5 m/s, smooth weight
    over 5-7 m/s", which is ``V_star = 6``, ``dV = 1``.
    """

    V_star: float = 6.0
    dV: float = 1.0
    tau: float = 0.10  #: relaxation time onto the kinematic manifold [s]


def blend_weight(V: float, bp: BlendParams) -> float:
    """``w(V) = 0.5 (1 + tanh((|V| - V*) / dV))``; 0 = kinematic, 1 = dynamic."""
    return float(0.5 * (1.0 + np.tanh((abs(V) - bp.V_star) / bp.dV)))


def kinematic_embedded_field(
    p: VehicleParams, bp: BlendParams
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """The kinematic bicycle lifted to ``z = [X, Y, psi, v_x, v_y, r]``.

    ``v_y`` and ``r`` are *relaxed* onto their kinematic values instead of
    being deleted, so the state dimension matches the dynamic field and the
    blend is well defined.
    """

    def f(z: np.ndarray, u: np.ndarray) -> np.ndarray:
        psi, v_x, v_y, r = z[2], z[3], z[4], z[5]
        a, delta = u
        beta = geometric_sideslip(delta, p)
        v_y_star = v_x * np.tan(beta)
        r_star = v_x / p.l_r * np.tan(beta)
        return np.array(
            [
                v_x * np.cos(psi) - v_y * np.sin(psi),
                v_x * np.sin(psi) + v_y * np.cos(psi),
                r,
                a,
                (v_y_star - v_y) / bp.tau,
                (r_star - r) / bp.tau,
            ]
        )

    return f


def dynamic_embedded_field(
    p: VehicleParams, mu: float | None = None
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """A six-state dynamic bicycle with linear tires, in the same coordinates.

    Deliberately linear-tire: this field is the *prediction model* handed to an
    optimizer, and the deck's rule is that the linear dynamic bicycle is a
    middle regime -- valid while slip stays small, not a substitute for tire
    saturation.  The plant in :mod:`avsim.models.dynamic_bicycle` saturates;
    the difference between them is model error, and it is measured, not
    assumed.
    """
    mu = p.mu if mu is None else mu

    def f(z: np.ndarray, u: np.ndarray) -> np.ndarray:
        psi, v_x, v_y, r = z[2], z[3], z[4], z[5]
        a, delta = u
        vx_reg = max(abs(v_x), SLIP_EPS)
        alpha_f = delta - np.arctan2(v_y + p.l_f * r, vx_reg)
        alpha_r = -np.arctan2(v_y - p.l_r * r, vx_reg)
        F_yf = np.clip(p.C_f * alpha_f, -mu * p.F_z_front_static, mu * p.F_z_front_static)
        F_yr = np.clip(p.C_r * alpha_r, -mu * p.F_z_rear_static, mu * p.F_z_rear_static)
        return np.array(
            [
                v_x * np.cos(psi) - v_y * np.sin(psi),
                v_x * np.sin(psi) + v_y * np.cos(psi),
                r,
                a + r * v_y,
                (F_yf * np.cos(delta) + F_yr) / p.m - r * v_x,
                (p.l_f * F_yf * np.cos(delta) - p.l_r * F_yr) / p.I_z,
            ]
        )

    return f


def embedded_field(
    p: VehicleParams, bp: BlendParams | None = None, mu: float | None = None
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """The blended field ``(1 - w) f_kin_emb + w f_dyn`` on the common state.

    Use this as the MPC prediction model when a manoeuvre spans standstill and
    speed -- the unprotected left turn of the running example does exactly
    that: braking, then low-speed turning, then acceleration.
    """
    bp = bp or BlendParams()
    f_kin = kinematic_embedded_field(p, bp)
    f_dyn = dynamic_embedded_field(p, mu)

    def f(z: np.ndarray, u: np.ndarray) -> np.ndarray:
        w = blend_weight(z[3], bp)
        return (1.0 - w) * f_kin(z, u) + w * f_dyn(z, u)

    return f
