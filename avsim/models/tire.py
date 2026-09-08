"""Tire force models: slip definitions, linear slope, Magic Formula, friction ellipse.

Lecture 2/3, *Tire Forces*:

    Cornering stiffness ``C_alpha`` is the **local slope** at the operating
    point, not a global property.  Linear tire models are local models.

Sign convention (see :mod:`avsim.core.conventions`):
``alpha = wheel heading - velocity direction``, ``F_y = C_alpha alpha`` with
``C_alpha > 0``.

Every function here divides by a *regularized* longitudinal speed.  Both the
slip angle and the slip ratio are ill-conditioned as ``v_x -> 0``; the
regularization ``eps`` is a modelling parameter, and the low-speed behaviour of
the whole vehicle model depends on it.
"""

from __future__ import annotations

import numpy as np

from .params import TireParams

#: Speed below which slip angles stop being a meaningful description [m/s].
SLIP_EPS = 0.5


def slip_angle_front(v_x: float, v_y: float, r: float, delta: float, l_f: float, eps: float = SLIP_EPS) -> float:
    """``alpha_f = delta - arctan((v_y + l_f r) / v_x)``."""
    vx = max(abs(v_x), eps) * (1.0 if v_x >= 0 else -1.0)
    return float(delta - np.arctan2(v_y + l_f * r, abs(vx)))


def slip_angle_rear(v_x: float, v_y: float, r: float, l_r: float, eps: float = SLIP_EPS) -> float:
    """``alpha_r = -arctan((v_y - l_r r) / v_x)``."""
    vx = max(abs(v_x), eps)
    return float(-np.arctan2(v_y - l_r * r, vx))


def slip_ratio(omega_w: float, v_x: float, R_w: float, eps: float = SLIP_EPS) -> float:
    """``kappa_x = (R_w omega_w - v_x) / max(|v_x|, |R_w omega_w|, eps)``.

    Positive when the driven wheel spins faster than ground speed, negative
    under braking.  Planners care because wheel force is not commanded torque
    and because hard braking consumes the same friction budget the turn needs.
    """
    den = max(abs(v_x), abs(R_w * omega_w), eps)
    return float((R_w * omega_w - v_x) / den)


class LinearTire:
    """``F_y = C_alpha alpha``, clipped at the friction limit.

    The clip is not cosmetic: without it the linear model happily returns
    30 kN from a 1500 kg car, and any KPI built on "did we exceed the friction
    budget" becomes meaningless.
    """

    def __init__(self, C_alpha: float, mu: float = 0.9):
        if C_alpha <= 0:
            raise ValueError("C_alpha > 0 in this convention")
        self.C_alpha = float(C_alpha)
        self.mu = float(mu)

    def lateral_force(self, alpha: float, F_z: float, mu: float | None = None) -> float:
        mu = self.mu if mu is None else mu
        F_max = mu * max(F_z, 0.0)
        return float(np.clip(self.C_alpha * alpha, -F_max, F_max))

    def cornering_stiffness(self, alpha: float = 0.0, F_z: float = 0.0) -> float:
        return self.C_alpha


class PacejkaTire:
    """Simplified Magic Formula lateral tire model.

    ``F_y = D sin(C arctan(B a - E (B a - arctan(B a))))``,  ``D = mu F_z``

    The three shape parameters are *not* independent of the linear model:

    * ``B C D`` is the slope at ``alpha = 0``, i.e. the cornering stiffness;
    * the peak occurs where ``C arctan(y) = pi/2``, which fixes ``E`` once
      ``B``, ``C`` and the desired peak slip angle are chosen.

    Constructing this class directly gives you the raw Magic Formula.  Use
    :meth:`from_stiffness_and_peak` to obtain a tire that reproduces a **given**
    cornering stiffness at the origin *and* peaks at a **given** slip angle --
    that is what makes a linear-vs-nonlinear model comparison meaningful, since
    any difference is then genuine saturation rather than a parameterization
    artefact.

    Note on the reference vehicle.  With ``C_f = 80 kN/rad`` and
    ``mu F_z,f = 7.4 kN`` the linear reach ``D / C_alpha`` is only 5.3 deg, so
    the peak cannot sit far beyond that without a strongly negative ``E``.
    That tension is a property of the given parameters, not of this code, and
    :meth:`from_stiffness_and_peak` reports the ``E`` it had to use.
    """

    def __init__(self, B: float, C: float, mu: float, E: float, F_z_ref: float, relaxation_length: float = 0.3):
        self.B = float(B)
        self.C = float(C)
        self.mu = float(mu)
        self.E = float(E)
        self.F_z_ref = float(F_z_ref)
        self.relaxation_length = float(relaxation_length)

    # --- construction --------------------------------------------------------

    @classmethod
    def from_stiffness_and_peak(
        cls,
        C_alpha: float,
        F_z: float,
        mu: float = 0.9,
        alpha_peak: float = np.deg2rad(8.0),
        C: float = 1.9,
        relaxation_length: float = 0.3,
    ) -> "PacejkaTire":
        """Solve ``(B, E)`` so the curve has slope ``C_alpha`` and peaks at ``alpha_peak``.

        ``B = C_alpha / (C D)`` comes straight from the slope condition.  With
        ``z = B alpha_peak`` and ``y* = tan(pi / 2C)`` the peak condition gives

            ``E = (z - y*) / (z - arctan z)``

        ``E`` is clamped to ``E <= 1`` (the Magic Formula's validity bound); if
        the clamp binds, the requested peak is not reachable with this ``C_alpha``
        and ``mu``, and the actual peak is later than requested.
        """
        D = mu * max(F_z, 1e-6)
        B = C_alpha / (C * D)
        z = B * float(alpha_peak)
        y_star = np.tan(np.pi / (2.0 * C))
        denom = z - np.arctan(z)
        E = (z - y_star) / denom if abs(denom) > 1e-12 else 0.0
        E = float(min(E, 1.0))
        return cls(B=B, C=C, mu=mu, E=E, F_z_ref=F_z, relaxation_length=relaxation_length)

    @classmethod
    def for_axle(cls, C_alpha: float, F_z: float, params: TireParams, alpha_peak: float = np.deg2rad(8.0)) -> "PacejkaTire":
        """Convenience constructor from a :class:`TireParams` bundle."""
        return cls.from_stiffness_and_peak(
            C_alpha=C_alpha,
            F_z=F_z,
            mu=params.mu,
            alpha_peak=alpha_peak,
            C=params.C,
            relaxation_length=params.relaxation_length,
        )

    # --- evaluation ----------------------------------------------------------

    def lateral_force(self, alpha: float, F_z: float, mu: float | None = None) -> float:
        """Lateral force [N] at slip angle ``alpha`` under normal load ``F_z``.

        Load sensitivity: ``D`` scales with ``F_z`` and ``B`` inversely, so the
        slope ``B C D`` stays constant with load while the peak force grows.
        Real tires lose stiffness at high load; that refinement is deliberately
        omitted and noted here rather than hidden.
        """
        mu = self.mu if mu is None else mu
        F_z = max(F_z, 0.0)
        D = mu * F_z
        if D <= 1e-9:
            return 0.0
        B = self.B * (self.mu * self.F_z_ref) / D
        Ba = B * alpha
        return float(D * np.sin(self.C * np.arctan(Ba - self.E * (Ba - np.arctan(Ba)))))

    def cornering_stiffness(self, alpha: float = 0.0, F_z: float | None = None, mu: float | None = None) -> float:
        """Local slope ``dF_y/dalpha`` at ``alpha`` -- the honest ``C_alpha``."""
        F_z = self.F_z_ref if F_z is None else F_z
        h = 1e-5
        return float(
            (self.lateral_force(alpha + h, F_z, mu) - self.lateral_force(alpha - h, F_z, mu)) / (2 * h)
        )

    def peak_slip_angle(self, F_z: float | None = None, mu: float | None = None) -> float:
        """Slip angle at which the lateral force peaks [rad], found by search.

        The lecture's rule -- "keep the tire slip angle below the peak" -- needs
        a number; this computes it for the actual parameters in use instead of
        assuming the textbook 6-8 deg.
        """
        F_z = self.F_z_ref if F_z is None else F_z
        grid = np.linspace(0.0, np.deg2rad(30.0), 1201)
        forces = np.array([self.lateral_force(a, F_z, mu) for a in grid])
        return float(grid[int(np.argmax(forces))])


def friction_ellipse_usage(F_x: float, F_y: float, F_z: float, mu: float) -> float:
    """``sqrt((F_x / mu F_z)^2 + (F_y / mu F_z)^2)`` -- the combined-slip budget.

    ``<= 1`` means the contact patch can deliver the requested force.  Above 1
    the request is refused by physics, whatever the controller asked for.  This
    is reported as a KPI (``max_friction_usage``) for every scenario.
    """
    cap = mu * max(F_z, 1e-6)
    return float(np.hypot(F_x / cap, F_y / cap))


def derate_lateral_for_longitudinal(F_x: float, F_z: float, mu: float) -> float:
    """Lateral force still available after spending ``F_x`` on the same patch.

    ``F_y,avail = sqrt((mu F_z)^2 - F_x^2)``, floored at zero.  Hard braking
    shrinks the cornering capacity; this is the term that makes a
    brake-then-turn plan infeasible when it was computed axle by axle.
    """
    cap = mu * max(F_z, 0.0)
    return float(np.sqrt(max(cap**2 - min(abs(F_x), cap) ** 2, 0.0)))


def apply_combined_slip(F_x: float, F_y_demand: float, F_z: float, mu: float) -> tuple[float, float]:
    """Scale a (F_x, F_y) request back onto the friction ellipse.

    Longitudinal force is treated as the committed request (the driver pressed
    the pedal); the lateral force absorbs the shortfall.  Returns the achieved
    pair.
    """
    cap = mu * max(F_z, 0.0)
    F_x = float(np.clip(F_x, -cap, cap))
    F_y_max = derate_lateral_for_longitudinal(F_x, F_z, mu)
    return F_x, float(np.clip(F_y_demand, -F_y_max, F_y_max))
