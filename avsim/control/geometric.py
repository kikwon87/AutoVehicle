"""Pure pursuit and Stanley -- the two geometric path-tracking baselines.

Lecture 2/3, *Two Geometric Controllers That Ignore the Tires*:

    Both are excellent geometric baselines **precisely because their failure
    modes are easy to interpret**: they ignore tire-force limits, obstacles and
    actuator rate, and the sign conventions vary across implementations.
    Regularize low speed and fix the signs before comparing two papers'
    results.

Conventions used here, stated once so they can be checked:

* the vehicle reference point for pure pursuit is the **rear axle**, which is
  what makes ``kappa = 2 sin(alpha) / l_d`` exact for the kinematic bicycle;
* Stanley regulates the **front-axle** cross-track error, which is what makes
  its heading term appear without a gain;
* ``delta > 0`` steers left, and a positive cross-track error means the vehicle
  is to the **left** of the path (ISO ``y``).

Both controllers take an optional feedforward from the path curvature.  Adding
``delta_ff = (L + K_us V^2) kappa`` is what stops the feedback term from having
to generate a steady-state steering demand the geometry already knows about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.conventions import wrap_to_pi
from ..models.params import VehicleParams
from ..world.path import ReferencePath


@dataclass
class PurePursuit:
    """``delta = arctan(2 L sin(alpha) / l_d)`` with a speed-scheduled lookahead.

    ``l_d = clip(k_v * v + l_d0, l_d_min, l_d_max)``.

    Larger lookahead: smoother, slower to respond, more corner cutting.
    Smaller: faster response, more oscillation.  The scheduling exists because a
    fixed lookahead is either unstable at speed or sluggish at parking pace.
    """

    params: VehicleParams
    k_v: float = 0.6
    l_d0: float = 4.0
    l_d_min: float = 3.0
    l_d_max: float = 25.0
    #: Add only the **force-dependent** part of the feedforward,
    #: ``K_us V^2 kappa``.  The geometric part ``arctan(kappa L)`` is already
    #: what the pure-pursuit law produces, so adding the full ``delta_ss``
    #: would double-count it.  See :class:`Stanley` for the other case.
    use_feedforward: bool = False

    def lookahead(self, v: float) -> float:
        return float(np.clip(self.k_v * abs(v) + self.l_d0, self.l_d_min, self.l_d_max))

    def __call__(
        self,
        x: float,
        y: float,
        psi: float,
        v: float,
        path: ReferencePath,
        s_guess: float | None = None,
    ) -> tuple[float, dict]:
        """Return ``(delta, info)`` for a **rear-axle** reference point."""
        p = self.params
        l_d = self.lookahead(v)
        s0 = path.project(x, y, s_guess)
        s_target = min(s0 + l_d, path.length)
        target = path.position(s_target)

        dx, dy = target[0] - x, target[1] - y
        # Bearing to the target expressed in the body frame.
        alpha = wrap_to_pi(np.arctan2(dy, dx) - psi)
        dist = max(float(np.hypot(dx, dy)), 1e-3)

        kappa = 2.0 * np.sin(alpha) / dist
        delta = float(np.arctan(kappa * p.L))

        if self.use_feedforward:
            delta += p.steady_state_steer(path.curvature(s0), abs(v)) - np.arctan(
                path.curvature(s0) * p.L
            )

        delta = float(np.clip(delta, -p.actuator.delta_max, p.actuator.delta_max))
        return delta, {
            "s": s0,
            "s_target": s_target,
            "lookahead": l_d,
            "alpha": float(alpha),
            "cross_track": path.lateral_offset(x, y, s0),
            "heading_error": path.heading_error(psi, s0),
        }


@dataclass
class Stanley:
    """``delta = e_psi + arctan(k e_fa / (v + eps))`` on the **front axle**.

    Under the ideal assumptions the front-axle cross-track error obeys
    ``e_fa' = -k e_fa``: exponential decay for small errors, saturating for
    large ones.

    ``eps`` is not cosmetic.  Without it the gain term is unbounded at
    standstill and the controller commands full lock the instant the vehicle
    stops with any offset at all.
    """

    params: VehicleParams
    k: float = 2.0
    eps: float = 1.0
    k_soft_yaw: float = 0.0  #: optional yaw-rate damping term
    #: Add the **full** feedforward ``delta_ss = (L + K_us V^2) kappa``.
    #: Unlike pure pursuit, the Stanley law contains no curvature term at all,
    #: so on a curved path it holds a steady-state error unless the whole
    #: feedforward is supplied.  The two controllers therefore take *different*
    #: feedforward terms under the same flag name -- stated here because a
    #: silent mismatch of exactly this kind is what makes two implementations
    #: of "Stanley with feedforward" disagree.
    use_feedforward: bool = False

    def __call__(
        self,
        x: float,
        y: float,
        psi: float,
        v: float,
        path: ReferencePath,
        s_guess: float | None = None,
        yaw_rate: float = 0.0,
    ) -> tuple[float, dict]:
        """``(x, y, psi)`` is the **rear-axle** pose; the front axle is derived."""
        p = self.params
        xf = x + p.L * np.cos(psi)
        yf = y + p.L * np.sin(psi)

        s = path.project(xf, yf, s_guess)
        e_fa = path.lateral_offset(xf, yf, s)
        e_psi = path.heading_error(psi, s)

        # A positive cross-track error means the vehicle is left of the path,
        # so the correction must steer right: hence the minus sign.
        delta = -e_psi - np.arctan2(self.k * e_fa, abs(v) + self.eps)
        if self.k_soft_yaw:
            delta -= self.k_soft_yaw * (yaw_rate - abs(v) * path.curvature(s))
        if self.use_feedforward:
            delta += p.steady_state_steer(path.curvature(s), abs(v))

        delta = float(np.clip(delta, -p.actuator.delta_max, p.actuator.delta_max))
        return delta, {
            "s": s,
            "cross_track_front": float(e_fa),
            "heading_error": float(e_psi),
        }
