"""Roll / pitch suspension, used to redistribute the tire normal loads.

The vehicle in this package moves in the plane.  A suspension is nevertheless
worth modelling because of the last bullet of the *Combined Slip* slide:

    Load transfer: each tire has a different ``F_z``, and therefore a different
    budget.

So roll and pitch are internal degrees of freedom here.  They never move the
body in X-Y; they set the four normal loads, which set the four friction
budgets, which is what actually limits the manoeuvre.

Two fidelities are provided:

``quasi_static_load_transfer``
    Algebraic: the load transfer that a rigid axle would see instantly.  Cheap,
    no extra states, and correct in steady state.

:class:`SuspensionState` + :func:`suspension_derivative`
    Two second-order oscillators driven by the body accelerations::

        I_x phi''   + C_roll  phi'   + (K_roll - m g h_roll) phi = m a_y h_roll
        I_y theta'' + C_pitch theta' + K_pitch          theta    = m a_x h_pitch

    The ``- m g h_roll`` term is the destabilizing gravity moment of a CG above
    the roll axis; it is what makes a soft, tall vehicle roll further than the
    static stiffness alone suggests.  Keeping it explicit means a badly chosen
    ``K_roll`` shows up as an unstable roll mode instead of quietly producing
    optimistic loads.

Both routines return loads in the order ``[FL, FR, RL, RR]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.conventions import GRAVITY
from .params import VehicleParams

FL, FR, RL, RR = 0, 1, 2, 3


@dataclass
class SuspensionState:
    """Roll angle / rate and pitch angle / rate [rad, rad/s]."""

    roll: float = 0.0
    roll_rate: float = 0.0
    pitch: float = 0.0
    pitch_rate: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.roll, self.roll_rate, self.pitch, self.pitch_rate])

    @staticmethod
    def from_array(v: np.ndarray) -> "SuspensionState":
        return SuspensionState(*(float(x) for x in v))


def suspension_derivative(state: np.ndarray, a_y: float, a_x: float, p: VehicleParams) -> np.ndarray:
    """Time derivative of ``[roll, roll_rate, pitch, pitch_rate]``."""
    s = p.suspension
    phi, dphi, th, dth = state
    # Effective roll stiffness net of the gravity moment about the roll axis.
    K_eff = s.K_roll - p.m * GRAVITY * s.h_roll
    ddphi = (p.m * a_y * s.h_roll - s.C_roll * dphi - K_eff * phi) / s.I_x
    ddth = (p.m * a_x * s.h_pitch - s.C_pitch * dth - s.K_pitch * th) / s.I_y
    return np.array([dphi, ddphi, dth, ddth])


def quasi_static_load_transfer(a_x: float, a_y: float, p: VehicleParams) -> np.ndarray:
    """Normal loads ``[FL, FR, RL, RR]`` [N] from instantaneous accelerations.

    * Longitudinal: ``dF_x = m a_x h_cg / L`` moves load rearwards on
      acceleration and forwards on braking.
    * Lateral: ``dF_y = m a_y h_cg / track``, split between the axles by the
      **roll stiffness distribution** -- the stiffer axle takes the larger
      share, which is the entire reason a roll bar changes the balance of a
      car without changing its mass.

    Loads are floored at zero: a lifted wheel carries no load and produces no
    force, and returning a negative ``F_z`` would silently produce a negative
    friction budget downstream.
    """
    s = p.suspension
    Fz_f = p.F_z_front_static
    Fz_r = p.F_z_rear_static

    dF_long = p.m * a_x * s.h_cg / p.L
    Fz_f_axle = Fz_f - dF_long
    Fz_r_axle = Fz_r + dF_long

    dF_lat_total = p.m * a_y * s.h_cg / s.track
    kf = s.roll_stiffness_front_fraction
    dF_lat_f = kf * dF_lat_total
    dF_lat_r = (1.0 - kf) * dF_lat_total

    # a_y > 0 (leftward acceleration) loads the RIGHT wheels: y points left.
    loads = np.array(
        [
            0.5 * Fz_f_axle - dF_lat_f,  # FL
            0.5 * Fz_f_axle + dF_lat_f,  # FR
            0.5 * Fz_r_axle - dF_lat_r,  # RL
            0.5 * Fz_r_axle + dF_lat_r,  # RR
        ]
    )
    return np.maximum(loads, 0.0)


def load_transfer_from_attitude(roll: float, pitch: float, p: VehicleParams) -> np.ndarray:
    """Normal loads ``[FL, FR, RL, RR]`` [N] from the suspension deflections.

    Equivalent to :func:`quasi_static_load_transfer` in steady state, but it
    carries the suspension's own dynamics: during a fast steer reversal the
    loads lag the accelerations, which is exactly when a limit manoeuvre is
    lost.
    """
    s = p.suspension
    # Restoring moments carried by each axle's springs.
    M_roll_f = s.K_roll_f * roll
    M_roll_r = s.K_roll_r * roll
    M_pitch = s.K_pitch * pitch

    dF_lat_f = M_roll_f / s.track
    dF_lat_r = M_roll_r / s.track
    dF_long = M_pitch / p.L

    Fz_f_axle = p.F_z_front_static - dF_long
    Fz_r_axle = p.F_z_rear_static + dF_long

    loads = np.array(
        [
            0.5 * Fz_f_axle - dF_lat_f,
            0.5 * Fz_f_axle + dF_lat_f,
            0.5 * Fz_r_axle - dF_lat_r,
            0.5 * Fz_r_axle + dF_lat_r,
        ]
    )
    return np.maximum(loads, 0.0)


def axle_loads(loads: np.ndarray) -> tuple[float, float]:
    """Collapse four wheel loads to ``(F_z,front_axle, F_z,rear_axle)``.

    The single-track models need axle loads; the four-wheel split still matters
    because it is where the load transfer was computed.
    """
    return float(loads[FL] + loads[FR]), float(loads[RL] + loads[RR])


def rollover_index(a_y: float, p: VehicleParams) -> float:
    """Static rollover propensity, ``|a_y| / (g * SSF)`` with ``SSF = t / 2h``.

    ``>= 1`` means the inside wheels lift.  Reported as a KPI so that a plan
    which is friction-feasible but geometrically unsafe is still flagged.
    """
    s = p.suspension
    ssf = s.track / (2.0 * s.h_cg)
    return float(abs(a_y) / (GRAVITY * ssf))
