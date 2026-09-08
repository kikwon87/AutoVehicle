"""Vehicle, tire, actuator and suspension parameters.

The defaults are the **reference vehicle** of Lecture 2/3:

===========  ==========  ==========================================
symbol       value       meaning
===========  ==========  ==========================================
``m``        1500 kg     total mass
``I_z``      2400 kgm^2  yaw inertia about the CG
``l_f``      1.2 m       CG to front axle
``l_r``      1.5 m       CG to rear axle
``L``        2.7 m       wheelbase (``l_f + l_r``)
``C_f``      80 kN/rad   front cornering stiffness (axle, both tires)
``C_r``      100 kN/rad  rear cornering stiffness (axle, both tires)
``mu``       scenario    tire-road friction coefficient
===========  ==========  ==========================================

Everything else (drivetrain, brakes, suspension, actuator lags) is required to
simulate a *front-wheel-drive* car with throttle, brake and a real steering
actuator, and is documented at the field.

A parameter set is a value object: construct it once per scenario, pass it
down, never mutate it.  ``__post_init__`` enforces the invariants that would
otherwise show up as a silently wrong model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from ..core.conventions import GRAVITY

DriveLayout = Literal["fwd", "rwd", "awd"]


@dataclass(frozen=True)
class TireParams:
    """Pacejka Magic Formula coefficients plus the linear-regime slope.

    ``F_y = D sin(C arctan(B alpha - E (B alpha - arctan(B alpha))))`` with
    ``D = mu * F_z``.  The linear cornering stiffness satisfies
    ``C_alpha = B * C * D`` at ``alpha = 0``, so ``B`` is derived from the
    axle stiffness rather than tuned independently -- otherwise the linear and
    the nonlinear tire disagree at the origin, and every comparison between the
    linear and nonlinear vehicle model is polluted by that mismatch.
    """

    C: float = 1.6        #: shape factor
    E: float = 0.97       #: curvature factor
    mu: float = 0.9       #: peak friction coefficient
    #: relaxation length [m]; ``0`` disables the first-order force lag
    relaxation_length: float = 0.3

    def B_from_stiffness(self, C_alpha: float, F_z: float) -> float:
        """Solve ``C_alpha = B C D`` for ``B`` given the static load."""
        D = max(self.mu * F_z, 1e-6)
        return C_alpha / (self.C * D)


@dataclass(frozen=True)
class ActuatorParams:
    """Steering and drivetrain actuator limits and first-order lags.

    A planner that ignores these produces plans the car cannot execute; an MPC
    that includes them as rate constraints (``|delta_k - delta_{k-1}| <=
    delta_rate_max * h``) produces plans it can.
    """

    delta_max: float = np.deg2rad(35.0)      #: road-wheel angle limit [rad]
    delta_rate_max: float = np.deg2rad(40.0)  #: road-wheel rate limit [rad/s]
    tau_steer: float = 0.10                   #: steering first-order lag [s]

    a_max: float = 3.0        #: max commanded longitudinal acceleration [m/s^2]
    a_min: float = -8.0       #: max commanded deceleration [m/s^2]
    jerk_max: float = 5.0     #: commanded jerk limit [m/s^3]
    tau_drive: float = 0.20   #: powertrain force first-order lag [s]
    tau_brake: float = 0.08   #: brake force first-order lag [s]


@dataclass(frozen=True)
class SuspensionParams:
    """Roll / pitch suspension used for **load transfer only**.

    The vehicle stays planar (X-Y): roll and pitch are internal degrees of
    freedom whose only job is to redistribute the normal loads ``F_z,i``, which
    in turn set each tire's friction budget.  This is the honest way to keep a
    suspension in a 2-D model -- it changes what the tires can do, not where
    the body is.

    ``I_x phi'' + c_phi phi' + (K_phi - m g h_roll) phi = m a_y h_roll``
    ``I_y theta'' + c_theta theta' + K_theta theta   = m a_x h_pitch``
    """

    h_cg: float = 0.55        #: CG height above ground [m]
    h_roll: float = 0.45      #: CG height above the roll axis [m]
    h_pitch: float = 0.50     #: CG height above the pitch axis [m]
    track: float = 1.6        #: track width [m]

    I_x: float = 550.0        #: roll inertia [kg m^2]
    I_y: float = 2200.0       #: pitch inertia [kg m^2]
    K_roll_f: float = 60000.0  #: front roll stiffness [Nm/rad]
    K_roll_r: float = 40000.0  #: rear roll stiffness [Nm/rad]
    C_roll: float = 6000.0     #: roll damping [Nms/rad]
    K_pitch: float = 130000.0  #: pitch stiffness [Nm/rad]
    C_pitch: float = 14000.0   #: pitch damping [Nms/rad]

    #: If False the suspension states are skipped and load transfer is
    #: computed quasi-statically from the instantaneous accelerations.
    dynamic: bool = True

    @property
    def K_roll(self) -> float:
        return self.K_roll_f + self.K_roll_r

    @property
    def roll_stiffness_front_fraction(self) -> float:
        """Share of the lateral load transfer taken by the front axle."""
        return self.K_roll_f / max(self.K_roll, 1e-9)


@dataclass(frozen=True)
class VehicleParams:
    """The full parameter set. Frozen; use :meth:`with_` to derive variants."""

    m: float = 1500.0
    I_z: float = 2400.0
    l_f: float = 1.2
    l_r: float = 1.5
    C_f: float = 80_000.0
    C_r: float = 100_000.0

    #: bounding box used for collision checking and rendering [m]
    length: float = 4.6
    width: float = 1.85
    #: distance from the rear axle to the rear bumper [m]
    rear_overhang: float = 0.9

    wheel_radius: float = 0.32
    drive_layout: DriveLayout = "fwd"
    #: fraction of the *braking* force applied at the front axle
    brake_bias_front: float = 0.65
    #: peak tractive force at the driven axle [N] (motor/engine limit)
    max_drive_force: float = 6000.0
    #: peak braking force, all axles combined [N]
    max_brake_force: float = 14000.0
    #: quadratic drag coefficient ``F = c_d v^2`` [N s^2/m^2]
    c_drag: float = 0.40
    #: rolling resistance coefficient
    c_rr: float = 0.013

    tire: TireParams = field(default_factory=TireParams)
    actuator: ActuatorParams = field(default_factory=ActuatorParams)
    suspension: SuspensionParams = field(default_factory=SuspensionParams)

    def __post_init__(self) -> None:
        if self.l_f <= 0 or self.l_r <= 0:
            raise ValueError("l_f and l_r must be positive")
        if self.m <= 0 or self.I_z <= 0:
            raise ValueError("mass and yaw inertia must be positive")
        if not 0.0 <= self.brake_bias_front <= 1.0:
            raise ValueError("brake_bias_front must lie in [0, 1]")
        if self.C_f <= 0 or self.C_r <= 0:
            raise ValueError(
                "cornering stiffnesses are positive in this deck's convention "
                "(F_y = C_alpha * alpha with alpha = wheel heading - velocity "
                "direction)"
            )

    # --- derived quantities --------------------------------------------------

    @property
    def L(self) -> float:
        """Wheelbase."""
        return self.l_f + self.l_r

    @property
    def mu(self) -> float:
        return self.tire.mu

    @property
    def F_z_front_static(self) -> float:
        """Static front-axle normal load ``m g l_r / L``."""
        return self.m * GRAVITY * self.l_r / self.L

    @property
    def F_z_rear_static(self) -> float:
        return self.m * GRAVITY * self.l_f / self.L

    @property
    def understeer_gradient(self) -> float:
        """``K_us = (m / L) (l_r / C_f - l_f / C_r)`` [s^2/m].

        Positive is understeer.  For the reference car this is
        ``3.75e-3 s^2/m``.
        """
        return (self.m / self.L) * (self.l_r / self.C_f - self.l_f / self.C_r)

    @property
    def characteristic_speed(self) -> float:
        """``v_ch = sqrt(L / K_us)`` [m/s]; 26.8 m/s for the reference car.

        Only defined for an understeering vehicle; an oversteering vehicle has
        a *critical* speed instead, returned as ``inf`` here to force the
        caller to handle that case explicitly.
        """
        K = self.understeer_gradient
        return float(np.sqrt(self.L / K)) if K > 0 else float("inf")

    @property
    def critical_speed(self) -> float:
        """Oversteer critical speed, ``inf`` when the vehicle understeers."""
        K = self.understeer_gradient
        return float(np.sqrt(-self.L / K)) if K < 0 else float("inf")

    def steady_state_steer(self, kappa: float, V: float) -> float:
        """``delta_ss = (L + K_us V^2) kappa`` -- the feedforward steering angle.

        At 30 m/s on a 200 m radius the force-dependent term is the *larger* of
        the two; geometry alone under-steers by more than half the required
        angle.
        """
        return (self.L + self.understeer_gradient * V**2) * kappa

    def max_lateral_accel(self, use_fraction: float = 1.0) -> float:
        """Friction-limited lateral acceleration ``mu g`` [m/s^2].

        The conservative planning rule of the lecture is
        ``|a_y| <= 0.5 g ~ 4.4 m/s^2`` at ``mu = 0.9``; pass
        ``use_fraction=0.5`` to get it.
        """
        return use_fraction * self.mu * GRAVITY

    def with_(self, **changes) -> "VehicleParams":
        """Return a copy with fields replaced (the frozen-dataclass idiom)."""
        return replace(self, **changes)


#: The lecture's reference vehicle, used by every default scenario.
REFERENCE_VEHICLE = VehicleParams()
