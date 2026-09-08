"""Front-wheel-drive powertrain, brakes and steering actuator.

The vehicle is FWD: **all** tractive force is applied at the front axle, which
is also the steered axle.  That single fact is what makes the combined-slip
budget of the *front* contact patches the binding constraint during a
brake-then-turn or a power-on exit, and it is why an FWD car understeers when
you add throttle mid-corner.

Braking is split front/rear by a fixed bias (``brake_bias_front``), as on a
production car with a proportioning valve.

Actuator dynamics are first order with a rate limit:

* steering: ``tau_steer delta' = delta_cmd - delta``, ``|delta'| <= rate_max``;
* drive / brake force: ``tau F' = F_cmd - F``.

A planner that ignores these produces plans the car cannot execute.  The MPC in
:mod:`avsim.control.mpc` includes the rate limit as an explicit constraint
``|delta_k - delta_{k-1}| <= delta_rate_max * h``, which is the discrete-time
statement of the same fact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.conventions import GRAVITY
from .params import VehicleParams


@dataclass
class ActuatorState:
    """Realized steering angle [rad] and axle forces [N]."""

    delta: float = 0.0
    F_drive: float = 0.0   #: >= 0, applied at the driven axle
    F_brake: float = 0.0   #: >= 0 magnitude, split by bias, opposes motion

    def to_array(self) -> np.ndarray:
        return np.array([self.delta, self.F_drive, self.F_brake])

    @staticmethod
    def from_array(v: np.ndarray) -> "ActuatorState":
        return ActuatorState(*(float(x) for x in v))


def split_acceleration_command(a_cmd: float, v_x: float, p: VehicleParams) -> tuple[float, float]:
    """Turn a commanded acceleration into ``(F_drive, F_brake)`` requests [N].

    Resistances (aerodynamic drag and rolling resistance) are compensated by
    the drive force, so ``a_cmd`` means "acceleration the body should see", not
    "force at the wheel".  That is the quantity a planner reasons about.

    The split is the usual one-pedal logic: positive demand goes to the
    powertrain, negative demand to the brakes.  Engine braking is not modelled
    separately; it is folded into the resistance term.
    """
    F_resist = resistance_force(v_x, p)
    F_req = p.m * a_cmd + F_resist
    if F_req >= 0.0:
        return float(min(F_req, p.max_drive_force)), 0.0
    return 0.0, float(min(-F_req, p.max_brake_force))


def resistance_force(v_x: float, p: VehicleParams) -> float:
    """Aerodynamic drag plus rolling resistance [N], always opposing motion."""
    drag = p.c_drag * v_x * abs(v_x)
    rr = p.c_rr * p.m * GRAVITY * np.tanh(v_x / 0.5)
    return float(drag + rr)


def axle_longitudinal_forces(
    F_drive: float, F_brake: float, p: VehicleParams
) -> tuple[float, float]:
    """Distribute drive and brake force to ``(F_x,front, F_x,rear)`` [N].

    Drive force goes entirely to the driven axle (front for FWD); brake force
    is split by ``brake_bias_front``.  Both are returned as *signed*
    longitudinal forces in the body frame.
    """
    if p.drive_layout == "fwd":
        Fx_f_drive, Fx_r_drive = F_drive, 0.0
    elif p.drive_layout == "rwd":
        Fx_f_drive, Fx_r_drive = 0.0, F_drive
    else:  # awd -- split by static load
        frac = p.F_z_front_static / (p.F_z_front_static + p.F_z_rear_static)
        Fx_f_drive, Fx_r_drive = frac * F_drive, (1.0 - frac) * F_drive

    Fx_f_brake = -p.brake_bias_front * F_brake
    Fx_r_brake = -(1.0 - p.brake_bias_front) * F_brake
    return float(Fx_f_drive + Fx_f_brake), float(Fx_r_drive + Fx_r_brake)


def actuator_derivative(
    state: np.ndarray, delta_cmd: float, a_cmd: float, v_x: float, p: VehicleParams
) -> np.ndarray:
    """Time derivative of ``[delta, F_drive, F_brake]`` under first-order lags.

    The steering rate is saturated *inside* the derivative, so the rate limit
    is enforced by the plant regardless of what the controller asked for --
    which is precisely how the real actuator behaves, and how you find out that
    your controller was relying on an impossible slew.
    """
    act = p.actuator
    delta, F_drive, F_brake = state

    rate = (delta_cmd - delta) / act.tau_steer
    rate = float(np.clip(rate, -act.delta_rate_max, act.delta_rate_max))

    F_drive_cmd, F_brake_cmd = split_acceleration_command(a_cmd, v_x, p)
    dF_drive = (F_drive_cmd - F_drive) / act.tau_drive
    dF_brake = (F_brake_cmd - F_brake) / act.tau_brake
    return np.array([rate, dF_drive, dF_brake])


def traction_limit(F_z_driven: float, mu: float) -> float:
    """Largest longitudinal force the driven axle can transmit [N].

    For an FWD car this is ``mu * F_z,front``, and front axle load *drops*
    under acceleration -- which is why FWD traction is worst exactly when it is
    most wanted.
    """
    return float(mu * max(F_z_driven, 0.0))
