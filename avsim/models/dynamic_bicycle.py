"""Nonlinear dynamic single-track (bicycle) model -- the simulation plant.

Lecture 2/3, *Dynamic Bicycle: Newton-Euler Plus Tire Forces*::

    m (v_x' - r v_y) = F_xf cos d - F_yf sin d + F_xr
    m (v_y' + r v_x) = F_yf cos d + F_xf sin d + F_yr
    I_z r'           = l_f (F_yf cos d + F_xf sin d) - l_r F_yr
    X' = v_x cos psi - v_y sin psi,  Y' = v_x sin psi + v_y cos psi,  psi' = r

    The terms ``-r v_y`` and ``+r v_x`` are transport terms from expressing
    velocity in a rotating body frame.  They are not optional.

    ``F_x`` and ``F_y`` close the model through the tire law, which is where the
    hard modelling decision lives -- not in the Newton-Euler mechanics.

This module supplies the closure for a **front-wheel-drive** car:

* drive force at the front (steered) axle only;
* brake force split front/rear by a fixed bias;
* Magic Formula lateral forces with load-dependent peak;
* combined slip: the longitudinal request is honoured first and the lateral
  force is derated onto the remaining friction ellipse;
* normal loads from the roll/pitch suspension, so a fast steer reversal
  transfers load with a lag;
* first-order steering and force actuators with a steering rate limit.

The full plant state is 13-dimensional:

===========  =================================================
indices      contents
===========  =================================================
``0:6``      ``[X, Y, psi, v_x, v_y, r]`` -- rigid body at the CG
``6:10``     ``[roll, roll', pitch, pitch']`` -- suspension
``10:13``    ``[delta, F_drive, F_brake]`` -- actuators
===========  =================================================

Input is ``u = [a_cmd, delta_cmd]``: the acceleration the *body* should see and
the commanded road-wheel angle.  Everything a controller emits in this package
is that pair, whatever model it used to produce it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..core.conventions import DYN_NX, GRAVITY, wrap_to_pi
from . import powertrain, suspension as susp
from .params import VehicleParams
from .tire import LinearTire, PacejkaTire, SLIP_EPS, apply_combined_slip, friction_ellipse_usage

#: Speed below which the tires stop producing lateral force [m/s].  A modelling
#: parameter, not a physical constant: it sets where the slip-angle description
#: is abandoned, and it must be validated like any other.
LATERAL_FORCE_SPEED_EPS = 0.7

IDX_BODY = slice(0, 6)
IDX_SUSP = slice(6, 10)
IDX_ACT = slice(10, 13)
PLANT_NX = 13


@dataclass
class PlantDiagnostics:
    """Per-step internal quantities a KPI or a plot may want.

    Returned alongside the derivative rather than recomputed, because
    recomputing them from the state is how a diagnostic silently stops
    describing the simulation it claims to describe.
    """

    alpha_f: float = 0.0
    alpha_r: float = 0.0
    F_yf: float = 0.0
    F_yr: float = 0.0
    F_xf: float = 0.0
    F_xr: float = 0.0
    F_z: np.ndarray = field(default_factory=lambda: np.zeros(4))
    a_x: float = 0.0
    a_y: float = 0.0
    usage_front: float = 0.0
    usage_rear: float = 0.0
    beta: float = 0.0


class DynamicBicycle:
    """Nonlinear FWD dynamic bicycle with tires, suspension and actuators."""

    nx = PLANT_NX
    nu = 2

    def __init__(
        self,
        params: VehicleParams | None = None,
        alpha_peak: float = np.deg2rad(8.0),
        tire_model: str = "pacejka",
    ):
        """
        ``tire_model`` selects the closure of the Newton-Euler mechanics:

        ``"pacejka"``
            Saturating Magic Formula.  This is the plant.
        ``"linear"``
            ``F_y = C_alpha alpha`` clipped at ``mu F_z``.  Use it to verify
            that the simulator reproduces the *linear* predictions of the
            lecture -- the understeer gradient, the characteristic speed, the
            linear single-track eigenvalues -- before asking it about the limit.

        Note on the reference parameters.  With ``C_f = 80 kN/rad`` on
        ``mu F_z,f = 7.4 kN`` and ``C_r = 100 kN/rad`` on ``mu F_z,r = 5.9 kN``,
        the rear axle's linear reach (3.4 deg) is shorter than the front's
        (5.3 deg).  The car is therefore *linearly understeering*
        (``K_us = +3.75e-3``) but saturates at the rear first, so it drifts
        towards limit oversteer above roughly 5 m/s^2 of lateral acceleration.
        That is a real property of this parameter set, not an artefact, and it
        is why :meth:`limit_balance` is reported rather than assumed.
        """
        self.p = params or VehicleParams()
        self.tire_model = tire_model
        if tire_model == "pacejka":
            self.tire_f = PacejkaTire.for_axle(self.p.C_f, self.p.F_z_front_static, self.p.tire, alpha_peak)
            self.tire_r = PacejkaTire.for_axle(self.p.C_r, self.p.F_z_rear_static, self.p.tire, alpha_peak)
        elif tire_model == "linear":
            self.tire_f = LinearTire(self.p.C_f, self.p.mu)
            self.tire_r = LinearTire(self.p.C_r, self.p.mu)
        else:
            raise ValueError(f"unknown tire_model {tire_model!r}; use 'pacejka' or 'linear'")
        self.last_diagnostics = PlantDiagnostics()

    def limit_balance(self, mu: float | None = None) -> dict:
        """Which axle saturates first, in slip-angle terms.

        Returns the peak slip angle of each axle and the ``linear reach``
        ``mu F_z / C_alpha``.  Front reach shorter than rear means the front
        gives up first: terminal understeer.  The reverse means the rear does:
        terminal oversteer, and a plan that assumed ``K_us > 0`` all the way to
        the limit was wrong about this car.
        """
        p = self.p
        mu = p.mu if mu is None else mu
        reach_f = mu * p.F_z_front_static / p.C_f
        reach_r = mu * p.F_z_rear_static / p.C_r
        return {
            "linear_reach_front_deg": float(np.rad2deg(reach_f)),
            "linear_reach_rear_deg": float(np.rad2deg(reach_r)),
            "terminal_behaviour": "understeer" if reach_f < reach_r else "oversteer",
            "understeer_gradient": p.understeer_gradient,
            "linear_behaviour": "understeer" if p.understeer_gradient > 0 else "oversteer",
        }

    # --- state helpers -------------------------------------------------------

    def initial_state(self, x: float = 0.0, y: float = 0.0, psi: float = 0.0, v: float = 0.0) -> np.ndarray:
        """A settled state at speed ``v``: suspension at rest, actuators at zero."""
        z = np.zeros(PLANT_NX)
        z[0], z[1], z[2], z[3] = x, y, psi, v
        return z

    @staticmethod
    def pose(z: np.ndarray) -> np.ndarray:
        return np.array([z[0], z[1], z[2]])

    @staticmethod
    def speed(z: np.ndarray) -> float:
        """Magnitude of the CG velocity [m/s]."""
        return float(np.hypot(z[3], z[4]))

    @staticmethod
    def sideslip(z: np.ndarray) -> float:
        """Dynamic sideslip ``beta = arctan(v_y / v_x)`` [rad]."""
        return float(np.arctan2(z[4], max(abs(z[3]), 1e-3)))

    # --- the field -----------------------------------------------------------

    def derivative(self, z: np.ndarray, u: np.ndarray, mu: float | None = None) -> np.ndarray:
        """Continuous-time derivative of the 13-state plant.

        ``mu`` overrides the tire friction coefficient for this evaluation,
        which is how a low-friction patch is injected by the world without
        rebuilding the model.
        """
        p = self.p
        mu = p.mu if mu is None else mu

        psi, v_x, v_y, r = z[2], z[3], z[4], z[5]
        roll, pitch = z[6], z[8]
        delta, F_drive, F_brake = z[10], z[11], z[12]

        # --- normal loads (suspension state -> friction budget) --------------
        if p.suspension.dynamic:
            F_z = susp.load_transfer_from_attitude(roll, pitch, p)
        else:
            d = self.last_diagnostics
            F_z = susp.quasi_static_load_transfer(d.a_x, d.a_y, p)
        F_zf, F_zr = susp.axle_loads(F_z)

        # --- longitudinal forces ---------------------------------------------
        F_xf_req, F_xr_req = powertrain.axle_longitudinal_forces(F_drive, F_brake, p)
        F_resist = powertrain.resistance_force(v_x, p)
        # Resistance acts on the body, not on one axle; remove it after the
        # tire ellipse so it cannot consume grip it does not use.

        # --- slip angles and lateral forces ----------------------------------
        vx_reg = max(abs(v_x), SLIP_EPS)
        alpha_f = float(delta - np.arctan2(v_y + p.l_f * r, vx_reg))
        alpha_r = float(-np.arctan2(v_y - p.l_r * r, vx_reg))

        # A slip angle is a *ratio of velocities*: at zero speed there is no
        # slip and therefore no lateral force, however large the regularized
        # angle computes to be.  Without this factor a stopped vehicle keeps
        # generating a yaw moment from its own residual v_y and r, and rotates
        # on the spot -- which shows up as tens of degrees of heading error in
        # every scenario that ends with the vehicle standing still.
        speed_factor = float(np.tanh(abs(v_x) / LATERAL_FORCE_SPEED_EPS))
        F_yf_demand = self.tire_f.lateral_force(alpha_f, F_zf, mu) * speed_factor
        F_yr_demand = self.tire_r.lateral_force(alpha_r, F_zr, mu) * speed_factor

        # Combined slip, per axle: the longitudinal request is committed and
        # the lateral force absorbs the shortfall.
        F_xf, F_yf = apply_combined_slip(F_xf_req, F_yf_demand, F_zf, mu)
        F_xr, F_yr = apply_combined_slip(F_xr_req, F_yr_demand, F_zr, mu)

        # --- Newton-Euler in the rotating body frame -------------------------
        cd, sd = np.cos(delta), np.sin(delta)
        Fx_body = F_xf * cd - F_yf * sd + F_xr - F_resist * np.sign(v_x if abs(v_x) > 1e-3 else 1.0)
        Fy_body = F_yf * cd + F_xf * sd + F_yr
        Mz = p.l_f * (F_yf * cd + F_xf * sd) - p.l_r * F_yr

        dv_x = Fx_body / p.m + r * v_y
        dv_y = Fy_body / p.m - r * v_x
        dr = Mz / p.I_z

        # Body-frame accelerations felt at the CG (what the suspension sees).
        a_x = dv_x - r * v_y
        a_y = dv_y + r * v_x

        # --- assemble ---------------------------------------------------------
        dz = np.zeros(PLANT_NX)
        dz[0] = v_x * np.cos(psi) - v_y * np.sin(psi)
        dz[1] = v_x * np.sin(psi) + v_y * np.cos(psi)
        dz[2] = r
        dz[3] = dv_x
        dz[4] = dv_y
        dz[5] = dr
        dz[IDX_SUSP] = susp.suspension_derivative(z[IDX_SUSP], a_y, a_x, p)
        dz[IDX_ACT] = powertrain.actuator_derivative(z[IDX_ACT], u[1], u[0], v_x, p)

        self.last_diagnostics = PlantDiagnostics(
            alpha_f=alpha_f,
            alpha_r=alpha_r,
            F_yf=F_yf,
            F_yr=F_yr,
            F_xf=F_xf,
            F_xr=F_xr,
            F_z=F_z,
            a_x=a_x,
            a_y=a_y,
            usage_front=friction_ellipse_usage(F_xf, F_yf, F_zf, mu),
            usage_rear=friction_ellipse_usage(F_xr, F_yr, F_zr, mu),
            beta=self.sideslip(z),
        )
        return dz

    def field(self, mu: float | None = None) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Bind ``mu`` and return a plain ``f(z, u)`` for the integrators."""

        def f(z: np.ndarray, u: np.ndarray) -> np.ndarray:
            return self.derivative(z, u, mu)

        return f

    # --- post-step housekeeping ----------------------------------------------

    def sanitize(self, z: np.ndarray) -> np.ndarray:
        """Clamp the physically-bounded states after an integration step.

        Wraps the heading, enforces the steering limit and keeps the forces
        non-negative.  An integrator has no idea these bounds exist; leaving
        them unenforced is how a plant quietly reports 400 deg of steering.
        """
        z = np.asarray(z, dtype=float).copy()
        z[2] = wrap_to_pi(z[2])
        z[10] = float(np.clip(z[10], -self.p.actuator.delta_max, self.p.actuator.delta_max))
        z[11] = max(z[11], 0.0)
        z[12] = max(z[12], 0.0)
        # A braking car must stop, not reverse: below walking pace the brake
        # can only bring v_x to zero.
        if z[3] < 0.0 and z[12] > 0.0:
            z[3] = 0.0
        # A stopped car does not rotate.  The lateral force already vanishes at
        # zero speed, but nothing then removes whatever yaw rate was left over,
        # so it is removed here explicitly.
        if abs(z[3]) < 0.05:
            z[4] = 0.0
            z[5] = 0.0
        return z


def linear_single_track(V: float, p: VehicleParams) -> tuple[np.ndarray, np.ndarray]:
    """``A(V), B`` of the linear single-track model, ``x_lat = [v_y, r]``.

    From *Linear Single-Track State-Space Model*::

        v_y' = -(C_f + C_r)/(m V) v_y + ((l_r C_r - l_f C_f)/(m V) - V) r + C_f/m d
        r'   =  (l_r C_r - l_f C_f)/(I_z V) v_y - (l_f^2 C_f + l_r^2 C_r)/(I_z V) r
                 + l_f C_f / I_z d

    Every ``1/V`` term enters through the slip-angle approximation; the separate
    ``-V r`` term comes from the rotating-frame force balance.  The two
    mechanisms are different, and the model is ill-conditioned as ``V -> 0``
    for the first reason, not the second.
    """
    if V <= 0:
        raise ValueError(
            "the linear single-track model is undefined at V = 0; this is a "
            "model failure, not a fast vehicle -- use a kinematic or blended "
            "model near standstill"
        )
    m, I_z, l_f, l_r, C_f, C_r = p.m, p.I_z, p.l_f, p.l_r, p.C_f, p.C_r
    A = np.array(
        [
            [-(C_f + C_r) / (m * V), (l_r * C_r - l_f * C_f) / (m * V) - V],
            [(l_r * C_r - l_f * C_f) / (I_z * V), -(l_f**2 * C_f + l_r**2 * C_r) / (I_z * V)],
        ]
    )
    B = np.array([[C_f / m], [l_f * C_f / I_z]])
    return A, B
