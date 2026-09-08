"""Speed profiles along a path: curvature, limits, stops and comfort.

A path without a speed profile is not a plan.  The intersection built in
:mod:`avsim.world.network` makes the point concretely: its right-turn connector
has ``R = 5.75 m``, so entering at the 13.9 m/s speed limit demands
``a_y = 33.6 m/s^2`` -- four times what the tires can deliver.  The geometry is
drivable, the geometry *at that speed* is not.

The profile is built by the standard three-pass construction:

1. **pointwise ceiling** -- the smallest of the speed limit, the curvature
   limit ``sqrt(a_y_max / |kappa|)`` and any explicit stop;
2. **backward pass** -- propagate stopping requirements upstream at the braking
   limit, so the vehicle is already slow when it arrives;
3. **forward pass** -- propagate the achievable acceleration downstream, so the
   profile never asks for more than ``a_max``.

Both passes use ``v^2`` arithmetic (``v_{k}^2 = v_{k+1}^2 + 2 a ds``), which is
exact for constant acceleration over a distance step and avoids the time
integration a naive implementation would need.

Combined slip is respected in the *ceiling*, not only in the passes: the
longitudinal acceleration available at a given lateral demand is reduced by the
friction ellipse, so a plan does not simultaneously request maximum braking and
maximum cornering from the same contact patch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.params import VehicleParams
from ..world.path import ReferencePath


@dataclass
class SpeedConstraint:
    """A speed ceiling at one arc length (``v_max = 0`` is a stop)."""

    s: float
    v_max: float
    label: str = ""


@dataclass
class VelocityProfile:
    """Sampled ``v(s)`` plus the arc-length grid it was built on."""

    s: np.ndarray
    v: np.ndarray
    a_lat_limit: float
    a_lon_max: float
    a_lon_min: float

    def speed_at(self, s: float) -> float:
        return float(np.interp(np.clip(s, self.s[0], self.s[-1]), self.s, self.v))

    def accel_at(self, s: float) -> float:
        """``a = v dv/ds`` -- the acceleration implied by the profile."""
        s = float(np.clip(s, self.s[0], self.s[-1]))
        dv_ds = float(np.interp(s, self.s, np.gradient(self.v, self.s)))
        return self.speed_at(s) * dv_ds

    def time_to(self, s_from: float, s_to: float) -> float:
        """Travel time between two arc lengths under this profile [s]."""
        mask = (self.s >= s_from) & (self.s <= s_to)
        if mask.sum() < 2:
            return 0.0
        ss, vv = self.s[mask], np.maximum(self.v[mask], 0.05)
        return float(np.trapezoid(1.0 / vv, ss))

    def sample_time_grid(self, s0: float, times: np.ndarray) -> np.ndarray:
        """Arc lengths reached at the given times, starting from ``s0``.

        Integrates ``ds/dt = v(s)`` with RK2 on the time grid.  This is what
        turns a spatial profile into the time-indexed reference an MPC needs.
        """
        times = np.asarray(times, dtype=float)
        out = np.empty_like(times)
        s = float(s0)
        prev_t = times[0]
        out[0] = s
        for i in range(1, len(times)):
            dt = times[i] - prev_t
            k1 = self.speed_at(s)
            k2 = self.speed_at(s + 0.5 * dt * k1)
            s = min(s + dt * k2, self.s[-1])
            out[i] = s
            prev_t = times[i]
        return out


def build_velocity_profile(
    path: ReferencePath,
    params: VehicleParams,
    v_limit: float,
    constraints: list[SpeedConstraint] | None = None,
    ds: float = 0.5,
    lateral_use: float = 0.5,
    longitudinal_use: float = 0.6,
    v_start: float | None = None,
    s_end: float | None = None,
    curvature_lookahead: float = 0.0,
) -> VelocityProfile:
    """Build ``v(s)`` for a path.

    ``lateral_use`` and ``longitudinal_use`` are the fractions of ``mu g`` the
    plan is allowed to spend.  The lecture's conservative planning rule is
    ``|a_y| <= 0.5 g ~ 4.4 m/s^2`` at ``mu = 0.9``, which is
    ``lateral_use = 0.5``.  Leaving headroom is not timidity: the plan is
    executed by a controller that has its own tracking error, and a plan at
    100% of ``mu`` has nothing left for it.

    ``curvature_lookahead`` widens the curvature used at each point to the
    maximum over a window ahead, which anticipates the step in ``kappa`` where a
    straight meets an arc.  Without it the profile only begins to slow *at* the
    joint and the backward pass has to brake harder than necessary.
    """
    s_end = path.length if s_end is None else min(s_end, path.length)
    n = max(int(s_end / ds) + 1, 3)
    s = np.linspace(0.0, s_end, n)

    a_lat_max = params.max_lateral_accel(lateral_use)
    a_lon_max = min(longitudinal_use * params.mu * 9.80665, params.actuator.a_max)
    a_lon_min = max(-longitudinal_use * params.mu * 9.80665, params.actuator.a_min)

    kappa = np.array([abs(path.curvature(si)) for si in s])
    if curvature_lookahead > 0:
        w = max(int(curvature_lookahead / ds), 1)
        kappa = np.array([kappa[i : i + w + 1].max() for i in range(len(kappa))])

    # 1. pointwise ceiling
    v = np.full(n, float(v_limit))
    curved = kappa > 1e-6
    v[curved] = np.minimum(v[curved], np.sqrt(a_lat_max / kappa[curved]))
    for c in constraints or []:
        v[s >= c.s] = np.minimum(v[s >= c.s], c.v_max) if c.v_max > 0 else v[s >= c.s]
        idx = int(np.clip(np.searchsorted(s, c.s), 0, n - 1))
        v[idx] = min(v[idx], c.v_max)
        if c.v_max <= 1e-6:
            v[idx:] = 0.0

    v = np.maximum(v, 0.0)

    # 2. backward pass: can we still stop / slow down in time?
    for i in range(n - 2, -1, -1):
        ds_i = s[i + 1] - s[i]
        a_avail = _longitudinal_budget(v[i + 1], kappa[i + 1], params, a_lon_min, a_lat_max)
        v[i] = min(v[i], float(np.sqrt(max(v[i + 1] ** 2 - 2.0 * a_avail * ds_i, 0.0))))

    # 3. forward pass: can we actually accelerate that fast?
    if v_start is not None:
        v[0] = min(v[0], float(v_start))
    for i in range(1, n):
        ds_i = s[i] - s[i - 1]
        a_avail = _longitudinal_budget(v[i - 1], kappa[i - 1], params, a_lon_max, a_lat_max)
        v[i] = min(v[i], float(np.sqrt(max(v[i - 1] ** 2 + 2.0 * a_avail * ds_i, 0.0))))

    return VelocityProfile(s=s, v=v, a_lat_limit=a_lat_max, a_lon_max=a_lon_max, a_lon_min=a_lon_min)


def _longitudinal_budget(
    v: float, kappa: float, params: VehicleParams, a_nominal: float, a_lat_max: float
) -> float:
    """Longitudinal acceleration still available at this lateral demand.

    ``(a_x / a_x_max)^2 + (a_y / a_y_max)^2 <= 1`` -- the friction ellipse
    applied to the *plan*.  Returns a magnitude with the sign of ``a_nominal``.
    """
    a_y = v * v * kappa
    frac = np.clip(1.0 - (a_y / max(a_lat_max, 1e-6)) ** 2, 0.0, 1.0)
    return float(np.sign(a_nominal) * abs(a_nominal) * np.sqrt(frac))


def stopping_distance(v: float, a_brake: float, reaction_time: float = 0.0) -> float:
    """Distance to stop from ``v`` at deceleration ``|a_brake|``."""
    return float(v * reaction_time + v * v / (2.0 * max(abs(a_brake), 1e-6)))
