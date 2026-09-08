"""A geometric controller worth comparing against.

Stanley cross-track control at the front axle, a constant-time-headway gap
policy for the vehicle ahead, curvature and signal speed limits, and
``diagnostics()`` so the run log says why it did what it did.

This is roughly the level of a competent hand-written baseline: it will hold a
lane, follow a leader and stop for red, and it will lose to the built-in MPC on
anything that needs planning around an obstacle.  That gap is the point of the
platform.
"""

import math


class StanleyController:
    name = "Stanley + headway"
    description = (
        "Front-axle Stanley steering with a steady-state feedforward, and a "
        "constant-time-headway longitudinal policy with curvature and signal caps."
    )

    def __init__(self, k_cross: float = 1.6, k_soft: float = 1.2,
                 headway: float = 1.6, a_y_max: float = 2.5):
        self.k_cross = float(k_cross)
        self.k_soft = float(k_soft)
        self.headway = float(headway)
        self.a_y_max = float(a_y_max)
        self._integral = 0.0
        self._last = {}

    def reset(self, context) -> None:
        self._integral = 0.0
        self._last = {}

    # --- lateral ------------------------------------------------------------
    def _steer(self, obs) -> float:
        v = obs.vehicle
        # Stanley uses the *front* axle, so the cross-track error is taken there.
        e_front = obs.e_y + v.wheelbase * math.sin(obs.e_psi)
        cross = math.atan2(-self.k_cross * e_front, self.k_soft + obs.ego.v)

        # Feedforward from the lecture's steady-state relation:
        #   delta_ss = (L + K_us V^2) * kappa
        kappa = obs.route_at(max(3.0, 0.5 * obs.ego.v)).curvature
        ff = (v.wheelbase + v.understeer_gradient * obs.ego.v ** 2) * kappa

        delta = ff - obs.e_psi + cross
        limit = v.delta_max * 0.95
        return max(min(delta, limit), -limit) / v.delta_max

    # --- longitudinal --------------------------------------------------------
    def _target_speed(self, obs) -> float:
        want = obs.speed_limit

        # Curvature cap: hold the lateral acceleration inside the budget.
        kappa = max(abs(obs.route_at(d).curvature) for d in (5.0, 15.0, 30.0))
        if kappa > 1e-4:
            want = min(want, math.sqrt(self.a_y_max / kappa))

        # Signal cap: a square-root ramp onto the stop line.
        sig = obs.signal
        if sig is not None and sig.colour in ("red", "yellow"):
            stop_in = sig.distance - 0.5 * obs.vehicle.length - 1.5
            if sig.colour == "yellow" and stop_in < 0.4 * obs.ego.v * 2.0:
                pass                      # too close to stop comfortably; go
            else:
                want = min(want, math.sqrt(max(2.0 * 2.2 * stop_in, 0.0)))

        # Gap policy: keep a constant time headway behind the leader.
        lead = obs.lead_object(half_width=1.7, max_range=70.0)
        if lead is not None:
            gap = lead.range - 0.5 * (obs.vehicle.length + lead.length)
            desired = 4.0 + self.headway * obs.ego.v
            want = min(want, max(lead.v + 0.6 * (gap - desired), 0.0))
            self._last["gap"] = gap
        else:
            self._last["gap"] = float("inf")

        self._last["target_speed"] = want
        return want

    def control(self, obs) -> dict:
        steer = self._steer(obs)

        err = self._target_speed(obs) - obs.ego.v
        self._integral = max(min(self._integral + err * obs.dt, 15.0), -15.0)
        a = 1.1 * err + 0.2 * self._integral
        if err < -2.0:
            self._integral = 0.0          # do not wind up while braking hard

        if a >= 0.0:
            throttle, brake = a / obs.vehicle.a_max, 0.0
        else:
            throttle, brake = 0.0, a / obs.vehicle.a_min

        self._last["e_y"] = obs.e_y
        self._last["n_objects"] = len(obs.objects)
        return {"steer": steer, "throttle": throttle, "brake": brake,
                "info": {"target_speed": self._last["target_speed"]}}

    def diagnostics(self) -> dict:
        return dict(self._last)


def create_controller(**options):
    return StanleyController(**options)
