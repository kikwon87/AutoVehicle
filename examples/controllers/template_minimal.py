"""The smallest controller that drives.

Proportional lane keeping on the Frenet error the platform already computes,
plus a PI on speed.  It imports nothing from ``avsim``: the platform duck-types
the controller and accepts a plain dict as the command.

Copy this file, rename the class, and start replacing the two ``control`` lines.
"""


class MyController:
    name = "Minimal P lane-keeper"
    description = "Proportional steering on (e_y, e_psi) with a speed PI."

    def __init__(self, target_speed: float = 10.0):
        self.target_speed = float(target_speed)
        self._integral = 0.0

    # Called once before each run.  Everything that carries state across ticks
    # is cleared here -- the same object is reused across a batch of runs.
    def reset(self, context) -> None:
        self._integral = 0.0

    def control(self, obs) -> dict:
        # --- lateral: steer towards the centreline ---------------------------
        # e_y > 0 means the vehicle sits left of the path, so the correction is
        # negative (steer right).  delta_max normalizes to the [-1, 1] command.
        delta = -0.35 * obs.e_y - 1.1 * obs.e_psi
        steer = delta / obs.vehicle.delta_max

        # --- longitudinal: hold a speed, and stop for red -------------------
        want = min(self.target_speed, obs.speed_limit)
        if obs.signal is not None and obs.signal.colour == "red":
            # A crude but honest stop: scale the target down inside 30 m.
            want = min(want, max(obs.signal.distance - 6.0, 0.0) * 0.4)

        err = want - obs.ego.v
        self._integral = max(min(self._integral + err * obs.dt, 20.0), -20.0)
        a = 0.9 * err + 0.15 * self._integral

        if a >= 0.0:
            throttle, brake = a / obs.vehicle.a_max, 0.0
        else:
            throttle, brake = 0.0, a / obs.vehicle.a_min

        return {"steer": steer, "throttle": throttle, "brake": brake}


def create_controller(target_speed: float = 10.0, **_):
    """The factory the loader looks for first; UI options arrive as kwargs."""
    return MyController(target_speed)
