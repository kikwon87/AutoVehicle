"""The shape a learned policy takes.

Features in, two normalized numbers out.  The weights here are hand-set so the
file runs without a training step -- replace ``W`` and ``b`` with something
learned and the rest of the file is unchanged.

Training data: every run writes a per-tick log (observation summary, command,
diagnostics).  ``RunResult.log`` is that dataset; the same seven features
computed below are what the built-in ``linear_policy`` controller uses, so a
policy trained on one is loadable by the other.
"""

import numpy as np

#: Feature names, in order.  Keep this next to the weights -- a policy whose
#: feature order drifts from its training set fails silently and drives badly.
FEATURES = (
    "e_y", "e_psi", "v_norm", "curvature_10m", "delta_norm", "gap_norm", "rel_v_norm",
)

#: Hand-set weights standing in for a trained matrix: row 0 -> steer,
#: row 1 -> acceleration (positive throttle, negative brake).
W_DEFAULT = np.array([
    [-0.45, -1.20, 0.00, 6.00, -0.15, 0.00, 0.00],
    [0.00, 0.00, -1.60, -2.50, 0.00, 0.90, 0.60],
])
B_DEFAULT = np.array([0.0, 0.55])


class PolicyController:
    name = "Linear policy (template)"
    description = "tanh(W x + b) on seven normalized features."

    def __init__(self, weights: str | None = None):
        if weights:
            z = np.load(weights)
            self.W, self.b = np.asarray(z["W"], float), np.asarray(z["b"], float)
        else:
            self.W, self.b = W_DEFAULT.copy(), B_DEFAULT.copy()
        if self.W.shape != (2, len(FEATURES)):
            raise ValueError(f"W must be (2, {len(FEATURES)}), got {self.W.shape}")

    def reset(self, context) -> None:
        # Seed anything stochastic from the run's seed: the platform is
        # bit-reproducible and a policy that is not makes comparison meaningless.
        self.rng = np.random.default_rng(context.seed)

    def features(self, obs) -> np.ndarray:
        lead = obs.lead_object(half_width=1.7, max_range=80.0)
        gap = lead.range if lead is not None else 80.0
        rel_v = (lead.v - obs.ego.v) if lead is not None else 0.0
        return np.array([
            np.clip(obs.e_y / 2.0, -2.0, 2.0),
            np.clip(obs.e_psi, -1.0, 1.0),
            obs.ego.v / max(obs.speed_limit, 1e-3),
            np.clip(obs.route_at(10.0).curvature * 20.0, -1.0, 1.0),
            obs.ego.delta / max(obs.vehicle.delta_max, 1e-6),
            np.clip(gap / 80.0, 0.0, 1.0),
            np.clip(rel_v / 10.0, -1.0, 1.0),
        ])

    def control(self, obs) -> dict:
        y = np.tanh(self.W @ self.features(obs) + self.b)
        steer, accel = float(y[0]), float(y[1])
        return {"steer": steer,
                "throttle": max(accel, 0.0),
                "brake": max(-accel, 0.0)}


def create_controller(weights: str | None = None, **_):
    return PolicyController(weights)
