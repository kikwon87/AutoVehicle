"""Controllers the platform ships with.

Three, chosen to span the space a user is asked to choose from:

:class:`MPCController`
    The full stack of this package -- prediction, behaviour, Frenet lattice and
    the constrained NMPC.  This is the reference: a new algorithm is worth
    keeping if it beats this on the KPI you care about.

:class:`PurePursuitController`
    Geometry and a PI on speed.  No obstacle avoidance beyond stopping for a
    lead vehicle, no constraints.  Present because a baseline whose failures are
    obvious is worth more than one whose failures are not.

:class:`LinearPolicyController`
    A feature vector in, an action out -- the shape a learned policy takes.  Its
    weights are hand-set rather than trained, and it says so: it exists to
    exercise the *interface* an ML controller plugs into, and to be replaced by
    ``LinearPolicyController.from_npz(...)`` once weights exist.

All three obey the same contract as a user plug-in, and the platform does not
know which is which.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..autonomy.stack import AutonomyConfig, AutonomyStack
from ..control.mpc import MPCConfig, VehicleMPC
from ..core.conventions import wrap_to_pi
from ..models.params import VehicleParams
from ..world.network import RoadNetwork
from ..world.path import ReferencePath
from .controller_api import (
    ControlCommand,
    Controller,
    Observation,
    ScenarioContext,
)


class MPCController(Controller):
    """The package's own stack, behind the plug-in interface.

    The stack does its own planning and needs the route and the road network,
    so it is constructed by the platform rather than by a plug-in file.  It
    reads its perception from ``obs.extras`` -- the tracks the platform already
    computed -- so that it is scored against exactly the observations an
    external controller receives.
    """

    name = "MPC (built-in stack)"
    description = (
        "Multi-modal prediction, behaviour FSM, Frenet lattice and a constrained "
        "iLQR NMPC on the kinematic bicycle with an understeer correction."
    )

    def __init__(
        self,
        params: VehicleParams,
        network: RoadNetwork,
        route: ReferencePath,
        config: AutonomyConfig | None = None,
        route_signals=None,
        lights=None,
        crossing_conflict: bool = True,
        mpc_config: MPCConfig | None = None,
    ):
        cfg = config or AutonomyConfig()
        # No wall-clock budget on the platform.
        #
        # On a vehicle the deadline is real and the budget belongs there. Here
        # the job is to *compare* algorithms, and a solver cut short by machine
        # load makes the same configuration produce different results on
        # different runs -- which is the one thing a comparison cannot survive.
        # The work is bounded by iterations, and the compute time is measured
        # and scored (``real_time_factor``) rather than enforced.
        mpc_cfg = mpc_config or MPCConfig(
            a_y_max=params.max_lateral_accel(cfg.lateral_use) * cfg.lateral_margin,
            time_budget=None,
        )
        self.stack = AutonomyStack(
            params, network, route, cfg,
            mpc=VehicleMPC(params, mpc_cfg),
            lights=lights, route_signals=route_signals,
            crossing_conflict=crossing_conflict,
        )

    def reset(self, context: ScenarioContext) -> None:
        self.stack.reset()

    def control(self, obs: Observation) -> ControlCommand:
        ego_actor = obs.extras["ego_actor"]
        x_rear = np.array([obs.ego.x, obs.ego.y, obs.ego.psi, obs.ego.v])
        a, delta = self.stack.step(
            obs.t, ego_actor, x_rear, obs.ego.delta,
            obs.extras.get("actors", []), tracks=obs.extras.get("tracks"),
        )
        return ControlCommand.from_physical(a, delta, obs.vehicle)

    def diagnostics(self) -> dict:
        if not self.stack.telemetry:
            return {}
        r = self.stack.telemetry[-1]
        return {
            "behavior": r.behavior,
            "reason": r.reason,
            "mpc_status": r.mpc_status,
            "mpc_time": r.mpc_time,
            "solver_iterations": r.mpc_iterations,
            "fallback": bool(r.used_fallback_control or r.used_fallback_plan),
        }


@dataclass
class PurePursuitController(Controller):
    """Pure pursuit for steering, a PI for speed, and stop for the lead vehicle.

    The whole thing is about forty lines, which is the point: it is what a
    plug-in author's first working controller looks like, and it is scored on
    the same KPIs as the MPC.
    """

    name: str = "Pure pursuit + PI"
    description: str = "Geometric steering with a speed-scheduled lookahead; PI on speed."
    k_lookahead: float = 0.7
    lookahead_min: float = 4.0
    lookahead_max: float = 22.0
    kp_speed: float = 0.9
    ki_speed: float = 0.12
    time_gap: float = 1.6
    min_gap: float = 5.0
    _integral: float = field(default=0.0, init=False)

    def reset(self, context: ScenarioContext) -> None:
        self._integral = 0.0

    def control(self, obs: Observation) -> ControlCommand:
        ego, veh = obs.ego, obs.vehicle

        # --- steering: aim at a point on the route ---------------------------
        l_d = float(np.clip(self.k_lookahead * ego.v + self.lookahead_min,
                            self.lookahead_min, self.lookahead_max))
        target = obs.route_at(l_d)
        dx, dy = target.x - ego.x, target.y - ego.y
        alpha = wrap_to_pi(np.arctan2(dy, dx) - ego.psi)
        dist = max(float(np.hypot(dx, dy)), 1e-3)
        delta = float(np.arctan(2.0 * veh.wheelbase * np.sin(alpha) / dist))

        # --- speed target: limit, curvature, lead vehicle, red light ---------
        v_target = min(obs.speed_limit, target.speed_limit)
        kappa = max(abs(obs.route_at(d).curvature) for d in (5.0, 15.0, 30.0))
        if kappa > 1e-4:
            v_target = min(v_target, float(np.sqrt(0.4 * veh.mu * 9.81 / kappa)))

        lead = obs.lead_object()
        if lead is not None:
            gap = lead.range - 0.5 * (veh.length + lead.length)
            desired = self.min_gap + self.time_gap * ego.v
            v_target = min(v_target, max(lead.v + 0.5 * (gap - desired), 0.0))

        if obs.signal is not None and obs.signal.colour != "green" and obs.signal.distance > 0.0:
            # Brake to a stop at the line: v^2 = 2 a d, at a comfortable a.
            v_target = min(v_target, float(np.sqrt(max(2.0 * 2.5 * (obs.signal.distance - 3.0), 0.0))))

        err = v_target - ego.v
        self._integral = float(np.clip(self._integral + err * obs.dt, -8.0, 8.0))
        a_cmd = self.kp_speed * err + self.ki_speed * self._integral
        if err < -2.0:
            self._integral = 0.0  # do not wind up through a hard deceleration

        return ControlCommand.from_physical(a_cmd, delta, veh, v_target=v_target)


@dataclass
class LinearPolicyController(Controller):
    """A features-in, action-out policy -- the shape a learned controller takes.

    ``action = tanh(W @ features + b)`` with ``action = (steer, accel)``, both
    normalized.  The default ``W`` is **hand-set, not trained**, and is here so
    that the ML path through the platform is exercised rather than merely
    described.  Replace it with :meth:`from_npz` once real weights exist.

    Features (all dimensionless, roughly unit-scaled):

    ==  =========================================================
    0   lateral offset from the route ``e_y / 3.5``
    1   heading error ``e_psi / 0.5``
    2   speed error ``(v - v_limit) / 10``
    3   route curvature 15 m ahead ``kappa * 50``
    4   inverse lead-vehicle gap ``10 / (10 + gap)``
    5   signal urgency ``1`` if a non-green light is within 40 m
    6   bias
    ==  =========================================================
    """

    name: str = "Linear policy (ML-shaped)"
    description: str = (
        "features -> tanh(Wx + b) -> (steer, accel). Weights are hand-set, not "
        "trained; use from_npz() to load real ones."
    )
    W: np.ndarray | None = None
    b: np.ndarray | None = None
    trained: bool = False

    N_FEATURES = 7

    def __post_init__(self) -> None:
        if self.W is None:
            # steer  <- -e_y, -e_psi, curvature feedforward
            # accel  <- -speed error, -curvature, -lead proximity, -signal
            self.W = np.array([
                [-0.55, -0.85, 0.0, 0.30, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.75, -0.25, -0.90, -1.10, 0.22],
            ])
        if self.b is None:
            self.b = np.zeros(2)
        self.W = np.asarray(self.W, dtype=float).reshape(2, self.N_FEATURES)
        self.b = np.asarray(self.b, dtype=float).reshape(2)

    @classmethod
    def from_npz(cls, path: str | Path, **kwargs) -> "LinearPolicyController":
        """Load ``W`` and ``b`` from an ``.npz`` produced by training."""
        data = np.load(Path(path).expanduser())
        return cls(W=data["W"], b=data.get("b", np.zeros(2)), trained=True, **kwargs)

    def features(self, obs: Observation) -> np.ndarray:
        veh = obs.vehicle
        lead = obs.lead_object()
        gap = (lead.range - 0.5 * (veh.length + lead.length)) if lead is not None else 200.0
        signal_urgent = (
            1.0 if (obs.signal is not None and obs.signal.colour != "green"
                    and 0.0 < obs.signal.distance < 40.0) else 0.0
        )
        return np.array([
            obs.e_y / 3.5,
            obs.e_psi / 0.5,
            (obs.ego.v - obs.speed_limit) / 10.0,
            obs.route_at(15.0).curvature * 50.0,
            10.0 / (10.0 + max(gap, 0.0)),
            signal_urgent,
            1.0,
        ])

    def control(self, obs: Observation) -> ControlCommand:
        action = np.tanh(self.W @ self.features(obs) + self.b)
        steer, accel = float(action[0]), float(action[1])
        return ControlCommand(
            steer=steer,
            throttle=max(accel, 0.0),
            brake=max(-accel, 0.0),
            info={"trained": self.trained},
        ).clipped()


#: Name -> factory. The platform lists these in its controller dropdown; a
#: plug-in file adds itself to the same list at load time.
BUILTIN_CONTROLLERS = {
    "mpc": MPCController,
    "pure_pursuit": PurePursuitController,
    "linear_policy": LinearPolicyController,
}
