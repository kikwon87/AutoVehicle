"""The 2-D world: road network, signals, other traffic and the ego plant.

The world owns *ground truth* and nothing else.  It advances the ego plant
under a control input, advances the sync source, evaluates the signals, and
answers questions about collisions and road departure.  It contains no
perception, no planning and no control: those modules read from the world only
through the interfaces they are supposed to use, which is the only way to be
sure the planner is not quietly reading a state no sensor could provide.

Friction is spatial.  :class:`FrictionPatch` marks a region with a different
``mu``; the plant is evaluated with the local value.  A wet or icy patch is
then a property of the road rather than a global switch, and a scenario can ask
what happens when the friction drops **inside** a corner the planner already
committed to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..core.geometry import obb_penetration, polygon_distance, rect_corners, time_to_collision
from ..core.integrators import get_method
from ..models.dynamic_bicycle import DynamicBicycle
from ..models.params import VehicleParams
from .actors import ActorState, SyncSource
from .network import SIGNAL_GROUP, RoadNetwork
from .traffic_light import SignalState, TrafficLightController


@dataclass(frozen=True)
class FrictionPatch:
    """A rectangular region with a different tire-road friction coefficient."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    mu: float
    label: str = "low-mu"

    def contains(self, x: float, y: float) -> bool:
        return bool(self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max)


@dataclass
class WorldSnapshot:
    """Everything a logger, a renderer or a KPI needs from one tick."""

    t: float
    ego: np.ndarray                      #: 13-state plant vector
    ego_state: ActorState
    actors: list[ActorState]
    u: np.ndarray                        #: the applied ``[a_cmd, delta_cmd]``
    mu: float
    diagnostics: dict = field(default_factory=dict)
    signals: dict[str, str] = field(default_factory=dict)


class World:
    """Ground-truth 2-D simulation container."""

    def __init__(
        self,
        network: RoadNetwork,
        params: VehicleParams | None = None,
        sync_source: SyncSource | None = None,
        lights: TrafficLightController | None = None,
        friction_patches: Sequence[FrictionPatch] = (),
        dt: float = 0.02,
        integrator: str = "rk4",
        plant_substeps: int = 1,
        tire_model: str = "pacejka",
    ):
        self.network = network
        self.params = params or VehicleParams()
        self.plant = DynamicBicycle(self.params, tire_model=tire_model)
        self.sync_source = sync_source
        self.lights = lights
        self.friction_patches = list(friction_patches)
        self.dt = float(dt)
        self.integrator = get_method(integrator)
        self.plant_substeps = int(plant_substeps)

        self.t = 0.0
        self.ego = self.plant.initial_state()
        self.history: list[WorldSnapshot] = []

    # --- setup ---------------------------------------------------------------

    def reset(self, ego_state: np.ndarray | None = None, seed: int = 0) -> None:
        self.t = 0.0
        self.ego = self.plant.initial_state() if ego_state is None else np.asarray(ego_state, float).copy()
        self.history = []
        if self.sync_source is not None:
            self.sync_source.reset(seed)

    def place_ego(self, path, s: float = 0.0, v: float = 0.0, e_y: float = 0.0) -> None:
        """Put the ego on a reference path at arc length ``s``.

        The path describes the **rear axle** (it is a lane centerline and the
        planning model is the rear-axle bicycle), while the plant state is at
        the CG, so the lever arm is applied here rather than left to the caller.
        """
        p = path.to_cartesian(s, e_y)
        psi = path.heading(s)
        self.ego = self.plant.initial_state(
            x=float(p[0] + self.params.l_r * np.cos(psi)),
            y=float(p[1] + self.params.l_r * np.sin(psi)),
            psi=float(psi),
            v=float(v),
        )

    # --- queries -------------------------------------------------------------

    def mu_at(self, x: float, y: float) -> float:
        for patch in self.friction_patches:
            if patch.contains(x, y):
                return patch.mu
        return self.params.mu

    def ego_actor(self) -> ActorState:
        """The ego as an :class:`ActorState`, for uniform collision handling."""
        z = self.ego
        return ActorState(
            id="ego",
            x=float(z[0]),
            y=float(z[1]),
            psi=float(z[2]),
            v=float(np.hypot(z[3], z[4])),
            length=self.params.length,
            width=self.params.width,
            kind="ego",
        )

    def ego_rear_axle(self) -> np.ndarray:
        """``[X_r, Y_r, psi, v]`` -- the state the planning model is written in."""
        z = self.ego
        return np.array(
            [
                z[0] - self.params.l_r * np.cos(z[2]),
                z[1] - self.params.l_r * np.sin(z[2]),
                z[2],
                float(np.hypot(z[3], z[4])),
            ]
        )

    def signal_states(self) -> dict[str, str]:
        if self.lights is None:
            return {}
        return {g: self.lights.state(g, self.t).value for g in ("NS", "EW")}

    def signal_for_approach(self, approach: str) -> SignalState:
        if self.lights is None:
            return SignalState.GREEN
        return self.lights.state(SIGNAL_GROUP[approach], self.t)

    # --- collision and road departure ---------------------------------------

    def collisions(self, actors: Sequence[ActorState] | None = None) -> list[tuple[str, float]]:
        """Actors currently overlapping the ego, with penetration depth [m]."""
        actors = self.current_actors() if actors is None else actors
        ego = self.ego_actor().corners()
        out = []
        for a in actors:
            depth = obb_penetration(ego, a.corners())
            if depth > 0.0:
                out.append((a.id, float(depth)))
        return out

    def min_clearance(self, actors: Sequence[ActorState] | None = None) -> float:
        """Smallest box-to-box distance to any actor [m]; ``inf`` when alone."""
        actors = self.current_actors() if actors is None else actors
        if not actors:
            return float("inf")
        ego = self.ego_actor().corners()
        return float(min(polygon_distance(ego, a.corners()) for a in actors))

    def min_ttc(self, actors: Sequence[ActorState] | None = None) -> float:
        actors = self.current_actors() if actors is None else actors
        if not actors:
            return float("inf")
        e = self.ego_actor()
        radius = 0.5 * (self.params.length + self.params.width) * 0.5
        return float(
            min(
                time_to_collision(
                    e.position, e.velocity, a.position, a.velocity,
                    radius + 0.5 * (a.length + a.width) * 0.5,
                )
                for a in actors
            )
        )

    def lateral_error_to(self, path, s_guess: float | None = None) -> tuple[float, float]:
        """``(s, e_y)`` of the ego **rear axle** relative to a reference path."""
        x = self.ego_rear_axle()
        s = path.project(x[0], x[1], s_guess)
        return s, path.lateral_offset(x[0], x[1], s)

    def current_actors(self) -> list[ActorState]:
        return [] if self.sync_source is None else self.sync_source.actors()

    # --- stepping ------------------------------------------------------------

    def step(self, u: np.ndarray) -> WorldSnapshot:
        """Advance the world by ``dt`` under the control input ``u = [a, delta]``.

        The plant may be integrated with ``plant_substeps`` internal steps while
        the control is held: that refines the *integration* without changing the
        zero-order hold, so the plant can be more accurate than the controller's
        prediction model without the two disagreeing about what was commanded.
        """
        u = np.asarray(u, dtype=float).reshape(2)
        mu = self.mu_at(float(self.ego[0]), float(self.ego[1]))
        f = self.plant.field(mu)

        h = self.dt / self.plant_substeps
        z = self.ego
        for _ in range(self.plant_substeps):
            z = self.plant.sanitize(self.integrator.step(f, z, u, h))
        self.ego = z

        actors = [] if self.sync_source is None else self.sync_source.step(self.t, self.dt, self.ego_actor())
        self.t += self.dt

        d = self.plant.last_diagnostics
        snap = WorldSnapshot(
            t=self.t,
            ego=self.ego.copy(),
            ego_state=self.ego_actor(),
            actors=list(actors),
            u=u.copy(),
            mu=mu,
            diagnostics={
                "a_x": d.a_x,
                "a_y": d.a_y,
                "alpha_f": d.alpha_f,
                "alpha_r": d.alpha_r,
                "usage_front": d.usage_front,
                "usage_rear": d.usage_rear,
                "beta": d.beta,
                "F_z": d.F_z.tolist(),
                "roll": float(self.ego[6]),
                "pitch": float(self.ego[8]),
                "delta_actual": float(self.ego[10]),
            },
            signals=self.signal_states(),
        )
        self.history.append(snap)
        return snap
