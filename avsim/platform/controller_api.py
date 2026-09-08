"""The controller plug-in contract.

Everything a control algorithm receives, everything it may command, and how a
``.py`` file supplying one is loaded.  The prose version of this file, meant to
be handed to somebody writing their own controller, is ``docs/controller_api.md``.

The contract in one sentence
----------------------------
A controller is a callable object that receives an :class:`Observation` once per
control tick and returns a :class:`ControlCommand`; it may keep state between
ticks and must survive :meth:`Controller.reset`.

What is *not* in the observation is as much a part of the contract as what is.
There is no ground truth: other vehicles arrive as :class:`DetectedObject`,
already degraded by the sensor and smoothed by the tracker, and they may be
missing, late or duplicated.  A controller that needs the true state of the
world cannot be evaluated on this platform, which is the point.

Units and signs follow :mod:`avsim.core.conventions` throughout: metres,
seconds, radians; ``x`` forward, ``y`` left, yaw counter-clockwise, steering
positive to the left.
"""

from __future__ import annotations

import importlib.util
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

CONTRACT_VERSION = "1.0"


# --- what the controller sees --------------------------------------------------

@dataclass(frozen=True)
class EgoState:
    """The ego vehicle's own state, as onboard sensing would report it.

    ``x``, ``y``, ``psi`` are the **rear axle** pose, because that is the
    reference point of the kinematic bicycle every planning model here uses.
    ``v`` is the speed at that point; ``v_x``/``v_y``/``yaw_rate`` are the
    body-frame velocities at the CG, which is what an IMU measures.
    """

    x: float
    y: float
    psi: float
    v: float
    v_x: float
    v_y: float
    yaw_rate: float
    #: realized road-wheel angle [rad] -- the actuator's state, not the command
    delta: float
    a_x: float
    a_y: float
    #: dynamic sideslip ``atan2(v_y, v_x)`` [rad]
    beta: float


@dataclass(frozen=True)
class DetectedObject:
    """One tracked object. Positions and velocities are **estimates**.

    ``id`` is a *track* id, not a ground-truth id: it changes when the tracker
    loses and re-acquires an object, and two objects passing close by can swap
    theirs.  A controller that assumes id stability will be surprised, and the
    KPI report will show why.
    """

    id: int
    x: float
    y: float
    psi: float
    v: float
    v_x: float
    v_y: float
    length: float
    width: float
    #: range and bearing from the ego, for convenience [m, rad]
    range: float
    bearing: float
    #: consecutive updates this track has existed
    age: int
    #: trace of the position covariance [m^2] -- how much to trust the above
    position_variance: float


@dataclass(frozen=True)
class RoutePoint:
    """A sample of the reference path ahead of the vehicle."""

    s: float
    x: float
    y: float
    heading: float
    curvature: float
    speed_limit: float


@dataclass(frozen=True)
class SignalInfo:
    """The next signalized stop line on the route, if any."""

    group: str
    #: ``"green"``, ``"yellow"`` or ``"red"``
    colour: str
    #: arc-length distance from the ego to the stop line [m]; negative once past
    distance: float
    #: seconds until this signal next changes colour
    time_to_change: float


@dataclass(frozen=True)
class VehicleInfo:
    """The parameters of the vehicle being driven.

    Supplied so a controller can be written once and remain correct when the
    platform's parameter sliders change the car underneath it.
    """

    wheelbase: float
    l_f: float
    l_r: float
    mass: float
    yaw_inertia: float
    length: float
    width: float
    delta_max: float
    delta_rate_max: float
    a_max: float
    a_min: float
    #: tire-road friction the *platform* is configured with; the road may differ
    mu: float
    understeer_gradient: float


@dataclass(frozen=True)
class Observation:
    """Everything the controller is given, once per control tick."""

    t: float
    dt: float
    ego: EgoState
    objects: tuple[DetectedObject, ...]
    #: reference path samples from the ego forward, ``route_horizon`` metres
    route: tuple[RoutePoint, ...]
    #: Frenet state on the route: arc length, lateral offset, heading error
    s: float
    e_y: float
    e_psi: float
    #: lateral corridor ``(min, max)`` in metres, positive to the left
    lane_bounds: tuple[float, float]
    speed_limit: float
    goal_s: float
    signal: SignalInfo | None
    vehicle: VehicleInfo
    #: free-form extras; nothing in the contract depends on them
    extras: dict = field(default_factory=dict)

    # --- conveniences, so a simple controller stays simple --------------------

    def route_at(self, distance: float) -> RoutePoint:
        """The route sample ``distance`` metres ahead of the ego."""
        target = self.s + distance
        best = self.route[0]
        for p in self.route:
            if p.s >= target:
                return p
            best = p
        return best

    def nearest_object(self) -> DetectedObject | None:
        return min(self.objects, key=lambda o: o.range) if self.objects else None

    def lead_object(self, half_width: float = 2.0, max_range: float = 80.0) -> DetectedObject | None:
        """The nearest object ahead and roughly in the ego's path.

        A crude in-lane test in the ego's body frame -- deliberately crude, so
        that a controller wanting something better writes it itself.
        """
        best = None
        for o in self.objects:
            if o.range > max_range:
                continue
            dx = (o.x - self.ego.x) * np.cos(self.ego.psi) + (o.y - self.ego.y) * np.sin(self.ego.psi)
            dy = -(o.x - self.ego.x) * np.sin(self.ego.psi) + (o.y - self.ego.y) * np.cos(self.ego.psi)
            if dx <= 0.0 or abs(dy) > half_width:
                continue
            if best is None or dx < best[0]:
                best = (dx, o)
        return best[1] if best else None


# --- what the controller commands ------------------------------------------------

@dataclass
class ControlCommand:
    """The command sent to the vehicle.

    ``steer``    normalized road-wheel angle in ``[-1, 1]``; ``+1`` is full
                 left lock, i.e. ``delta = steer * delta_max``.
    ``throttle`` accelerator level in ``[0, 1]``; ``a = throttle * a_max``.
    ``brake``    brake level in ``[0, 1]``; ``a = brake * a_min`` (``a_min`` is
                 negative).

    Both pedals at once is not an error and is not silently averaged: **brake
    wins**, because that is what a real brake-override does and because a
    controller that leaks throttle during a stop should be visible in the log,
    not smoothed over.

    ``info`` is free-form and is written into the run log next to the command,
    which is the cheapest way to debug a controller after the fact.
    """

    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    info: dict = field(default_factory=dict)

    def clipped(self) -> "ControlCommand":
        return ControlCommand(
            steer=float(np.clip(self.steer, -1.0, 1.0)),
            throttle=float(np.clip(self.throttle, 0.0, 1.0)),
            brake=float(np.clip(self.brake, 0.0, 1.0)),
            info=self.info,
        )

    def to_physical(self, vehicle: VehicleInfo) -> tuple[float, float]:
        """``(a_cmd [m/s^2], delta_cmd [rad])`` -- what the plant is given."""
        c = self.clipped()
        delta = c.steer * vehicle.delta_max
        a = c.brake * vehicle.a_min if c.brake > 0.0 else c.throttle * vehicle.a_max
        return float(a), float(delta)

    @classmethod
    def from_physical(cls, a: float, delta: float, vehicle: VehicleInfo, **info) -> "ControlCommand":
        """Build a command from physical units, for controllers that think in them."""
        steer = delta / max(vehicle.delta_max, 1e-6)
        if a >= 0.0:
            throttle, brake = a / max(vehicle.a_max, 1e-6), 0.0
        else:
            throttle, brake = 0.0, a / min(vehicle.a_min, -1e-6)
        return cls(steer=steer, throttle=throttle, brake=brake, info=dict(info)).clipped()


# --- the controller itself ------------------------------------------------------

@dataclass
class ScenarioContext:
    """Handed to :meth:`Controller.reset` once, before a run begins."""

    scenario: str
    vehicle: VehicleInfo
    control_dt: float
    goal_s: float
    route_length: float
    speed_limit: float
    seed: int = 0
    options: dict = field(default_factory=dict)


class Controller(ABC):
    """Base class for a control algorithm.

    Subclass it, or supply any object with the same two methods -- the platform
    duck-types, so a plain class needs no import from this package to work.
    """

    #: shown in the UI and written into the report
    name: str = "unnamed"
    #: free text describing the algorithm, shown next to the results
    description: str = ""

    def reset(self, context: ScenarioContext) -> None:
        """Called once before each run. Clear all per-run state here."""

    @abstractmethod
    def control(self, obs: Observation) -> ControlCommand:
        """Return the command for this tick."""

    def diagnostics(self) -> dict:
        """Per-tick extras for the log; called right after :meth:`control`."""
        return {}


# --- loading a user-supplied file --------------------------------------------------

class ControllerLoadError(RuntimeError):
    """Raised when a plug-in file cannot be turned into a controller."""


def load_controller_module(path: str | Path):
    """Import a ``.py`` file as a module.

    **This executes the file.** The platform exists to run control algorithms
    people write, so that is the intended behaviour and not a lapse -- but it
    means a plug-in is exactly as trustworthy as its author, and the file should
    be read before it is loaded, the same as any other code you run.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.suffix != ".py":
        raise ControllerLoadError(f"{path} is not a .py file")

    name = f"avsim_plugin_{path.stem}_{abs(hash(str(path))) % 10**8}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ControllerLoadError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - the author's error, reported as theirs
        raise ControllerLoadError(f"{path.name} raised while importing: {exc}") from exc
    return module


def instantiate(module, **kwargs) -> Controller:
    """Find the controller a plug-in module provides.

    Looked for in order:

    1. ``create_controller(**kwargs)`` -- a factory function;
    2. a class named ``Controller``;
    3. the single subclass of :class:`Controller` defined in the module;
    4. an already-built object named ``controller``.

    The order matters: a factory is tried first so a plug-in can take options
    from the platform, and the single-subclass rule is deliberately *single* --
    two candidates is an ambiguity the loader will not resolve on the author's
    behalf.
    """
    factory = getattr(module, "create_controller", None)
    if callable(factory):
        obj = factory(**kwargs)
        return _validate(obj, module)

    cls = getattr(module, "Controller", None)
    if isinstance(cls, type) and cls is not Controller:
        return _validate(cls(**kwargs) if kwargs else cls(), module)

    subclasses = [
        v for v in vars(module).values()
        if isinstance(v, type) and issubclass(v, Controller) and v is not Controller
    ]
    if len(subclasses) == 1:
        return _validate(subclasses[0](**kwargs) if kwargs else subclasses[0](), module)
    if len(subclasses) > 1:
        raise ControllerLoadError(
            f"{module.__name__} defines {len(subclasses)} controllers "
            f"({', '.join(c.__name__ for c in subclasses)}); add a "
            "create_controller() factory to say which one to use"
        )

    obj = getattr(module, "controller", None)
    if obj is not None:
        return _validate(obj, module)

    raise ControllerLoadError(
        f"{module.__name__} provides no controller: define create_controller(), "
        "a class named Controller, a single Controller subclass, or an object "
        "named `controller`"
    )


def _validate(obj: Any, module) -> Controller:
    if not callable(getattr(obj, "control", None)):
        raise ControllerLoadError(
            f"{type(obj).__name__} from {module.__name__} has no control(obs) method"
        )
    if not hasattr(obj, "name") or not obj.name or obj.name == "unnamed":
        try:
            obj.name = getattr(module, "__name__", "plugin").split("_")[-1]
        except AttributeError:  # pragma: no cover - frozen or slotted objects
            pass
    return obj


def load_controller(path: str | Path, **kwargs) -> Controller:
    """Load a controller from a ``.py`` file. See :func:`load_controller_module`."""
    return instantiate(load_controller_module(path), **kwargs)


def check_command(value: Any, who: str = "controller") -> ControlCommand:
    """Coerce and sanity-check whatever a controller returned.

    A tuple or a dict is accepted, because that is what a first attempt usually
    returns, and rejecting it would teach nothing.  ``NaN`` is *not* accepted:
    it propagates silently into the plant and shows up ten seconds later as a
    vehicle at infinity.
    """
    if isinstance(value, ControlCommand):
        cmd = value
    elif isinstance(value, dict):
        cmd = ControlCommand(**value)
    elif isinstance(value, (tuple, list)) and len(value) in (2, 3):
        cmd = ControlCommand(*value) if len(value) == 3 else ControlCommand(
            steer=value[0],
            throttle=max(float(value[1]), 0.0),
            brake=max(-float(value[1]), 0.0),
        )
    else:
        raise TypeError(
            f"{who} returned {type(value).__name__}; expected a ControlCommand, "
            "a dict, (steer, accel) or (steer, throttle, brake)"
        )
    for field_name in ("steer", "throttle", "brake"):
        v = float(getattr(cmd, field_name))
        if not np.isfinite(v):
            raise ValueError(f"{who} returned a non-finite {field_name}: {v}")
    return cmd.clipped()
