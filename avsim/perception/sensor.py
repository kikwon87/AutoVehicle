"""A deliberately simple forward vision sensor.

The course assumes perception is solved and studies behaviour; this module
therefore models only the properties of a camera that a *planner* has to cope
with, and models them explicitly rather than hiding them:

* a finite **field of view** and range, so objects appear and disappear;
* **occlusion**: a vehicle hidden behind another is not detected, which is what
  makes an intersection dangerous rather than merely busy;
* **range-dependent noise**.  For a monocular camera the depth error grows with
  the square of the range while the lateral error grows linearly, because
  depth comes from a disparity or a size cue and bearing comes from a pixel
  offset.  Using a single isotropic sigma would make far-away objects look far
  better localized than they are;
* **misses** (a detection probability that falls with range) and **latency**.

What is *not* modelled: false positives from clutter, classification error,
extrinsic calibration error, and rolling shutter.  Those matter in practice and
their absence is the reason a KPI measured here is optimistic about perception
even when it is honest about control.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..core.conventions import wrap_to_pi
from ..core.geometry import segments_intersect
from ..world.actors import ActorState


@dataclass
class Detection:
    """One measurement of one object, in **world** coordinates.

    Positions are reported in the world frame for the tracker's convenience,
    but they were *measured* in the sensor frame, so the covariance is the
    rotated sensor-frame covariance rather than a diagonal.  Reporting a
    diagonal here is the standard way to make a tracker overconfident about
    range.
    """

    x: float
    y: float
    psi: float
    covariance: np.ndarray            #: 2x2 position covariance, world frame
    length: float = 4.6
    width: float = 1.85
    range: float = 0.0
    bearing: float = 0.0
    #: ground-truth id, retained for evaluation only -- never for association
    truth_id: str | None = None


@dataclass
class VisionSensor:
    """Forward-looking camera-like detector."""

    fov: float = np.deg2rad(110.0)
    max_range: float = 80.0
    min_range: float = 0.5
    #: depth error ``sigma_r = sigma_r0 + k_r * range^2`` [m].  At the defaults
    #: this is 0.2 m at 5 m, 1.5 m at 40 m and 5.3 m at 80 m -- roughly 6% of
    #: range at the far end, which is the order a monocular depth cue achieves.
    sigma_r0: float = 0.2
    k_r: float = 0.0008
    #: lateral error ``sigma_t = k_t * range`` [m]; bearing is far better
    #: conditioned than depth, which is the whole point of separating them.
    k_t: float = 0.006
    sigma_heading: float = np.deg2rad(6.0)
    #: Detection probability decays as ``p0 * exp(-range / range_scale)``:
    #: 0.97 at 10 m, 0.90 at 40 m, 0.81 at 80 m.  Misses are what force the
    #: tracker to coast, and coasting is where a constant-velocity prediction
    #: does its damage -- so the rate matters more than it looks.
    p_detect_near: float = 0.99
    range_scale: float = 400.0
    occlusion: bool = True
    #: whole frames of delay before a detection is released to the tracker
    latency_steps: int = 1
    #: sensor origin ahead of the CG along the body x-axis [m]
    mount_offset: float = 1.5

    _buffer: deque = field(default_factory=deque, init=False, repr=False)

    def reset(self) -> None:
        self._buffer = deque()

    # --- internals -----------------------------------------------------------

    def _origin(self, ego: ActorState) -> np.ndarray:
        return ego.position + self.mount_offset * np.array([np.cos(ego.psi), np.sin(ego.psi)])

    def _visible(self, origin: np.ndarray, target: ActorState, others: list[ActorState]) -> bool:
        """Line-of-sight test against every other actor's bounding box."""
        if not self.occlusion:
            return True
        for blocker in others:
            if blocker is target:
                continue
            corners = blocker.corners()
            for i in range(4):
                if segments_intersect(origin, target.position, corners[i], corners[(i + 1) % 4]):
                    return False
        return True

    # --- public API ----------------------------------------------------------

    def observe(
        self, ego: ActorState, actors: list[ActorState], rng: np.random.Generator
    ) -> list[Detection]:
        """Detections released **this** frame, i.e. delayed by ``latency_steps``."""
        fresh = self._measure(ego, actors, rng)
        self._buffer.append(fresh)
        while len(self._buffer) > self.latency_steps + 1:
            self._buffer.popleft()
        return self._buffer[0] if len(self._buffer) == self.latency_steps + 1 else []

    def _measure(
        self, ego: ActorState, actors: list[ActorState], rng: np.random.Generator
    ) -> list[Detection]:
        origin = self._origin(ego)
        out: list[Detection] = []
        for a in actors:
            if a.id == ego.id:
                continue
            d = a.position - origin
            rng_m = float(np.linalg.norm(d))
            if not (self.min_range <= rng_m <= self.max_range):
                continue
            bearing = wrap_to_pi(np.arctan2(d[1], d[0]) - ego.psi)
            if abs(bearing) > 0.5 * self.fov:
                continue
            if not self._visible(origin, a, actors):
                continue
            p = self.p_detect_near * np.exp(-rng_m / self.range_scale)
            if rng.random() > p:
                continue

            sigma_r = self.sigma_r0 + self.k_r * rng_m**2
            sigma_t = max(self.k_t * rng_m, 1e-3)
            # Sample in the sensor's (range, cross-range) frame, then rotate.
            e_r, e_t = rng.normal(0.0, sigma_r), rng.normal(0.0, sigma_t)
            los = d / max(rng_m, 1e-9)
            perp = np.array([-los[1], los[0]])
            pos = a.position + e_r * los + e_t * perp

            R_los = np.column_stack([los, perp])
            cov = R_los @ np.diag([sigma_r**2, sigma_t**2]) @ R_los.T

            out.append(
                Detection(
                    x=float(pos[0]),
                    y=float(pos[1]),
                    psi=float(wrap_to_pi(a.psi + rng.normal(0.0, self.sigma_heading))),
                    covariance=cov,
                    length=a.length,
                    width=a.width,
                    range=rng_m,
                    bearing=float(bearing),
                    truth_id=a.id,
                )
            )
        return out

    def visible_ids(self, ego: ActorState, actors: list[ActorState]) -> set[str]:
        """Ground-truth ids inside the FOV and unoccluded, ignoring noise and misses.

        Used by the KPI layer to separate "the sensor could not see it" from
        "the sensor saw it and the tracker dropped it".
        """
        origin = self._origin(ego)
        out = set()
        for a in actors:
            if a.id == ego.id:
                continue
            d = a.position - origin
            r = float(np.linalg.norm(d))
            if not (self.min_range <= r <= self.max_range):
                continue
            if abs(wrap_to_pi(np.arctan2(d[1], d[0]) - ego.psi)) > 0.5 * self.fov:
                continue
            if self._visible(origin, a, actors):
                out.add(a.id)
        return out
