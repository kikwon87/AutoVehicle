"""Predicting where the other vehicles will be.

Two predictors, and the difference between them is the point:

:class:`ConstantVelocityPredictor`
    Straight-line extrapolation of the tracked velocity.  Correct for a
    vehicle going straight, and *systematically wrong* for one turning -- it
    places a left-turning car outside the intersection while the car is
    actually crossing the ego's path.

:class:`LaneFollowingPredictor`
    Snaps each track to the nearest compatible lane and rolls it forward along
    that lane's centerline at its current speed.  Correct for a vehicle that
    stays on the road, and wrong when one does not.

Both return the same object, a :class:`Prediction` with a time grid and a
covariance that **grows with horizon**.  A prediction without growing
uncertainty invites a planner to treat a 5-second forecast as a fact, and the
resulting plan is confident exactly where it should not be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..core.conventions import wrap_to_pi
from ..perception.tracker import Track
from ..world.network import RoadNetwork


@dataclass
class Prediction:
    """Predicted positions of one object on a shared time grid."""

    track_id: int
    times: np.ndarray            #: (T,)
    positions: np.ndarray        #: (T, 2)
    headings: np.ndarray         #: (T,)
    #: 1-sigma **along** the object's heading at each time [m].  Grows fast: an
    #: unmodelled acceleration integrates twice.
    sigma_lon: np.ndarray
    #: 1-sigma **across** the heading [m].  Grows slowly, because a vehicle is
    #: expected to stay in its lane; treating it like the longitudinal figure
    #: forbids lateral passes that are in fact wide open.
    sigma_lat: np.ndarray
    length: float = 4.6
    width: float = 1.85
    mode: str = "constant_velocity"
    truth_id: str | None = None
    #: probability of this mode; modes of one track sum to 1
    probability: float = 1.0

    @property
    def sigma(self) -> np.ndarray:
        """Isotropic, conservative sigma -- the larger of the two axes."""
        return np.maximum(self.sigma_lon, self.sigma_lat)

    def radius(self, inflate: float = 1.0) -> np.ndarray:
        """Conservative isotropic occupancy radius per time step.

        Used by the coarse space-time conflict check in the behaviour layer,
        where over-conservatism only means yielding a little early.  The
        trajectory planner uses :meth:`ellipse_axes` instead, because there the
        same conservatism would forbid every lane change.
        """
        half_diag = 0.5 * float(np.hypot(self.length, self.width))
        return half_diag + inflate * self.sigma

    def ellipse_axes(self, inflate: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """``(a, b)`` semi-axes of the uncertainty ellipse at each time [m]."""
        return inflate * self.sigma_lon, inflate * self.sigma_lat

    def position_at(self, t: float) -> np.ndarray:
        return np.array(
            [np.interp(t, self.times, self.positions[:, 0]),
             np.interp(t, self.times, self.positions[:, 1])]
        )


def _uncertainty(times: np.ndarray, speed: float, sigma0: float, accel_sigma: float, lat_rate: float):
    """Longitudinal and lateral 1-sigma growth of a constant-velocity forecast.

    Longitudinal: an unknown acceleration of standard deviation ``accel_sigma``
    integrates to ``0.5 a t^2``.  Lateral: a vehicle drifts across its lane
    slowly and in proportion to how fast it is going, so ``lat_rate * v * t``,
    floored by the filter's own uncertainty.
    """
    sig_lon = sigma0 + 0.5 * accel_sigma * times**2
    sig_lat = sigma0 + lat_rate * max(speed, 0.0) * times
    return sig_lon, sig_lat


class ConstantVelocityPredictor:
    """Straight-line extrapolation with anisotropic, growing uncertainty."""

    def __init__(self, accel_sigma: float = 1.0, sigma0: float = 0.4, lat_rate: float = 0.05):
        self.accel_sigma = float(accel_sigma)
        self.sigma0 = float(sigma0)
        self.lat_rate = float(lat_rate)

    def __call__(self, tracks: Sequence[Track], times: np.ndarray) -> list[Prediction]:
        times = np.asarray(times, dtype=float)
        out = []
        for tr in tracks:
            pos = tr.predict_positions(times)
            sig_lon, sig_lat = _uncertainty(
                times, tr.speed, self.sigma0, self.accel_sigma, self.lat_rate
            )
            # The filter's own position uncertainty is the floor for both axes.
            floor = float(np.sqrt(max(np.trace(tr.P[:2, :2]) / 2.0, 0.0)))
            sig_lon = np.maximum(sig_lon, floor)
            sig_lat = np.maximum(sig_lat, floor)
            out.append(
                Prediction(
                    track_id=tr.id,
                    times=times,
                    positions=pos,
                    headings=np.full_like(times, tr.heading),
                    sigma_lon=sig_lon,
                    sigma_lat=sig_lat,
                    length=tr.length,
                    width=tr.width,
                    mode="constant_velocity",
                    truth_id=tr.truth_id,
                )
            )
        return out


#: Prior over which connector a vehicle takes at an intersection.  A flat
#: placeholder for a learned intention model -- stated as a constant so that a
#: scenario failure can be traced to *this* assumption rather than to an
#: unexamined default buried in the predictor.
MANOEUVRE_PRIOR = {
    "connector_straight": 0.6,
    "connector_right": 0.2,
    "connector_left": 0.2,
}


class LaneFollowingPredictor:
    """Roll each track forward along the lanes it can reach, one mode per route.

    A track is assigned to the lane whose centerline it is closest to *and*
    whose heading it agrees with to within ``heading_tol``; the heading test is
    what stops an oncoming vehicle from being snapped onto the ego's own lane.

    Where the assigned lane forks -- which at an intersection is always -- the
    predictor emits **one prediction per branch**, weighted by
    :data:`MANOEUVRE_PRIOR`.  Collapsing the fork to its most likely branch is
    what makes a predictor confidently place a left-turning car on the far side
    of the intersection while it is crossing in front of the ego.

    Tracks that match no lane fall back to constant velocity and say so in
    ``mode``, rather than pretending to be lane-aware.
    """

    def __init__(
        self,
        network: RoadNetwork,
        heading_tol: float = np.deg2rad(35.0),
        max_lateral: float = 3.0,
        accel_sigma: float = 0.7,
        sigma0: float = 0.4,
        lat_rate: float = 0.03,
        max_modes: int = 3,
    ):
        self.network = network
        self.heading_tol = float(heading_tol)
        self.max_lateral = float(max_lateral)
        self.accel_sigma = float(accel_sigma)
        self.sigma0 = float(sigma0)
        self.lat_rate = float(lat_rate)
        self.max_modes = int(max_modes)
        # The fallback is deliberately less certain: a track that matches no
        # lane is doing something the road model does not explain.
        self._fallback = ConstantVelocityPredictor(accel_sigma * 1.7, sigma0, lat_rate * 2.0)

    def _assign_lane(self, pos: np.ndarray, heading: float):
        best, best_cost = None, np.inf
        for lane in self.network.lanes.values():
            s = lane.centerline.project(pos[0], pos[1])
            if not (0.0 <= s <= lane.length):
                continue
            e_y = lane.centerline.lateral_offset(pos[0], pos[1], s)
            if abs(e_y) > self.max_lateral:
                continue
            dpsi = abs(wrap_to_pi(heading - lane.centerline.heading(s)))
            if dpsi > self.heading_tol:
                continue
            cost = abs(e_y) + 4.0 * dpsi
            if cost < best_cost:
                best, best_cost = (lane, s, e_y), cost
        return best

    def _routes(self, lane_id: str, s0: float, need: float) -> list[tuple[list[str], float]]:
        """Enumerate ``(lane_ids, probability)`` covering ``need`` metres ahead."""
        lane = self.network.lanes[lane_id]
        remaining = need - (lane.length - s0)
        if remaining <= 0 or not lane.successors:
            return [([lane_id], 1.0)]

        weights = []
        for sid in lane.successors:
            kind = self.network.lanes[sid].kind
            weights.append(MANOEUVRE_PRIOR.get(kind, 1.0))
        total = sum(weights) or 1.0

        out: list[tuple[list[str], float]] = []
        for sid, w in zip(lane.successors, weights):
            for tail, p_tail in self._routes(sid, 0.0, remaining):
                out.append(([lane_id] + tail, (w / total) * p_tail))
        out.sort(key=lambda r: -r[1])
        out = out[: self.max_modes]
        norm = sum(p for _, p in out) or 1.0
        return [(ids, p / norm) for ids, p in out]

    def _roll(self, lane_ids: list[str], s0: float, e_y: float, distances: np.ndarray):
        """Positions and headings at the given travelled distances along a route."""
        pos = np.empty((len(distances), 2))
        head = np.empty(len(distances))
        for i, d in enumerate(distances):
            s = s0 + d
            for lid in lane_ids:
                lane = self.network.lanes[lid]
                if s <= lane.length:
                    pos[i] = lane.centerline.to_cartesian(s, e_y)
                    head[i] = lane.centerline.heading(s)
                    break
                s -= lane.length
            else:
                lane = self.network.lanes[lane_ids[-1]]
                th = lane.centerline.heading(lane.length)
                pos[i] = lane.centerline.to_cartesian(lane.length, e_y) + s * np.array(
                    [np.cos(th), np.sin(th)]
                )
                head[i] = th
        return pos, head

    def __call__(self, tracks: Sequence[Track], times: np.ndarray) -> list[Prediction]:
        times = np.asarray(times, dtype=float)
        out: list[Prediction] = []
        for tr in tracks:
            match = self._assign_lane(tr.position, tr.heading)
            if match is None:
                out.extend(self._fallback([tr], times))
                continue
            lane, s0, e_y = match
            v = tr.speed
            distances = v * times
            sig_lon, sig_lat = _uncertainty(times, v, self.sigma0, self.accel_sigma, self.lat_rate)
            for lane_ids, prob in self._routes(lane.id, s0, float(distances[-1]) + 1.0):
                pos, head = self._roll(lane_ids, s0, e_y, distances)
                out.append(
                    Prediction(
                        track_id=tr.id,
                        times=times,
                        positions=pos,
                        headings=head,
                        sigma_lon=sig_lon,
                        sigma_lat=sig_lat,
                        length=tr.length,
                        width=tr.width,
                        mode="lane:" + ">".join(lane_ids),
                        truth_id=tr.truth_id,
                        probability=prob,
                    )
                )
        return out
