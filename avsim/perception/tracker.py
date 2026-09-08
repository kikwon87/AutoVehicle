"""Multi-object tracking: constant-velocity Kalman filters with gated association.

The planner needs *velocities*, and a detector supplies only positions.  A
tracker is the cheapest thing that turns one into the other, and it introduces
two failure modes the planner must survive:

* **lag** -- a constant-velocity filter is biased during a turn or a braking
  manoeuvre, precisely when the prediction matters;
* **identity switches** -- two vehicles passing close by can swap tracks, after
  which their predicted futures are exchanged.

Association is greedy nearest neighbour under a **Mahalanobis gate**, so a
detection is only matched when it is statistically compatible with the track's
own uncertainty.  Gating on Euclidean distance instead would associate a
confident near track with a noisy far detection, which is where most identity
switches come from.

Track management is M-of-N: a track is *tentative* until confirmed by
``n_confirm`` hits and is deleted after ``n_miss`` consecutive misses.  Only
confirmed tracks are published, because acting on a single detection is how a
planner brakes for a shadow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.conventions import wrap_to_pi
from .sensor import Detection


@dataclass
class Track:
    """A tracked object with state ``[x, y, vx, vy]``."""

    id: int
    x: np.ndarray                 #: 4-vector
    P: np.ndarray                 #: 4x4 covariance
    psi: float = 0.0
    length: float = 4.6
    width: float = 1.85
    hits: int = 1
    misses: int = 0
    age: int = 0
    confirmed: bool = False
    truth_id: str | None = None   #: evaluation only

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.x[2:]))

    @property
    def heading(self) -> float:
        """Heading from the velocity when moving, from the measurement when not.

        A stationary object has no velocity direction, and reading one out of
        filter noise gives a heading that spins -- which then propagates into a
        prediction that fans out in every direction.
        """
        return float(np.arctan2(self.x[3], self.x[2])) if self.speed > 0.5 else self.psi

    def predict_positions(self, times: np.ndarray) -> np.ndarray:
        """Constant-velocity prediction at the given times, shape ``(len(times), 2)``.

        The honest baseline.  It is wrong for anything turning, and the
        scenarios that depend on it say so.
        """
        t = np.asarray(times, dtype=float).reshape(-1, 1)
        return self.position[None, :] + t * self.velocity[None, :]


class MultiObjectTracker:
    """Greedy nearest-neighbour tracker over constant-velocity Kalman filters."""

    def __init__(
        self,
        dt: float,
        #: Process noise as an acceleration standard deviation.  It must cover
        #: the *manoeuvres* the tracked objects actually perform: a leader
        #: braking at 4 m/s^2 violates a constant-velocity model badly enough
        #: that a tight gate rejects its own detections, deletes the track and
        #: re-spawns it with zero velocity -- precisely when the follow
        #: controller needs the velocity most.
        process_accel_std: float = 3.0,
        gate_chi2: float = 13.8,   #: 99.9% for 2 dof
        n_confirm: int = 2,
        n_miss: int = 5,
        max_tracks: int = 64,
    ):
        self.dt = float(dt)
        self.q = float(process_accel_std)
        self.gate = float(gate_chi2)
        self.n_confirm = int(n_confirm)
        self.n_miss = int(n_miss)
        self.max_tracks = int(max_tracks)
        self.tracks: list[Track] = []
        self._next_id = 0

        d = self.dt
        self.F = np.array([[1, 0, d, 0], [0, 1, 0, d], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        # Piecewise-constant white-acceleration process noise.
        g = np.array([[0.5 * d * d, 0], [0, 0.5 * d * d], [d, 0], [0, d]])
        self.Q = g @ (self.q**2 * np.eye(2)) @ g.T
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 0

    # --- steps ---------------------------------------------------------------

    def predict(self) -> None:
        for t in self.tracks:
            t.x = self.F @ t.x
            t.P = self.F @ t.P @ self.F.T + self.Q
            t.age += 1

    def _gate_matrix(self, dets: list[Detection]) -> np.ndarray:
        """Mahalanobis distances, ``inf`` where the pair fails the gate."""
        D = np.full((len(self.tracks), len(dets)), np.inf)
        for i, tr in enumerate(self.tracks):
            S_base = self.H @ tr.P @ self.H.T
            for j, det in enumerate(dets):
                S = S_base + det.covariance
                nu = np.array([det.x, det.y]) - self.H @ tr.x
                try:
                    d2 = float(nu @ np.linalg.solve(S, nu))
                except np.linalg.LinAlgError:  # pragma: no cover - singular S
                    continue
                if d2 <= self.gate:
                    D[i, j] = d2
        return D

    def update(self, dets: list[Detection]) -> list[Track]:
        """One predict/associate/update/manage cycle; returns confirmed tracks."""
        self.predict()

        D = self._gate_matrix(dets)
        matched_t: set[int] = set()
        matched_d: set[int] = set()
        # Greedy: repeatedly take the globally best remaining compatible pair.
        while np.isfinite(D).any():
            i, j = np.unravel_index(int(np.argmin(D)), D.shape)
            if not np.isfinite(D[i, j]):
                break
            self._correct(self.tracks[i], dets[j])
            matched_t.add(int(i))
            matched_d.add(int(j))
            D[i, :] = np.inf
            D[:, j] = np.inf

        for i, tr in enumerate(self.tracks):
            if i not in matched_t:
                tr.misses += 1

        for j, det in enumerate(dets):
            if j not in matched_d and len(self.tracks) < self.max_tracks:
                self.tracks.append(self._spawn(det))

        self.tracks = [t for t in self.tracks if t.misses <= self.n_miss]
        for t in self.tracks:
            if t.hits >= self.n_confirm:
                t.confirmed = True
        return self.confirmed_tracks()

    def _correct(self, tr: Track, det: Detection) -> None:
        z = np.array([det.x, det.y])
        S = self.H @ tr.P @ self.H.T + det.covariance
        K = tr.P @ self.H.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ (z - self.H @ tr.x)
        I_KH = np.eye(4) - K @ self.H
        # Joseph form: stays symmetric positive-definite under round-off, which
        # the short form does not over a long run.
        tr.P = I_KH @ tr.P @ I_KH.T + K @ det.covariance @ K.T
        tr.psi = float(wrap_to_pi(det.psi))
        tr.length, tr.width = det.length, det.width
        tr.hits += 1
        tr.misses = 0
        tr.truth_id = det.truth_id

    def _spawn(self, det: Detection) -> Track:
        self._next_id += 1
        P = np.eye(4) * 1.0
        P[:2, :2] = det.covariance
        P[2:, 2:] = np.eye(2) * (15.0**2)  # velocity unknown on the first frame
        return Track(
            id=self._next_id,
            x=np.array([det.x, det.y, 0.0, 0.0]),
            P=P,
            psi=det.psi,
            length=det.length,
            width=det.width,
            truth_id=det.truth_id,
        )

    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.confirmed]
