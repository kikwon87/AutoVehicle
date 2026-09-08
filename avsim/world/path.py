"""Reference paths: arc-length parameterization, curvature, and projection.

A reference path is the object the Frenet frame follows and the object the
planner offsets from.  It must supply, for any arc length ``s``:

* ``position(s)``  -- the point ``p_r(s)``;
* ``heading(s)``   -- the tangent angle ``theta_r(s)``;
* ``curvature(s)`` -- ``kappa(s) = d theta_r / d s``;

and, given a world point, the projection ``s`` of that point onto the path.

Two implementations share this interface:

:class:`PrimitivePath`
    A chain of straights and circular arcs.  Position, heading and curvature
    are **exact**, and ``kappa`` is piecewise constant.  Used to lay out roads
    and intersection turns, where the geometry is known analytically and a
    spline would only add approximation error.

:class:`SplinePath`
    A natural cubic spline through waypoints, resampled uniformly in arc
    length.  Used for paths that come from data -- a recorded route, or a
    planner's own output.

Projection uses a **seeded local search** (``s_guess``).  Re-projecting
globally every step is what makes a controller jump between the two branches
of a hairpin; seeding with the previous ``s`` keeps the branch continuous.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..core.conventions import wrap_to_pi


class ReferencePath:
    """Interface shared by all reference paths."""

    length: float

    def position(self, s: float) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def heading(self, s: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def curvature(self, s: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    # --- shared helpers ------------------------------------------------------

    def frame(self, s: float) -> tuple[np.ndarray, float, float]:
        """``(position, heading, curvature)`` in one call."""
        return self.position(s), self.heading(s), self.curvature(s)

    def normal(self, s: float) -> np.ndarray:
        """Left-pointing unit normal ``n(s) = [-sin theta, cos theta]``.

        Left-pointing so that a positive lateral offset ``e_y`` is to the left,
        consistent with the ISO ``y``-axis used everywhere else.
        """
        th = self.heading(s)
        return np.array([-math.sin(th), math.cos(th)])

    def to_cartesian(self, s: float, e_y: float) -> np.ndarray:
        """Frenet ``(s, e_y)`` to world ``(x, y)``."""
        return self.position(s) + e_y * self.normal(s)

    def table(self, ds: float = 0.25) -> dict:
        """Cached tabulation of the path for **vectorized** queries.

        Returns ``{"s", "x", "y", "theta", "kappa"}`` on a uniform grid, with
        ``theta`` unwrapped so linear interpolation across ``+-pi`` is correct.

        Scalar ``position``/``heading``/``curvature`` calls are exact but cost a
        Python call each; a trajectory lattice makes tens of thousands of them
        per plan, which dominates the planning time by an order of magnitude
        over the optimizer it feeds.  Interpolating a 0.25 m table is accurate
        to well under a centimetre on these geometries and turns that loop into
        three :func:`numpy.interp` calls.
        """
        cached = getattr(self, "_table_cache", None)
        if cached is not None and abs(cached["ds"] - ds) < 1e-12:
            return cached
        n = max(int(self.length / ds) + 1, 4)
        ss = np.linspace(0.0, self.length, n)
        xy = np.array([self.position(v) for v in ss])
        th = np.unwrap(np.array([self.heading(v) for v in ss]))
        ka = np.array([self.curvature(v) for v in ss])
        cached = {"ds": ds, "s": ss, "x": xy[:, 0], "y": xy[:, 1], "theta": th, "kappa": ka}
        self._table_cache = cached
        return cached

    def frames(self, s: np.ndarray, ds: float = 0.25):
        """Vectorized ``(x, y, theta, kappa)`` at many arc lengths."""
        tab = self.table(ds)
        sc = np.clip(np.asarray(s, dtype=float), 0.0, self.length)
        return (
            np.interp(sc, tab["s"], tab["x"]),
            np.interp(sc, tab["s"], tab["y"]),
            wrap_to_pi(np.interp(sc, tab["s"], tab["theta"])),
            np.interp(sc, tab["s"], tab["kappa"]),
        )

    def sample(self, ds: float = 0.5) -> np.ndarray:
        """Uniformly sampled polyline ``(N, 2)`` for rendering and collision tests."""
        n = max(int(self.length / ds) + 1, 2)
        x, y, _, _ = self.frames(np.linspace(0.0, self.length, n))
        return np.column_stack([x, y])

    def project(self, x: float, y: float, s_guess: float | None = None, window: float = 25.0) -> float:
        """Arc length of the closest path point, searched near ``s_guess``.

        Coarse scan over a window, then a few Newton steps on
        ``d/ds ||p - p_r(s)||^2 = -2 (p - p_r) . t(s) = 0``.
        """
        p = np.array([x, y], dtype=float)
        if s_guess is None:
            lo, hi = 0.0, self.length
        else:
            lo = max(0.0, s_guess - window)
            hi = min(self.length, s_guess + window)
        if hi <= lo:
            return float(np.clip(s_guess or 0.0, 0.0, self.length))

        # The coarse scan goes through the cached table rather than through
        # ``position`` per sample.  A 25 m window at 0.25 m is 200 samples, and
        # a planning tick projects tens of points onto the same path -- as a
        # Python loop that is by far the most expensive thing in the stack, and
        # as three ``np.interp`` calls it is free.  Accuracy is unaffected: the
        # scan only chooses which basin the Newton iteration below starts in,
        # and the table's spacing is the scan's spacing.
        grid = np.linspace(lo, hi, max(int((hi - lo) / 0.25) + 2, 8))
        gx, gy, _, _ = self.frames(grid)
        d2 = (p[0] - gx) ** 2 + (p[1] - gy) ** 2
        s = float(grid[int(np.argmin(d2))])

        for _ in range(12):
            pr = self.position(s)
            th = self.heading(s)
            t = np.array([math.cos(th), math.sin(th)])
            k = self.curvature(s)
            n = np.array([-math.sin(th), math.cos(th)])
            d = p - pr
            g = -float(d @ t)                       # dF/ds, F = 0.5 ||d||^2
            H = 1.0 - k * float(d @ n)              # d2F/ds2
            if abs(H) < 1e-9:
                break
            ds = -g / H
            ds = float(np.clip(ds, -1.0, 1.0))
            s_new = float(np.clip(s + ds, 0.0, self.length))
            # A tenth of a micron is converged by any standard this simulation
            # cares about; iterating to 1e-10 only bought more scalar segment
            # lookups, three per iteration, on the hottest call in the stack.
            if abs(s_new - s) < 1e-7:
                s = s_new
                break
            s = s_new
        return s

    def lateral_offset(self, x: float, y: float, s: float) -> float:
        """Signed offset of ``(x, y)`` from the path at ``s``, positive to the left."""
        pr = self.position(s)
        th = self.heading(s)
        return float(-(x - pr[0]) * math.sin(th) + (y - pr[1]) * math.cos(th))

    def heading_error(self, psi: float, s: float) -> float:
        return float(wrap_to_pi(psi - self.heading(s)))


# --- primitive-based paths ----------------------------------------------------

@dataclass(frozen=True)
class Straight:
    """A straight segment of length ``length`` starting at ``(x, y, theta)``."""

    x: float
    y: float
    theta: float
    length: float

    @property
    def curvature(self) -> float:
        return 0.0

    def at(self, ds: float) -> tuple[np.ndarray, float]:
        return (
            np.array([self.x + ds * math.cos(self.theta), self.y + ds * math.sin(self.theta)]),
            self.theta,
        )

    def end(self) -> tuple[float, float, float]:
        p, th = self.at(self.length)
        return float(p[0]), float(p[1]), th


@dataclass(frozen=True)
class Arc:
    """A circular arc of signed curvature ``kappa`` (positive = left turn)."""

    x: float
    y: float
    theta: float
    length: float
    kappa: float

    @property
    def curvature(self) -> float:
        return self.kappa

    def at(self, ds: float) -> tuple[np.ndarray, float]:
        k = self.kappa
        if abs(k) < 1e-12:
            return Straight(self.x, self.y, self.theta, self.length).at(ds)
        th = self.theta + k * ds
        # Centre of the circle, offset along the left normal by the radius.
        cx = self.x - math.sin(self.theta) / k
        cy = self.y + math.cos(self.theta) / k
        return np.array([cx + math.sin(th) / k, cy - math.cos(th) / k]), th

    def end(self) -> tuple[float, float, float]:
        p, th = self.at(self.length)
        return float(p[0]), float(p[1]), th


class PrimitivePath(ReferencePath):
    """A chain of :class:`Straight` and :class:`Arc` segments.

    Curvature is piecewise constant and therefore **discontinuous** at segment
    joints.  That is geometrically honest for a road built from straights and
    arcs, but it means the steering feedforward ``delta_ff = (L + K_us V^2)
    kappa`` steps at the joint.  Real roads use clothoids for exactly this
    reason; :meth:`smoothed_curvature` provides a bounded-rate alternative for
    controllers that cannot accept the step.
    """

    def __init__(self, segments: Sequence[Straight | Arc]):
        if not segments:
            raise ValueError("a path needs at least one segment")
        self.segments = list(segments)
        lengths = np.array([seg.length for seg in self.segments], dtype=float)
        if np.any(lengths <= 0):
            raise ValueError("every segment must have positive length")
        self._s0 = np.concatenate([[0.0], np.cumsum(lengths)])
        self.length = float(self._s0[-1])

    @classmethod
    def chain(cls, x: float, y: float, theta: float, spec: Iterable[tuple[str, float, float]]) -> "PrimitivePath":
        """Build a connected chain from ``(kind, length, kappa)`` triples.

        Each segment starts where the previous one ended, so the path is
        ``C^0`` in position and ``C^0`` in heading by construction.
        """
        segs: list[Straight | Arc] = []
        for kind, length, kappa in spec:
            if kind == "straight":
                seg: Straight | Arc = Straight(x, y, theta, length)
            elif kind == "arc":
                seg = Arc(x, y, theta, length, kappa)
            else:
                raise ValueError(f"unknown primitive {kind!r}")
            segs.append(seg)
            x, y, theta = seg.end()
        return cls(segs)

    def _locate(self, s: float) -> tuple[int, float]:
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.searchsorted(self._s0, s, side="right") - 1)
        i = min(max(i, 0), len(self.segments) - 1)
        return i, s - self._s0[i]

    def position(self, s: float) -> np.ndarray:
        i, ds = self._locate(s)
        return self.segments[i].at(ds)[0]

    def heading(self, s: float) -> float:
        # Wrapped, so that a chain of arcs does not accumulate multiples of
        # 2*pi and hand a controller a heading of 270 degrees for due south.
        i, ds = self._locate(s)
        return float(wrap_to_pi(self.segments[i].at(ds)[1]))

    def curvature(self, s: float) -> float:
        i, _ = self._locate(s)
        return float(self.segments[i].curvature)

    def smoothed_curvature(self, s: float, window: float = 4.0) -> float:
        """Curvature averaged over ``+-window/2``, to bound the feedforward rate.

        This is a controller-side remedy for a geometry-side discontinuity; it
        does not change the path, only what the feedforward believes about it.
        """
        lo, hi = max(0.0, s - window / 2), min(self.length, s + window / 2)
        grid = np.linspace(lo, hi, 9)
        return float(np.mean([self.curvature(t) for t in grid]))


# --- spline paths -------------------------------------------------------------

def _natural_cubic_spline(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Second derivatives of the natural cubic spline through ``(t, y)``.

    Solves the standard symmetric tridiagonal system with the natural boundary
    condition ``y'' = 0`` at both ends, by the Thomas algorithm.  Written out
    rather than pulled from scipy so the core of this package stays numpy-only.
    """
    n = len(t)
    if n < 3:
        return np.zeros(n)
    h = np.diff(t)
    alpha = np.zeros(n)
    alpha[1:-1] = 3.0 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])

    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    for i in range(1, n - 1):
        l[i] = 2.0 * (t[i + 1] - t[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    c = np.zeros(n)
    for i in range(n - 2, -1, -1):
        c[i] = z[i] - mu[i] * c[i + 1]
    return 2.0 * c  # convert from the c-coefficient to y''


class SplinePath(ReferencePath):
    """Natural cubic spline through waypoints, resampled uniformly in arc length.

    The two-pass construction (spline in chord length, then resample by arc
    length) is what makes ``s`` an actual arc length -- interpolating directly
    in chord length gives a parameter that is *not* arc length, and every
    curvature and every Frenet velocity computed from it is then wrong by the
    local speed factor.
    """

    def __init__(self, waypoints: np.ndarray, ds: float = 0.2, closed: bool = False):
        wp = np.asarray(waypoints, dtype=float)
        if wp.ndim != 2 or wp.shape[1] != 2 or len(wp) < 2:
            raise ValueError("waypoints must be an (N, 2) array with N >= 2")
        if closed and not np.allclose(wp[0], wp[-1]):
            wp = np.vstack([wp, wp[:1]])

        chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))])
        if len(wp) == 2:
            dense_t = np.linspace(0.0, chord[-1], 200)
            dense = np.column_stack(
                [np.interp(dense_t, chord, wp[:, 0]), np.interp(dense_t, chord, wp[:, 1])]
            )
        else:
            ddx = _natural_cubic_spline(chord, wp[:, 0])
            ddy = _natural_cubic_spline(chord, wp[:, 1])
            dense_t = np.linspace(0.0, chord[-1], max(int(chord[-1] / (ds / 4)) + 2, 400))
            dense = np.column_stack(
                [
                    _spline_eval(chord, wp[:, 0], ddx, dense_t),
                    _spline_eval(chord, wp[:, 1], ddy, dense_t),
                ]
            )

        seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(arc[-1])

        n = max(int(self.length / ds) + 1, 4)
        self.s = np.linspace(0.0, self.length, n)
        self.xy = np.column_stack(
            [np.interp(self.s, arc, dense[:, 0]), np.interp(self.s, arc, dense[:, 1])]
        )
        d1 = np.gradient(self.xy, self.s, axis=0)
        d2 = np.gradient(d1, self.s, axis=0)
        self.theta = np.unwrap(np.arctan2(d1[:, 1], d1[:, 0]))
        speed = np.maximum(np.linalg.norm(d1, axis=1), 1e-9)
        self.kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / speed**3

    def position(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, self.length))
        return np.array([np.interp(s, self.s, self.xy[:, 0]), np.interp(s, self.s, self.xy[:, 1])])

    def heading(self, s: float) -> float:
        s = float(np.clip(s, 0.0, self.length))
        return float(wrap_to_pi(np.interp(s, self.s, self.theta)))

    def curvature(self, s: float) -> float:
        s = float(np.clip(s, 0.0, self.length))
        return float(np.interp(s, self.s, self.kappa))


def _spline_eval(t: np.ndarray, y: np.ndarray, ypp: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Evaluate a cubic spline given nodal values and second derivatives."""
    i = np.clip(np.searchsorted(t, q, side="right") - 1, 0, len(t) - 2)
    h = t[i + 1] - t[i]
    a = (t[i + 1] - q) / h
    b = (q - t[i]) / h
    return (
        a * y[i]
        + b * y[i + 1]
        + ((a**3 - a) * ypp[i] + (b**3 - b) * ypp[i + 1]) * (h**2) / 6.0
    )


class ConcatPath(ReferencePath):
    """Several reference paths chained into one, by arc length.

    Used to turn a *route* -- a sequence of lanes and intersection connectors --
    into a single path the Frenet frame can follow end to end.

    The chain is checked for ``C^0`` continuity in position and heading at
    construction: a route assembled from mismatched lanes produces a path with
    an invisible jump, and every controller downstream then reports a
    cross-track error it cannot explain.  Curvature is *not* required to be
    continuous, because a straight meeting an arc genuinely steps.
    """

    def __init__(self, paths: Sequence[ReferencePath], tol_pos: float = 0.05, tol_heading: float = 1e-3):
        if not paths:
            raise ValueError("ConcatPath needs at least one path")
        self.paths = list(paths)
        for a, b in zip(self.paths[:-1], self.paths[1:]):
            gap = float(np.linalg.norm(a.position(a.length) - b.position(0.0)))
            dth = abs(float(wrap_to_pi(a.heading(a.length) - b.heading(0.0))))
            if gap > tol_pos or dth > tol_heading:
                raise ValueError(
                    f"route segments do not connect: position gap {gap:.3f} m, "
                    f"heading gap {np.rad2deg(dth):.3f} deg"
                )
        lengths = np.array([p.length for p in self.paths], dtype=float)
        self._s0 = np.concatenate([[0.0], np.cumsum(lengths)])
        self.length = float(self._s0[-1])

    def _locate(self, s: float) -> tuple[int, float]:
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.searchsorted(self._s0, s, side="right") - 1)
        i = min(max(i, 0), len(self.paths) - 1)
        return i, s - self._s0[i]

    def position(self, s: float) -> np.ndarray:
        i, ds = self._locate(s)
        return self.paths[i].position(ds)

    def heading(self, s: float) -> float:
        i, ds = self._locate(s)
        return self.paths[i].heading(ds)

    def curvature(self, s: float) -> float:
        i, ds = self._locate(s)
        return self.paths[i].curvature(ds)

    def segment_start(self, index: int) -> float:
        """Arc length at which sub-path ``index`` begins."""
        return float(self._s0[index])
