"""Planar geometry: oriented boxes, separating axes, and point-to-polyline distance.

Collision checking in this package is done with **oriented bounding boxes** and
the separating-axis theorem.  Two boxes are disjoint iff some axis separates
them, and for rectangles it suffices to test the four face normals.  Circles
would be cheaper but a 4.6 x 1.85 m car inscribed in a circle is 2.5 m wide,
which turns every ordinary lane change into a reported near-miss.

:func:`obb_penetration` returns a signed depth rather than a boolean, so a
scenario can report *how close* it came instead of only whether it touched --
which is what a safety KPI actually needs.
"""

from __future__ import annotations

import numpy as np


def rect_corners(x: float, y: float, psi: float, length: float, width: float, offset: float = 0.0) -> np.ndarray:
    """Corners of a rectangle centred ``offset`` ahead of ``(x, y)`` along its axis."""
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, -s], [s, c]])
    hl, hw = 0.5 * length, 0.5 * width
    local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]]) + np.array([offset, 0.0])
    return (R @ local.T).T + np.array([x, y])


def _axes(corners: np.ndarray) -> np.ndarray:
    """The two unique face normals of a rectangle given its corners."""
    e1 = corners[1] - corners[0]
    e2 = corners[2] - corners[1]
    out = []
    for e in (e1, e2):
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-12:
            out.append(n / norm)
    return np.array(out)


def obb_penetration(a: np.ndarray, b: np.ndarray) -> float:
    """Signed overlap depth between two convex polygons, by separating axes.

    Positive: the polygons overlap, and the value is the minimum translation
    depth.  Negative: they are disjoint, and the magnitude is the largest gap
    found over the tested axes -- a lower bound on the true clearance, which is
    the conservative direction for a safety metric.
    """
    best = np.inf
    for poly in (a, b):
        for axis in _axes(poly):
            pa = a @ axis
            pb = b @ axis
            overlap = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
            best = min(best, float(overlap))
    return best


def obb_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """True iff two oriented boxes intersect."""
    return obb_penetration(a, b) > 0.0


def polygon_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two convex polygons; ``0`` when they overlap.

    Computed as the minimum over all vertex-to-edge distances in both
    directions.  Exact for convex polygons, and cheap at these sizes.
    """
    if obb_overlap(a, b):
        return 0.0
    return min(_poly_point_min(a, b), _poly_point_min(b, a))


def _poly_point_min(poly: np.ndarray, pts: np.ndarray) -> float:
    return min(point_to_polygon_distance(p, poly) for p in pts)


def point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-15 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def point_to_polygon_distance(p: np.ndarray, poly: np.ndarray) -> float:
    n = len(poly)
    return min(point_to_segment_distance(p, poly[i], poly[(i + 1) % n]) for i in range(n))


def point_to_polyline_distance(p: np.ndarray, line: np.ndarray) -> float:
    return min(point_to_segment_distance(p, line[i], line[i + 1]) for i in range(len(line) - 1))


def segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
    """Proper segment intersection test, used for line-of-sight occlusion."""

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1, d2 = cross(q1, q2, p1), cross(q1, q2, p2)
    d3, d4 = cross(p1, p2, q1), cross(p1, p2, q2)
    return bool(((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)))


def time_to_collision(
    p_ego: np.ndarray,
    v_ego: np.ndarray,
    p_obj: np.ndarray,
    v_obj: np.ndarray,
    radius: float,
) -> float:
    """Time until two discs of combined ``radius`` touch, under constant velocity.

    Returns ``inf`` when they never do.  A disc approximation is deliberate
    here: TTC is a *scalar warning*, and its value would be dominated by
    orientation bookkeeping if boxes were used, while the quantity people
    compare across papers is the disc version.
    """
    dp = np.asarray(p_obj, dtype=float) - np.asarray(p_ego, dtype=float)
    dv = np.asarray(v_obj, dtype=float) - np.asarray(v_ego, dtype=float)
    a = float(dv @ dv)
    if a < 1e-12:
        return float("inf")
    b = 2.0 * float(dp @ dv)
    c = float(dp @ dp) - radius**2
    if c <= 0.0:
        return 0.0
    disc = b * b - 4 * a * c
    if disc < 0.0:
        return float("inf")
    root = np.sqrt(disc)
    t1, t2 = (-b - root) / (2 * a), (-b + root) / (2 * a)
    ts = [t for t in (t1, t2) if t >= 0.0]
    return float(min(ts)) if ts else float("inf")


def multi_circle_cover(length: float, width: float, n: int = 3) -> tuple[np.ndarray, float]:
    """Cover a rectangle with ``n`` equal circles along its axis.

    Returns ``(offsets, radius)`` where ``offsets`` are the circle centres along
    the body x-axis, measured from the rectangle centre.

    A single enclosing circle has radius ``hypot(L, W) / 2`` -- 2.48 m for a
    4.6 x 1.85 m car -- which is 2.7 times the car's actual half-width.  Every
    lateral clearance computed from it is wrong by more than a metre, and a
    planner using it reports an ordinary lane change as a collision.  Three
    circles bring the radius down to 1.20 m at a cost of nine distance
    evaluations instead of one.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    seg = length / n
    radius = 0.5 * float(np.hypot(seg, width))
    offsets = (np.arange(n) - (n - 1) / 2.0) * seg
    return offsets, radius


def circle_centres(x: float, y: float, psi: float, offsets: np.ndarray, body_offset: float = 0.0) -> np.ndarray:
    """World positions of body-frame axial offsets, shape ``(n, 2)``."""
    d = np.array([np.cos(psi), np.sin(psi)])
    return np.array([x, y])[None, :] + (np.asarray(offsets) + body_offset)[:, None] * d[None, :]


def ellipse_clearance(
    delta: np.ndarray, heading: float, a: float, b: float
) -> float:
    """Normalized clearance of a separation vector against an oriented ellipse.

    ``delta`` is the vector from the ellipse centre to the query point and
    ``heading`` the ellipse's major-axis direction.  Returns
    ``sqrt((d_lon/a)^2 + (d_lat/b)^2) - 1``: negative inside, zero on the
    boundary, positive outside.

    The anisotropy is the point.  A vehicle's predicted position is uncertain
    mostly *along* its direction of travel -- it may brake or accelerate -- and
    only slightly across it, because it is expected to stay in its lane.  An
    isotropic radius inherits the longitudinal growth in the lateral direction
    and forbids passes that are in fact wide open.
    """
    c, s = np.cos(heading), np.sin(heading)
    d_lon = delta[0] * c + delta[1] * s
    d_lat = -delta[0] * s + delta[1] * c
    return float(np.hypot(d_lon / max(a, 1e-6), d_lat / max(b, 1e-6)) - 1.0)
