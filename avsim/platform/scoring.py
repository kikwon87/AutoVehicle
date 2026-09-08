"""Turning a run into one number, with the weights left in the user's hands.

The request names the difficulty exactly: mission time, safety and energy are
"서로 다른 차원의 문제" -- different dimensions -- so no fixed formula combines
them honestly.  What this module does instead:

1. Each raw metric is mapped to a **0-100 sub-score** by two thresholds the
   user sets: ``good`` (scores 100) and ``bad`` (scores 0), linear in between
   and clamped outside.  Two numbers per metric, both in the metric's own unit,
   so "a 2 second lap penalty" and "half a metre of clearance" are stated where
   they are meaningful rather than buried in a coefficient.
2. Sub-scores are combined by **weights**, also the user's, first within a
   category and then across categories.
3. A collision is not a weighted term. It is a **hard outcome**, handled by an
   explicit policy, because averaging a crash against a good lap time is the
   one thing a scoring function must not quietly do.

Every threshold and weight is data, serialisable to JSON, and the platform
persists whatever the user last used.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class MetricSpec:
    """How one raw measurement becomes a sub-score."""

    key: str
    label: str
    category: str
    unit: str
    good: float          #: value scoring 100
    bad: float           #: value scoring 0
    weight: float = 1.0
    help: str = ""

    @property
    def lower_is_better(self) -> bool:
        return self.good < self.bad

    def score(self, value: float) -> float:
        """Map a raw value onto ``[0, 100]``.

        Infinity is not a failure by itself -- its meaning depends on the
        metric's direction.  ``min_clearance = inf`` means there was nothing to
        be close to and is the best possible outcome; ``time_to_goal = inf``
        means the mission was never completed and is the worst.  Scoring both as
        zero, as a bare finiteness check does, punishes an empty road.
        """
        if value is None:
            return 0.0
        if np.isnan(value):
            return 0.0
        if np.isinf(value):
            best_is_large = not self.lower_is_better
            positive = value > 0
            return 100.0 if positive == best_is_large else 0.0
        span = self.bad - self.good
        if abs(span) < 1e-12:
            return 100.0 if value == self.good else 0.0
        frac = (self.bad - value) / span
        return float(np.clip(frac, 0.0, 1.0) * 100.0)


#: The default metric set.  Every number here is a starting point meant to be
#: argued with in the UI, which is why they are all editable.
DEFAULT_METRICS: tuple[MetricSpec, ...] = (
    # --- mission ----------------------------------------------------------------
    MetricSpec("time_to_goal", "Time to goal 임무 완료 시간", "mission", "s",
               good=20.0, bad=90.0, weight=1.0,
               help="Wall time of the run to the goal arc length. Not finishing scores 0."),
    MetricSpec("progress_ratio", "Route completed 경로 완주율", "mission", "-",
               good=1.0, bad=0.3, weight=1.0,
               help="Fraction of the route covered; separates 'slow' from 'stuck'."),
    MetricSpec("mean_speed", "Mean speed 평균 속도", "mission", "m/s",
               good=12.0, bad=2.0, weight=0.5),

    # --- safety -----------------------------------------------------------------
    MetricSpec("min_clearance", "Minimum clearance 최소 이격거리", "safety", "m",
               good=3.0, bad=0.2, weight=2.0,
               help="Closest box-to-box distance to any other vehicle."),
    MetricSpec("min_ttc", "Minimum TTC 최소 충돌시간", "safety", "s",
               good=4.0, bad=0.8, weight=1.5),
    MetricSpec("time_below_ttc", "Time in low TTC 위험 노출 시간", "safety", "s",
               good=0.0, bad=6.0, weight=1.0,
               help="Seconds spent with TTC under the safety threshold."),
    MetricSpec("max_friction_usage", "Peak friction usage 마찰 사용률", "safety", "-",
               good=0.6, bad=1.0, weight=1.0,
               help="Peak point on the friction ellipse; 1.0 is the tire's limit."),
    MetricSpec("corridor_exit_time", "Lane departure 차로 이탈 시간", "safety", "s",
               good=0.0, bad=4.0, weight=1.0),
    MetricSpec("red_light_violations", "Red-light violations 신호 위반", "safety", "count",
               good=0.0, bad=1.0, weight=2.0),

    # --- energy -------------------------------------------------------------------
    MetricSpec("steering_effort", "Steering effort 조타 사용량", "energy", "rad",
               good=2.0, bad=25.0, weight=1.0,
               help="Integral of |steering rate| -- total steering travel."),
    MetricSpec("accel_effort", "Pedal effort 엑셀·브레이크 사용량", "energy", "m/s",
               good=10.0, bad=90.0, weight=1.0,
               help="Integral of |commanded acceleration|."),
    MetricSpec("tractive_energy", "Tractive energy 구동 에너지", "energy", "kJ",
               good=200.0, bad=2500.0, weight=1.0,
               help="Integral of positive traction power; the physical energy bill."),

    # --- comfort ---------------------------------------------------------------------
    MetricSpec("max_lat_accel", "Peak lateral accel 최대 횡가속", "comfort", "m/s²",
               good=2.0, bad=7.0, weight=0.5),
    MetricSpec("jerk_rms", "Jerk (RMS) 저크", "comfort", "m/s³",
               good=1.0, bad=8.0, weight=0.5),
    MetricSpec("cross_track_rms", "Path error (RMS) 경로 추종 오차", "comfort", "m",
               good=0.15, bad=1.5, weight=1.0),

    # --- compute -----------------------------------------------------------------------
    MetricSpec("real_time_factor", "Real-time factor 실시간 계수", "compute", "-",
               good=0.2, bad=1.0, weight=0.5,
               help="Mean controller compute time over the control period; >1 is not real time."),
)

DEFAULT_CATEGORY_WEIGHTS = {
    "mission": 1.0,
    "safety": 3.0,
    "energy": 1.0,
    "comfort": 0.7,
    "compute": 0.3,
}

CATEGORY_LABELS = {
    "mission": "Mission 임무",
    "safety": "Safety 안전",
    "energy": "Energy 에너지",
    "comfort": "Comfort 승차감",
    "compute": "Compute 연산",
}


@dataclass
class ScoreConfig:
    """The user's scoring policy: thresholds, weights and how a crash counts."""

    metrics: list[MetricSpec] = field(
        default_factory=lambda: [MetricSpec(**asdict(m)) for m in DEFAULT_METRICS]
    )
    category_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CATEGORY_WEIGHTS)
    )
    #: "zero" -> a collision scores 0 overall; "penalty" -> subtract points
    collision_policy: str = "zero"
    collision_penalty: float = 60.0
    #: TTC below this counts towards ``time_below_ttc``
    ttc_threshold: float = 2.0

    # --- serialisation ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "metrics": [asdict(m) for m in self.metrics],
            "category_weights": dict(self.category_weights),
            "collision_policy": self.collision_policy,
            "collision_penalty": self.collision_penalty,
            "ttc_threshold": self.ttc_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "ScoreConfig":
        if not data:
            return cls()
        base = cls()
        by_key = {m.key: m for m in base.metrics}
        for entry in data.get("metrics", []):
            key = entry.get("key")
            if key in by_key:
                m = by_key[key]
                for f in ("good", "bad", "weight"):
                    if f in entry:
                        setattr(m, f, float(entry[f]))
        base.category_weights.update(
            {k: float(v) for k, v in (data.get("category_weights") or {}).items()}
        )
        base.collision_policy = str(data.get("collision_policy", base.collision_policy))
        base.collision_penalty = float(data.get("collision_penalty", base.collision_penalty))
        base.ttc_threshold = float(data.get("ttc_threshold", base.ttc_threshold))
        return base

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class ScoreBreakdown:
    """The result: one total, per-category totals, and every sub-score."""

    total: float
    categories: dict[str, float]
    metrics: list[dict]
    collided: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "categories": self.categories,
            "metrics": self.metrics,
            "collided": self.collided,
            "notes": self.notes,
        }


def score_run(raw: dict, config: ScoreConfig | None = None) -> ScoreBreakdown:
    """Score a run.

    ``raw`` is the flat dictionary of measurements produced by
    :func:`avsim.platform.session.collect_metrics`; metrics it does not contain
    are skipped rather than scored as zero, so adding a metric to the config
    before the session produces it degrades gracefully.
    """
    cfg = config or ScoreConfig()
    rows: list[dict] = []
    per_category: dict[str, list[tuple[float, float]]] = {}

    for spec in cfg.metrics:
        if spec.key not in raw:
            continue
        value = raw[spec.key]
        sub = spec.score(value)
        rows.append({
            "key": spec.key,
            "label": spec.label,
            "category": spec.category,
            "unit": spec.unit,
            "value": None if value is None or not np.isfinite(value) else float(value),
            "score": sub,
            "weight": spec.weight,
            "good": spec.good,
            "bad": spec.bad,
        })
        if spec.weight > 0:
            per_category.setdefault(spec.category, []).append((sub, spec.weight))

    categories: dict[str, float] = {}
    for cat, entries in per_category.items():
        w = sum(x[1] for x in entries)
        categories[cat] = float(sum(s * x for s, x in entries) / w) if w > 0 else 0.0

    total_w = sum(cfg.category_weights.get(c, 0.0) for c in categories)
    total = (
        float(sum(categories[c] * cfg.category_weights.get(c, 0.0) for c in categories) / total_w)
        if total_w > 0 else 0.0
    )

    notes: list[str] = []
    collided = bool(raw.get("collisions", 0) > 0)
    if collided:
        if cfg.collision_policy == "zero":
            notes.append("Collision: total forced to 0 (정책: zero).")
            total = 0.0
        else:
            notes.append(
                f"Collision: {cfg.collision_penalty:g} points subtracted (정책: penalty)."
            )
            total = max(total - cfg.collision_penalty, 0.0)
    if not raw.get("goal_reached", True):
        notes.append("Goal not reached within the time limit (임무 미완료).")

    return ScoreBreakdown(
        total=total, categories=categories, metrics=rows, collided=collided, notes=notes
    )


def compare(runs: Sequence[tuple[str, ScoreBreakdown]]) -> list[dict]:
    """Rank scored runs, best first -- the table the platform shows after a batch."""
    ranked = sorted(runs, key=lambda r: -r[1].total)
    return [
        {
            "rank": i + 1,
            "name": name,
            "total": b.total,
            "categories": b.categories,
            "collided": b.collided,
        }
        for i, (name, b) in enumerate(ranked)
    ]
