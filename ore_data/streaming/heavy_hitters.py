"""Heavy-hitter detection with Space-Saving summaries (Phase F, Steps 35-36).

Space-Saving maintains a bounded set of monitored counters; any unmonitored
key's true frequency is at most the total stream weight divided by the
capacity, so every reported estimate upper-bounds and lower-bounds the true
count within a documented error guarantee (Step 35). Candidates are
validated against exact aggregation over a bounded retained window before
high-impact decisions; approximate rankings stay qualified as approximate
(Step 36).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class HeavyHitterCandidate:
    """One heavy-hitter candidate with its documented error guarantee (Step 35)."""

    key: str
    estimate: float
    lower_bound: float
    upper_bound: float
    stream_weight: float
    capacity: int
    algorithm: str = "space_saving"
    algorithm_version: str = "0.1.0"
    timestamp: str = ""
    source_watermark: int = 0

    def __post_init__(self) -> None:
        """Stamp the candidate."""
        if not self.timestamp:
            object.__setattr__(self, "timestamp", _now())

    @property
    def exact(self) -> bool:
        """Heavy-hitter estimates carry error guarantees, not exact counts."""
        return False

    def guarantee_description(self) -> str:
        """Return the documented guarantee of this candidate (Step 35)."""
        return (
            "true frequency lies within [estimate - W/k, estimate] where W is "
            f"the total stream weight ({self.stream_weight:.0f}) and k the "
            f"capacity ({self.capacity}); the ranking is approximate"
        )

    def to_dict(self) -> dict[str, object]:
        """Return the candidate with its full metadata (section 5.1 requirement)."""
        return {
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "key": self.key,
            "estimate": self.estimate,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "stream_weight": self.stream_weight,
            "capacity": self.capacity,
            "guarantee": self.guarantee_description(),
            "timestamp": self.timestamp,
            "source_watermark": self.source_watermark,
            "exact": self.exact,
        }


@dataclass(frozen=True, slots=True)
class HeavyHitterValidation:
    """Validation of heavy-hitter candidates against exact aggregation (Step 36)."""

    requested_top_k: int
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    spurious: tuple[str, ...]
    recall: float
    precision: float
    exact_top_k: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the validation report as a plain dictionary."""
        return {
            "requested_top_k": self.requested_top_k,
            "matched": list(self.matched),
            "missing": list(self.missing),
            "spurious": list(self.spurious),
            "recall": self.recall,
            "precision": self.precision,
            "exact_top_k": list(self.exact_top_k),
        }


def _now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class SpaceSavingSummary:
    """Space-Saving heavy-hitter summary over weighted streams (Step 35).

    The stream may be unweighted or weighted; weights are validated to be
    finite and non-negative. The classic guarantee: for a stream of total
    weight W and capacity k, every unmonitored key has true frequency at
    most W/k, and every monitored key's true frequency lies within
    [estimate - W/k, estimate].
    """

    def __init__(
        self,
        capacity: int,
        *,
        namespace: str = "ore.data.heavy_hitters",
        retain_window: int = 0,
    ) -> None:
        """Create one empty summary with capacity k and optional retained window."""
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if retain_window < 0:
            raise ValueError("retain_window must be non-negative")
        self.capacity = capacity
        self.namespace = namespace
        self._counters: dict[str, float] = {}
        self._total_weight = 0.0
        self._watermark = 0
        self._retained: list[str] = [] if retain_window else []
        self._retain_limit = retain_window

    # -- update path -------------------------------------------------------

    def update(self, key: str, weight: float = 1.0, watermark: int | None = None) -> None:
        """Update one key's counter with a validated weight (Step 35)."""
        if not isinstance(weight, (int, float)) or not math.isfinite(float(weight)):
            raise ValueError("weight must be a finite number")
        if weight < 0:
            raise ValueError("weights must be non-negative")
        if not key.strip():
            raise ValueError("key must be non-empty")
        if key in self._counters:
            self._counters[key] += weight
        else:
            if len(self._counters) >= self.capacity:
                minimum_key = min(self._counters, key=lambda item: self._counters[item])
                base = self._counters.pop(minimum_key)
                self._counters[key] = base + weight
            else:
                self._counters[key] = weight
        self._total_weight += weight
        if watermark is not None:
            self._watermark = max(self._watermark, int(watermark))
        if self._retain_limit:
            self._retained.append(key)
            if len(self._retained) > self._retain_limit:
                self._retained = self._retained[-self._retain_limit :]

    def update_all(self, entries: Iterable[tuple[str, float]]) -> None:
        """Update many (key, weight) pairs."""
        for key, weight in entries:
            self.update(key, weight)

    # -- query ---------------------------------------------------------------

    def top(self, top_k: int | None = None) -> tuple[HeavyHitterCandidate, ...]:
        """Return the heavy-hitter candidates, each with its error guarantee (Step 35)."""
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when given")
        limit = top_k or self.capacity
        error_bound = self._total_weight / self.capacity
        ordered = sorted(self._counters.items(), key=lambda pair: pair[1], reverse=True)
        return tuple(
            HeavyHitterCandidate(
                key=key,
                estimate=estimate,
                lower_bound=max(0.0, estimate - error_bound),
                upper_bound=estimate,
                stream_weight=self._total_weight,
                capacity=self.capacity,
                timestamp=_now(),
                source_watermark=self._watermark,
            )
            for key, estimate in ordered[:limit]
        )

    def error_bound(self) -> float:
        """Return the maximum frequency error for any unmonitored key: W/k."""
        return self._total_weight / self.capacity

    # -- validation (Step 36) ---------------------------------------------------

    def exact_aggregation(self) -> dict[str, float]:
        """Aggregate exact counts over the bounded retained window (Step 36)."""
        if not self._retained:
            raise ValueError("no retained window; configure retain_window for validation")
        counts: dict[str, float] = {}
        for key in self._retained:
            counts[key] = counts.get(key, 0.0) + 1.0
        return counts

    def validate_against_exact(
        self,
        exact_counts: Mapping[str, float],
        top_k: int,
    ) -> HeavyHitterValidation:
        """Measure recall and precision against exact top-k aggregation (Step 36)."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        exact_top = set(
            key
            for key, _ in sorted(exact_counts.items(), key=lambda pair: pair[1], reverse=True)[
                :top_k
            ]
        )
        reported = tuple(candidate.key for candidate in self.top(top_k))
        reported_set = set(reported)
        matched = tuple(key for key in reported if key in exact_top)
        missing = tuple(key for key in sorted(exact_top - reported_set))
        spurious = tuple(key for key in reported if key not in exact_top)
        recall = len(matched) / len(exact_top) if exact_top else 1.0
        precision = len(matched) / len(reported_set) if reported_set else 1.0
        return HeavyHitterValidation(
            requested_top_k=top_k,
            matched=matched,
            missing=missing,
            spurious=spurious,
            recall=recall,
            precision=precision,
            exact_top_k=tuple(sorted(exact_top)),
        )

    # -- introspection -----------------------------------------------------------

    @property
    def total_weight(self) -> float:
        """Return the total stream weight W."""
        return self._total_weight

    @property
    def watermark(self) -> int:
        """Return the source watermark of this summary's updates."""
        return self._watermark

    def memory_bytes(self) -> int:
        """Return the summary's allocation size."""
        return len(self._counters) * 24

    def describe(self) -> dict[str, object]:
        """Return the summary metadata for observability."""
        return {
            "algorithm": "space_saving",
            "version": "0.1.0",
            "namespace": self.namespace,
            "capacity": self.capacity,
            "monitored": len(self._counters),
            "total_weight": self._total_weight,
            "error_bound": self.error_bound(),
            "watermark": self._watermark,
            "memory_bytes": self.memory_bytes(),
        }


def heavy_hitter_candidates_are_not_authoritative(
    candidates: Sequence[HeavyHitterCandidate],
) -> str:
    """Document that candidates guide investigation, caching and dashboards (Step 36)."""
    if not candidates:
        return "no heavy-hitter candidates reported"
    return (
        "approximate ranking: "
        + ", ".join(f"{candidate.key}~{candidate.estimate:.0f}" for candidate in candidates)
        + "; high-impact decisions require exact validation over a retained window"
    )


__all__ = [
    "HeavyHitterCandidate",
    "HeavyHitterValidation",
    "SpaceSavingSummary",
    "heavy_hitter_candidates_are_not_authoritative",
]
