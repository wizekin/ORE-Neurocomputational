"""Transparent weighted evidence aggregation without treating scores as probabilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class EvidenceContribution:
    """Represent one source's signed support for or against a candidate hypothesis."""

    source_id: str
    contribution: float
    weight: float = 1.0
    correlation_group: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Require finite evidence values and non-negative source reliability weights."""
        if not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        if not math.isfinite(self.contribution) or not math.isfinite(self.weight):
            raise ValueError("evidence contribution and weight must be finite")
        if self.weight < 0:
            raise ValueError("evidence weight cannot be negative")
        if self.correlation_group is not None and not self.correlation_group.strip():
            raise ValueError("correlation_group must be non-empty when supplied")


@dataclass(frozen=True, slots=True)
class EvidenceScore:
    """Return an uncalibrated weighted evidence score and the evidence used to calculate it."""

    score: float
    used_evidence: tuple[EvidenceContribution, ...]
    all_evidence: tuple[EvidenceContribution, ...]
    suppressed_correlated_sources: tuple[str, ...] = ()
    calibrated_probability: bool = False


class HypothesisScorer:
    """Compute S(h)=sum_i w_i*e_i(h) with optional conservative correlation grouping."""

    def score(self, evidence: Iterable[EvidenceContribution]) -> EvidenceScore:
        """Sum unique-source evidence and avoid counting multiple correlated sources twice."""
        items = tuple(evidence)
        if any(not isinstance(item, EvidenceContribution) for item in items):
            raise TypeError("evidence must contain EvidenceContribution objects")
        groups: dict[str, list[EvidenceContribution]] = {}
        for item in items:
            group = item.correlation_group or f"source:{item.source_id}"
            groups.setdefault(group, []).append(item)
        used: list[EvidenceContribution] = []
        suppressed: list[str] = []
        for group in sorted(groups):
            candidates = groups[group]
            selected = max(
                candidates,
                key=lambda item: (abs(item.weight * item.contribution), item.source_id),
            )
            used.append(selected)
            suppressed.extend(item.source_id for item in candidates if item is not selected)
        total = sum(item.weight * item.contribution for item in used)
        return EvidenceScore(
            score=total,
            used_evidence=tuple(used),
            all_evidence=items,
            suppressed_correlated_sources=tuple(sorted(suppressed)),
            calibrated_probability=False,
        )