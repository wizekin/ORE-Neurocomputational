"""Graph prediction, semantic-safety and operational metrics (spec section 8).

Graph prediction: MRR, Hits@1/3/10, Precision@K, Recall@K. Semantic safety:
false-positive rate among promoted candidates, verifier rejection/ambiguity,
evidence-backed support and calibration error. Every aggregate is reported by
subgroup (relation type, domain, split) so poor subgroups are never hidden inside a
single overall number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..schemas.candidate_schema import (
    CandidateDisposition,
    GraphCandidate,
)
from ..verification.promotion_gate import VerificationResult


def mean_reciprocal_rank(ranks: Sequence[int]) -> float:
    """MRR over 1-based rank positions (graph prediction metric)."""
    if not ranks:
        return 0.0
    return float(np.mean([1.0 / rank for rank in ranks if rank >= 1]))


def hits_at_k(ranks: Sequence[int], k: int) -> float:
    """Fraction of ranked items whose 1-based rank is at most k."""
    if not ranks or k < 1:
        return 0.0
    return float(np.mean([1 if rank <= k else 0 for rank in ranks]))


def precision_at_k(ranked: Sequence[GraphCandidate], relevant: frozenset, k: int) -> float:
    """Precision@K: fraction of the top-k ranked candidates that are relevant."""
    if k < 1 or not ranked:
        return 0.0
    top = list(ranked)[:k]
    hits = sum(1 for item in top if item.triple() in relevant)
    return hits / len(top) if top else 0.0


def recall_at_k(ranked: Sequence[GraphCandidate], relevant: frozenset, k: int) -> float:
    """Recall@K: fraction of relevant items recovered in the top-k ranked candidates."""
    if k < 1 or not relevant:
        return 0.0
    top = list(ranked)[:k]
    hits = sum(1 for item in top if item.triple() in relevant)
    return hits / len(relevant)


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    bins: int = 10,
) -> float:
    """Binned expected calibration error (semantic-safety metric)."""
    if len(probabilities) != len(labels) or not probabilities:
        return 0.0
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)
        if not mask.any():
            continue
        bin_confidence = float(probabilities[mask].mean())
        bin_accuracy = float(labels[mask].mean())
        total += float(mask.sum()) / len(probabilities) * abs(bin_confidence - bin_accuracy)
    return total


@dataclass
class SemanticSafetyMetrics:
    """Verifier outcome fractions for one evaluation group (spec section 8)."""

    n_candidates: int = 0
    n_supported: int = 0
    n_rejected: int = 0
    n_ambiguous: int = 0
    n_unverifiable: int = 0
    evidence_backed_supported: int = 0
    contradiction_recall_known: float | None = None

    @classmethod
    def from_results(cls, results: Sequence[VerificationResult]) -> "SemanticSafetyMetrics":
        metrics = cls(n_candidates=len(results))
        for result in results:
            if result.disposition == CandidateDisposition.SUPPORTED:
                metrics.n_supported += 1
                if result.evidence_classification in ("direct_source", "rule_derived"):
                    metrics.evidence_backed_supported += 1
            elif result.disposition == CandidateDisposition.REJECTED:
                metrics.n_rejected += 1
            elif result.disposition == CandidateDisposition.AMBIGUOUS:
                metrics.n_ambiguous += 1
            else:
                metrics.n_unverifiable += 1
        return metrics

    def supported_fraction(self) -> float:
        return self.n_supported / self.n_candidates if self.n_candidates else 0.0

    def rejected_fraction(self) -> float:
        return self.n_rejected / self.n_candidates if self.n_candidates else 0.0

    def evidence_backed_fraction(self) -> float:
        """Fraction of SUPPORTED dispositions that identify evidence or rules (test 15)."""
        return self.evidence_backed_supported / self.n_supported if self.n_supported else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "n_candidates": self.n_candidates, "n_supported": self.n_supported,
            "n_rejected": self.n_rejected, "n_ambiguous": self.n_ambiguous,
            "n_unverifiable": self.n_unverifiable,
            "evidence_backed_fraction": round(self.evidence_backed_fraction(), 4),
        }


def false_positive_rate(
    results: Sequence[VerificationResult],
    adjudicated: Mapping[tuple[str, str, str], bool],
) -> float | None:
    """False-positive rate among promoted candidates (semantic safety).

    ``adjudicated`` maps candidate triples to the ground-truth decision produced by the
    authorized workflow; only SUPPORTED candidates the workflow later refuted count.
    """
    promoted = [
        result for result in results
        if result.disposition == CandidateDisposition.SUPPORTED
    ]
    checked = [result for result in promoted if result.triple in adjudicated]
    if not checked:
        return None
    false = sum(1 for result in checked if adjudicated[result.triple] is False)
    return false / len(checked)


@dataclass
class MetricSnapshot:
    """One named metrics group; subgroups stay separately visible."""

    name: str
    metrics: dict[str, float] = field(default_factory=dict)
    subgroups: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "subgroups": {
                group: {k: round(v, 6) for k, v in values.items()}
                for group, values in self.subgroups.items()
            },
        }


def relation_type_subgroup_metrics(
    results: Sequence[VerificationResult],
) -> dict[str, dict[str, float]]:
    """Semantic-safety metrics split per relation type (never one aggregate only)."""
    grouped: dict[str, list[VerificationResult]] = {}
    for result in results:
        grouped.setdefault(result.triple[1], []).append(result)
    subgroups: dict[str, dict[str, float]] = {}
    for relation_type, items in sorted(grouped.items()):
        metrics = SemanticSafetyMetrics.from_results(items)
        subgroups[relation_type] = {
            "n_candidates": float(metrics.n_candidates),
            "supported_fraction": metrics.supported_fraction(),
            "rejected_fraction": metrics.rejected_fraction(),
            "evidence_backed_fraction": metrics.evidence_backed_fraction(),
        }
    return subgroups


def graph_prediction_metrics(
    ranked: Sequence[GraphCandidate],
    relevant: frozenset,
    *,
    ranks: Sequence[int] | None = None,
) -> dict[str, float]:
    """MRR / Hits@K / Precision@K / Recall@K for one ranked candidate list."""
    metrics = {
        "precision_at_10": precision_at_k(ranked, relevant, 10),
        "recall_at_10": recall_at_k(ranked, relevant, 10),
    }
    if ranks:
        metrics.update({
            "mrr": mean_reciprocal_rank(ranks),
            "hits_at_1": hits_at_k(ranks, 1),
            "hits_at_3": hits_at_k(ranks, 3),
            "hits_at_10": hits_at_k(ranks, 10),
        })
    return metrics


__all__ = [
    "MetricSnapshot", "SemanticSafetyMetrics",
    "expected_calibration_error", "false_positive_rate",
    "graph_prediction_metrics", "hits_at_k", "mean_reciprocal_rank",
    "precision_at_k", "recall_at_k", "relation_type_subgroup_metrics",
]
