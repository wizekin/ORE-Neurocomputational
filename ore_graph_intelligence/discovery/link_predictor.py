"""Complementary candidate scoring and deterministic baselines (ORE Point 8 steps 13, 27).

Learned scores from embedding and GNN models are combined with deterministic typed
baselines. Component scores are normalized on validation data and *preserved separately*:
an uncalibrated average is never called a probability. Learned models are always compared
against deterministic baselines before being trusted for guidance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..gnn.inference import DeterministicGraphBaselines
from ..projection.typed_graph import TypedGraphSnapshot
from ..schemas.candidate_schema import ComponentScores, GraphCandidate


@dataclass(frozen=True, slots=True)
class ScoreNormalization:
    """Validation-derived per-component affine normalization (mean/std by component)."""

    component_statistics: Mapping[str, tuple[float, float]]

    def normalize(self, component: str, value: float) -> float:
        mean, std = self.component_statistics.get(component, (0.0, 1.0))
        if std == 0.0:
            std = 1.0
        return float((value - mean) / std)

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            component: {"mean": mean, "std": std}
            for component, (mean, std) in self.component_statistics.items()
        }


class ComplementaryScorer:
    """Score candidates with deterministic, embedding, and GNN components (step 27)."""

    def __init__(
        self,
        snapshot: TypedGraphSnapshot,
        *,
        embedding_scorer=None,
        gnn_scorer=None,
        normalization: ScoreNormalization | None = None,
    ) -> None:
        """Bind one snapshot to optional learned scorers; baselines are always active."""
        self.snapshot = snapshot
        self.embedding_scorer = embedding_scorer
        self.gnn_scorer = gnn_scorer
        self.baselines = DeterministicGraphBaselines(
            [(edge.source_id, edge.relation_type, edge.target_id) for edge in snapshot.edges]
        )
        self.normalization = normalization

    def score(self, candidate: GraphCandidate) -> ComponentScores:
        """Fill component scores for one candidate; every score stays a raw ranking value."""
        subject, relation, obj = candidate.triple()
        baseline = float(
            self.baselines.common_neighbor_score(subject, obj)
            + self.baselines.relation_frequency_score(relation)
        )
        embedding = None
        if self.embedding_scorer is not None:
            try:
                embedding = self.embedding_scorer.raw_score(subject, relation, obj)
            except KeyError:
                embedding = None
        gnn = None
        if self.gnn_scorer is not None:
            try:
                gnn = self.gnn_scorer(subject, relation, obj)
            except KeyError:
                gnn = None
        components = ComponentScores(
            embedding_score=embedding,
            gnn_score=gnn,
            baseline_score=baseline,
        )
        return self._normalize(components)

    def _normalize(self, components: ComponentScores) -> ComponentScores:
        """Normalize with validation statistics when available; preserve components."""
        if self.normalization is None:
            return components
        return ComponentScores(
            embedding_score=(
                self.normalization.normalize("embedding", components.embedding_score)
                if components.embedding_score is not None else None
            ),
            gnn_score=(
                self.normalization.normalize("gnn", components.gnn_score)
                if components.gnn_score is not None else None
            ),
            baseline_score=self.normalization.normalize("baseline", components.baseline_score),
        )

    def fit_normalization(
        self,
        validation_triples: Sequence[tuple[str, str, str]],
    ) -> ScoreNormalization:
        """Fit per-component mean/std on held-out validation triples (step 27)."""
        statistics: dict[str, tuple[float, float]] = {}
        baseline_values = [
            self.baselines.common_neighbor_score(head, tail)
            + self.baselines.relation_frequency_score(relation)
            for head, relation, tail in validation_triples
        ] or [0.0]
        statistics["baseline"] = (
            float(np.mean(baseline_values)), float(np.std(baseline_values)),
        )
        if self.embedding_scorer is not None:
            embedding_values = []
            for head, relation, tail in validation_triples:
                try:
                    embedding_values.append(self.embedding_scorer.raw_score(head, relation, tail))
                except KeyError:
                    continue
            if embedding_values:
                statistics["embedding"] = (
                    float(np.mean(embedding_values)), float(np.std(embedding_values)),
                )
        if self.gnn_scorer is not None:
            gnn_values = []
            for head, relation, tail in validation_triples:
                try:
                    gnn_values.append(self.gnn_scorer(head, relation, tail))
                except KeyError:
                    continue
            if gnn_values:
                statistics["gnn"] = (
                    float(np.mean(gnn_values)), float(np.std(gnn_values)),
                )
        self.normalization = ScoreNormalization(statistics)
        return self.normalization


class LinkPredictor:
    """Attach complementary scores to generated candidates (steps 27-28 helper)."""

    def __init__(self, scorer: ComplementaryScorer) -> None:
        self.scorer = scorer

    def score_all(self, candidates: Sequence[GraphCandidate]) -> list[GraphCandidate]:
        """Score every candidate in deterministic order, preserving component scores."""
        scored: list[GraphCandidate] = []
        for candidate in candidates:
            candidate.component_scores = self.scorer.score(candidate)
            scored.append(candidate)
        return scored


__all__ = ["ComplementaryScorer", "LinkPredictor", "ScoreNormalization"]
