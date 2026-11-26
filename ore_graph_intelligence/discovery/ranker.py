"""Task-utility ranking with hard semantic rejection (ORE Point 8 step 30).

S(c) = w1 R(c) + w2 E(c) + w3 T(c) + w4 N(c) - w5 C(c) with relevance, evidence, type
compatibility, novelty, and cost normalized into [0, 1]. Hard semantic violations reject
outright - they are never converted into a small score penalty. Component weights are
explicit and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..projection.typed_graph import TypedGraphSnapshot
from ..schemas.candidate_schema import (
    CandidateStatus,
    ComponentScores,
    GraphCandidate,
    UtilityComponents,
)

DEFAULT_UTILITY_WEIGHTS: dict[str, float] = {
    "relevance": 0.30,
    "evidence": 0.25,
    "type_compatibility": 0.20,
    "novelty": 0.15,
    "cost": 0.10,
}


@dataclass(frozen=True, slots=True)
class NoveltyContext:
    """Graph facts consulted to compute novelty and contradictions (step 29)."""

    existing_triples: frozenset[tuple[str, str, str]]
    inverse_known: frozenset[tuple[str, str, str]]


class UtilityRanker:
    """Rank scored candidates by task utility; hard violations are already rejected."""

    def __init__(
        self,
        snapshot: TypedGraphSnapshot,
        *,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.weights = dict(DEFAULT_UTILITY_WEIGHTS)
        if weights:
            unknown = set(weights) - set(self.weights)
            if unknown:
                raise ValueError(f"unknown utility weights: {sorted(unknown)}")
            self.weights.update({name: float(value) for name, value in weights.items()})
        if any(value < 0 for value in self.weights.values()):
            raise ValueError("utility weights cannot be negative")
        self.existing = snapshot.existing_triples()

    def utility_components(self, candidate: GraphCandidate) -> UtilityComponents:
        """Compute normalized R/E/T/N/C components for one scored candidate."""
        subject, relation, obj = candidate.triple()
        relevance = self._relevance(candidate.component_scores)
        evidence = self._evidence(candidate)
        type_compatibility = self._type_compatibility(candidate)
        novelty = self._novelty(candidate)
        cost = self._cost(candidate)
        return UtilityComponents(
            relevance=relevance,
            evidence=evidence,
            type_compatibility=type_compatibility,
            novelty=novelty,
            cost=cost,
        )

    def rank(self, candidates: Sequence[GraphCandidate]) -> list[GraphCandidate]:
        """Attach utility components and total utility, returning candidates sorted."""
        for candidate in candidates:
            if candidate.status == CandidateStatus.REJECTED_PRE_RANK:
                continue
            components = self.utility_components(candidate)
            candidate.utility_components = components
            candidate.utility = components.weighted(self.weights)
            candidate.status = CandidateStatus.RANKED
        ranked = sorted(
            (item for item in candidates if item.utility is not None),
            key=lambda item: (-item.utility, item.candidate_id),
        )
        return ranked

    # ------------------------------------------------------------------ components

    def _relevance(self, scores: ComponentScores) -> float:
        """Relevance: mean of available normalized component scores, clipped to [0, 1]."""
        available = [value for value in (
            scores.embedding_score, scores.gnn_score, scores.baseline_score,
        ) if value is not None]
        if not available:
            return 0.0
        return float(min(max(sum(available) / len(available), 0.0), 1.0))

    def _evidence(self, candidate: GraphCandidate) -> float:
        """Evidence: normalized count of evidence identifiers supporting the candidate."""
        count = len(candidate.evidence_ids) + len(candidate.supporting_edge_ids)
        if count == 0:
            return 0.0
        return float(min(count / 3.0, 1.0))

    def _type_compatibility(self, candidate: GraphCandidate) -> float:
        """Type compatibility: 1.0 when the schema signature fully admits the pair."""
        if candidate.rejection_reasons:
            return 0.0
        signature = self.snapshot.schema.signature(candidate.relation_type)
        if signature is None:
            return 0.5  # undeclared relation type: neither violated nor fully compatible
        node_by_id = {node.node_id: node for node in self.snapshot.nodes}
        subject_kind = node_by_id.get(candidate.subject_id)
        object_kind = node_by_id.get(candidate.object_id)
        compatible = (
            signature.domain_allows(subject_kind.node_kind if subject_kind else "")
            and signature.range_allows(object_kind.node_kind if object_kind else "")
        )
        return 1.0 if compatible else 0.0

    def _novelty(self, candidate: GraphCandidate) -> float:
        """Novelty: 1.0 unless the fact (or its known inverse) already exists."""
        subject, relation, obj = candidate.triple()
        if (subject, relation, obj) in self.existing:
            return 0.0
        signature = self.snapshot.schema.signature(relation)
        if signature is not None and signature.inverse_of is not None:
            if (obj, signature.inverse_of, subject) in self.existing:
                return 0.0
        return 1.0

    def _cost(self, candidate: GraphCandidate) -> float:
        """Cost: normalized retrieval cost from the explanation neighborhood size."""
        path_cost = sum(len(path) for path in candidate.explanation_paths)
        if path_cost == 0:
            return 1.0
        return float(min(path_cost / 8.0, 1.0))


__all__ = ["DEFAULT_UTILITY_WEIGHTS", "NoveltyContext", "UtilityRanker"]
