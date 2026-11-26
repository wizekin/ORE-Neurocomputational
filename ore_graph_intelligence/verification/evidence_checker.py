"""Evidence retrieval and classification (ORE Point 8 step 35).

Distinguishes direct source support, rule-derived inference, model-only suggestion, and
absence of evidence. Every classification records its derivation chain so downstream
consumers can tell hypotheses from evidence-backed conclusions. Score alone is never
sufficient for the SUPPORTED disposition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..projection.typed_graph import TypedGraphSnapshot
from ..schemas.candidate_schema import EvidenceClassification, GraphCandidate


@dataclass(frozen=True, slots=True)
class EvidenceCheck:
    """One evidence classification with its derivation chain."""

    candidate_id: str
    classification: str
    evidence_ids: tuple[str, ...]
    supporting_edge_ids: tuple[str, ...]
    derivation: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "classification": self.classification,
            "evidence_ids": list(self.evidence_ids),
            "supporting_edge_ids": list(self.supporting_edge_ids),
            "derivation": list(self.derivation),
        }


class EvidenceChecker:
    """Classify the evidence available for one candidate (step 35)."""

    def __init__(self, snapshot: TypedGraphSnapshot) -> None:
        self.snapshot = snapshot
        self.existing = snapshot.existing_triples()
        self.edges_by_pair: dict[tuple[str, str], list] = {}
        for edge in snapshot.edges:
            self.edges_by_pair.setdefault((edge.source_id, edge.target_id), []).append(edge)
        self.signature_by_type = {
            signature.relation_type: signature
            for signature in snapshot.schema.relation_signatures
        }

    def check(self, candidate: GraphCandidate) -> EvidenceCheck:
        """Classify evidence as direct, rule-derived, model-only, or absent."""
        subject, relation, obj = candidate.triple()
        derivation: list[str] = []
        direct_edges: list = []
        for edge in self.edges_by_pair.get((subject, obj), ()):
            if edge.relation_type == relation:
                direct_edges.append(edge)
        if direct_edges:
            evidence_ids = tuple(sorted({
                evidence_id for edge in direct_edges for evidence_id in edge.evidence_ids
            }))
            derivation.append(
                f"direct:{len(direct_edges)} projected edge(s) with predicate {relation}"
            )
            return EvidenceCheck(
                candidate.candidate_id,
                EvidenceClassification.DIRECT_SOURCE,
                evidence_ids,
                tuple(sorted(edge.edge_id for edge in direct_edges)),
                tuple(derivation),
            )
        if self._rule_derived(candidate, derivation):
            return EvidenceCheck(
                candidate.candidate_id,
                EvidenceClassification.RULE_DERIVED,
                (),
                (),
                tuple(derivation),
            )
        if candidate.component_scores.available():
            derivation.append(
                "model_only: no projected support; learned scores only"
            )
            return EvidenceCheck(
                candidate.candidate_id,
                EvidenceClassification.MODEL_ONLY,
                (),
                (),
                tuple(derivation),
            )
        derivation.append("absent: no evidence reachable for this candidate")
        return EvidenceCheck(
            candidate.candidate_id,
            EvidenceClassification.ABSENT,
            (),
            (),
            tuple(derivation),
        )

    def _rule_derived(
        self, candidate: GraphCandidate, derivation: list[str],
    ) -> bool:
        """Derive support from declared transitive relations within a bounded depth."""
        subject, relation, obj = candidate.triple()
        signature = self.signature_by_type.get(relation)
        if signature is None or not signature.transitive:
            return False
        adjacency: dict[str, list] = {}
        for edge in self.snapshot.edges:
            if edge.relation_type == relation:
                adjacency.setdefault(edge.source_id, []).append(edge.target_id)
        frontier = {subject}
        depth = 0
        max_depth = 4
        while frontier and depth < max_depth:
            next_frontier = set()
            for node in frontier:
                for neighbor in adjacency.get(node, ()):
                    if neighbor == obj:
                        derivation.append(
                            f"rule_derived: transitive {relation} path from {subject} "
                            f"to {obj} within depth {depth + 1}"
                        )
                        return True
                    next_frontier.add(neighbor)
            frontier = next_frontier
            depth += 1
        return False

    def check_all(self, candidates: Sequence[GraphCandidate]) -> list[EvidenceCheck]:
        """Classify evidence for every candidate in deterministic order."""
        return [self.check(candidate) for candidate in candidates]


__all__ = ["EvidenceCheck", "EvidenceChecker"]
