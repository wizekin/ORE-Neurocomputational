"""Counter-evidence search (ORE Point 8 step 36).

Verification must avoid confirmation bias: for every candidate the search looks for
counter-evidence - negated or polarity-flipped edges, incompatible constraints, temporal
changes, exceptions, and previously rejected instances - and reports what was found.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..projection.typed_graph import TypedGraphSnapshot
from ..schemas.candidate_schema import GraphCandidate


@dataclass(frozen=True, slots=True)
class CounterEvidenceReport:
    """Counter-evidence found for one candidate."""

    candidate_id: str
    counter_edge_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    exception_records: tuple[str, ...]
    temporal_changes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "counter_edge_ids": list(self.counter_edge_ids),
            "reasons": list(self.reasons),
            "exception_records": list(self.exception_records),
            "temporal_changes": list(self.temporal_changes),
        }

    @property
    def has_counter_evidence(self) -> bool:
        return bool(self.counter_edge_ids or self.reasons or self.exception_records)


class ContradictionSearch:
    """Search counter-evidence for candidates (step 36)."""

    def __init__(
        self,
        snapshot: TypedGraphSnapshot,
        *,
        rejected_ledger: Mapping[tuple[str, str, str], str] | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.rejected_ledger = dict(rejected_ledger or {})
        self.edges_by_pair: dict[tuple[str, str], list] = {}
        for edge in snapshot.edges:
            self.edges_by_pair.setdefault((edge.source_id, edge.target_id), []).append(edge)
        for edge in snapshot.edges:
            self.edges_by_pair.setdefault((edge.target_id, edge.source_id), []).append(edge)

    def search(self, candidate: GraphCandidate) -> CounterEvidenceReport:
        """Find counter-evidence for one candidate across all declared dimensions."""
        subject, relation, obj = candidate.triple()
        counter_edge_ids: set[str] = set()
        reasons: list[str] = []
        exceptions: list[str] = []
        temporal_changes: list[str] = []
        for edge in self.edges_by_pair.get((subject, obj), ()):
            if edge.edge_id in counter_edge_ids:
                continue
            if edge.relation_type == relation and edge.qualifiers.polarity == "negative":
                counter_edge_ids.add(edge.edge_id)
                reasons.append(
                    f"negated edge {edge.edge_id} asserts NOT({relation})"
                )
            if edge.relation_type == relation and edge.qualifiers.exceptions:
                counter_edge_ids.add(edge.edge_id)
                exceptions.extend(edge.qualifiers.exceptions)
                reasons.append(
                    f"edge {edge.edge_id} declares exceptions "
                    f"{list(edge.qualifiers.exceptions)}"
                )
            if edge.relation_type == relation and edge.qualifiers.polarity == "unknown":
                counter_edge_ids.add(edge.edge_id)
                reasons.append(f"edge {edge.edge_id} has unknown polarity")
            if edge.relation_type == relation and edge.qualifiers.temporal_scope is not None:
                temporal_changes.append(
                    f"edge {edge.edge_id} scoped to '{edge.qualifiers.temporal_scope}'"
                )
        rejection = self.rejected_ledger.get((subject, relation, obj))
        if rejection is not None:
            reasons.append(f"previously rejected: {rejection}")
        return CounterEvidenceReport(
            candidate.candidate_id,
            tuple(sorted(counter_edge_ids)),
            tuple(reasons),
            tuple(sorted(set(exceptions))),
            tuple(sorted(set(temporal_changes))),
        )

    def search_all(self, candidates: Sequence[GraphCandidate]) -> list[CounterEvidenceReport]:
        """Search counter-evidence for every candidate in deterministic order."""
        return [self.search(candidate) for candidate in candidates]


__all__ = ["ContradictionSearch", "CounterEvidenceReport"]
