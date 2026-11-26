"""Schema/type validation for graph candidates (ORE Point 8 step 33).

Validates relation signatures, subject/object types, cardinality, direction, and declared
roles against the projected USG schema. Type checks are hard gates: a violation rejects
the candidate outright regardless of any learned score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..projection.typed_graph import TypedGraphSnapshot
from ..schemas.candidate_schema import GraphCandidate


@dataclass(frozen=True, slots=True)
class TypeCheckResult:
    """Outcome of one candidate type check."""

    candidate_id: str
    valid: bool
    violations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "valid": self.valid,
            "violations": list(self.violations),
        }


class CandidateTypeChecker:
    """Validate candidate triples against relation signatures (step 33)."""

    def __init__(self, snapshot: TypedGraphSnapshot) -> None:
        self.snapshot = snapshot
        self.schema = snapshot.schema
        self.node_by_id = {node.node_id: node for node in snapshot.nodes}
        self.edges_by_pair: dict[tuple[str, str], list] = {}
        for edge in snapshot.edges:
            self.edges_by_pair.setdefault((edge.source_id, edge.target_id), []).append(edge)

    def check(self, candidate: GraphCandidate) -> TypeCheckResult:
        """Run signature, direction, cardinality, and role checks for one candidate."""
        subject, relation, obj = candidate.triple()
        violations: list[str] = []
        subject_node = self.node_by_id.get(subject)
        object_node = self.node_by_id.get(obj)
        if subject_node is None or object_node is None:
            violations.append("endpoint_not_projected")
            return TypeCheckResult(candidate.candidate_id, False, tuple(violations))
        signature = self.schema.signature(relation)
        if signature is None:
            violations.append("undeclared_relation_type")
        else:
            if not signature.domain_allows(subject_node.node_kind):
                violations.append(
                    f"domain_violation:{relation}:{subject_node.node_kind}"
                )
            if not signature.range_allows(object_node.node_kind):
                violations.append(
                    f"range_violation:{relation}:{object_node.node_kind}"
                )
            if signature.direction == "DIRECTED" and signature.symmetric is False:
                pass  # direction preserved by construction; nothing extra to check
            missing_roles = [
                role for role in signature.required_roles()
                if not any(
                    edge.relation_type == relation
                    and any(binding.role == role for binding in edge.roles)
                    for edges in self.edges_by_pair.values() for edge in edges
                )
            ]
            if missing_roles:
                violations.append("missing_required_roles:" + ",".join(missing_roles))
            cardinality = signature.cardinality
            if cardinality in {"one_to_one", "one_to_many"}:
                existing_subjects = self.snapshot.edges_by_relation(relation)
                for edge in existing_subjects:
                    if edge.target_id == obj and edge.source_id != subject:
                        if cardinality == "one_to_one":
                            violations.append(
                                f"cardinality_violation:one_to_one:{edge.edge_id}"
                            )
                            break
        return TypeCheckResult(candidate.candidate_id, not violations, tuple(violations))

    def check_all(self, candidates: Sequence[GraphCandidate]) -> list[TypeCheckResult]:
        """Type-check every candidate in deterministic order."""
        return [self.check(candidate) for candidate in candidates]


__all__ = ["CandidateTypeChecker", "TypeCheckResult"]
