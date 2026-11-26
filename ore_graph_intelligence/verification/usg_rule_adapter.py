"""USG rule adapter: logical compatibility checks (ORE Point 8 step 34).

Checks candidates against USG rules and formal constraints derived from the projection:
contradictions, polarity errors, scope changes, modality loss, and
conditional-to-asserted leakage. Rules are declarative and recorded per check so every
verification decision names the rules that produced it.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..projection.typed_graph import TypedEdge, TypedGraphSnapshot
from ..schemas.candidate_schema import GraphCandidate


@dataclass(frozen=True, slots=True)
class RuleViolation:
    """One logical-compatibility violation reported by a named rule."""

    rule: str
    message: str
    edge_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {"rule": self.rule, "message": self.message, "edge_ids": list(self.edge_ids)}


class USGRuleAdapter:
    """Evaluate USG logical rules over one projection (step 34)."""

    def __init__(self, snapshot: TypedGraphSnapshot) -> None:
        self.snapshot = snapshot
        self.existing = snapshot.existing_triples()
        self.edges_by_pair: dict[tuple[str, str], list[TypedEdge]] = {}
        for edge in snapshot.edges:
            self.edges_by_pair.setdefault((edge.source_id, edge.target_id), []).append(edge)
        self.edges_by_relation: dict[str, list[TypedEdge]] = {}
        for edge in snapshot.edges:
            self.edges_by_relation.setdefault(edge.relation_type, []).append(edge)

    def check(self, candidate: GraphCandidate) -> list[RuleViolation]:
        """Run every logical compatibility rule for one candidate."""
        violations: list[RuleViolation] = []
        subject, relation, obj = candidate.triple()
        for rule in (
            self._rule_polarity_contradiction,
            self._rule_scope_change,
            self._rule_modality_loss,
            self._rule_conditional_to_asserted,
            self._rule_inverse_consistency,
            self._rule_temporal_conflict,
        ):
            violations.extend(rule(candidate))
        return violations

    # ------------------------------------------------------------------ rules

    def _rule_polarity_contradiction(
        self, candidate: GraphCandidate,
    ) -> list[RuleViolation]:
        """A projected negative-polarity edge contradicts an asserted positive candidate."""
        subject, relation, obj = candidate.triple()
        violations: list[RuleViolation] = []
        for edge in self.edges_by_pair.get((subject, obj), ()):
            if edge.relation_type == relation and edge.qualifiers.polarity == "negative":
                violations.append(RuleViolation(
                    "polarity_contradiction",
                    f"projected edge {edge.edge_id} asserts {relation} with negative polarity",
                    (edge.edge_id,),
                ))
        return violations

    def _rule_scope_change(
        self, candidate: GraphCandidate,
    ) -> list[RuleViolation]:
        """Candidate widens a temporal or attribution scope of an existing edge."""
        subject, relation, obj = candidate.triple()
        violations: list[RuleViolation] = []
        for edge in self.edges_by_pair.get((subject, obj), ()):
            if edge.relation_type != relation:
                continue
            if edge.qualifiers.temporal_scope is not None:
                violations.append(RuleViolation(
                    "scope_change",
                    f"projected edge {edge.edge_id} restricts temporal scope to "
                    f"'{edge.qualifiers.temporal_scope}'",
                    (edge.edge_id,),
                ))
        return violations

    def _rule_modality_loss(
        self, candidate: GraphCandidate,
    ) -> list[RuleViolation]:
        """Asserting an edge that exists only under a non-factive modality."""
        subject, relation, obj = candidate.triple()
        violations: list[RuleViolation] = []
        for edge in self.edges_by_pair.get((subject, obj), ()):
            if edge.relation_type != relation:
                continue
            modality = edge.qualifiers.modality
            if modality is not None and modality not in {"factive", "asserted"}:
                violations.append(RuleViolation(
                    "modality_loss",
                    f"projected edge {edge.edge_id} only supports modality '{modality}'",
                    (edge.edge_id,),
                ))
        return violations

    def _rule_conditional_to_asserted(
        self, candidate: GraphCandidate,
    ) -> list[RuleViolation]:
        """Candidate asserts unconditionally what the projection holds only conditionally."""
        subject, relation, obj = candidate.triple()
        violations: list[RuleViolation] = []
        for edge in self.edges_by_pair.get((subject, obj), ()):
            if edge.relation_type != relation and edge.qualifiers.condition is not None:
                continue
            if edge.relation_type == relation and edge.qualifiers.condition is not None:
                violations.append(RuleViolation(
                    "conditional_to_asserted",
                    f"projected edge {edge.edge_id} holds only under condition "
                    f"'{edge.qualifiers.condition}'",
                    (edge.edge_id,),
                ))
        return violations

    def _rule_inverse_consistency(
        self, candidate: GraphCandidate,
    ) -> list[RuleViolation]:
        """Declared inverse present in the opposite direction with conflicting content."""
        subject, relation, obj = candidate.triple()
        signature = self.snapshot.schema.signature(relation)
        if signature is None or signature.inverse_of is None:
            return []
        violations: list[RuleViolation] = []
        for edge in self.edges_by_pair.get((obj, subject), ()):
            if edge.relation_type == signature.inverse_of and edge.qualifiers.polarity == "negative":
                violations.append(RuleViolation(
                    "inverse_consistency",
                    f"inverse edge {edge.edge_id} carries negative polarity",
                    (edge.edge_id,),
                ))
        return violations

    def _rule_temporal_conflict(
        self, candidate: GraphCandidate,
    ) -> list[RuleViolation]:
        """Same predicate and endpoints under different temporal scopes."""
        subject, relation, obj = candidate.triple()
        signature = self.snapshot.schema.signature(relation)
        if signature is None or not signature.transitive:
            return []
        violations: list[RuleViolation] = []
        for edge in self.edges_by_pair.get((subject, obj), ()):
            if edge.relation_type != relation:
                continue
            existing_scope = edge.qualifiers.temporal_scope
            candidate_scope = "asserted_current"
            if existing_scope is not None and existing_scope != candidate_scope:
                violations.append(RuleViolation(
                    "temporal_conflict",
                    f"projected edge {edge.edge_id} uses temporal scope '{existing_scope}'",
                    (edge.edge_id,),
                ))
        return violations


__all__ = ["RuleViolation", "USGRuleAdapter"]
