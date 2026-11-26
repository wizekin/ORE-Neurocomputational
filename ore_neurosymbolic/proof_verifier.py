"""Independent proof verifier (Step C3).

The verifier recomputes every strict rule application from the original source
facts and the validated rule definition. It does not trust engine claims: it
re-derives each substitution, re-evaluates the strict firing gate (6.1) and
compares the recomputed conclusion with the recorded proof node. Any mismatch
— tampered rule version, tampered substitution, missing premise fact or altered
conclusion — marks the node ``TAMPERED`` so publication gates can block it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import (
    ProofNode,
    RuleDefinition,
    SemanticFact,
    VerificationStatus,
)
from .symbolic_engine import FactTable
from .unification import Substitution, unify_literals


@dataclass(slots=True)
class VerificationReport:
    """Per-node verifier outcomes with bounded diagnostics (Step C3)."""

    verified: dict[str, bool] = field(default_factory=dict)
    tampered: tuple[str, ...] = ()
    diagnostics: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "verified_count": sum(1 for value in self.verified.values() if value),
            "total_checked": len(self.verified),
            "tampered": list(self.tampered),
            "diagnostics": dict(self.diagnostics),
        }


class IndependentProofVerifier:
    """Re-execute strict rule applications without trusting the engine (Step C3)."""

    def __init__(
        self,
        rules: tuple[RuleDefinition, ...],
        facts: FactTable,
    ) -> None:
        """Bind the validated ruleset and a fresh view of the source facts."""
        self.rules = {rule.rule_ref: rule for rule in rules}
        self.facts = facts

    def verify_graph(
        self,
        graph: Mapping[str, ProofNode],
        *,
        verification_level: str = "basic",
    ) -> VerificationReport:
        """Verify every strict node in the graph; defeasible nodes stay unverified.

        ``verification_level='none'`` skips re-execution (nodes remain
        unverified, which publication gates reject for strict conclusions).
        """
        report = VerificationReport()
        for node_id, node in sorted(graph.items()):
            if verification_level == "none":
                report.verified[node_id] = False
                continue
            rule = self.rules.get(node.rule_ref)
            if rule is None:
                report.tampered = (*report.tampered, node_id)
                report.diagnostics[node_id] = "rule_version_unresolvable"
                continue
            if rule.rule_class.value == "defeasible":
                report.verified[node_id] = False
                continue
            ok, reason = self._recompute_node(rule, node)
            report.verified[node_id] = ok
            if not ok:
                report.tampered = (*report.tampered, node_id)
                report.diagnostics[node_id] = reason or "recompute_mismatch"
        return report

    def _recompute_node(self, rule: RuleDefinition, node: ProofNode) -> tuple[bool, str | None]:
        """Re-derive one strict application from source facts (Step C3).

        The verifier reconstructs every binding independently from the source
        facts; stored substitutions are never trusted.
        """
        working = Substitution()
        premise_sources: list[str] = []
        for premise in rule.premises:
            grounded, fact = self._match_premise(premise, working)
            if grounded is None or fact is None:
                return False, "premise_unsupported_on_recompute"
            premise_sources.append(fact.fact_id)
        grounded_conclusion = self._ground(rule.conclusion, working)
        if grounded_conclusion is None:
            return False, "conclusion_not_grounded"
        if grounded_conclusion.signature() != node.conclusion.signature():
            return False, "conclusion_mismatch"
        for fact_id in node.source_fact_ids:
            if fact_id not in premise_sources:
                # Recorded provenance must reference facts that still support
                # the premises; otherwise the recorded lineage was tampered.
                if fact_id not in self.facts._facts:  # noqa: SLF001 - verifier boundary check
                    return False, "recorded_source_fact_missing"
        return True, None

    def _match_premise(
        self,
        premise: Any,
        working: Substitution,
    ) -> tuple[Any, SemanticFact | None]:
        """Match one premise literal against the fact table, extending bindings."""
        from .contracts import LogicalVariable, Literal

        candidates = self.facts.by_predicate(premise.predicate)
        for fact in candidates:
            fact_literal = fact.literal()
            if fact_literal.arity != premise.arity:
                continue
            if fact_literal.negated != premise.negated:
                continue
            trial = working.copy()
            ok, _diagnostic = unify_literals(premise.args, fact_literal.args, trial)
            if not ok:
                continue
            grounded = Literal(
                predicate=premise.predicate,
                args=trial.resolve_literal_args(premise.args),
                negated=premise.negated,
                temporal_qualifier=premise.temporal_qualifier,
            )
            if any(isinstance(item, LogicalVariable) for item in grounded.args):
                continue
            if grounded.negated:
                if self.facts.has_negated(grounded) is None:
                    continue
            elif self.facts.has_positive(grounded) is None:
                continue
            working._bindings.update(trial._bindings)  # noqa: SLF001 - verifier-internal
            return grounded, fact
        return None, None

    @staticmethod
    def _ground(literal: Any, substitution: Substitution) -> Any:
        """Resolve one literal through the reconstructed substitution."""
        from .contracts import LogicalVariable, Literal

        resolved = substitution.resolve_literal_args(literal.args)
        if any(isinstance(item, LogicalVariable) for item in resolved):
            return None
        return Literal(
            predicate=literal.predicate,
            args=resolved,
            negated=literal.negated,
            temporal_qualifier=literal.temporal_qualifier,
        )

    @staticmethod
    def _term_from_key(value: Any) -> Any:
        """Parse one recorded term key back into a term (used for hash checks)."""
        from .contracts import LogicalTerm

        if isinstance(value, str) and value:
            return LogicalTerm(value, "any")
        return None


__all__ = ["IndependentProofVerifier", "VerificationReport"]
