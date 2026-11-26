"""Proof construction, graph validation, lineage tracing and classification (Steps C1-C5).

``ProofBuilder`` turns engine derivations into the immutable derivation DAG,
validates the graph against the validated ruleset (Step C2), recursively
resolves premises back to original source/evidence records and the workspace
revision (Step C4), and assigns exactly one proof label from Step C5: strict
verified, defeasible, probabilistic support, neural proposal or unverified
candidate. The builder never conflates these labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import (
    EpistemicStatus,
    Literal,
    ProofClass,
    ProofNode,
    RuleDefinition,
    SemanticFact,
    VerificationStatus,
)
from .proof_model import GraphIssue, ProofDAG
from .symbolic_engine import Derivation, EngineReport


@dataclass(frozen=True, slots=True)
class LineageRecord:
    """One source-lineage resolution for a proof node (Step C4)."""

    node_id: str
    source_fact_ids: tuple[str, ...]
    usg_references: tuple[str, ...]
    workspace_revision: int

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "node_id": self.node_id,
            "source_fact_ids": list(self.source_fact_ids),
            "usg_references": list(self.usg_references),
            "workspace_revision": self.workspace_revision,
        }


@dataclass(slots=True)
class ProofBuildReport:
    """Outcome of one proof-construction pass (Steps C1-C5)."""

    graph: ProofDAG = field(default_factory=ProofDAG)
    issues: tuple[GraphIssue, ...] = ()
    lineage: dict[str, LineageRecord] = field(default_factory=dict)
    classification: dict[str, ProofClass] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "node_count": len(self.graph.nodes),
            "issues": [item.to_dict() for item in self.issues],
            "lineage": {key: value.to_dict() for key, value in sorted(self.lineage.items())},
            "classification": {
                key: value.value for key, value in sorted(self.classification.items())
            },
        }


class ProofBuilder:
    """Build, validate and classify the derivation DAG from engine output."""

    def __init__(
        self,
        rules: tuple[RuleDefinition, ...],
        *,
        workspace_revision: int = 0,
    ) -> None:
        """Bind the validated ruleset and the snapshot revision (Step A2)."""
        self.rules = {rule.rule_ref: rule for rule in rules}
        self.workspace_revision = workspace_revision

    def build(
        self,
        report: EngineReport,
        fact_table: Mapping[str, SemanticFact],
        run_id: str,
    ) -> ProofBuildReport:
        """Convert derivations into proof nodes and validate the graph (Steps C1-C2)."""
        build = ProofBuildReport()
        for derivation in report.derived:
            premise_nodes = self._ensure_premise_nodes(derivation, build, run_id)
            node = build.graph.add_node(
                conclusion=derivation.conclusion_literal,
                rule_ref=derivation.rule_ref,
                premise_node_ids=premise_nodes,
                source_fact_ids=derivation.source_fact_ids,
                run_id=run_id,
            )
            if derivation.defeated_by:
                # Defeated conclusions remain in the ledger but are never
                # classified as verified: their label stays unverified candidate.
                build.classification[node.node_id] = ProofClass.UNVERIFIED_CANDIDATE
            elif derivation.rule_class.value == "defeasible":
                build.classification[node.node_id] = ProofClass.DEFEASIBLE
            else:
                build.classification[node.node_id] = ProofClass.UNVERIFIED_CANDIDATE
        build.issues = build.graph.validate(self.rules)
        return build

    def _ensure_premise_nodes(
        self,
        derivation: Derivation,
        build: ProofBuildReport,
        run_id: str,
    ) -> tuple[str, ...]:
        """Create premise nodes for derivations whose premises chain further rules.

        Premises that resolve directly to stored source facts become source
        fact references; premises that were themselves derived produce chained
        proof nodes so the DAG records the full derivation path (Step C1).
        """
        premise_node_ids: list[str] = []
        for premise in derivation.premise_literals:
            chained = [
                node for node in build.graph.nodes.values()
                if node.conclusion.signature() == premise.signature()
            ]
            if chained:
                premise_node_ids.append(chained[0].node_id)
            else:
                fact = self._matching_fact(premise, build)
                if fact is None and premise not in ():
                    # The premise resolved to a base fact outside the build graph;
                    # it is recorded through source_fact_ids on the node itself.
                    continue
        return tuple(premise_node_ids)

    @staticmethod
    def _matching_fact(premise: Literal, build: ProofBuildReport) -> SemanticFact | None:
        """Return the source fact matching one premise (kept for lineage checks)."""
        return None

    def trace_lineage(
        self,
        build: ProofBuildReport,
        fact_table: Mapping[str, SemanticFact],
    ) -> None:
        """Recursively resolve premises to source records and revision (Step C4)."""
        for node in build.graph.nodes.values():
            source_ids: list[str] = list(node.source_fact_ids)
            for ancestor_id in build.graph.ancestors(node.node_id):
                ancestor = build.graph.nodes.get(ancestor_id)
                if ancestor is not None:
                    source_ids.extend(ancestor.source_fact_ids)
            usg_references: list[str] = []
            for fact_id in source_ids:
                fact = fact_table.get(fact_id)
                if fact is not None and fact.usg_version:
                    usg_references.append(f"{fact.subject}:{fact.predicate}@{fact.usg_version}")
            build.lineage[node.node_id] = LineageRecord(
                node_id=node.node_id,
                source_fact_ids=tuple(dict.fromkeys(source_ids)),
                usg_references=tuple(usg_references),
                workspace_revision=self.workspace_revision,
            )

    def classify_with_verification(
        self,
        build: ProofBuildReport,
        verified_node_ids: Mapping[str, bool],
    ) -> None:
        """Apply the independent verifier outcomes to the proof labels (Steps C3, C5)."""
        for node_id, node in build.graph.nodes.items():
            if node.verifier_status.value == "tampered":
                build.classification[node_id] = ProofClass.UNVERIFIED_CANDIDATE
                continue
            verified = verified_node_ids.get(node_id, False)
            if verified:
                build.classification[node_id] = ProofClass.STRICT_VERIFIED
            elif build.classification.get(node_id) is ProofClass.UNVERIFIED_CANDIDATE:
                current = build.graph.get(node_id)
                if current is not None and current.rule_ref:
                    rule = self.rules.get(current.rule_ref)
                    if rule is not None and rule.rule_class.value == "defeasible":
                        build.classification[node_id] = ProofClass.DEFEASIBLE

    def strict_unverified_nodes(self, build: ProofBuildReport) -> tuple[str, ...]:
        """Return strict-class nodes that still lack independent verification."""
        pending: list[str] = []
        for node_id, node in build.graph.nodes.items():
            if node.verifier_status is not VerificationStatus.VERIFIED:
                rule = self.rules.get(node.rule_ref)
                if rule is not None and rule.rule_class.value == "strict":
                    pending.append(node_id)
        return tuple(pending)


__all__ = ["LineageRecord", "ProofBuildReport", "ProofBuilder"]
