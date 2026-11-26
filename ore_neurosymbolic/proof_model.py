"""Immutable, content-addressed derivation DAG (Steps C1-C2, Section 6.8).

The proof ledger is append-only: identical content addresses to the same node
(deduplication), no node can be mutated after insertion, and every stored
``content_hash`` implements ``h_v = H(conclusion || ruleVersion || h_p1 || ...
|| h_pn)``. Hashing detects tampering and pins references; it does not prove
logical validity — that is the independent verifier's job (Step C3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import (
    Literal,
    ProofClass,
    ProofNode,
    RuleDefinition,
    VerificationStatus,
    make_id,
)


@dataclass(frozen=True, slots=True)
class GraphIssue:
    """One structural problem found while validating the derivation graph."""

    code: str
    message: str
    node_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {"code": self.code, "message": self.message, "node_id": self.node_id}


@dataclass(slots=True)
class ProofDAG:
    """The immutable derivation DAG with content addressing and validation."""

    nodes: dict[str, ProofNode] = field(default_factory=dict)
    _hashes: dict[str, str] = field(default_factory=dict)
    _tampered: dict[str, str] = field(default_factory=dict)

    def add_node(
        self,
        conclusion: Literal,
        rule_ref: tuple[str, str],
        premise_node_ids: tuple[str, ...],
        source_fact_ids: tuple[str, ...],
        run_id: str,
        node_id: str | None = None,
    ) -> ProofNode:
        """Insert one node, reusing the identical content-addressed node (Step C1)."""
        premise_hashes = tuple(self._hashes.get(item, "") for item in premise_node_ids)
        draft = ProofNode(
            node_id=node_id or make_id("proof"),
            conclusion=conclusion,
            rule_ref=rule_ref,
            premise_node_ids=tuple(premise_node_ids),
            source_fact_ids=tuple(source_fact_ids),
            run_id=run_id,
        )
        content_hash = draft.compute_content_hash(premise_hashes)
        existing = self._hashes.get(content_hash)
        if existing is not None:
            return self.nodes[existing]
        node = ProofNode(
            node_id=draft.node_id,
            conclusion=conclusion,
            rule_ref=rule_ref,
            premise_node_ids=tuple(premise_node_ids),
            source_fact_ids=tuple(source_fact_ids),
            run_id=run_id,
            verifier_status=VerificationStatus.UNVERIFIED,
            content_hash=content_hash,
        )
        self.nodes[node.node_id] = node
        self._hashes[content_hash] = node.node_id
        self._hashes[node.node_id] = content_hash
        return node

    def get(self, node_id: str) -> ProofNode | None:
        """Return one node by identity."""
        return self.nodes.get(node_id)

    def nodes_for_conclusion(self, signature: str) -> tuple[ProofNode, ...]:
        """Return nodes whose conclusion signature matches."""
        return tuple(
            node for node in self.nodes.values()
            if node.conclusion.signature() == signature
        )

    def mark_verifier_status(self, node_id: str, status: VerificationStatus) -> bool:
        """Record the independent verifier outcome on one node (Step C3).

        This replaces the node with an equal-content node carrying the new
        verification status; the content hash deliberately stays stable so the
        verifier cannot silently alter the derivation it validates.
        """
        node = self.nodes.get(node_id)
        if node is None:
            return False
        replacement = ProofNode(
            node_id=node.node_id,
            conclusion=node.conclusion,
            rule_ref=node.rule_ref,
            premise_node_ids=node.premise_node_ids,
            source_fact_ids=node.source_fact_ids,
            run_id=node.run_id,
            verifier_status=status,
            content_hash=node.content_hash,
        )
        self.nodes[node_id] = replacement
        return True

    def validate(self, rules: Mapping[tuple[str, str], RuleDefinition]) -> tuple[GraphIssue, ...]:
        """Validate graph structure (Step C2): premises resolve, rule versions resolve.

        Every premise node ID must exist in the graph, every referenced rule
        ``(rule_id, version)`` must resolve in the validated ruleset, and the
        conclusion arity must match the rule's declared conclusion arity so all
        recorded substitutions are type-correct at the recorded rule version.
        """
        issues: list[GraphIssue] = []
        for node in self.nodes.values():
            for premise_id in node.premise_node_ids:
                if premise_id not in self.nodes:
                    issues.append(GraphIssue(
                        code="unresolved_premise",
                        message=f"premise node {premise_id!r} is not in the graph",
                        node_id=node.node_id,
                    ))
            rule = rules.get(node.rule_ref)
            if rule is None:
                issues.append(GraphIssue(
                    code="unresolved_rule_version",
                    message=(
                        f"rule {node.rule_ref[0]!r}@{node.rule_ref[1]!r} is not in the "
                        "validated ruleset"
                    ),
                    node_id=node.node_id,
                ))
            elif rule.conclusion.predicate != node.conclusion.predicate:
                issues.append(GraphIssue(
                    code="substitution_type_error",
                    message=(
                        f"conclusion predicate {node.conclusion.predicate!r} does not match "
                        f"rule conclusion {rule.conclusion.predicate!r}"
                    ),
                    node_id=node.node_id,
                ))
            elif rule.conclusion.arity != node.conclusion.arity:
                issues.append(GraphIssue(
                    code="substitution_type_error",
                    message="conclusion arity does not match the rule declaration",
                    node_id=node.node_id,
                ))
        if not issues and not self._integrity_ok():
            issues.append(GraphIssue(
                code="content_hash_mismatch",
                message="at least one stored node fails its content hash",
            ))
        return tuple(issues)

    def _integrity_ok(self) -> bool:
        """Recompute every node hash and detect any mutation (Section 6.8)."""
        for node_id, node in self.nodes.items():
            premise_hashes = tuple(
                self._hashes.get(item, "") for item in node.premise_node_ids
            )
            recomputed = node.compute_content_hash(premise_hashes)
            if recomputed != node.content_hash:
                self._tampered[node_id] = recomputed
                return False
        return True

    def verify_integrity(self) -> tuple[bool, tuple[str, ...]]:
        """Return overall integrity plus tampered node IDs (tamper detection)."""
        self._tampered.clear()
        for node_id, node in self.nodes.items():
            premise_hashes = tuple(
                self._hashes.get(item, "") for item in node.premise_node_ids
            )
            recomputed = node.compute_content_hash(premise_hashes)
            if recomputed != node.content_hash:
                self._tampered[node_id] = recomputed
        return not self._tampered, tuple(sorted(self._tampered))

    def classify(self, node_id: str) -> ProofClass:
        """Classify one node's proof label (Step C5) without conflating them."""
        node = self.nodes.get(node_id)
        if node is None:
            return ProofClass.UNVERIFIED_CANDIDATE
        if node.verifier_status is VerificationStatus.TAMPERED:
            return ProofClass.UNVERIFIED_CANDIDATE
        if node.verifier_status is VerificationStatus.VERIFIED:
            return ProofClass.STRICT_VERIFIED
        if node.verifier_status is VerificationStatus.FAILED:
            return ProofClass.UNVERIFIED_CANDIDATE
        return ProofClass.DEFEASIBLE

    def ancestors(self, node_id: str) -> tuple[str, ...]:
        """Return all transitive premise node IDs of one node (lineage closure)."""
        seen: list[str] = []
        stack = [node_id]
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            node = self.nodes.get(current)
            if node is None:
                continue
            for premise_id in node.premise_node_ids:
                if premise_id not in visited:
                    seen.append(premise_id)
                    stack.append(premise_id)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        """Return the detached JSON-compatible graph record."""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "integrity_ok": not self._tampered,
            "tampered_nodes": sorted(self._tampered),
        }


__all__ = ["GraphIssue", "ProofDAG"]
