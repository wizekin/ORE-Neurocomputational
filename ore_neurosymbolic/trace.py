"""Provenance trace reconstruction for the reasoning layer (Step H1 / Section 2).

The trace graph links every conclusion to its proof nodes, source facts,
rule versions, USG bundle references and the workspace revision it was bound
to (Section 2: every conclusion carries its exact premises, rule ID/version,
source lineage, workspace revision and inference-run ID). Trace completeness
is computed from conclusions whose lineage reaches a source root.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .contracts import sha256_json


@dataclass(frozen=True, slots=True)
class TraceNode:
    """One traceable entity in the reasoning trace graph."""

    node_id: str
    kind: str
    label: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TraceEdge:
    """One directed lineage edge in the trace graph."""

    source: str
    target: str
    relation: str


class NSRLTraceBuilder:
    """Build the auditable explanation graph for one reasoning run."""

    def __init__(self, run_id: str, workspace_revision: int) -> None:
        """Anchor the trace at the run root bound to one workspace revision."""
        self.run_id = run_id
        self.workspace_revision = workspace_revision
        self.nodes: dict[str, TraceNode] = {
            f"run:{run_id}": TraceNode(
                node_id=f"run:{run_id}",
                kind="run_root",
                label=f"nsrl:{run_id}",
                details={"workspace_revision": workspace_revision},
            ),
        }
        self.edges: list[TraceEdge] = []

    def add_usg_bundle(self, bundle_id: str, digest: str) -> None:
        """Register the read-only USG knowledge root (Section 7)."""
        node_id = f"usg:{bundle_id}"
        self.nodes[node_id] = TraceNode(
            node_id=node_id,
            kind="imported_knowledge",
            label=f"usg:{bundle_id}",
            details={"bundle_digest": digest, "read_only": True},
        )

    def add_source_fact(self, fact_id: str, *, epistemic_status: str, usg_version: str | None) -> None:
        """Register one source fact node (immutable, never overwritten)."""
        self.nodes[fact_id] = TraceNode(
            node_id=fact_id,
            kind="source_fact",
            label=fact_id,
            details={
                "epistemic_status": epistemic_status,
                "usg_version": usg_version,
            },
        )

    def add_proof_node(self, node_id: str, *, rule_ref: str, verified: bool) -> None:
        """Register one proof node with its rule reference and verification."""
        self.nodes[node_id] = TraceNode(
            node_id=node_id,
            kind="proof_node",
            label=node_id,
            details={"rule_ref": rule_ref, "verified": verified},
        )

    def link_proof_premises(self, node_id: str, premise_node_ids: Iterable[str]) -> None:
        """Link one proof node to its premise nodes."""
        for premise_id in premise_node_ids:
            self.edges.append(TraceEdge(premise_id, node_id, "premise_of"))

    def link_fact_to_proof(self, fact_id: str, node_id: str) -> None:
        """Link a source fact to the proof node that consumed it."""
        self.edges.append(TraceEdge(fact_id, node_id, "supports"))

    def link_conclusion(self, finding_id: str, *, proof_ref: str | None) -> None:
        """Link one published finding to its proof reference."""
        self.nodes[finding_id] = TraceNode(
            node_id=finding_id,
            kind="finding",
            label=finding_id,
            details={"proof_ref": proof_ref},
        )
        if proof_ref:
            self.edges.append(TraceEdge(proof_ref, finding_id, "proves"))

    def add_conflict(self, conflict_id: str, claim_signatures: tuple[str, ...]) -> None:
        """Register one preserved conflict and its claims."""
        self.nodes[conflict_id] = TraceNode(
            node_id=conflict_id,
            kind="conflict",
            label=conflict_id,
            details={"claims": list(claim_signatures), "status": "preserved"},
        )
        for signature in claim_signatures:
            self.edges.append(TraceEdge(signature, conflict_id, "retained_in_conflict"))

    def completeness(self, finding_ids: Iterable[str]) -> tuple[float, tuple[str, ...]]:
        """Return lineage completeness over the requested findings."""
        finals = tuple(finding_ids)
        if not finals:
            return 1.0, ()
        parents: dict[str, list[str]] = {}
        for edge in self.edges:
            if edge.relation in {"premise_of", "supports", "proves"}:
                parents.setdefault(edge.target, []).append(edge.source)
        roots = {
            node.node_id for node in self.nodes.values()
            if node.kind in {"run_root", "imported_knowledge", "source_fact"}
        }
        traceable: list[str] = []
        untraceable: list[str] = []
        for finding_id in finals:
            stack = [finding_id]
            visited: set[str] = set()
            reachable = False
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                if current in roots:
                    reachable = True
                for parent in parents.get(current, ()):
                    stack.append(parent)
            (traceable if reachable else untraceable).append(finding_id)
        return len(traceable) / max(1, len(finals)), tuple(untraceable)

    def to_dict(self) -> dict[str, Any]:
        """Return the detached JSON-compatible trace graph."""
        return {
            "nodes": [
                {
                    "node_id": node.node_id, "kind": node.kind, "label": node.label,
                    "details": dict(node.details),
                }
                for node in sorted(self.nodes.values(), key=lambda item: item.node_id)
            ],
            "edges": [
                {"source": edge.source, "target": edge.target, "relation": edge.relation}
                for edge in sorted(
                    self.edges, key=lambda item: (item.source, item.target, item.relation),
                )
            ],
        }

    @staticmethod
    def state_digest(state: Mapping[str, Any]) -> str:
        """Digest one checkpoint state payload."""
        return sha256_json(dict(state))


__all__ = ["NSRLTraceBuilder", "TraceEdge", "TraceNode"]
