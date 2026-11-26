"""Auditable reasoning and provenance trace reconstruction (Step 27).

The trace graph ``G_T = (V_T, E_T)`` links every accepted, pending, and rejected
finding to its producer invocation, its dependencies, the input evidence root,
and any imported USG knowledge. Trace completeness ``T_c`` is computed only from
final claims whose complete lineage reaches an input or imported evidence root.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .contracts import InputRoot, sha256_json


@dataclass(frozen=True, slots=True)
class TraceNode:
    """Represent one traceable entity in the reasoning graph (Step 27)."""

    node_id: str
    kind: str
    label: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TraceEdge:
    """Represent one directed lineage edge in the reasoning graph (Step 27)."""

    source: str
    target: str
    relation: str


class ProvenanceTraceBuilder:
    """Build the explanation graph from workspace history and execution records (Step 27)."""

    def __init__(self, input_root: InputRoot) -> None:
        """Anchor the trace at the immutable input evidence root."""
        self.input_root = input_root
        self.nodes: dict[str, TraceNode] = {
            input_root.root_id: TraceNode(
                node_id=input_root.root_id,
                kind="input_root",
                label=f"input:{input_root.input_digest[:12]}",
                details={"input_digest": input_root.input_digest},
            )
        }
        self.edges: list[TraceEdge] = []

    def add_usg_bundle(self, bundle_id: str, digest: str) -> None:
        """Register the imported knowledge root and its digest (Step 27)."""
        node_id = f"usg:{bundle_id}"
        self.nodes[node_id] = TraceNode(
            node_id=node_id,
            kind="imported_knowledge",
            label=f"usg:{bundle_id}",
            details={"bundle_digest": digest, "read_only": True},
        )

    def add_invocation(
        self, module_id: str, invocation_id: str, *, status: str, duration_ms: float,
        suppressed: bool = False, error: str | None = None,
    ) -> None:
        """Register one module execution node (Step 27)."""
        self.nodes[invocation_id] = TraceNode(
            node_id=invocation_id,
            kind="module_execution",
            label=f"{module_id}@{invocation_id[:12]}",
            details={
                "module_id": module_id,
                "status": status,
                "duration_ms": round(duration_ms, 3),
                "suppressed": suppressed,
                "error": error,
            },
        )

    def link_input_root(self, input_root_finding_id: str) -> None:
        """Link the input-root observation finding to the evidence root node."""
        self.edges.append(TraceEdge(
            self.input_root.root_id, input_root_finding_id, "rooted_in",
        ))

    def add_finding(
        self,
        finding_id: str,
        *,
        producer_module: str,
        decision: str,
        target_field: str,
        payload: Mapping[str, Any],
        dependencies: Iterable[str] = (),
        usg_references: Iterable[Mapping[str, Any]] = (),
        span: tuple[int, int] | None = None,
    ) -> None:
        """Register one finding with its lineage edges (Step 27)."""
        self.nodes[finding_id] = TraceNode(
            node_id=finding_id,
            kind="finding",
            label=f"{producer_module}:{target_field}",
            details={
                "decision": decision,
                "field": target_field,
                "producer_module": producer_module,
                "span": list(span) if span else None,
                "payload_digest": self.payload_digest(payload),
            },
        )
        for dependency in dependencies:
            self.edges.append(TraceEdge(dependency, finding_id, "depends_on"))
        for reference in usg_references:
            bundle_id = reference.get("bundle_id")
            if bundle_id:
                self.edges.append(TraceEdge(f"usg:{bundle_id}", finding_id, "uses_usg_record"))

    def link_invocation_output(self, invocation_id: str, finding_id: str) -> None:
        """Connect a module execution to the finding it produced."""
        self.edges.append(TraceEdge(invocation_id, finding_id, "produced"))

    def add_event(self, event_id: str, revision: int, changed_fields: Iterable[str]) -> None:
        """Register one committed workspace event node."""
        self.nodes[event_id] = TraceNode(
            node_id=event_id,
            kind="workspace_event",
            label=f"event@revision:{revision}",
            details={"revision": revision, "changed_fields": sorted(changed_fields)},
        )

    def add_goal_evaluation(self, node_id: str, outcome: str, score: float, round_index: int) -> None:
        """Register one goal-evaluation node."""
        self.nodes[node_id] = TraceNode(
            node_id=node_id,
            kind="goal_evaluation",
            label=f"goal@round:{round_index}",
            details={"outcome": outcome, "score": round(score, 4), "round": round_index},
        )

    def add_conflict(self, conflict_id: str, finding_ids: Iterable[str]) -> None:
        """Register one conflict record and its retained alternatives."""
        self.nodes[conflict_id] = TraceNode(
            node_id=conflict_id,
            kind="conflict",
            label=f"conflict:{conflict_id[:12]}",
            details={"alternatives": list(finding_ids)},
        )
        for finding_id in finding_ids:
            self.edges.append(TraceEdge(finding_id, conflict_id, "retained_in_conflict"))

    @staticmethod
    def payload_digest(payload: Mapping[str, Any]) -> str:
        """Digest a finding payload without volatile execution-only fields (Step 17)."""
        stable = {
            key: value for key, value in payload.items()
            if key not in {
                "finding_id", "producer_module", "finding_type", "timestamp",
                "dependencies", "provenance", "confidence",
            }
        }
        return sha256_json(stable)

    def completeness(self, final_finding_ids: Iterable[str]) -> tuple[float, tuple[str, ...]]:
        """Return ``T_c = |C_traceable| / max(1, |C_final|)`` and untraceable IDs (Step 27)."""
        finals = tuple(final_finding_ids)
        if not finals:
            return 1.0, ()
        parents: dict[str, list[str]] = {}
        for edge in self.edges:
            if edge.relation in ("depends_on", "produced", "rooted_in"):
                parents.setdefault(edge.target, []).append(edge.source)
        roots = {
            node.node_id for node in self.nodes.values()
            if node.kind in ("input_root", "imported_knowledge")
        }
        traceable: list[str] = []
        untraceable: list[str] = []
        for finding_id in finals:
            seen = {finding_id}
            stack = [finding_id]
            reachable_root = False
            while stack:
                current = stack.pop()
                for parent in parents.get(current, []):
                    if parent in roots:
                        reachable_root = True
                    if parent not in seen:
                        seen.add(parent)
                        stack.append(parent)
                if current in roots:
                    reachable_root = True
            (traceable if reachable_root else untraceable).append(finding_id)
        completeness = len(traceable) / max(1, len(finals))
        return completeness, tuple(untraceable)

    def verify_input_integrity(self, workspace_input: str) -> bool:
        """Verify the stored input digest before presenting reproducibility (Step 27)."""
        expected = sha256_json({
            "kind": "ore_blackboard_input_root",
            "sources": [
                {"source_id": item.source_id, "group": item.source_group, "text": item.text}
                for item in self.input_root.sources
            ],
        })
        if expected != self.input_root.input_digest:
            return False
        return self.input_root.primary_text() == workspace_input

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible trace graph record."""
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


__all__ = ["ProvenanceTraceBuilder", "TraceEdge", "TraceNode"]
