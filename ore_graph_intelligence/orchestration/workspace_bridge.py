"""Shared cognitive workspace bridge for graph intelligence (step 32 and step 39).

Hypotheses and verification outcomes are published through the ORE workspace's
controlled publication pipeline with *distinct finding types* so downstream modules
always distinguish unverified hypotheses from verified facts (acceptance test 14).
The bridge holds no mutable workspace state: publication goes through
:class:`ore_workspace.publication.FindingPublisher`, and nothing here can write
authoritative USG records.

The workspace must grant the producer ``ore_graph_intelligence`` write access to the
hypothesis field; :func:`graph_workspace_permissions` builds that policy from the
workspace's own defaults (no wildcard escalation).
"""
from __future__ import annotations

from typing import Any, Sequence

from ..schemas.candidate_schema import GraphCandidate
from ..verification.promotion_gate import VerificationResult

#: Workspace field that graph hypotheses publish into (registered, append-only).
GRAPH_HYPOTHESIS_FIELD = "semantic_hypotheses"


def graph_workspace_permissions(
    *, producer: str = "ore_graph_intelligence",
) -> dict[str, frozenset[str]]:
    """Field-level permissions for the graph intelligence producer."""
    return {producer: frozenset({GRAPH_HYPOTHESIS_FIELD})}


class WorkspaceGraphPublisher:
    """Publish graph hypotheses and verification outcomes to a SharedWorkspace."""

    def __init__(
        self,
        workspace: Any,
        *,
        producer: str = "ore_graph_intelligence",
        field: str = GRAPH_HYPOTHESIS_FIELD,
    ) -> None:
        from ore_workspace.publication import FindingPublisher

        self.workspace = workspace
        self.producer = producer
        self.field = field
        self.publisher = FindingPublisher(workspace, producer)

    def publish_hypothesis(
        self,
        candidate: GraphCandidate,
        namespace: str,
    ) -> dict[str, object] | None:
        """Publish one candidate as a hypothesis in the dedicated namespace (step 32)."""
        payload = {**candidate.as_payload(), "namespace": namespace}
        result = self.publisher.publish(
            self.field,
            payload,
            key=f"{namespace}:{candidate.candidate_id}",
            finding_type="graph_hypothesis",
        )
        if result is None:
            return None
        return {
            "candidate_id": candidate.candidate_id,
            "finding_id": result.finding_id,
            "decision": str(result.decision),
            "finding_type": "graph_hypothesis",
            "namespace": namespace,
        }

    def publish_verifications(
        self, results: Sequence[VerificationResult],
    ) -> list[dict[str, object]]:
        """Publish verification dispositions distinctly from hypotheses (step 39)."""
        receipts: list[dict[str, object]] = []
        for result in results:
            published = self.publisher.publish(
                self.field,
                result.as_dict(),
                key=f"graph-verification:{result.candidate_id}",
                finding_type="graph_verification",
            )
            if published is None:
                continue
            receipts.append({
                "candidate_id": result.candidate_id,
                "finding_id": published.finding_id,
                "decision": str(published.decision),
                "finding_type": "graph_verification",
                "disposition": result.disposition,
            })
        return receipts

    def workspace_findings(self) -> list[dict[str, object]]:
        """Return the workspace's recorded findings for audit inspection."""
        findings = getattr(self.workspace, "findings", None) or {}
        return [
            {
                "finding_id": finding.finding_id,
                "finding_type": finding.finding_type,
                "target_field": finding.target.field,
                "target_key": finding.target.key,
                "producer": finding.producer_module,
            }
            for finding in findings.values()
        ]


__all__ = [
    "GRAPH_HYPOTHESIS_FIELD", "WorkspaceGraphPublisher", "graph_workspace_permissions",
]
