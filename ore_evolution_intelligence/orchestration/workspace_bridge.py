"""Shared-workspace bridge for evolution intelligence (integration map).

The ORE workspace stores experiment requests, hypotheses, results, evidence and
unresolved issues (integration map row ``ore_workspace / Shared Cognitive
Workspace``). The bridge publishes through the workspace's controlled publication
pipeline (``FindingPublisher``) into registered append-only fields with distinct
finding types so downstream modules always distinguish hypotheses from verified
results and open issues. The bridge holds no mutable workspace state and can never
write authoritative USG records.
"""
from __future__ import annotations

from typing import Any, Sequence

#: Registered append-only workspace fields used by the evolution producer.
EVOLUTION_ARTIFACT_FIELD = "artifact_references"
EVOLUTION_ISSUE_FIELD = "unresolved_questions"


def evolution_workspace_permissions(
    *,
    producer: str = "ore_evolution_intelligence",
) -> dict[str, frozenset[str]]:
    """Field-level permissions for the evolution producer (no wildcard escalation)."""
    return {
        producer: frozenset({EVOLUTION_ARTIFACT_FIELD, EVOLUTION_ISSUE_FIELD}),
    }


class WorkspaceEvolutionPublisher:
    """Publish evolution findings into the Shared Cognitive Workspace."""

    def __init__(
        self,
        workspace: Any,
        *,
        producer: str = "ore_evolution_intelligence",
    ) -> None:
        from ore_workspace.publication import FindingPublisher
        self.workspace = workspace
        self.producer = producer
        self.publisher = FindingPublisher(workspace, producer)

    def publish_experiment_request(
        self, run_id: str, objective: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Publish the experiment request (objective, budget, baseline hash)."""
        result = self.publisher.publish(
            EVOLUTION_ARTIFACT_FIELD,
            {"kind": "experiment_request", "run_id": run_id, **objective},
            key=f"evolution_request:{run_id}",
            finding_type="evolution_request",
        )
        if result is None:
            return None
        return {
            "run_id": run_id, "finding_id": result.finding_id,
            "decision": str(result.decision), "field": EVOLUTION_ARTIFACT_FIELD,
        }

    def publish_hypothesis(
        self, candidate_id: str, payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Publish one candidate strategy as a hypothesis (never as fact)."""
        result = self.publisher.publish(
            EVOLUTION_ARTIFACT_FIELD,
            {"kind": "strategy_hypothesis", "candidate_id": candidate_id, **payload},
            key=f"evolution_hypothesis:{candidate_id}",
            finding_type="evolution_hypothesis",
        )
        if result is None:
            return None
        return {
            "candidate_id": candidate_id, "finding_id": result.finding_id,
            "decision": str(result.decision),
        }

    def publish_result(
        self, run_id: str, payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Publish verified experiment results and evidence."""
        result = self.publisher.publish(
            EVOLUTION_ARTIFACT_FIELD,
            {"kind": "experiment_result", "run_id": run_id, **payload},
            key=f"evolution_result:{run_id}",
            finding_type="evolution_result",
        )
        if result is None:
            return None
        return {
            "run_id": run_id, "finding_id": result.finding_id,
            "decision": str(result.decision),
        }

    def publish_verification(
        self, candidate_id: str, payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Publish verification outcomes distinctly from hypotheses."""
        result = self.publisher.publish(
            EVOLUTION_ARTIFACT_FIELD,
            {"kind": "strategy_verification", "candidate_id": candidate_id, **payload},
            key=f"evolution_verification:{candidate_id}",
            finding_type="evolution_verification",
        )
        if result is None:
            return None
        return {
            "candidate_id": candidate_id, "finding_id": result.finding_id,
            "decision": str(result.decision),
        }

    def publish_issue(
        self, issue_id: str, payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Publish unresolved issues from the run (underexplored regions etc.)."""
        result = self.publisher.publish(
            EVOLUTION_ISSUE_FIELD,
            {"kind": "evolution_issue", "issue_id": issue_id, **payload},
            key=f"evolution_issue:{issue_id}",
            finding_type="evolution_issue",
        )
        if result is None:
            return None
        return {
            "issue_id": issue_id, "finding_id": result.finding_id,
            "decision": str(result.decision), "field": EVOLUTION_ISSUE_FIELD,
        }

    def workspace_findings(self) -> list[dict[str, Any]]:
        """Return recorded findings for audit inspection."""
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
    "EVOLUTION_ARTIFACT_FIELD", "EVOLUTION_ISSUE_FIELD",
    "WorkspaceEvolutionPublisher", "evolution_workspace_permissions",
]
