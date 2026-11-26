"""Aggregation and integration of accepted results (spec Section 4G).

Aggregation validates schema and scope, confirms evidence lineage, aggregates only
compatible results while preserving source-level traceability, resolves conflicts
explicitly through the existing conflict machinery, and publishes every accepted
contribution through the existing workspace integration path (Steps 44-48). Only
accepted outputs satisfy downstream prerequisites (Step 49), and goal progress is
evaluated with the existing goal evaluator from accepted outcomes (Step 50).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ore_workspace import UpdateDecision

from .adjudication import AdjudicationReport
from .contracts import (
    AdjudicationDecision,
    AgentDescriptor,
    AgentResult,
    AgentTask,
)
from .ledger import PerformanceLedger
from .workspace_adapter import WorkspaceArtifactAdapter, is_stale_decision


@dataclass(frozen=True, slots=True)
class PublicationRecord:
    """Record one workspace publication decision for an accepted result (Step 48)."""

    task_id: str
    agent_id: str
    field: str
    decision: str
    message: str
    workspace_revision: int
    conflict_id: str | None = None
    stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a detached publication record."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "field": self.field,
            "decision": self.decision,
            "message": self.message,
            "workspace_revision": self.workspace_revision,
            "conflict_id": self.conflict_id,
            "stale": self.stale,
        }


@dataclass
class AggregationReport:
    """Summarize integration outcomes for one coordination round (Steps 44-48)."""

    publications: list[PublicationRecord] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        """Return the number of accepted publications."""
        return sum(
            1 for record in self.publications
            if record.decision in (UpdateDecision.ACCEPTED.value, UpdateDecision.MERGED.value)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached aggregation report."""
        return {
            "publications": [record.to_dict() for record in self.publications],
            "rejected": list(self.rejected),
            "deferred": list(self.deferred),
            "conflicts": list(self.conflicts),
            "duplicates": list(self.duplicates),
            "accepted_count": self.accepted_count,
        }


#: Mapping from capability to the workspace field its results propose into.
CAPABILITY_FIELD_MAP: Mapping[str, str] = {
    "macl.capability.lexical_analysis": "lexical_hypotheses",
    "macl.capability.syntactic_analysis": "syntactic_hypotheses",
    "macl.capability.semantic_interpretation": "semantic_hypotheses",
    "macl.capability.evidence_verification": "artifact_references",
    "macl.capability.critic_review": "artifact_references",
}

#: Capabilities whose outputs are structurally combinable in one aggregate.
_COMPATIBLE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"macl.capability.lexical_analysis"}),
    frozenset({"macl.capability.syntactic_analysis"}),
    frozenset({"macl.capability.semantic_interpretation"}),
    frozenset({"macl.capability.evidence_verification", "macl.capability.critic_review"}),
)


def compatible_for_aggregation(capability_a: str, capability_b: str) -> bool:
    """Return whether two capabilities share a compatible aggregation group (Step 46)."""
    return any(
        capability_a in group and capability_b in group for group in _COMPATIBLE_GROUPS
    )


class ResultAggregator:
    """Publish accepted, compatible results through the existing workspace (Steps 46-48)."""

    def __init__(
        self,
        adapter: WorkspaceArtifactAdapter,
        *,
        ledger: PerformanceLedger | None = None,
    ) -> None:
        """Bind the aggregator to the workspace adapter and optional ledger."""
        self.adapter = adapter
        self.ledger = ledger

    def publish(
        self,
        task: AgentTask,
        result: AgentResult,
        report: AdjudicationReport,
        descriptor: AgentDescriptor,
    ) -> PublicationRecord:
        """Integrate one accepted result through the workspace validation pipeline."""
        if report.decision is not AdjudicationDecision.ACCEPTED:
            raise ValueError("only accepted results may be integrated")
        field = self._target_field(task)
        if field is None or field not in descriptor.writable_workspace_fields:
            return PublicationRecord(
                task_id=task.task_id, agent_id=result.agent_id, field=field or "none",
                decision=UpdateDecision.REJECTED.value,
                message="no writable workspace field is mapped for this capability",
                workspace_revision=self.adapter.revision,
            )
        if field == "lexical_hypotheses":
            return self._publish_lexical_tokens(task, result, descriptor, field)
        payload = self._proposal_payload(task, result)
        validation = self.adapter.propose(
            descriptor, task,
            field=field, payload=payload, key=None,
            evidence=[item.to_dict() for item in result.evidence_refs],
            rules_applied=(f"macl:{task.required_capabilities[0]}",),
            confidence=result.self_confidence,
            base_revision=task.workspace_revision,
        )
        stale = is_stale_decision(validation, task, self.adapter.revision)
        record = PublicationRecord(
            task_id=task.task_id,
            agent_id=result.agent_id,
            field=field,
            decision=validation.decision.value,
            message=validation.message,
            workspace_revision=validation.workspace_revision,
            conflict_id=validation.conflict_id,
            stale=stale,
        )
        if validation.decision is UpdateDecision.CONFLICT and self.ledger is not None:
            self.ledger.record_conflict(result.agent_id)
        return record

    def _publish_lexical_tokens(
        self,
        task: AgentTask,
        result: AgentResult,
        descriptor: AgentDescriptor,
        field: str,
    ) -> PublicationRecord:
        """Publish one keyed finding per token: the lexical merge policy is keyed (Step 29)."""
        token_hypotheses = result.outputs.get("token_hypotheses")
        if not isinstance(token_hypotheses, Mapping) or not token_hypotheses:
            return PublicationRecord(
                task_id=task.task_id, agent_id=result.agent_id, field=field,
                decision=UpdateDecision.REJECTED.value,
                message="lexical result carries no token_hypotheses to integrate",
                workspace_revision=self.adapter.revision,
            )
        evidence = [item.to_dict() for item in result.evidence_refs]
        rule = f"macl:{task.required_capabilities[0]}"
        integrated = 0
        duplicates = 0
        conflicts: list[str] = []
        rejected: list[str] = []
        latest_revision = self.adapter.revision
        for token, hypotheses in token_hypotheses.items():
            candidates = hypotheses if isinstance(hypotheses, list) else [hypotheses]
            top = candidates[0] if isinstance(candidates[0], Mapping) else {"candidate": str(candidates[0])}
            payload = {
                "word": token,
                "candidates": list(candidates),
                "candidate": str(top.get("candidate", "")),
                "coverage": result.outputs.get("coverage"),
                "source_agent": result.agent_id,
                "source_task": task.task_id,
                "uncertainty": result.uncertainty,
                "assumptions": list(result.assumptions),
            }
            validation = self.adapter.propose(
                descriptor, task,
                field=field, payload=payload, key=str(token),
                evidence=evidence,
                rules_applied=(rule,),
                confidence=result.self_confidence,
                base_revision=task.workspace_revision,
            )
            latest_revision = validation.workspace_revision
            if validation.decision is UpdateDecision.ACCEPTED:
                integrated += 1
            elif validation.decision is UpdateDecision.DUPLICATE:
                duplicates += 1
            elif validation.decision is UpdateDecision.CONFLICT:
                conflicts.append(validation.conflict_id or "")
            else:
                rejected.append(f"{token}: {validation.message}")
        total = len(token_hypotheses)
        integrated_like = integrated + duplicates
        conflict_id = conflicts[0] if conflicts else None
        if integrated_like == total and not conflicts:
            message = (
                f"integrated {integrated} lexical token findings "
                f"({duplicates} already present)"
            )
            decision = UpdateDecision.ACCEPTED.value
        elif integrated_like > 0:
            decision = (
                UpdateDecision.CONFLICT.value if conflicts else UpdateDecision.MERGED.value
            )
            message = (
                f"partially integrated {integrated_like}/{total} lexical tokens "
                f"({len(rejected)} rejected"
                + (f", {len(conflicts)} conflicts" if conflicts else "")
                + ")"
            )
        else:
            decision, message = (
                UpdateDecision.REJECTED.value,
                "no lexical token finding passed publication validation: "
                + "; ".join(rejected[:3]),
            )
        if conflicts and self.ledger is not None:
            self.ledger.record_conflict(result.agent_id)
        return PublicationRecord(
            task_id=task.task_id,
            agent_id=result.agent_id,
            field=field,
            decision=decision,
            message=message,
            workspace_revision=latest_revision,
            conflict_id=conflict_id,
        )

    def aggregate_round(
        self,
        entries: Sequence[tuple[AgentTask, AgentResult, AdjudicationReport, AgentDescriptor]],
    ) -> AggregationReport:
        """Publish every accepted entry, aggregating only compatible results (Step 46)."""
        summary = AggregationReport()
        published: list[tuple[str, str]] = []
        for task, result, report, descriptor in entries:
            if report.decision is not AdjudicationDecision.ACCEPTED:
                if report.decision is AdjudicationDecision.DUPLICATE:
                    summary.duplicates.append(task.task_id)
                elif report.decision is AdjudicationDecision.DEFERRED:
                    summary.deferred.append(task.task_id)
                else:
                    summary.rejected.append(task.task_id)
                continue
            capability = task.required_capabilities[0]
            if any(
                not compatible_for_aggregation(capability, published_capability)
                for published_capability, _ in published
            ):
                summary.rejected.append(task.task_id)
                continue
            record = self.publish(task, result, report, descriptor)
            summary.publications.append(record)
            if record.decision == UpdateDecision.REJECTED.value:
                summary.rejected.append(task.task_id)
            elif record.decision == UpdateDecision.DEFERRED.value:
                summary.deferred.append(task.task_id)
            elif record.decision == UpdateDecision.DUPLICATE.value:
                summary.duplicates.append(task.task_id)
            elif record.conflict_id is not None:
                summary.conflicts.append(record.conflict_id)
            if record.decision != UpdateDecision.REJECTED.value:
                published.append((capability, task.task_id))
        return summary

    def _target_field(self, task: AgentTask) -> str | None:
        """Resolve the writable workspace field for the task's primary capability."""
        for capability in task.required_capabilities:
            field = CAPABILITY_FIELD_MAP.get(capability)
            if field is not None:
                return field
        return None

    @staticmethod
    def _proposal_payload(task: AgentTask, result: AgentResult) -> dict[str, Any]:
        """Wrap the result outputs with source-level traceability (Step 46)."""
        return {
            **dict(result.outputs),
            "candidate": str(result.outputs.get("interpretation", result.outputs.get("coverage", ""))),
            "source_agent": result.agent_id,
            "source_task": task.task_id,
            "uncertainty": result.uncertainty,
            "assumptions": list(result.assumptions),
        }


def evaluate_goal_progress(evaluator, workspace, *, component_scores=None):
    """Evaluate the goal with the existing evaluator over accepted state (Step 50)."""
    return workspace.evaluate_goal(evaluator, component_scores)


def update_dependencies(
    dag,
    *,
    accepted_task_ids: Sequence[str],
) -> tuple[str, ...]:
    """Activate tasks whose prerequisites are now satisfied by accepted outputs (Step 49)."""
    accepted = frozenset(accepted_task_ids)
    return tuple(
        task.task_id for task in dag.ready_tasks(accepted)
        if task.task_id not in accepted
    )
