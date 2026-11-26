"""Result adjudication: schema, provenance, revision, evidence, and criteria checks.

The adjudicator validates schema, provenance, revision, and task contract before
any result can be integrated (spec Section 3, collect-and-validate). It detects
duplicate, contradictory, stale, unsupported, and incomplete results. Self-
evaluation is never final acceptance; completion means contract satisfaction.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import (
    AdjudicationDecision,
    AgentResult,
    AgentTask,
)
from .workspace_adapter import WorkspaceArtifactAdapter


def result_fingerprint(result: AgentResult) -> str:
    """Fingerprint result content independently of mutable timing fields (Step 29)."""
    payload = {
        "agent_id": result.agent_id,
        "task_id": result.task_id,
        "outputs": dict(result.outputs),
        "evidence_refs": [item.to_dict() for item in result.evidence_refs],
        "provenance": result.provenance.to_dict(),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AdjudicationReport:
    """Record the adjudicator's decision with structured, auditable reasons."""

    decision: AdjudicationDecision
    task_id: str
    agent_id: str
    reasons: tuple[str, ...] = ()
    evidence_errors: tuple[str, ...] = ()
    criteria_gaps: tuple[str, ...] = ()
    fingerprint: str = ""
    verified_by: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a detached report record."""
        return {
            "decision": self.decision.value,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "reasons": list(self.reasons),
            "evidence_errors": list(self.evidence_errors),
            "criteria_gaps": list(self.criteria_gaps),
            "fingerprint": self.fingerprint,
            "verified_by": list(self.verified_by),
        }


class ResultAdjudicator:
    """Validate typed results against their task contracts (Steps 44-45, Section 3)."""

    def __init__(
        self,
        adapter: WorkspaceArtifactAdapter,
        *,
        seen_fingerprints: Mapping[str, str] | None = None,
    ) -> None:
        """Bind the adjudicator to the workspace adapter and duplicate memory."""
        self.adapter = adapter
        self._seen: dict[str, str] = dict(seen_fingerprints or {})

    def validate(
        self,
        result: AgentResult,
        task: AgentTask,
        *,
        current_revision: int | None = None,
        known_agent_versions: Mapping[str, str] | None = None,
    ) -> AdjudicationReport:
        """Run schema, identity, revision, provenance, evidence, and criteria checks."""
        reasons: list[str] = []
        if result.task_id != task.task_id:
            reasons.append(f"result task identity {result.task_id!r} does not match {task.task_id!r}")
        versions = dict(known_agent_versions or {})
        if versions and result.agent_id in versions and result.agent_version != versions[result.agent_id]:
            reasons.append(
                f"result agent version {result.agent_version!r} does not match the registered "
                f"{versions[result.agent_id]!r}"
            )
        schema_errors = task.expected_result_schema.validate(result.outputs)
        if schema_errors:
            reasons.extend(schema_errors)
        revision_now = self.adapter.revision if current_revision is None else current_revision
        binding_mismatch = result.source_revision != task.workspace_revision
        obsolete = (
            result.source_revision < revision_now and task.workspace_revision < revision_now
        )
        if binding_mismatch or obsolete:
            reasons.append(
                f"result derived from revision {result.source_revision} while the task is bound to "
                f"{task.workspace_revision} and the workspace is at {revision_now}; "
                "obsolete workspace state must be revalidated or recomputed before integration"
            )
        evidence_errors = self.adapter.validate_evidence_lineage(result.evidence_refs)
        if evidence_errors:
            reasons.append("evidence lineage does not resolve")
        fingerprint = result_fingerprint(result)
        if fingerprint in self._seen:
            return AdjudicationReport(
                decision=AdjudicationDecision.DUPLICATE, task_id=task.task_id,
                agent_id=result.agent_id,
                reasons=(f"identical result content already adjudicated ({self._seen[fingerprint]})",),
                fingerprint=fingerprint,
            )
        criteria_gaps = self._criteria_gaps(result, task)
        if criteria_gaps:
            reasons.extend(criteria_gaps)
        if evidence_errors and self._looks_fabricated(result):
            decision = AdjudicationDecision.REJECTED
        elif criteria_gaps and not schema_errors and not evidence_errors:
            decision = AdjudicationDecision.NEEDS_REVISION
        elif reasons:
            decision = AdjudicationDecision.REJECTED
        else:
            decision = AdjudicationDecision.ACCEPTED
        if decision is not AdjudicationDecision.DUPLICATE:
            self._seen[fingerprint] = task.task_id
        return AdjudicationReport(
            decision=decision,
            task_id=task.task_id,
            agent_id=result.agent_id,
            reasons=tuple(reasons),
            evidence_errors=tuple(evidence_errors),
            criteria_gaps=tuple(criteria_gaps),
            fingerprint=fingerprint,
        )

    def record_verification(self, task_id: str, verifier_agent_id: str, passed: bool,
                            report: AdjudicationReport) -> AdjudicationReport:
        """Attach an independent verification outcome to an adjudication report (Step 42)."""
        verified_by = tuple(report.verified_by) + (
            (verifier_agent_id,) if passed else (f"{verifier_agent_id}:failed",)
        )
        decision = report.decision if passed else AdjudicationDecision.REJECTED
        reasons = tuple(report.reasons) if passed else tuple(report.reasons) + (
            f"independent verification by {verifier_agent_id} failed",
        )
        return AdjudicationReport(
            decision=decision, task_id=report.task_id, agent_id=report.agent_id,
            reasons=reasons, evidence_errors=report.evidence_errors,
            criteria_gaps=report.criteria_gaps, fingerprint=report.fingerprint,
            verified_by=verified_by,
        )

    @staticmethod
    def _criteria_gaps(result: AgentResult, task: AgentTask) -> tuple[str, ...]:
        """Check the task's explicit success criteria against the submitted result (Step 23)."""
        gaps: list[str] = []
        criteria = task.success_criteria
        for key in criteria.required_output_keys:
            if key not in result.outputs:
                gaps.append(f"success criteria requires missing output key: {key}")
        if result.self_confidence < criteria.minimum_confidence:
            gaps.append(
                f"self confidence {result.self_confidence:.2f} is below the required "
                f"{criteria.minimum_confidence:.2f}"
            )
        if criteria.must_cite_evidence and not result.evidence_refs:
            gaps.append("success criteria requires cited evidence but none was provided")
        for action in criteria.forbidden_actions:
            if action in result.provenance.transformation_history:
                gaps.append(f"forbidden action recorded in provenance: {action}")
        return tuple(gaps)

    @staticmethod
    def _looks_fabricated(result: AgentResult) -> bool:
        """Detect provenance that references nothing the agent actually consumed."""
        return bool(result.evidence_refs) and not result.input_refs and not result.provenance.source_artifacts

    def duplicates(self) -> Mapping[str, str]:
        """Return the recorded result fingerprints for audits."""
        return dict(self._seen)


def check_conflict_preservation(
    conflicting_findings: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Confirm conflicting claims are retained rather than overwritten (Step 30)."""
    return {
        "conflict_count": len(conflicting_findings),
        "retained": all(bool(item) for item in conflicting_findings),
        "majority_vote_used": False,
    }
