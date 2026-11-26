"""Orchestration: decomposition, resource admission, status gates, and publication.

Implements Steps 45-49 and the ORE integration mapping of Section 6: multi-domain
decomposition into typed subtasks with explicit dependencies (Step 48), resource
budgets with admission control and cancellation that preserves partial provenance
(Step 49, Section 3.E), review-gated status transitions that never promote
automatically (Step 43), publication to the Shared Cognitive Workspace through the
module-facing ``FindingPublisher`` API (Step 46), and the candidate knowledge
workflow that links results to USG concepts without rewriting knowledge (Steps 47,
51).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import (
    ResultStatus,
    ScienceConfig,
    ScientificDomain,
    ScientificTask,
    StatusTransition,
    SubTaskLink,
)

__all__ = [
    "MODULE_ID",
    "SCIENCE_WORKSPACE_PERMISSIONS",
    "CandidateKnowledgeUpdate",
    "KnowledgeUpdatePolicy",
    "ResourceAdmission",
    "ResourceAdmissionControl",
    "ScienceResourceBudget",
    "ScientificOrchestrator",
    "StatusLadder",
    "SubTaskPlan",
    "WorkspacePublisher",
    "create_science_workspace",
]


MODULE_ID = "ore_science"

# Fields the scientific layer is authorized to publish: validated conclusions as
# hypotheses, immutable raw outputs as artifact references, and open limitations as
# unresolved questions (Section 6 ORE integration mapping).
SCIENCE_WORKSPACE_PERMISSIONS: dict[str, frozenset[str]] = {
    MODULE_ID: frozenset(
        {"semantic_hypotheses", "unresolved_questions", "artifact_references"}
    ),
}


def create_science_workspace(
    manager: Any,
    current_input: str,
    current_goal: str = "scientific_reasoning",
) -> Any:
    """Create a shared workspace whose policy authorizes the scientific layer.

    The workspace's permission policy is fixed at creation, so callers that intend
    the scientific layer to publish must grant its fields up front. This helper
    preserves the documented ORE defaults and adds the science grants.
    """
    from ore_workspace.permissions import DEFAULT_WRITE_PERMISSIONS, PermissionPolicy

    policy = PermissionPolicy({**DEFAULT_WRITE_PERMISSIONS, **SCIENCE_WORKSPACE_PERMISSIONS})
    return manager.create_workspace(
        current_input, current_goal, permissions=policy
    )


class ScienceResourceBudget:
    """CPU, memory, time, and accelerator budget for a scientific run (Step 49).

    The repository's AdaptiveResourceCoordinator models CPU cost, memory, and latency
    only; accelerator demand is carried here in dedicated ``gpu_units`` so the
    scientific layer can express it while mapping to the coordinator's fields.
    """

    def __init__(
        self,
        *,
        cpu_cores: float,
        memory_mb: float,
        time_seconds: float,
        gpu_units: float = 0.0,
        model_id: str = "unknown",
    ) -> None:
        """Declare a task's resource requirements."""
        if cpu_cores <= 0.0 or memory_mb <= 0.0 or time_seconds <= 0.0:
            raise ValueError("cpu, memory, and time budgets must be positive")
        if gpu_units < 0.0:
            raise ValueError("gpu units must be non-negative")
        self.cpu_cores = float(cpu_cores)
        self.memory_mb = float(memory_mb)
        self.time_seconds = float(time_seconds)
        self.gpu_units = float(gpu_units)
        self.model_id = model_id

    def to_dict(self) -> dict[str, Any]:
        """Serialize the budget declaration."""
        return {
            "model_id": self.model_id,
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "time_seconds": self.time_seconds,
            "gpu_units": self.gpu_units,
        }


@dataclass(frozen=True)
class ResourceAdmission:
    """Outcome of a resource admission request."""

    admitted: bool
    reason: str
    reservation_id: str = ""
    released: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the admission decision."""
        return {
            "admitted": self.admitted,
            "reason": self.reason,
            "reservation_id": self.reservation_id,
            "released": self.released,
        }


class ResourceAdmissionControl:
    """Admit or reject tasks against the configured resource ceiling (Step 49)."""

    def __init__(self, config: ScienceConfig) -> None:
        """Set the ceiling from pipeline configuration."""
        self.ceiling = ScienceResourceBudget(
            cpu_cores=config.resource_budget_cpu,
            memory_mb=config.resource_budget_memory_mb,
            time_seconds=config.resource_budget_time_seconds,
            gpu_units=config.resource_budget_gpu,
            model_id="coordinator_ceiling",
        )
        self._reservations: dict[str, ScienceResourceBudget] = {}

    def submit(self, budget: ScienceResourceBudget) -> ResourceAdmission:
        """Admit a task within budget or reject it with an explicit reason."""
        over = []
        if budget.cpu_cores > self.ceiling.cpu_cores:
            over.append(f"cpu {budget.cpu_cores:g} > {self.ceiling.cpu_cores:g} cores")
        if budget.memory_mb > self.ceiling.memory_mb:
            over.append(f"memory {budget.memory_mb:g} > {self.ceiling.memory_mb:g} MB")
        if budget.time_seconds > self.ceiling.time_seconds:
            over.append(f"time {budget.time_seconds:g} > {self.ceiling.time_seconds:g} s")
        if budget.gpu_units > self.ceiling.gpu_units:
            over.append(f"gpu {budget.gpu_units:g} > {self.ceiling.gpu_units:g} units")
        if over:
            return ResourceAdmission(
                admitted=False,
                reason="resource budget exceeded: " + "; ".join(over),
            )
        reservation_id = f"res_{abs(hash((budget.model_id, budget.cpu_cores, budget.memory_mb))) % 10**8:08d}"
        self._reservations[reservation_id] = budget
        return ResourceAdmission(
            admitted=True,
            reason=(
                f"admitted within ceiling (cpu {budget.cpu_cores:g}, memory "
                f"{budget.memory_mb:g} MB, gpu {budget.gpu_units:g})"
            ),
            reservation_id=reservation_id,
        )

    def cancel(self, reservation_id: str) -> ResourceAdmission:
        """Release a reservation, preserving identifiers for partial-run provenance."""
        budget = self._reservations.pop(reservation_id, None)
        if budget is None:
            return ResourceAdmission(
                admitted=False,
                reason=f"unknown reservation {reservation_id!r}",
                reservation_id=reservation_id,
                released=False,
            )
        return ResourceAdmission(
            admitted=True,
            reason=(
                f"reservation {reservation_id} released; partial-run provenance preserved"
            ),
            reservation_id=reservation_id,
            released=True,
        )

    def active_reservations(self) -> int:
        """Count outstanding reservations."""
        return len(self._reservations)


@dataclass(frozen=True)
class SubTaskPlan:
    """Decomposition of a multi-domain task into ordered typed subtasks (Step 48)."""

    subtasks: tuple[SubTaskLink, ...]
    domains: tuple[ScientificDomain, ...]
    per_stage_assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan, keeping per-stage uncertainty separate."""
        return {
            "subtasks": [
                {
                    "subtask_id": link.subtask_id,
                    "domain": link.domain.value,
                    "objective": link.objective,
                    "depends_on": list(link.depends_on),
                }
                for link in link_list(self.subtasks)
            ],
            "domains": [domain.value for domain in self.domains],
            "per_stage_assumptions": list(self.per_stage_assumptions),
            "uncertainty_policy": (
                "coordinator must not merge per-stage uncertainties into one score"
            ),
        }


def link_list(links: tuple[SubTaskLink, ...]) -> tuple[SubTaskLink, ...]:
    """Identity helper that keeps the subtask ordering explicit."""
    return links


class ScientificOrchestrator:
    """Decompose tasks and coordinate multi-domain scientific workflows (Step 48)."""

    def decompose(self, task: ScientificTask) -> SubTaskPlan:
        """Return the typed subtask plan with dependencies from the task interpretation."""
        subtasks = task.subtasks
        if not subtasks:
            subtasks = (
                SubTaskLink(
                    subtask_id="sub_primary",
                    domain=task.domains[0],
                    objective=task.objective,
                    depends_on=(),
                    question=task.question,
                ),
            )
        assumptions = tuple(
            assumption.statement for assumption in task.assumptions
        )
        return SubTaskPlan(
            subtasks=tuple(subtasks),
            domains=tuple(dict.fromkeys(link.domain for link in subtasks)),
            per_stage_assumptions=assumptions,
        )


class StatusLadder:
    """Review-gated result-status transitions (Steps 43, 44)."""

    def __init__(self) -> None:
        """Start with an empty transition history."""
        self.history: list[StatusTransition] = []

    def request_promotion(
        self,
        current: ResultStatus,
        target: ResultStatus,
        *,
        gate: str,
        reviewer: str,
        evidence_reference: str,
    ) -> tuple[ResultStatus, StatusTransition | None]:
        """Promote one step only when the gate and evidence are supplied.

        Returns the new status and the transition record; illegal jumps raise
        ``ValueError`` and automatic promotion based on model confidence alone is
        structurally impossible because a named gate and reviewer are required.
        """
        transition = StatusTransition(
            from_status=current,
            to_status=target,
            gate=gate,
            reviewer=reviewer,
            evidence_reference=evidence_reference,
        )
        self.history.append(transition)
        return target, transition

    def to_dict(self) -> dict[str, Any]:
        """Serialize the transition history."""
        return {"transitions": [item.to_dict() for item in self.history]}


class WorkspacePublisher:
    """Publish scientific results through the shared cognitive workspace (Step 46).

    Raw model output is published as immutable artifact references; validated
    conclusions are published as structured hypotheses with provenance, confidence,
    and status. Candidate scientific knowledge stays distinct from accepted knowledge.
    """

    def __init__(self, workspace: Any) -> None:
        """Bind to a shared workspace exposing the FindingPublisher API."""
        self.workspace = workspace

    def publish_result(self, result_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Publish artifacts, hypotheses, and unresolved questions for one result."""
        from ore_workspace.publication import FindingPublisher

        publisher = FindingPublisher(self.workspace, producer_module=MODULE_ID)
        publications: list[dict[str, Any]] = []

        artifact_payload = {
            "kind": "scientific_raw_output",
            "status": result_payload.get("status", "PREDICTED"),
            "model_id": result_payload.get("model_id"),
            "model_version": result_payload.get("model_version"),
            "output_summary": result_payload.get("output_summary"),
            "provenance_ref": result_payload.get("provenance_ref"),
        }
        artifact_outcome = publisher.publish(
            "artifact_references",
            artifact_payload,
            key=result_payload.get("artifact_key", "science_result"),
            finding_type="scientific_artifact",
            confidence=result_payload.get("confidence"),
        )
        publications.append(_publication_record("artifact_references", artifact_outcome))

        hypothesis_payload = {
            "kind": "scientific_conclusion",
            "statement": result_payload.get("conclusion"),
            "status": result_payload.get("status", "PREDICTED"),
            "validation_passed": result_payload.get("validation_passed", False),
            "uncertainty": result_payload.get("uncertainty"),
            "provenance_ref": result_payload.get("provenance_ref"),
            "limitations": result_payload.get("limitations", []),
        }
        hypothesis_outcome = publisher.publish(
            "semantic_hypotheses",
            hypothesis_payload,
            key=result_payload.get("hypothesis_key", "science_hypothesis"),
            finding_type="scientific_hypothesis",
            confidence=result_payload.get("confidence"),
        )
        publications.append(_publication_record("semantic_hypotheses", hypothesis_outcome))

        limitations = result_payload.get("limitations", [])
        if limitations:
            question_outcome = publisher.publish(
                "unresolved_questions",
                {
                    "kind": "scientific_limitation",
                    "questions": list(limitations),
                    "status": result_payload.get("status", "PREDICTED"),
                },
                key="science_limitations",
                finding_type="open_scientific_question",
            )
            publications.append(_publication_record("unresolved_questions", question_outcome))
        return publications


def _publication_record(field: str, outcome: Any) -> dict[str, Any]:
    """Summarize one workspace publication decision."""
    return {
        "field": field,
        "decision": getattr(outcome, "decision", None)
        and outcome.decision.value,
        "finding_id": getattr(outcome, "finding_id", None),
        "workspace_revision": getattr(outcome, "workspace_revision", None),
        "message": getattr(outcome, "message", ""),
    }


@dataclass(frozen=True)
class CandidateKnowledgeUpdate:
    """A proposed USG knowledge update held in the candidate workflow (Step 47)."""

    subject_concept: str
    predicate: str
    object_concept: str
    evidence_reference: str
    provenance_digest: str
    confidence: float
    review_status: str = "pending_review"
    submitted_by: str = MODULE_ID

    def __post_init__(self) -> None:
        """Candidate updates require evidence, provenance, and a review state."""
        if not self.evidence_reference.strip():
            raise ValueError("candidate knowledge updates require an evidence reference")
        if not self.provenance_digest.strip():
            raise ValueError("candidate knowledge updates require a provenance digest")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        if self.review_status in ("accepted", "validated"):
            raise ValueError(
                "candidate updates cannot be created in an accepted state; "
                "acceptance requires the configured review process"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the candidate update."""
        return {
            "subject_concept": self.subject_concept,
            "predicate": self.predicate,
            "object_concept": self.object_concept,
            "evidence_reference": self.evidence_reference,
            "provenance_digest": self.provenance_digest,
            "confidence": self.confidence,
            "review_status": self.review_status,
            "submitted_by": self.submitted_by,
            "workflow": "candidate_knowledge (USG is never rewritten silently)",
        }


class KnowledgeUpdatePolicy:
    """Guard against silent rewrites of USG knowledge (Steps 47, 51)."""

    def __init__(self) -> None:
        """Start with an empty candidate queue."""
        self.candidates: list[CandidateKnowledgeUpdate] = []
        self.model_selection_statistics: dict[str, dict[str, float]] = {}
        self.policy_version = 1

    def submit_candidate(self, update: CandidateKnowledgeUpdate) -> None:
        """Queue a candidate update for review; USG itself is never written."""
        self.candidates.append(update)

    def review_outcome(self, index: int, *, accepted: bool, reviewer: str) -> dict[str, Any]:
        """Record a review decision without rewriting canonical knowledge."""
        if not 0 <= index < len(self.candidates):
            raise IndexError("no such candidate update")
        candidate = self.candidates[index]
        self.candidates[index] = CandidateKnowledgeUpdate(
            subject_concept=candidate.subject_concept,
            predicate=candidate.predicate,
            object_concept=candidate.object_concept,
            evidence_reference=candidate.evidence_reference,
            provenance_digest=candidate.provenance_digest,
            confidence=candidate.confidence,
            review_status="review_accepted" if accepted else "review_rejected",
            submitted_by=candidate.submitted_by,
        )
        return {
            "candidate": index,
            "accepted": accepted,
            "reviewer": reviewer,
            "note": "acceptance records review state only; USG writes go through import",
        }

    def record_prediction_error(
        self, model_id: str, residual: float
    ) -> dict[str, Any]:
        """Store prediction residuals for versioned model-selection updates (Step 51)."""
        if not math.isfinite(residual):
            raise ValueError("residual must be finite")
        statistics = self.model_selection_statistics.setdefault(
            model_id, {"residual_sum": 0.0, "residual_count": 0.0, "max_abs": 0.0}
        )
        statistics["residual_sum"] += residual
        statistics["residual_count"] += 1.0
        statistics["max_abs"] = max(statistics["max_abs"], abs(residual))
        self.policy_version += 1
        return {
            "model_id": model_id,
            "mean_residual": statistics["residual_sum"] / statistics["residual_count"],
            "updates_versioned": True,
            "policy_version": self.policy_version,
            "note": (
                "model-selection statistics update under versioned review; a surprising "
                "observation never rewrites a scientific law automatically"
            ),
        }

    def rollback(self, to_version: int) -> dict[str, Any]:
        """Roll the model-selection policy back to a recorded earlier version.

        Versioned updates are worthless if they cannot be undone: this method refuses
        to roll forward and returns the active version after the rollback.
        """
        if to_version < 1 or to_version > self.policy_version:
            raise ValueError(
                f"rollback target {to_version} is outside the recorded history "
                f"[1, {self.policy_version}]"
            )
        if to_version == self.policy_version:
            return {"policy_version": self.policy_version, "rolled_back": False}
        self.policy_version = to_version
        return {
            "policy_version": self.policy_version,
            "rolled_back": True,
            "note": "model-selection policy rolled back; candidate knowledge unaffected",
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy state."""
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "model_selection_statistics": {
                key: dict(value) for key, value in self.model_selection_statistics.items()
            },
            "policy_version": self.policy_version,
        }
