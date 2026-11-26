"""The validated workflow task state machine (Step 17).

Transitions are checked against an explicit table and persisted
transactionally through an optional observer, so an invalid transition is
rejected rather than silently accepted.  Workflow-level status transitions
follow the same explicit-table discipline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..contracts import TaskState, WorkflowStatus, utc_now

__all__ = (
    "InvalidTransitionError",
    "StateTransitionRecord",
    "TASK_TRANSITIONS",
    "WORKFLOW_TRANSITIONS",
    "WorkflowStateMachine",
)


TASK_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.PENDING: frozenset({TaskState.READY, TaskState.CANCELLED}),
    TaskState.READY: frozenset({TaskState.LEASED, TaskState.CANCELLED}),
    TaskState.LEASED: frozenset({TaskState.RUNNING, TaskState.READY, TaskState.CANCELLED}),
    TaskState.RUNNING: frozenset(
        {TaskState.WAITING, TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.WAITING: frozenset(
        {TaskState.RUNNING, TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.SUCCEEDED: frozenset({TaskState.COMPENSATED}),
    TaskState.FAILED: frozenset({TaskState.COMPENSATED, TaskState.PENDING}),
    TaskState.CANCELLED: frozenset(),
    TaskState.COMPENSATED: frozenset(),
}

WORKFLOW_TRANSITIONS: dict[WorkflowStatus, frozenset[WorkflowStatus]] = {
    WorkflowStatus.COMPILED: frozenset({WorkflowStatus.RUNNING,
                                        WorkflowStatus.WAITING_APPROVAL,
                                        WorkflowStatus.WAITING_CLARIFICATION,
                                        WorkflowStatus.REJECTED}),
    WorkflowStatus.RUNNING: frozenset({WorkflowStatus.SUCCEEDED, WorkflowStatus.PARTIAL,
                                       WorkflowStatus.FAILED, WorkflowStatus.CANCELLED,
                                       WorkflowStatus.WAITING_APPROVAL}),
    WorkflowStatus.WAITING_APPROVAL: frozenset({WorkflowStatus.RUNNING,
                                                WorkflowStatus.REJECTED,
                                                WorkflowStatus.CANCELLED}),
    WorkflowStatus.WAITING_CLARIFICATION: frozenset({WorkflowStatus.REJECTED}),
    WorkflowStatus.SUCCEEDED: frozenset({WorkflowStatus.COMPENSATED}),
    WorkflowStatus.PARTIAL: frozenset({WorkflowStatus.COMPENSATED}),
    WorkflowStatus.FAILED: frozenset(),
    WorkflowStatus.CANCELLED: frozenset(),
    WorkflowStatus.COMPENSATED: frozenset(),
    WorkflowStatus.REJECTED: frozenset(),
    WorkflowStatus.REUSED: frozenset(),
}


class InvalidTransitionError(ValueError):
    """Raised when a state transition is not in the validated table (Step 17)."""


@dataclass(frozen=True, slots=True)
class StateTransitionRecord:
    """One transactionally recorded transition (Step 17)."""

    workflow_id: str
    task_id: str
    from_state: TaskState
    to_state: TaskState
    actor: str
    reason: str = ""
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record."""
        return {
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "actor": self.actor,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True, slots=True)
class BranchDecisionRecord:
    """One recorded conditional-branch evaluation (Step 18)."""

    predicate: str
    inputs: Mapping[str, Any]
    rule_version: str
    outcome: str  # "taken" | "not_taken" | "clarification"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record."""
        return {"predicate": self.predicate, "inputs": dict(self.inputs),
                "rule_version": self.rule_version, "outcome": self.outcome,
                "reason": self.reason}


def evaluate_branch(predicate: str, inputs: Mapping[str, Any],
                    *, rule_version: str = "branch-v1") -> BranchDecisionRecord:
    """Evaluate one predicate over typed, provenance-bearing values (Step 18).

    Predicate inputs and the rule version are recorded; a missing or
    explicitly-unknown value routes to clarification, never silent guessing.
    """
    if predicate not in inputs or inputs[predicate] is None:
        return BranchDecisionRecord(
            predicate=predicate, inputs=dict(inputs), rule_version=rule_version,
            outcome="clarification",
            reason="missing or unknown value routed to clarification")
    value = inputs[predicate]
    taken = bool(value) if not isinstance(value, str) else value.strip().lower() in (
        "true", "yes", "taken")
    return BranchDecisionRecord(
        predicate=predicate, inputs=dict(inputs), rule_version=rule_version,
        outcome="taken" if taken else "not_taken",
        reason=f"predicate {predicate!r} evaluated over recorded inputs")


class WorkflowStateMachine:
    """Track task and workflow states with validated transitions (Step 17).

    An optional ``observer`` callback receives every transition record so that
    persistence layers can store transitions transactionally.
    """

    def __init__(self, workflow_id: str,
                 observer: Callable[[StateTransitionRecord], None] | None = None) -> None:
        """Initialize the machine for one workflow."""
        if not workflow_id or not workflow_id.strip():
            raise ValueError("workflow_id must be a non-empty string")
        self.workflow_id = workflow_id
        self._observer = observer
        self._task_states: dict[str, TaskState] = {}
        self._workflow_status = WorkflowStatus.COMPILED
        self.history: list[StateTransitionRecord] = []

    @property
    def workflow_status(self) -> WorkflowStatus:
        """Return the current workflow-level status."""
        return self._workflow_status

    def task_state(self, task_id: str) -> TaskState:
        """Return the current state of a task (PENDING if never announced)."""
        return self._task_states.get(task_id, TaskState.PENDING)

    def announce(self, task_id: str) -> TaskState:
        """Register a task in the PENDING state."""
        if task_id in self._task_states:
            raise InvalidTransitionError(f"task {task_id} already announced")
        self._task_states[task_id] = TaskState.PENDING
        return TaskState.PENDING

    def transition(self, task_id: str, target: TaskState, *, actor: str = "supervisor",
                   reason: str = "") -> StateTransitionRecord:
        """Validate and apply one task-state transition (Step 17)."""
        current = self.task_state(task_id)
        if target not in TASK_TRANSITIONS[current]:
            raise InvalidTransitionError(
                f"invalid transition for task {task_id}: "
                f"{current.value} -> {target.value}")
        record = StateTransitionRecord(self.workflow_id, task_id, current, target,
                                       actor=actor, reason=reason)
        self._task_states[task_id] = target
        self.history.append(record)
        if self._observer is not None:
            self._observer(record)
        return record

    def transition_workflow(self, target: WorkflowStatus, *,
                            actor: str = "supervisor",
                            reason: str = "") -> WorkflowStatus:
        """Validate and apply one workflow-level status transition."""
        current = self._workflow_status
        if target not in WORKFLOW_TRANSITIONS[current]:
            raise InvalidTransitionError(
                f"invalid workflow transition: {current.value} -> {target.value}")
        self._workflow_status = target
        record = StateTransitionRecord(
            self.workflow_id, "workflow", TaskState.PENDING, TaskState.PENDING,
            actor=actor, reason=f"{current.value}->{target.value}: {reason}".strip(": "))
        self.history.append(record)
        if self._observer is not None:
            self._observer(record)
        return target

    def snapshot(self) -> dict[str, str]:
        """Return the current task states for checkpointing (Step 20)."""
        return {task_id: state.value for task_id, state in self._task_states.items()}

    def restore(self, task_states: Mapping[str, str]) -> None:
        """Restore task states from a checkpoint (Step 21)."""
        for task_id, state in task_states.items():
            try:
                self._task_states[task_id] = TaskState(state)
            except ValueError as exc:
                raise InvalidTransitionError(
                    f"unknown checkpoint state {state!r} for task {task_id}") from exc

    def counts(self) -> dict[str, int]:
        """Return per-state task counts for observability (Step 49)."""
        counts: dict[str, int] = {}
        for state in self._task_states.values():
            counts[state.value] = counts.get(state.value, 0) + 1
        return counts
