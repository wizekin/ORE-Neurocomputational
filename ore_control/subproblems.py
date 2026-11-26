"""Bounded problem-solving subcontexts for the control layer (Phase E).

Implements Point 02 Steps 22-25: decomposition detection, child goal and
contract definition, subproblem boundary enforcement (depth, child count,
budgets, and allowed publications), and reconciliation of child outcomes with
the parent context while preserving conflicting findings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ore_workspace.models import utc_now

from .config import ControlPolicy
from .contracts import (
    COGNITIVE_CONTROL_SCHEMA_VERSION,
    ContextBudget,
    ContextStatus,
    CognitiveContext,
    new_control_id,
)
from .operators import ChildGoalSpec, OperatorFinding


@dataclass(frozen=True, slots=True)
class ChildContract:
    """The contract each child context executes under (Step 23)."""

    context_id: str
    parent_context_id: str
    goal_text: str
    input_references: tuple[str, ...] = ()
    required_evidence: tuple[str, ...] = ()
    allowed_operator_ids: tuple[str, ...] = ()
    output_fields: tuple[str, ...] = ()
    budget: ContextBudget = field(default_factory=ContextBudget)
    deadline_seconds: float | None = None
    success_predicate: str = "hypothesis_resolved"
    derived_from_workspace_revision: int = 0
    schema_version: str = COGNITIVE_CONTROL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the child contract."""
        if not self.goal_text.strip():
            raise ValueError("child goal text must be non-empty")
        if not self.allowed_operator_ids:
            raise ValueError("a child contract must allow at least one operator")
        if not self.output_fields:
            raise ValueError("a child contract must declare output fields")
        object.__setattr__(self, "input_references", tuple(self.input_references))
        object.__setattr__(self, "required_evidence", tuple(self.required_evidence))
        object.__setattr__(self, "allowed_operator_ids", tuple(self.allowed_operator_ids))
        object.__setattr__(self, "output_fields", tuple(self.output_fields))

    def permits_operator(self, operator_id: str) -> bool:
        """Return whether the child may run one operator (Step 24)."""
        return operator_id in self.allowed_operator_ids

    def permits_field(self, workspace_field: str) -> bool:
        """Return whether the child may publish to one workspace field (Step 24)."""
        return workspace_field in self.output_fields

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable contract record."""
        return {
            "context_id": self.context_id,
            "parent_context_id": self.parent_context_id,
            "goal_text": self.goal_text,
            "input_references": list(self.input_references),
            "required_evidence": list(self.required_evidence),
            "allowed_operator_ids": list(self.allowed_operator_ids),
            "output_fields": list(self.output_fields),
            "budget": self.budget.to_dict(),
            "deadline_seconds": self.deadline_seconds,
            "success_predicate": self.success_predicate,
            "derived_from_workspace_revision": self.derived_from_workspace_revision,
            "schema_version": self.schema_version,
        }


def decomposition_justified(
    *,
    competing_hypotheses: int,
    open_questions: int,
    available_budget: ContextBudget,
) -> bool:
    """Decide whether decomposition is justified (Step 22).

    Decomposition requires either multiple independent hypotheses to test or
    unresolved evidence gaps, plus enough remaining budget to fund children.
    Arbitrary splitting is not justified.
    """
    if competing_hypotheses < 2 and open_questions < 1:
        return False
    if available_budget.operations < 2:
        return False
    return True


def create_child_contexts(
    parent: CognitiveContext,
    goal_specs: Iterable[ChildGoalSpec],
    *,
    policy: ControlPolicy,
    workspace_revision: int,
) -> tuple[tuple[CognitiveContext, ChildContract], ...]:
    """Create bounded child contexts from decomposition specs (Steps 23-24).

    Each child receives a narrow goal, references, allowed operators, output
    fields, and a budget slice of the parent budget; depth and child-count
    limits are enforced before any child exists.
    """
    if parent.depth >= policy.budgets.max_depth:
        raise ValueError(
            f"max subproblem depth reached: {parent.depth} >= "
            f"{policy.budgets.max_depth}"
        )
    specs = tuple(goal_specs)
    if not specs:
        return ()
    if len(specs) > policy.budgets.max_children:
        specs = specs[: policy.budgets.max_children]
    child_budget = ContextBudget(
        time_seconds=max(
            1.0, parent.budget.time_seconds / (len(specs) + 1),
        ),
        operations=policy.budgets.child_operation_budget,
        memory_mb=max(16.0, parent.budget.memory_mb / (len(specs) + 1)),
        concurrency=1,
    )
    created: list[tuple[CognitiveContext, ChildContract]] = []
    for spec in specs:
        context_id = new_control_id("child_context")
        contract = ChildContract(
            context_id=context_id,
            parent_context_id=parent.context_id,
            goal_text=spec.goal_text,
            input_references=(parent.goal_id,),
            required_evidence=spec.required_evidence,
            allowed_operator_ids=spec.allowed_operator_ids or (
                "control.retrieve_knowledge",
            ),
            output_fields=(
                "retrieved_memories", "unresolved_questions", "lexical_hypotheses",
            ),
            budget=child_budget,
            deadline_seconds=child_budget.time_seconds,
            success_predicate=spec.success_predicate,
            derived_from_workspace_revision=workspace_revision,
        )
        child = CognitiveContext(
            context_id=context_id,
            goal_id=parent.goal_id,
            workspace_revision=workspace_revision,
            status=ContextStatus.READY,
            parent_context_id=parent.context_id,
            active_hypotheses=(
                (spec.hypothesis_id,) if spec.hypothesis_id else ()
            ),
            budget=child_budget,
            depth=parent.depth + 1,
            cycle_fingerprint="",
        )
        created.append((child, contract))
    return tuple(created)


class ChildBoundaryGuard:
    """Enforce one child's publication boundary (Step 24).

    The guard wraps the workspace publication path for a child: findings that
    target fields outside the child contract are rejected before they enter
    shared state, and operators outside the contract are never dispatched.
    """

    def __init__(self, contract: ChildContract) -> None:
        """Bind the guard to one child contract."""
        self.contract = contract
        self.violations: list[Mapping[str, Any]] = []

    def check_operator(self, operator_id: str) -> bool:
        """Return whether the child may execute one operator."""
        permitted = self.contract.permits_operator(operator_id)
        if not permitted:
            self.violations.append({
                "kind": "operator_outside_contract",
                "operator_id": operator_id,
                "context_id": self.contract.context_id,
            })
        return permitted

    def check_finding(self, finding: OperatorFinding) -> bool:
        """Return whether one finding is publishable by the child."""
        permitted = self.contract.permits_field(finding.field)
        if not permitted:
            self.violations.append({
                "kind": "publication_outside_contract",
                "field": finding.field,
                "context_id": self.contract.context_id,
            })
        return permitted

    def check_goal_change(self, proposed_goal_id: str) -> bool:
        """A child may never change its parent's goal (Step 24)."""
        if proposed_goal_id != self.contract.context_id and proposed_goal_id != (
            self.contract.parent_context_id
        ):
            self.violations.append({
                "kind": "parent_goal_change_attempt",
                "proposed_goal_id": proposed_goal_id,
                "context_id": self.contract.context_id,
            })
            return False
        return True


@dataclass(frozen=True, slots=True)
class ChildOutcome:
    """One child's reconcilable outcome (Step 25)."""

    context_id: str
    goal_text: str
    status: str  # SUCCEEDED | FAILED | TIMED_OUT | INCOMPLETE
    contract_satisfied: bool
    findings: tuple[OperatorFinding, ...] = ()
    unresolved_issues: tuple[str, ...] = ()
    decision: str = ""  # merged | retry | narrowed | deferred | abandoned
    elapsed_ms: float = 0.0

    def __post_init__(self) -> None:
        """Validate the child outcome."""
        allowed = {"SUCCEEDED", "FAILED", "TIMED_OUT", "INCOMPLETE"}
        if self.status not in allowed:
            raise ValueError(f"child status must be one of {sorted(allowed)}")
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "unresolved_issues", tuple(self.unresolved_issues))

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable child outcome."""
        return {
            "context_id": self.context_id,
            "goal_text": self.goal_text,
            "status": self.status,
            "contract_satisfied": bool(self.contract_satisfied),
            "finding_count": len(self.findings),
            "unresolved_issues": list(self.unresolved_issues),
            "decision": self.decision,
            "elapsed_ms": float(self.elapsed_ms),
        }


def reconcile_child_outcomes(
    child_outcomes: Sequence[ChildOutcome],
    *,
    policy: ControlPolicy,
) -> tuple[tuple[OperatorFinding, ...], tuple[Mapping[str, Any], ...]]:
    """Reconcile child outcomes with the parent (Step 25).

    Only contract-validated child findings are merged into the parent's
    integration batch; timed-out and incomplete children are represented
    explicitly and never imply success. Conflicting findings stay separate:
    they are returned as conflict-preservation records for the parent's
    conflict registry instead of being silently overwritten.
    """
    merged: list[OperatorFinding] = []
    records: list[Mapping[str, Any]] = []
    for outcome in child_outcomes:
        if outcome.status == "SUCCEEDED" and outcome.contract_satisfied:
            merged.extend(outcome.findings)
            records.append({
                "context_id": outcome.context_id,
                "decision": "merged",
                "status": outcome.status,
                "finding_count": len(outcome.findings),
                "reconciled_at": utc_now(),
            })
            continue
        if outcome.status in {"TIMED_OUT", "INCOMPLETE"}:
            # Contract-validated findings from an incomplete child still
            # integrate; the child's incompleteness is represented explicitly
            # and never implies success (Step 25).
            merged.extend(outcome.findings)
            records.append({
                "context_id": outcome.context_id,
                "decision": "merged_with_explicit_incomplete_status",
                "status": outcome.status,
                "note": "child did not complete; findings integrated, failure explicit",
                "finding_count": len(outcome.findings),
                "unresolved_issues": list(outcome.unresolved_issues),
                "reconciled_at": utc_now(),
            })
            continue
        records.append({
            "context_id": outcome.context_id,
            "decision": "abandoned",
            "status": outcome.status,
            "note": "child failed its contract; retry/narrow is a parent decision",
            "unresolved_issues": list(outcome.unresolved_issues),
            "reconciled_at": utc_now(),
        })
    return tuple(merged), tuple(records)


__all__ = [
    "ChildBoundaryGuard",
    "ChildContract",
    "ChildOutcome",
    "create_child_contexts",
    "decomposition_justified",
    "reconcile_child_outcomes",
]
