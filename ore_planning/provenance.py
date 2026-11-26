"""Provenance records and the 40-step planner trace for auditability.

Every planner run records which of the forty specification steps executed, with
what outcome, correlated by goal, plan, task, module, and workspace revision.
Records are kept in memory as typed structures and optionally mirrored into the
shared ore_logger SQLite event store using the same annotation-linked pattern as
ore_contracts components.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .contracts import _new_id, _utc_now_iso

PLANNER_STEPS: tuple[tuple[str, int, str], ...] = (
    ("A", 1, "Receive goal, scope, deadline, quality threshold, and resource budget"),
    ("A", 2, "Bind to a consistent workspace revision and record the goal/version"),
    ("A", 3, "Convert the goal into typed predicates and explicit success conditions"),
    ("A", 4, "Preserve ambiguous interpretations as bounded alternatives"),
    ("A", 5, "Identify hard constraints, forbidden operations, and critical conditions"),
    ("A", 6, "Query the module registry for capability- and schema-matching operations"),
    ("A", 7, "Normalize operator preconditions, effects, schemas, costs, and resources"),
    ("A", 8, "Reject operators with incompatible schemas or missing capabilities"),
    ("B", 9, "Select an applicable HTN method by task type, context, and priority"),
    ("B", 10, "Expand the task into typed subtasks with completion predicates"),
    ("B", 11, "Recursively decompose compound subtasks to primitive operations"),
    ("B", 12, "Detect decomposition cycles and enforce depth and node limits"),
    ("B", 13, "Check method preconditions before accepting each decomposition"),
    ("B", 14, "Preserve alternative decompositions with ranking and rationale"),
    ("B", 15, "Attach provenance to every decomposition choice"),
    ("B", 16, "Verify each leaf maps to an eligible capability or mark it unsatisfied"),
    ("C", 17, "Build the directed dependency graph from preconditions and causal links"),
    ("C", 18, "Leave independent operations unordered for concurrency"),
    ("C", 19, "Detect mutex/resource conflicts and serialize only conflicting operations"),
    ("C", 20, "Check graph acyclicity and identify ready tasks"),
    ("C", 21, "Add protection conditions for consumed intermediate results"),
    ("D", 22, "Define the planner state from revision, facts, and pending tasks"),
    ("D", 23, "Define successor generation from applicable operators and methods"),
    ("D", 24, "Run bounded cost-aware A* search when costs are defined"),
    ("D", 25, "Compute g(n) from measured costs and h(n) from a documented heuristic"),
    ("D", 26, "Require admissibility for optimality claims, else label best-effort"),
    ("D", 27, "Use beam search for large branching with recorded pruning"),
    ("D", 28, "Apply duplicate detection, dominance checks, and cycle prevention"),
    ("E", 29, "Select the plan that satisfies hard constraints and policy best"),
    ("E", 30, "Submit ready tasks to the existing coordinator with dependencies"),
    ("E", 31, "Allow concurrent execution of independent ready tasks"),
    ("E", 32, "Integrate outputs through workspace publication and revision validation"),
    ("E", 33, "Update achieved predicates only from accepted outputs"),
    ("E", 34, "Compare actual cost, latency, and success with predictions"),
    ("E", 35, "Recompute preconditions and goal progress after each revision"),
    ("E", 36, "Replan on failure, evidence change, budget change, or better plans"),
    ("E", 37, "Stop on verified success, budget limits, no plan, or human decision"),
    ("E", 38, "Return partial status and unresolved subgoals on early termination"),
    ("E", 39, "Persist plan, checkpoint, decisions, and feedback for reproducibility"),
    ("E", 40, "Persist calibration data for future cost estimates"),
)
_STEP_INDEX = {(phase, number): description for phase, number, description in PLANNER_STEPS}


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """Record the outcome of one planner step for one run."""

    phase: str
    step: int
    description: str
    outcome: str

    def __post_init__(self) -> None:
        """Validate the step reference against the specification registry."""
        if (self.phase, self.step) not in _STEP_INDEX:
            raise ValueError(f"unknown planner step: {self.phase}{self.step}")
        if _STEP_INDEX[(self.phase, self.step)] != self.description:
            raise ValueError("step description does not match the specification registry")

    def to_dict(self) -> dict[str, Any]:
        """Return the step outcome record."""
        return {
            "phase": self.phase,
            "step": self.step,
            "description": self.description,
            "outcome": self.outcome,
        }


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Correlate one planning event with its goal, plan, modules, and revision."""

    record_id: str
    event: str
    goal_id: str | None
    plan_id: str | None
    node_ids: tuple[str, ...]
    module_ids: tuple[str, ...]
    workspace_revision: int
    payload: Mapping[str, Any]
    timestamp: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        """Normalize correlation identifiers."""
        object.__setattr__(self, "node_ids", tuple(self.node_ids))
        object.__setattr__(self, "module_ids", tuple(self.module_ids))
        object.__setattr__(self, "payload", dict(self.payload))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible provenance record."""
        return {
            "record_id": self.record_id,
            "event": self.event,
            "goal_id": self.goal_id,
            "plan_id": self.plan_id,
            "node_ids": list(self.node_ids),
            "module_ids": list(self.module_ids),
            "workspace_revision": self.workspace_revision,
            "payload": dict(self.payload),
            "timestamp": self.timestamp,
        }


class PlanningProvenance:
    """Collect step outcomes and correlated provenance records for one planner run."""

    def __init__(self) -> None:
        """Initialize empty, append-only collections."""
        self._steps: list[StepOutcome] = []
        self._records: list[ProvenanceRecord] = []

    def record_step(self, phase: str, step: int, outcome: str) -> StepOutcome:
        """Append one step outcome and return it."""
        entry = StepOutcome(
            phase=phase, step=step,
            description=_STEP_INDEX[(phase, step)], outcome=outcome,
        )
        self._steps.append(entry)
        return entry

    def record(
        self,
        event: str,
        *,
        goal_id: str | None = None,
        plan_id: str | None = None,
        node_ids: Iterable[str] = (),
        module_ids: Iterable[str] = (),
        workspace_revision: int = 0,
        payload: Mapping[str, Any] | None = None,
    ) -> ProvenanceRecord:
        """Append one correlated provenance record and return it."""
        entry = ProvenanceRecord(
            record_id=_new_id("planprov"),
            event=event,
            goal_id=goal_id,
            plan_id=plan_id,
            node_ids=tuple(node_ids),
            module_ids=tuple(module_ids),
            workspace_revision=workspace_revision,
            payload=payload or {},
        )
        self._records.append(entry)
        return entry

    @property
    def steps(self) -> tuple[StepOutcome, ...]:
        """Return every recorded step outcome in execution order."""
        return tuple(self._steps)

    @property
    def records(self) -> tuple[ProvenanceRecord, ...]:
        """Return every provenance record in append order."""
        return tuple(self._records)

    def for_goal(self, goal_id: str) -> tuple[ProvenanceRecord, ...]:
        """Return every record correlated with one goal."""
        return tuple(item for item in self._records if item.goal_id == goal_id)

    def for_plan(self, plan_id: str) -> tuple[ProvenanceRecord, ...]:
        """Return every record correlated with one plan."""
        return tuple(item for item in self._records if item.plan_id == plan_id)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible provenance trace."""
        return {
            "steps": [item.to_dict() for item in self._steps],
            "records": [item.to_dict() for item in self._records],
        }


class PlanningEventLogger:
    """Mirror provenance events into the shared ore_logger SQLite store.

    The bridge reuses ore_contracts.SQLiteModuleEventLogger so planner events land
    in the same runs/events/annotations tables as registry and coordinator events,
    with method-linked annotations for ore_planning classes.
    """

    def __init__(self, logger: Any) -> None:
        """Wrap one already-open SQLiteModuleEventLogger context."""
        self._logger = logger

    def record(
        self,
        method_name: str,
        metadata: Mapping[str, Any],
        *,
        class_name: str = "HierarchicalPlanner",
        module_name: str = "ore_planning.planner",
    ) -> dict[str, Any] | None:
        """Write one event when the underlying logger is available."""
        if self._logger is None:
            return None
        return self._logger.record(
            method_name, dict(metadata), class_name=class_name, module_name=module_name,
        )


__all__ = [
    "PLANNER_STEPS",
    "PlanningEventLogger",
    "PlanningProvenance",
    "ProvenanceRecord",
    "StepOutcome",
]
