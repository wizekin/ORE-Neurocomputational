"""Typed, versioned planning contracts for ORE Point 12 search and hierarchical planning.

Every record is immutable, validated, and serializable. The contracts follow the
repository conventions used by ore_contracts: frozen slots dataclasses, explicit
validation in __post_init__, and JSON-compatible to_dict() outputs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from ore_contracts.module_manifest import validate_semantic_version

PLANNING_CONTRACTS_VERSION = "1.0.0"
_PREDICATE_NAME = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)*$")
_OPERATOR_ID = re.compile(r"^[a-z][a-z0-9_.-]*$")


def _utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _new_id(prefix: str) -> str:
    """Create a readable, collision-resistant identifier with a domain prefix."""
    return f"{prefix}_{uuid4().hex}"


def canonical_planning_json(value: Any) -> str:
    """Serialize nested planning data deterministically for fingerprints and state hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def planning_fingerprint(value: Any) -> str:
    """Return a SHA-256 fingerprint of a JSON-compatible planning value."""
    return hashlib.sha256(canonical_planning_json(value).encode("utf-8")).hexdigest()


def _require_finite(value: float, name: str, *, minimum: float | None = None) -> None:
    """Reject non-finite and out-of-range numeric contract fields."""
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and float(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


class PlanNodeStatus(str, Enum):
    """Describe the execution state of one plan node."""

    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    UNSATISFIED = "UNSATISFIED"


class DependencyType(str, Enum):
    """Describe why one plan node must precede another."""

    CAUSAL = "causal"
    DATA = "data"
    RESOURCE = "resource"
    ORDERING = "ordering"
    PROTECTION = "protection"


class PlanRunStatus(str, Enum):
    """Report overall plan-run status without ever masking incomplete work."""

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    STOPPED = "STOPPED"


@dataclass(frozen=True, slots=True)
class Predicate:
    """Represent one typed, ground first-order predicate used by goals and effects."""

    name: str
    arguments: tuple[str, ...] = ()
    polarity: bool = True

    def __post_init__(self) -> None:
        """Validate the predicate name and detach caller-owned arguments."""
        if not _PREDICATE_NAME.fullmatch(self.name):
            raise ValueError("predicate name must be a lowercase dotted identifier")
        arguments = tuple(str(item) for item in self.arguments)
        if any(not item.strip() for item in arguments):
            raise ValueError("predicate arguments must be non-empty strings")
        object.__setattr__(self, "arguments", arguments)

    def to_atom(self) -> str:
        """Return the canonical atom string used for fact hashing and lookups."""
        base = f"{self.name}({','.join(self.arguments)})"
        return base if self.polarity else f"not({base})"

    @classmethod
    def from_atom(cls, atom: str) -> Predicate:
        """Reconstruct a predicate from a canonical atom string."""
        text = atom.strip()
        negative = text.startswith("not(") and text.endswith(")")
        if negative:
            text = text[4:-1]
        name, _, remainder = text.partition("(")
        if not remainder.endswith(")"):
            raise ValueError(f"malformed predicate atom: {atom}")
        arguments = tuple(item for item in remainder[:-1].split(",") if item)
        return cls(name=name, arguments=arguments, polarity=not negative)

    def satisfied_by(self, facts: frozenset[str] | set[str]) -> bool:
        """Check this predicate against a set of canonical fact atoms."""
        return (self.to_atom() in facts) if self.polarity else (self.to_atom() not in facts)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible predicate record."""
        return {"name": self.name, "arguments": list(self.arguments), "polarity": self.polarity}


@dataclass(frozen=True, slots=True)
class SuccessCondition:
    """Weight one goal success predicate; critical conditions act as hard gates."""

    predicate: Predicate
    weight: float = 1.0
    critical: bool = False

    def __post_init__(self) -> None:
        """Validate the weight range and require an explicit predicate."""
        if not isinstance(self.predicate, Predicate):
            raise TypeError("predicate must be a Predicate")
        _require_finite(self.weight, "weight", minimum=0.0)


@dataclass(frozen=True, slots=True)
class GoalSpec:
    """Bind a formalized goal to typed predicates, success conditions, and constraints."""

    goal_id: str
    description: str
    predicates: tuple[Predicate, ...]
    success_conditions: tuple[SuccessCondition, ...]
    priority: float = 0.5
    deadline_seconds: float | None = None
    critical_constraints: tuple[str, ...] = ()
    forbidden_operations: tuple[str, ...] = ()
    version: str = PLANNING_CONTRACTS_VERSION
    interpretation_label: str = "primary"

    def __post_init__(self) -> None:
        """Validate goal identity, conditions, and constraint fields."""
        if not self.goal_id.strip() or not self.description.strip():
            raise ValueError("goal_id and description must be non-empty")
        if not self.predicates:
            raise ValueError("a goal requires at least one typed predicate")
        if not self.success_conditions:
            raise ValueError("a goal requires at least one success condition")
        _require_finite(self.priority, "priority")
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError("priority must be in [0, 1]")
        if self.deadline_seconds is not None:
            _require_finite(self.deadline_seconds, "deadline_seconds", minimum=0.0)
        if len(set(self.forbidden_operations)) != len(self.forbidden_operations):
            raise ValueError("forbidden_operations must not contain duplicates")
        validate_semantic_version(self.version, "version")
        object.__setattr__(self, "predicates", tuple(self.predicates))
        object.__setattr__(self, "success_conditions", tuple(self.success_conditions))

    @property
    def critical_predicates(self) -> tuple[Predicate, ...]:
        """Return the hard-gate predicates that no weighted score can compensate for."""
        return tuple(
            condition.predicate
            for condition in self.success_conditions
            if condition.critical
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible goal record."""
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "predicates": [item.to_dict() for item in self.predicates],
            "success_conditions": [
                {
                    "predicate": item.predicate.to_dict(),
                    "weight": item.weight,
                    "critical": item.critical,
                }
                for item in self.success_conditions
            ],
            "priority": self.priority,
            "deadline_seconds": self.deadline_seconds,
            "critical_constraints": list(self.critical_constraints),
            "forbidden_operations": list(self.forbidden_operations),
            "version": self.version,
            "interpretation_label": self.interpretation_label,
        }


@dataclass(frozen=True, slots=True)
class CostVector:
    """Measure one operation along time, memory, risk, uncertainty, and failure recovery."""

    time_ms: float = 1.0
    memory_mb: float = 0.0
    risk: float = 0.0
    uncertainty: float = 0.0
    failure_recovery: float = 0.0

    def __post_init__(self) -> None:
        """Reject non-finite or negative cost components."""
        for name in ("time_ms", "memory_mb", "risk", "uncertainty", "failure_recovery"):
            _require_finite(getattr(self, name), name, minimum=0.0)
        for name in ("risk", "uncertainty", "failure_recovery"):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must be normalized to [0, 1]")

    def to_dict(self) -> dict[str, float]:
        """Return the five measured cost components."""
        return {
            "time_ms": self.time_ms,
            "memory_mb": self.memory_mb,
            "risk": self.risk,
            "uncertainty": self.uncertainty,
            "failure_recovery": self.failure_recovery,
        }


@dataclass(frozen=True, slots=True)
class CostWeights:
    """Version the relative weights applied to the five cost components."""

    time: float = 0.35
    memory: float = 0.15
    risk: float = 0.2
    uncertainty: float = 0.15
    failure_recovery: float = 0.15
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        """Require finite, non-negative weights and a semantic version."""
        for name in ("time", "memory", "risk", "uncertainty", "failure_recovery"):
            _require_finite(getattr(self, name), name, minimum=0.0)
        validate_semantic_version(self.version, "version")

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned weight record."""
        return {
            "time": self.time, "memory": self.memory, "risk": self.risk,
            "uncertainty": self.uncertainty, "failure_recovery": self.failure_recovery,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    """Declare per-operator resource units and exclusively-held resources."""

    required_units: Mapping[str, float] = field(default_factory=dict)
    exclusive_resources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate resource keys and detach caller-owned mappings."""
        units = dict(self.required_units)
        for key, value in units.items():
            if not key.strip():
                raise ValueError("resource unit keys must be non-empty")
            _require_finite(value, f"resource unit {key}", minimum=0.0)
        if len(set(self.exclusive_resources)) != len(self.exclusive_resources):
            raise ValueError("exclusive_resources must not contain duplicates")
        if any(not item.strip() for item in self.exclusive_resources):
            raise ValueError("exclusive resources must be non-empty")
        object.__setattr__(self, "required_units", units)
        object.__setattr__(self, "exclusive_resources", tuple(self.exclusive_resources))

    def to_dict(self) -> dict[str, Any]:
        """Return the detached resource specification."""
        return {
            "required_units": dict(self.required_units),
            "exclusive_resources": list(self.exclusive_resources),
        }


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """Describe one executable cognitive operation with preconditions, effects, and cost."""

    operator_id: str
    preconditions: tuple[Predicate, ...]
    effects: tuple[Predicate, ...]
    input_schema: Mapping[str, tuple[str, ...]]
    output_schema: Mapping[str, tuple[str, ...]]
    module_capability: str
    estimated_cost: CostVector = field(default_factory=CostVector)
    risk: float = 0.0
    resource_requirements: ResourceSpec = field(default_factory=ResourceSpec)
    version: str = PLANNING_CONTRACTS_VERSION
    module_id: str | None = None
    execution_modes: tuple[str, ...] = ("FUNCTIONAL",)

    def __post_init__(self) -> None:
        """Validate operator identity, schemas, and normalized risk."""
        if not _OPERATOR_ID.fullmatch(self.operator_id):
            raise ValueError("operator_id must be a stable lowercase identifier")
        if not self.module_capability.strip():
            raise ValueError("module_capability must be non-empty")
        _require_finite(self.risk, "risk")
        if not 0.0 <= self.risk <= 1.0:
            raise ValueError("risk must be normalized to [0, 1]")
        input_schema = {key: tuple(value) for key, value in self.input_schema.items()}
        output_schema = {key: tuple(value) for key, value in self.output_schema.items()}
        for schema in (input_schema, output_schema):
            for port_id, versions in schema.items():
                if not port_id.strip() or not versions:
                    raise ValueError("schema ports require an id and at least one version")
                for version in versions:
                    validate_semantic_version(version, "schema version")
        if len(set(self.preconditions)) != len(self.preconditions):
            raise ValueError("preconditions must not contain duplicates")
        if len(set(self.effects)) != len(self.effects):
            raise ValueError("effects must not contain duplicates")
        validate_semantic_version(self.version, "version")
        object.__setattr__(self, "input_schema", input_schema)
        object.__setattr__(self, "output_schema", output_schema)
        object.__setattr__(self, "preconditions", tuple(self.preconditions))
        object.__setattr__(self, "effects", tuple(self.effects))
        object.__setattr__(self, "execution_modes", tuple(self.execution_modes))

    def provides(self, fact: str) -> bool:
        """Check whether a positive effect of this operator establishes a fact atom."""
        return any(effect.to_atom() == fact for effect in self.effects if effect.polarity)

    def negates(self, fact: str) -> bool:
        """Check whether a negative effect of this operator removes a fact atom."""
        return any(effect.to_atom() == f"not({fact})" for effect in self.effects)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible operator record."""
        return {
            "operator_id": self.operator_id,
            "preconditions": [item.to_dict() for item in self.preconditions],
            "effects": [item.to_dict() for item in self.effects],
            "input_schema": {key: list(value) for key, value in self.input_schema.items()},
            "output_schema": {key: list(value) for key, value in self.output_schema.items()},
            "module_capability": self.module_capability,
            "estimated_cost": self.estimated_cost.to_dict(),
            "risk": self.risk,
            "resource_requirements": self.resource_requirements.to_dict(),
            "version": self.version,
            "module_id": self.module_id,
            "execution_modes": list(self.execution_modes),
        }


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Declare one typed task with an explicit completion predicate."""

    task_id: str
    task_type: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    completion: Predicate | None = None
    compound: bool = False
    operator_id: str | None = None

    def __post_init__(self) -> None:
        """Validate task identity, typing, and completion semantics."""
        if not self.task_id.strip() or not self.task_type.strip():
            raise ValueError("task_id and task_type must be non-empty")
        if self.compound and self.operator_id is not None:
            raise ValueError("a compound task cannot bind an operator directly")
        if not self.compound and self.completion is None:
            raise ValueError("primitive tasks require a completion predicate")
        object.__setattr__(self, "parameters", copy.deepcopy(dict(self.parameters)))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible task record."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "parameters": copy.deepcopy(dict(self.parameters)),
            "completion": self.completion.to_dict() if self.completion else None,
            "compound": self.compound,
            "operator_id": self.operator_id,
        }


@dataclass(frozen=True, slots=True)
class HTNMethod:
    """Version one hierarchical decomposition from a task type to ordered subtasks."""

    method_id: str
    task_type: str
    applicability_conditions: tuple[Predicate, ...]
    subtasks: tuple[TaskSpec, ...]
    ordering: str = "partial"
    constraints: tuple[str, ...] = ()
    version: str = "1.0.0"
    priority: float = 0.5

    def __post_init__(self) -> None:
        """Validate method identity, ordering semantics, and version."""
        if not self.method_id.strip() or not self.task_type.strip():
            raise ValueError("method_id and task_type must be non-empty")
        if self.ordering not in {"ordered", "partial"}:
            raise ValueError("ordering must be 'ordered' or 'partial'")
        if not self.subtasks:
            raise ValueError("an HTN method must declare at least one subtask")
        if any(not item.strip() for item in self.constraints):
            raise ValueError("constraints must be non-empty strings")
        _require_finite(self.priority, "priority")
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError("method priority must be in [0, 1]")
        if len(set(self.applicability_conditions)) != len(self.applicability_conditions):
            raise ValueError("applicability conditions must not contain duplicates")
        validate_semantic_version(self.version, "version")
        object.__setattr__(self, "subtasks", tuple(self.subtasks))
        object.__setattr__(self, "applicability_conditions", tuple(self.applicability_conditions))

    def applicable_to(self, facts: frozenset[str] | set[str]) -> bool:
        """Check every applicability condition against the current fact set."""
        return all(condition.satisfied_by(facts) for condition in self.applicability_conditions)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible method record."""
        return {
            "method_id": self.method_id,
            "task_type": self.task_type,
            "applicability_conditions": [
                item.to_dict() for item in self.applicability_conditions
            ],
            "subtasks": [item.to_dict() for item in self.subtasks],
            "ordering": self.ordering,
            "constraints": list(self.constraints),
            "version": self.version,
            "priority": self.priority,
        }


@dataclass(frozen=True, slots=True)
class DecompositionProvenance:
    """Attach method/version, applicability facts, and the parent task to one expansion."""

    method_id: str
    method_version: str
    applicability_facts: tuple[str, ...]
    parent_task_type: str

    def __post_init__(self) -> None:
        """Validate provenance references."""
        if not self.method_id.strip() or not self.parent_task_type.strip():
            raise ValueError("provenance method and parent task must be non-empty")
        validate_semantic_version(self.method_version, "method_version")
        object.__setattr__(self, "applicability_facts", tuple(sorted(self.applicability_facts)))

    def to_dict(self) -> dict[str, Any]:
        """Return the provenance record."""
        return {
            "method_id": self.method_id,
            "method_version": self.method_version,
            "applicability_facts": list(self.applicability_facts),
            "parent_task_type": self.parent_task_type,
        }


@dataclass(frozen=True, slots=True)
class PlanNode:
    """Hold one executable plan step with status, expected effects, and cost estimate."""

    node_id: str
    task: TaskSpec
    operator_id: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    status: PlanNodeStatus = PlanNodeStatus.PENDING
    expected_effects: tuple[Predicate, ...] = ()
    cost_estimate: CostVector = field(default_factory=CostVector)
    provenance: DecompositionProvenance | None = None

    def __post_init__(self) -> None:
        """Validate node identity and normalize status and parameters."""
        if not self.node_id.strip():
            raise ValueError("node_id must be non-empty")
        if not isinstance(self.task, TaskSpec):
            raise TypeError("task must be a TaskSpec")
        if not isinstance(self.status, PlanNodeStatus):
            object.__setattr__(self, "status", PlanNodeStatus(self.status))
        object.__setattr__(self, "parameters", copy.deepcopy(dict(self.parameters)))
        object.__setattr__(self, "expected_effects", tuple(self.expected_effects))

    def with_status(self, status: PlanNodeStatus) -> PlanNode:
        """Return a copy of this node with an updated execution status."""
        return PlanNode(
            self.node_id, self.task, self.operator_id, self.parameters,
            status, self.expected_effects, self.cost_estimate, self.provenance,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible node record."""
        return {
            "node_id": self.node_id,
            "task": self.task.to_dict(),
            "operator_id": self.operator_id,
            "parameters": copy.deepcopy(dict(self.parameters)),
            "status": self.status.value,
            "expected_effects": [item.to_dict() for item in self.expected_effects],
            "cost_estimate": self.cost_estimate.to_dict(),
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }


@dataclass(frozen=True, slots=True)
class PlanEdge:
    """Record why one plan node must precede another."""

    predecessor: str
    successor: str
    dependency_type: DependencyType = DependencyType.CAUSAL
    causal_link: str | None = None

    def __post_init__(self) -> None:
        """Validate endpoint identity and normalize the dependency type."""
        if not self.predecessor.strip() or not self.successor.strip():
            raise ValueError("edge endpoints must be non-empty node IDs")
        if self.predecessor == self.successor:
            raise ValueError("an edge cannot point to its own node")
        if not isinstance(self.dependency_type, DependencyType):
            object.__setattr__(
                self, "dependency_type", DependencyType(self.dependency_type),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the edge record with its causal link atom."""
        return {
            "predecessor": self.predecessor,
            "successor": self.successor,
            "dependency_type": self.dependency_type.value,
            "causal_link": self.causal_link,
        }


@dataclass(frozen=True, slots=True)
class PlanState:
    """Bind one search state to a workspace revision, facts, and pending work."""

    workspace_revision: int
    achieved_facts: frozenset[str]
    pending_tasks: tuple[str, ...]
    resources: Mapping[str, float] = field(default_factory=dict)
    hypotheses: Mapping[str, Any] = field(default_factory=dict)
    plan_version: str = PLANNING_CONTRACTS_VERSION

    def __post_init__(self) -> None:
        """Validate revision binding and detach mutable mappings."""
        if self.workspace_revision < 0:
            raise ValueError("workspace_revision cannot be negative")
        object.__setattr__(self, "achieved_facts", frozenset(self.achieved_facts))
        object.__setattr__(self, "pending_tasks", tuple(self.pending_tasks))
        object.__setattr__(self, "resources", dict(self.resources))
        object.__setattr__(self, "hypotheses", copy.deepcopy(dict(self.hypotheses)))

    def state_key(self) -> str:
        """Return the deterministic hash used for duplicate detection and dominance."""
        return planning_fingerprint({
            "revision": self.workspace_revision,
            "facts": sorted(self.achieved_facts),
            "pending": list(self.pending_tasks),
            "plan_version": self.plan_version,
        })

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible state record."""
        return {
            "workspace_revision": self.workspace_revision,
            "achieved_facts": sorted(self.achieved_facts),
            "pending_tasks": list(self.pending_tasks),
            "resources": dict(self.resources),
            "hypotheses": copy.deepcopy(dict(self.hypotheses)),
            "plan_version": self.plan_version,
        }


@dataclass(frozen=True, slots=True)
class SearchRecord:
    """Store one search-tree expansion with its evaluated costs and chosen operator."""

    state_hash: str
    g_cost: float
    h_cost: float
    parent: str | None
    chosen_operator: str | None
    depth: int

    def __post_init__(self) -> None:
        """Validate search record costs and depth."""
        _require_finite(self.g_cost, "g_cost", minimum=0.0)
        _require_finite(self.h_cost, "h_cost", minimum=0.0)
        if self.depth < 0:
            raise ValueError("depth cannot be negative")

    @property
    def f_cost(self) -> float:
        """Return f(n) = g(n) + h(n) for this record."""
        return self.g_cost + self.h_cost

    def to_dict(self) -> dict[str, Any]:
        """Return the search record."""
        return {
            "state_hash": self.state_hash,
            "g_cost": self.g_cost,
            "h_cost": self.h_cost,
            "f_cost": self.f_cost,
            "parent": self.parent,
            "chosen_operator": self.chosen_operator,
            "depth": self.depth,
        }


@dataclass(frozen=True, slots=True)
class PlanDecision:
    """Explain the selected plan, rejected alternatives, budget, and heuristic version."""

    selected_plan: str
    rejected_alternatives: tuple[tuple[str, str], ...]
    rationale: str
    budget: Mapping[str, Any] = field(default_factory=dict)
    heuristic_version: str = "unspecified"
    optimality_claim: bool = False
    search_mode: str = "astar"

    def __post_init__(self) -> None:
        """Validate alternative records and detach the budget mapping."""
        if not self.selected_plan.strip():
            raise ValueError("selected_plan must be a non-empty plan identifier")
        object.__setattr__(
            self, "rejected_alternatives",
            tuple((str(key), str(value)) for key, value in self.rejected_alternatives),
        )
        object.__setattr__(self, "budget", dict(self.budget))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible decision record."""
        return {
            "selected_plan": self.selected_plan,
            "rejected_alternatives": [
                {"plan": key, "rationale": value} for key, value in self.rejected_alternatives
            ],
            "rationale": self.rationale,
            "budget": dict(self.budget),
            "heuristic_version": self.heuristic_version,
            "optimality_claim": self.optimality_claim,
            "search_mode": self.search_mode,
        }


@dataclass(frozen=True, slots=True)
class ExecutionFeedback:
    """Compare one executed task's actual outcome against its plan prediction."""

    task_id: str
    node_id: str | None
    operator_id: str | None
    actual_cost: CostVector
    predicted_cost: CostVector
    outputs: tuple[str, ...]
    success: bool
    failure_class: str | None = None
    evidence_refs: tuple[str, ...] = ()
    module_id: str | None = None

    def __post_init__(self) -> None:
        """Validate feedback identity and failure reporting."""
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if self.success and self.failure_class is not None:
            raise ValueError("a successful task cannot carry a failure class")
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))

    @property
    def cost_deviation(self) -> float:
        """Return the scalar difference between actual and predicted cost totals."""
        actual = sum(self.actual_cost.to_dict().values())
        predicted = sum(self.predicted_cost.to_dict().values())
        return actual - predicted

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible feedback record."""
        return {
            "task_id": self.task_id,
            "node_id": self.node_id,
            "operator_id": self.operator_id,
            "actual_cost": self.actual_cost.to_dict(),
            "predicted_cost": self.predicted_cost.to_dict(),
            "cost_deviation": self.cost_deviation,
            "outputs": list(self.outputs),
            "success": self.success,
            "failure_class": self.failure_class,
            "evidence_refs": list(self.evidence_refs),
            "module_id": self.module_id,
        }


@dataclass(frozen=True, slots=True)
class PlannerBudget:
    """Bound planner nodes, search effort, plan cost, and decomposition size."""

    max_search_nodes: int = 2000
    max_time_seconds: float = 30.0
    max_plan_cost: float = 1000.0
    max_depth: int = 12
    max_htn_nodes: int = 500
    beam_width: int = 5

    def __post_init__(self) -> None:
        """Reject non-positive planner limits."""
        for name in (
            "max_search_nodes", "max_time_seconds", "max_plan_cost",
            "max_depth", "max_htn_nodes", "beam_width",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return the enforced budget values."""
        return {
            "max_search_nodes": self.max_search_nodes,
            "max_time_seconds": self.max_time_seconds,
            "max_plan_cost": self.max_plan_cost,
            "max_depth": self.max_depth,
            "max_htn_nodes": self.max_htn_nodes,
            "beam_width": self.beam_width,
        }


@dataclass(frozen=True, slots=True)
class PlanCheckpoint:
    """Persist completed work so resume never duplicates executions or publications."""

    plan_id: str
    goal_id: str
    workspace_revision: int
    completed_node_ids: tuple[str, ...]
    completed_module_results: Mapping[str, Any] = field(default_factory=dict)
    feedback: tuple[ExecutionFeedback, ...] = ()
    created_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        """Validate checkpoint identity and detach stored mappings."""
        if not self.plan_id.strip() or not self.goal_id.strip():
            raise ValueError("checkpoint plan and goal identifiers must be non-empty")
        if self.workspace_revision < 0:
            raise ValueError("workspace_revision cannot be negative")
        object.__setattr__(self, "completed_node_ids", tuple(self.completed_node_ids))
        object.__setattr__(self, "completed_module_results", dict(self.completed_module_results))
        object.__setattr__(self, "feedback", tuple(self.feedback))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible checkpoint record."""
        return {
            "plan_id": self.plan_id,
            "goal_id": self.goal_id,
            "workspace_revision": self.workspace_revision,
            "completed_node_ids": list(self.completed_node_ids),
            "completed_module_results": {
                key: getattr(value, "to_dict", dict)() if hasattr(value, "to_dict")
                else str(value)
                for key, value in self.completed_module_results.items()
            },
            "feedback": [item.to_dict() for item in self.feedback],
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class PlanRunResult:
    """Report plan-run status, progress, unresolved work, and termination reason."""

    plan_id: str
    status: PlanRunStatus
    goal_progress_score: float
    critical_satisfied: bool
    unresolved_subgoals: tuple[str, ...]
    termination_reason: str
    feedback: tuple[ExecutionFeedback, ...] = ()
    replans: int = 0
    workspace_revision: int = 0
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate run status and normalize enum fields."""
        if not isinstance(self.status, PlanRunStatus):
            object.__setattr__(self, "status", PlanRunStatus(self.status))
        _require_finite(self.goal_progress_score, "goal_progress_score")
        if not 0.0 <= self.goal_progress_score <= 1.0:
            raise ValueError("goal_progress_score must be in [0, 1]")
        object.__setattr__(self, "unresolved_subgoals", tuple(self.unresolved_subgoals))
        object.__setattr__(self, "feedback", tuple(self.feedback))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        if self.status is PlanRunStatus.COMPLETE:
            if not self.critical_satisfied:
                raise ValueError("an incomplete goal can never be reported as COMPLETE")
            if self.unresolved_subgoals:
                raise ValueError("a COMPLETE run cannot carry unresolved subgoals")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible run result."""
        return {
            "plan_id": self.plan_id,
            "status": self.status.value,
            "goal_progress_score": self.goal_progress_score,
            "critical_satisfied": self.critical_satisfied,
            "unresolved_subgoals": list(self.unresolved_subgoals),
            "termination_reason": self.termination_reason,
            "feedback": [item.to_dict() for item in self.feedback],
            "replans": self.replans,
            "workspace_revision": self.workspace_revision,
            "evidence_refs": list(self.evidence_refs),
        }


__all__ = [
    "PLANNING_CONTRACTS_VERSION",
    "CostVector",
    "CostWeights",
    "DecompositionProvenance",
    "DependencyType",
    "ExecutionFeedback",
    "GoalSpec",
    "HTNMethod",
    "OperatorSpec",
    "PlanCheckpoint",
    "PlanDecision",
    "PlanEdge",
    "PlanNode",
    "PlanNodeStatus",
    "PlanRunResult",
    "PlanRunStatus",
    "PlanState",
    "PlannerBudget",
    "Predicate",
    "ResourceSpec",
    "SearchRecord",
    "SuccessCondition",
    "TaskSpec",
    "canonical_planning_json",
    "planning_fingerprint",
]
