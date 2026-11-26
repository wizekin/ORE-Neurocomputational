"""Hierarchical task network decomposition with versioned methods and bounded expansion.

The decomposer expands high-level goals into typed subtasks using versioned
decomposition methods. It preserves alternative decompositions when several
methods apply, attaches provenance to every expansion, detects cycles and
repeated subgoals, and enforces explicit depth and node budgets. Primitive
leaves are capability-bound tasks; a leaf is accepted only when the capability
maps to at least one eligible registered operator, otherwise it is explicitly
marked UNSATISFIED. The concrete operator for a leaf is selected later by the
search engine, so provider choice is never prematurely committed here.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .capability_adapter import OperatorDiscovery
from .contracts import (
    DecompositionProvenance,
    GoalSpec,
    HTNMethod,
    PlannerBudget,
    PlanNodeStatus,
    Predicate,
    TaskSpec,
    planning_fingerprint,
)

GOAL_TASK_TYPE = "goal.achieve"
_MAX_ALTERNATIVES = 8


@dataclass(frozen=True, slots=True)
class TaskTree:
    """Hold one decomposition node with its children and expansion provenance."""

    task: TaskSpec
    children: tuple[TaskTree, ...] = ()
    provenance: DecompositionProvenance | None = None
    status: PlanNodeStatus = PlanNodeStatus.PENDING
    depth: int = 0

    def __post_init__(self) -> None:
        """Normalize children, status, and depth."""
        object.__setattr__(self, "children", tuple(self.children))
        if not isinstance(self.status, PlanNodeStatus):
            object.__setattr__(self, "status", PlanNodeStatus(self.status))
        if self.depth < 0:
            raise ValueError("depth cannot be negative")

    def leaves(self) -> tuple[TaskTree, ...]:
        """Return every primitive leaf of this subtree in deterministic order."""
        if not self.children:
            return (self,)
        collected: list[TaskTree] = []
        for child in self.children:
            collected.extend(child.leaves())
        return tuple(collected)

    def node_count(self) -> int:
        """Count every node in this subtree including the root."""
        return 1 + sum(child.node_count() for child in self.children)

    def to_dict(self) -> dict:
        """Return a detached, JSON-compatible tree record."""
        return {
            "task": self.task.to_dict(),
            "children": [child.to_dict() for child in self.children],
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "status": self.status.value,
            "depth": self.depth,
        }


@dataclass(frozen=True, slots=True)
class DecompositionResult:
    """Expose alternative decompositions, rejections, and budget diagnostics."""

    alternatives: tuple[TaskTree, ...]
    rejected_methods: tuple[tuple[str, str], ...]
    unsatisfied_leaves: tuple[str, ...]
    cycle_count: int
    node_count: int
    truncated: bool

    def __post_init__(self) -> None:
        """Normalize decomposition diagnostics."""
        object.__setattr__(self, "alternatives", tuple(self.alternatives))
        object.__setattr__(
            self, "rejected_methods",
            tuple((str(key), str(value)) for key, value in self.rejected_methods),
        )
        object.__setattr__(self, "unsatisfied_leaves", tuple(self.unsatisfied_leaves))

    @property
    def primary(self) -> TaskTree | None:
        """Return the best-ranked alternative, if any decomposition succeeded."""
        return self.alternatives[0] if self.alternatives else None

    def to_dict(self) -> dict:
        """Return a detached, JSON-compatible result record."""
        return {
            "alternatives": [item.to_dict() for item in self.alternatives],
            "rejected_methods": [
                {"method": key, "reason": value} for key, value in self.rejected_methods
            ],
            "unsatisfied_leaves": list(self.unsatisfied_leaves),
            "cycle_count": self.cycle_count,
            "node_count": self.node_count,
            "truncated": self.truncated,
        }


class HTNDecomposer:
    """Expand typed tasks recursively with cycle, depth, and node budget guards."""

    def __init__(
        self,
        methods: Iterable[HTNMethod],
        *,
        budget: PlannerBudget | None = None,
    ) -> None:
        """Store the versioned method library indexed by task type."""
        self.methods = tuple(methods)
        seen: set[str] = set()
        for method in self.methods:
            key = f"{method.method_id}@{method.version}"
            if key in seen:
                raise ValueError(f"duplicate method registration: {key}")
            seen.add(key)
        self.budget = budget or PlannerBudget()

    def methods_for(self, task_type: str) -> tuple[HTNMethod, ...]:
        """Return methods for one task type ordered by priority then identifier."""
        candidates = [method for method in self.methods if method.task_type == task_type]
        return tuple(sorted(candidates, key=lambda item: (-item.priority, item.method_id)))

    @staticmethod
    def _task_signature(task: TaskSpec) -> str:
        """Fingerprint a task for repeated-subgoal and cycle detection."""
        return planning_fingerprint({
            "task_type": task.task_type,
            "parameters": task.parameters,
        })

    def decompose(
        self,
        task: TaskSpec,
        facts: frozenset[str] | set[str],
        discovery: OperatorDiscovery,
    ) -> DecompositionResult:
        """Expand one task into ranked alternative decompositions."""
        alternatives, rejections, unsatisfied, cycles, nodes, truncated = self._expand(
            task, frozenset(facts), discovery, chain=(), depth=0,
        )
        if not alternatives:
            alternatives = (TaskTree(task=task, status=PlanNodeStatus.UNSATISFIED),)
        return DecompositionResult(
            alternatives=tuple(alternatives[:_MAX_ALTERNATIVES]),
            rejected_methods=tuple(rejections),
            unsatisfied_leaves=tuple(sorted(set(unsatisfied))),
            cycle_count=cycles,
            node_count=nodes,
            truncated=truncated,
        )

    def _expand(
        self,
        task: TaskSpec,
        facts: frozenset[str],
        discovery: OperatorDiscovery,
        *,
        chain: tuple[str, ...],
        depth: int,
    ) -> tuple[list[TaskTree], list[tuple[str, str]], list[str], int, int, bool]:
        """Recursively expand one task under cycle, depth, and node guards."""
        node_count = 1
        if not task.compound:
            eligible = discovery.operators_for_capability(task.task_type)
            if not eligible:
                return (
                    [TaskTree(task=task, status=PlanNodeStatus.UNSATISFIED, depth=depth)],
                    [], [task.task_id], 0, node_count, False,
                )
            return ([TaskTree(task=task, depth=depth)], [], [], 0, node_count, False)
        signature = self._task_signature(task)
        if signature in chain:
            return ([], [], [], 1, node_count, False)
        if depth >= self.budget.max_depth:
            return (
                [TaskTree(task=task, status=PlanNodeStatus.UNSATISFIED, depth=depth)],
                [], [task.task_id], 0, node_count, True,
            )
        alternatives: list[TaskTree] = []
        rejections: list[tuple[str, str]] = []
        unsatisfied: list[str] = []
        cycle_count = 0
        truncated = False
        for method in self.methods_for(task.task_type):
            if node_count >= self.budget.max_htn_nodes:
                truncated = True
                break
            if not method.applicable_to(facts):
                rejections.append((
                    f"{method.method_id}@{method.version}",
                    "applicability conditions not satisfied",
                ))
                continue
            child_trees: list[TaskTree] = []
            child_nodes = 0
            child_failed = False
            for subtask in method.subtasks:
                if node_count + child_nodes >= self.budget.max_htn_nodes:
                    truncated = True
                    child_failed = True
                    break
                sub = self._expand(
                    subtask, facts, discovery,
                    chain=chain + (signature,), depth=depth + 1,
                )
                sub_alternatives, sub_rejections, sub_unsatisfied, sub_cycles, sub_nodes, sub_truncated = sub
                child_nodes += sub_nodes
                cycle_count += sub_cycles
                truncated = truncated or sub_truncated
                rejections.extend(sub_rejections)
                unsatisfied.extend(sub_unsatisfied)
                if not sub_alternatives:
                    child_failed = True
                    break
                child_trees.append(sub_alternatives[0])
            node_count += child_nodes
            if child_failed:
                continue
            alternatives.append(TaskTree(
                task=task,
                children=tuple(child_trees),
                provenance=DecompositionProvenance(
                    method_id=method.method_id,
                    method_version=method.version,
                    applicability_facts=tuple(
                        fact for fact in sorted(facts)
                        if any(condition.to_atom() == fact
                               for condition in method.applicability_conditions)
                    ),
                    parent_task_type=task.task_type,
                ),
                depth=depth,
            ))
            if len(alternatives) >= _MAX_ALTERNATIVES:
                break
        if not alternatives:
            alternatives = [TaskTree(task=task, status=PlanNodeStatus.UNSATISFIED, depth=depth)]
            unsatisfied.append(task.task_id)
        return alternatives, rejections, unsatisfied, cycle_count, node_count, truncated


def build_default_methods(goal: GoalSpec, discovery: OperatorDiscovery) -> tuple[HTNMethod, ...]:
    """Construct versioned default methods that decompose the goal into capability tasks.

    Two ranked alternatives are produced: a parallel-first decomposition and a
    serialized economy decomposition. Both use capability-bound primitive leaves
    so the search engine retains provider choice. Goal capabilities complete on
    their goal.satisfied predicate; transitively discovered support capabilities
    complete on the contract they produce for downstream consumers.
    """
    goal_capabilities = {
        condition.predicate.arguments[0]
        for condition in goal.success_conditions
        if condition.predicate.arguments
    }
    subtasks: list[TaskSpec] = []
    for capability, operator_ids in sorted(discovery.capability_map.items()):
        if not operator_ids:
            continue
        if capability in goal_capabilities:
            completion = Predicate("goal.satisfied", (capability,))
        else:
            providers = discovery.operators_for_capability(capability)
            produced = next(
                (
                    effect.arguments[0]
                    for operator in providers
                    for effect in operator.effects
                    if effect.polarity and effect.name == "artifact.available"
                ),
                capability,
            )
            completion = Predicate("artifact.available", (produced,))
        subtasks.append(TaskSpec(
            task_id=f"task_{len(subtasks):02d}_{capability.replace('.', '_')}",
            task_type=capability,
            parameters={
                "capability": capability,
                "support": capability not in goal_capabilities,
            },
            completion=completion,
            compound=False,
        ))
    if not subtasks:
        return ()
    return (
        HTNMethod(
            method_id="goal.achieve.parallel",
            task_type=GOAL_TASK_TYPE,
            applicability_conditions=(),
            subtasks=tuple(subtasks),
            ordering="partial",
            constraints=("independent capability tasks may run concurrently",),
            version="1.0.0",
            priority=0.6,
        ),
        HTNMethod(
            method_id="goal.achieve.serial",
            task_type=GOAL_TASK_TYPE,
            applicability_conditions=(),
            subtasks=tuple(subtasks),
            ordering="ordered",
            constraints=("serialize capability tasks to minimize resource contention",),
            version="1.0.0",
            priority=0.4,
        ),
    )


__all__ = [
    "GOAL_TASK_TYPE",
    "DecompositionResult",
    "HTNDecomposer",
    "TaskTree",
    "build_default_methods",
]
