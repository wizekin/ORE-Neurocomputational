"""Workflow graph schema: nodes, edges, branches, joins and terminal states.

A workflow is a DAG W = (V, E) where each node is a typed task and edge (i, j)
means task j depends on task i.  The graph validates structure (Steps 5, 8),
orders execution topologically, computes the critical path, and exposes the
independent frontier for map-reduce dispatch (Step 13).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..contracts import JoinSemantics, TaskSpec, content_digest

__all__ = (
    "CycleError",
    "GraphValidationError",
    "TaskGraph",
    "topological_order",
)


class GraphValidationError(ValueError):
    """Raised when a task graph violates the workflow schema (Step 8)."""


class CycleError(GraphValidationError):
    """Raised when the graph contains an unbounded dependency cycle (Step 5)."""


def topological_order(tasks: Mapping[str, TaskSpec]) -> list[str]:
    """Return a valid execution order, or raise CycleError (Step 5)."""
    order: list[str] = []
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> None:
        if state.get(node) == 2:
            return
        if state.get(node) == 1:
            cycle = stack[stack.index(node):] + [node]
            raise CycleError("unbounded dependency cycle: " + " -> ".join(cycle))
        state[node] = 1
        stack.append(node)
        for dependency in tasks[node].depends_on:
            visit(dependency)
        stack.pop()
        state[node] = 2
        order.append(node)

    for node in tasks:
        visit(node)
    return order


@dataclass(slots=True)
class TaskGraph:
    """A validated directed task graph with joins and conditional edges (Step 5)."""

    tasks: dict[str, TaskSpec] = field(default_factory=dict)

    def add(self, task: TaskSpec, *, replace: bool = False) -> TaskSpec:
        """Add one typed task; duplicate IDs require replace."""
        if task.task_id in self.tasks and not replace:
            raise GraphValidationError(f"duplicate task id: {task.task_id}")
        self.tasks[task.task_id] = task
        return task

    def validate(self) -> list[str]:
        """Validate structure, dependencies, schemas and terminal conditions (Step 8).

        Returns the ordered list of topological task IDs; raises on any
        violation.  Schemas must be mappings; every dependency must exist; the
        graph must be acyclic; join tasks with quorum semantics need a quorum
        fraction; and at least one terminal task must exist.
        """
        if not self.tasks:
            raise GraphValidationError("a workflow graph requires at least one task")
        for task in self.tasks.values():
            unknown = [dep for dep in task.depends_on if dep not in self.tasks]
            if unknown:
                raise GraphValidationError(
                    f"task {task.task_id} depends on unknown tasks: {unknown}")
            if task.join.semantics in (JoinSemantics.QUORUM, JoinSemantics.WEIGHTED_QUORUM):
                if not 0.0 < task.join.quorum <= 1.0:
                    raise GraphValidationError(
                        f"join task {task.task_id} needs a quorum fraction in (0, 1]")
            if not isinstance(task.input_schema, Mapping) or not isinstance(
                    task.output_schema, Mapping):
                raise GraphValidationError(
                    f"task {task.task_id} schemas must be mappings")
            if not task.capability or not task.capability.strip():
                raise GraphValidationError(
                    f"task {task.task_id} must declare a capability")
        order = topological_order(self.tasks)
        terminals = [task_id for task_id, task in self.tasks.items()
                     if not self.dependents(task_id)]
        if not terminals:
            raise GraphValidationError("the workflow graph has no terminal task")
        return order

    def dependents(self, task_id: str) -> tuple[str, ...]:
        """Return the tasks that directly depend on the given task."""
        return tuple(sorted(tid for tid, task in self.tasks.items()
                            if task_id in task.depends_on))

    def dependencies(self, task_id: str) -> tuple[str, ...]:
        """Return the direct dependencies of a task."""
        return tuple(self.tasks[task_id].depends_on)

    def frontier(self, completed: Iterable[str]) -> tuple[str, ...]:
        """Return tasks whose dependencies are all complete (Step 13)."""
        done = set(completed)
        ready = []
        for task_id, task in self.tasks.items():
            if task_id in done:
                continue
            if all(dep in done for dep in task.depends_on):
                ready.append(task_id)
        return tuple(sorted(ready))

    def critical_path_seconds(self) -> float:
        """Return T_crit = max over paths p of sum(t_i for i in p) (math foundations)."""
        longest: dict[str, float] = {}
        for task_id in topological_order(self.tasks):
            task = self.tasks[task_id]
            best_predecessor = max((longest.get(dep, 0.0)
                                    for dep in task.depends_on), default=0.0)
            longest[task_id] = best_predecessor + task.cost.seconds
        return max(longest.values(), default=0.0)

    def aggregate_cost(self) -> dict[str, float]:
        """Return the aggregate cost vector over all tasks (Step 7)."""
        totals = {"cpu_units": 0.0, "memory_mb": 0.0, "io_units": 0.0,
                  "network_units": 0.0, "seconds": 0.0}
        for task in self.tasks.values():
            for key in totals:
                totals[key] += float(getattr(task.cost, key))
        return totals

    def plan_hash(self) -> str:
        """Return the immutable plan content digest (Step 8)."""
        return content_digest({tid: task.to_dict() for tid, task in
                               sorted(self.tasks.items())})

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph."""
        return {"tasks": {tid: task.to_dict() for tid, task in
                          sorted(self.tasks.items())},
                "plan_hash": self.plan_hash()}
