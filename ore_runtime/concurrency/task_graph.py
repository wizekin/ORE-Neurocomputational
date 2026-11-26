"""Task nodes, dependency edges, cycle detection, readiness, and iterative regions."""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..contracts import TaskState, TaskSpec
from ..exceptions import CycleRejectedError, PlanValidationError


@dataclass(slots=True)
class TaskNode:
    """Track the mutable runtime state of one task inside the execution graph."""

    spec: TaskSpec
    state: TaskState = TaskState.CREATED
    attempts: int = 0
    enqueued_at: float | None = None
    assigned_worker: str | None = None
    read_version: int | None = None
    commit_version: int | None = None
    last_error: str | None = None
    output: Mapping[str, Any] = field(default_factory=dict)
    activated_children: tuple[str, ...] = ()
    deadline_exceeded: bool = False

    @property
    def task_id(self) -> str:
        """Return the stable identifier of this node's task."""
        return self.spec.task_id

    @property
    def unfinished(self) -> bool:
        """Return whether the task still owes work before the goal can complete."""
        return self.state not in {
            TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED,
        }


@dataclass(frozen=True, slots=True)
class IterativeRegionSpec:
    """Declare a bounded iterative subplan replacing an intentional dependency cycle."""

    region_id: str
    member_task_ids: tuple[str, ...]
    update: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None
    distance: Callable[[Mapping[str, Any], Mapping[str, Any]], float] | None = None
    validation_predicate: Callable[[Mapping[str, Any]], bool] | None = None
    max_iterations: int = 64
    epsilon: float = 1e-6
    initial_state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate region identity, membership, and convergence bounds."""
        if not self.region_id.strip():
            raise ValueError("region_id must be non-empty.")
        if not self.member_task_ids:
            raise ValueError("member_task_ids must be non-empty.")
        if len(set(self.member_task_ids)) != len(self.member_task_ids):
            raise ValueError("member_task_ids must not contain duplicates.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        object.__setattr__(self, "initial_state", dict(self.initial_state))
        if self.update is None:
            object.__setattr__(
                self, "update", lambda state: dict(state),
            )


@dataclass(frozen=True, slots=True)
class IterationResult:
    """Report the bounded convergence outcome of one iterative region."""

    region_id: str
    converged: bool
    iterations: int
    final_state: Mapping[str, Any]
    distances: tuple[float, ...]
    stopped_because: str

    def __post_init__(self) -> None:
        """Detach the final state mapping from caller mutation."""
        object.__setattr__(self, "final_state", dict(self.final_state))


def _default_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Measure mean absolute numeric difference between two iteration states."""
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    total = 0.0
    for key in keys:
        left_value = left.get(key)
        right_value = right.get(key)
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            total += abs(float(left_value) - float(right_value))
        else:
            total += 0.0 if left_value == right_value else 1.0
    return total / len(keys)


class TaskGraph:
    """Maintain the DAG of task nodes with readiness, cycles, and iterative regions."""

    def __init__(self, specs: Sequence[TaskSpec] = ()) -> None:
        """Build the graph from a task manifest with known dependencies.

        All nodes are created before edges are wired so manifests may declare
        tasks in any order; unknown dependencies and self-dependencies are
        rejected during edge wiring.
        """
        self._nodes: dict[str, TaskNode] = {}
        self._dependents: dict[str, tuple[str, ...]] = {}
        self._iterative_regions: dict[str, IterativeRegionSpec] = {}
        for spec in specs:
            self._add_node(spec)
        self._wire_edges()

    def _add_node(self, spec: TaskSpec) -> TaskNode:
        """Insert one task node without registering its edges yet."""
        if spec.task_id in self._nodes:
            raise ValueError(f"task '{spec.task_id}' is already present in the graph.")
        node = TaskNode(spec=spec)
        self._nodes[spec.task_id] = node
        self._dependents.setdefault(spec.task_id, ())
        return node

    def _wire_one(self, spec: TaskSpec) -> None:
        """Validate and register one task's dependency edges."""
        for dep in spec.dependencies:
            if dep == spec.task_id:
                raise CycleRejectedError(
                    "a task cannot depend on itself",
                    diagnostic_path=(spec.task_id,),
                )
            if dep not in self._nodes:
                raise ValueError(
                    f"task '{spec.task_id}' declares unknown dependency '{dep}'"
                )
            known = list(self._dependents.setdefault(dep, ()))
            self._dependents[dep] = tuple(known + [spec.task_id])

    def _wire_edges(self) -> None:
        """Wire every declared edge after all nodes exist."""
        for node in self._nodes.values():
            self._wire_one(node.spec)

    def add_task(self, spec: TaskSpec) -> TaskNode:
        """Insert one task node and register its dependency edges immediately."""
        node = self._add_node(spec)
        self._wire_one(spec)
        return node

    # ------------------------------------------------------------------ queries

    def node(self, task_id: str) -> TaskNode:
        """Return the runtime node for a known task identifier."""
        if task_id not in self._nodes:
            raise KeyError(f"unknown task '{task_id}'")
        return self._nodes[task_id]

    @property
    def nodes(self) -> tuple[TaskNode, ...]:
        """Return every node in deterministic insertion order."""
        return tuple(self._nodes.values())

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Return every task identifier in insertion order."""
        return tuple(self._nodes.keys())

    def dependents_of(self, task_id: str) -> tuple[str, ...]:
        """Return the tasks that require the given task's committed output."""
        return self._dependents.get(task_id, ())

    def dependencies_of(self, task_id: str) -> tuple[str, ...]:
        """Return the tasks whose committed outputs the given task requires."""
        return self._nodes[task_id].spec.dependencies

    def unfinished_tasks(self) -> tuple[TaskNode, ...]:
        """Return every node that still owes work before the goal completes."""
        return tuple(node for node in self._nodes.values() if node.unfinished)

    def is_complete(self) -> bool:
        """Return whether every task has reached a terminal state."""
        return all(not node.unfinished for node in self._nodes.values())

    def ready_tasks(self) -> tuple[TaskNode, ...]:
        """Return tasks whose predecessors all succeeded and whose state allows dispatch.

        Readiness follows the specification's predicate generalized to tasks
        already sitting in the ready queue: state CREATED/WAITING tasks become
        eligible once every dependency SUCCEEDED, and READY tasks remain
        dispatchable until a worker claims them.
        """
        ready: list[TaskNode] = []
        for node in self._nodes.values():
            if node.state not in {TaskState.CREATED, TaskState.WAITING, TaskState.READY}:
                continue
            if all(
                self._nodes[dep].state is TaskState.SUCCEEDED
                for dep in node.spec.dependencies
            ):
                ready.append(node)
        return tuple(ready)

    def mark_ready(self) -> tuple[str, ...]:
        """Promote eligible CREATED/WAITING tasks to READY and return their IDs."""
        promoted: list[str] = []
        for node in self.ready_tasks():
            node.state = TaskState.READY
            promoted.append(node.task_id)
        return tuple(promoted)

    def terminal_states(self) -> dict[str, str]:
        """Return the terminal state of every task for invariant checks and reports."""
        return {
            task_id: node.state.value
            for task_id, node in self._nodes.items()
        }

    def critical_path_length(self) -> float:
        """Return the maximum summed estimated cost along any dependency path."""
        if not self._nodes:
            return 0.0
        longest: dict[str, float] = {}

        def longest_for(task_id: str, visiting: frozenset[str]) -> float:
            """Compute the longest chain ending at one task with cycle protection."""
            if task_id in longest:
                return longest[task_id]
            if task_id in visiting:
                raise CycleRejectedError(
                    "cycle detected while computing the critical path",
                    diagnostic_path=(task_id,),
                )
            deps = self._nodes[task_id].spec.dependencies
            inbound = max(
                (longest_for(dep, visiting | {task_id}) for dep in deps),
                default=0.0,
            )
            longest[task_id] = inbound + self._nodes[task_id].spec.estimated_cost
            return longest[task_id]

        return max(longest_for(task_id, frozenset()) for task_id in self._nodes)

    # ------------------------------------------------------------------ cycles

    def detect_cycles(self) -> tuple[tuple[str, ...], ...]:
        """Return every dependency cycle as an ordered diagnostic path."""
        color: dict[str, int] = {task_id: 0 for task_id in self._nodes}
        stack: list[str] = []
        cycles: list[tuple[str, ...]] = []

        def visit(task_id: str) -> None:
            """Depth-first search marking one node and recording completed cycles."""
            color[task_id] = 1
            stack.append(task_id)
            for dependent in self._dependents.get(task_id, ()):
                if color[dependent] == 0:
                    visit(dependent)
                elif color[dependent] == 1:
                    start = stack.index(dependent)
                    cycles.append(tuple(stack[start:]))
            stack.pop()
            color[task_id] = 2

        for task_id in self._nodes:
            if color[task_id] == 0:
                visit(task_id)
        return tuple(cycles)

    def topological_order(self) -> tuple[str, ...]:
        """Return a Kahn topological order over current dependency edges."""
        indegree = {
            task_id: len(self._nodes[task_id].spec.dependencies)
            for task_id in self._nodes
        }
        frontier = sorted(
            task_id for task_id, degree in indegree.items() if degree == 0
        )
        order: list[str] = []
        while frontier:
            task_id = frontier.pop(0)
            order.append(task_id)
            for dependent in sorted(self._dependents.get(task_id, ())):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    frontier.append(dependent)
            frontier.sort()
        if len(order) != len(self._nodes):
            remaining = sorted(set(self._nodes) - set(order))
            raise CycleRejectedError(
                "the task graph contains an unresolved dependency cycle",
                diagnostic_path=tuple(remaining),
            )
        return tuple(order)

    def validate_plan(self) -> tuple[str, ...]:
        """Reject accidental cycles with a diagnostic path and return the valid order."""
        cycles = self.detect_cycles()
        if cycles:
            cycle = cycles[0]
            raise CycleRejectedError(
                "accidental dependency cycle rejected: " + " -> ".join(cycle + (cycle[0],)),
                diagnostic_path=cycle,
            )
        return self.topological_order()

    # -------------------------------------------------------- iterative regions

    def declare_iterative_region(self, spec: IterativeRegionSpec) -> None:
        """Register a bounded iterative region that replaces an intentional cycle."""
        unknown = [task_id for task_id in spec.member_task_ids
                   if task_id not in self._nodes]
        if unknown:
            raise PlanValidationError(
                "iterative region references unknown tasks",
                diagnostic_path=tuple(unknown),
            )
        if spec.region_id in self._iterative_regions:
            raise ValueError(f"iterative region '{spec.region_id}' already exists.")
        self._iterative_regions[spec.region_id] = spec

    @property
    def iterative_regions(self) -> tuple[IterativeRegionSpec, ...]:
        """Return every declared iterative region."""
        return tuple(self._iterative_regions.values())

    def run_iterative_region(
        self, spec: IterativeRegionSpec,
        distance: Callable[[Mapping[str, Any], Mapping[str, Any]], float] | None = None,
    ) -> IterationResult:
        """Iterate x_(k+1)=F(x_k) until convergence, iteration limit, or predicate failure."""
        measure = distance or spec.distance or _default_distance
        state = dict(spec.initial_state)
        history: list[float] = []
        for iteration in range(1, spec.max_iterations + 1):
            if spec.validation_predicate is not None and not spec.validation_predicate(state):
                return IterationResult(
                    region_id=spec.region_id, converged=False, iterations=iteration - 1,
                    final_state=state, distances=tuple(history),
                    stopped_because="validation_predicate_failed",
                )
            next_state = dict(spec.update(state))
            step = measure(state, next_state)
            history.append(step)
            state = next_state
            if not math.isfinite(step):
                return IterationResult(
                    region_id=spec.region_id, converged=False, iterations=iteration,
                    final_state=state, distances=tuple(history),
                    stopped_because="divergence_detected",
                )
            if step < spec.epsilon:
                return IterationResult(
                    region_id=spec.region_id, converged=True, iterations=iteration,
                    final_state=state, distances=tuple(history),
                    stopped_because="converged_below_epsilon",
                )
        return IterationResult(
            region_id=spec.region_id, converged=False, iterations=spec.max_iterations,
            final_state=state, distances=tuple(history),
            stopped_because="iteration_limit_reached",
        )

    # ------------------------------------------------------------------ reports

    def summary(self) -> dict[str, Any]:
        """Return a compact graph report for engine traces and metrics."""
        return {
            "task_count": len(self._nodes),
            "states": self.terminal_states(),
            "ready": [node.task_id for node in self.ready_tasks()],
            "critical_path_length": self.critical_path_length(),
            "iterative_regions": list(self._iterative_regions),
        }
