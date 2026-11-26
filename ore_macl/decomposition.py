"""Typed task decomposition and cycle-checked dependency DAGs (spec Section 4B).

Decomposition produces bounded objectives with explicit deliverables, acceptance
conditions, input artifacts, and dependencies (Step 8). The dependency graph is
checked for cycles before any dispatch (Step 9), and independent work is separated
from dependent work so only genuinely parallelizable tasks run concurrently
(Step 10).
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .contracts import (
    AgentTask,
    AgentTaskKind,
    SchemaSpec,
    SuccessCriteria,
    TaskTopology,
)
from .errors import CyclicDependencyError


@dataclass
class TaskDAG:
    """Hold typed tasks and their dependency edges with cycle detection (Step 9)."""

    goal_id: str
    tasks: dict[str, AgentTask] = field(default_factory=dict)

    def add(self, task: AgentTask) -> AgentTask:
        """Insert one task and verify its declared dependencies are internally consistent."""
        if task.goal_id != self.goal_id:
            raise ValueError(f"task {task.task_id} belongs to goal {task.goal_id!r}, not {self.goal_id!r}")
        if task.task_id in self.tasks:
            raise ValueError(f"duplicate task ID: {task.task_id}")
        for dependency in task.dependencies:
            if dependency == task.task_id:
                raise ValueError(f"task {task.task_id} depends on itself")
        self.tasks[task.task_id] = task
        return task

    def link(self, task_id: str, dependency_id: str) -> None:
        """Add one dependency edge after both endpoints exist."""
        for identifier in (task_id, dependency_id):
            if identifier not in self.tasks:
                raise KeyError(f"unknown task: {identifier}")
        if task_id == dependency_id:
            raise ValueError("a task cannot depend on itself")
        task = self.tasks[task_id]
        if dependency_id not in task.dependencies:
            object.__setattr__(
                self.tasks[task_id], "dependencies", tuple(task.dependencies) + (dependency_id,),
            )

    def detect_cycles(self) -> tuple[str, ...]:
        """Return nodes participating in at least one dependency cycle (Kahn's algorithm)."""
        incoming = {task_id: 0 for task_id in self.tasks}
        for task in self.tasks.values():
            for dependency in task.dependencies:
                if dependency in incoming:
                    incoming[task.task_id] += 1
        ready = deque(sorted(task_id for task_id, count in incoming.items() if count == 0))
        visited: set[str] = set()
        while ready:
            node = ready.popleft()
            visited.add(node)
            for task in self.tasks.values():
                if node in task.dependencies and task.task_id not in visited:
                    incoming[task.task_id] -= 1
                    if incoming[task.task_id] == 0:
                        ready.append(task.task_id)
        return tuple(sorted(set(self.tasks) - visited))

    def validate_acyclic(self) -> None:
        """Raise CyclicDependencyError when the DAG contains a cycle."""
        cycles = self.detect_cycles()
        if cycles:
            raise CyclicDependencyError(cycles)

    def ready_tasks(self, accepted_task_ids: Iterable[str]) -> tuple[AgentTask, ...]:
        """Return tasks whose dependencies are satisfied by accepted outputs (Step 49)."""
        accepted = frozenset(accepted_task_ids)
        ready = [
            task for task in self.tasks.values()
            if task.dependency_complete(accepted)
        ]
        return tuple(sorted(ready, key=lambda task: (-task.priority, task.task_id)))

    def topological_layers(self) -> tuple[tuple[str, ...], ...]:
        """Group task IDs into parallelizable layers; cycles raise before returning."""
        self.validate_acyclic()
        remaining = set(self.tasks)
        layers: list[tuple[str, ...]] = []
        while remaining:
            layer = sorted(
                task_id for task_id in remaining
                if all(dep not in remaining for dep in self.tasks[task_id].dependencies)
            )
            if not layer:
                raise CyclicDependencyError(tuple(sorted(remaining)))
            layers.append(tuple(layer))
            remaining.difference_update(layer)
        return tuple(layers)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible DAG description."""
        return {
            "goal_id": self.goal_id,
            "tasks": [task.to_dict() for task in sorted(self.tasks.values(), key=lambda t: t.task_id)],
            "layers": [list(layer) for layer in self.topological_layers()],
        }


class TaskDecomposer:
    """Decompose a validated goal into typed tasks with dependencies (Steps 8-11)."""

    LEXICAL_SCHEMA = SchemaSpec({
        "token_hypotheses": "dict",
        "coverage": "number",
    })
    SYNTACTIC_SCHEMA = SchemaSpec({
        "parse": "dict",
        "interpretation": "string",
        "ambiguity": "number",
    })
    SEMANTIC_SCHEMA = SchemaSpec({
        "meaning": "dict",
        "interpretation": "string",
        "confidence": "number",
    })
    VERIFICATION_SCHEMA = SchemaSpec({
        "verified": "boolean",
        "checks": "list",
        "findings": "list",
    })

    def __init__(self, *, workspace_revision: int = 0) -> None:
        """Bind the decomposer to the workspace revision tasks are derived from."""
        self.workspace_revision = workspace_revision

    def decompose_language_comprehension(self, goal_id: str) -> TaskDAG:
        """Produce the default typed task plan for an understand_sentence goal."""
        dag = TaskDAG(goal_id)
        common = {"goal_id": goal_id, "workspace_revision": self.workspace_revision}
        dag.add(AgentTask(
            task_id=f"{goal_id}:lexical", objective="Produce lexical hypotheses for every input token",
            required_capabilities=("macl.capability.lexical_analysis",),
            expected_result_schema=self.LEXICAL_SCHEMA,
            permitted_workspace_fields=("current_input", "lexical_hypotheses"),
            success_criteria=SuccessCriteria(
                required_output_keys=("token_hypotheses", "coverage"),
                must_cite_evidence=True,
            ),
            priority=0.8, **common,
        ))
        dag.add(AgentTask(
            task_id=f"{goal_id}:syntactic", objective="Propose a syntactic parse with an explicit interpretation",
            required_capabilities=("macl.capability.syntactic_analysis",),
            expected_result_schema=self.SYNTACTIC_SCHEMA,
            dependencies=(f"{goal_id}:lexical",),
            permitted_workspace_fields=("current_input", "lexical_hypotheses", "syntactic_hypotheses"),
            success_criteria=SuccessCriteria(required_output_keys=("parse", "interpretation")),
            priority=0.7, **common,
        ))
        dag.add(AgentTask(
            task_id=f"{goal_id}:semantic", objective="Integrate lexical and syntactic evidence into a semantic interpretation",
            required_capabilities=("macl.capability.semantic_interpretation",),
            expected_result_schema=self.SEMANTIC_SCHEMA,
            dependencies=(f"{goal_id}:lexical", f"{goal_id}:syntactic"),
            permitted_workspace_fields=(
                "current_input", "lexical_hypotheses", "syntactic_hypotheses",
                "semantic_hypotheses", "discourse_context",
            ),
            success_criteria=SuccessCriteria(
                required_output_keys=("meaning", "interpretation"),
                must_cite_evidence=True,
            ),
            priority=0.6, **common,
        ))
        dag.add(AgentTask(
            task_id=f"{goal_id}:verification", objective="Independently verify accepted findings and their evidence lineage",
            required_capabilities=("macl.capability.evidence_verification",),
            expected_result_schema=self.VERIFICATION_SCHEMA,
            dependencies=(f"{goal_id}:semantic",),
            permitted_workspace_fields=(
                "current_input", "lexical_hypotheses", "syntactic_hypotheses",
                "semantic_hypotheses", "artifact_references",
            ),
            requires_independent_verification=False,
            verifier_requirements=("deterministic_checks", "lineage_checks"),
            priority=0.9, **common,
        ))
        dag.validate_acyclic()
        return dag

    def decompose_debate(self, goal_id: str, *, competing_capability: str, schema: SchemaSpec,
                         objective: str) -> TaskDAG:
        """Produce a debate-topology plan for competing hypotheses (Steps 16, 37-43)."""
        dag = TaskDAG(goal_id)
        common = {"goal_id": goal_id, "workspace_revision": self.workspace_revision}
        dag.add(AgentTask(
            task_id=f"{goal_id}:debate", objective=objective,
            required_capabilities=(competing_capability,),
            expected_result_schema=schema,
            topology=TaskTopology.DEBATE,
            topology_justification="Competing hypotheses require independent proposals and critic review",
            requires_independent_verification=True,
            permitted_workspace_fields=("current_input", "lexical_hypotheses", "semantic_hypotheses"),
            priority=0.85, **common,
        ))
        dag.validate_acyclic()
        return dag

    def decompose_module_task(self, goal_id: str, *, objective: str,
                              required_capabilities: tuple[str, ...],
                              module_route: Mapping[str, Any]) -> TaskDAG:
        """Produce a deterministic-module task routed through the existing coordinator."""
        dag = TaskDAG(goal_id)
        dag.add(AgentTask(
            task_id=f"{goal_id}:module", objective=objective,
            required_capabilities=required_capabilities,
            expected_result_schema=SchemaSpec({"module_status": "string", "artifact_count": "integer"}),
            kind=AgentTaskKind.DETERMINISTIC_MODULE,
            module_route=dict(module_route),
            goal_id=goal_id,
            workspace_revision=self.workspace_revision,
            priority=0.75,
        ))
        dag.validate_acyclic()
        return dag


def identify_required_roles(dag: TaskDAG) -> dict[str, tuple[str, ...]]:
    """Map each task to the capability roles it requires (Step 11)."""
    roles: dict[str, tuple[str, ...]] = {}
    for task in sorted(dag.tasks.values(), key=lambda item: item.task_id):
        roles[task.task_id] = tuple(sorted(task.required_capabilities))
    return roles


def separate_independent_work(dag: TaskDAG) -> dict[str, tuple[str, ...]]:
    """Group tasks into layers separating independent and dependent work (Step 10)."""
    return {
        f"layer_{index}": layer
        for index, layer in enumerate(dag.topological_layers(), start=1)
    }
