"""Plan execution adapter that submits planned work through existing ORE components.

The adapter converts a selected plan into a standard ore_contracts CognitiveTask
and submits it to the existing CognitiveTaskCoordinator. Task execution, worker
scheduling, artifact validation, and workspace publication remain entirely with
the coordinator; this adapter never invokes module backends directly. Completed
module results are captured as checkpoints so resuming a plan never duplicates
executions or publications.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ore_contracts import ModuleRegistry, RunContext
from ore_contracts.coordinator import CognitiveTask, CognitiveTaskCoordinator, CognitiveTaskResult
from ore_contracts.module_protocol import ModuleResult, ModuleStatus

from .contracts import (
    CostVector,
    ExecutionFeedback,
    GoalSpec,
    PlanCheckpoint,
)
from .cost import CostModel
from .partial_order import PartialOrderPlan
from .search import SearchOutcome


@dataclass(frozen=True, slots=True)
class PlanExecutionReport:
    """Summarize one submission round through the existing coordinator."""

    plan_id: str
    task_id: str
    coordinator_status: str
    feedback: tuple[ExecutionFeedback, ...]
    workspace_id: str
    workspace_revision: int
    module_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    module_results: Mapping[str, ModuleResult] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize report collections."""
        object.__setattr__(self, "feedback", tuple(self.feedback))
        object.__setattr__(self, "module_ids", tuple(self.module_ids))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        object.__setattr__(self, "module_results", dict(self.module_results))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible report record."""
        return {
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "coordinator_status": self.coordinator_status,
            "feedback": [item.to_dict() for item in self.feedback],
            "workspace_id": self.workspace_id,
            "workspace_revision": self.workspace_revision,
            "module_ids": list(self.module_ids),
            "evidence_refs": list(self.evidence_refs),
            "module_result_statuses": {
                key: value.status.value for key, value in self.module_results.items()
            },
        }


class PlanExecutorAdapter:
    """Submit planned task graphs to the existing coordinator and collect feedback."""

    def __init__(
        self,
        coordinator: CognitiveTaskCoordinator,
        registry: ModuleRegistry,
        cost_model: CostModel | None = None,
    ) -> None:
        """Bind the adapter to the shared coordinator, registry, and cost model."""
        if not isinstance(coordinator, CognitiveTaskCoordinator):
            raise TypeError("coordinator must be a CognitiveTaskCoordinator")
        if not isinstance(registry, ModuleRegistry):
            raise TypeError("registry must be a ModuleRegistry")
        self.coordinator = coordinator
        self.registry = registry
        self.cost_model = cost_model or CostModel()

    def _capability_preferences(
        self, plan: PartialOrderPlan, outcome: SearchOutcome
    ) -> dict[str, str]:
        """Map each capability in the plan to the operator the search selected."""
        chosen = dict(outcome.operator_sequence)
        preferences: dict[str, str] = {}
        for node in plan.nodes:
            operator_id = chosen.get(node.node_id)
            if operator_id is None:
                continue
            preferences[node.task.task_type] = operator_id
        return preferences

    def _predicted_costs(
        self, plan: PartialOrderPlan, outcome: SearchOutcome
    ) -> dict[str, CostVector]:
        """Return the predicted cost vector for every chosen operator."""
        predicted: dict[str, CostVector] = {}
        chosen = dict(outcome.operator_sequence)
        for node in plan.nodes:
            operator_id = chosen.get(node.node_id)
            if operator_id is None:
                continue
            predicted[operator_id] = node.cost_estimate
        return predicted

    def submit(
        self,
        plan: PartialOrderPlan,
        goal: GoalSpec,
        outcome: SearchOutcome,
        *,
        input_artifacts: Mapping[str, Any] | None = None,
        run_context: RunContext | None = None,
        checkpoint: PlanCheckpoint | None = None,
    ) -> PlanExecutionReport:
        """Build one CognitiveTask from the plan and execute it via the coordinator."""
        if not outcome.operator_sequence and checkpoint is None:
            raise ValueError("refusing to submit an empty plan without a checkpoint")
        preferences = self._capability_preferences(plan, outcome)
        predicted = self._predicted_costs(plan, outcome)
        required_capabilities = tuple(sorted(preferences))
        completed: dict[str, ModuleResult] = {}
        if checkpoint is not None:
            if checkpoint.plan_id != plan.plan_id:
                raise ValueError("checkpoint belongs to a different plan")
            completed = {
                key: value for key, value in checkpoint.completed_module_results.items()
                if isinstance(value, ModuleResult)
            }
        metadata: dict[str, Any] = {
            "ore_planner_plan_id": plan.plan_id,
            "ore_planner_goal_id": goal.goal_id,
            "ore_planner_workspace_revision": plan.workspace_revision,
            "resource_module_preferences": dict(preferences),
            "_completed_module_results": dict(completed),
        }
        task_inputs = self._carry_forward_artifacts(
            dict(input_artifacts or {}), completed,
        )
        task = CognitiveTask(
            goal=goal.description,
            required_capabilities=required_capabilities,
            input_artifacts=task_inputs,
            execution_mode="FUNCTIONAL",
            priority=goal.priority,
            metadata=metadata,
        )
        result: CognitiveTaskResult = self.coordinator.coordinate(task, run_context)
        feedback = self._to_feedback(plan, result, predicted, completed)
        return PlanExecutionReport(
            plan_id=plan.plan_id,
            task_id=result.task_id,
            coordinator_status=result.status,
            feedback=tuple(feedback),
            workspace_id=result.workspace_id,
            workspace_revision=result.workspace_revision,
            module_ids=tuple(sorted(result.module_results)),
            evidence_refs=tuple(
                record.artifact_id for record in result.artifacts
            ),
            module_results=dict(result.module_results),
        )

    def _carry_forward_artifacts(
        self,
        base_inputs: dict[str, Any],
        completed: Mapping[str, ModuleResult],
    ) -> dict[str, Any]:
        """Expose prior validated artifacts under compatible consumer input port IDs.

        This mirrors the existing AdaptiveResourceCoordinator carry-forward
        convention so the coordinator's artifact ledger recognizes the lineage of
        completed modules' outputs on resume.
        """
        inputs = dict(base_inputs)
        carried: dict[str, list[Any]] = {}
        for module_result in completed.values():
            for artifact in module_result.artifacts:
                for manifest in self.registry.list_manifests():
                    for port in manifest.input_ports:
                        if (
                            artifact.envelope.contract_id == port.contract_id
                            and artifact.envelope.schema_version in port.accepted_versions
                        ):
                            bucket = carried.setdefault(port.port_id, [])
                            if artifact not in bucket:
                                bucket.append(artifact)
        for port_id, artifacts in carried.items():
            existing = inputs.get(port_id)
            existing_values = (
                tuple(existing) if isinstance(existing, (tuple, list))
                else (() if existing is None else (existing,))
            )
            merged = {item.envelope.fingerprint(): item for item in (*existing_values, *artifacts)}
            inputs[port_id] = tuple(merged[key] for key in sorted(merged))
        return inputs

    def _to_feedback(
        self,
        plan: PartialOrderPlan,
        result: CognitiveTaskResult,
        predicted: Mapping[str, CostVector],
        completed: Mapping[str, ModuleResult],
    ) -> list[ExecutionFeedback]:
        """Convert coordinator module results into prediction-aware feedback records."""
        del plan
        feedback: list[ExecutionFeedback] = []
        for module_id, module_result in sorted(result.module_results.items()):
            if module_id in completed:
                continue
            success = module_result.status is ModuleStatus.SUCCESS
            duration = module_result.processing_metadata.get("duration_ms")
            memory = module_result.processing_metadata.get("memory_mb")
            actual = CostVector(
                time_ms=float(duration) if duration is not None else 1.0,
                memory_mb=float(memory) if memory is not None else 0.0,
                uncertainty=0.0 if success else 0.5,
            )
            failure_class = None if success else module_result.status.value
            outputs = tuple(
                artifact.envelope.fingerprint() for artifact in module_result.artifacts
            )
            feedback.append(ExecutionFeedback(
                task_id=result.task_id,
                node_id=None,
                operator_id=module_id,
                actual_cost=actual,
                predicted_cost=predicted.get(module_id, CostVector()),
                outputs=outputs,
                success=success,
                failure_class=failure_class,
                evidence_refs=outputs,
                module_id=module_id,
            ))
        return feedback

    @staticmethod
    def checkpoint(
        plan: PartialOrderPlan,
        goal: GoalSpec,
        report: PlanExecutionReport,
        result: CognitiveTaskResult | None = None,
    ) -> PlanCheckpoint:
        """Capture completed module results so resume never re-executes them."""
        source = result.module_results if result is not None else report.module_results
        completed = {
            key: value for key, value in source.items()
            if value.status is ModuleStatus.SUCCESS
        }
        completed_ids = tuple(sorted(completed)) or report.module_ids
        return PlanCheckpoint(
            plan_id=plan.plan_id,
            goal_id=goal.goal_id,
            workspace_revision=report.workspace_revision,
            completed_node_ids=completed_ids,
            completed_module_results=dict(completed),
            feedback=report.feedback,
        )


def completed_operators(reports: Iterable[PlanExecutionReport]) -> tuple[str, ...]:
    """Return every operator that already succeeded across submission rounds."""
    succeeded: set[str] = set()
    for report in reports:
        succeeded.update(
            item.operator_id for item in report.feedback
            if item.success and item.operator_id is not None
        )
    return tuple(sorted(succeeded))


__all__ = [
    "PlanExecutionReport",
    "PlanExecutorAdapter",
    "completed_operators",
]
