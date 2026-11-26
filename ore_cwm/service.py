"""Cognitive World Model service: narrow prediction API, MPC loop, and advisory bridges.

CWM is a predictive advisor, not a controller: the shared workspace stays authoritative
for state, the CognitiveTaskCoordinator stays responsible for dispatch, the
AdaptiveResourceCoordinator stays responsible for resource safeguards, modules execute,
validators and workspace integration accept artifacts, and USG remains the knowledge
source. CWM predicts and advises; it never dispatches around the coordinator.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from ore_contracts.coordinator import CognitiveTask
from ore_workspace.goals import GoalEvaluator
from ore_workspace.models import WorkspaceSnapshot
from ore_workspace.permissions import PermissionPolicy

from .baselines import DEFAULT_OPERATION_CLASS_BASELINES
from .contracts import (
    MISSING,
    AdvisoryMode,
    ArtifactLifecycle,
    CognitiveOperation,
    CognitiveStateFeatures,
    CWMBudget,
    CWMConfig,
    ExecutionStatus,
    ObservedOutcome,
    OperationContext,
    OperationPrecondition,
    OperationType,
    OutcomeDimensions,
    PredictionRecord,
    ResourceState,
    _utc_now,
)
from .features import FeatureExtractor
from .learning import (
    ExecutionLedger,
    PredictionErrorReport,
)
from .observability import SQLiteCWMObserver, SQLitePredictionStore
from .planner import CWMPlan, MPCObjectiveWeights, MPCPlanner
from .policy import ConfidenceGate, FallbackPolicy, HardConstraintSet, build_gate
from .predictor import StatisticalTransitionModel, classify_operation
from .registry import OPERATION_REGISTRY, registry_as_dicts

ModuleRegistry = Any


class CoordinatorLike(Protocol):
    """The minimal coordinator surface CWM requires to dispatch one first action."""

    def coordinate(self, task: CognitiveTask, context: Any | None = None) -> Any: ...


CandidatesProvider = Callable[[WorkspaceSnapshot], Sequence[CognitiveOperation]]


@dataclass(frozen=True, slots=True)
class CWMStepRecord:
    """Summarize one receding-horizon iteration: prediction, dispatch, and observation."""

    iteration: int
    workspace_revision_before: int
    workspace_revision_after: int
    selected_operation_id: str | None
    selected_module_id: str | None
    predicted_success: float | None
    predicted_duration_ms: float | None
    observed_artifact_lifecycle: str | None
    prediction_error: PredictionErrorReport | None
    fallback_trigger: str | None
    advisory_task_modified: bool
    timestamp: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible step record."""
        return {
            "iteration": self.iteration,
            "workspace_revision_before": self.workspace_revision_before,
            "workspace_revision_after": self.workspace_revision_after,
            "selected_operation_id": self.selected_operation_id,
            "selected_module_id": self.selected_module_id,
            "predicted_success": self.predicted_success,
            "predicted_duration_ms": self.predicted_duration_ms,
            "observed_artifact_lifecycle": self.observed_artifact_lifecycle,
            "prediction_error": (
                None if self.prediction_error is None else self.prediction_error.to_dict()),
            "fallback_trigger": self.fallback_trigger,
            "advisory_task_modified": self.advisory_task_modified,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True, slots=True)
class CWMRunResult:
    """Report one receding-horizon run with its steps, outcomes, and learning summaries."""

    workspace_id: str
    goal: str
    advisory_mode: str
    steps: tuple[CWMStepRecord, ...] = ()
    outcomes: tuple[ObservedOutcome, ...] = ()
    initial_goal_score: float = 0.0
    final_goal_score: float = 0.0
    stop_reason: str = ""
    operation_trace: tuple[str, ...] = ()
    calibration: dict[str, Any] = field(default_factory=dict)
    drift: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)

    @property
    def goal_progress(self) -> float:
        """Return the observed goal-score change achieved by the run."""
        return max(0.0, self.final_goal_score - self.initial_goal_score)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible run report."""
        return {
            "workspace_id": self.workspace_id,
            "goal": self.goal,
            "advisory_mode": self.advisory_mode,
            "steps": [item.to_dict() for item in self.steps],
            "outcomes": [item.to_dict() for item in self.outcomes],
            "initial_goal_score": self.initial_goal_score,
            "final_goal_score": self.final_goal_score,
            "goal_progress": self.goal_progress,
            "stop_reason": self.stop_reason,
            "operation_trace": list(self.operation_trace),
            "calibration": dict(self.calibration),
            "drift": dict(self.drift),
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class ShadowReport:
    """Compare shadow-mode forecasts with actual outcomes without altering execution."""

    workspace_id: str
    goal: str
    steps: tuple[CWMStepRecord, ...] = ()
    comparisons: tuple[PredictionErrorReport, ...] = ()
    calibration: dict[str, Any] = field(default_factory=dict)
    drift: dict[str, Any] = field(default_factory=dict)
    tasks_modified: bool = False
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible shadow report."""
        return {
            "workspace_id": self.workspace_id,
            "goal": self.goal,
            "steps": [item.to_dict() for item in self.steps],
            "comparisons": [item.to_dict() for item in self.comparisons],
            "calibration": dict(self.calibration),
            "drift": dict(self.drift),
            "tasks_modified": self.tasks_modified,
            "created_at": self.created_at,
        }


class CognitiveWorldModel:
    """Predictive advisor that ranks candidate operations; coordinators stay authoritative."""

    def __init__(
        self,
        *,
        config: CWMConfig | None = None,
        goal_evaluator: GoalEvaluator | None = None,
        objective: MPCObjectiveWeights | None = None,
        hard_constraints: HardConstraintSet | None = None,
        ledger: ExecutionLedger | None = None,
        transition_model: StatisticalTransitionModel | None = None,
        planner: MPCPlanner | None = None,
        observer: SQLiteCWMObserver | None = None,
        prediction_store: SQLitePredictionStore | None = None,
        repository_root: str | Path | None = None,
        logger_database_path: str | Path | None = None,
        enable_logging: bool = False,
    ) -> None:
        """Assemble the advisory subsystem around one shared execution ledger."""
        self.config = config or CWMConfig()
        self.goal_evaluator = goal_evaluator or GoalEvaluator()
        self.features = FeatureExtractor(self.goal_evaluator)
        self.ledger = ledger or ExecutionLedger(self.config)
        self.transition_model = transition_model or StatisticalTransitionModel(
            self.ledger, self.config)
        self.hard_constraints = hard_constraints or HardConstraintSet()
        self.confidence_gate: ConfidenceGate = build_gate(self.config)
        self.fallback_policy = FallbackPolicy()
        self.planner = planner or MPCPlanner(
            self.transition_model, self.hard_constraints, objective,
            self.confidence_gate, self.fallback_policy, self.config)
        self.observer = observer
        if enable_logging and observer is None:
            self.observer = SQLiteCWMObserver(logger_database_path, repository_root).open()
        self.prediction_store = prediction_store

    # -- Step 43: the narrow prediction API ---------------------------------

    def predict_outcomes(
        self,
        snapshot: WorkspaceSnapshot,
        candidates: Sequence[CognitiveOperation],
        *,
        horizon: int | None = None,
        budget: CWMBudget | None = None,
        advisory_mode: str = AdvisoryMode.SHADOW.value,
        resource_state: ResourceState | None = None,
        unresolved_conflicts: int | None = None,
        belief_probabilities: Sequence[float] | None = None,
    ) -> tuple[PredictionRecord, ...]:
        """Predict each candidate's outcome tied to the exact snapshot revision."""
        _require_snapshot(snapshot)
        features = self._extract_features(
            snapshot, resource_state=resource_state,
            unresolved_conflicts=unresolved_conflicts)
        budget = budget or CWMBudget(
            horizon=self.config.default_horizon if horizon is None else horizon)
        records: list[PredictionRecord] = []
        for operation in candidates:
            context = self._context_for(features, operation)
            transition = self.transition_model.transition(
                features, operation, context, belief_probabilities)
            record = self._prediction_record(features, transition, budget, advisory_mode)
            self.ledger.register_prediction(record)
            records.append(record)
        self._emit(
            "predict_outcomes",
            {"workspace_revision": snapshot.revision, "candidates": len(candidates),
             "advisory_mode": advisory_mode, "records": len(records)},
            class_name="CognitiveWorldModel", module_name="ore_cwm.service")
        return tuple(records)

    def plan(
        self,
        snapshot: WorkspaceSnapshot,
        candidates: Sequence[CognitiveOperation],
        *,
        horizon: int | None = None,
        budget: CWMBudget | None = None,
        hard_constraints: HardConstraintSet | None = None,
        advisory_mode: str = AdvisoryMode.SHADOW.value,
        expected_revision: int | None = None,
        resource_state: ResourceState | None = None,
        unresolved_conflicts: int | None = None,
        belief_probabilities: Sequence[float] | None = None,
    ) -> CWMPlan:
        """Plan one bounded rollout from a consistent snapshot and return the advisory plan."""
        _require_snapshot(snapshot)
        features = self._extract_features(
            snapshot, resource_state=resource_state,
            unresolved_conflicts=unresolved_conflicts)
        planner = self._planner_for(hard_constraints)
        result = planner.plan(
            features, candidates,
            budget=budget or CWMBudget(
                horizon=self.config.default_horizon if horizon is None else horizon),
            expected_revision=expected_revision,
            belief_probabilities=belief_probabilities,
            advisory_mode=advisory_mode,
        )
        for record in result.predictions:
            self.ledger.register_prediction(record)
        self._emit(
            "plan",
            {"workspace_revision": snapshot.revision, "candidates": len(candidates),
             "fallback": None if result.fallback is None else result.fallback.trigger.value,
             "sequences": len(result.ranked_sequences), "advisory_mode": advisory_mode},
            class_name="CognitiveWorldModel", module_name="ore_cwm.service")
        return result

    # -- Steps 44-45: advisory bridges ---------------------------------------

    def advise_resource_coordinator(
        self,
        snapshot: WorkspaceSnapshot,
        candidates: Sequence[CognitiveOperation],
    ) -> dict[str, Any]:
        """Translate CWM predictions into ModuleResourceEstimate evidence for ORE."""
        from ore_contracts.resource_profiles import ModuleResourceEstimate

        records = self.predict_outcomes(snapshot, candidates)
        estimates: dict[str, Any] = {}
        for operation, record in zip(candidates, records):
            predicted = record.predicted
            duration = predicted.expected_cost.expected_duration_ms
            memory = predicted.expected_cost.expected_memory_mb
            cpu = predicted.expected_cost.expected_cpu_cost
            gain = min(1.0, predicted.information_gain.value)
            estimates[operation.module_id] = ModuleResourceEstimate(
                module_id=operation.module_id,
                expected_latency_ms=(
                    1000.0 if duration is MISSING else float(duration)),
                expected_memory_mb=(
                    128.0 if memory is MISSING else float(memory)),
                expected_cpu_cost=(1.0 if cpu is MISSING else float(cpu)),
                expected_information_gain=float(gain),
                estimated_success_probability=float(predicted.success_probability),
                estimate_source="cwm_prediction",
            )
        return estimates

    def advise_task_coordinator(self, task: CognitiveTask, plan: CWMPlan) -> CognitiveTask:
        """Rank candidates through the coordinator's documented metadata advisory channel."""
        if plan.fallback is not None or plan.first_action is None:
            return task
        if plan.first_action.operation_type is not OperationType.MODULE_EXECUTION:
            return task
        preferences = {plan.first_action.capability_id: plan.first_action.module_id}
        return replace(
            task,
            metadata={
                **task.metadata,
                "resource_module_preferences": preferences,
                "cwm_advisory": {
                    "prediction_ids": [
                        item.prediction_id for item in plan.predictions[:1]
                    ],
                    "model_version": self.transition_model.predictor_version,
                },
            },
        )

    # -- Steps 34-35: receding-horizon loop ----------------------------------

    def run_receding_horizon(
        self,
        workspace: Any,
        coordinator: CoordinatorLike,
        task: CognitiveTask,
        *,
        candidates_provider: CandidatesProvider | Sequence[CognitiveOperation],
        hard_constraints: HardConstraintSet | None = None,
        budget: CWMBudget | None = None,
        max_iterations: int = 8,
        mode: AdvisoryMode = AdvisoryMode.SHADOW,
        outcome_extractor: Callable[[Any], OutcomeDimensions] | None = None,
    ) -> CWMRunResult:
        """Dispatch only first actions through the coordinator and replan after each."""
        if max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        planner = self._planner_for(hard_constraints)
        steps: list[CWMStepRecord] = []
        outcomes: list[ObservedOutcome] = []
        comparisons: list[PredictionErrorReport] = []
        initial_score = self.goal_evaluator.evaluate(workspace.snapshot()).score
        expected_revision = workspace.revision
        stop_reason = "iteration_limit_reached"
        for iteration in range(max_iterations):
            snapshot = workspace.snapshot()
            features = self._extract_features(
                snapshot, recent_executions=outcomes[-self.features.recent_window:])
            candidates = (
                candidates_provider(snapshot) if callable(candidates_provider)
                else list(candidates_provider))
            result = planner.plan(
                features, candidates, budget=budget,
                expected_revision=expected_revision, advisory_mode=mode.value)
            for record in result.predictions:
                self.ledger.register_prediction(record)
            if result.stopped:
                stop_reason = "goal_satisfied_planner_stopped"
                steps.append(self._step_record(
                    iteration, workspace.revision, workspace.revision, result,
                    None, None, advisory_task_modified=False))
                break
            selected_prediction = self._prediction_for(result)
            advisory_mode_on = (
                mode is AdvisoryMode.ADVISORY and result.first_action is not None
                and result.first_action.operation_type is OperationType.MODULE_EXECUTION)
            advised_task = (
                self.advise_task_coordinator(task, result)
                if advisory_mode_on else task)
            revision_before = workspace.revision
            task_result = coordinator.coordinate(advised_task)
            revision_after = getattr(task_result, "workspace_revision", workspace.revision)
            outcome = self.observe(
                task_result,
                operation=result.first_action,
                workspace=workspace,
                revision_before=revision_before,
                baseline_goal_score=float(features.value("goal_completion"))
                if features.value("goal_completion") is not MISSING else None,
                dimensions=(
                    outcome_extractor(task_result) if outcome_extractor else None),
            )
            error_report = None
            if selected_prediction is not None:
                outcome = replace(
                    outcome, prediction_id=selected_prediction.prediction_id)
                error_report = self.ledger.align(selected_prediction, outcome)
                comparisons.append(error_report)
                if self.prediction_store is not None:
                    self.prediction_store.record_alignment(
                        selected_prediction, outcome, error_report)
            outcomes.append(outcome)
            steps.append(self._step_record(
                iteration, revision_before, revision_after, result,
                selected_prediction, outcome, advisory_task_modified=advisory_mode_on,
                error_report=error_report))
            self._emit(
                "run_receding_horizon_step",
                {"iteration": iteration, "revision_before": revision_before,
                 "revision_after": revision_after,
                 "module_id": result.first_action.module_id if result.first_action else None,
                 "advisory_mode": mode.value,
                 "prediction_error": (
                     None if error_report is None else error_report.to_dict())},
                class_name="CognitiveWorldModel", module_name="ore_cwm.service")
            evaluation = self.goal_evaluator.evaluate(workspace.snapshot())
            if evaluation.outcome.value in ("completed", "completed_with_uncertainty"):
                stop_reason = "goal_completed"
                break
            if evaluation.outcome.value in ("failed", "blocked"):
                stop_reason = f"goal_{evaluation.outcome.value}"
                break
            expected_revision = revision_after
        final_score = self.goal_evaluator.evaluate(workspace.snapshot()).score
        return CWMRunResult(
            workspace_id=workspace.workspace_id,
            goal=workspace.state.current_goal,
            advisory_mode=mode.value,
            steps=tuple(steps),
            outcomes=tuple(outcomes),
            initial_goal_score=initial_score,
            final_goal_score=final_score,
            stop_reason=stop_reason,
            operation_trace=self._operation_trace(steps),
            calibration=self.ledger.calibration.report().to_dict(),
            drift=self.ledger.drift.status().to_dict(),
        )

    # -- Step 51: shadow evaluation -----------------------------------------

    def run_shadow(
        self,
        workspace: Any,
        coordinator: CoordinatorLike,
        task: CognitiveTask,
        *,
        candidates_provider: CandidatesProvider | Sequence[CognitiveOperation],
        hard_constraints: HardConstraintSet | None = None,
        budget: CWMBudget | None = None,
        max_iterations: int = 8,
        outcome_extractor: Callable[[Any], OutcomeDimensions] | None = None,
    ) -> ShadowReport:
        """Predict without changing execution decisions and compare against outcomes."""
        run = self.run_receding_horizon(
            workspace, coordinator, task,
            candidates_provider=candidates_provider,
            hard_constraints=hard_constraints, budget=budget,
            max_iterations=max_iterations, mode=AdvisoryMode.SHADOW,
            outcome_extractor=outcome_extractor)
        comparisons = list(self.ledger.error_reports())
        relevant = comparisons[-len(run.steps):] if run.steps else []
        return ShadowReport(
            workspace_id=run.workspace_id,
            goal=run.goal,
            steps=run.steps,
            comparisons=tuple(relevant),
            calibration=run.calibration,
            drift=run.drift,
            tasks_modified=False,
        )

    # -- observation ---------------------------------------------------------

    def observe(
        self,
        task_result: Any,
        *,
        operation: CognitiveOperation | None,
        workspace: Any,
        revision_before: int,
        baseline_goal_score: float | None = None,
        dimensions: OutcomeDimensions | None = None,
        execution_status: ExecutionStatus | None = None,
        notes: str = "",
    ) -> ObservedOutcome:
        """Extract the observed outcome of one dispatched action with identical metrics."""
        status_text = str(getattr(task_result, "status", "RESOLVED")).upper()
        status = execution_status or _map_execution_status(status_text)
        module_results = dict(getattr(task_result, "module_results", {}) or {})
        artifacts = tuple(getattr(task_result, "artifacts", ()) or ())
        revision_after = int(getattr(task_result, "workspace_revision", workspace.revision))
        generated = len(artifacts)
        accepted = revision_after > revision_before or any(
            str(getattr(value, "status", "")).upper() == "SUCCESS"
            for value in module_results.values())
        lifecycle = ArtifactLifecycle.ACCEPTED if accepted else ArtifactLifecycle.GENERATED
        progress_delta = MISSING
        current_score = self.goal_evaluator.evaluate(workspace.snapshot()).score
        if baseline_goal_score is not None:
            progress_delta = current_score - float(baseline_goal_score)
        validity = MISSING
        if module_results:
            successful = sum(
                1 for value in module_results.values()
                if str(getattr(value, "status", "")).upper() == "SUCCESS")
            validity = successful / len(module_results)
        if dimensions is None:
            dimensions = OutcomeDimensions(
                goal_progress_delta=progress_delta,
                output_validity=validity,
                failure_class=(
                    status_text.lower() if status_text in ("FAILED", "BLOCKED")
                    else None),
            )
        return ObservedOutcome(
            operation_id=(operation.operation_id if operation is not None else "unattributed"),
            module_id=(operation.module_id if operation is not None else "unknown_module"),
            workspace_id=workspace.workspace_id,
            workspace_revision_before=revision_before,
            workspace_revision_after=revision_after,
            execution_status=status,
            dimensions=dimensions,
            artifact_lifecycle=lifecycle,
            artifacts_generated=generated,
            artifacts_accepted=generated if accepted else 0,
            notes=notes,
        )

    # -- Step 9: candidates from existing ORE control logic ------------------

    @staticmethod
    def candidates_from_registry(
        registry: ModuleRegistry,
        task: CognitiveTask,
        *,
        permission_policy: PermissionPolicy | None = None,
    ) -> tuple[CognitiveOperation, ...]:
        """Obtain feasible candidates from the existing module registry contracts."""
        manifests = {
            manifest.identity.module_id: manifest
            for manifest in registry.list_manifests()
        }
        policy = permission_policy or PermissionPolicy()
        operations: list[CognitiveOperation] = []
        for capability in task.required_capabilities:
            for record in registry.find_by_capability(capability):
                if not record.runnable:
                    continue
                manifest = manifests.get(record.module_id)
                if manifest is None:
                    continue
                preconditions = tuple(
                    OperationPrecondition(
                        "artifact",
                        f"{port.port_id}:{port.contract_id}@{version}")
                    for port in manifest.input_ports
                    for version in port.accepted_versions
                )
                output_schema = tuple(
                    f"{port.contract_id}@{port.contract_version}"
                    for port in manifest.output_ports
                )
                operations.append(CognitiveOperation(
                    operation_type=OperationType.MODULE_EXECUTION,
                    capability_id=capability,
                    module_id=record.module_id,
                    preconditions=preconditions,
                    output_schema=output_schema,
                    produced_fields=tuple(sorted(policy.allowed_fields(record.module_id))),
                ))
        return tuple(operations)

    # -- internals -----------------------------------------------------------

    def _extract_features(
        self,
        snapshot: WorkspaceSnapshot,
        *,
        resource_state: ResourceState | None = None,
        unresolved_conflicts: int | None = None,
        recent_executions: Sequence[ObservedOutcome] = (),
    ) -> CognitiveStateFeatures:
        """Extract the versioned feature projection for one snapshot (Steps 1-3)."""
        return self.features.extract(
            snapshot, resource_state=resource_state,
            unresolved_conflicts=unresolved_conflicts,
            recent_executions=recent_executions or self.ledger.outcomes(),
        )

    def _context_for(
        self, features: CognitiveStateFeatures, operation: CognitiveOperation,
    ) -> OperationContext:
        """Build the conditioning context for one candidate (Step 12)."""
        input_size = MISSING
        if operation.input_artifact_refs:
            input_size = len(operation.input_artifact_refs)
        return OperationContext(
            operation_type=operation.operation_type.value,
            module_id=operation.module_id,
            goal_class=features.goal,
            input_size=input_size,
            dependency_depth=len(operation.depends_on),
        )

    def _prediction_record(
        self,
        features: CognitiveStateFeatures,
        transition: Any,
        budget: CWMBudget,
        advisory_mode: str,
    ) -> PredictionRecord:
        """Build one provisional prediction record (Step 6)."""
        return PredictionRecord(
            workspace_id=features.workspace_id,
            workspace_revision=features.workspace_revision,
            goal_id=features.goal,
            operation_id=transition.operation.operation_id,
            feature_snapshot_hash=features.feature_snapshot_hash,
            predicted=transition.predicted_outcome,
            predictor_version=self.transition_model.predictor_version,
            horizon=budget.horizon,
            constraints_considered=self.hard_constraints.constraint_names(),
            advisory_mode=advisory_mode,
            notes=(
                f"operation_class={classify_operation(transition.operation)};"
                f"module_id={transition.operation.module_id}"
            ),
        )

    def _planner_for(self, hard_constraints: HardConstraintSet | None) -> MPCPlanner:
        """Return the planner to use, honoring per-call constraint overrides."""
        if hard_constraints is None or hard_constraints is self.hard_constraints:
            return self.planner
        return MPCPlanner(
            self.transition_model, hard_constraints, self.planner.objective,
            self.confidence_gate, self.fallback_policy, self.config)

    def _prediction_for(self, result: CWMPlan) -> PredictionRecord | None:
        """Return the prediction record matching the selected first action."""
        if result.first_action is None:
            return None
        for record in result.predictions:
            if record.operation_id == result.first_action.operation_id:
                return record
        return None

    def _step_record(
        self,
        iteration: int,
        revision_before: int,
        revision_after: int,
        result: CWMPlan,
        selected_prediction: PredictionRecord | None,
        outcome: ObservedOutcome | None,
        advisory_task_modified: bool,
        error_report: PredictionErrorReport | None = None,
    ) -> CWMStepRecord:
        """Freeze one loop iteration into a step record."""
        first = result.first_action
        predicted_success = None
        predicted_duration = None
        if selected_prediction is not None:
            predicted_success = float(selected_prediction.predicted.success_probability)
            duration = selected_prediction.predicted.expected_cost.expected_duration_ms
            predicted_duration = None if duration is MISSING else float(duration)
        return CWMStepRecord(
            iteration=iteration,
            workspace_revision_before=revision_before,
            workspace_revision_after=revision_after,
            selected_operation_id=None if first is None else first.operation_id,
            selected_module_id=None if first is None else first.module_id,
            predicted_success=predicted_success,
            predicted_duration_ms=predicted_duration,
            observed_artifact_lifecycle=(
                None if outcome is None else outcome.artifact_lifecycle.value),
            prediction_error=error_report,
            fallback_trigger=(
                None if result.fallback is None else result.fallback.trigger.value),
            advisory_task_modified=advisory_task_modified,
        )

    def _operation_trace(self, steps: Sequence[CWMStepRecord]) -> tuple[str, ...]:
        """Map executed pipeline phases onto the registered CWM operation inventory."""
        trace = ["CWM-01", "CWM-02", "CWM-03", "CWM-09", "CWM-12", "CWM-27", "CWM-28",
                 "CWM-29", "CWM-30", "CWM-31", "CWM-32", "CWM-33"]
        if steps:
            trace.extend(["CWM-34", "CWM-35", "CWM-36", "CWM-37", "CWM-38", "CWM-42"])
        trace.extend(["CWM-20", "CWM-39", "CWM-41", "CWM-43", "CWM-46", "CWM-47",
                      "CWM-48", "CWM-49", "CWM-50"])
        if any(item.observed_artifact_lifecycle is not None for item in steps):
            trace.extend(["CWM-40"])
        return tuple(trace)

    def _emit(
        self, method_name: str, metadata: Mapping[str, Any], *,
        class_name: str, module_name: str,
    ) -> None:
        """Record one observer event when logging is enabled (Step 47)."""
        if self.observer is None:
            return
        try:
            self.observer.record(
                method_name, metadata, class_name=class_name, module_name=module_name)
        except Exception:  # observability must never break prediction (Step 48)
            return


def _require_snapshot(snapshot: WorkspaceSnapshot) -> None:
    """Require a consistent authoritative snapshot object (Step 1)."""
    if not isinstance(snapshot, WorkspaceSnapshot):
        raise TypeError("snapshot must be an ore_workspace WorkspaceSnapshot")


def _map_execution_status(status_text: str) -> ExecutionStatus:
    """Map one coordinator status onto the CWM execution-status vocabulary."""
    if status_text in ("RESOLVED", "PARTIAL", "AMBIGUOUS", "ABSTAINED"):
        return ExecutionStatus.EXECUTED
    if status_text == "BLOCKED":
        return ExecutionStatus.ABORTED
    return ExecutionStatus.EXECUTED


def record_result_to_sqlite(
    result: CWMRunResult,
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist the ordered CWM operation trace and per-step metrics to SQLite."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    database = LoggerDatabase(database_path)
    catalogue = sync_repository(database, root, include_paths={"ore_cwm"})
    run_id = database.create_run(
        script_path="ore_cwm/cognitive_world_model",
        module_name="ore_cwm.service",
        entry_kind="cognitive_world_model",
        working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        sequence = 0
        for operation_id in result.operation_trace:
            sequence += 1
            database.record_event(
                run_id=run_id,
                sequence_no=sequence,
                event_kind="operation",
                source_path="ore_cwm/registry.py",
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                metadata={"operation_id": operation_id,
                           "registry": registry_as_dicts()},
            )
        for step in result.steps:
            sequence += 1
            database.record_event(
                run_id=run_id,
                sequence_no=sequence,
                event_kind="cognitive_world_model_step",
                source_path="ore_cwm/service.py",
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                metadata=step.to_dict(),
            )
        database.finish_run(run_id, "completed",
                            duration_ms=round((time.perf_counter() - started) * 1000.0, 3))
    except Exception as exc:
        database.finish_run(
            run_id, "failed",
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            error_summary=f"{type(exc).__name__}: {exc}")
        raise
    return {"run_id": run_id, "annotation_catalogue": catalogue,
            "database_summary": database.summary()}


__all__ = [
    "DEFAULT_OPERATION_CLASS_BASELINES",
    "OPERATION_REGISTRY",
    "CWMRunResult",
    "CWMStepRecord",
    "CognitiveWorldModel",
    "CoordinatorLike",
    "ShadowReport",
    "record_result_to_sqlite",
    "registry_as_dicts",
]
