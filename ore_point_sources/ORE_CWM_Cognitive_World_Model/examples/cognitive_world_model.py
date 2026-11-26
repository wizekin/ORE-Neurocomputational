"""Demonstrate the Cognitive World Model advisory loop over a shared workspace.

The example seeds execution history, predicts candidate outcomes from a consistent
workspace snapshot, plans a bounded MPC rollout, and runs shadow and advisory loops
with a scripted coordinator double. CWM itself never publishes workspace findings.
"""
from __future__ import annotations

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ore_contracts.coordinator import CognitiveTask  # noqa: E402
from ore_workspace import WorkspaceManager  # noqa: E402

from ore_cwm import (  # noqa: E402
    AdvisoryMode,
    ArtifactLifecycle,
    CognitiveOperation,
    CognitiveWorldModel,
    CWMConfig,
    ExecutionLedger,
    ExecutionStatus,
    ObservedOutcome,
    OperationType,
    OutcomeClass,
    OutcomeDimensions,
    PredictedOutcome,
    PredictionRecord,
    PredictionUncertainty,
    ResourceEnvelope,
    RiskAssessment,
    record_result_to_sqlite,
)
from ore_cwm.contracts import InformationGainEstimate  # noqa: E402


def candidate(module_id: str, capability: str, produced: tuple[str, ...],
              cpu_cost: float) -> CognitiveOperation:
    """Build one demonstration candidate standing in for a registry-provided operation."""
    return CognitiveOperation(
        operation_type=OperationType.MODULE_EXECUTION,
        capability_id=capability,
        module_id=module_id,
        output_schema=(f"{module_id}.output@1.0.0",),
        produced_fields=produced,
        resource_envelope=ResourceEnvelope(expected_cpu_cost=cpu_cost),
    )


def seed(ledger: ExecutionLedger, module_id: str, operation_class: str,
         accepted: bool, duration_ms: float, cpu_cost: float, progress: float) -> None:
    """Align one synthetic prediction/outcome pair into the ledger estimators."""
    probability = 0.8 if accepted else 0.2
    predicted = PredictedOutcome(
        success_probability=probability,
        outcome_class_probabilities={OutcomeClass.SUCCESS.value: 1.0},
        expected_cost=ResourceEnvelope(
            expected_duration_ms=duration_ms, expected_cpu_cost=cpu_cost,
            expected_memory_mb=64.0, sample_count=1),
        predicted_feature_delta={},
        goal_progress_estimate=progress,
        information_gain=InformationGainEstimate(0.1, "proxy", "seed observation"),
        risks=RiskAssessment(failure_risk=max(0.0, 1.0 - probability)),
        uncertainty=PredictionUncertainty(
            success_credible_interval=(max(0.0, probability - 0.2),
                                       min(1.0, probability + 0.2)),
            sample_count=1),
    )
    prediction = PredictionRecord(
        workspace_id="seed_workspace", workspace_revision=0,
        goal_id="understand_sentence",
        operation_id=f"seed_{module_id}_{len(ledger.error_reports())}",
        feature_snapshot_hash="0" * 64, predicted=predicted,
        predictor_version="seed",
        notes=f"operation_class={operation_class};module_id={module_id}")
    outcome = ObservedOutcome(
        operation_id=prediction.operation_id, module_id=module_id,
        workspace_id="seed_workspace", workspace_revision_before=0,
        workspace_revision_after=1 if accepted else 0,
        execution_status=ExecutionStatus.EXECUTED,
        dimensions=OutcomeDimensions(
            goal_progress_delta=progress, elapsed_ms=duration_ms, cpu_cost=cpu_cost),
        artifact_lifecycle=(
            ArtifactLifecycle.ACCEPTED if accepted else ArtifactLifecycle.GENERATED),
        artifacts_generated=1, artifacts_accepted=1 if accepted else 0)
    ledger.register_prediction(prediction)
    ledger.align(prediction, outcome)


class ScriptedCoordinator:
    """Deterministic coordinator double publishing one field per dispatch."""

    def __init__(self, workspace: object, default_module: str,
                 fields: dict[str, tuple[str, ...]]) -> None:
        """Configure the workspace, the default module policy, and field ownership."""
        self.workspace = workspace
        self.default_module = default_module
        self.fields = fields
        self.executed: list[str] = []

    def coordinate(self, task: CognitiveTask, context: object | None = None) -> object:
        """Dispatch exactly one module action through the workspace publication path."""
        preferences = task.metadata.get("resource_module_preferences", {})
        module_id = next(iter(preferences.values()), self.default_module)
        self.executed.append(module_id)
        from ore_workspace import FindingPublisher

        producer = {"lexical_processor": "lexical_processor",
                    "semantic_processor": "semantic_processor"}.get(module_id, module_id)
        publisher = FindingPublisher(self.workspace, producer)
        for field in self.fields.get(module_id, ()):
            publisher.publish(field, {"module": module_id, "value": 1.0},
                               key=f"{module_id}-{field}")
        return type("TaskResult", (), {
            "status": "RESOLVED",
            "module_results": {module_id: type("Stub", (), {"status": "SUCCESS"})()},
            "artifacts": (type("ArtifactStub", (), {"contract_id": "stub"})(),),
            "workspace_revision": self.workspace.revision,
        })()


def main() -> None:
    """Run shadow and advisory loops and persist the trace to SQLite."""
    ledger = ExecutionLedger(CWMConfig(min_support=2, max_candidates=4))
    for _ in range(4):
        seed(ledger, "lexical_processor", "module_execution", True, 280.0, 0.8, 0.35)
    for _ in range(4):
        seed(ledger, "semantic_processor", "composition", True, 480.0, 1.5, 0.80)
    model = CognitiveWorldModel(config=CWMConfig(min_support=2, max_candidates=4))
    model.ledger = ledger
    model.transition_model.ledger = ledger

    manager = WorkspaceManager(enable_logging=False)
    workspace = manager.create_workspace(
        current_input="The bilingual student translated the ambiguous sentence.",
        current_goal="understand_sentence",
    )
    coordinator = ScriptedCoordinator(workspace, default_module="lexical_processor",
                                      fields={"lexical_processor": ("lexical_hypotheses",),
                                              "semantic_processor": ("semantic_hypotheses",)})
    task = CognitiveTask(goal="understand_sentence",
                         required_capabilities=("language.lexical_processing",))
    candidates = (
        candidate("lexical_processor", "language.lexical_processing",
                  ("lexical_hypotheses",), 0.8),
        candidate("semantic_processor", "language.semantic_composition",
                  ("semantic_hypotheses",), 1.5),
    )

    snapshot = workspace.snapshot()
    records = model.predict_outcomes(snapshot, candidates, advisory_mode="shadow")
    print(f"predicted {len(records)} candidates at revision {snapshot.revision}")
    for record in records:
        outcome = record.predicted
        print(f"  {record.notes}: success={outcome.success_probability:.2f} "
              f"support={outcome.uncertainty.sample_count} "
              f"duration_ms={outcome.expected_cost.expected_duration_ms:.0f}")

    shadow = model.run_shadow(
        workspace, coordinator, task,
        candidates_provider=lambda _snapshot: candidates, max_iterations=2)
    print(f"shadow: steps={len(shadow.steps)} comparisons={len(shadow.comparisons)} "
          f"tasks_modified={shadow.tasks_modified}")

    advisory_coordinator = ScriptedCoordinator(
        workspace, default_module="lexical_processor",
        fields={"lexical_processor": ("lexical_hypotheses",),
                "semantic_processor": ("semantic_hypotheses",)})
    run = model.run_receding_horizon(
        workspace, advisory_coordinator, task,
        candidates_provider=lambda _snapshot: candidates,
        max_iterations=1, mode=AdvisoryMode.ADVISORY)
    print(f"advisory: modules={advisory_coordinator.executed} "
          f"goal_progress={run.goal_progress:.3f} stop_reason={run.stop_reason}")

    database = Path("ore_logs") / "cwm_example.sqlite3"
    database.parent.mkdir(exist_ok=True)
    summary = record_result_to_sqlite(run, database, repository_root=REPOSITORY_ROOT)
    print(f"persisted run {summary['run_id']} to {database}")
    manager.close()


if __name__ == "__main__":
    main()
