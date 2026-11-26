"""Acceptance tests for the ORE Cognitive World Model (Point 04, ranked 13).

Every test maps onto one acceptance criterion from the algorithm specification,
section 8: predictions stay provisional, missing measurements never read as zero,
hard constraints override predicted utility, dispatch only happens through the
existing coordinator, fallbacks are deterministic, learning survives retries and
duplicates, calibration is chronological, and shadow mode never alters execution.
"""
import json
import math
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from ore_contracts import (
    ComputeBudget,
    ContractArtifact,
    ContractCatalog,
    InputPort,
    ModuleIdentity,
    ModuleManifest,
    ModuleRegistry,
    ModuleResult,
    ModuleStatus,
    OperationMode,
    OutputPort,
    RunContext,
    make_envelope,
)
from ore_contracts.coordinator import CognitiveTask, CognitiveTaskCoordinator
from ore_contracts.resource_profiles import ModuleResourceEstimate
from ore_cwm import (
    MISSING,
    OPERATION_REGISTRY,
    AdvisoryMode,
    ArtifactLifecycle,
    BetaBinomialEstimator,
    CognitiveOperation,
    CognitiveWorldModel,
    CWMBudget,
    CWMConfig,
    ExecutionLedger,
    ExecutionStatus,
    ExponentialMovingEstimator,
    HardConstraintSet,
    ModelVersion,
    ModelVersionRegistry,
    ObservedOutcome,
    OperationType,
    OutcomeClass,
    OutcomeDimensions,
    PredictedOutcome,
    PredictionRecord,
    PredictionUncertainty,
    ResourceEnvelope,
    RiskAssessment,
    RobustEstimator,
    SQLiteCWMObserver,
    SQLitePredictionStore,
    StaleRevisionError,
    evaluate_chronological,
    evidence_coverage_proxy,
    expected_information_gain,
    fallback_trigger_for,
    record_result_to_sqlite,
    registry_as_dicts,
)
from ore_cwm.contracts import CWM_PREDICTOR_VERSION, InformationGainEstimate
from ore_cwm.observability import register_cwm_annotations
from ore_workspace import WorkspaceManager

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def make_operation(
    module_id,
    capability="language.lexical_processing",
    produced=("lexical_hypotheses",),
    cpu_cost=None,
    timeout_ms=None,
    depends_on=(),
    safety=None,
):
    """Build one candidate operation standing in for a registry-provided candidate."""
    from ore_cwm import SafetyClass

    envelope = ResourceEnvelope.missing()
    if cpu_cost is not None:
        envelope = ResourceEnvelope(expected_cpu_cost=cpu_cost)
    return CognitiveOperation(
        operation_type=OperationType.MODULE_EXECUTION,
        capability_id=capability,
        module_id=module_id,
        output_schema=(f"{module_id}.output@1.0.0",),
        produced_fields=tuple(produced),
        resource_envelope=envelope,
        timeout_ms=MISSING if timeout_ms is None else timeout_ms,
        depends_on=tuple(depends_on),
        safety=safety or SafetyClass.SAFE,
    )


def make_model(config=None, **overrides):
    """Build one Cognitive World Model with test-friendly defaults."""
    settings = {"min_support": 2, "max_candidates": 4, "default_horizon": 2}
    settings.update(overrides)
    configuration = config or CWMConfig(**settings)
    return CognitiveWorldModel(config=configuration)


def make_workspace(current_input="The bilingual student translated the sentence."):
    """Create one isolated workspace without SQLite logging."""
    manager = WorkspaceManager(enable_logging=False)
    workspace = manager.create_workspace(
        current_input=current_input, current_goal="understand_sentence")
    return manager, workspace


def seeded_prediction(
    operation_id="seed_operation",
    module_id="seed_module",
    probability=0.7,
    duration_ms=300.0,
    cpu_cost=1.0,
    progress=0.4,
    operation_class="module_execution",
):
    """Build one minimal prediction record used to seed execution history."""
    predicted = PredictedOutcome(
        success_probability=probability,
        outcome_class_probabilities={OutcomeClass.SUCCESS.value: 1.0},
        expected_cost=ResourceEnvelope(
            expected_duration_ms=duration_ms,
            expected_cpu_cost=cpu_cost,
            expected_memory_mb=64.0,
            sample_count=1,
        ),
        predicted_feature_delta={},
        goal_progress_estimate=progress,
        information_gain=InformationGainEstimate(0.1, "proxy", "seed observation"),
        risks=RiskAssessment(failure_risk=max(0.0, 1.0 - probability)),
        uncertainty=PredictionUncertainty(
            success_credible_interval=(max(0.0, probability - 0.2), min(1.0, probability + 0.2)),
            sample_count=1,
        ),
    )
    return PredictionRecord(
        workspace_id="seed_workspace",
        workspace_revision=0,
        goal_id="understand_sentence",
        operation_id=operation_id,
        feature_snapshot_hash="0" * 64,
        predicted=predicted,
        predictor_version="seed",
        notes=f"operation_class={operation_class};module_id={module_id}",
    )


def seed_observation(
    ledger,
    module_id,
    *,
    accepted=True,
    probability=0.8,
    duration_ms=300.0,
    cpu_cost=1.0,
    progress=0.4,
    execution_status=ExecutionStatus.EXECUTED,
    operation_class="module_execution",
):
    """Align one synthetic prediction/outcome pair into the ledger estimators."""
    prediction = seeded_prediction(
        operation_id=f"seed_{module_id}_{ledger_calibration_count(ledger)}",
        module_id=module_id,
        probability=probability,
        duration_ms=duration_ms,
        cpu_cost=cpu_cost,
        progress=progress,
        operation_class=operation_class,
    )
    outcome = ObservedOutcome(
        operation_id=prediction.operation_id,
        module_id=module_id,
        workspace_id="seed_workspace",
        workspace_revision_before=0,
        workspace_revision_after=1 if accepted else 0,
        execution_status=execution_status,
        dimensions=OutcomeDimensions(
            goal_progress_delta=progress,
            elapsed_ms=duration_ms,
            cpu_cost=cpu_cost,
        ),
        artifact_lifecycle=(
            ArtifactLifecycle.ACCEPTED if accepted else ArtifactLifecycle.GENERATED),
        artifacts_generated=1,
        artifacts_accepted=1 if accepted else 0,
    )
    ledger.register_prediction(prediction)
    return ledger.align(prediction, outcome)


def ledger_calibration_count(ledger):
    """Return a monotonically increasing seed counter for unique operation IDs."""
    count = ledger.to_dict().get("error_reports", 0)
    return count


class RecordingCoordinator:
    """Deterministic coordinator double that records every task it dispatches."""

    def __init__(self, workspace, default_module="lexical_processor",
                 modules=None, costs=None, progress=None):
        """Configure the double with a default module policy and per-module effects."""
        self.workspace = workspace
        self.default_module = default_module
        self.modules = modules or {
            "lexical_processor": ("lexical_hypotheses",),
            "semantic_processor": ("semantic_hypotheses",),
        }
        self.costs = costs or {"lexical_processor": 1.0, "semantic_processor": 2.0}
        self.progress = progress or {
            "lexical_processor": 0.1, "semantic_processor": 0.9}
        self.received = []
        self.executed = []

    def coordinate(self, task, context=None):
        """Dispatch exactly one module action and record the advisory task verbatim."""
        self.received.append(json.loads(json.dumps(dict(task.metadata))))
        preferences = task.metadata.get("resource_module_preferences", {})
        module_id = next(iter(preferences.values()), self.default_module)
        self.executed.append(module_id)
        fields = self.modules.get(module_id, ())
        revision_after = self.workspace.revision
        for field in fields:
            publisher = _publisher_for(self.workspace, module_id, field)
            publisher.publish(field, {"module": module_id, "value": 1.0},
                               key=f"{module_id}-{field}")
            revision_after = self.workspace.revision
        return _task_result(module_id, revision_after)


def _publisher_for(workspace, module_id, field):
    """Create the field-authorized FindingPublisher for one module."""
    from ore_workspace import FindingPublisher

    producer = {
        "lexical_hypotheses": "lexical_processor",
        "semantic_hypotheses": "semantic_processor",
        "syntactic_hypotheses": "syntax_processor",
    }.get(field, module_id)
    return FindingPublisher(workspace, producer)


def _task_result(module_id, revision_after):
    """Build one duck-typed coordinator result for the CWM observation path."""
    return type("TaskResult", (), {
        "status": "RESOLVED",
        "module_results": {module_id: type("ModuleResultStub", (), {"status": "SUCCESS"})()},
        "artifacts": (type("ArtifactStub", (), {"contract_id": "stub"})(),),
        "workspace_revision": revision_after,
    })()


def test_registry_contains_all_52_ordered_operations():
    """The 52 algorithmic steps register in source order with stable identifiers."""
    assert len(OPERATION_REGISTRY) == 52
    expected = [f"CWM-{index:02d}" for index in range(1, 53)]
    assert [item.operation_id for item in OPERATION_REGISTRY] == expected
    sections = {item.section for item in OPERATION_REGISTRY}
    assert any(section.startswith("A.") for section in sections)
    assert any(section.startswith("G.") for section in sections)
    assert registry_as_dicts()[0]["operation_id"] == "CWM-01"


def test_prediction_record_ties_to_revision_goal_operation_and_model_version():
    """Every prediction is tied to the exact workspace revision, goal, and model version."""
    manager, workspace = make_workspace()
    model = make_model()
    candidate = make_operation("lexical_processor")
    snapshot = workspace.snapshot()
    records = model.predict_outcomes(snapshot, (candidate,))
    features = model.features.extract(snapshot)
    assert len(records) == 1
    record = records[0]
    assert record.workspace_revision == snapshot.revision
    assert record.workspace_id == workspace.workspace_id
    assert record.goal_id == "understand_sentence"
    assert record.operation_id == candidate.operation_id
    assert record.predictor_version == CWM_PREDICTOR_VERSION
    assert record.feature_snapshot_hash == features.feature_snapshot_hash
    assert record.record_kind == "provisional_planning"
    manager.close()


def test_predictions_are_provisional_and_never_become_workspace_findings():
    """Predictions cannot be mistaken for accepted findings and never mutate the workspace."""
    manager, workspace = make_workspace()
    model = make_model()
    candidates = (make_operation("lexical_processor"),
                  make_operation("semantic_processor"))
    snapshot = workspace.snapshot()
    revision_before = workspace.revision
    plan = model.plan(snapshot, candidates)
    model.predict_outcomes(snapshot, candidates)
    assert workspace.revision == revision_before
    assert workspace.findings == {}
    assert workspace.events == []
    assert plan.first_action is None  # no history yet: the gate defers
    assert plan.fallback is not None
    assert plan.fallback.trigger.value == "low_confidence"
    for prediction in plan.predictions:
        assert prediction.record_kind == "provisional_planning"
    manager.close()


def test_missing_measurements_stay_missing_and_never_read_as_zero():
    """Unavailable measurements stay missing in serialization, arithmetic, and vectors."""
    dimensions = OutcomeDimensions.all_missing()
    assert dimensions.elapsed_ms is MISSING
    assert dimensions.to_dict()["elapsed_ms"] is None
    assert not MISSING  # the sentinel is falsy but never numerically zero
    envelope = ResourceEnvelope.missing().to_dict()
    assert envelope["expected_duration_ms"] is None
    manager, workspace = make_workspace()
    model = make_model()
    snapshot = workspace.snapshot()
    features = model.features.extract(snapshot, unresolved_conflicts=None)
    assert features.value("unresolved_conflicts") is MISSING
    with pytest.raises(ValueError):
        features.as_vector(defaults={})  # missing feature demands an explicit default
    vector = features.as_vector(defaults={
        "unresolved_conflicts": 1.0, "recent_module_success": 0.0,
        "resource_pressure": 1.0})
    assert vector[features.names().index("unresolved_conflicts")] == 1.0
    manager.close()


def test_estimators_match_hand_calculated_fixtures():
    """Beta-Binomial, EMA, and robust estimators match hand-computed values."""
    estimator = BetaBinomialEstimator(1.0, 1.0)
    for success in [True] * 8 + [False] * 2:
        estimator.record(success)
    assert estimator.posterior_mean() == pytest.approx(9.0 / 12.0)
    low, high = estimator.credible_interval(0.90)
    assert 0.0 <= low < 0.75 < high <= 1.0
    assert estimator.sample_count() == 10
    moving = ExponentialMovingEstimator(rho=0.5)
    moving.update(10.0)
    assert moving.mean() == pytest.approx(10.0)
    moving.update(20.0)
    assert moving.mean() == pytest.approx(15.0)
    assert moving.variance() == pytest.approx(6.25)
    robust = RobustEstimator(window=8)
    for value in (1.0, 2.0, 3.0, 4.0, 100.0):
        robust.record(value)
    assert robust.estimate() == pytest.approx(3.0)
    deviations = [0.0, 1.0, 1.0, 2.0, 97.0]
    ordered = sorted(deviations)
    mad = ordered[len(ordered) // 2]
    assert robust.dispersion() == pytest.approx(1.4826 * mad)


def test_information_gain_entropy_and_proxy_labeling():
    """Entropy estimates are exact and proxies are labeled as proxies."""
    prior = [0.5, 0.5]
    posterior_success = [0.8, 0.2]
    posterior_failure = [0.5, 0.5]
    gain = expected_information_gain(
        prior, {"success": posterior_success, "failure": posterior_failure},
        {"success": 0.8, "failure": 0.2})
    expected = (
        -(0.5 * math.log(0.5) + 0.5 * math.log(0.5))
        - 0.8 * -(0.8 * math.log(0.8) + 0.2 * math.log(0.2))
        - 0.2 * -(0.5 * math.log(0.5) + 0.5 * math.log(0.5))
    )
    assert gain == pytest.approx(expected)
    assert evidence_coverage_proxy(0.4, 0.6) == pytest.approx(0.2)
    estimate = InformationGainEstimate(0.1, "proxy", "coverage proxy")
    assert estimate.kind == "proxy"
    with pytest.raises(ValueError):
        InformationGainEstimate(0.1, "causal", "unsupported labeling")


def test_predict_outcomes_carries_uncertainty_support_and_calibration():
    """Predictions carry intervals, sample counts, risks, and calibration status."""
    ledger = ExecutionLedger(CWMConfig(min_support=2))
    for _ in range(4):
        seed_observation(ledger, "lexical_processor", accepted=True)
    model = make_model()
    model.ledger = ledger
    model.transition_model.ledger = ledger
    manager, workspace = make_workspace()
    snapshot = workspace.snapshot()
    records = model.predict_outcomes(snapshot, (make_operation("lexical_processor"),))
    predicted = records[0].predicted
    assert predicted.uncertainty.sample_count >= 4
    low, high = predicted.uncertainty.success_credible_interval
    assert low < predicted.success_probability < high
    assert predicted.uncertainty.calibration_status in ("calibrated", "uncalibrated")
    assert predicted.risks.failure_risk == pytest.approx(1.0 - predicted.success_probability)
    assert predicted.information_gain.kind in ("entropy", "proxy")
    assert predicted.explanation.startswith("predictive estimate")
    manager.close()


def test_hard_constraints_override_predicted_utility():
    """A sequence exceeding the resource budget is rejected regardless of its score."""
    ledger = ExecutionLedger(CWMConfig(min_support=2))
    for _ in range(4):
        seed_observation(ledger, "expensive_module", cpu_cost=8.0, progress=0.9)
    model = make_model()
    model.ledger = ledger
    model.transition_model.ledger = ledger
    constraints = HardConstraintSet(max_total_cost=5.0)
    manager, workspace = make_workspace()
    candidates = (make_operation("expensive_module", cpu_cost=8.0),)
    plan = model.plan(workspace.snapshot(), candidates, hard_constraints=constraints)
    budget_violations = [
        name for sequence in plan.ranked_sequences + plan.rejected_sequences
        for name, _ in sequence.violations if name == "resource_budget"
    ]
    assert budget_violations, "the over-budget sequence must carry the violation"
    for sequence in plan.ranked_sequences:
        if sequence.feasible:
            assert sequence.total_cost <= 5.0
    assert plan.first_action is None or plan.first_action.module_id != "expensive_module"
    manager.close()


def test_mpc_sequences_respect_deadlines_permissions_and_dependencies():
    """Sequences violating deadlines, permissions, or dependencies are infeasible."""
    ledger = ExecutionLedger(CWMConfig(min_support=2))
    for _ in range(4):
        seed_observation(ledger, "slow_module", duration_ms=400.0)
    model = make_model()
    model.ledger = ledger
    model.transition_model.ledger = ledger
    manager, workspace = make_workspace()
    snapshot = workspace.snapshot()
    deadline_constraints = HardConstraintSet(deadline_ms=100.0)
    plan = model.plan(
        snapshot, (make_operation("slow_module"),), hard_constraints=deadline_constraints)
    deadline_violations = [
        name for sequence in plan.ranked_sequences + plan.rejected_sequences
        for name, _ in sequence.violations if name == "deadline"
    ]
    assert deadline_violations
    permission_constraints = HardConstraintSet(allowed_modules=frozenset({"other_module"}))
    plan = model.plan(
        snapshot, (make_operation("slow_module"),), hard_constraints=permission_constraints)
    permission_violations = [
        name for sequence in plan.ranked_sequences + plan.rejected_sequences
        for name, _ in sequence.violations if name == "module_permissions"
    ]
    assert permission_violations
    dependency = make_operation("upstream_module")
    dependent = make_operation("downstream_module", depends_on=(dependency.operation_id,))
    plan = model.plan(snapshot, (dependent,))
    assert plan.fallback is not None
    assert plan.fallback.trigger.value == "no_safe_candidate"
    manager.close()


def test_verify_and_stop_actions_are_available_to_the_planner():
    """Weak evidence requires a verification step and satisfied goals may stop early."""
    ledger = ExecutionLedger(CWMConfig(min_support=2))
    for _ in range(4):
        seed_observation(ledger, "lexical_processor")
    model = make_model()
    model.ledger = ledger
    model.transition_model.ledger = ledger
    manager, workspace = make_workspace()
    constraints = HardConstraintSet(mandatory_verification_below_coverage=0.5)
    plan = model.plan(
        workspace.snapshot(), (make_operation("lexical_processor"),),
        hard_constraints=constraints)
    verification_violations = [
        name for sequence in plan.ranked_sequences + plan.rejected_sequences
        for name, _ in sequence.violations
        if name == "mandatory_verification"
    ]
    assert verification_violations
    from ore_cwm import stop_action, verification_action

    assert stop_action().operation_type is OperationType.STOP
    assert verification_action().operation_type is OperationType.VERIFY
    manager.close()


def test_only_first_action_executes_then_replans_from_new_revision():
    """One dispatch per iteration, each replan starts from the new authoritative revision."""
    ledger = ExecutionLedger(CWMConfig(min_support=2))
    for _ in range(4):
        seed_observation(ledger, "lexical_processor", progress=0.4)
    model = make_model()
    model.ledger = ledger
    model.transition_model.ledger = ledger
    manager, workspace = make_workspace()
    coordinator = RecordingCoordinator(workspace)
    task = CognitiveTask(
        goal="understand_sentence",
        required_capabilities=("language.lexical_processing",),
    )
    candidates = (make_operation("lexical_processor"),
                  make_operation("semantic_processor"))
    run = model.run_receding_horizon(
        workspace, coordinator, task, candidates_provider=lambda _snapshot: candidates,
        max_iterations=2, mode=AdvisoryMode.SHADOW)
    assert len(coordinator.received) == 2
    assert len(run.steps) == 2
    assert run.steps[0].workspace_revision_before == 0
    assert run.steps[1].workspace_revision_before == run.steps[0].workspace_revision_after
    assert run.steps[1].workspace_revision_before > 0
    snapshots_revisions = [item.workspace_revision_before for item in run.steps]
    assert snapshots_revisions == sorted(set(snapshots_revisions))
    manager.close()


def test_shadow_mode_does_not_alter_execution_decisions():
    """Shadow predictions leave every dispatched task unmodified and the workspace intact."""
    ledger = ExecutionLedger(CWMConfig(min_support=2))
    for _ in range(4):
        seed_observation(ledger, "lexical_processor")
    model = make_model()
    model.ledger = ledger
    model.transition_model.ledger = ledger
    manager, workspace = make_workspace()
    coordinator = RecordingCoordinator(workspace, default_module="lexical_processor")
    task = CognitiveTask(
        goal="understand_sentence",
        required_capabilities=("language.lexical_processing",),
        metadata={"session": "shadow-test"},
    )
    candidates = (make_operation("lexical_processor"),
                  make_operation("semantic_processor"))
    report = model.run_shadow(
        workspace, coordinator, task,
        candidates_provider=lambda _snapshot: candidates, max_iterations=2)
    assert report.tasks_modified is False
    assert coordinator.received, "the coordinator must still dispatch unmodified tasks"
    for metadata in coordinator.received:
        assert "resource_module_preferences" not in metadata
        assert "cwm_advisory" not in metadata
        assert metadata == {"session": "shadow-test"}
    for step in report.steps:
        assert step.advisory_task_modified is False
    assert report.comparisons
    assert report.calibration["sample_count"] > 0
    manager.close()


def test_advisory_mode_injects_only_the_documented_metadata_channel():
    """Advisory influence flows only through the coordinator's documented channel."""
    ledger = ExecutionLedger(CWMConfig(min_support=2))
    for _ in range(4):
        seed_observation(ledger, "lexical_processor", progress=0.4)
    model = make_model()
    model.ledger = ledger
    model.transition_model.ledger = ledger
    manager, workspace = make_workspace()
    coordinator = RecordingCoordinator(workspace)
    task = CognitiveTask(
        goal="understand_sentence",
        required_capabilities=("language.lexical_processing",),
        metadata={"session": "advisory-test"},
    )
    candidates = (make_operation("lexical_processor"),)
    run = model.run_receding_horizon(
        workspace, coordinator, task, candidates_provider=lambda _snapshot: candidates,
        max_iterations=1, mode=AdvisoryMode.ADVISORY)
    assert coordinator.received, "dispatch still flows through the coordinator"
    for metadata in coordinator.received:
        assert metadata.get("resource_module_preferences") == {
            "language.lexical_processing": "lexical_processor"}
        assert metadata["session"] == "advisory-test"
    assert run.steps[0].advisory_task_modified is True
    assert not hasattr(model, "dispatch_module"), "CWM exposes no direct dispatch path"
    manager.close()


def test_fallback_triggers_are_deterministic_and_mapped():
    """Stale revisions, cancellations, and cold starts trigger deterministic fallbacks."""
    model = make_model()
    manager, workspace = make_workspace()
    snapshot = workspace.snapshot()
    plan = model.plan(snapshot, (make_operation("lexical_processor"),),
                      expected_revision=99)
    assert plan.fallback.trigger.value == "stale_revision"
    assert plan.deferred_to_coordinator_defaults is True
    plan = model.plan(snapshot, (make_operation("lexical_processor"),),
                      budget=CWMBudget(cancelled=True))
    assert plan.fallback.trigger.value == "timeout"
    cold_model = make_model(min_support=3)
    plan = cold_model.plan(snapshot, (make_operation("lexical_processor"),))
    assert plan.fallback.trigger.value == "low_confidence"
    with pytest.raises(TypeError):
        model.plan("not-a-snapshot", ())
    assert fallback_trigger_for(StaleRevisionError("stale")).value == "stale_revision"
    assert fallback_trigger_for(TimeoutError()).value == "timeout"
    assert fallback_trigger_for(RuntimeError("model")).value == "model_error"
    manager.close()


def test_alignment_survives_retries_cancellation_and_duplicate_events():
    """Retried executions learn from the final attempt; duplicates are labeled, not counted."""
    ledger = ExecutionLedger(CWMConfig(min_support=2))
    for _ in range(3):
        seed_observation(ledger, "retry_module", accepted=True)
    before = ledger.stats_for("module_execution", "retry_module").success.successes
    prediction = seeded_prediction(operation_id="retry_prediction", module_id="retry_module")
    ledger.register_prediction(prediction)
    cancelled = ObservedOutcome(
        operation_id="retry_prediction", module_id="retry_module",
        workspace_id="seed_workspace", workspace_revision_before=0,
        workspace_revision_after=0, execution_status=ExecutionStatus.CANCELLED,
        dimensions=OutcomeDimensions.all_missing())
    cancelled_report = ledger.align(prediction, cancelled)
    assert cancelled_report.excluded_from_learning
    assert "cancelled" in cancelled_report.exclusion_reason
    executed = ObservedOutcome(
        operation_id="retry_prediction", module_id="retry_module",
        workspace_id="seed_workspace", workspace_revision_before=0,
        workspace_revision_after=1, execution_status=ExecutionStatus.EXECUTED,
        dimensions=OutcomeDimensions(goal_progress_delta=0.4, elapsed_ms=300.0),
        artifact_lifecycle=ArtifactLifecycle.ACCEPTED, artifacts_generated=1,
        artifacts_accepted=1)
    executed_report = ledger.align(prediction, executed)
    assert not executed_report.excluded_from_learning
    after = ledger.stats_for("module_execution", "retry_module").success.successes
    assert after == before + 1
    duplicate_report = ledger.align(prediction, executed)
    assert duplicate_report.excluded_from_learning
    assert "duplicate" in duplicate_report.exclusion_reason
    assert ledger.stats_for("module_execution", "retry_module").success.successes == after


def test_calibration_uses_held_out_chronological_data():
    """Calibration is evaluated on the latest held-out segment in time order."""
    pairs = [(0.5, index % 2 == 0) for index in range(14)]
    pairs += [(0.9, False) for _ in range(6)]
    report = evaluate_chronological(pairs, holdout_fraction=0.3)
    assert report.sample_count == 6
    assert report.brier_score == pytest.approx(0.81, abs=1e-9)
    assert report.log_loss == pytest.approx(-math.log(0.1), abs=1e-6)
    assert report.status in ("over_confident", "under_confident", "uncalibrated")
    with pytest.raises(ValueError):
        evaluate_chronological(pairs, holdout_fraction=0.0)


def test_drift_lowers_confidence_and_restores_validated_baseline():
    """Drift lowers the confidence multiplier and rolls back to a validated baseline."""
    from ore_cwm import DriftMonitor

    monitor = DriftMonitor(baseline_error=0.05, window=8, drift_ratio=2.0, min_samples=4)
    for _ in range(4):
        monitor.record(0.05)
    assert not monitor.status().drifted
    for _ in range(6):
        monitor.record(0.40)
    status = monitor.status()
    assert status.drifted
    assert status.confidence_multiplier < 1.0
    assert status.recommendation == "revert_to_validated_baseline"
    registry = ModelVersionRegistry()
    baseline = ModelVersion("v1", "ore_cwm.features.1", "digest-1", validated=True)
    registry.register(baseline)
    registry.deploy("v1")
    candidate = ModelVersion("v2", "ore_cwm.features.1", "digest-2")
    registry.register(candidate)
    registry.deploy("v2")
    assert registry.active().version_id == "v2"
    restored = registry.rollback_to_validated()
    assert restored.version_id == "v1"
    assert registry.active().version_id == "v1"
    ledger = ExecutionLedger(CWMConfig(min_support=2, calibration_window=8))
    for _ in range(4):
        seed_observation(ledger, "drifting_module", probability=0.9, accepted=True)
    for _ in range(20):
        ledger.drift.record(0.5)
    assert ledger.drift.status().drifted
    model = make_model()
    model.ledger = ledger
    model.transition_model.ledger = ledger
    manager, workspace = make_workspace()
    records = model.predict_outcomes(
        workspace.snapshot(), (make_operation("drifting_module"),))
    assert abs(records[0].predicted.success_probability - 0.5) < 0.35
    manager.close()


def test_prediction_store_preserves_missing_values_as_null(tmp_path):
    """The SQLite alignment store keeps missing measurements as NULL, never zero."""
    ledger = ExecutionLedger(CWMConfig(min_support=1))
    prediction = seeded_prediction(operation_id="store_prediction", module_id="store_module")
    ledger.register_prediction(prediction)
    outcome = ObservedOutcome(
        operation_id="store_prediction", module_id="store_module",
        workspace_id="seed_workspace", workspace_revision_before=0,
        workspace_revision_after=1, execution_status=ExecutionStatus.EXECUTED,
        dimensions=OutcomeDimensions.all_missing(),
        artifact_lifecycle=ArtifactLifecycle.ACCEPTED, artifacts_generated=1)
    report = ledger.align(prediction, outcome)
    store = SQLitePredictionStore(tmp_path / "cwm.sqlite3")
    store.record_alignment(prediction, outcome, report)
    rows = store.recent()
    assert len(rows) == 1
    row = rows[0]
    assert row["predicted_duration_ms"] == pytest.approx(300.0)
    assert row["observed_duration_ms"] is None
    assert row["observed_goal_progress"] is None
    assert row["execution_status"] == "executed"
    assert row["model_version"] == "seed"


def test_observer_events_preserve_function_and_class_identity(tmp_path):
    """Logger events link back to the annotated CWM methods and classes."""
    database_path = tmp_path / "observer.sqlite3"
    report = register_cwm_annotations(database_path, REPOSITORY_ROOT)
    assert report["scope"] == "ore_cwm"
    observer = SQLiteCWMObserver(database_path, REPOSITORY_ROOT).open()
    try:
        observer.record(
            "plan", {"prediction_id": "p1"},
            class_name="CognitiveWorldModel", module_name="ore_cwm.service")
        observer.record(
            "align", {"outcome_id": "o1"},
            class_name="ExecutionLedger", module_name="ore_cwm.learning")
    finally:
        observer.close()
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        events = connection.execute(
            "SELECT metadata_json FROM events ORDER BY sequence_no").fetchall()
        annotations = connection.execute(
            "SELECT qualified_name FROM annotations WHERE module_name = 'ore_cwm.service'"
        ).fetchall()
    assert len(events) == 2
    first = json.loads(events[0]["metadata_json"])
    assert first["annotated_method"] == "CognitiveWorldModel.plan"
    assert first["method_annotation_found"] is True
    assert first["class_annotation_found"] is True
    second = json.loads(events[1]["metadata_json"])
    assert second["annotated_method"] == "ExecutionLedger.align"
    assert second["class_annotation_found"] is True
    assert any("CognitiveWorldModel" in item["qualified_name"] for item in annotations)


def test_record_result_to_sqlite_persists_the_operation_trace(tmp_path):
    """Run reports persist their ordered operation trace through the shared logger."""
    ledger = ExecutionLedger(CWMConfig(min_support=2))
    for _ in range(4):
        seed_observation(ledger, "lexical_processor")
    model = make_model()
    model.ledger = ledger
    model.transition_model.ledger = ledger
    manager, workspace = make_workspace()
    coordinator = RecordingCoordinator(workspace)
    task = CognitiveTask(goal="understand_sentence",
                         required_capabilities=("language.lexical_processing",))
    run = model.run_receding_horizon(
        workspace, coordinator, task,
        candidates_provider=lambda _snapshot: (make_operation("lexical_processor"),),
        max_iterations=1, mode=AdvisoryMode.ADVISORY)
    summary = record_result_to_sqlite(
        run, tmp_path / "trace.sqlite3", repository_root=REPOSITORY_ROOT)
    assert summary["run_id"]
    with sqlite3.connect(tmp_path / "trace.sqlite3") as connection:
        connection.row_factory = sqlite3.Row
        events = connection.execute(
            "SELECT event_kind, metadata_json FROM events ORDER BY sequence_no").fetchall()
        run_row = connection.execute("SELECT status FROM runs").fetchone()
    assert run_row["status"] == "completed"
    assert len(events) == len(run.operation_trace) + len(run.steps)
    operation_events = [item for item in events if item["event_kind"] == "operation"]
    assert operation_events[0]["metadata_json"].count("CWM-01") > 0
    manager.close()


def test_benchmark_progress_per_resource_unit_beats_naive_ordering():
    """CWM advice improves accepted goal progress per resource unit over a naive default."""
    ledger = ExecutionLedger(CWMConfig(min_support=2))
    for _ in range(4):
        seed_observation(ledger, "cheap_weak", cpu_cost=1.0, progress=0.10)
    for _ in range(4):
        seed_observation(ledger, "costly_strong", cpu_cost=2.0, progress=0.90,
                         operation_class="composition")
    candidates = (
        make_operation("cheap_weak", capability="language.lexical_processing",
                       produced=("lexical_hypotheses",), cpu_cost=1.0),
        make_operation("costly_strong", capability="language.semantic_composition",
                       produced=("semantic_hypotheses",), cpu_cost=2.0),
    )
    task = CognitiveTask(goal="understand_sentence",
                         required_capabilities=("language.lexical_processing",))
    manager, workspace = make_workspace()
    advisory_coordinator = RecordingCoordinator(
        workspace, default_module="cheap_weak",
        modules={"cheap_weak": ("lexical_hypotheses",),
                 "costly_strong": ("semantic_hypotheses",)})
    model = make_model()
    model.ledger = ledger
    model.transition_model.ledger = ledger
    benchmark_constraints = HardConstraintSet(mandatory_verification_below_coverage=0.0)
    run = model.run_receding_horizon(
        workspace, advisory_coordinator, task,
        candidates_provider=lambda _snapshot: candidates,
        hard_constraints=benchmark_constraints,
        max_iterations=1, mode=AdvisoryMode.ADVISORY)
    executed_module = advisory_coordinator.executed[0]
    manager.close()

    manager, workspace = make_workspace()
    naive_coordinator = RecordingCoordinator(workspace, default_module="cheap_weak")
    naive_model = make_model()
    naive_run = naive_model.run_receding_horizon(
        workspace, naive_coordinator, task,
        candidates_provider=lambda _snapshot: candidates,
        max_iterations=1, mode=AdvisoryMode.SHADOW)
    naive_module = naive_coordinator.executed[0]
    manager.close()

    cwm_metric = run.goal_progress / 2.0
    naive_metric = naive_run.goal_progress / 1.0
    assert executed_module == "costly_strong", "CWM must prefer the better supported module"
    assert naive_module == "cheap_weak", "the naive default stays cheapest-first"
    assert cwm_metric > naive_metric
    critical_failures = sum(
        1 for outcome in run.outcomes
        if outcome.artifact_lifecycle.value in ("rejected", "invalidated"))
    naive_failures = sum(
        1 for outcome in naive_run.outcomes
        if outcome.artifact_lifecycle.value in ("rejected", "invalidated"))
    assert critical_failures == naive_failures == 0


def test_advise_resource_coordinator_produces_module_estimates():
    """Predictions translate into ModuleResourceEstimate evidence for ORE resource logic."""
    ledger = ExecutionLedger(CWMConfig(min_support=2))
    for _ in range(4):
        seed_observation(ledger, "lexical_processor", duration_ms=320.0, cpu_cost=0.8)
    model = make_model()
    model.ledger = ledger
    model.transition_model.ledger = ledger
    manager, workspace = make_workspace()
    candidates = (make_operation("lexical_processor"),)
    estimates = model.advise_resource_coordinator(workspace.snapshot(), candidates)
    estimate = estimates["lexical_processor"]
    assert isinstance(estimate, ModuleResourceEstimate)
    assert estimate.estimate_source == "cwm_prediction"
    assert estimate.expected_latency_ms == pytest.approx(320.0, rel=0.3)
    assert 0.0 <= estimate.estimated_success_probability <= 1.0
    manager.close()


def test_candidates_come_from_the_existing_module_registry():
    """Candidate operations originate from the registry, never from CWM invention."""
    catalog = ContractCatalog()
    catalog.register_model("ORE.TEST.INPUT", "1.0.0", str)
    catalog.register_model("ORE.TEST.ENCODED", "1.0.0", dict)
    manifest = ModuleManifest(
        ModuleIdentity("ore.test_encoder", "1.0.0", "Test encoder"),
        ("encode the external input",),
        (InputPort("source", "ORE.TEST.INPUT", ("1.0.0",),
                   expected_data_type="builtins.str"),),
        (OutputPort("encoded", "ORE.TEST.ENCODED", "1.0.0", "builtins.dict",
                    "encoded test representation"),),
        ("cognition.test_encoding",),
    )
    registry = ModuleRegistry(catalog)
    registry.register(manifest)
    registry.bind_backend(
        "ore.test_encoder",
        _Backend(manifest, "ORE.TEST.ENCODED", "source"))
    task = CognitiveTask(
        goal="encode the input",
        required_capabilities=("cognition.test_encoding",),
    )
    operations = CognitiveWorldModel.candidates_from_registry(registry, task)
    assert len(operations) == 1
    operation = operations[0]
    assert operation.module_id == "ore.test_encoder"
    assert operation.capability_id == "cognition.test_encoding"
    assert operation.operation_type is OperationType.MODULE_EXECUTION
    assert any("ORE.TEST.INPUT" in item.requirement for item in operation.preconditions)
    assert operation.output_schema == ("ORE.TEST.ENCODED@1.0.0",)
    registry.set_enabled("ore.test_encoder", False)
    operations = CognitiveWorldModel.candidates_from_registry(registry, task)
    assert operations == (), "disabled modules are not feasible candidates"


class _Backend:
    """Provide deterministic contract-producing behavior for the registry test."""

    def __init__(self, manifest, output_contract, source_port):
        """Store the manifest and contract names used by this backend."""
        self.manifest = manifest
        self.output_contract = output_contract
        self.source_port = source_port
        self.calls = 0

    def describe(self):
        """Return the exact manifest registered for this backend."""
        return self.manifest

    def execute(self, inputs, context):
        """Create one typed output whose lineage points to the exact input artifact."""
        self.calls += 1
        source = inputs[self.source_port]
        output = ContractArtifact(
            make_envelope(
                self.output_contract,
                context.run_id,
                self.manifest.identity.module_id,
                parent_artifact_ids=(source.envelope.fingerprint(),),
            ),
            {"derived_from": source.payload,
             "producer": self.manifest.identity.module_id},
        )
        return ModuleResult(
            self.manifest.identity.module_id,
            self.manifest.identity.module_version,
            ModuleStatus.SUCCESS,
            artifacts=(output,),
        )


def test_integration_with_real_cognitive_task_coordinator():
    """CWM advice steers the real coordinator through its documented channel only."""
    catalog = ContractCatalog()
    catalog.register_model("ORE.TEST.INPUT", "1.0.0", str)
    catalog.register_model("ORE.TEST.ENCODED", "1.0.0", dict)
    fast = ModuleManifest(
        ModuleIdentity("ore.test_encoder_fast", "1.0.0", "Fast encoder"),
        ("encode the external input",),
        (InputPort("source", "ORE.TEST.INPUT", ("1.0.0",),
                   expected_data_type="builtins.str"),),
        (OutputPort("encoded", "ORE.TEST.ENCODED", "1.0.0", "builtins.dict",
                    "fast encoded representation"),),
        ("cognition.test_encoding",),
    )
    slow = ModuleManifest(
        ModuleIdentity("ore.test_encoder_slow", "1.0.0", "Slow but reliable encoder"),
        ("encode the external input",),
        (InputPort("source", "ORE.TEST.INPUT", ("1.0.0",),
                   expected_data_type="builtins.str"),),
        (OutputPort("encoded", "ORE.TEST.ENCODED", "1.0.0", "builtins.dict",
                    "slow encoded representation"),),
        ("cognition.test_encoding",),
    )
    registry = ModuleRegistry(catalog)
    for manifest in (fast, slow):
        registry.register(manifest)
    fast_backend = _Backend(fast, "ORE.TEST.ENCODED", "source")
    slow_backend = _Backend(slow, "ORE.TEST.ENCODED", "source")
    registry.bind_backend("ore.test_encoder_fast", fast_backend)
    registry.bind_backend("ore.test_encoder_slow", slow_backend)
    context = RunContext(
        run_id="cwm-acceptance",
        mode=OperationMode.FUNCTIONAL,
        deterministic_seed=11,
        compute_budget=ComputeBudget(),
    )
    source = ContractArtifact(
        make_envelope("ORE.TEST.INPUT", context.run_id, "ore.test_input"),
        "The brain coordinates specialized modules.",
    )
    task = CognitiveTask(
        goal="encode the input",
        required_capabilities=("cognition.test_encoding",),
        input_artifacts={"source": source},
        metadata={"session": "integration"},
    )
    ledger = ExecutionLedger(CWMConfig(min_support=2))
    for _ in range(4):
        seed_observation(ledger, "ore.test_encoder_slow", duration_ms=250.0,
                          cpu_cost=0.5, progress=0.5)
    model = make_model()
    model.ledger = ledger
    model.transition_model.ledger = ledger
    candidates = CognitiveWorldModel.candidates_from_registry(registry, task)
    assert {item.module_id for item in candidates} == {
        "ore.test_encoder_fast", "ore.test_encoder_slow"}
    coordinator = CognitiveTaskCoordinator(
        registry, catalog, enable_logging=False, max_workers=1)
    spy = _CoordinatorSpy(coordinator)
    manager, workspace = make_workspace("The brain coordinates specialized modules.")
    run = model.run_receding_horizon(
        workspace, spy, task,
        candidates_provider=lambda _snapshot: candidates,
        max_iterations=1, mode=AdvisoryMode.ADVISORY)
    assert run.steps, "the coordinator must have dispatched the advised first action"
    assert spy.results, "the real coordinator must have returned a task result"
    executed_modules = set(spy.results[-1].module_results)
    for result in spy.results[-1].module_results.values():
        assert result.status is ModuleStatus.SUCCESS
    assert "ore.test_encoder_slow" in executed_modules
    assert fast_backend.calls == 0, "CWM advice must not execute the unadvised module"
    assert slow_backend.calls == 1
    assert run.steps[0].advisory_task_modified is True
    assert run.calibration["sample_count"] >= 1
    manager.close()


class _CoordinatorSpy:
    """Wrap one real coordinator and retain every dispatched task result."""

    def __init__(self, inner):
        """Store the wrapped coordinator and initialize the result log."""
        self.inner = inner
        self.results = []

    def coordinate(self, task, context=None):
        """Delegate to the real coordinator and retain its result for assertions."""
        result = self.inner.coordinate(task, context)
        self.results.append(result)
        return result


def test_contract_validation_rejects_invalid_records():
    """Typed contracts reject malformed identities and distributions."""
    prediction = seeded_prediction()
    with pytest.raises(ValueError):
        replace(prediction, feature_snapshot_hash="short")
    with pytest.raises(ValueError):
        replace(prediction, record_kind="accepted_finding")
    with pytest.raises(ValueError):
        PredictedOutcome(
            success_probability=0.5,
            outcome_class_probabilities={"success": 0.5, "failure": 0.2},
            expected_cost=ResourceEnvelope(expected_duration_ms=1.0),
            predicted_feature_delta={},
            goal_progress_estimate=0.2,
            information_gain=InformationGainEstimate(0.0, "proxy", "test"),
            risks=RiskAssessment(failure_risk=0.5),
            uncertainty=PredictionUncertainty((0.2, 0.8), sample_count=1),
        )
    with pytest.raises(ValueError):
        OutcomeDimensions(elapsed_ms=-5.0)
    with pytest.raises(ValueError):
        CWMBudget(horizon=9)
    with pytest.raises(ValueError):
        make_operation("module", depends_on=("",))


def test_pyproject_registers_the_cwm_package():
    """The package is registered in the repository's setuptools include list."""
    content = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"ore_cwm*"' in content
