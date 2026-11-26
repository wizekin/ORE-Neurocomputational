"""Acceptance tests: monitoring, lifecycle operations, and the 56-step pipeline."""
from __future__ import annotations

import sqlite3
import time

import numpy as np
import pytest

from ore_learning.adapters import MockRegressionAdapter
from ore_learning.audit import AuditStore
from ore_learning.config import LearningConfig
from ore_learning.contracts import (
    Capability,
    CapabilityRequest,
    CognitiveTaskRecord,
    LifecycleState,
    Modality,
    RiskClass,
)
from ore_learning.monitoring import (
    BehaviorMonitor,
    CanaryRouter,
    DriftMonitor,
    QuarantinePlaybook,
    ReferenceWindow,
    TaskQualityMonitor,
    execute_quarantine_and_rollback,
    quality_metrics,
)
from ore_learning.registry import ModelRegistry
from ore_learning.routing import ModelRouter, SelectionDecision

from learning_point14_helpers import make_manifest, make_request


# --------------------------------------------------------------- Steps 48-50


def test_calibration_and_quality_monitoring():
    """Step 48: Brier, ECE, reliability, and classification metrics."""
    rng = np.random.default_rng(2)
    probabilities = np.clip(0.5 + 0.3 * rng.normal(size=100), 0.01, 0.99)
    outcomes = (rng.random(100) < probabilities).astype(float)
    monitor = TaskQualityMonitor()
    report = monitor.calibration_report("m1", probabilities, outcomes,
                                        task_class="classification")
    assert 0.0 <= report.brier <= 1.0
    assert 0.0 <= report.ece <= 1.0
    assert report.sample_size == 100
    assert len(report.reliability["bin_counts"]) == 10
    assert report.comparable_across_models is False  # calibration context matters
    metrics = quality_metrics(probabilities, outcomes, threshold=0.5)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "false_positive_rate" in metrics and "false_negative_rate" in metrics
    monitor.record_execution("classification", latency_ms=42.0, memory_mb=1.5)
    summary = monitor.summarize("classification")
    assert summary["entries"] == 2 and summary["p50_latency_ms"] == 42.0


def test_drift_monitor_detects_a_controlled_synthetic_degradation():
    """Acceptance: drift monitor detects a controlled synthetic degradation."""
    config = LearningConfig(drift_min_sample_size=30)
    monitor = DriftMonitor(config)
    rng = np.random.default_rng(2)
    reference = ReferenceWindow("m1", rng.normal(0.0, 1.0, size=(200, 3)),
                                canary_quality=0.92)
    # 20 clean checks: no alert fires (false-alarm control)
    for _ in range(20):
        clean = monitor.check(reference, rng.normal(0.0, 1.0, size=(60, 3)), 0.91)
        assert clean is None
    # the controlled degradation: shifted inputs plus a canary quality drop
    alert = monitor.check(reference, rng.normal(2.0, 1.5, size=(60, 3)), 0.60)
    assert alert is not None
    assert alert.psi > config.drift_psi_alert
    assert alert.canary_drop > config.drift_canary_quality_alert
    assert alert.critical
    assert any("PSI" in reason for reason in alert.reasons)
    assert any("canary" in reason for reason in alert.reasons)
    # small windows below the minimum sample size never alert
    tiny = monitor.check(reference, rng.normal(5.0, 1.0, size=(10, 3)), 0.1)
    assert tiny is None


def test_behavior_monitor_separates_model_from_upstream_failures():
    """Step 50: model degradation vs upstream data or infrastructure failures."""
    monitor = BehaviorMonitor()
    for _ in range(6):
        monitor.record("schema_violation", "m1")
    verdict = monitor.verdict("m1", total_calls=100)
    assert verdict.regression_detected
    assert verdict.likely_cause == "model_degradation"
    assert "quarantine" in verdict.recommendation
    # correlated upstream errors shift the likely cause
    upstream = BehaviorMonitor()
    for _ in range(6):
        upstream.record("schema_violation", "m1")
        upstream.record("upstream_data_error", "pipeline")
    upstream_verdict = upstream.verdict("m1", total_calls=100)
    assert upstream_verdict.regression_detected
    assert upstream_verdict.likely_cause == "upstream_data"
    # safety events always regress
    safety = BehaviorMonitor()
    safety.record("safety_event", "m1")
    assert safety.verdict("m1", total_calls=1000).regression_detected


# --------------------------------------------------------------- Steps 51-52


def validated_registry() -> ModelRegistry:
    """A registry with an active model and a validated known-good fallback."""
    registry = ModelRegistry()
    registry.register(make_manifest("model.primary", quality=0.92))
    registry.register(make_manifest("model.knowngood", quality=0.85,
                                    dataset="ds-fallback",
                                    state=LifecycleState.VALIDATED))
    return registry


def test_drift_alert_triggers_quarantine_and_known_good_fallback():
    """Acceptance: drift alert triggers quarantine and known-good fallback."""
    registry = validated_registry()
    playbook = QuarantinePlaybook(registry)
    # a critical drift alert drives the deterministic Step 51 procedure
    recovery = execute_quarantine_and_rollback(
        registry, "model.primary", "1.0.0", "model.knowngood", "1.0.0",
        reason="critical drift alert (psi=1.9, canary drop 0.35)")
    assert recovery["quarantine"]["status"] == "quarantined"
    assert recovery["quarantine"]["new_assignments_stopped"]
    assert recovery["quarantine"]["logs_preserved"]
    assert recovery["rollback"]["status"] == "rolled-back"
    assert registry.require("model.primary", "1.0.0").lifecycle_state \
        is LifecycleState.QUARANTINED
    assert registry.require("model.knowngood", "1.0.0").lifecycle_state \
        is LifecycleState.ACTIVE
    # the quarantined model no longer receives new assignments
    selection = ModelRouter(registry).select(make_request())
    assert all(manifest.model_id != "model.primary"
               for manifest in selection.selected)
    # double quarantine is idempotent
    again = playbook.quarantine("model.primary", "1.0.0", reason="repeat")
    assert again["status"] == "already-quarantined"


def test_canary_deployment_limits_candidate_exposure_to_its_fraction():
    """Acceptance: canary deployment limits candidate exposure to its fraction."""
    canary = CanaryRouter(("cand", "1.0.0"), ("inc", "1.0.0"),
                          LearningConfig(), fraction=0.2)
    assignments = [canary.route(f"task-{i}", i) for i in range(100)]
    candidate_count = sum(1 for a in assignments if a.routed_to == "candidate")
    assert candidate_count == 20  # exactly the configured 20%
    summary = canary.exposure_summary()
    assert summary["candidate_share"] == pytest.approx(0.2)
    # a different fraction holds exactly as well
    quarter = CanaryRouter(("cand", "1.0.0"), ("inc", "1.0.0"),
                           LearningConfig(), fraction=0.25)
    routed = [quarter.route(f"t{i}", i) for i in range(200)]
    assert sum(1 for a in routed if a.routed_to == "candidate") == 50
    # determinism: the same ordinal always routes identically
    assert canary.route("t-x", 7).routed_to == canary.route("t-y", 7).routed_to


def test_candidate_rolls_back_when_regression_threshold_is_crossed():
    """Acceptance: candidate rolls back when regression threshold is crossed."""
    registry = ModelRegistry()
    registry.register(make_manifest("cand", quality=0.60))
    canary = CanaryRouter(("cand", "1.0.0"), ("inc", "1.0.0"),
                          LearningConfig(canary_regression_rollback_threshold=0.05),
                          fraction=0.2, registry=registry)
    for index in range(20):
        assignment = canary.route(f"t{index}", index)
        quality = 0.60 if assignment.routed_to == "candidate" else 0.90
        canary.record_outcome(assignment.task_id, assignment.routed_to, quality)
    gate = canary.gate()
    assert gate["decision"] == "rollback"
    assert "trails" in gate["reason"]
    assert canary.rolled_back
    # the registry quarantined the regressing candidate
    assert registry.require("cand", "1.0.0").lifecycle_state \
        is LifecycleState.QUARANTINED
    # after rollback every task routes to the incumbent
    assert canary.route("t-after", 999).routed_to == "incumbent"
    # an improving candidate expands instead
    improving = CanaryRouter(("good", "1.0.0"), ("inc", "1.0.0"),
                             LearningConfig(), fraction=0.2)
    for index in range(20):
        assignment = improving.route(f"t{index}", index)
        quality = 0.95 if assignment.routed_to == "candidate" else 0.90
        improving.record_outcome(assignment.task_id, assignment.routed_to, quality)
    assert improving.gate()["decision"] == "expand"


# --------------------------------------------------------------- pipeline (A-J)


def make_engine(registry: ModelRegistry | None = None, *,
                with_default_model: bool | None = None, **router_kwargs):
    """An engine wired with an in-memory audit store.

    A caller-supplied registry is trusted as-is; otherwise (or when
    ``with_default_model`` is explicitly True) a default specialist is registered.
    """
    from ore_learning.pipeline import LearnedModelEngine

    audit = AuditStore(":memory:")
    register_default = with_default_model if with_default_model is not None \
        else registry is None
    effective_registry = registry or ModelRegistry()
    if register_default:
        effective_registry.register(make_manifest("reg.main", quality=0.91,
                                                  latency_ms=60.0))
    router = ModelRouter(effective_registry, **router_kwargs)
    engine = LearnedModelEngine(
        LearningConfig(), registry=effective_registry, router=router,
        audit_hook=lambda operation, entry: audit.record(
            "ore_learning.pipeline", operation, "operation",
            task_id="task-01", status=entry.get("status", "completed"),
            metadata={"step": entry.get("step"),
                      "detail": str(entry.get("detail", ""))[:120]}))
    return engine, audit, effective_registry


def make_task() -> CognitiveTaskRecord:
    """A fully populated cognitive task record."""
    return CognitiveTaskRecord(
        task_id="task-01", goal="Predict the next measured level.",
        original_request="Predict the next level from the sensor feed.",
        source_references=("sensor://feed-a",), constraints=("latency < 200 ms",))


def test_engine_executes_the_full_56_step_pipeline():
    """End-to-end: phases A-J run, all 56 steps traced, proposals audited."""
    engine, audit, registry = make_engine()
    result = engine.run(make_task(), make_request(),
                        inputs=({"values": [0.2, -0.1, 0.5, 0.05]},),
                        adapter_for=MockRegressionAdapter())
    assert result.decision == "completed"
    assert [entry["step"] for entry in result.operation_trace] == list(range(1, 57))
    assert all(entry["status"] in ("completed", "skipped") for entry in
               result.operation_trace)
    phases = {entry["phase"] for entry in result.operation_trace}
    assert phases == set("ABCDEFGHIJ")
    # selection and inference happened through the typed path
    assert result.selection.decision is SelectionDecision.SELECTED
    assert result.selection.primary.model_id == "reg.main"
    assert result.inference.status == "completed"
    assert result.inference.validation.verified  # adapter output passed its schema
    assert result.proposals and result.proposals[0].status.value == "proposed"
    # raw proposals are refused at the authority boundary
    assert len(result.authority_refusals) == 1
    # PRM and validator separation holds inside the pipeline
    assert result.reasoning_trace is not None
    assert result.reward_record is not None
    # every step landed in the audit store without leaking payloads
    assert len(audit.query(task_id="task-01")) == 56
    assert audit.assert_no_forbidden_keys() == []


def test_engine_defers_gracefully_when_no_model_is_available():
    """An empty registry defers with a complete, honest trace."""
    engine, audit, _ = make_engine(ModelRegistry(), with_default_model=False)
    result = engine.run(make_task(), make_request(),
                        inputs=({"values": [0.1, 0.2, 0.3, 0.4]},),
                        adapter_for=MockRegressionAdapter())
    assert result.decision == "deferred"
    assert result.selection.decision is SelectionDecision.DEFERRED
    assert result.proposals == []
    # every one of the 56 steps still has a trace entry with an explicit reason
    assert len(result.operation_trace) == 56
    skipped = [entry for entry in result.operation_trace if entry["status"] == "skipped"]
    assert skipped and all(entry["detail"] for entry in skipped)


def test_engine_quarantines_and_rolls_back_on_critical_drift():
    """Phase J: a critical drift alert triggers quarantine and known-good fallback."""
    registry = ModelRegistry()
    registry.register(make_manifest("reg.main", quality=0.91))
    registry.register(make_manifest("reg.backup", quality=0.80,
                                    dataset="ds-backup",
                                    state=LifecycleState.VALIDATED))
    engine, audit, registry = make_engine(registry)
    rng = np.random.default_rng(2)
    reference = ReferenceWindow("reg.main", rng.normal(0.0, 1.0, size=(200, 3)),
                                canary_quality=0.92)
    result = engine.run(
        make_task(), make_request(),
        inputs=({"values": [0.2, -0.1, 0.5, 0.05]},),
        adapter_for=MockRegressionAdapter(),
        reference_window=reference,
        recent_window=rng.normal(2.0, 1.5, size=(60, 3)),
        recent_canary_quality=0.55)
    assert result.drift_alert is not None and result.drift_alert.critical
    assert result.diagnostics["quarantine"]["status"] == "quarantined"
    assert result.diagnostics["rollback"]["status"] == "rolled-back"
    assert registry.require("reg.main", "1.0.0").lifecycle_state \
        is LifecycleState.QUARANTINED
    assert registry.require("reg.backup", "1.0.0").lifecycle_state \
        is LifecycleState.ACTIVE
    # the quarantine/rollback procedure landed in the audit trail
    operations = {event["operation"] for event in audit.query(task_id="task-01")}
    assert {"quarantine", "rollback"} <= operations


def test_module_registration_with_the_ore_registry():
    """Integration: contracts, manifest, and backend register with ModuleRegistry."""
    from ore_contracts import ModuleRegistry, RunContext

    from ore_learning.registration import (
        ORE_LEARNING_CAPABILITIES,
        ORE_LEARNING_CONTRACTS,
        ORE_LEARNING_MODULE_ID,
        LearningModuleBackend,
        build_ore_learning_manifest,
        register_ore_learning_contracts,
        register_ore_learning_module,
    )

    from ore_contracts import ContractCatalog

    registry = ModuleRegistry(ContractCatalog())
    contracts = register_ore_learning_contracts(registry.contract_catalog)
    assert len(contracts) == len(ORE_LEARNING_CONTRACTS)
    backend = register_ore_learning_module(registry)
    manifest = registry.get_manifest(ORE_LEARNING_MODULE_ID)
    assert manifest is not None and manifest.identity.module_name.startswith("ORE")
    assert set(ORE_LEARNING_CAPABILITIES).issubset(set(manifest.capabilities))
    # the backend executes the pipeline through the module protocol
    engine, _, _ = make_engine()
    backend = LearningModuleBackend(build_ore_learning_manifest(), engine)
    from ore_contracts import ComputeBudget, OperationMode

    context = RunContext("learning-test", OperationMode.INFER, 0, ComputeBudget())
    module_result = backend.execute(
        {"task": make_task(), "capability_request": make_request()}, context)
    assert module_result.status.value == "success"
    assert module_result.processing_metadata["operation_steps"] == 56
    assert module_result.diagnostics["result"]["decision"] == "completed"
    # malformed inputs abstain rather than crash
    abstained = backend.execute({"task": "not-a-task"}, context)
    assert abstained.status.value == "abstained"


def test_annotations_sync_and_persist_to_sqlite(tmp_path):
    """ore_logger integration: annotation sync and per-step operation events."""
    from ore_learning.annotations import (
        LearningAnnotationRecorder,
        record_result_to_sqlite,
        register_annotations,
    )
    from ore_learning.pipeline import LearnedModelEngine

    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    root = tmp_path
    catalogue = register_annotations(root / "logger.sqlite3", repo_root)
    assert catalogue["scope"] == "ore_learning"
    assert catalogue["symbols_written"] >= 50  # the package is fully indexed

    engine, _, _ = make_engine()
    result = engine.run(make_task(), make_request(),
                        inputs=({"values": [0.2, -0.1, 0.5, 0.05]},),
                        adapter_for=MockRegressionAdapter())
    persisted = record_result_to_sqlite(result, root / "runs.sqlite3", repo_root)
    assert persisted["events_written"] == 56
    assert persisted["class_annotation_linked"] is True
    assert persisted["database_summary"]["runs"] >= 1

    with LearningAnnotationRecorder(root / "recorder.sqlite3", root) as recorder:
        event = recorder.record("execute", task_id="task-01", model="reg.main",
                                metadata={"step": 20, "detail": "selection"},
                                duration_ms=1.5)
        assert event["metadata"] == {"step": 20, "detail": "selection"}
        leaked = recorder.record("execute", task_id="task-01",
                                 metadata={"secret": "x", "raw_payload": {"a": 1},
                                           "status": "completed"})
        assert "secret" not in leaked["metadata"]
        assert "raw_payload" not in leaked["metadata"]
    with sqlite3.connect(root / "recorder.sqlite3") as connection:
        rows = connection.execute(
            "SELECT COUNT(*) FROM runs").fetchone()
        assert rows[0] >= 1


def test_workspace_publication_of_proposals():
    """Integration: proposals publish to the shared cognitive workspace."""
    pytest.importorskip("ore_workspace")
    from ore_workspace import SharedWorkspace

    from ore_learning import WorkspaceBridge, build_coordinator_notification

    engine, _, _ = make_engine()
    result = engine.run(make_task(), make_request(),
                        inputs=({"values": [0.2, -0.1, 0.5, 0.05]},),
                        adapter_for=MockRegressionAdapter())
    proposal = result.proposals[0]
    workspace = SharedWorkspace("analyze measurements", "analysis")
    record = WorkspaceBridge().publish(proposal, workspace)
    assert record["published"]
    assert record["decision"] in ("accepted", "merged")
    notification = build_coordinator_notification(proposal)
    assert notification["model"] == "reg.main@1.0.0"
    assert "never approves" in notification["note"]
    # publications are append-only: a second proposal never overwrites the first
    second = engine.run(
        CognitiveTaskRecord(task_id="task-02", goal="again",
                            original_request="again"),
        make_request(task_id="task-02"),
        inputs=({"values": [0.1, 0.0, 0.3, 0.2]},),
        adapter_for=MockRegressionAdapter())
    second_record = WorkspaceBridge().publish(second.proposals[0], workspace)
    assert second_record["finding_id"] != record["finding_id"]
