"""Point 14 demonstration: the ORE Replaceable Learned-Model Subsystem.

This example exercises the full Point 14 lifecycle on one cognitive task: registry
admission under provenance and license policy, capability-based routing with hard
eligibility gates and CROS reservations, bounded inference behind a typed adapter
with an enforced timeout, evidence-bearing proposals refused at the USG authority
boundary, deterministic process validation with PRM screening kept separate,
drift-triggered quarantine and rollback to a known-good fallback, canary routing
with exact exposure control, and audit events persisted to SQLite without raw
payloads.

Run from the repository root (consistent with the other point examples, which expect
the repository packages to be importable, e.g. after ``pip install -e .``):

    python -m examples.point14_rlms
"""
from __future__ import annotations

import time

import numpy as np

from ore_learning import (
    AuthorityBoundary,
    CanaryRouter,
    DriftMonitor,
    LearnedModelEngine,
    LearningConfig,
    ReferenceWindow,
    WorkspaceBridge,
    build_coordinator_notification,
)
from ore_learning.adapters import ModalityPayload, ObservationGateway
from ore_learning.audit import AuditStore
from ore_learning.contracts import (
    ArchitectureFamily,
    BenchmarkResult,
    CalibrationSummary,
    Capability,
    CapabilityRequest,
    CognitiveTaskRecord,
    LifecycleState,
    Modality,
    ModelManifest,
    ResourceProfile,
    RiskClass,
    SchemaDescriptor,
    SecurityReview,
    TrainingDataProvenance,
)
from ore_learning.registry import ModelRegistry
from ore_learning.routing import ModelRouter

SCHEMA = SchemaDescriptor("structured", "regression", ("values", "intervals"))


def make_manifest(model_id: str, *, quality: float, latency_ms: float,
                  dataset: str = "ds-open-1",
                  state: LifecycleState = LifecycleState.ACTIVE,
                  cpu_cores: float = 1.0) -> ModelManifest:
    """Build a fully populated manifest (identity, provenance, license, benchmarks)."""
    import hashlib

    return ModelManifest(
        model_id=model_id,
        semantic_name=f"{model_id} reference specialist",
        version="1.0.0",
        artifact_digest=hashlib.sha256(model_id.encode()).hexdigest()[:16],
        architecture_family=ArchitectureFamily.MLP,
        framework="numpy-reference",
        input_schema=SCHEMA,
        output_schema=SCHEMA,
        schema_version="regression",
        supported_tasks=(Capability.REGRESSION.value,),
        modalities=(Modality.STRUCTURED,),
        training_data_provenance=TrainingDataProvenance(
            dataset, "mit", "2023-Q3..2024-Q2",
            filtering_rules=("drop-nulls", "unit-check")),
        license="mit",
        resource_profile=ResourceProfile(cpu_cores=cpu_cores, memory_mb=128.0,
                                         estimated_latency_ms=latency_ms),
        benchmark_results=(BenchmarkResult("regression", "accuracy", quality),),
        calibration_report=CalibrationSummary(0.06, 0.05, 400),
        known_limitations=("reference implementation on toy data",),
        security_review=SecurityReview(reviewed=True, verdict="approved",
                                       reviewer="example-reviewer"),
        lifecycle_state=state,
    )


def main() -> None:
    """Run the Point 14 demonstration end to end."""
    config = LearningConfig()
    audit = AuditStore(":memory:")

    # -- registry: lifecycle events flow into the audit store as they happen
    registry = ModelRegistry(event_listener=lambda event: audit.record(
        "ore_learning.registry", event.event_type.value, "lifecycle",
        model_id=event.model_id, model_version=event.model_version,
        metadata={"from_state": event.from_state.value if event.from_state else None,
                  "to_state": event.to_state.value, "reason": event.reason}))
    registry.register(make_manifest("reg.large", quality=0.93, latency_ms=80.0))
    registry.register(make_manifest("reg.small", quality=0.88, latency_ms=30.0,
                                    dataset="ds-open-2"))
    registry.register(make_manifest("reg.fallback", quality=0.80, latency_ms=5.0,
                                    dataset="ds-fallback", cpu_cores=0.25,
                                    state=LifecycleState.VALIDATED))

    # -- the cognitive task and its capability request (capabilities, not model names)
    task = CognitiveTaskRecord(
        task_id="example-task-001",
        goal="Predict the next measured level from the sensor feed.",
        original_request="Please predict the next level and explain the drivers.",
        source_references=("sensor://feed-a", "calibration://v3"),
        constraints=("latency < 200 ms", "no production state changes"))
    request = CapabilityRequest(
        task_id=task.task_id, capabilities=(Capability.REGRESSION,),
        modality=Modality.STRUCTURED, output_contract=SCHEMA,
        accuracy_target=0.85, latency_target_ms=200.0, risk_class=RiskClass.MEDIUM)

    # -- engine: phases A-J with a full 56-step operation trace
    router = ModelRouter(registry, config=config,
                         fallback_model_id="reg.fallback")
    engine = LearnedModelEngine(
        config, registry=registry, router=router,
        audit_hook=lambda operation, entry: audit.record(
            "ore_learning.pipeline", operation, "operation",
            task_id=task.task_id, status=entry.get("status", "completed"),
            metadata={"step": entry.get("step"),
                      "detail": str(entry.get("detail", ""))[:120]}))

    print("== Point 14 RLMS demonstration ==")
    print("registry:", registry.describe())

    # -- multimodal gateway: an image observation arrives alongside the structured input
    gateway = ObservationGateway(config)
    image_payload = ModalityPayload(
        Modality.IMAGE, "camera/frame-77", time.time(),
        np.random.default_rng(3).normal(size=(16, 16)), encoding="float64")
    normalized = gateway.validate(image_payload)
    print("gateway:", normalized.source_ref, normalized.shape, normalized.checks)

    # -- run the pipeline; a recent distribution shift is supplied for the drift check
    rng = np.random.default_rng(11)
    result = engine.run(
        task, request,
        inputs=({"values": [0.2, -0.1, 0.5, 0.05], "timestamp": 1000.0},),
        modality_payloads=[image_payload],
        reference_window=ReferenceWindow(
            "reg.large", rng.normal(0.0, 1.0, size=(200, 3)), canary_quality=0.92),
        recent_window=rng.normal(0.0, 1.0, size=(60, 3)),  # no drift this time
        recent_canary_quality=0.91)

    summary = result.to_dict()
    print("\n-- pipeline result --")
    print("decision:", summary["decision"])
    print("selected:", summary["selected_models"],
          f"({summary['selection_decision']})")
    print("inference:", summary["inference_status"],
          "| adapter output validated:",
          result.inference.validation.verified if result.inference else None)
    print("proposals:", summary["proposal_ids"],
          "| trace verified:", summary["trace_verified"],
          "| reward:", summary["reward"])
    print("drift alert:", bool(summary["drift_alert"]))
    print("steps traced:", summary["steps_executed"])

    # -- authority boundary: the raw proposal is refused until validated + approved
    boundary = AuthorityBoundary()
    refusal = boundary.request_canonical_update(result.proposals[0])
    print("\n-- authority boundary --")
    print("unvalidated proposal:", refusal["decision"], "-", refusal["reason"])

    # -- workspace publication (append-only findings, permission handshake)
    workspace = None
    try:
        from ore_workspace import SharedWorkspace

        workspace = SharedWorkspace("predict the next level", "prediction")
    except Exception:  # noqa: BLE001 - publication is optional in this demo
        pass
    publication = WorkspaceBridge().publish(result.proposals[0], workspace)
    print("workspace publication:", publication["decision"])
    print("coordinator notification:",
          build_coordinator_notification(result.proposals[0])["model"])

    # -- a critical drift alert quarantines the model and rolls back to known-good
    print("\n-- lifecycle recovery drill --")
    drifted = engine.run(
        CognitiveTaskRecord("drill-001", "degradation drill", "degradation drill"),
        CapabilityRequest("drill-001", (Capability.REGRESSION,), Modality.STRUCTURED,
                          SCHEMA, risk_class=RiskClass.LOW),
        inputs=({"values": [0.1, 0.0, 0.3, 0.2]},),
        reference_window=ReferenceWindow(
            "reg.large", rng.normal(0.0, 1.0, size=(200, 3)), canary_quality=0.92),
        recent_window=rng.normal(2.5, 1.5, size=(60, 3)),
        recent_canary_quality=0.55)
    print("drift critical:", drifted.drift_alert.critical if drifted.drift_alert else None,
          "| quarantine:", drifted.diagnostics.get("quarantine", {}).get("status"),
          "| rollback:", drifted.diagnostics.get("rollback", {}).get("status"))
    print("registry after recovery:", registry.describe()["states"])

    # -- canary routing: exposure is bounded to exactly the configured fraction
    print("\n-- canary routing --")
    canary = CanaryRouter(("cand", "1.0.0"), ("inc", "1.0.0"), config, fraction=0.2)
    assignments = [canary.route(f"task-{i}", i) for i in range(50)]
    candidate_share = sum(1 for a in assignments if a.routed_to == "candidate") / 50
    print("candidate exposure:", candidate_share, "(configured 0.2)")

    # -- audit: every step and lifecycle event persisted, nothing raw leaked
    print("\n-- audit --")
    print(audit.summary())
    print("forbidden-key scan:", audit.assert_no_forbidden_keys() or "clean")

    print("\nlimitations:")
    for limitation in summary["limitations"]:
        print("  -", limitation)


if __name__ == "__main__":
    main()
