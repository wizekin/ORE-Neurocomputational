"""Reference demo of the Point 14 Replaceable Learned-Model Subsystem.

Run with ``python -m ore_learning``.  The demo registers two specialists plus a
deterministic fallback, routes a capability request, executes the selected model
behind its adapter, validates the process, and prints the bounded result summary.
"""
from __future__ import annotations

import numpy as np

from .adapters import MockRegressionAdapter
from .audit import AuditStore
from .config import LearningConfig
from .contracts import (
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
from .monitoring import DriftMonitor, ReferenceWindow
from .pipeline import LearnedModelEngine
from .registry import ModelRegistry
from .routing import ModelRouter

SCHEMA = SchemaDescriptor("structured", "regression", ("values", "intervals"))


def make_manifest(model_id: str, quality: float, latency_ms: float, *,
                  license_id: str = "mit", dataset: str = "ds-open-1",
                  state: LifecycleState = LifecycleState.ACTIVE) -> ModelManifest:
    """Build a fully populated demo manifest."""
    import hashlib

    digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:16]
    return ModelManifest(
        model_id=model_id, semantic_name=f"{model_id} reference specialist",
        version="1.0.0", artifact_digest=digest,
        architecture_family=ArchitectureFamily.MLP, framework="numpy-reference",
        input_schema=SCHEMA, output_schema=SCHEMA, schema_version="regression",
        supported_tasks=(Capability.REGRESSION.value,),
        modalities=(Modality.STRUCTURED,),
        training_data_provenance=TrainingDataProvenance(
            dataset, license_id, "2023-Q3..2024-Q2",
            filtering_rules=("drop-nulls", "unit-check")),
        license=license_id,
        resource_profile=ResourceProfile(cpu_cores=1.0, memory_mb=128.0,
                                         estimated_latency_ms=latency_ms),
        benchmark_results=(BenchmarkResult("regression", "accuracy", quality),),
        calibration_report=CalibrationSummary(0.06, 0.05, 400),
        known_limitations=("reference implementation on toy data",),
        security_review=SecurityReview(reviewed=True, verdict="approved",
                                       reviewer="demo-reviewer"),
        lifecycle_state=state)


def main() -> None:
    """Run the reference demonstration."""
    config = LearningConfig()
    audit = AuditStore(":memory:")
    registry = ModelRegistry(event_listener=lambda event: audit.record(
        "ore_learning.registry", event.event_type.value, "lifecycle",
        model_id=event.model_id, model_version=event.model_version,
        metadata={"from_state": event.from_state.value if event.from_state else None,
                  "to_state": event.to_state.value, "reason": event.reason}))
    registry.register(make_manifest("reg.large", 0.93, 80.0))
    registry.register(make_manifest("reg.small", 0.88, 30.0, dataset="ds-open-2"))
    registry.register(make_manifest("reg.fallback", 0.80, 5.0,
                                    state=LifecycleState.VALIDATED))

    router = ModelRouter(registry, config=config,
                         fallback_model_id="reg.fallback")
    engine = LearnedModelEngine(
        config, registry=registry, router=router,
        audit_hook=lambda operation, entry: audit.record(
            "ore_learning.pipeline", operation, "operation",
            task_id="demo-task", status=entry.get("status", "completed"),
            metadata={"step": entry.get("step"), "detail": entry.get("detail", "")[:120]}))

    task = CognitiveTaskRecord(
        task_id="demo-task", goal="Predict the next measured level.",
        original_request="Please predict the next measured level from the sensor feed.",
        source_references=("sensor://feed-a",), constraints=("latency < 200 ms",))
    request = CapabilityRequest(
        task_id="demo-task", capabilities=(Capability.REGRESSION,),
        modality=Modality.STRUCTURED, output_contract=SCHEMA,
        accuracy_target=0.85, latency_target_ms=200.0, risk_class=RiskClass.MEDIUM)

    inputs = [{"values": [0.2, -0.1, 0.5, 0.05], "timestamp": 1000.0}]
    result = engine.run(task, request, inputs=inputs,
                        adapter_for=MockRegressionAdapter(),
                        reference_window=ReferenceWindow(
                            "reg.large", np.random.default_rng(1).normal(size=(200, 3)),
                            canary_quality=0.92),
                        recent_window=np.random.default_rng(2).normal(size=(60, 3)),
                        recent_canary_quality=0.91)

    summary = result.to_dict()
    print("== Point 14 RLMS reference demo ==")
    print("decision:", summary["decision"])
    print("selected:", summary["selected_models"],
          "| selection:", summary["selection_decision"])
    print("inference:", summary["inference_status"],
          "| adapter output validated:", result.diagnostics.get(
              "adapter_output_validated"))
    print("proposals:", summary["proposal_ids"])
    print("trace verified:", summary["trace_verified"],
          "| reward:", summary["reward"])
    print("drift alert:", bool(summary["drift_alert"]))
    print("authority refusals (expected for unvalidated proposals):",
          len(summary["authority_refusals"]))
    print("steps recorded:", summary["steps_executed"])
    print("audit events:", audit.summary()["events"],
          "| leak check:", audit.assert_no_forbidden_keys() or "clean")
    print("limitations:")
    for limitation in summary["limitations"]:
        print("  -", limitation)


if __name__ == "__main__":
    main()
