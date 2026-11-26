"""Shared fixtures for the Point 14 RLMS acceptance tests (specification Section 7)."""
from __future__ import annotations

import hashlib

from ore_learning.contracts import (
    ArchitectureFamily,
    BenchmarkResult,
    CalibrationSummary,
    Capability,
    CapabilityRequest,
    LifecycleState,
    Modality,
    ModelManifest,
    ResourceProfile,
    RiskClass,
    SchemaDescriptor,
    SecurityReview,
    TrainingDataProvenance,
)

REGRESSION_SCHEMA = SchemaDescriptor("structured", "regression",
                                     ("values", "intervals"))
SENTIMENT_SCHEMA = SchemaDescriptor("text", "sentiment", ("label", "score"))


def digest_for(name: str) -> str:
    """A stable 16-character artifact digest for a model name."""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


def make_manifest(model_id: str, *, version: str = "1.0.0",
                  quality: float = 0.90,
                  latency_ms: float = 60.0,
                  license_id: str = "mit",
                  dataset: str = "ds-open-1",
                  state: LifecycleState = LifecycleState.ACTIVE,
                  security_verdict: str = "approved",
                  calibration: CalibrationSummary | None = None,
                  schema: SchemaDescriptor | None = None,
                  supported_tasks: tuple[str, ...] = (Capability.REGRESSION.value,),
                  modalities: tuple[Modality, ...] = (Modality.STRUCTURED,),
                  cpu_cores: float = 1.0,
                  memory_mb: float = 128.0,
                  ) -> ModelManifest:
    """Build a fully populated manifest for tests."""
    effective_schema = schema or REGRESSION_SCHEMA
    return ModelManifest(
        model_id=model_id,
        semantic_name=f"{model_id} test specialist",
        version=version,
        artifact_digest=digest_for(f"{model_id}:{version}"),
        architecture_family=ArchitectureFamily.MLP,
        framework="numpy-reference",
        input_schema=effective_schema,
        output_schema=effective_schema,
        schema_version=effective_schema.schema_version,
        supported_tasks=supported_tasks,
        modalities=modalities,
        training_data_provenance=TrainingDataProvenance(
            dataset, license_id, "2023-Q3..2024-Q2",
            filtering_rules=("drop-nulls",)),
        license=license_id,
        resource_profile=ResourceProfile(cpu_cores=cpu_cores, memory_mb=memory_mb,
                                         estimated_latency_ms=latency_ms),
        benchmark_results=(BenchmarkResult(supported_tasks[0], "accuracy", quality),),
        calibration_report=calibration or CalibrationSummary(0.05, 0.04, 400),
        known_limitations=("reference implementation on toy data",),
        security_review=SecurityReview(reviewed=security_verdict != "pending",
                                       verdict=security_verdict,
                                       reviewer="test-reviewer"),
        lifecycle_state=state,
    )


def make_request(task_id: str = "task-001", *,
                 capabilities: tuple[Capability, ...] = (Capability.REGRESSION,),
                 modality: Modality = Modality.STRUCTURED,
                 schema: SchemaDescriptor | None = None,
                 risk_class: RiskClass = RiskClass.LOW,
                 accuracy_target: float = 0.8,
                 latency_target_ms: float = 200.0) -> CapabilityRequest:
    """Build a fully populated capability request for tests."""
    return CapabilityRequest(
        task_id=task_id, capabilities=capabilities, modality=modality,
        output_contract=schema or REGRESSION_SCHEMA,
        accuracy_target=accuracy_target, latency_target_ms=latency_target_ms,
        risk_class=risk_class)
