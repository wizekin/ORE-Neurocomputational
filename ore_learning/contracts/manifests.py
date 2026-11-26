"""Model manifests and schema descriptors for the RLMS registry (Point 14, Section 5.1).

A :class:`ModelManifest` is the immutable identity card of a learned model: who it is,
what it accepts and produces, where it came from, under which license, what it costs,
how well it measured, and where it sits in the lifecycle.  Registry admission and
eligibility filtering both consume manifests; nothing else about a model is trusted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence

__all__ = (
    "AdapterKind",
    "ArchitectureFamily",
    "BenchmarkResult",
    "CalibrationSummary",
    "LifecycleState",
    "Modality",
    "ModelManifest",
    "ResourceProfile",
    "SchemaDescriptor",
    "SecurityReview",
    "TrainingDataProvenance",
    "canonical_payload_digest",
    "default_schema_catalog",
    "validate_schema_against_catalog",
)


def _nonempty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _digest(name: str, value: str) -> str:
    text = _nonempty(name, value)
    if len(text) < 8 or any(character.isspace() for character in text):
        raise ValueError(f"{name} must be a compact digest string of at least 8 characters")
    return text


class LifecycleState(str, Enum):
    """Registry lifecycle states (Section 2.2)."""

    PROPOSED = "proposed"
    QUARANTINED = "quarantined"
    VALIDATED = "validated"
    CANARY = "canary"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"
    RETIRED = "retired"


class Modality(str, Enum):
    """Input/output modalities handled by the observation gateway (Step 18)."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SPATIAL = "spatial"
    STRUCTURED = "structured"


class ArchitectureFamily(str, Enum):
    """Architecture families named by the source point (Sections 3 and 14.7)."""

    LINEAR = "linear"
    MLP = "mlp"
    TRANSFORMER = "transformer"
    SPARSE_MOE = "sparse_moe"
    HYBRID_SSM = "hybrid_state_space"
    CNN = "cnn"
    GAN = "gan"
    DIFFUSION = "diffusion"
    QUANTIZED_EDGE = "quantized_edge"


class AdapterKind(str, Enum):
    """The adapter category that isolates a model behind ORE contracts."""

    MOCK = "mock"
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    SPATIAL = "spatial"
    FRAMEWORK = "framework"


@dataclass(frozen=True, slots=True)
class SchemaDescriptor:
    """A typed, versioned input/output schema (Step 2.2 core contracts)."""

    schema_type: str
    schema_version: str
    fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the descriptor shape."""
        object.__setattr__(self, "schema_type", _nonempty("schema_type", self.schema_type))
        object.__setattr__(self, "schema_version",
                           _nonempty("schema_version", self.schema_version))
        if not all(isinstance(name, str) and name for name in self.fields):
            raise ValueError("schema fields must be non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"schema_type": self.schema_type,
                "schema_version": self.schema_version,
                "fields": list(self.fields)}


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    """Declared resource requirements used for CROS admission (Step 11)."""

    cpu_cores: float = 1.0
    memory_mb: float = 256.0
    accelerator_mb: float = 0.0
    storage_mb: float = 64.0
    max_concurrency: int = 1
    estimated_latency_ms: float = 50.0

    def __post_init__(self) -> None:
        """Resource requirements must be positive."""
        for name in ("cpu_cores", "memory_mb", "storage_mb", "estimated_latency_ms"):
            value = getattr(self, name)
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
        if float(self.accelerator_mb) < 0.0 or int(self.max_concurrency) < 1:
            raise ValueError("accelerator_mb must be non-negative and max_concurrency >= 1")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"cpu_cores": self.cpu_cores, "memory_mb": self.memory_mb,
                "accelerator_mb": self.accelerator_mb, "storage_mb": self.storage_mb,
                "max_concurrency": self.max_concurrency,
                "estimated_latency_ms": self.estimated_latency_ms}


@dataclass(frozen=True, slots=True)
class TrainingDataProvenance:
    """Training-data lineage recorded at registration (Section 2.2)."""

    dataset_id: str
    license: str
    collection_period: str
    filtering_rules: tuple[str, ...] = ()
    retention_policy: str = "default"
    access_policy: str = "restricted"

    def __post_init__(self) -> None:
        """Identity, license, and period are mandatory provenance fields."""
        object.__setattr__(self, "dataset_id", _nonempty("dataset_id", self.dataset_id))
        object.__setattr__(self, "license", _nonempty("license", self.license))
        object.__setattr__(self, "collection_period",
                           _nonempty("collection_period", self.collection_period))

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"dataset_id": self.dataset_id, "license": self.license,
                "collection_period": self.collection_period,
                "filtering_rules": list(self.filtering_rules),
                "retention_policy": self.retention_policy,
                "access_policy": self.access_policy}


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One measured benchmark entry on a named evaluation suite (Step 8 evidence)."""

    suite: str
    metric: str
    value: float
    held_out: bool = True
    lower_is_better: bool = False

    def __post_init__(self) -> None:
        """Values must be finite."""
        if not math.isfinite(float(self.value)):
            raise ValueError("benchmark value must be finite")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"suite": self.suite, "metric": self.metric, "value": self.value,
                "held_out": self.held_out, "lower_is_better": self.lower_is_better}


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    """Calibration evidence carried with the manifest (Step 48 comparability)."""

    brier_score: float
    expected_calibration_error: float
    sample_size: int
    method: str = "reliability-bins"

    def __post_init__(self) -> None:
        """Scores must be finite and sample sizes positive."""
        if not (math.isfinite(self.brier_score) and math.isfinite(
                self.expected_calibration_error)):
            raise ValueError("calibration scores must be finite")
        if int(self.sample_size) <= 0:
            raise ValueError("sample_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"brier_score": self.brier_score,
                "expected_calibration_error": self.expected_calibration_error,
                "sample_size": self.sample_size, "method": self.method}


@dataclass(frozen=True, slots=True)
class SecurityReview:
    """Security posture recorded at registration (Step 6 provenance completeness)."""

    reviewed: bool
    verdict: str
    reviewer: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        """Verdict vocabulary is fixed."""
        if self.verdict not in ("approved", "pending", "rejected", "not_required"):
            raise ValueError("verdict must be approved/pending/rejected/not_required")
        if self.reviewed and self.verdict == "pending":
            raise ValueError("a reviewed manifest cannot carry a pending verdict")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"reviewed": self.reviewed, "verdict": self.verdict,
                "reviewer": self.reviewer, "notes": self.notes}


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Immutable registry entry for one model version (Section 2.2, Section 5.1)."""

    model_id: str
    semantic_name: str
    version: str
    artifact_digest: str
    architecture_family: ArchitectureFamily
    framework: str
    input_schema: SchemaDescriptor
    output_schema: SchemaDescriptor
    schema_version: str
    supported_tasks: tuple[str, ...]
    modalities: tuple[Modality, ...]
    languages: tuple[str, ...] = ("en",)
    training_data_provenance: TrainingDataProvenance | None = None
    adaptation_history: tuple[str, ...] = ()
    license: str = "research-only"
    resource_profile: ResourceProfile = field(default_factory=ResourceProfile)
    benchmark_results: tuple[BenchmarkResult, ...] = ()
    calibration_report: CalibrationSummary | None = None
    known_limitations: tuple[str, ...] = ()
    security_review: SecurityReview = SecurityReview(reviewed=False, verdict="pending")
    execution_permissions: tuple[str, ...] = ("inference:local",)
    lifecycle_state: LifecycleState = LifecycleState.PROPOSED
    adapter_kind: AdapterKind = AdapterKind.MOCK

    def __post_init__(self) -> None:
        """Structural validation performed on every construction (acceptance 1-4)."""
        object.__setattr__(self, "model_id", _nonempty("model_id", self.model_id))
        object.__setattr__(self, "semantic_name",
                           _nonempty("semantic_name", self.semantic_name))
        object.__setattr__(self, "version", _nonempty("version", self.version))
        object.__setattr__(self, "framework", _nonempty("framework", self.framework))
        object.__setattr__(self, "schema_version",
                           _nonempty("schema_version", self.schema_version))
        object.__setattr__(self, "artifact_digest",
                           _digest("artifact_digest", self.artifact_digest))
        if not self.supported_tasks:
            raise ValueError("supported_tasks must list at least one capability")
        if not self.modalities:
            raise ValueError("modalities must list at least one modality")
        if self.training_data_provenance is None:
            raise ValueError("training_data_provenance is required for registry admission")
        if self.license not in ("mit", "apache-2.0", "cc-by-4.0", "proprietary",
                                "research-only", "internal"):
            raise ValueError(f"unsupported license {self.license!r}")

    @property
    def identity(self) -> str:
        """Stable ``model_id@version`` identity string."""
        return f"{self.model_id}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view of the manifest."""
        return {
            "model_id": self.model_id,
            "semantic_name": self.semantic_name,
            "version": self.version,
            "artifact_digest": self.artifact_digest,
            "architecture_family": self.architecture_family.value,
            "framework": self.framework,
            "input_schema": self.input_schema.to_dict(),
            "output_schema": self.output_schema.to_dict(),
            "schema_version": self.schema_version,
            "supported_tasks": list(self.supported_tasks),
            "modalities": [item.value for item in self.modalities],
            "languages": list(self.languages),
            "training_data_provenance": (self.training_data_provenance.to_dict()
                                         if self.training_data_provenance else None),
            "adaptation_history": list(self.adaptation_history),
            "license": self.license,
            "resource_profile": self.resource_profile.to_dict(),
            "benchmark_results": [item.to_dict() for item in self.benchmark_results],
            "calibration_report": (self.calibration_report.to_dict()
                                   if self.calibration_report else None),
            "known_limitations": list(self.known_limitations),
            "security_review": self.security_review.to_dict(),
            "execution_permissions": list(self.execution_permissions),
            "lifecycle_state": self.lifecycle_state.value,
            "adapter_kind": self.adapter_kind.value,
        }

    def with_state(self, state: LifecycleState) -> "ModelManifest":
        """Return a copy carrying a new lifecycle state (transitions happen elsewhere)."""
        return ModelManifest(
            model_id=self.model_id, semantic_name=self.semantic_name, version=self.version,
            artifact_digest=self.artifact_digest,
            architecture_family=self.architecture_family, framework=self.framework,
            input_schema=self.input_schema, output_schema=self.output_schema,
            schema_version=self.schema_version, supported_tasks=self.supported_tasks,
            modalities=self.modalities, languages=self.languages,
            training_data_provenance=self.training_data_provenance,
            adaptation_history=self.adaptation_history, license=self.license,
            resource_profile=self.resource_profile,
            benchmark_results=self.benchmark_results,
            calibration_report=self.calibration_report,
            known_limitations=self.known_limitations,
            security_review=self.security_review,
            execution_permissions=self.execution_permissions,
            lifecycle_state=state, adapter_kind=self.adapter_kind)


def canonical_payload_digest(payload: Mapping[str, Any]) -> str:
    """Digest an arbitrary typed payload for provenance records."""
    import hashlib
    import json as _json

    blob = _json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def default_schema_catalog() -> dict[tuple[str, str], tuple[str, ...]]:
    """Known schema descriptors with their field contracts.

    Registration rejects manifests whose declared schemas are absent from this
    catalog (acceptance: "Registry rejects incompatible input/output schemas").
    """
    catalog: dict[tuple[str, str], tuple[str, ...]] = {}
    descriptors = (
        ("text", "tokens", ("text", "language")),
        ("text", "sentiment", ("label", "score")),
        ("text", "generation", ("candidate_text", "purpose")),
        ("image", "features", ("region", "descriptor")),
        ("image", "caption", ("region", "caption", "confidence")),
        ("audio", "transcript", ("interval", "transcript", "confidence")),
        ("audio", "features", ("interval", "descriptor")),
        ("video", "events", ("interval", "event_label", "confidence")),
        ("spatial", "coordinates", ("frame", "points")),
        ("structured", "label_distribution", ("labels", "probabilities")),
        ("structured", "regression", ("values", "intervals")),
        ("structured", "ranking", ("items", "scores")),
        ("structured", "code", ("language", "code_text")),
    )
    for schema_type, schema_version, fields in descriptors:
        catalog[(schema_type, schema_version)] = fields
    return catalog


def validate_schema_against_catalog(descriptor: SchemaDescriptor,
                                    catalog: Mapping[tuple[str, str], tuple[str, ...]] | None = None
                                    ) -> tuple[bool, str]:
    """Check that a schema descriptor exists in the catalog and declares its fields."""
    known = catalog if catalog is not None else default_schema_catalog()
    key = (descriptor.schema_type, descriptor.schema_version)
    if key not in known:
        return False, f"unknown schema {descriptor.schema_type}@{descriptor.schema_version}"
    expected = known[key]
    if descriptor.fields and tuple(descriptor.fields) != tuple(expected):
        return False, (f"schema fields {list(descriptor.fields)} do not match the catalog "
                       f"contract {list(expected)}")
    return True, "schema known to the catalog"
