"""Typed public request, hardware, plan, and result contracts for the HACOL layer.

Implements the "ore_compute: public request/result contracts" package boundary of
"ORE Integrated Algorithm -- Point 21: Hardware-Aware Numerical Computing and
Quantum Algorithms" (Most Intelligent Systems 01-24, Point 21, sections 21.1-21.8).

All contracts are typed, versioned, serializable, and independently testable.
Kernels never mutate coordinator internals and never bypass CROS.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

MODULE_VERSION = "1.0.0"
ENVELOPE_VERSION = "1.0"
SPEC_SOURCE = (
    "ORE Integrated Algorithm -- Point 21: Hardware-Aware Numerical Computing and "
    "Quantum Algorithms (Most Intelligent Systems 01-24.txt, sections 21.1-21.8)"
)
REPOSITORY_SOURCE = "https://github.com/wizekin/ORE-Neurocomputational"

#: Dtype declarations accepted by HACOL. "bfloat16" is emulated on CPU via
#: round-to-nearest-even mantissa truncation of float32 (see ore_hacol.precision).
SUPPORTED_DTYPES = ("float16", "bfloat16", "float32", "float64")

#: Layout declarations currently understood by the reference CPU backends.
SUPPORTED_LAYOUTS = ("row-major", "column-major")


class RequestValidationError(ValueError):
    """Raised when an optimization request lacks identity or declares invalid data."""


class OperationCancelled(RuntimeError):
    """Raised inside the pipeline when a cancellation token fires between steps."""


class OperationKind(str, Enum):
    """Mathematical primitive classification required by integrated step 2."""

    ATTENTION = "attention"
    MATRIX_MULTIPLY = "matrix_multiply"
    SIMILARITY = "similarity"
    FACTORIZATION = "factorization"
    LEAST_SQUARES = "least_squares"
    TRANSFORM = "transform"
    REDUCTION = "reduction"
    OPTIMIZATION = "optimization"
    OTHER = "other"

    @classmethod
    def coerce(cls, value: "OperationKind | str") -> "OperationKind":
        """Accept an enum member or one of the canonical lowercase names."""
        if isinstance(value, cls):
            return value
        name = str(value).strip().lower()
        for member in cls:
            if member.value == name:
                return member
        raise RequestValidationError(f"unknown operation_kind: {value!r}")


class ConsequenceLevel(str, Enum):
    """Consequence classification for integrated step 10."""

    EXPLORATORY = "exploratory"
    ADVISORY = "advisory"
    DECISION_SUPPORT = "decision_support"
    SCIENTIFIC = "scientific"
    HIGH_ASSURANCE = "high_assurance"


class ValidationStatus(str, Enum):
    """Lifecycle status recorded in the versioned result envelope."""

    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    DEFERRED = "deferred"
    FALLBACK = "fallback"
    REFERENCE_ONLY = "reference_only"
    CANCELLED = "cancelled"


def _positive_int(name: str, value: int) -> int:
    number = int(value)
    if number <= 0:
        raise RequestValidationError(f"{name} must be a positive integer")
    return number


def _positive_float(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RequestValidationError(f"{name} must be finite and positive")
    return number


def _non_negative_float(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise RequestValidationError(f"{name} must be finite and non-negative")
    return number


def canonical_config_hash(parameters: Mapping[str, Any]) -> str:
    """Hash a plan configuration into a stable reproducibility key (spec section 8)."""
    payload = json.dumps(parameters, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class CorrectnessContract:
    """Correctness requirements for one operation (integrated step 9).

    Tolerances are per-operation allowed error budgets; high-consequence use
    must tighten them and require independent verification (step 10).
    """

    exactness_required: bool = False
    abs_tolerance: float = 1e-9
    rel_tolerance: float = 1e-6
    determinism_required: bool = True
    feasibility_required: bool = False
    conservation_required: bool = False
    monotonicity_required: bool = False
    all_masked_row_policy: str = "zeros"
    consequence_level: ConsequenceLevel = ConsequenceLevel.ADVISORY
    downstream_sensitivity: float = 0.5

    def __post_init__(self) -> None:
        _non_negative_float("abs_tolerance", self.abs_tolerance)
        _non_negative_float("rel_tolerance", self.rel_tolerance)
        _non_negative_float("downstream_sensitivity", self.downstream_sensitivity)
        if self.all_masked_row_policy not in ("zeros", "uniform", "reject"):
            raise RequestValidationError(
                "all_masked_row_policy must be 'zeros', 'uniform', or 'reject'"
            )
        if not isinstance(self.consequence_level, ConsequenceLevel):
            raise RequestValidationError("consequence_level must be a ConsequenceLevel")

    def tighten(self, factor: float = 0.1) -> "CorrectnessContract":
        """Return a tightened copy used for high-consequence outputs (step 10)."""
        return CorrectnessContract(
            exactness_required=self.exactness_required,
            abs_tolerance=self.abs_tolerance * factor,
            rel_tolerance=self.rel_tolerance * factor,
            determinism_required=True,
            feasibility_required=self.feasibility_required,
            conservation_required=self.conservation_required,
            monotonicity_required=self.monotonicity_required,
            all_masked_row_policy=self.all_masked_row_policy,
            consequence_level=self.consequence_level,
            downstream_sensitivity=self.downstream_sensitivity,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "consequence_level": self.consequence_level.value,
        }


@dataclass(frozen=True)
class SensitivityProfile:
    """Numerical sensitivity evidence collected by integrated step 12."""

    condition_estimate: float | None = None
    dynamic_range: float | None = None
    accumulation_depth: int = 1
    near_singular: bool = False
    cancellation_risk: float = 0.0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"notes": list(self.notes)}


@dataclass(frozen=True)
class OptimizationRequest:
    """One optimization request received by integrated step 1.

    Requires operation identity, input schema/dimensions, correctness contract,
    budgets, precision requirements, and whether approximation is allowed.
    """

    task_id: str
    operation_id: str
    operation_kind: OperationKind | str
    input_shape: Sequence[int]
    dtype: str = "float64"
    layout: str = "row-major"
    sparsity: float = 0.0
    correctness_contract: CorrectnessContract | None = None
    approximation_allowed: bool = False
    latency_budget_seconds: float | None = None
    memory_budget_bytes: float | None = None
    resource_budget: float | None = None
    calling_module: str = ""
    calling_function: str = ""
    call_frequency: float = 1.0
    description: str = ""

    def __post_init__(self) -> None:
        for name, value in (("task_id", self.task_id), ("operation_id", self.operation_id)):
            if not str(value).strip():
                raise RequestValidationError(f"{name} must be non-empty")
        shape = tuple(int(dim) for dim in self.input_shape)
        if not shape or any(dim <= 0 for dim in shape):
            raise RequestValidationError(
                f"input_shape must contain positive dimensions: {self.input_shape}"
            )
        if self.dtype not in SUPPORTED_DTYPES:
            raise RequestValidationError(
                f"dtype {self.dtype!r} not in supported set {SUPPORTED_DTYPES}"
            )
        if self.layout not in SUPPORTED_LAYOUTS:
            raise RequestValidationError(f"layout {self.layout!r} not in {SUPPORTED_LAYOUTS}")
        _non_negative_float("sparsity", self.sparsity)
        _positive_float("call_frequency", self.call_frequency)
        for name, value in (
            ("latency_budget_seconds", self.latency_budget_seconds),
            ("memory_budget_bytes", self.memory_budget_bytes),
            ("resource_budget", self.resource_budget),
        ):
            if value is not None:
                _positive_float(name, value)
        if self.correctness_contract is not None and not isinstance(
            self.correctness_contract, CorrectnessContract
        ):
            raise RequestValidationError("correctness_contract must be a CorrectnessContract")

    @property
    def kind(self) -> OperationKind:
        """Validated operation classification (integrated step 2)."""
        return OperationKind.coerce(self.operation_kind)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.input_shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "operation_id": self.operation_id,
            "operation_kind": self.kind.value,
            "input_shape": list(self.shape),
            "dtype": self.dtype,
            "layout": self.layout,
            "sparsity": self.sparsity,
            "approximation_allowed": self.approximation_allowed,
            "latency_budget_seconds": self.latency_budget_seconds,
            "memory_budget_bytes": self.memory_budget_bytes,
            "calling_module": self.calling_module,
            "calling_function": self.calling_function,
            "call_frequency": self.call_frequency,
            "description": self.description,
            "correctness_contract": (
                self.correctness_contract.to_dict() if self.correctness_contract else None
            ),
        }


@dataclass(frozen=True)
class HardwareProfile:
    """Discovered hardware capabilities (integrated step 13)."""

    cpu_model: str = "unknown"
    cpu_features: tuple[str, ...] = ()
    core_count: int = 1
    memory_capacity_bytes: int = 0
    memory_available_bytes: int = 0
    bandwidth_bytes_per_second: float = 0.0
    compute_rate_flops_per_second: float = 0.0
    accelerator_types: tuple[str, ...] = ()
    supported_dtypes: tuple[str, ...] = SUPPORTED_DTYPES
    runtime_versions: Mapping[str, str] = field(default_factory=dict)
    utilization: Mapping[str, float] = field(default_factory=dict)
    discovery_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_model": self.cpu_model,
            "cpu_features": list(self.cpu_features),
            "core_count": self.core_count,
            "memory_capacity_bytes": self.memory_capacity_bytes,
            "memory_available_bytes": self.memory_available_bytes,
            "bandwidth_bytes_per_second": self.bandwidth_bytes_per_second,
            "compute_rate_flops_per_second": self.compute_rate_flops_per_second,
            "accelerator_types": list(self.accelerator_types),
            "supported_dtypes": list(self.supported_dtypes),
            "runtime_versions": dict(self.runtime_versions),
            "utilization": dict(self.utilization),
            "discovery_notes": list(self.discovery_notes),
        }

    def hardware_identity(self) -> str:
        """Stable identity recorded in result envelopes (acceptance: capability identity)."""
        payload = {
            "cpu_model": self.cpu_model,
            "core_count": self.core_count,
            "cpu_features": sorted(self.cpu_features),
            "accelerator_types": list(self.accelerator_types),
            "runtime_versions": dict(self.runtime_versions),
        }
        return "hw:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]


@dataclass(frozen=True)
class ResourceAsk:
    """Resource demand HACOL submits to CROS for admission (integrated step 14)."""

    cpu_share: float = 1.0
    memory_bytes: float = 0.0
    accelerator_slots: int = 0
    deadline_seconds: float | None = None
    queue_priority: int = 5

    def __post_init__(self) -> None:
        _positive_float("cpu_share", self.cpu_share)
        _non_negative_float("memory_bytes", self.memory_bytes)
        if int(self.accelerator_slots) < 0:
            raise RequestValidationError("accelerator_slots must be non-negative")


@dataclass(frozen=True)
class AdmissionDecision:
    """CROS response. HACOL recommends; CROS retains global allocation authority."""

    granted: bool
    reason: str
    granted_memory_bytes: float = 0.0
    granted_cpu_share: float = 0.0
    queue_position: int = 0
    deadline_slack_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CostEstimate:
    """Analytical cost model for one algorithm/backend pair (integrated step 15)."""

    flops: float = 0.0
    bytes_moved: float = 0.0
    temporary_bytes: float = 0.0
    estimated_seconds: float = 0.0
    setup_seconds: float = 0.0
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OptimizationPlan:
    """Selected execution plan with fallback and rationale (integrated step 16)."""

    algorithm: str
    backend: str
    precision: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    estimated_cost: CostEstimate | None = None
    expected_error: float | None = None
    fallback_algorithm: str = "reference_implementation"
    rationale: str = ""
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not self.algorithm.strip():
            raise RequestValidationError("plan algorithm must be non-empty")
        if self.precision not in SUPPORTED_DTYPES:
            raise RequestValidationError(f"plan precision {self.precision!r} unsupported")
        object.__setattr__(self, "parameters", dict(self.parameters))
        if not self.config_hash:
            object.__setattr__(
                self,
                "config_hash",
                canonical_config_hash(
                    {
                        "algorithm": self.algorithm,
                        "backend": self.backend,
                        "precision": self.precision,
                        "parameters": dict(self.parameters),
                    }
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "backend": self.backend,
            "precision": self.precision,
            "parameters": dict(self.parameters),
            "estimated_cost": self.estimated_cost.to_dict() if self.estimated_cost else None,
            "expected_error": self.expected_error,
            "fallback_algorithm": self.fallback_algorithm,
            "rationale": self.rationale,
            "config_hash": self.config_hash,
        }


@dataclass(frozen=True)
class ErrorReport:
    """Numerical comparison of optimized and reference outputs (Phase G)."""

    max_abs: float = 0.0
    mean_abs: float = 0.0
    max_rel: float = 0.0
    mean_rel: float = 0.0
    within_tolerance: bool = False
    tolerance_basis: str = "none"
    sample_count: int = 0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"notes": list(self.notes)}


@dataclass(frozen=True)
class ProvenanceRecord:
    """Provenance attached to every published envelope (Phase G)."""

    spec_source: str = SPEC_SOURCE
    repository: str = REPOSITORY_SOURCE
    module_version: str = MODULE_VERSION
    envelope_version: str = ENVELOPE_VERSION
    seed: int | None = None
    config_hash: str = ""
    recorded_at_utc: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationResult:
    """Measured, validated outcome envelope for one optimization request (Phase G)."""

    task_id: str
    operation_id: str
    operation_kind: str
    algorithm_identity: str
    backend: str
    precision: str
    hardware_identity: str
    output_reference: Any
    reference_output: Any
    elapsed_seconds: float
    kernel_seconds: float
    orchestration_seconds: float
    peak_memory_bytes: int
    transfer_cost_bytes: float
    measured_error: ErrorReport | None
    validation_status: ValidationStatus
    plan: OptimizationPlan | None
    provenance: ProvenanceRecord | None
    step_trace: tuple[Mapping[str, Any], ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_envelope(self) -> dict[str, Any]:
        """Versioned result envelope: metadata only, never raw tensors (Phase G)."""
        return {
            "envelope_version": ENVELOPE_VERSION,
            "task_id": self.task_id,
            "operation_id": self.operation_id,
            "operation_kind": self.operation_kind,
            "algorithm_identity": self.algorithm_identity,
            "backend": self.backend,
            "hardware_identity": self.hardware_identity,
            "dtype": self.precision,
            "configuration": self.plan.to_dict() if self.plan else None,
            "approximation_metadata": self.metrics.get("approximation_metadata"),
            "error": self.measured_error.to_dict() if self.measured_error else None,
            "validation_status": self.validation_status.value,
            "timing": {
                "elapsed_seconds": self.elapsed_seconds,
                "kernel_seconds": self.kernel_seconds,
                "orchestration_seconds": self.orchestration_seconds,
                "latency_p50_seconds": self.metrics.get("latency_p50_seconds"),
                "latency_p95_seconds": self.metrics.get("latency_p95_seconds"),
                "latency_p99_seconds": self.metrics.get("latency_p99_seconds"),
            },
            "memory": {
                "peak_memory_bytes": self.peak_memory_bytes,
                "transfer_cost_bytes": self.transfer_cost_bytes,
            },
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "step_count": len(self.step_trace),
            "notes": list(self.metrics.get("notes", [])),
        }


@dataclass(frozen=True)
class ContractAssessment:
    """Phase B admission evidence: correctness, consequence, permission, sensitivity."""

    contract: CorrectnessContract
    consequence_level: ConsequenceLevel
    approximation_permitted: bool
    sensitivity: SensitivityProfile
    rationale: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract.to_dict(),
            "consequence_level": self.consequence_level.value,
            "approximation_permitted": self.approximation_permitted,
            "sensitivity": self.sensitivity.to_dict(),
            "rationale": list(self.rationale),
        }


@dataclass(frozen=True)
class ValueEstimate:
    """Optimization value estimate from integrated step 8."""

    call_frequency: float
    per_call_savings_seconds: float
    setup_cost_seconds: float
    maintenance_cost_seconds: float
    net_value_seconds: float
    recommend_optimization: bool
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
