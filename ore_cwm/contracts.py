"""Typed contracts for the ORE Cognitive World Model prediction and advisory subsystem.

Every record produced here is a provisional planning object. Predictions describe imagined
consequences of candidate cognitive operations; they are never committed as observed
workspace findings and never replace the authoritative shared workspace state.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

CWM_FEATURE_SCHEMA_VERSION = "ore_cwm.features.1"
CWM_PREDICTOR_VERSION = "ore_cwm.predictor.1"


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _new_id(prefix: str) -> str:
    """Create a readable, collision-resistant identifier with a CWM domain prefix."""
    return f"{prefix}_{uuid4().hex}"


class _MissingMeasurement:
    """Singleton marker for measurements that are unavailable rather than numerically zero."""

    _instance: _MissingMeasurement | None = None

    def __new__(cls) -> _MissingMeasurement:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False

    def __copy__(self) -> _MissingMeasurement:
        return MISSING

    def __deepcopy__(self, memo: dict[int, Any]) -> _MissingMeasurement:
        return MISSING


MISSING = _MissingMeasurement()
MissingMeasurement = _MissingMeasurement


def is_missing(value: Any) -> bool:
    """Return whether a measurement is explicitly unavailable."""
    return value is MISSING


def measurement(value: float | _MissingMeasurement) -> float | None:
    """Convert a measurement to a plain float or None when it is unavailable."""
    return None if value is MISSING else float(value)


def _require_number(name: str, value: Any, *, minimum: float | None = None,
                   maximum: float | None = None) -> float:
    """Validate one finite numeric field and return it as a float."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def _require_measurement(name: str, value: Any, *, minimum: float | None = None,
                         maximum: float | None = None) -> None:
    """Validate a possibly-missing measurement without treating missing as zero."""
    if value is MISSING:
        return
    _require_number(name, value, minimum=minimum, maximum=maximum)


class CWMError(Exception):
    """Base class for recoverable Cognitive World Model failures."""


class StaleRevisionError(CWMError):
    """The snapshot revision no longer matches the authoritative workspace revision."""


class InvalidPredictionSchemaError(CWMError):
    """A candidate or prediction does not satisfy the declared CWM schema contracts."""


class PlanningBudgetExceededError(CWMError):
    """Planning exceeded its declared runtime, memory, or branching budget."""


class UnsupportedCandidateError(CWMError):
    """A candidate operation cannot be predicted or executed under current policies."""


class FeatureProvenance(str, Enum):
    """Label whether one feature was observed, deterministically derived, or predicted."""

    OBSERVED = "observed"
    DERIVED = "derived"
    PREDICTED = "predicted"


class OutcomeClass(str, Enum):
    """Classify the predicted effect of one operation on goal progress."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    NO_PROGRESS = "no_progress"
    RECOVERABLE_FAILURE = "recoverable_failure"
    HARD_FAILURE = "hard_failure"


class SafetyClass(str, Enum):
    """Classify the execution-safety posture of one candidate operation."""

    SAFE = "safe"
    CONSTRAINED = "constrained"


class RetryPolicy(str, Enum):
    """Declare retry and idempotency semantics attached to one operation."""

    IDEMPOTENT = "idempotent"
    AT_MOST_ONCE = "at_most_once"
    RETRY_ONCE = "retry_once"


class OperationType(str, Enum):
    """Distinguish module dispatches from planner-internal control actions."""

    MODULE_EXECUTION = "module_execution"
    VERIFY = "verify"
    STOP = "stop"


class ExecutionStatus(str, Enum):
    """Label how one execution ended so contaminated events stay outside learning."""

    EXECUTED = "executed"
    ABORTED = "aborted"
    CANCELLED = "cancelled"
    DUPLICATED = "duplicated"
    CORRUPTED = "corrupted"
    POLICY_VIOLATING = "policy_violating"


class ArtifactLifecycle(str, Enum):
    """Distinguish generated, accepted, deferred, rejected, and invalidated artifacts."""

    GENERATED = "generated"
    ACCEPTED = "accepted"
    DEFERRED = "deferred"
    REJECTED = "rejected"
    INVALIDATED = "invalidated"


class FallbackTrigger(str, Enum):
    """Name the deterministic fallback triggers required by the CWM safeguards."""

    TIMEOUT = "timeout"
    STALE_REVISION = "stale_revision"
    INVALID_SCHEMA = "invalid_schema"
    MISSING_MEASUREMENT = "missing_measurement"
    MODEL_ERROR = "model_error"
    LOW_CONFIDENCE = "low_confidence"
    DRIFT = "drift"
    NO_SAFE_CANDIDATE = "no_safe_candidate"


class AdvisoryMode(str, Enum):
    """Separate shadow evaluation from controlled advisory influence."""

    SHADOW = "shadow"
    ADVISORY = "advisory"


@dataclass(frozen=True, slots=True)
class FeatureValue:
    """Hold one compact cognitive-state feature with explicit provenance labeling."""

    name: str
    value: float | _MissingMeasurement
    provenance: FeatureProvenance
    definition: str = ""

    def __post_init__(self) -> None:
        """Reject unnamed features and non-finite values while keeping missing values distinct."""
        if not self.name.strip():
            raise ValueError("FeatureValue.name must be non-empty")
        if self.provenance is FeatureProvenance.PREDICTED and self.value is MISSING:
            raise ValueError("predicted features must carry a numeric value")
        _require_measurement(
            "FeatureValue.value", self.value if self.value is not MISSING else MISSING)
        object.__setattr__(self, "name", self.name.strip())

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible feature representation."""
        return {
            "name": self.name,
            "value": measurement(self.value),
            "provenance": self.provenance.value,
            "definition": self.definition,
        }


@dataclass(frozen=True, slots=True)
class CognitiveStateFeatures:
    """Versioned predictive projection of one authoritative workspace snapshot.

    The feature vector is a compact projection used for prediction only; it never replaces
    the workspace state representation and every value keeps its provenance label.
    """

    schema_version: str
    workspace_id: str
    workspace_revision: int
    goal: str
    feature_snapshot_hash: str
    values: Mapping[str, FeatureValue]
    created_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        """Validate identities and detach the feature mapping from caller ownership."""
        for name in ("schema_version", "workspace_id", "goal", "feature_snapshot_hash"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.workspace_revision < 0:
            raise ValueError("workspace_revision cannot be negative")
        values = dict(self.values)
        if not values:
            raise ValueError("CognitiveStateFeatures requires at least one feature value")
        if len({item.name for item in values.values()}) != len(values):
            raise ValueError("feature names must be unique")
        object.__setattr__(self, "values", values)

    def value(self, name: str) -> float | _MissingMeasurement:
        """Return one feature value that stays missing rather than reading as zero."""
        item = self.values.get(name)
        return MISSING if item is None else item.value

    def names(self) -> tuple[str, ...]:
        """Return feature names in deterministic sorted order."""
        return tuple(sorted(self.values))

    def as_vector(self, defaults: Mapping[str, float]) -> tuple[float, ...]:
        """Project features onto a numeric vector using explicit conservative defaults.

        Missing features use the supplied per-name defaults so an unavailable measurement
        can never be silently substituted by an implicit zero.
        """
        vector = []
        for name in self.names():
            current = self.values[name].value
            if current is MISSING:
                if name not in defaults:
                    raise ValueError(f"missing feature {name!r} requires an explicit default")
                vector.append(float(defaults[name]))
            else:
                vector.append(float(current))
        return tuple(vector)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible feature snapshot."""
        return {
            "schema_version": self.schema_version,
            "workspace_id": self.workspace_id,
            "workspace_revision": self.workspace_revision,
            "goal": self.goal,
            "feature_snapshot_hash": self.feature_snapshot_hash,
            "values": {name: item.to_dict() for name, item in sorted(self.values.items())},
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class ResourceEnvelope:
    """Estimated execution cost envelope with explicit missing-measurement handling."""

    expected_duration_ms: float | _MissingMeasurement = MISSING
    expected_cpu_cost: float | _MissingMeasurement = MISSING
    expected_memory_mb: float | _MissingMeasurement = MISSING
    expected_io_cost: float | _MissingMeasurement = MISSING
    sample_count: int = 0

    def __post_init__(self) -> None:
        """Validate every available cost component without coercing missing values."""
        for name in ("expected_duration_ms", "expected_cpu_cost",
                     "expected_memory_mb", "expected_io_cost"):
            _require_measurement(name, getattr(self, name), minimum=0.0)
        if self.sample_count < 0:
            raise ValueError("sample_count cannot be negative")

    @classmethod
    def missing(cls) -> ResourceEnvelope:
        """Return a fully unavailable envelope for candidates without estimates."""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible cost representation."""
        return {
            "expected_duration_ms": measurement(self.expected_duration_ms),
            "expected_cpu_cost": measurement(self.expected_cpu_cost),
            "expected_memory_mb": measurement(self.expected_memory_mb),
            "expected_io_cost": measurement(self.expected_io_cost),
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True, slots=True)
class OperationPrecondition:
    """Declare one checkable precondition attached to a candidate operation."""

    kind: str
    requirement: str

    def __post_init__(self) -> None:
        """Reject empty precondition kinds and requirements."""
        if not self.kind.strip() or not self.requirement.strip():
            raise ValueError("precondition kind and requirement must be non-empty")


@dataclass(frozen=True, slots=True)
class CognitiveOperation:
    """Describe one executable candidate cognitive operation offered by ORE itself.

    CWM never invents module identifiers: module execution candidates originate from the
    module registry and coordinator contracts; verify and stop are planner-internal control
    actions that are never dispatched to module backends.
    """

    operation_type: OperationType
    capability_id: str
    module_id: str
    operation_id: str = field(default_factory=lambda: _new_id("cwm_operation"))
    input_artifact_refs: tuple[str, ...] = ()
    preconditions: tuple[OperationPrecondition, ...] = ()
    output_schema: tuple[str, ...] = ()
    produced_fields: tuple[str, ...] = ()
    resource_envelope: ResourceEnvelope = field(default_factory=ResourceEnvelope.missing)
    timeout_ms: float | _MissingMeasurement = MISSING
    retry_policy: RetryPolicy = RetryPolicy.IDEMPOTENT
    safety: SafetyClass = SafetyClass.SAFE
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the operation contract while normalizing sequence fields."""
        if not self.capability_id.strip() or not self.module_id.strip():
            raise ValueError("capability_id and module_id must be non-empty")
        if not self.operation_id.strip():
            raise ValueError("operation_id must be non-empty")
        object.__setattr__(self, "input_artifact_refs", tuple(self.input_artifact_refs))
        object.__setattr__(self, "preconditions", tuple(self.preconditions))
        object.__setattr__(self, "output_schema", tuple(self.output_schema))
        object.__setattr__(self, "produced_fields", tuple(self.produced_fields))
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        _require_measurement("timeout_ms", self.timeout_ms, minimum=0.0)
        for reference in self.depends_on:
            if not reference.strip():
                raise ValueError("dependency references must be non-empty")
        for field_name in self.produced_fields:
            if not field_name.strip():
                raise ValueError("produced field names must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible operation contract."""
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type.value,
            "capability_id": self.capability_id,
            "module_id": self.module_id,
            "input_artifact_refs": list(self.input_artifact_refs),
            "preconditions": [
                {"kind": item.kind, "requirement": item.requirement} for item in self.preconditions
            ],
            "output_schema": list(self.output_schema),
            "produced_fields": list(self.produced_fields),
            "resource_envelope": self.resource_envelope.to_dict(),
            "timeout_ms": measurement(self.timeout_ms),
            "retry_policy": self.retry_policy.value,
            "safety": self.safety.value,
            "depends_on": list(self.depends_on),
        }


@dataclass(frozen=True, slots=True)
class ResourceState:
    """Capture current resource measurements used to condition predictions."""

    cpu_load: float | _MissingMeasurement = MISSING
    memory_used_mb: float | _MissingMeasurement = MISSING
    memory_budget_mb: float | _MissingMeasurement = MISSING
    io_rate: float | _MissingMeasurement = MISSING
    timestamp: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        """Validate available resource components without coercing missing values."""
        for name in ("cpu_load", "memory_used_mb", "memory_budget_mb", "io_rate"):
            _require_measurement(name, getattr(self, name), minimum=0.0)

    def resource_pressure(self) -> float | _MissingMeasurement:
        """Return memory utilization when measurable, else CPU load, else missing."""
        if self.memory_used_mb is not MISSING and self.memory_budget_mb is not MISSING:
            budget = float(self.memory_budget_mb)
            if budget <= 0.0:
                return MISSING
            return min(1.0, float(self.memory_used_mb) / budget)
        if self.cpu_load is not MISSING:
            return min(1.0, float(self.cpu_load))
        return MISSING

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible resource measurement snapshot."""
        return {
            "cpu_load": measurement(self.cpu_load),
            "memory_used_mb": measurement(self.memory_used_mb),
            "memory_budget_mb": measurement(self.memory_budget_mb),
            "io_rate": measurement(self.io_rate),
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True, slots=True)
class OperationContext:
    """Encode the conditioning context under which one candidate is predicted."""

    operation_type: str
    module_id: str
    goal_class: str
    input_size: int | _MissingMeasurement = MISSING
    dependency_depth: int = 0
    resource_load: float | _MissingMeasurement = MISSING

    def __post_init__(self) -> None:
        """Validate context components without coercing missing measurements."""
        for name in ("operation_type", "module_id", "goal_class"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.input_size is not MISSING and int(self.input_size) < 0:
            raise ValueError("input_size cannot be negative")
        if self.dependency_depth < 0:
            raise ValueError("dependency_depth cannot be negative")
        _require_measurement("resource_load", self.resource_load, minimum=0.0, maximum=1.0)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible context representation."""
        return {
            "operation_type": self.operation_type,
            "module_id": self.module_id,
            "goal_class": self.goal_class,
            "input_size": None if self.input_size is MISSING else int(self.input_size),
            "dependency_depth": self.dependency_depth,
            "resource_load": measurement(self.resource_load),
        }


@dataclass(frozen=True, slots=True)
class OutcomeDimensions:
    """Measure one execution along goal, evidence, uncertainty, validity, and cost axes.

    Units and windows: durations in milliseconds, memory in megabytes, CPU and I/O cost in
    abstract cost units, and every metric is measured over exactly one operation execution.
    Unavailable metrics remain MISSING and are never silently converted to zero.
    """

    goal_progress_delta: float | _MissingMeasurement = MISSING
    accepted_evidence_delta: float | _MissingMeasurement = MISSING
    uncertainty_reduction: float | _MissingMeasurement = MISSING
    output_validity: float | _MissingMeasurement = MISSING
    elapsed_ms: float | _MissingMeasurement = MISSING
    cpu_cost: float | _MissingMeasurement = MISSING
    memory_mb: float | _MissingMeasurement = MISSING
    io_cost: float | _MissingMeasurement = MISSING
    retries: int | _MissingMeasurement = MISSING
    failure_class: str | None = None
    side_effects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate every available measurement while preserving explicit missingness."""
        for name in ("accepted_evidence_delta", "uncertainty_reduction", "elapsed_ms",
                     "cpu_cost", "memory_mb", "io_cost"):
            _require_measurement(name, getattr(self, name), minimum=0.0)
        _require_measurement("goal_progress_delta", self.goal_progress_delta)
        _require_measurement("output_validity", self.output_validity, minimum=0.0, maximum=1.0)
        if self.retries is not MISSING and int(self.retries) < 0:
            raise ValueError("retries cannot be negative")
        object.__setattr__(self, "side_effects", tuple(self.side_effects))

    @classmethod
    def all_missing(cls) -> OutcomeDimensions:
        """Return dimensions with every metric explicitly unavailable."""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible measurement record."""
        return {
            "goal_progress_delta": measurement(self.goal_progress_delta),
            "accepted_evidence_delta": measurement(self.accepted_evidence_delta),
            "uncertainty_reduction": measurement(self.uncertainty_reduction),
            "output_validity": measurement(self.output_validity),
            "elapsed_ms": measurement(self.elapsed_ms),
            "cpu_cost": measurement(self.cpu_cost),
            "memory_mb": measurement(self.memory_mb),
            "io_cost": measurement(self.io_cost),
            "retries": None if self.retries is MISSING else int(self.retries),
            "failure_class": self.failure_class,
            "side_effects": list(self.side_effects),
        }


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """Separate failure, conflict, invalidation, timeout, and policy violation risks.

    Rare severe risks are preserved through severe_tail_notes instead of being averaged
    away inside a single aggregate risk number.
    """

    failure_risk: float
    conflict_creation_risk: float = 0.0
    artifact_invalidation_risk: float = 0.0
    timeout_risk: float = 0.0
    policy_violation_risk: float = 0.0
    severe_tail_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate that every risk probability is finite and inside the unit interval."""
        for name in ("failure_risk", "conflict_creation_risk",
                     "artifact_invalidation_risk", "timeout_risk", "policy_violation_risk"):
            _require_number(name, getattr(self, name), minimum=0.0, maximum=1.0)
        object.__setattr__(self, "severe_tail_notes", tuple(self.severe_tail_notes))

    def aggregate(self) -> float:
        """Return a documented upper-bound aggregate used only for sequence scoring."""
        return min(1.0, max(
            self.failure_risk, self.conflict_creation_risk, self.artifact_invalidation_risk,
            self.timeout_risk, self.policy_violation_risk))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible risk representation."""
        return {
            "failure_risk": self.failure_risk,
            "conflict_creation_risk": self.conflict_creation_risk,
            "artifact_invalidation_risk": self.artifact_invalidation_risk,
            "timeout_risk": self.timeout_risk,
            "policy_violation_risk": self.policy_violation_risk,
            "severe_tail_notes": list(self.severe_tail_notes),
        }


@dataclass(frozen=True, slots=True)
class PredictionUncertainty:
    """Attach intervals, dispersion, support, and calibration status to one prediction."""

    success_credible_interval: tuple[float, float]
    sample_count: int
    duration_interval_ms: tuple[float, float] | None = None
    duration_variance_ms2: float | _MissingMeasurement = MISSING
    calibration_status: str = "uncalibrated"
    out_of_distribution: bool = False
    confidence_multiplier: float = 1.0

    def __post_init__(self) -> None:
        """Validate interval ordering and support counts without coercing missingness."""
        low, high = self.success_credible_interval
        if not (0.0 <= low <= high <= 1.0):
            raise ValueError("success_credible_interval must be an ordered subinterval of [0, 1]")
        if self.duration_interval_ms is not None:
            duration_low, duration_high = self.duration_interval_ms
            if duration_high < duration_low or duration_low < 0.0:
                raise ValueError("duration_interval_ms must be ordered and non-negative")
        _require_measurement("duration_variance_ms2", self.duration_variance_ms2, minimum=0.0)
        if self.sample_count < 0:
            raise ValueError("sample_count cannot be negative")
        if not self.calibration_status.strip():
            raise ValueError("calibration_status must be non-empty")
        _require_number("confidence_multiplier", self.confidence_multiplier,
                        minimum=0.0, maximum=1.0)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible uncertainty representation."""
        return {
            "success_credible_interval": list(self.success_credible_interval),
            "duration_interval_ms": (
                list(self.duration_interval_ms) if self.duration_interval_ms is not None else None
            ),
            "duration_variance_ms2": measurement(self.duration_variance_ms2),
            "sample_count": self.sample_count,
            "calibration_status": self.calibration_status,
            "out_of_distribution": self.out_of_distribution,
            "confidence_multiplier": self.confidence_multiplier,
        }


@dataclass(frozen=True, slots=True)
class InformationGainEstimate:
    """Report expected information gain with an explicit entropy or proxy labeling."""

    value: float
    kind: str
    basis: str

    def __post_init__(self) -> None:
        """Validate the estimate and require an explicit labeling vocabulary."""
        _require_number("value", self.value, minimum=0.0)
        if self.kind not in ("entropy", "proxy"):
            raise ValueError("kind must be 'entropy' or 'proxy'")
        if not self.basis.strip():
            raise ValueError("basis must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible information-gain representation."""
        return {"value": self.value, "kind": self.kind, "basis": self.basis}


@dataclass(frozen=True, slots=True)
class ArtifactTypeEffect:
    """Predict artifact categories, schemas, and quantities, never semantic content."""

    contract_id: str
    category: str
    expected_quantity: int

    def __post_init__(self) -> None:
        """Validate the typed artifact effect contract."""
        if not self.contract_id.strip() or not self.category.strip():
            raise ValueError("contract_id and category must be non-empty")
        if self.expected_quantity < 0:
            raise ValueError("expected_quantity cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible artifact-type prediction."""
        return {
            "contract_id": self.contract_id,
            "category": self.category,
            "expected_quantity": self.expected_quantity,
        }


@dataclass(frozen=True, slots=True)
class PredictedOutcome:
    """Hold one statistical prediction of the consequences of a candidate operation."""

    success_probability: float
    outcome_class_probabilities: Mapping[str, float]
    expected_cost: ResourceEnvelope
    predicted_feature_delta: Mapping[str, float]
    goal_progress_estimate: float
    information_gain: InformationGainEstimate
    risks: RiskAssessment
    uncertainty: PredictionUncertainty
    predicted_artifacts: tuple[ArtifactTypeEffect, ...] = ()
    explanation: str = ""

    def __post_init__(self) -> None:
        """Validate the predicted distributions and detach mutable mappings."""
        _require_number("success_probability", self.success_probability, minimum=0.0, maximum=1.0)
        _require_number("goal_progress_estimate", self.goal_progress_estimate)
        probabilities = dict(self.outcome_class_probabilities)
        if not probabilities:
            raise ValueError("outcome_class_probabilities cannot be empty")
        total = 0.0
        for name, probability in probabilities.items():
            if not name.strip():
                raise ValueError("outcome class names must be non-empty")
            _require_number(f"probability[{name}]", probability, minimum=0.0, maximum=1.0)
            total += float(probability)
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("outcome class probabilities must sum to one")
        object.__setattr__(self, "outcome_class_probabilities", probabilities)
        deltas = {name: float(value) for name, value in self.predicted_feature_delta.items()}
        object.__setattr__(self, "predicted_feature_delta", deltas)
        object.__setattr__(self, "predicted_artifacts", tuple(self.predicted_artifacts))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible prediction body."""
        return {
            "success_probability": self.success_probability,
            "outcome_class_probabilities": dict(self.outcome_class_probabilities),
            "expected_cost": self.expected_cost.to_dict(),
            "predicted_feature_delta": dict(self.predicted_feature_delta),
            "goal_progress_estimate": self.goal_progress_estimate,
            "information_gain": self.information_gain.to_dict(),
            "risks": self.risks.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "predicted_artifacts": [item.to_dict() for item in self.predicted_artifacts],
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """Tie one prediction to the exact workspace revision, goal, operation, and model version."""

    workspace_id: str
    workspace_revision: int
    goal_id: str
    operation_id: str
    feature_snapshot_hash: str
    predicted: PredictedOutcome
    predictor_version: str
    prediction_id: str = field(default_factory=lambda: _new_id("cwm_prediction"))
    sequence_id: str | None = None
    horizon: int = 1
    constraints_considered: tuple[str, ...] = ()
    advisory_mode: str = AdvisoryMode.SHADOW.value
    record_kind: str = "provisional_planning"
    timestamp: str = field(default_factory=_utc_now)
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate the record identity and its explicit provisional planning status."""
        for name in ("workspace_id", "goal_id", "operation_id", "predictor_version"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.workspace_revision < 0:
            raise ValueError("workspace_revision cannot be negative")
        if len(self.feature_snapshot_hash.strip()) < 8:
            raise ValueError("feature_snapshot_hash must identify the exact feature state")
        if not self.prediction_id.strip():
            raise ValueError("prediction_id must be non-empty")
        if self.horizon < 1:
            raise ValueError("horizon must be at least one")
        if self.record_kind != "provisional_planning":
            raise ValueError("predictions are always provisional planning objects")
        object.__setattr__(self, "constraints_considered", tuple(self.constraints_considered))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible prediction record with full lineage fields."""
        return {
            "prediction_id": self.prediction_id,
            "workspace_id": self.workspace_id,
            "workspace_revision": self.workspace_revision,
            "goal_id": self.goal_id,
            "operation_id": self.operation_id,
            "sequence_id": self.sequence_id,
            "feature_snapshot_hash": self.feature_snapshot_hash,
            "predictor_version": self.predictor_version,
            "horizon": self.horizon,
            "predicted": self.predicted.to_dict(),
            "constraints_considered": list(self.constraints_considered),
            "advisory_mode": self.advisory_mode,
            "record_kind": self.record_kind,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class ObservedOutcome:
    """Capture the actual result of one dispatched operation with identical metric definitions."""

    operation_id: str
    module_id: str
    workspace_id: str
    workspace_revision_before: int
    workspace_revision_after: int
    execution_status: ExecutionStatus
    dimensions: OutcomeDimensions
    outcome_id: str = field(default_factory=lambda: _new_id("cwm_outcome"))
    prediction_id: str | None = None
    artifact_lifecycle: ArtifactLifecycle = ArtifactLifecycle.GENERATED
    artifacts_generated: int = 0
    artifacts_accepted: int = 0
    timestamp: str = field(default_factory=_utc_now)
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate the observed record identity and revision window."""
        for name in ("operation_id", "module_id", "workspace_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.workspace_revision_before < 0 or self.workspace_revision_after < 0:
            raise ValueError("workspace revisions cannot be negative")
        if self.artifacts_generated < 0 or self.artifacts_accepted < 0:
            raise ValueError("artifact counts cannot be negative")
        if not self.outcome_id.strip():
            raise ValueError("outcome_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible observed outcome."""
        return {
            "outcome_id": self.outcome_id,
            "prediction_id": self.prediction_id,
            "operation_id": self.operation_id,
            "module_id": self.module_id,
            "workspace_id": self.workspace_id,
            "workspace_revision_before": self.workspace_revision_before,
            "workspace_revision_after": self.workspace_revision_after,
            "execution_status": self.execution_status.value,
            "dimensions": self.dimensions.to_dict(),
            "artifact_lifecycle": self.artifact_lifecycle.value,
            "artifacts_generated": self.artifacts_generated,
            "artifacts_accepted": self.artifacts_accepted,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class PredictionLineage:
    """Link one prediction through dispatch, execution, artifacts, validation, and outcome."""

    prediction_id: str
    selected: bool
    dispatched_task_id: str | None = None
    module_execution_ids: tuple[str, ...] = ()
    output_artifact_ids: tuple[str, ...] = ()
    validation_decisions: tuple[str, ...] = ()
    observed_outcome_id: str | None = None

    def __post_init__(self) -> None:
        """Normalize lineage tuples and require a non-empty prediction reference."""
        if not self.prediction_id.strip():
            raise ValueError("prediction_id must be non-empty")
        object.__setattr__(self, "module_execution_ids", tuple(self.module_execution_ids))
        object.__setattr__(self, "output_artifact_ids", tuple(self.output_artifact_ids))
        object.__setattr__(self, "validation_decisions", tuple(self.validation_decisions))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible lineage record."""
        return {
            "prediction_id": self.prediction_id,
            "selected": self.selected,
            "dispatched_task_id": self.dispatched_task_id,
            "module_execution_ids": list(self.module_execution_ids),
            "output_artifact_ids": list(self.output_artifact_ids),
            "validation_decisions": list(self.validation_decisions),
            "observed_outcome_id": self.observed_outcome_id,
        }


@dataclass(frozen=True, slots=True)
class FallbackDecision:
    """Record one deterministic deferral to existing ORE control logic."""

    trigger: FallbackTrigger
    reason: str
    deferred_to: str = "cognitive_task_coordinator.deterministic_policy"
    timestamp: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        """Require an explicit reason for every fallback."""
        if not self.reason.strip() or not self.deferred_to.strip():
            raise ValueError("fallback reason and deferral target must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible fallback decision."""
        return {
            "trigger": self.trigger.value,
            "reason": self.reason,
            "deferred_to": self.deferred_to,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True, slots=True)
class CWMConfig:
    """Configure estimator priors, gating thresholds, and bounded-planning defaults.

    All values are engineering starting points that require calibration against replayed
    execution history before they are treated as more than conservative defaults.
    """

    success_prior_alpha: float = 1.0
    success_prior_beta: float = 1.0
    cost_ema_rho: float = 0.35
    robust_window: int = 32
    min_support: int = 5
    max_success_interval_width: float = 0.6
    default_horizon: int = 2
    max_candidates: int = 8
    max_sequences: int = 64
    planning_time_budget_ms: float = 250.0
    uncertainty_margin: float = 0.5
    ood_feature_distance: float = 1.0
    calibration_window: int = 64
    drift_error_ratio: float = 2.0
    goal_threshold: float = 0.75

    def __post_init__(self) -> None:
        """Validate estimator and planning parameters with documented ranges."""
        for name in ("success_prior_alpha", "success_prior_beta", "cost_ema_rho",
                     "max_success_interval_width", "planning_time_budget_ms",
                     "uncertainty_margin", "ood_feature_distance", "goal_threshold"):
            _require_number(name, getattr(self, name), minimum=0.0)
        for name in ("cost_ema_rho", "max_success_interval_width", "goal_threshold"):
            _require_number(name, getattr(self, name), maximum=1.0)
        for name in ("robust_window", "min_support", "default_horizon",
                     "max_candidates", "max_sequences", "calibration_window"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.drift_error_ratio < 1.0:
            raise ValueError("drift_error_ratio must be at least one")


@dataclass(frozen=True, slots=True)
class CWMBudget:
    """Bound one planning invocation's horizon, branching, and runtime consumption."""

    horizon: int = 2
    max_candidates: int = 8
    max_sequences: int = 64
    planning_time_budget_ms: float = 250.0
    cancelled: bool = False

    def __post_init__(self) -> None:
        """Validate the compute bounds declared for one planning call."""
        if not 1 <= self.horizon <= 8:
            raise ValueError("horizon must be between one and eight operations")
        for name in ("max_candidates", "max_sequences"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.planning_time_budget_ms <= 0.0:
            raise ValueError("planning_time_budget_ms must be positive")
