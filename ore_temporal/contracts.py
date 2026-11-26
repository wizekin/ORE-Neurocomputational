"""Typed contracts for the Point 13 Temporal and Event-Driven Intelligence Layer (TEDIL).

These contracts implement the request, observation, event, and result interfaces proposed
in the Point 13 specification ("ORE Integrated Algorithm -- Point 13: Continuous-Time
Neural Networks and Neuromorphic Computing").  They are deliberately explicit about
timestamp semantics, units, quality flags, and provenance so that downstream symbolic
modules can never confuse a prediction with a measurement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
import math
from typing import Any, Mapping

__all__ = (
    "Actionability",
    "AdmissionDecision",
    "BackpressurePolicy",
    "BaselineComparison",
    "CancellationSignal",
    "ClassificationResult",
    "EpistemicStatus",
    "Estimate",
    "EventSeverity",
    "ExecutionRecord",
    "MissingDataPolicy",
    "ModelFamily",
    "ModelIdentity",
    "ObservationChannel",
    "QualityFlag",
    "ResultEnvelope",
    "ResultType",
    "SelectionDecision",
    "TaskStatus",
    "TemporalObservation",
    "TemporalProblemClass",
    "TemporalRequest",
    "TemporalResult",
    "TimestampBasis",
    "TypedEventRecord",
    "UncertaintyReport",
)


def _required(name: str, value: str) -> str:
    """Require a meaningful identifier or descriptive string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _finite(name: str, value: float) -> float:
    """Validate a finite scalar."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return numeric


def _nonnegative(name: str, value: float) -> float:
    """Validate a finite non-negative scalar."""
    numeric = _finite(name, value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return numeric


def _unit_interval(name: str, value: float) -> float:
    """Validate a finite scalar bounded to [0, 1]."""
    numeric = _finite(name, value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return numeric


class TimestampBasis(str, Enum):
    """Distinguish wall-clock, monotonic-process, device, and simulation time bases."""

    WALL_CLOCK = "wall_clock"
    MONOTONIC = "monotonic"
    DEVICE = "device"
    SIMULATION = "simulation"


class TemporalProblemClass(str, Enum):
    """Classify a temporal problem before any model family is considered."""

    STATIC = "static"
    REGULARLY_SAMPLED = "regularly_sampled"
    IRREGULARLY_SAMPLED = "irregularly_sampled"
    CONTINUOUS_DYNAMIC = "continuous_dynamic"
    SPARSE_EVENTS = "sparse_events"


class ModelFamily(str, Enum):
    """Candidate representation families considered by the selection policy."""

    BASELINE = "baseline"
    LNN = "lnn"
    CFC = "cfc"
    NEURAL_ODE = "neural_ode"
    SSM = "ssm"
    SNN = "snn"


class MissingDataPolicy(str, Enum):
    """Explicit policies for handling missing observations."""

    NO_UPDATE = "no_update"
    INTERPOLATE = "interpolate"
    MODEL_NATIVE = "model_native"
    IMPUTE = "impute"


class ResultType(str, Enum):
    """Typed result categories produced by the temporal layer."""

    FORECAST = "forecast"
    EVENT = "event"
    ANOMALY = "anomaly"
    STATE_SUMMARY = "state_summary"
    COMPARISON = "comparison"


class ValidationStatus(str, Enum):
    """Validation outcome attached to every result."""

    VALID = "valid"
    PARTIAL = "partial"
    FALLBACK = "fallback"
    INVALID = "invalid"


class EpistemicStatus(str, Enum):
    """Separate measured, imputed, predicted, inferred, and externally confirmed values."""

    MEASURED = "measured"
    IMPUTED = "imputed"
    PREDICTED = "predicted"
    INFERRED = "inferred"
    EXTERNALLY_CONFIRMED = "externally_confirmed"


class QualityFlag(str, Enum):
    """Quality flags carried by every observation record."""

    GOOD = "good"
    SUSPECT = "suspect"
    OUT_OF_RANGE = "out_of_range"
    MISSING = "missing"
    LATE = "late"
    DUPLICATE = "duplicate"
    IMPUTED = "imputed"


class EventSeverity(str, Enum):
    """Severity levels for detected temporal events."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Actionability(str, Enum):
    """Actionability decisions; neural evidence alone never authorizes high-impact action."""

    NOT_ACTIONABLE = "not_actionable"
    REVIEW_REQUIRED = "review_required"
    ACTIONABLE_WITH_CONSTRAINTS = "actionable_with_constraints"


class BackpressurePolicy(str, Enum):
    """Documented responses when ingestion exceeds capacity."""

    AGGREGATE = "aggregate"
    SAMPLE = "sample"
    REJECT_LATE = "reject_late"
    BLOCK = "block"


class TaskStatus(str, Enum):
    """Terminal statuses for a temporal-analysis task."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FALLBACK = "fallback"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ObservationChannel:
    """Declare one observed channel with unit, physical bounds, and missing-data policy."""

    name: str
    unit: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    missing_policy: MissingDataPolicy = MissingDataPolicy.NO_UPDATE
    description: str = ""

    def __post_init__(self) -> None:
        """Validate channel identity and bounds."""
        _required("name", self.name)
        _required("unit", self.unit)
        if self.lower_bound is not None and self.upper_bound is not None:
            if float(self.lower_bound) > float(self.upper_bound):
                raise ValueError("lower_bound must not exceed upper_bound")
        if not isinstance(self.missing_policy, MissingDataPolicy):
            raise ValueError("missing_policy must be a MissingDataPolicy")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the channel declaration."""
        return {
            "name": self.name,
            "unit": self.unit,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "missing_policy": self.missing_policy.value,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class TemporalRequest:
    """Step 1 -- a temporal-analysis request with objective, schema, budgets, and contract.

    Requests are rejected before resources are allocated when identity, schema, or budget
    fields are malformed.
    """

    task_id: str
    source_id: str
    objective: str
    observation_schema: tuple[ObservationChannel, ...]
    timestamp_semantics: TimestampBasis = TimestampBasis.WALL_CLOCK
    horizon: float = 1.0
    latency_budget: float = 1.0
    resource_budget: float = 1.0
    output_contract: str = "typed_estimates+events+uncertainty"
    required_capabilities: tuple[str, ...] = ()
    requested_family: ModelFamily | None = None

    def __post_init__(self) -> None:
        """Reject malformed requests before allocating resources (Step 1)."""
        _required("task_id", self.task_id)
        _required("source_id", self.source_id)
        _required("objective", self.objective)
        if not isinstance(self.observation_schema, tuple) or not self.observation_schema:
            raise ValueError("observation_schema must be a non-empty tuple of ObservationChannel")
        if any(not isinstance(channel, ObservationChannel) for channel in self.observation_schema):
            raise ValueError("observation_schema entries must be ObservationChannel objects")
        names = [channel.name for channel in self.observation_schema]
        if len(set(names)) != len(names):
            raise ValueError("observation_schema channel names must be unique")
        if not isinstance(self.timestamp_semantics, TimestampBasis):
            raise ValueError("timestamp_semantics must be a TimestampBasis")
        _nonnegative("horizon", self.horizon)
        _nonnegative("latency_budget", self.latency_budget)
        _nonnegative("resource_budget", self.resource_budget)
        _required("output_contract", self.output_contract)
        if self.requested_family is not None and not isinstance(self.requested_family, ModelFamily):
            raise ValueError("requested_family must be a ModelFamily or None")

    def channel(self, name: str) -> ObservationChannel:
        """Return the declared channel by name."""
        for item in self.observation_schema:
            if item.name == name:
                return item
        raise KeyError(f"channel {name!r} is not declared in the observation schema")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the request without flattening typed temporal structure."""
        return {
            "task_id": self.task_id,
            "source_id": self.source_id,
            "objective": self.objective,
            "observation_schema": [item.to_dict() for item in self.observation_schema],
            "timestamp_semantics": self.timestamp_semantics.value,
            "horizon": self.horizon,
            "latency_budget": self.latency_budget,
            "resource_budget": self.resource_budget,
            "output_contract": self.output_contract,
            "required_capabilities": list(self.required_capabilities),
            "requested_family": self.requested_family.value if self.requested_family else None,
        }


@dataclass(frozen=True, slots=True)
class TemporalObservation:
    """Step 8 -- one timestamped observation with units, quality flags, and provenance.

    Units are never inferred from magnitude; they must be declared per value.
    """

    source_id: str
    event_id: str
    timestamp: float
    time_basis: TimestampBasis
    values: tuple[float, ...]
    units: tuple[str, ...]
    quality_flags: tuple[QualityFlag, ...] = (QualityFlag.GOOD,)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identity, timestamp finiteness, value/unit alignment, and flags."""
        _required("source_id", self.source_id)
        _required("event_id", self.event_id)
        object.__setattr__(self, "timestamp", _finite("timestamp", self.timestamp))
        if not isinstance(self.time_basis, TimestampBasis):
            raise ValueError("time_basis must be a TimestampBasis")
        if not isinstance(self.values, tuple) or not self.values:
            raise ValueError("values must be a non-empty tuple")
        if len(self.units) != len(self.values):
            raise ValueError("units must be provided for every value; "
                             "never infer units from magnitude")
        object.__setattr__(self, "values", tuple(_finite(f"values[{index}]", item)
                                                 for index, item in enumerate(self.values)))
        if any(not isinstance(flag, QualityFlag) for flag in self.quality_flags):
            raise ValueError("quality_flags entries must be QualityFlag values")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a mapping")

    @property
    def identity(self) -> tuple[str, str]:
        """Return the stable (source, event) identity used for deduplication."""
        return (self.source_id, self.event_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the observation while preserving clock basis and provenance."""
        return {
            "source_id": self.source_id,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "time_basis": self.time_basis.value,
            "values": list(self.values),
            "units": list(self.units),
            "quality_flags": [flag.value for flag in self.quality_flags],
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class TypedEventRecord:
    """Step 26 -- a typed event estimate; hidden vectors are never exposed as conclusions."""

    event_id: str
    event_type: str
    timestamp: float
    timestamp_end: float | None = None
    magnitude: float | None = None
    probability: float | None = None
    severity: EventSeverity = EventSeverity.INFO
    confidence: float | None = None
    source_id: str = ""
    cause_hypothesis: str = "uncertain"
    epistemic_status: EpistemicStatus = EpistemicStatus.INFERRED
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate event identity and bounded estimates."""
        _required("event_id", self.event_id)
        _required("event_type", self.event_type)
        object.__setattr__(self, "timestamp", _finite("timestamp", self.timestamp))
        if self.timestamp_end is not None:
            end = _finite("timestamp_end", self.timestamp_end)
            if end < self.timestamp:
                raise ValueError("timestamp_end must not precede timestamp")
        if self.magnitude is not None:
            _finite("magnitude", self.magnitude)
        if self.probability is not None:
            _unit_interval("probability", self.probability)
        if self.confidence is not None:
            _unit_interval("confidence", self.confidence)
        if not isinstance(self.severity, EventSeverity):
            raise ValueError("severity must be an EventSeverity")
        if not isinstance(self.epistemic_status, EpistemicStatus):
            raise ValueError("epistemic_status must be an EpistemicStatus")

    @property
    def interval(self) -> tuple[float, float]:
        """Return the [start, end] interval covered by the event."""
        end = self.timestamp_end if self.timestamp_end is not None else self.timestamp
        return (self.timestamp, end)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the typed event record."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "timestamp_end": self.timestamp_end,
            "magnitude": self.magnitude,
            "probability": self.probability,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "source_id": self.source_id,
            "cause_hypothesis": self.cause_hypothesis,
            "epistemic_status": self.epistemic_status.value,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True, slots=True)
class Estimate:
    """Step 26 -- a typed scalar estimate with unit, horizon, and epistemic status."""

    channel: str
    value: float
    unit: str
    timestamp: float
    horizon: float = 0.0
    confidence_interval: tuple[float, float] | None = None
    epistemic_status: EpistemicStatus = EpistemicStatus.PREDICTED

    def __post_init__(self) -> None:
        """Validate the estimate fields."""
        _required("channel", self.channel)
        object.__setattr__(self, "value", _finite("value", self.value))
        _required("unit", self.unit)
        object.__setattr__(self, "timestamp", _finite("timestamp", self.timestamp))
        _nonnegative("horizon", self.horizon)
        if self.confidence_interval is not None:
            low = _finite("confidence_interval[0]", self.confidence_interval[0])
            high = _finite("confidence_interval[1]", self.confidence_interval[1])
            if low > high:
                raise ValueError("confidence_interval lower bound must not exceed upper bound")
        if not isinstance(self.epistemic_status, EpistemicStatus):
            raise ValueError("epistemic_status must be an EpistemicStatus")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the typed estimate."""
        return {
            "channel": self.channel,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "horizon": self.horizon,
            "confidence_interval": (list(self.confidence_interval)
                                    if self.confidence_interval else None),
            "epistemic_status": self.epistemic_status.value,
        }


@dataclass(frozen=True, slots=True)
class UncertaintyReport:
    """Step 29 -- forecast uncertainty with calibration lineage and OOD marking."""

    method: str
    ensemble_mean: float | None = None
    ensemble_variance: float | None = None
    interval: tuple[float, float] | None = None
    coverage_target: float = 0.9
    calibration_version: str = "uncalibrated"
    calibration_observations: int = 0
    out_of_distribution: bool = False
    ood_reason: str = ""

    def __post_init__(self) -> None:
        """Validate uncertainty quantities."""
        _required("method", self.method)
        _unit_interval("coverage_target", self.coverage_target)
        for name in ("ensemble_mean", "ensemble_variance"):
            if getattr(self, name) is not None:
                _finite(name, getattr(self, name))
        if self.ensemble_variance is not None and self.ensemble_variance < 0.0:
            raise ValueError("ensemble_variance must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the uncertainty report."""
        return {
            "method": self.method,
            "ensemble_mean": self.ensemble_mean,
            "ensemble_variance": self.ensemble_variance,
            "interval": list(self.interval) if self.interval else None,
            "coverage_target": self.coverage_target,
            "calibration_version": self.calibration_version,
            "calibration_observations": self.calibration_observations,
            "out_of_distribution": self.out_of_distribution,
            "ood_reason": self.ood_reason,
        }


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Identify the model version, configuration hash, and backend used for a result."""

    family: ModelFamily
    model_version: str
    config_hash: str
    backend: str = "cpu_reference"
    artifact_hash: str = ""
    parameters_owner: str = "ore_temporal.seeded_reference"

    def __post_init__(self) -> None:
        """Validate model identity fields."""
        if not isinstance(self.family, ModelFamily):
            raise ValueError("family must be a ModelFamily")
        _required("model_version", self.model_version)
        _required("config_hash", self.config_hash)
        _required("backend", self.backend)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model identity."""
        return {
            "family": self.family.value,
            "model_version": self.model_version,
            "config_hash": self.config_hash,
            "backend": self.backend,
            "artifact_hash": self.artifact_hash,
            "parameters_owner": self.parameters_owner,
        }


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    """Deterministic-baseline comparison attached to every model-benefit claim."""

    baseline_family: ModelFamily
    baseline_metric: float
    model_metric: float
    metric_name: str
    beats_baseline: bool
    materially_complements: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the baseline comparison."""
        return {
            "baseline_family": self.baseline_family.value,
            "baseline_metric": self.baseline_metric,
            "model_metric": self.model_metric,
            "metric_name": self.metric_name,
            "beats_baseline": self.beats_baseline,
            "materially_complements": self.materially_complements,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class ResultEnvelope:
    """Step 50 -- explainable result envelope returned with every temporal result."""

    interval: tuple[float, float]
    prediction_status: ValidationStatus
    baseline: str
    model_identity: ModelIdentity
    limitations: tuple[str, ...]
    recommended_verification: str
    evidence_summary: str = ""
    uncertainty_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result envelope."""
        return {
            "interval": list(self.interval),
            "prediction_status": self.prediction_status.value,
            "baseline": self.baseline,
            "model_identity": self.model_identity.to_dict(),
            "limitations": list(self.limitations),
            "recommended_verification": self.recommended_verification,
            "evidence_summary": self.evidence_summary,
            "uncertainty_summary": self.uncertainty_summary,
        }


@dataclass(frozen=True, slots=True)
class TemporalResult:
    """Steps 26 and 50 -- the complete typed temporal result with provenance and status."""

    task_id: str
    source_id: str
    interval: tuple[float, float]
    result_type: ResultType
    estimates: tuple[Estimate, ...] = ()
    events: tuple[TypedEventRecord, ...] = ()
    uncertainty: UncertaintyReport | None = None
    evidence_refs: tuple[str, ...] = ()
    model_identity: ModelIdentity | None = None
    baseline_comparison: BaselineComparison | None = None
    validation_status: ValidationStatus = ValidationStatus.VALID
    problem_class: TemporalProblemClass = TemporalProblemClass.IRREGULARLY_SAMPLED
    operation_trace: tuple[Mapping[str, Any], ...] = ()
    limitations: tuple[str, ...] = ()
    recommended_verification: str = ""
    envelope: ResultEnvelope | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result identity and interval."""
        _required("task_id", self.task_id)
        _required("source_id", self.source_id)
        start = _finite("interval[0]", self.interval[0])
        end = _finite("interval[1]", self.interval[1])
        if end < start:
            raise ValueError("interval end must not precede start")
        if not isinstance(self.result_type, ResultType):
            raise ValueError("result_type must be a ResultType")
        if not isinstance(self.validation_status, ValidationStatus):
            raise ValueError("validation_status must be a ValidationStatus")
        if not isinstance(self.problem_class, TemporalProblemClass):
            raise ValueError("problem_class must be a TemporalProblemClass")

    def build_envelope(self) -> ResultEnvelope:
        """Step 50 -- materialize the explainable envelope when not already attached."""
        if self.envelope is not None:
            return self.envelope
        identity = self.model_identity or ModelIdentity(ModelFamily.BASELINE, "1.0.0", "none")
        uncertainty_summary = "no uncertainty estimate" if self.uncertainty is None else (
            f"method={self.uncertainty.method}; "
            f"calibration={self.uncertainty.calibration_version}; "
            f"ood={self.uncertainty.out_of_distribution}"
        )
        return ResultEnvelope(
            interval=self.interval,
            prediction_status=self.validation_status,
            baseline=(self.baseline_comparison.baseline_family.value
                      if self.baseline_comparison else "none"),
            model_identity=identity,
            limitations=self.limitations,
            recommended_verification=self.recommended_verification or (
                "Compare against the deterministic baseline on a held-out chronological window."),
            evidence_summary=(f"{len(self.evidence_refs)} evidence references; "
                              f"{len(self.estimates)} typed estimates; "
                              f"{len(self.events)} typed events"),
            uncertainty_summary=uncertainty_summary,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result for publication without raw stream payloads."""
        return {
            "task_id": self.task_id,
            "source_id": self.source_id,
            "interval": list(self.interval),
            "result_type": self.result_type.value,
            "estimates": [item.to_dict() for item in self.estimates],
            "events": [item.to_dict() for item in self.events],
            "uncertainty": self.uncertainty.to_dict() if self.uncertainty else None,
            "evidence_refs": list(self.evidence_refs),
            "model_identity": self.model_identity.to_dict() if self.model_identity else None,
            "baseline_comparison": (self.baseline_comparison.to_dict()
                                    if self.baseline_comparison else None),
            "validation_status": self.validation_status.value,
            "problem_class": self.problem_class.value,
            "limitations": list(self.limitations),
            "recommended_verification": self.recommended_verification,
            "diagnostics": _plain_dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """Step 7 -- an auditable record of model, configuration, data window, and versions."""

    task_id: str
    model_identity: ModelIdentity
    data_window: tuple[float, float]
    normalization_version: str
    seed: int
    solver_settings: Mapping[str, Any]
    software_versions: Mapping[str, Any]
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the execution record."""
        return {
            "task_id": self.task_id,
            "model_identity": self.model_identity.to_dict(),
            "data_window": list(self.data_window),
            "normalization_version": self.normalization_version,
            "seed": self.seed,
            "solver_settings": _plain_dict(self.solver_settings),
            "software_versions": _plain_dict(self.software_versions),
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    """Steps 4-5 -- logged, testable model-family selection with rationale and caution."""

    chosen_family: ModelFamily
    alternatives: tuple[ModelFamily, ...]
    rationale: str
    caution: str
    baseline_family: ModelFamily = ModelFamily.BASELINE
    profile: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the selection decision."""
        return {
            "chosen_family": self.chosen_family.value,
            "alternatives": [item.value for item in self.alternatives],
            "rationale": self.rationale,
            "caution": self.caution,
            "baseline_family": self.baseline_family.value,
            "profile": _plain_dict(self.profile),
        }


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Step 6 -- resource admission decision before any weights are loaded."""

    admitted: bool
    reason: str
    deferred: bool = False
    downsample: bool = False
    fallback_to_baseline: bool = False
    estimated_cost: float = 0.0
    budget: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the admission decision."""
        return {
            "admitted": self.admitted,
            "reason": self.reason,
            "deferred": self.deferred,
            "downsample": self.downsample,
            "fallback_to_baseline": self.fallback_to_baseline,
            "estimated_cost": self.estimated_cost,
            "budget": self.budget,
        }


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Step 2 -- the recorded temporal-problem classification and its confidence."""

    problem_class: TemporalProblemClass
    confidence: float
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the classification result."""
        if not isinstance(self.problem_class, TemporalProblemClass):
            raise ValueError("problem_class must be a TemporalProblemClass")
        _unit_interval("confidence", self.confidence)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the classification result."""
        return {
            "problem_class": self.problem_class.value,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


class CancellationSignal:
    """Cooperative cancellation token checked between pipeline phases (Step 38)."""

    def __init__(self) -> None:
        """Initialize an uncancelled signal."""
        self._cancelled = False
        self.reason = ""

    @property
    def cancelled(self) -> bool:
        """Report whether cancellation was requested."""
        return self._cancelled

    def cancel(self, reason: str = "requested") -> None:
        """Request cancellation; running phases finish safely at the next check."""
        self._cancelled = True
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        """Serialize the cancellation state."""
        return {"cancelled": self._cancelled, "reason": self.reason}


def _plain_dict(value: Any) -> dict[str, Any]:
    """Convert nested dataclasses and mappings into plain JSON-compatible dictionaries."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_dict(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
