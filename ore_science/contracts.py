"""Typed schemas for scientific tasks, variables, constraints, results, and uncertainty.

This module implements the ``ore_science.contracts`` boundary from the integrated
algorithm: every scientific task carries explicit variables, units, constraints,
assumptions, and evidence requirements; every outcome is one of a closed set of typed
result kinds so that invalid input, unsupported domains, and validation failures can
never masquerade as successful scientific conclusions.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .units import Quantity, Unit

__all__ = [
    "Assumption",
    "AssumptionSource",
    "ConsequenceLevel",
    "Constraint",
    "DomainDecision",
    "EquationModel",
    "EvidenceRequirements",
    "ExplanationBlock",
    "ObjectiveKind",
    "OutcomeKind",
    "ProblemClass",
    "ResultStatus",
    "ScienceConfig",
    "ScienceRequest",
    "ScienceResult",
    "ScientificDomain",
    "ScientificPrediction",
    "ScientificTask",
    "ScientificVariable",
    "StatusTransition",
    "SubTaskLink",
    "UncertaintyReport",
    "ValidationCheck",
    "ValidationReport",
    "VariableRole",
]


class ScientificDomain(str, Enum):
    """Domains recognized by the scientific task interpreter (Step 1)."""

    MOLECULAR_STRUCTURE = "MOLECULAR_STRUCTURE"
    MOLECULAR_INTERACTION = "MOLECULAR_INTERACTION"
    PROTEIN_DESIGN = "PROTEIN_DESIGN"
    MATERIALS = "MATERIALS"
    PHYSICAL_SIMULATION = "PHYSICAL_SIMULATION"
    GEOMETRIC = "GEOMETRIC"
    FORECASTING = "FORECASTING"
    LOCALIZATION = "LOCALIZATION"
    GENERAL_SCIENCE = "GENERAL_SCIENCE"


class ObjectiveKind(str, Enum):
    """Explicit objective types extracted from a natural-language request (Step 2)."""

    PREDICT_STRUCTURE = "PREDICT_STRUCTURE"
    ESTIMATE_FIELD = "ESTIMATE_FIELD"
    FORECAST_DISTRIBUTION = "FORECAST_DISTRIBUTION"
    INFER_STATE = "INFER_STATE"
    GENERATE_CANDIDATE = "GENERATE_CANDIDATE"
    OPTIMIZE_PROPERTY = "OPTIMIZE_PROPERTY"
    EXPLAIN_RELATION = "EXPLAIN_RELATION"
    VALIDATE_RESULT = "VALIDATE_RESULT"


class ProblemClass(str, Enum):
    """Mathematical problem classes that determine admissible solvers (Step 6)."""

    FORWARD = "FORWARD"
    INVERSE = "INVERSE"
    PARAMETER_ESTIMATION = "PARAMETER_ESTIMATION"
    CONSTRAINED_OPTIMIZATION = "CONSTRAINED_OPTIMIZATION"
    PROBABILISTIC_PREDICTION = "PROBABILISTIC_PREDICTION"
    GENERATIVE_DESIGN = "GENERATIVE_DESIGN"
    STATE_ESTIMATION = "STATE_ESTIMATION"
    HYBRID = "HYBRID"


class ResultStatus(str, Enum):
    """Explicit evidence states for scientific results (Step 43).

    CANDIDATE means generated; PREDICTED means a model output exists;
    COMPUTATIONALLY_VALIDATED means independent computational checks passed;
    EXPERIMENTALLY_CONFIRMED means a relevant experiment supports the claim.
    Status promotion always requires an explicit review gate; it is never automatic.
    """

    CANDIDATE = "CANDIDATE"
    PREDICTED = "PREDICTED"
    COMPUTATIONALLY_VALIDATED = "COMPUTATIONALLY_VALIDATED"
    EXPERIMENTALLY_CONFIRMED = "EXPERIMENTALLY_CONFIRMED"


_VALID_TRANSITIONS: dict[ResultStatus, tuple[ResultStatus, ...]] = {
    ResultStatus.CANDIDATE: (ResultStatus.PREDICTED,),
    ResultStatus.PREDICTED: (ResultStatus.COMPUTATIONALLY_VALIDATED,),
    ResultStatus.COMPUTATIONALLY_VALIDATED: (ResultStatus.EXPERIMENTALLY_CONFIRMED,),
    ResultStatus.EXPERIMENTALLY_CONFIRMED: (),
}


class AssumptionSource(str, Enum):
    """Where an assumption came from: user, USG retrieval, inference, or model design."""

    USER = "USER"
    USG = "USG"
    INFERRED = "INFERRED"
    MODEL_IMPOSED = "MODEL_IMPOSED"


class VariableRole(str, Enum):
    """Role of a variable in the task's variable registry (Step 3)."""

    OBSERVED = "OBSERVED"
    LATENT = "LATENT"
    PARAMETER = "PARAMETER"
    CONTROL = "CONTROL"
    PREDICTED = "PREDICTED"


class OutcomeKind(str, Enum):
    """Typed terminal outcomes for scientific reasoning runs.

    These distinguish invalid input, unsupported domains, incompatible representations,
    resource exhaustion, solver failure, and validation failure from successful
    completion, as required by the suggested-interface contract.
    """

    COMPLETED = "COMPLETED"
    INVALID_INPUT = "INVALID_INPUT"
    UNSUPPORTED_DOMAIN = "UNSUPPORTED_DOMAIN"
    INCOMPATIBLE_REPRESENTATION = "INCOMPATIBLE_REPRESENTATION"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    SOLVER_FAILURE = "SOLVER_FAILURE"
    VALIDATION_FAILURE = "VALIDATION_FAILURE"
    LIMITED_RESULT = "LIMITED_RESULT"


class ConsequenceLevel(str, Enum):
    """Consequence level of a result, which sets the strength of required validation."""

    EXPLORATORY = "EXPLORATORY"
    RESEARCH_SUPPORTING = "RESEARCH_SUPPORTING"
    CONSEQUENTIAL = "CONSEQUENTIAL"


@dataclass(frozen=True, slots=True)
class ScientificVariable:
    """One entry of the variable registry: symbol, meaning, value, unit, and provenance."""

    symbol: str
    meaning: str
    role: VariableRole
    unit: Unit
    value: float | None = None
    value_uncertainty: float | None = None
    domain: str = ""
    coordinate_frame: str = "global"
    provenance: str = "user_supplied"
    usg_concept_id: str | None = None

    def __post_init__(self) -> None:
        """Require a symbol, meaning, role, and unit, and a finite value when present."""
        if not self.symbol.strip():
            raise ValueError("variable symbol must be non-empty")
        if not self.meaning.strip():
            raise ValueError(f"variable {self.symbol} must declare its semantic meaning")
        if not isinstance(self.unit, Unit):
            raise TypeError(f"variable {self.symbol} requires a typed Unit")
        if self.value is not None and not math.isfinite(float(self.value)):
            raise ValueError(f"variable {self.symbol} value must be finite")
        if self.value_uncertainty is not None and self.value_uncertainty < 0.0:
            raise ValueError(f"variable {self.symbol} uncertainty must be non-negative")

    @property
    def is_unknown(self) -> bool:
        """A variable is unknown when no value has been supplied or observed."""
        return self.value is None

    def quantity(self) -> Quantity | None:
        """Return the variable value as a typed Quantity, or None when unknown."""
        return None if self.value is None else Quantity(float(self.value), self.unit)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the variable registry entry for provenance and workspace records."""
        return {
            "symbol": self.symbol,
            "meaning": self.meaning,
            "role": self.role.value,
            "unit": self.unit.symbol,
            "dimension": self.unit.dimension.describe(),
            "value": self.value,
            "value_uncertainty": self.value_uncertainty,
            "domain": self.domain,
            "coordinate_frame": self.coordinate_frame,
            "provenance": self.provenance,
            "usg_concept_id": self.usg_concept_id,
            "unknown": self.is_unknown,
        }


@dataclass(frozen=True, slots=True)
class Constraint:
    """An explicit law, boundary/initial condition, or admissible range (Step 4)."""

    kind: str
    expression: str
    description: str
    source: AssumptionSource
    tolerance: float | None = None
    applies_to: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require a typed constraint with a non-empty statement."""
        if not self.kind.strip() or not self.expression.strip():
            raise ValueError("constraint kind and expression must be non-empty")
        if not self.description.strip():
            raise ValueError("constraint description must be non-empty")
        if self.tolerance is not None and (not math.isfinite(self.tolerance) or self.tolerance < 0.0):
            raise ValueError("constraint tolerance must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the constraint for validation and provenance records."""
        return {
            "kind": self.kind,
            "expression": self.expression,
            "description": self.description,
            "source": self.source.value,
            "tolerance": self.tolerance,
            "applies_to": list(self.applies_to),
        }


@dataclass(frozen=True, slots=True)
class EquationModel:
    """A governing equation with term-level units for dimensional validation (Step 22).

    ``terms`` lists the additive terms of the equation with the composed unit of each
    term, so that dimensional homogeneity can be checked across them. ``symbols`` lists
    the individual quantities appearing in the equation with their own units, so the
    variable registry can be linked to the equation.
    """

    equation_id: str
    name: str
    form: str
    terms: tuple[tuple[str, Unit], ...]
    symbols: tuple[tuple[str, Unit], ...] = ()
    conditions: tuple[str, ...] = ()
    usg_reference: str | None = None
    domain: ScientificDomain = ScientificDomain.PHYSICAL_SIMULATION

    def __post_init__(self) -> None:
        """Require an identified equation with at least two typed additive terms."""
        if not self.equation_id.strip() or not self.name.strip():
            raise ValueError("equation id and name must be non-empty")
        if not self.form.strip():
            raise ValueError("equation form must be non-empty")
        if len(self.terms) < 2:
            raise ValueError("equations require at least two terms for homogeneity checks")
        for symbol, unit in self.terms:
            if not symbol.strip() or not isinstance(unit, Unit):
                raise ValueError("equation terms must pair a label with a typed Unit")
        for symbol, unit in self.symbols:
            if not symbol.strip() or not isinstance(unit, Unit):
                raise ValueError("equation symbols must pair a symbol with a typed Unit")

    def term_quantities(self) -> list[Quantity]:
        """Return unit placeholders for each additive term, for homogeneity checks."""
        return [Quantity(1.0, unit) for _, unit in self.terms]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the equation record with term and symbol units."""
        return {
            "equation_id": self.equation_id,
            "name": self.name,
            "form": self.form,
            "terms": [[symbol, unit.symbol] for symbol, unit in self.terms],
            "symbols": [[symbol, unit.symbol] for symbol, unit in self.symbols],
            "conditions": list(self.conditions),
            "usg_reference": self.usg_reference,
            "domain": self.domain.value,
        }


@dataclass(frozen=True, slots=True)
class Assumption:
    """One declared assumption with its source tag (Step 4)."""

    statement: str
    source: AssumptionSource
    detail: str = ""

    def __post_init__(self) -> None:
        """Require a non-empty assumption statement."""
        if not self.statement.strip():
            raise ValueError("assumption statement must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the assumption record."""
        return {
            "statement": self.statement,
            "source": self.source.value,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class EvidenceRequirements:
    """Precision, confidence, and consequence requirements (Step 7)."""

    accuracy: float | None = None
    confidence_level: float = 0.95
    resolution: str = ""
    prediction_horizon: str = ""
    consequence_level: ConsequenceLevel = ConsequenceLevel.RESEARCH_SUPPORTING

    def __post_init__(self) -> None:
        """Confidence must be a valid probability and accuracy a positive fraction."""
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly between 0 and 1")
        if self.accuracy is not None and (not math.isfinite(self.accuracy) or self.accuracy <= 0.0):
            raise ValueError("accuracy must be a positive finite number")

    def to_dict(self) -> dict[str, Any]:
        """Serialize evidence requirements."""
        return {
            "accuracy": self.accuracy,
            "confidence_level": self.confidence_level,
            "resolution": self.resolution,
            "prediction_horizon": self.prediction_horizon,
            "consequence_level": self.consequence_level.value,
        }


@dataclass(frozen=True, slots=True)
class SubTaskLink:
    """A typed dependency between subtasks of a multi-domain scientific workflow."""

    subtask_id: str
    domain: ScientificDomain
    objective: str
    depends_on: tuple[str, ...] = ()
    question: str = ""

    def __post_init__(self) -> None:
        """Require an identified subtask with an objective."""
        if not self.subtask_id.strip() or not self.objective.strip():
            raise ValueError("subtask id and objective must be non-empty")


@dataclass(frozen=True, slots=True)
class ScientificTask:
    """The structured output of the scientific task interpreter (Phase I, Steps 1-7)."""

    question: str
    domains: tuple[ScientificDomain, ...]
    objective: str
    objective_kind: ObjectiveKind
    problem_class: ProblemClass
    variables: tuple[ScientificVariable, ...]
    constraints: tuple[Constraint, ...]
    equations: tuple[EquationModel, ...]
    assumptions: tuple[Assumption, ...]
    evidence_requirements: EvidenceRequirements
    subtasks: tuple[SubTaskLink, ...] = ()
    source_reference: str = "unspecified"
    task_id: str = "scientific_task"

    def __post_init__(self) -> None:
        """A task must state a question, at least one domain, and an objective."""
        if not self.question.strip():
            raise ValueError("scientific task requires a question")
        if not self.domains:
            raise ValueError("scientific task requires at least one domain")
        if not self.objective.strip():
            raise ValueError("scientific task requires an explicit objective")

    def variable(self, symbol: str) -> ScientificVariable | None:
        """Return the registry entry for a symbol, or None when absent."""
        for candidate in self.variables:
            if candidate.symbol == symbol:
                return candidate
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full task interpretation for logs and provenance."""
        return {
            "task_id": self.task_id,
            "question": self.question,
            "domains": [domain.value for domain in self.domains],
            "objective": self.objective,
            "objective_kind": self.objective_kind.value,
            "problem_class": self.problem_class.value,
            "variables": [variable.to_dict() for variable in self.variables],
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "equations": [equation.to_dict() for equation in self.equations],
            "assumptions": [assumption.to_dict() for assumption in self.assumptions],
            "evidence_requirements": self.evidence_requirements.to_dict(),
            "subtasks": [
                {
                    "subtask_id": link.subtask_id,
                    "domain": link.domain.value,
                    "objective": link.objective,
                    "depends_on": list(link.depends_on),
                }
                for link in self.subtasks
            ],
            "source_reference": self.source_reference,
        }


@dataclass(frozen=True, slots=True)
class DomainDecision:
    """Result of Step 1 domain detection: domains plus the evidence for the routing."""

    domains: tuple[ScientificDomain, ...]
    confidence: float
    matched_keywords: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        """Routing decisions must name at least one domain with a stated reason."""
        if not self.domains:
            raise ValueError("domain decision requires at least one domain")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("domain decision confidence must lie in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the routing decision."""
        return {
            "domains": [domain.value for domain in self.domains],
            "confidence": self.confidence,
            "matched_keywords": list(self.matched_keywords),
            "reason": self.reason,
        }


@dataclass(slots=True)
class ScientificPrediction:
    """A model output with its identity, units, status, and references (Phase III)."""

    model_id: str
    model_version: str
    model_family: str
    description: str
    output_unit: Unit
    payload: dict[str, Any]
    status: ResultStatus = ResultStatus.PREDICTED
    confidence_information: dict[str, Any] = field(default_factory=dict)
    validation_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the prediction record without inlining large array payloads."""
        payload_summary: dict[str, Any] = {}
        for key, value in self.payload.items():
            payload_summary[key] = _summarize_payload_value(value)
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_family": self.model_family,
            "description": self.description,
            "output_unit": self.output_unit.symbol,
            "payload": payload_summary,
            "status": self.status.value,
            "confidence_information": dict(self.confidence_information),
            "validation_summary": dict(self.validation_summary),
        }


def _summarize_payload_value(value: Any) -> Any:
    """Reduce arrays and callables to compact descriptions for serialization."""
    if hasattr(value, "shape"):
        return {"kind": "array", "shape": list(value.shape)}
    if callable(value):
        return {"kind": "callable", "name": getattr(value, "__name__", "anonymous")}
    if isinstance(value, dict):
        return {key: _summarize_payload_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 8:
            return {"kind": type(value).__name__, "length": len(value)}
        return [_summarize_payload_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One named validation check with an explicit pass/fail decision and detail."""

    name: str
    passed: bool
    detail: str
    severity: str = "required"

    def __post_init__(self) -> None:
        """Checks must carry a non-empty name and explanation."""
        if not self.name.strip() or not self.detail.strip():
            raise ValueError("validation check name and detail must be non-empty")
        if self.severity not in {"required", "advisory"}:
            raise ValueError("validation severity must be 'required' or 'advisory'")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the check record."""
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "detail": self.detail,
            "severity": self.severity,
        }


@dataclass(slots=True)
class ValidationReport:
    """Aggregated validation outcome for one scientific prediction (Phase IV)."""

    checks: list[ValidationCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """A report passes when every required check passed."""
        return all(check.passed for check in self.checks if check.severity == "required")

    def add(self, check: ValidationCheck) -> None:
        """Append one check to the report."""
        if not isinstance(check, ValidationCheck):
            raise TypeError("add expects a ValidationCheck")
        self.checks.append(check)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report with an overall status."""
        return {
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class UncertaintyReport:
    """Uncertainty record separating supported uncertainty kinds and limitations."""

    method: str
    point_estimate: float | None = None
    interval: tuple[float, float] | None = None
    interval_confidence: float | None = None
    aleatoric: dict[str, Any] = field(default_factory=dict)
    epistemic: dict[str, Any] = field(default_factory=dict)
    alternative_futures: int = 0
    calibration_evidence: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the uncertainty record."""
        return {
            "method": self.method,
            "point_estimate": self.point_estimate,
            "interval": list(self.interval) if self.interval else None,
            "interval_confidence": self.interval_confidence,
            "aleatoric": dict(self.aleatoric),
            "epistemic": dict(self.epistemic),
            "alternative_futures": self.alternative_futures,
            "calibration_evidence": dict(self.calibration_evidence),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class StatusTransition:
    """An explicit, review-gated status change for a scientific result (Step 43)."""

    from_status: ResultStatus
    to_status: ResultStatus
    gate: str
    reviewer: str
    evidence_reference: str

    def __post_init__(self) -> None:
        """Only ladder-adjacent promotions are legal and the gate must be named."""
        if self.to_status not in _VALID_TRANSITIONS[self.from_status]:
            raise ValueError(
                f"illegal status transition {self.from_status.value} -> {self.to_status.value}: "
                "status changes must follow the candidate -> predicted -> validated -> "
                "experimentally confirmed ladder one step at a time"
            )
        if not self.gate.strip() or not self.reviewer.strip():
            raise ValueError("status transitions require a named gate and reviewer")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the transition record."""
        return {
            "from_status": self.from_status.value,
            "to_status": self.to_status.value,
            "gate": self.gate,
            "reviewer": self.reviewer,
            "evidence_reference": self.evidence_reference,
        }


def validate_status_transition(
    from_status: ResultStatus, to_status: ResultStatus
) -> bool:
    """Return True when a proposed status promotion follows the evidence ladder."""
    return to_status in _VALID_TRANSITIONS[from_status]


@dataclass(slots=True)
class ExplanationBlock:
    """A calibrated explanation of a scientific result (Step 52)."""

    representation_rationale: str = ""
    solver_rationale: str = ""
    applied_constraints: list[str] = field(default_factory=list)
    validation_summary: str = ""
    remaining_uncertainty: str = ""
    evidence_needed: str = ""
    statement_classes: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the explanation block."""
        return {
            "representation_rationale": self.representation_rationale,
            "solver_rationale": self.solver_rationale,
            "applied_constraints": list(self.applied_constraints),
            "validation_summary": self.validation_summary,
            "remaining_uncertainty": self.remaining_uncertainty,
            "evidence_needed": self.evidence_needed,
            "statement_classes": dict(self.statement_classes),
        }


@dataclass(frozen=True, slots=True)
class ScienceConfig:
    """Configuration for the scientific reasoning pipeline."""

    collocation_points: int = 61
    time_steps: int = 240
    refinement_levels: int = 3
    convergence_tolerance: float = 1.0e-3
    conservation_tolerance: float = 1.0e-3
    baseline_tolerance: float = 5.0e-2
    ensemble_members: int = 32
    pin_basis_size: int = 8
    pin_max_iterations: int = 200
    random_seed: int = 2026
    fail_safe_mode: bool = True
    resource_budget_cpu: float = 4.0
    resource_budget_memory_mb: float = 2048.0
    resource_budget_time_seconds: float = 60.0
    resource_budget_gpu: float = 0.0

    def __post_init__(self) -> None:
        """Reject non-positive counts, tolerances, and budgets."""
        positive_ints = (
            "collocation_points", "time_steps", "refinement_levels",
            "ensemble_members", "pin_basis_size", "pin_max_iterations",
        )
        for name in positive_ints:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        positive_floats = (
            "convergence_tolerance", "conservation_tolerance", "baseline_tolerance",
            "resource_budget_cpu", "resource_budget_memory_mb", "resource_budget_time_seconds",
        )
        for name in positive_floats:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
        if self.resource_budget_gpu < 0.0:
            raise ValueError("resource_budget_gpu must be non-negative")


@dataclass(frozen=True, slots=True)
class ScienceRequest:
    """A natural-language scientific request with optional structured facts."""

    question: str
    facts: Mapping[str, str] = field(default_factory=dict)
    domain_hint: str | None = None
    source_reference: str = "unspecified"
    request_id: str = "science_request"

    def __post_init__(self) -> None:
        """The request must be a non-empty question."""
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("science request requires a non-empty question")
        if not isinstance(self.facts, Mapping):
            raise TypeError("facts must be a mapping of variable names to '<value> <unit>' strings")


@dataclass(slots=True)
class ScienceResult:
    """Terminal result of one run through the 52-step scientific reasoning loop.

    Carries the typed outcome, the structured task, per-subtask predictions, validation
    and uncertainty records, provenance reference, full stage and operation traces, and
    diagnostics with an explicit ``valid`` flag.
    """

    request_id: str
    question: str
    outcome: OutcomeKind
    task: dict[str, Any] | None
    domain_decision: dict[str, Any] | None
    subtask_results: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    validation_reports: list[dict[str, Any]] = field(default_factory=list)
    uncertainty_reports: list[dict[str, Any]] = field(default_factory=list)
    result_status: str = ""
    status_history: list[dict[str, Any]] = field(default_factory=list)
    explanation: dict[str, Any] = field(default_factory=dict)
    knowledge_update: dict[str, Any] | None = None
    workspace_publications: list[dict[str, Any]] = field(default_factory=list)
    provenance_ref: dict[str, Any] | None = None
    stage_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    stage_trace: list[dict[str, Any]] = field(default_factory=list)
    operation_trace: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly view of the full result with a schema tag."""
        payload = {
            "schema": "ore.science.52_step.v1",
            "request_id": self.request_id,
            "question": self.question,
            "outcome": self.outcome.value,
            "task": self.task,
            "domain_decision": self.domain_decision,
            "subtask_results": self.subtask_results,
            "predictions": self.predictions,
            "validation_reports": self.validation_reports,
            "uncertainty_reports": self.uncertainty_reports,
            "result_status": self.result_status,
            "status_history": self.status_history,
            "explanation": self.explanation,
            "knowledge_update": self.knowledge_update,
            "workspace_publications": self.workspace_publications,
            "provenance_ref": self.provenance_ref,
            "stage_outputs": self.stage_outputs,
            "stage_trace": self.stage_trace,
            "operation_trace": self.operation_trace,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        }
        return _jsonable(payload)


def _jsonable(value: Any) -> Any:
    """Recursively convert a payload into JSON-safe primitives."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, int):
        return value
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "shape"):
        return {"kind": "array", "shape": list(value.shape)}
    return str(value)
