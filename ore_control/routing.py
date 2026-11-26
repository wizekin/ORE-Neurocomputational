"""Explicit/learned processing coordination for the control layer (Phase F).

Implements Point 02 Steps 26-28: processing-path classification, learned
component suitability checks through a replaceable registry, and the
proposal-only publication boundary for learned outputs. Learned outputs are
proposals with provenance and uncalibrated confidence; they never mutate
authoritative USG knowledge and always pass validation before publication.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .contracts import RiskClass, sha256_json

LEARNED_PROPOSAL_NOTE = (
    "Learned-component output is a proposal with provenance, not authoritative "
    "knowledge; confidence is uncalibrated and must not be read as a probability"
)


class ProcessingPath:
    """Processing-path classification labels (Step 26)."""

    EXPLICIT_SYMBOLIC = "EXPLICIT_SYMBOLIC"
    LEARNED_STATISTICAL = "LEARNED_STATISTICAL"
    HYBRID = "HYBRID"


@dataclass(frozen=True, slots=True)
class LearnedComponentRecord:
    """Replaceable learned-component declaration (Step 27)."""

    component_id: str
    model_version: str
    input_schema_id: str
    output_schema_id: str
    supported_tasks: tuple[str, ...] = ()
    resource_requirements: Mapping[str, float] = field(default_factory=dict)
    evaluation_provenance: Mapping[str, Any] = field(default_factory=dict)
    calibration_status: str = "uncalibrated"
    limitations: tuple[str, ...] = ()
    restrictions: tuple[str, ...] = ()
    risk_class: RiskClass = RiskClass.LOW
    ranker: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        """Validate the learned-component declaration."""
        if not self.component_id.strip() or not self.model_version.strip():
            raise ValueError("learned component identity and version are required")
        if not self.input_schema_id.strip() or not self.output_schema_id.strip():
            raise ValueError("learned component schemas are required")
        object.__setattr__(self, "supported_tasks", tuple(self.supported_tasks))
        object.__setattr__(self, "resource_requirements", dict(self.resource_requirements))
        object.__setattr__(self, "evaluation_provenance", dict(self.evaluation_provenance))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(self, "restrictions", tuple(self.restrictions))

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable declaration (handler excluded)."""
        return {
            "component_id": self.component_id,
            "model_version": self.model_version,
            "input_schema_id": self.input_schema_id,
            "output_schema_id": self.output_schema_id,
            "supported_tasks": list(self.supported_tasks),
            "resource_requirements": dict(self.resource_requirements),
            "evaluation_provenance": dict(self.evaluation_provenance),
            "calibration_status": self.calibration_status,
            "limitations": list(self.limitations),
            "restrictions": list(self.restrictions),
            "risk_class": self.risk_class.value,
        }


@dataclass(frozen=True, slots=True)
class LearnedComponentCheck:
    """Suitability verdict for one learned component (Step 27)."""

    component_id: str
    suitable: bool
    reasons: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable check record."""
        return {
            "component_id": self.component_id,
            "suitable": self.suitable,
            "reasons": dict(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class LearnedProposal:
    """One learned output wrapped as a non-authoritative proposal (Step 28)."""

    component_id: str
    model_version: str
    payload: Mapping[str, Any]
    provenance: Mapping[str, Any]
    confidence: float | None
    calibration_status: str
    requires_validation: bool = True

    def fingerprint(self) -> str:
        """Return the stable fingerprint of the proposal content."""
        return sha256_json({
            "component_id": self.component_id,
            "model_version": self.model_version,
            "payload": dict(self.payload),
        })

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable proposal record."""
        return {
            "component_id": self.component_id,
            "model_version": self.model_version,
            "payload": dict(self.payload),
            "provenance": dict(self.provenance),
            "confidence": self.confidence,
            "calibration_status": self.calibration_status,
            "requires_validation": bool(self.requires_validation),
            "note": LEARNED_PROPOSAL_NOTE,
            "fingerprint": self.fingerprint(),
        }


class LearnedComponentRegistry:
    """Registry of replaceable learned components (Step 27)."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._components: dict[str, LearnedComponentRecord] = {}

    def register(self, record: LearnedComponentRecord) -> None:
        """Register one learned component by stable identity."""
        if record.component_id in self._components:
            raise ValueError(
                f"learned component already registered: {record.component_id}"
            )
        self._components[record.component_id] = record

    def unregister(self, component_id: str) -> bool:
        """Withdraw one component; withdrawn components fail suitability."""
        return self._components.pop(component_id, None) is not None

    def get(self, component_id: str) -> LearnedComponentRecord | None:
        """Return one component record if registered."""
        return self._components.get(component_id)

    def records(self) -> tuple[LearnedComponentRecord, ...]:
        """Return all component records in stable ID order."""
        return tuple(
            self._components[key] for key in sorted(self._components)
        )

    def check(
        self,
        component_id: str,
        *,
        task_type: str = "sentence_understanding",
        input_schema_id: str = "control.operator.input/1.0",
        output_schema_id: str = "control.operator.output/1.0",
        available_memory_mb: float | None = None,
    ) -> LearnedComponentCheck:
        """Verify learned-component suitability before invocation (Step 27).

        Model identity/version, input/output schemas, supported task, resource
        requirements, evaluation provenance, calibration status, limitations,
        and restrictions are all checked; any failure makes the component
        unsuitable with a machine-readable reason.
        """
        record = self._components.get(component_id)
        if record is None:
            return LearnedComponentCheck(
                component_id, False, {"registration": "component_not_registered"},
            )
        reasons: dict[str, str] = {}
        if not record.model_version.strip():
            reasons["model_version"] = "missing_model_version"
        if record.input_schema_id != input_schema_id:
            reasons["input_schema"] = (
                f"schema_mismatch:{record.input_schema_id}!={input_schema_id}"
            )
        if record.output_schema_id != output_schema_id:
            reasons["output_schema"] = (
                f"schema_mismatch:{record.output_schema_id}!={output_schema_id}"
            )
        if record.supported_tasks and task_type not in record.supported_tasks:
            reasons["supported_tasks"] = f"task_not_supported:{task_type}"
        if available_memory_mb is not None:
            required = float(record.resource_requirements.get("memory_mb", 0.0))
            if required > available_memory_mb:
                reasons["resources"] = f"memory_requirement_exceeds_budget:{required}"
        if not record.evaluation_provenance.get("evaluation_recorded", False):
            reasons["evaluation"] = "evaluation_provenance_missing"
        if record.ranker is None:
            reasons["handler"] = "ranker_unavailable"
        for restriction in record.restrictions:
            reasons[f"restriction:{restriction}"] = "execution_restriction_active"
        return LearnedComponentCheck(
            component_id, not reasons, reasons,
        )


# ------------------------------------------------------------- reference model


def _stable_pseudo_frequency(sense_id: str) -> float:
    """Derive a deterministic pseudo-frequency for one sense identifier.

    The value is a stable hash-derived score in [0, 1]; it exists only to
    make the reference ranker deterministic and is explicitly not a corpus
    frequency estimate.
    """
    digest = sha256_json({"sense_id": sense_id})
    return int(digest[:8], 16) / 0xFFFFFFFF


class SenseRanker:
    """Deterministic reference learned component (toy model, Step 27 demo).

    Ranks sense candidates by a stable pseudo-frequency heuristic. It is a
    deliberately toy statistical component: it validates the routing and
    proposal machinery and is labelled uncalibrated everywhere.
    """

    component_id = "control.toy.sense_ranker"
    model_version = "0.1.0"

    def rank(
        self, term: str, senses: Sequence[tuple[str, str]],
    ) -> tuple[tuple[int, float, str], ...]:
        """Return (rank, score, sense_id) tuples ordered deterministically."""
        scored = [
            (index, _stable_pseudo_frequency(sense_id), sense_id)
            for index, (sense_id, _) in enumerate(senses)
        ]
        scored.sort(key=lambda item: (-item[1], item[2]))
        return tuple(
            (rank, score, sense_id) for rank, (_, score, sense_id) in enumerate(scored)
        )


def default_learned_components() -> LearnedComponentRegistry:
    """Return the registry with the reference toy sense ranker (Step 27)."""
    registry = LearnedComponentRegistry()
    ranker = SenseRanker()
    registry.register(LearnedComponentRecord(
        component_id=ranker.component_id,
        model_version=ranker.model_version,
        input_schema_id="control.operator.input/1.0",
        output_schema_id="control.operator.output/1.0",
        supported_tasks=("sentence_understanding",),
        resource_requirements={"memory_mb": 1.0, "compute": 0.01},
        evaluation_provenance={
            "evaluation_recorded": True,
            "evaluation": "reference smoke evaluation only; not a benchmark",
            "provenance": "ore_control.routing.default_learned_components",
        },
        calibration_status="uncalibrated",
        limitations=(
            "deterministic toy heuristic; no corpus evidence; ranking is a proposal",
        ),
        restrictions=(),
        risk_class=RiskClass.LOW,
        ranker=ranker.rank,
    ))
    return registry


def classify_processing_path(
    operator_processing_path: str,
    *,
    learned_check: LearnedComponentCheck | None = None,
) -> str:
    """Classify the processing path for one action (Step 26).

    USG rules, typed predicates, and deterministic constraints stay explicit;
    a HYBRID operator with an unsuitable learned component degrades to
    explicit symbolic processing rather than bypassing the suitability check.
    """
    if operator_processing_path == ProcessingPath.EXPLICIT_SYMBOLIC:
        return ProcessingPath.EXPLICIT_SYMBOLIC
    if operator_processing_path == ProcessingPath.LEARNED_STATISTICAL:
        if learned_check is not None and not learned_check.suitable:
            return ProcessingPath.EXPLICIT_SYMBOLIC
        return ProcessingPath.LEARNED_STATISTICAL
    if operator_processing_path == ProcessingPath.HYBRID:
        if learned_check is not None and not learned_check.suitable:
            return ProcessingPath.EXPLICIT_SYMBOLIC
        return ProcessingPath.HYBRID
    return ProcessingPath.EXPLICIT_SYMBOLIC


def validate_learned_component_declarations(
    records: Iterable[LearnedComponentRecord],
) -> tuple[Mapping[str, Any], ...]:
    """Return machine-readable validation issues for declarations (tests)."""
    issues: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if record.component_id in seen:
            issues.append({
                "component_id": record.component_id,
                "issue": "duplicate_component_id",
            })
        seen.add(record.component_id)
        if record.calibration_status not in {
            "uncalibrated", "calibrated_on_labeled_set",
        }:
            issues.append({
                "component_id": record.component_id,
                "issue": "unknown_calibration_status",
            })
    return tuple(issues)


__all__ = [
    "LEARNED_PROPOSAL_NOTE",
    "LearnedComponentCheck",
    "LearnedComponentRecord",
    "LearnedComponentRegistry",
    "LearnedProposal",
    "ProcessingPath",
    "SenseRanker",
    "classify_processing_path",
    "default_learned_components",
    "validate_learned_component_declarations",
]
