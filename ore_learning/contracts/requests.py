"""Task, capability, and inference request contracts (Point 14, Steps 1-5).

The Main Brain Coordinator hands the RLMS a typed task record; the planner decides
whether learned inference is necessary at all (Step 2), expresses the need as
capabilities rather than model names (Step 3), decomposes multimodal work at
verifiable boundaries (Step 4), and attaches an explicit evidence and authority
policy (Step 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

from .manifests import Modality, SchemaDescriptor, canonical_payload_digest

__all__ = (
    "ApprovalPolicy",
    "AuthorityPolicy",
    "Capability",
    "CapabilityRequest",
    "CognitiveTaskRecord",
    "EvidenceRequirements",
    "InferenceRequest",
    "RiskClass",
    "TaskStageContract",
)


def _nonempty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


class RiskClass(str, Enum):
    """Task risk classes that gate policy and approval requirements (Step 5)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Capability(str, Enum):
    """Requested operations expressed as capabilities, never model names (Step 3)."""

    IMAGE_FEATURE_EXTRACTION = "image_feature_extraction"
    IMAGE_CAPTIONING = "image_captioning"
    SPEECH_TRANSCRIPTION = "speech_transcription"
    AUDIO_EVENT_DETECTION = "audio_event_detection"
    VIDEO_EVENT_DETECTION = "video_event_detection"
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_GENERATION = "text_generation"
    SPATIAL_GEOMETRY = "spatial_geometry"
    REGRESSION = "regression"
    RANKING = "ranking"
    CANDIDATE_GENERATION = "candidate_generation"
    INTERMEDIATE_STEP_ASSESSMENT = "intermediate_step_assessment"


class ApprovalPolicy(str, Enum):
    """Who must approve consequential state changes (Step 5, Step 55)."""

    AUTOMATIC = "automatic"
    COORDINATOR = "coordinator"
    HUMAN = "human"


@dataclass(frozen=True, slots=True)
class EvidenceRequirements:
    """Which outputs are proposals and which checks apply (Step 5)."""

    deterministic_checks: tuple[str, ...] = ()
    requires_formal_proof: bool = False
    requires_human_approval: bool = False
    explanation_required: bool = False
    allow_model_initiated_actions: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"deterministic_checks": list(self.deterministic_checks),
                "requires_formal_proof": self.requires_formal_proof,
                "requires_human_approval": self.requires_human_approval,
                "explanation_required": self.explanation_required,
                "allow_model_initiated_actions": self.allow_model_initiated_actions}


@dataclass(frozen=True, slots=True)
class AuthorityPolicy:
    """Step 5 -- the authority boundary in force for this task.

    Learned models may propose; deterministic checks gate verification; formal proof
    and human approval gate durable state changes.  ``allow_model_initiated_actions``
    is refused for high-risk classes regardless of the flag.
    """

    risk_class: RiskClass = RiskClass.LOW
    output_kind: str = "proposal"
    approval_policy: ApprovalPolicy = ApprovalPolicy.COORDINATOR
    evidence: EvidenceRequirements = field(default_factory=EvidenceRequirements)
    notes: str = ""

    def __post_init__(self) -> None:
        """High-risk tasks can never authorize model-initiated actions."""
        if self.output_kind not in ("proposal", "reversible_action", "excluded"):
            raise ValueError("output_kind must be proposal/reversible_action/excluded")
        if (self.risk_class in (RiskClass.HIGH, RiskClass.CRITICAL)
                and self.evidence.allow_model_initiated_actions):
            raise ValueError("high-risk tasks cannot allow model-initiated actions")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"risk_class": self.risk_class.value,
                "output_kind": self.output_kind,
                "approval_policy": self.approval_policy.value,
                "evidence": self.evidence.to_dict(), "notes": self.notes}


@dataclass(frozen=True, slots=True)
class CognitiveTaskRecord:
    """Step 1 -- the normalized, versioned task record preserved with its origin."""

    task_id: str
    goal: str
    original_request: str
    source_references: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    terminology_note: str = "normalized through USG; ambiguity preserved, not discarded"
    record_version: str = "1.0.0"

    def __post_init__(self) -> None:
        """Identity and goal are mandatory."""
        object.__setattr__(self, "task_id", _nonempty("task_id", self.task_id))
        object.__setattr__(self, "goal", _nonempty("goal", self.goal))
        object.__setattr__(self, "original_request",
                           _nonempty("original_request", self.original_request))

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"task_id": self.task_id, "goal": self.goal,
                "original_request": self.original_request,
                "source_references": list(self.source_references),
                "constraints": list(self.constraints),
                "terminology_note": self.terminology_note,
                "record_version": self.record_version}


@dataclass(frozen=True, slots=True)
class CapabilityRequest:
    """Step 3 -- the requested operation expressed as typed capabilities."""

    task_id: str
    capabilities: tuple[Capability, ...]
    modality: Modality
    output_contract: SchemaDescriptor
    accuracy_target: float = 0.8
    latency_target_ms: float = 500.0
    risk_class: RiskClass = RiskClass.LOW
    languages: tuple[str, ...] = ("en",)

    def __post_init__(self) -> None:
        """Validate the capability bundle."""
        object.__setattr__(self, "task_id", _nonempty("task_id", self.task_id))
        if not self.capabilities:
            raise ValueError("at least one capability is required")
        for name, value in (("accuracy_target", self.accuracy_target),
                            ("latency_target_ms", self.latency_target_ms)):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 < self.accuracy_target <= 1.0:
            raise ValueError("accuracy_target must be a fraction in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"task_id": self.task_id,
                "capabilities": [item.value for item in self.capabilities],
                "modality": self.modality.value,
                "output_contract": self.output_contract.to_dict(),
                "accuracy_target": self.accuracy_target,
                "latency_target_ms": self.latency_target_ms,
                "risk_class": self.risk_class.value,
                "languages": list(self.languages)}


@dataclass(frozen=True, slots=True)
class TaskStageContract:
    """Step 4 -- one stage of a task decomposition at a verifiable boundary."""

    stage_id: str
    stage_kind: str
    description: str
    input_contract: SchemaDescriptor
    output_contract: SchemaDescriptor
    provenance_edge: str
    upstream_stage: str = ""

    def __post_init__(self) -> None:
        """Stage identity is mandatory."""
        object.__setattr__(self, "stage_id", _nonempty("stage_id", self.stage_id))
        object.__setattr__(self, "stage_kind", _nonempty("stage_kind", self.stage_kind))
        object.__setattr__(self, "description",
                           _nonempty("description", self.description))
        object.__setattr__(self, "provenance_edge",
                           _nonempty("provenance_edge", self.provenance_edge))
        if self.stage_kind not in ("perception", "semantic_normalization",
                                   "hypothesis_generation", "symbolic_validation"):
            raise ValueError(f"unsupported stage kind {self.stage_kind!r}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"stage_id": self.stage_id, "stage_kind": self.stage_kind,
                "description": self.description,
                "input_contract": self.input_contract.to_dict(),
                "output_contract": self.output_contract.to_dict(),
                "provenance_edge": self.provenance_edge,
                "upstream_stage": self.upstream_stage}


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """Section 5.1 -- typed inputs, budget, deadline, and permissions for one call."""

    task_id: str
    model_id: str
    inputs: tuple[dict[str, Any], ...]
    budget_ms: float = 1000.0
    deadline_seconds: float = 5.0
    permissions: tuple[str, ...] = ("inference:local",)
    cancellation: Any = None
    request_digest: str = ""

    def __post_init__(self) -> None:
        """Validate the call envelope."""
        object.__setattr__(self, "task_id", _nonempty("task_id", self.task_id))
        object.__setattr__(self, "model_id", _nonempty("model_id", self.model_id))
        if not self.inputs:
            raise ValueError("at least one typed input is required")
        for name, value in (("budget_ms", self.budget_ms),
                            ("deadline_seconds", self.deadline_seconds)):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not self.request_digest:
            object.__setattr__(self, "request_digest", canonical_payload_digest(
                {"task_id": self.task_id, "model_id": self.model_id,
                 "inputs": list(self.inputs)}))

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view (cancellation objects are referenced by type name only)."""
        return {"task_id": self.task_id, "model_id": self.model_id,
                "n_inputs": len(self.inputs), "budget_ms": self.budget_ms,
                "deadline_seconds": self.deadline_seconds,
                "permissions": list(self.permissions),
                "request_digest": self.request_digest,
                "cancellation": type(self.cancellation).__name__ if self.cancellation else None}
