"""Typed state, action, transition, reward, and decision schemas for ore_rl.

Point 10, Step 3: versioned schemas for OREState, OREAction, ORETransition,
OREReward, and PolicyDecision. Unknown schema versions are rejected instead of
being interpreted permissively (acceptance test 8.1).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


RL_SCHEMA_VERSION = "1.0.0"
_KNOWN_SCHEMA_VERSIONS = frozenset({"1.0.0"})


class UnknownSchemaVersionError(ValueError):
    """Raised when a payload declares a schema version this package cannot interpret."""


def require_schema_version(version: str, *, context: str = "payload") -> str:
    """Reject unknown schema versions rather than interpreting them permissively."""
    if version not in _KNOWN_SCHEMA_VERSIONS:
        raise UnknownSchemaVersionError(
            f"{context} declares unsupported schema version {version!r}; "
            f"known versions: {sorted(_KNOWN_SCHEMA_VERSIONS)}"
        )
    return version


def canonical_json(value: Any) -> str:
    """Deterministic JSON serialization used for hashing and integrity checks."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_fallback)


def _json_fallback(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return list(value)
    return str(value)


def stable_hash(payload: Any) -> str:
    """Content SHA-256 over the canonical JSON encoding of ``payload``."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class ActionType(str, Enum):
    """Named cognitive operations and control decisions (Step 15, Step 18)."""

    RUN_MODULE = "RUN_MODULE"
    RETRIEVE_KNOWLEDGE = "RETRIEVE_KNOWLEDGE"
    GENERATE_HYPOTHESES = "GENERATE_HYPOTHESES"
    CAUSAL_ANALYSIS = "CAUSAL_ANALYSIS"
    VERIFY = "VERIFY"
    ADJUST_DEPTH = "ADJUST_DEPTH"
    CONCURRENT_BATCH = "CONCURRENT_BATCH"
    STOP_SUCCESS = "STOP_SUCCESS"
    STOP_INSUFFICIENT_EVIDENCE = "STOP_INSUFFICIENT_EVIDENCE"
    DEFER_RESOURCE_LIMIT = "DEFER_RESOURCE_LIMIT"
    REQUEST_HUMAN_REVIEW = "REQUEST_HUMAN_REVIEW"


class OutcomeStatus(str, Enum):
    """Distinct execution outcomes (Step 20, acceptance test 8.1)."""

    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    TIMEOUT = "TIMEOUT"
    EXCEPTION = "EXCEPTION"
    CANCELLED = "CANCELLED"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    TERMINAL = "TERMINAL"


OUTCOME_STATUSES: tuple[OutcomeStatus, ...] = tuple(OutcomeStatus)


class RiskLevel(str, Enum):
    """Risk classification used by the limited-authority controller (Step 51)."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PolicyAuthority(str, Enum):
    """What a policy may do at a decision point (Step 1, Step 51)."""

    SHADOW = "SHADOW"
    PROPOSAL = "PROPOSAL"
    LIMITED_CONTROL = "LIMITED_CONTROL"


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Declarative parameter validation for actions (Step 16)."""

    name: str
    kind: str  # "int" | "float" | "str" | "bool" | "sequence"
    required: bool = True
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    description: str = ""

    def validate(self, value: Any) -> str | None:
        """Return an error message for an invalid value, or None when valid."""
        if value is None:
            return None if not self.required else f"parameter {self.name!r} is required"
        if self.kind == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                return f"parameter {self.name!r} must be an int"
        elif self.kind == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return f"parameter {self.name!r} must be numeric"
        elif self.kind == "str":
            if not isinstance(value, str):
                return f"parameter {self.name!r} must be a string"
        elif self.kind == "bool":
            if not isinstance(value, bool):
                return f"parameter {self.name!r} must be a bool"
        elif self.kind == "sequence":
            if isinstance(value, (str, bytes)) or not isinstance(
                value, (tuple, list)
            ):
                return f"parameter {self.name!r} must be a sequence"
        if self.choices and value not in self.choices:
            return f"parameter {self.name!r} must be one of {list(self.choices)}"
        if self.minimum is not None and isinstance(value, (int, float)):
            if value < self.minimum:
                return f"parameter {self.name!r} must be >= {self.minimum}"
        if self.maximum is not None and isinstance(value, (int, float)):
            if value > self.maximum:
                return f"parameter {self.name!r} must be <= {self.maximum}"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "required": self.required,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "choices": list(self.choices),
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    """Pre-execution cost estimate in ORE resource units (latency ms, CPU, memory MB)."""

    latency_ms: float
    cpu_cost: float
    memory_mb: float

    def to_dict(self) -> dict[str, Any]:
        return {"latency_ms": self.latency_ms, "cpu_cost": self.cpu_cost, "memory_mb": self.memory_mb}


@dataclass(frozen=True, slots=True)
class ActionSpecRecord:
    """Immutable action definition registered in the action catalog (Step 15-16)."""

    action_id: str
    action_type: ActionType
    description: str
    parameters: tuple[ParameterSpec, ...] = ()
    target_module: str | None = None
    capability: str | None = None
    prerequisite_capabilities: tuple[str, ...] = ()
    default_resource_estimate: ResourceEstimate = ResourceEstimate(0.0, 0.0, 0.0)
    risk: RiskLevel = RiskLevel.LOW
    depth_change: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "description": self.description,
            "parameters": [item.to_dict() for item in self.parameters],
            "target_module": self.target_module,
            "capability": self.capability,
            "prerequisite_capabilities": list(self.prerequisite_capabilities),
            "default_resource_estimate": self.default_resource_estimate.to_dict(),
            "risk": self.risk.value,
            "depth_change": self.depth_change,
        }


@dataclass(frozen=True, slots=True)
class OREAction:
    """A parameterized, contract-bound action proposal (Step 3, Step 16)."""

    schema_version: str = RL_SCHEMA_VERSION
    action_id: str = ""
    action_type: ActionType = ActionType.RUN_MODULE
    parameters: Mapping[str, Any] = field(default_factory=dict)
    target_module: str | None = None
    capability: str | None = None
    resource_estimate: ResourceEstimate = ResourceEstimate(0.0, 0.0, 0.0)
    risk: RiskLevel = RiskLevel.LOW
    step_budget_ms: float | None = None

    def __post_init__(self) -> None:
        require_schema_version(self.schema_version, context="OREAction")
        if not self.action_id:
            raise ValueError("action_id must be non-empty")

    @property
    def stable_id(self) -> str:
        """Identity used for indexing and logging: action id plus parameter signature."""
        signature = stable_hash({"parameters": dict(self.parameters)})
        return f"{self.action_id}:{signature[:12]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "parameters": dict(self.parameters),
            "target_module": self.target_module,
            "capability": self.capability,
            "resource_estimate": self.resource_estimate.to_dict(),
            "risk": self.risk.value,
            "step_budget_ms": self.step_budget_ms,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OREAction:
        require_schema_version(data["schema_version"], context="OREAction")
        estimate = data.get("resource_estimate") or {}
        return cls(
            schema_version=data["schema_version"],
            action_id=data["action_id"],
            action_type=ActionType(data["action_type"]),
            parameters=dict(data.get("parameters", {})),
            target_module=data.get("target_module"),
            capability=data.get("capability"),
            resource_estimate=ResourceEstimate(
                float(estimate.get("latency_ms", 0.0)),
                float(estimate.get("cpu_cost", 0.0)),
                float(estimate.get("memory_mb", 0.0)),
            ),
            risk=RiskLevel(data.get("risk", RiskLevel.LOW.value)),
            step_budget_ms=data.get("step_budget_ms"),
        )


@dataclass(frozen=True, slots=True)
class OREState:
    """Typed, schema-validated observation bound to a task and workspace revision (Steps 3, 7-13)."""

    schema_version: str = RL_SCHEMA_VERSION
    state_id: str = ""
    task_id: str = ""
    task_family: str = ""
    workspace_id: str = ""
    workspace_revision: int = 0
    episode_id: str = ""
    step_index: int = 0
    encoder_version: str = ""
    feature_names: tuple[str, ...] = ()
    features: Mapping[str, float] = field(default_factory=dict)
    missing_features: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_schema_version(self.schema_version, context="OREState")
        if not self.state_id:
            raise ValueError("state_id must be non-empty")
        if not self.encoder_version:
            raise ValueError("encoder_version must be recorded on every state")

    def vector(self, spec_names: Sequence[str] | None = None) -> "tuple[float, ...]":
        """Return features as an ordered tuple; missing entries use explicit NaN markers."""
        import math

        names = tuple(spec_names) if spec_names is not None else self.feature_names
        return tuple(
            float(self.features.get(name, math.nan)) if name not in self.missing_features
            else math.nan
            for name in names
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state_id": self.state_id,
            "task_id": self.task_id,
            "task_family": self.task_family,
            "workspace_id": self.workspace_id,
            "workspace_revision": self.workspace_revision,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "encoder_version": self.encoder_version,
            "feature_names": list(self.feature_names),
            "features": dict(self.features),
            "missing_features": list(self.missing_features),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OREState:
        require_schema_version(data["schema_version"], context="OREState")
        return cls(
            schema_version=data["schema_version"],
            state_id=data["state_id"],
            task_id=data.get("task_id", ""),
            task_family=data.get("task_family", ""),
            workspace_id=data.get("workspace_id", ""),
            workspace_revision=int(data.get("workspace_revision", 0)),
            episode_id=data.get("episode_id", ""),
            step_index=int(data.get("step_index", 0)),
            encoder_version=data["encoder_version"],
            feature_names=tuple(data.get("feature_names", ())),
            features=dict(data.get("features", {})),
            missing_features=tuple(data.get("missing_features", ())),
            provenance=dict(data.get("provenance", {})),
        )


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    """Measured execution result with distinct statuses (Step 20)."""

    status: OutcomeStatus
    measured_latency_ms: float = 0.0
    measured_cpu_cost: float = 0.0
    measured_memory_mb: float = 0.0
    artifact_refs: tuple[str, ...] = ()
    findings_published: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "measured_latency_ms": self.measured_latency_ms,
            "measured_cpu_cost": self.measured_cpu_cost,
            "measured_memory_mb": self.measured_memory_mb,
            "artifact_refs": list(self.artifact_refs),
            "findings_published": self.findings_published,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class OREReward:
    """Composite reward computed from independently evaluated outcomes (Steps 24-25).

    ``verified`` must be True for a reward to enter the trusted training set
    (acceptance test 8.1: unverified reward evidence cannot be trained on).
    """

    schema_version: str = RL_SCHEMA_VERSION
    quality_delta: float = 0.0
    evidence_delta: float = 0.0
    resource_cost: float = 0.0
    latency_cost: float = 0.0
    failure_cost: float = 0.0
    total: float = 0.0
    weights_version: str = ""
    reward_spec_fingerprint: str = ""
    evaluator_id: str = ""
    evaluator_version: str = ""
    attribution_method: str = "STEP"
    attribution_uncertainty: float = 0.0
    terminal: bool = False
    verified: bool = False
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_schema_version(self.schema_version, context="OREReward")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "quality_delta": self.quality_delta,
            "evidence_delta": self.evidence_delta,
            "resource_cost": self.resource_cost,
            "latency_cost": self.latency_cost,
            "failure_cost": self.failure_cost,
            "total": self.total,
            "weights_version": self.weights_version,
            "reward_spec_fingerprint": self.reward_spec_fingerprint,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "attribution_method": self.attribution_method,
            "attribution_uncertainty": self.attribution_uncertainty,
            "terminal": self.terminal,
            "verified": self.verified,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OREReward:
        require_schema_version(data["schema_version"], context="OREReward")
        return cls(
            schema_version=data["schema_version"],
            quality_delta=float(data.get("quality_delta", 0.0)),
            evidence_delta=float(data.get("evidence_delta", 0.0)),
            resource_cost=float(data.get("resource_cost", 0.0)),
            latency_cost=float(data.get("latency_cost", 0.0)),
            failure_cost=float(data.get("failure_cost", 0.0)),
            total=float(data.get("total", 0.0)),
            weights_version=data.get("weights_version", ""),
            reward_spec_fingerprint=data.get("reward_spec_fingerprint", ""),
            evaluator_id=data.get("evaluator_id", ""),
            evaluator_version=data.get("evaluator_version", ""),
            attribution_method=data.get("attribution_method", "STEP"),
            attribution_uncertainty=float(data.get("attribution_uncertainty", 0.0)),
            terminal=bool(data.get("terminal", False)),
            verified=bool(data.get("verified", False)),
            evidence_refs=tuple(data.get("evidence_refs", ())),
        )


@dataclass(frozen=True, slots=True)
class ORETransition:
    """One recorded state-action-outcome-next-state step (Steps 3, 29-30)."""

    schema_version: str = RL_SCHEMA_VERSION
    episode_id: str = ""
    step_index: int = 0
    state_id: str = ""
    action: OREAction | None = None
    outcome: ActionOutcome | None = None
    next_state_id: str | None = None
    terminal: bool = False
    simulated: bool = False
    eligible_action_ids: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_schema_version(self.schema_version, context="ORETransition")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "state_id": self.state_id,
            "action": self.action.to_dict() if self.action else None,
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "next_state_id": self.next_state_id,
            "terminal": self.terminal,
            "simulated": self.simulated,
            "eligible_action_ids": list(self.eligible_action_ids),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class RankedProposal:
    """One ranked action suggestion with uncertainty metadata (Step 1)."""

    action_id: str
    score: float
    uncertainty: float

    def to_dict(self) -> dict[str, Any]:
        return {"action_id": self.action_id, "score": self.score, "uncertainty": self.uncertainty}


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Proposal bundle returned by the policy service; never an execution order (Step 1)."""

    schema_version: str = RL_SCHEMA_VERSION
    policy_id: str = ""
    policy_version: str = ""
    ranked: tuple[RankedProposal, ...] = ()
    chosen_action_id: str = ""
    propensity: float = 1.0
    uncertainty: float = 0.0
    authority: PolicyAuthority = PolicyAuthority.PROPOSAL
    wall_time_ms: float = 0.0
    fallback_used: bool = False
    rationale: str = ""

    def __post_init__(self) -> None:
        require_schema_version(self.schema_version, context="PolicyDecision")
        if not self.policy_id or not self.policy_version:
            raise ValueError("policy_id and policy_version are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "ranked": [item.to_dict() for item in self.ranked],
            "chosen_action_id": self.chosen_action_id,
            "propensity": self.propensity,
            "uncertainty": self.uncertainty,
            "authority": self.authority.value,
            "wall_time_ms": self.wall_time_ms,
            "fallback_used": self.fallback_used,
            "rationale": self.rationale,
        }
