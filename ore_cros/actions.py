"""Typed resource actions and the runtime control state they mutate.

These types form the narrow interface between the MAPE-K planner (Step 11),
the hard-constraint filter (Step 13) and the controlled execution manager
(Step 14). Every action carries typed, validated parameters, predicted
effects, cost, rollback condition and full policy provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping

from .resource_model import (
    CLASS_ORDER, ResourceDemand, ResourceKind, WorkloadClass,
)


class ActionType(str, Enum):
    """The closed set of resource actions CROS may propose or execute."""

    KEEP_CURRENT_POLICY = "KEEP_CURRENT_POLICY"
    ADJUST_ADMISSION = "ADJUST_ADMISSION"
    ADJUST_WORKER_COUNT = "ADJUST_WORKER_COUNT"
    ADJUST_CPU_WEIGHTS = "ADJUST_CPU_WEIGHTS"
    DEFER_BACKGROUND = "DEFER_BACKGROUND"
    CHANGE_CACHE_TIER = "CHANGE_CACHE_TIER"
    SELECT_WORKER = "SELECT_WORKER"
    DETERMINISTIC_FALLBACK = "DETERMINISTIC_FALLBACK"


class PolicyOrigin(str, Enum):
    """Provenance of a candidate action (spec Step 12)."""

    DETERMINISTIC = "DETERMINISTIC"
    PREDICTIVE = "PREDICTIVE"
    LEARNED = "LEARNED"
    BASELINE = "BASELINE"


#: Workload classes that may legally be deferred by DEFER_BACKGROUND actions.
DEFERRABLE_CLASSES: frozenset[WorkloadClass] = frozenset({
    WorkloadClass.BATCH, WorkloadClass.EXPLORATORY, WorkloadClass.MAINTENANCE,
})


@dataclass(frozen=True)
class ResourceAction:
    """One candidate resource action with typed parameters and provenance."""

    action_id: str
    action_type: ActionType
    parameters: Mapping[str, Any]
    rationale: str
    policy_origin: PolicyOrigin
    policy_version: str
    state_version: int
    predicted_effect: ResourceDemand = field(default_factory=ResourceDemand)
    predicted_objective_delta: float = 0.0
    cost: float = 0.0
    rollback_condition: str = "constraint violation"
    high_impact: bool = False

    def __post_init__(self) -> None:
        """Validate identifiers, provenance and type-specific parameters."""
        if not isinstance(self.action_id, str) or not self.action_id.strip():
            raise ValueError("action_id must be a non-empty string")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError("rationale must be a non-empty string")
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise ValueError("policy_version must be a non-empty string")
        if self.state_version < 0:
            raise ValueError("state_version must be non-negative")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        object.__setattr__(self, "parameters", dict(self.parameters))
        for name in ("predicted_objective_delta", "cost"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Enforce the typed parameter schema for each action type."""
        params = self.parameters
        kind = self.action_type
        if kind is ActionType.ADJUST_WORKER_COUNT:
            target = params.get("target_worker_count")
            if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
                raise ValueError(
                    "ADJUST_WORKER_COUNT requires a positive integer "
                    "target_worker_count"
                )
        elif kind is ActionType.ADJUST_CPU_WEIGHTS:
            weights = params.get("cpu_weights")
            if not isinstance(weights, Mapping) or not weights:
                raise ValueError("ADJUST_CPU_WEIGHTS requires a non-empty cpu_weights mapping")
            for module_id, weight in weights.items():
                if not isinstance(module_id, str) or not module_id.strip():
                    raise ValueError("cpu_weights keys must be non-empty module ids")
                if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
                    raise ValueError(f"cpu weight for {module_id!r} must be finite and positive")
        elif kind is ActionType.ADJUST_ADMISSION:
            enabled = params.get("admission_enabled")
            if not isinstance(enabled, bool):
                raise ValueError("ADJUST_ADMISSION requires a boolean admission_enabled")
            overcommit = params.get("allow_overcommit", False)
            if not isinstance(overcommit, bool):
                raise ValueError("allow_overcommit must be boolean")
        elif kind is ActionType.DEFER_BACKGROUND:
            classes = params.get("defer_classes")
            if not isinstance(classes, (list, tuple)) or not classes:
                raise ValueError("DEFER_BACKGROUND requires a non-empty defer_classes sequence")
            for item in classes:
                if not isinstance(item, WorkloadClass):
                    raise ValueError("defer_classes entries must be WorkloadClass members")
                if item not in DEFERRABLE_CLASSES:
                    raise ValueError(
                        f"{item.value} cannot be deferred; only "
                        f"{sorted(item.value for item in DEFERRABLE_CLASSES)} may defer"
                    )
            defer = params.get("defer", True)
            if not isinstance(defer, bool):
                raise ValueError("defer must be boolean")
        elif kind is ActionType.CHANGE_CACHE_TIER:
            sizes = params.get("tier_sizes")
            if not isinstance(sizes, Mapping):
                raise ValueError("CHANGE_CACHE_TIER requires a tier_sizes mapping")
            for tier, size in sizes.items():
                if tier not in ("hot", "warm", "cold"):
                    raise ValueError(f"unknown cache tier {tier!r}")
                if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                    raise ValueError(f"tier {tier!r} size must be a non-negative integer")
        elif kind is ActionType.SELECT_WORKER:
            task_id = params.get("task_id")
            worker_id = params.get("worker_id")
            for name, value in (("task_id", task_id), ("worker_id", worker_id)):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"SELECT_WORKER requires a non-empty {name}")
        elif kind in (ActionType.KEEP_CURRENT_POLICY, ActionType.DETERMINISTIC_FALLBACK):
            pass

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the action."""
        return {
            "action_id": self.action_id, "action_type": self.action_type.value,
            "parameters": {
                key: (
                    [item.value if isinstance(item, WorkloadClass) else item for item in value]
                    if isinstance(value, (list, tuple)) else
                    value.value if isinstance(value, WorkloadClass) else value
                )
                for key, value in self.parameters.items()
            },
            "rationale": self.rationale,
            "policy_origin": self.policy_origin.value,
            "policy_version": self.policy_version,
            "state_version": self.state_version,
            "predicted_effect": self.predicted_effect.to_dict(),
            "predicted_objective_delta": self.predicted_objective_delta,
            "cost": self.cost,
            "rollback_condition": self.rollback_condition,
            "high_impact": self.high_impact,
        }


def keep_current_policy_action(
    state_version: int, policy_version: str, rationale: str = "no safe action available",
) -> ResourceAction:
    """Return the explicit KEEP_CURRENT_POLICY action (spec section 5)."""
    return ResourceAction(
        action_id="keep-current", action_type=ActionType.KEEP_CURRENT_POLICY,
        parameters={}, rationale=rationale, policy_origin=PolicyOrigin.BASELINE,
        policy_version=policy_version, state_version=state_version,
    )


def deterministic_fallback_action(
    state_version: int, policy_version: str, rationale: str,
) -> ResourceAction:
    """Return the explicit DETERMINISTIC_FALLBACK action (spec Step 52)."""
    return ResourceAction(
        action_id="deterministic-fallback", action_type=ActionType.DETERMINISTIC_FALLBACK,
        parameters={}, rationale=rationale, policy_origin=PolicyOrigin.BASELINE,
        policy_version=policy_version, state_version=state_version,
    )


#: Cache-tier capacity kinds handled by CHANGE_CACHE_TIER.
CACHE_TIER_KINDS: tuple[ResourceKind, ...] = (ResourceKind.RAM, ResourceKind.STORAGE)


@dataclass
class RuntimeControlState:
    """The mutable control state managed exclusively by the execution manager.

    This is the set of knobs the policy engine is allowed to change; nothing
    else in CROS mutates OS or process configuration directly.
    """

    admission_enabled: bool = True
    allow_overcommit: bool = False
    worker_count: int = 4
    cpu_weights: dict[str, float] = field(default_factory=dict)
    deferred_classes: set[WorkloadClass] = field(default_factory=set)
    tier_sizes: dict[str, int] = field(
        default_factory=lambda: {"hot": 64, "warm": 256, "cold": 1024}
    )
    version: int = 0

    def snapshot(self) -> "RuntimeControlState":
        """Return an independent deep copy for checkpointing and rollback."""
        return RuntimeControlState(
            admission_enabled=self.admission_enabled,
            allow_overcommit=self.allow_overcommit,
            worker_count=self.worker_count,
            cpu_weights=dict(self.cpu_weights),
            deferred_classes=set(self.deferred_classes),
            tier_sizes=dict(self.tier_sizes),
            version=self.version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the control state."""
        return {
            "admission_enabled": self.admission_enabled,
            "allow_overcommit": self.allow_overcommit,
            "worker_count": self.worker_count,
            "cpu_weights": dict(self.cpu_weights),
            "deferred_classes": sorted(
                (item.value for item in self.deferred_classes),
                key=lambda value: CLASS_ORDER[WorkloadClass(value)],
            ),
            "tier_sizes": dict(self.tier_sizes),
            "version": self.version,
        }


__all__ = [
    "ActionType", "PolicyOrigin", "ResourceAction", "RuntimeControlState",
    "DEFERRABLE_CLASSES", "keep_current_policy_action", "deterministic_fallback_action",
    "CACHE_TIER_KINDS",
]
