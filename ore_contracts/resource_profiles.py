"""Typed task requirements and auditable outputs for adaptive resource allocation."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Any, Mapping

@dataclass(frozen=True, slots=True)
class CapabilityDemand:
    """Describe the priority, mandatory status, and target support for one capability."""
    capability_id: str
    priority: float = 1.0
    mandatory: bool = True
    target_confidence: float = 0.9

    def __post_init__(self) -> None:
        """Reject incomplete capability IDs and non-finite or out-of-range demands."""
        if not isinstance(self.capability_id, str) or not self.capability_id.strip():
            raise ValueError("capability_id must be a non-empty string")
        if not math.isfinite(float(self.priority)) or not 0.0 <= self.priority <= 1.0:
            raise ValueError("priority must be finite and in [0, 1]")
        if not math.isfinite(float(self.target_confidence)) or not 0.0 <= self.target_confidence <= 1.0:
            raise ValueError("target_confidence must be finite and in [0, 1]")
        if not isinstance(self.mandatory, bool):
            raise TypeError("mandatory must be a bool")

@dataclass(frozen=True, slots=True)
class TaskProfile:
    """Define the capability demands and bounded resource policy for a task family."""
    task_id: str
    task_family: str
    demands: tuple[CapabilityDemand, ...]
    stop_confidence: float = 0.9
    max_module_calls: int = 10
    max_replans: int = 1
    execution_mode: str = "FUNCTIONAL"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize immutable requirements and validate task execution limits."""
        if not self.task_id.strip() or not self.task_family.strip():
            raise ValueError("task_id and task_family must be non-empty")
        demands = tuple(self.demands)
        if any(not isinstance(item, CapabilityDemand) for item in demands):
            raise TypeError("demands must contain CapabilityDemand instances")
        if len({item.capability_id for item in demands}) != len(demands):
            raise ValueError("capability demands must be unique")
        if not math.isfinite(float(self.stop_confidence)) or not 0.0 <= self.stop_confidence <= 1.0:
            raise ValueError("stop_confidence must be finite and in [0, 1]")
        if self.max_module_calls < 0 or self.max_replans < 0:
            raise ValueError("max_module_calls and max_replans cannot be negative")
        mode = str(self.execution_mode).upper()
        if mode not in {"FUNCTIONAL", "BIOLOGICAL", "HYBRID", "TRAIN", "INFER", "VALIDATE"}:
            raise ValueError("execution_mode is not a supported ORE operation mode")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "demands", demands)
        object.__setattr__(self, "execution_mode", mode)
        object.__setattr__(self, "metadata", dict(self.metadata))

@dataclass(frozen=True, slots=True)
class ModuleResourceEstimate:
    """Separate estimated execution cost and usefulness from measured runtime outcomes."""
    module_id: str
    expected_latency_ms: float
    expected_memory_mb: float
    expected_cpu_cost: float
    expected_information_gain: float
    estimated_success_probability: float
    estimate_source: str

    def __post_init__(self) -> None:
        """Validate finite non-negative costs and probabilities before scoring a module."""
        if not self.module_id.strip() or not self.estimate_source.strip():
            raise ValueError("module_id and estimate_source must be non-empty")
        costs = (self.expected_latency_ms, self.expected_memory_mb, self.expected_cpu_cost)
        if any(not math.isfinite(float(value)) or value < 0.0 for value in costs):
            raise ValueError("resource costs must be finite and non-negative")
        probabilities = (self.expected_information_gain, self.estimated_success_probability)
        if any(not math.isfinite(float(value)) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("information gain and success probability must be in [0, 1]")

@dataclass(frozen=True, slots=True)
class ResourceDecision:
    """Explain why one module was selected or rejected, including every score input."""
    module_id: str
    selected: bool
    score: float
    estimated_cost: float
    expected_information_gain: float
    reason: str
    capability_gain: float = 0.0
    success_probability: float = 0.5
    failure_penalty: float = 0.0
    estimate_source: str = "engineering_assumption"

    def __post_init__(self) -> None:
        """Ensure the decision's numeric diagnostics are finite and its reason is explicit."""
        if not self.module_id.strip() or not self.reason.strip():
            raise ValueError("module_id and reason must be non-empty")
        values = (
            self.score, self.estimated_cost, self.expected_information_gain,
            self.capability_gain, self.success_probability, self.failure_penalty,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("resource decision values must be finite")

@dataclass(frozen=True, slots=True)
class ResourceExecutionPlan:
    """Represent the greedy module set, dependency order, costs, and unmet requirements."""
    task_id: str
    selected_modules: tuple[str, ...]
    execution_order: tuple[str, ...]
    estimated_total_cost: float
    expected_uncertainty_reduction: float
    decisions: tuple[ResourceDecision, ...]
    unmet_capabilities: tuple[str, ...] = ()
    blocked_modules: tuple[str, ...] = ()
    preferred_modules_by_capability: Mapping[str, str] = field(default_factory=dict)
    uncertainty_by_capability: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy mapping inputs and require a finite, non-negative aggregate cost."""
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if not math.isfinite(float(self.estimated_total_cost)) or self.estimated_total_cost < 0:
            raise ValueError("estimated_total_cost must be finite and non-negative")
        if not math.isfinite(float(self.expected_uncertainty_reduction)):
            raise ValueError("expected_uncertainty_reduction must be finite")
        if len(set(self.selected_modules)) != len(self.selected_modules):
            raise ValueError("selected_modules must be unique")
        if len(set(self.execution_order)) != len(self.execution_order):
            raise ValueError("execution_order must be unique")
        if set(self.selected_modules) != set(self.execution_order):
            raise ValueError("execution_order must contain every selected module exactly once")
        object.__setattr__(self, "preferred_modules_by_capability", dict(self.preferred_modules_by_capability))
        object.__setattr__(self, "uncertainty_by_capability", dict(self.uncertainty_by_capability))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation without runtime backend objects."""
        return {
            "task_id": self.task_id,
            "selected_modules": list(self.selected_modules),
            "execution_order": list(self.execution_order),
            "estimated_total_cost": self.estimated_total_cost,
            "expected_uncertainty_reduction": self.expected_uncertainty_reduction,
            "decisions": [
                {
                    "module_id": item.module_id, "selected": item.selected, "score": item.score,
                    "estimated_cost": item.estimated_cost,
                    "expected_information_gain": item.expected_information_gain,
                    "reason": item.reason, "capability_gain": item.capability_gain,
                    "success_probability": item.success_probability,
                    "failure_penalty": item.failure_penalty, "estimate_source": item.estimate_source,
                }
                for item in self.decisions
            ],
            "unmet_capabilities": list(self.unmet_capabilities),
            "blocked_modules": list(self.blocked_modules),
            "preferred_modules_by_capability": dict(self.preferred_modules_by_capability),
            "uncertainty_by_capability": dict(self.uncertainty_by_capability),
        }

@dataclass(frozen=True, slots=True)
class CoordinationResult:
    """Return the final adaptive plan, all execution rounds, and evidence-backed support."""
    task_id: str
    execution_plans: tuple[ResourceExecutionPlan, ...]
    module_results: Mapping[str, Any]
    final_support: Mapping[str, float]
    total_module_calls: int
    completed: bool
    stop_reason: str

    def __post_init__(self) -> None:
        """Defensively copy returned mappings and validate the final call count."""
        if self.total_module_calls < 0:
            raise ValueError("total_module_calls cannot be negative")
        if not self.stop_reason.strip():
            raise ValueError("stop_reason must be non-empty")
        object.__setattr__(self, "module_results", dict(self.module_results))
        object.__setattr__(self, "final_support", dict(self.final_support))

class TaskProfileRegistry:
    """Register deterministic task templates by task family without inferring requirements from text."""

    def __init__(self) -> None:
        """Initialize an empty profile registry with stable family keys."""
        self._profiles: dict[str, TaskProfile] = {}

    def register(self, profile: TaskProfile, *, replace: bool = False) -> TaskProfile:
        """Register one task family and reject accidental replacement of an existing profile."""
        if not isinstance(profile, TaskProfile):
            raise TypeError("profile must be a TaskProfile")
        if profile.task_family in self._profiles and not replace:
            raise ValueError(f"task family is already registered: {profile.task_family}")
        self._profiles[profile.task_family] = profile
        return profile

    def resolve(self, task_family: str) -> TaskProfile:
        """Return the exact registered profile for a task family or raise KeyError."""
        try:
            return self._profiles[task_family]
        except KeyError as exc:
            raise KeyError(f"no task profile registered for {task_family!r}") from exc

    def list_profiles(self) -> tuple[TaskProfile, ...]:
        """Return registered task profiles in deterministic task-family order."""
        return tuple(self._profiles[key] for key in sorted(self._profiles))

__all__ = [
    "CapabilityDemand", "TaskProfile", "ModuleResourceEstimate", "ResourceDecision",
    "ResourceExecutionPlan", "CoordinationResult", "TaskProfileRegistry",
]
