"""ORE Point 23: Learning-Directed Operating Systems and Adaptive Resource Management.

``ore_cros`` implements the CROS (Cognitive Resource-Oriented System)
adaptive resource-management layer for ORE-Neurocomputational: a unified
resource-state model, unit-safe telemetry, task accounting, a MAPE-K
autonomic loop, deterministic and adaptive scheduling policies, memory
tiering, distributed placement, constrained learning and a safety layer
with hard constraints, watchdogs and deterministic fallback.

Scientific scope: this is an engineering implementation of the Point 23
specification. It does not replace the host operating system, and its
default parameters are starting points rather than platform constants.
"""
from __future__ import annotations

from .actions import (
    ActionType, PolicyOrigin, ResourceAction, RuntimeControlState,
    deterministic_fallback_action, keep_current_policy_action,
)
from .config import (
    AdaptationLimits, CROSConfig, ObjectiveWeights, QualityThresholds,
    WindowConfig, DEFAULT_OBJECTIVE_WEIGHTS,
)
from .cycle import CROSCoordinator, CycleResult
from .resource_model import (
    CriticalityLevel, EnvironmentProfile, ModuleBudget, ResourceDemand,
    ResourceDimension, ResourceKind, ResourceScope, ResourceState,
    ScopeLevel, SideEffectPolicy, TaskRecord, TaskState, WorkloadClass,
    build_resource_state, jain_fairness_index, machine_scope,
)

__version__ = "1.0.0"

__all__ = [
    "ActionType", "PolicyOrigin", "ResourceAction", "RuntimeControlState",
    "deterministic_fallback_action", "keep_current_policy_action",
    "AdaptationLimits", "CROSConfig", "ObjectiveWeights", "QualityThresholds",
    "WindowConfig", "DEFAULT_OBJECTIVE_WEIGHTS",
    "CROSCoordinator", "CycleResult",
    "CriticalityLevel", "EnvironmentProfile", "ModuleBudget", "ResourceDemand",
    "ResourceDimension", "ResourceKind", "ResourceScope", "ResourceState",
    "ScopeLevel", "SideEffectPolicy", "TaskRecord", "TaskState",
    "WorkloadClass", "build_resource_state", "jain_fairness_index",
    "machine_scope", "__version__",
]
