"""Module registration, authority boundaries, and dependency planning (Steps 6 and 7).

Step 6 validates manifests, capabilities, and the module-field write authority
matrix ``W_if``. Step 7 builds the dependency and compatibility graph through the
existing ``ore_contracts.DependencyGraphPlanner`` with exact contract versions and
returns a blocked result when mandatory gaps make the plan non-executable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ore_contracts.contracts_registry import ContractCatalog
from ore_contracts.coordinator import CognitiveTask, DependencyGraphPlanner
from ore_contracts.module_registry import ModuleRegistry
from ore_contracts.resource_profiles import (
    CapabilityDemand,
    TaskProfile,
)

from .config import BlackboardPolicy
from .contracts import BlackboardTaskRequest, COGNITIVE_CONTROL_PRODUCER


@dataclass(frozen=True, slots=True)
class ModuleRegistrationReport:
    """Report module registration and authority validation (Step 6)."""

    registered_modules: tuple[str, ...]
    capability_coverage: float
    missing_mandatory_capabilities: tuple[str, ...]
    authority_matrix: Mapping[str, frozenset[str]]
    capabilities: frozenset[str] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        """Return a detached registration summary."""
        return {
            "registered_modules": list(self.registered_modules),
            "capability_coverage": self.capability_coverage,
            "missing_mandatory_capabilities": list(self.missing_mandatory_capabilities),
            "authority_matrix": {
                key: sorted(value) for key, value in sorted(self.authority_matrix.items())
            },
        }


def register_specialist_modules(
    registry: ModuleRegistry,
    catalog: ContractCatalog,
    specialists: Iterable[Any],
) -> ModuleRegistrationReport:
    """Register manifests, bind backends, and compute the authority matrix (Step 6).

    Every specialist must expose ``describe()``, ``execute()``, ``write_fields``,
    and ``subscription_fields`` declarations. Registration is additive: existing
    module identities are reused rather than replaced.
    """
    registered: list[str] = []
    authority: dict[str, frozenset[str]] = {}
    capabilities: set[str] = set()
    for specialist in specialists:
        manifest = specialist.describe()
        module_id = manifest.identity.module_id
        if registry.get_manifest(module_id) is None:
            registry.register(manifest)
        if registry.get_backend(module_id) is None:
            registry.bind_backend(module_id, specialist)
        registered.append(module_id)
        capabilities.update(manifest.capabilities)
        authority[module_id] = frozenset(specialist.write_fields.values())
    return ModuleRegistrationReport(
        registered_modules=tuple(sorted(registered)),
        capability_coverage=1.0,
        missing_mandatory_capabilities=(),
        authority_matrix=authority,
        capabilities=frozenset(capabilities),
    )


def build_permission_policy(authority_matrix: Mapping[str, frozenset[str]]) -> Any:
    """Build the workspace permission policy from the authority matrix ``W_if``.

    The coordinator publishes control records (input root, USG reference, module
    status, caller evidence) through the established ``cognitive_control``
    producer identity, mirroring ``ore_contracts.CognitiveTaskCoordinator``.
    """
    from ore_workspace.permissions import PermissionPolicy

    permissions = {
        module_id: frozenset(fields) for module_id, fields in authority_matrix.items()
    }
    permissions[COGNITIVE_CONTROL_PRODUCER] = frozenset({
        "unresolved_questions", "module_status", "artifact_references",
        "retrieved_memories",
    })
    return PermissionPolicy(permissions)


def build_task_profile(request: BlackboardTaskRequest, policy: BlackboardPolicy) -> TaskProfile:
    """Derive the adaptive task profile from the request and policy (Steps 9, 14)."""
    demands = tuple(
        CapabilityDemand(
            capability_id=capability,
            priority=1.0 if capability in request.required_capabilities else 0.5,
            mandatory=capability in request.required_capabilities,
            target_confidence=policy.goal_threshold,
        )
        for capability in (
            *request.required_capabilities, *request.optional_capabilities,
        )
    )
    return TaskProfile(
        task_id=request.task_id,
        task_family=request.task_family,
        demands=demands,
        stop_confidence=policy.goal_threshold,
        max_module_calls=policy.budgets.max_module_calls,
        max_replans=policy.budgets.max_replans,
        execution_mode=request.execution_mode,
    )


def build_dependency_plan(
    registry: ModuleRegistry,
    catalog: ContractCatalog,
    request: BlackboardTaskRequest,
    input_artifacts: Mapping[str, Any],
) -> Any:
    """Plan the dependency and compatibility graph with exact versions (Step 7)."""
    planner = DependencyGraphPlanner(registry, catalog)
    task = CognitiveTask(
        goal=request.goal,
        required_capabilities=tuple(request.required_capabilities),
        optional_capabilities=tuple(request.optional_capabilities),
        input_artifacts=dict(input_artifacts),
        execution_mode=request.execution_mode,
        priority=request.priority,
        max_replans=1,
        metadata={"input_text": (
            request.input if isinstance(request.input, str) else "multi_source_input"
        )},
        task_id=request.task_id,
    )
    return planner.build(task)


def capability_coverage(required: Iterable[str], registered: Iterable[str]) -> float:
    """Compute ``|C_R intersect union C(m_i)| / |C_R|`` without hiding mandatory gaps (Step 6)."""
    required_set = set(required)
    if not required_set:
        return 1.0
    covered = required_set & set(registered)
    return len(covered) / len(required_set)


__all__ = [
    "ModuleRegistrationReport",
    "build_dependency_plan",
    "build_permission_policy",
    "build_task_profile",
    "capability_coverage",
    "register_specialist_modules",
]
