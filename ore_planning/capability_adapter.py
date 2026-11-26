"""Capability registry adapter for operator discovery and eligibility validation.

The adapter queries the existing ore_contracts ModuleRegistry for eligible
operations and normalizes registered manifests into typed OperatorSpec records.
It never registers modules, binds backends, or duplicates registry state: the
registry remains the single source of capability truth.

Causal atoms use contract granularity: an operator requires artifact.available(C)
for every required input port contract and provides artifact.available(C) for
every output port. Exact schema-version compatibility is still validated against
the contract catalog during eligibility checks; the coarser fact level only
drives planner-level ordering, while the coordinator validates exact versions at
execution time. Upstream producers are pulled into the operator pool transitively
so plans can express full dependency chains, and operators whose required inputs
have neither a producer nor an external source are rejected.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ore_contracts import ModuleRegistry

from .contracts import (
    CostVector,
    GoalSpec,
    OperatorSpec,
    Predicate,
    ResourceSpec,
)
from .cost import CostModel

DEFAULT_COST = CostVector(time_ms=10.0, memory_mb=1.0, uncertainty=0.1)


@dataclass(frozen=True, slots=True)
class OperatorRejection:
    """Record why one module was rejected from entering a plan."""

    module_id: str
    capability: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return the rejection record."""
        return {"module_id": self.module_id, "capability": self.capability, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class OperatorDiscovery:
    """Expose eligible operators, explicit rejections, and unsatisfied capabilities."""

    operators: tuple[OperatorSpec, ...]
    rejections: tuple[OperatorRejection, ...]
    unsatisfied_capabilities: tuple[str, ...]
    capability_map: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize collections to deterministic tuples."""
        object.__setattr__(self, "operators", tuple(self.operators))
        object.__setattr__(self, "rejections", tuple(self.rejections))
        object.__setattr__(
            self, "unsatisfied_capabilities", tuple(self.unsatisfied_capabilities),
        )
        object.__setattr__(
            self, "capability_map",
            {key: tuple(value) for key, value in self.capability_map.items()},
        )

    def operators_for_capability(self, capability: str) -> tuple[OperatorSpec, ...]:
        """Return eligible operators that provide one capability."""
        return tuple(
            operator for operator in self.operators
            if operator.module_capability == capability
        )

    def provided_contracts(self) -> frozenset[str]:
        """Return every contract an eligible operator can produce."""
        return frozenset(
            effect.arguments[0]
            for operator in self.operators
            for effect in operator.effects
            if effect.polarity and effect.name == "artifact.available"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible discovery record."""
        return {
            "operators": [item.to_dict() for item in self.operators],
            "rejections": [item.to_dict() for item in self.rejections],
            "unsatisfied_capabilities": list(self.unsatisfied_capabilities),
            "capability_map": {
                key: list(value) for key, value in self.capability_map.items()
            },
        }


class CapabilityRegistryAdapter:
    """Discover and validate eligible operators from the existing module registry."""

    def __init__(
        self,
        registry: ModuleRegistry,
        cost_model: CostModel | None = None,
        *,
        measured_costs: Mapping[str, CostVector] | None = None,
        declared_risks: Mapping[str, float] | None = None,
    ) -> None:
        """Bind the adapter to the shared registry and optional measured telemetry."""
        if not isinstance(registry, ModuleRegistry):
            raise TypeError("registry must be a ModuleRegistry")
        self.registry = registry
        self.cost_model = cost_model or CostModel()
        self.measured_costs = dict(measured_costs or {})
        self.declared_risks = dict(declared_risks or {})

    def _operator_cost(self, module_id: str) -> CostVector:
        """Prefer measured telemetry over the documented default cost."""
        return self.measured_costs.get(module_id, DEFAULT_COST)

    def _operator_from_manifest(self, capability: str, module_id: str) -> OperatorSpec:
        """Normalize one registered manifest into a typed operator specification."""
        manifest = self.registry.get_manifest(module_id)
        if manifest is None:
            raise KeyError(f"unknown module: {module_id}")
        preconditions = tuple(
            Predicate("artifact.available", (port.contract_id,))
            for port in manifest.input_ports
            if port.required
        )
        effects = tuple(
            Predicate("artifact.available", (port.contract_id,))
            for port in manifest.output_ports
        ) + (Predicate("goal.satisfied", (capability,)),)
        input_schema = {
            port.port_id: tuple(port.accepted_versions)
            for port in manifest.input_ports
        }
        output_schema = {
            port.port_id: (port.contract_version,)
            for port in manifest.output_ports
        }
        return OperatorSpec(
            operator_id=module_id,
            preconditions=preconditions,
            effects=effects,
            input_schema=input_schema,
            output_schema=output_schema,
            module_capability=capability,
            estimated_cost=self._operator_cost(module_id),
            risk=self.declared_risks.get(module_id, 0.1),
            resource_requirements=ResourceSpec(
                required_units={module_id: 1.0},
                exclusive_resources=(f"module:{module_id}",),
            ),
            module_id=module_id,
            execution_modes=tuple(manifest.execution_modes),
        )

    def _eligibility_rejection(
        self,
        capability: str,
        module_id: str,
        execution_mode: str,
        forbidden: Iterable[str],
    ) -> str | None:
        """Return a rejection reason when a module cannot enter any plan."""
        if module_id in set(forbidden):
            return "forbidden by goal constraints"
        manifest = self.registry.get_manifest(module_id)
        if manifest is None:
            return "module is not registered"
        if execution_mode not in manifest.execution_modes:
            return f"module does not support execution mode {execution_mode}"
        record = next(
            (item for item in self.registry.find_by_capability(capability)
             if item.module_id == module_id),
            None,
        )
        if record is None:
            return "capability is not declared by this module"
        if record.disabled:
            return "module is disabled in the registry"
        if not record.backend_bound:
            return "module backend is not bound"
        if not record.validated:
            return "module backend failed validation"
        if not record.runnable:
            return "module dependencies are unavailable"
        for port in (*manifest.input_ports, *manifest.output_ports):
            versions = (
                port.accepted_versions if hasattr(port, "accepted_versions")
                else (port.contract_version,)
            )
            for version in versions:
                if self.registry.contract_catalog.get_contract(port.contract_id, version) is None:
                    return (
                        f"port {port.port_id} references unregistered contract "
                        f"{port.contract_id}@{version}"
                    )
        return None

    def _producer_manifests(self, contract_id: str) -> list[Any]:
        """Return registered manifests that declare one output contract."""
        return [
            manifest for manifest in self.registry.list_manifests()
            if any(port.contract_id == contract_id for port in manifest.output_ports)
        ]

    def discover(
        self,
        goal: GoalSpec,
        *,
        execution_mode: str = "FUNCTIONAL",
        extra_capabilities: Iterable[str] = (),
        external_contracts: Iterable[str] = (),
    ) -> OperatorDiscovery:
        """Query the registry and return eligible operators for the goal's capabilities.

        The operator pool is closed under required-input producers: upstream
        modules are added transitively until every needed contract is either
        produced by an included operator or supplied externally. Operators whose
        required inputs cannot be satisfied are rejected for missing dependencies.
        """
        forbidden = set(goal.forbidden_operations)
        external = set(external_contracts)
        requested = sorted(
            {condition.predicate.arguments[0]
             for condition in goal.success_conditions if condition.predicate.arguments}
            | set(extra_capabilities)
        )
        operators: dict[str, OperatorSpec] = {}
        rejections: list[OperatorRejection] = []
        unsatisfied: list[str] = []

        def add_operator(capability: str, module_id: str) -> bool:
            """Validate and add one operator; return False when rejected."""
            reason = self._eligibility_rejection(
                capability, module_id, execution_mode, forbidden,
            )
            if reason is not None:
                rejections.append(OperatorRejection(module_id, capability, reason))
                return False
            operator = self._operator_from_manifest(capability, module_id)
            operators[f"{module_id}|{capability}"] = operator
            return True

        for capability in requested:
            records = self.registry.find_by_capability(capability)
            if not records:
                unsatisfied.append(capability)
                continue
            eligible = [
                record for record in sorted(records, key=lambda item: item.module_id)
                if add_operator(capability, record.module_id)
            ]
            if not eligible:
                unsatisfied.append(capability)

        pending = list(operators.values())
        while pending:
            operator = pending.pop()
            required_contracts = {
                precondition.arguments[0]
                for precondition in operator.preconditions
                if precondition.name == "artifact.available"
            }
            for contract in sorted(required_contracts):
                provided = {
                    effect.arguments[0]
                    for candidate in operators.values()
                    for effect in candidate.effects
                    if effect.polarity and effect.name == "artifact.available"
                } | external
                if contract in provided:
                    continue
                producer_ids: list[str] = []
                for manifest in sorted(
                    self._producer_manifests(contract),
                    key=lambda item: item.identity.module_id,
                ):
                    module_id = manifest.identity.module_id
                    if not manifest.capabilities:
                        continue
                    capability = manifest.capabilities[0]
                    key = f"{module_id}|{capability}"
                    if key in operators:
                        continue
                    if add_operator(capability, module_id):
                        producer_ids.append(module_id)
                        pending.append(operators[key])
                if not producer_ids and contract not in external:
                    rejections.append(OperatorRejection(
                        operator.operator_id or "unknown",
                        operator.module_capability,
                        f"missing dependency: no producer or external input for {contract}",
                    ))

        capability_map: dict[str, tuple[str, ...]] = {}
        for operator in operators.values():
            ids = capability_map.setdefault(operator.module_capability, [])
            if operator.operator_id not in ids:
                ids.append(operator.operator_id)
        requested_map = {
            capability: tuple(sorted(capability_map.get(capability, ())))
            for capability in requested
        }
        support_map = {
            capability: tuple(sorted(ids))
            for capability, ids in capability_map.items()
            if capability not in requested
        }
        return OperatorDiscovery(
            operators=tuple(operators[key] for key in sorted(operators)),
            rejections=tuple(rejections),
            unsatisfied_capabilities=tuple(sorted(set(unsatisfied))),
            capability_map={**requested_map, **support_map},
        )


__all__ = ["DEFAULT_COST", "CapabilityRegistryAdapter", "OperatorDiscovery", "OperatorRejection"]
