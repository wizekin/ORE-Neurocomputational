"""Central registration of the ore_temporal module with the ORE module registry.

This follows the repository's ``ore_usg/registration.py`` pattern: register contract
definitions first, build a validated ``ModuleManifest``, then register the manifest and
bind a backend that satisfies the ``OREModuleProtocol`` (``describe`` / ``execute``).
"""
from __future__ import annotations

from typing import Any, Mapping

from ore_contracts import (
    ContractCatalog,
    ContractDefinition,
    InputPort,
    ModuleDependency,
    ModuleIdentity,
    ModuleManifest,
    ModuleRegistry,
    ModuleResult,
    ModuleStatus,
    OutputPort,
    RunContext,
)

from .config import SOFTWARE_VERSIONS
from .contracts import TemporalRequest
from .pipeline import TemporalEngine

__all__ = (
    "ORE_TEMPORAL_MODULE_ID",
    "ORE_TEMPORAL_CAPABILITIES",
    "ORE_TEMPORAL_CONTRACTS",
    "TemporalModuleBackend",
    "build_ore_temporal_manifest",
    "register_ore_temporal_contracts",
    "register_ore_temporal_module",
)

ORE_TEMPORAL_MODULE_ID = "ore.temporal_intelligence"

ORE_TEMPORAL_CONTRACTS = (
    ("ORE.TEMPORAL.REQUEST", "1.0.0"),
    ("ORE.TEMPORAL.OBSERVATIONS", "1.0.0"),
    ("ORE.TEMPORAL.RESULT", "1.0.0"),
)

ORE_TEMPORAL_CAPABILITIES: tuple[str, ...] = (
    "temporal.irregular_sampling.forecast",
    "temporal.event_detection",
    "temporal.anomaly_detection",
    "temporal.continuous_state",
    "temporal.baseline_comparison",
)


def register_ore_temporal_contracts(catalog: ContractCatalog) -> tuple[ContractDefinition, ...]:
    """Register the temporal layer's typed contracts in the shared catalog."""
    if not isinstance(catalog, ContractCatalog):
        raise TypeError("catalog must be a ContractCatalog")
    registered: list[ContractDefinition] = []
    for contract_id, version in ORE_TEMPORAL_CONTRACTS:
        existing = catalog.get_contract(contract_id, version)
        if existing is not None:
            registered.append(existing)
            continue
        registered.append(catalog.register_model(
            contract_id=contract_id, version=version, model_type=dict,
            required_fields=(),
            optional_fields=(),
            description=f"Point 13 TEDIL contract {contract_id} at {version}"))
    return tuple(registered)


def build_ore_temporal_manifest() -> ModuleManifest:
    """Build the validated module manifest for the temporal intelligence layer."""
    identity = ModuleIdentity(
        module_id=ORE_TEMPORAL_MODULE_ID,
        module_version=SOFTWARE_VERSIONS["ore_temporal"],
        module_name="ORE Temporal and Event-Driven Intelligence Layer (Point 13 TEDIL)",
        package_path="ore_temporal",
        biological_reference=(
            "Continuous-time and event-driven families (LNN, CfC, Neural ODE, selective "
            "SSM, SNN) motivated by time-constant membrane dynamics, recurrent memory, "
            "and spiking computation; an engineering abstraction, not a brain simulation"),
    )
    input_ports = (
        InputPort("temporal_request", "ORE.TEMPORAL.REQUEST", ("1.0.0",), required=True,
                  expected_data_type="object", multiple=False),
        InputPort("observations", "ORE.TEMPORAL.OBSERVATIONS", ("1.0.0",), required=True,
                  expected_data_type="object", multiple=True),
    )
    output_ports = (
        OutputPort("temporal_result", "ORE.TEMPORAL.RESULT", "1.0.0", "object",
                   "typed temporal result with events, estimates, uncertainty, and envelope"),
        OutputPort("baseline_comparison", "ORE.TEMPORAL.RESULT", "1.0.0", "object",
                   "deterministic-baseline comparison for every model-benefit claim",
                   availability="always"),
    )
    return ModuleManifest(
        identity=identity,
        responsibilities=(
            "classify temporal problems and select the least expensive representation",
            "run a deterministic baseline as the control condition for every claim",
            "execute continuous-time and event-driven model families with bounded budgets",
            "detect changes and anomalies with calibrated uncertainty",
            "publish typed, versioned, provenance-carrying results to the shared workspace",
        ),
        input_ports=input_ports,
        output_ports=output_ports,
        capabilities=ORE_TEMPORAL_CAPABILITIES,
        dependencies=(
            ModuleDependency("ore.workspace", required=False,
                             purpose="shared cognitive workspace publication"),
        ),
        execution_modes=("FUNCTIONAL", "INFER", "VALIDATE"),
        limitations=(
            "seeded reference models use engineering default parameters",
            "temporal correlations are never treated as causal proof",
            "neural evidence alone never authorizes high-impact actions",
        ),
    )


class TemporalModuleBackend:
    """Adapt the TemporalEngine to the ORE module protocol (describe / execute)."""

    def __init__(self, manifest: ModuleManifest, engine: TemporalEngine | None = None) -> None:
        """Store the manifest and an optional preconfigured engine."""
        self._manifest = manifest
        self._engine = engine or TemporalEngine()

    def describe(self) -> ModuleManifest:
        """Return exactly the registered manifest."""
        return self._manifest

    def execute(self, inputs: Mapping[str, Any], context: RunContext) -> ModuleResult:
        """Run the 50-step pipeline over the supplied request and observations."""
        request = inputs.get("temporal_request")
        observations = inputs.get("observations")
        if not isinstance(request, TemporalRequest):
            return ModuleResult(
                module_id=self._manifest.identity.module_id,
                module_version=self._manifest.identity.module_version,
                status=ModuleStatus.ABSTAINED,
                warnings=("temporal_request input must be a TemporalRequest",))
        if isinstance(observations, Mapping) and "payload" in observations:
            observations = observations["payload"]
        if not observations:
            return ModuleResult(
                module_id=self._manifest.identity.module_id,
                module_version=self._manifest.identity.module_version,
                status=ModuleStatus.ABSTAINED,
                warnings=("observations input must be a non-empty sequence",))
        try:
            result = self._engine.run(request, observations)
        except (ValueError, TypeError) as exc:
            return ModuleResult(
                module_id=self._manifest.identity.module_id,
                module_version=self._manifest.identity.module_version,
                status=ModuleStatus.FAILED,
                diagnostics={"error": str(exc), "capability_support": {
                    capability: 0.0 for capability in ORE_TEMPORAL_CAPABILITIES}})
        support = 1.0 if result.validation_status.value == "valid" else 0.5
        return ModuleResult(
            module_id=self._manifest.identity.module_id,
            module_version=self._manifest.identity.module_version,
            status=ModuleStatus.SUCCESS,
            artifacts=(),
            diagnostics={
                "result": result.to_dict(),
                "capability_support": {capability: support
                                       for capability in ORE_TEMPORAL_CAPABILITIES},
            },
            warnings=tuple(result.limitations),
            provenance={"task_id": result.task_id, "source_id": result.source_id,
                        "interval": list(result.interval)},
            processing_metadata={"operation_steps": len(result.operation_trace),
                                 "task_status": result.diagnostics.get("task_status")})


def register_ore_temporal_module(registry: ModuleRegistry, *,
                                 engine: TemporalEngine | None = None,
                                 replace: bool = False) -> TemporalModuleBackend:
    """Register contracts, manifest, and backend with a ModuleRegistry."""
    if not isinstance(registry, ModuleRegistry):
        raise TypeError("registry must be a ModuleRegistry")
    register_ore_temporal_contracts(registry.contract_catalog)
    manifest = build_ore_temporal_manifest()
    existing = registry.get_manifest(ORE_TEMPORAL_MODULE_ID)
    if existing is None:
        registry.register(manifest)
    elif existing != manifest and not replace:
        raise ValueError(
            "a different ore_temporal manifest is already registered; pass replace=True "
            "to update it")
    elif existing != manifest and replace:
        registry.register(manifest, replace=True)
    if registry.get_backend(ORE_TEMPORAL_MODULE_ID) is not None and not replace:
        raise ValueError("the ore_temporal module already has a bound backend")
    backend = TemporalModuleBackend(manifest, engine)
    if replace:
        registry.unbind_backend(ORE_TEMPORAL_MODULE_ID)
    registry.bind_backend(ORE_TEMPORAL_MODULE_ID, backend)
    return backend
