"""Central registration of the ore_agentic module with the ORE module registry.

This follows the repository's ``ore_usg/registration.py``,
``ore_temporal/registration.py`` and ``ore_learning/registration.py`` pattern:
register contract definitions first, build a validated ``ModuleManifest``,
then register the manifest and bind a backend that satisfies the
``OREModuleProtocol`` (``describe`` / ``execute``).
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
from .pipeline import AgenticEngine
from .registry.agents import WorkerManifest

__all__ = (
    "ORE_AGENTIC_MODULE_ID",
    "ORE_AGENTIC_CAPABILITIES",
    "ORE_AGENTIC_CONTRACTS",
    "AgenticModuleBackend",
    "build_ore_agentic_manifest",
    "register_ore_agentic_contracts",
    "register_ore_agentic_module",
    "worker_manifest_from_dict",
)

ORE_AGENTIC_MODULE_ID = "ore.agentic_workflow"

ORE_AGENTIC_CONTRACTS = (
    ("ORE.AGENTIC.GOAL_REQUEST", "1.0.0"),
    ("ORE.AGENTIC.RESULT", "1.0.0"),
    ("ORE.AGENTIC.MEMORY_RECORD", "1.0.0"),
)

ORE_AGENTIC_CAPABILITIES: tuple[str, ...] = (
    "agentic.workflow.orchestrate",
    "agentic.workflow.recover",
    "agentic.delegation.supervise",
    "agentic.memory.three_tier",
    "agentic.retrieval.hybrid",
    "agentic.protocols.mcp",
    "agentic.protocols.a2a",
    "agentic.evaluation.critic",
)


def register_ore_agentic_contracts(catalog: ContractCatalog) -> tuple[ContractDefinition, ...]:
    """Register the agentic layer's typed contracts in the shared catalog."""
    if not isinstance(catalog, ContractCatalog):
        raise TypeError("catalog must be a ContractCatalog")
    registered: list[ContractDefinition] = []
    for contract_id, version in ORE_AGENTIC_CONTRACTS:
        existing = catalog.get_contract(contract_id, version)
        if existing is not None:
            registered.append(existing)
            continue
        registered.append(catalog.register_model(
            contract_id=contract_id, version=version, model_type=dict,
            required_fields=(),
            optional_fields=(),
            description=f"Point 15 agentic contract {contract_id} at {version}"))
    return tuple(registered)


def build_ore_agentic_manifest() -> ModuleManifest:
    """Build the validated module manifest for the agentic workflow layer."""
    identity = ModuleIdentity(
        module_id=ORE_AGENTIC_MODULE_ID,
        module_version=SOFTWARE_VERSIONS["ore_agentic"],
        module_name="ORE Agentic Workflow and Memory Layer (Point 15)",
        package_path="ore_agentic",
        biological_reference=(
            "Agentic coordination, three-tier memory and protocol interoperation "
            "are engineering abstractions motivated by prefrontal supervisory "
            "control, hippocampal episodic encoding and consolidated procedural "
            "skill; not a brain simulation"),
    )
    input_ports = (
        InputPort("goal_request", "ORE.AGENTIC.GOAL_REQUEST", ("1.0.0",),
                  required=True, expected_data_type="object", multiple=False),
    )
    output_ports = (
        OutputPort("agentic_result", "ORE.AGENTIC.RESULT", "1.0.0", "object",
                   "typed workflow result with outcomes, provenance and the "
                   "50-step operation trace"),
        OutputPort("memory_records", "ORE.AGENTIC.MEMORY_RECORD", "1.0.0", "object",
                   "episodic, procedural and semantic-proposal records with "
                   "typed links",
                   availability="always"),
    )
    return ModuleManifest(
        identity=identity,
        responsibilities=(
            "compile goals into validated typed task graphs with bounded budgets",
            "delegate to eligible workers under leases with fencing and backpressure",
            "guarantee side-effect idempotency, checkpoints and recovery",
            "evaluate outputs against predeclared contracts through a bounded critic loop",
            "maintain three-tier cognitive memory linked to the canonical USG",
            "expose controlled MCP capabilities and optional A2A interoperation",
        ),
        input_ports=input_ports,
        output_ports=output_ports,
        capabilities=ORE_AGENTIC_CAPABILITIES,
        dependencies=(
            ModuleDependency("ore.workspace", required=False,
                             purpose="shared cognitive workspace publication"),
            ModuleDependency("ore.usg_knowledge_import", required=False,
                             purpose="canonical semantic memory retrieval"),
        ),
        execution_modes=("FUNCTIONAL", "INFER", "VALIDATE"),
        limitations=(
            "reference workers are deterministic local callables behind typed adapters",
            "agent semantic output never directly modifies canonical USG knowledge",
            "A2A is used only for genuinely independent external agents",
        ),
    )


class AgenticModuleBackend:
    """Adapt the AgenticEngine to the ORE module protocol (describe / execute)."""

    def __init__(self, manifest: ModuleManifest,
                 engine: AgenticEngine | None = None) -> None:
        """Store the manifest and an optional preconfigured engine."""
        self._manifest = manifest
        self._engine = engine or AgenticEngine()

    def describe(self) -> ModuleManifest:
        """Return exactly the registered manifest."""
        return self._manifest

    def execute(self, inputs: Mapping[str, Any], context: RunContext) -> ModuleResult:
        """Run the 50-step pipeline over the supplied goal request."""
        from .contracts import GoalRequest

        request = inputs.get("goal_request")
        if not isinstance(request, GoalRequest):
            return ModuleResult(
                module_id=self._manifest.identity.module_id,
                module_version=self._manifest.identity.module_version,
                status=ModuleStatus.ABSTAINED,
                warnings=("goal_request input must be a GoalRequest",))
        try:
            result = self._engine.run(request)
        except (ValueError, TypeError) as exc:
            return ModuleResult(
                module_id=self._manifest.identity.module_id,
                module_version=self._manifest.identity.module_version,
                status=ModuleStatus.FAILED,
                diagnostics={"error": str(exc), "capability_support": {
                    capability: 0.0 for capability in ORE_AGENTIC_CAPABILITIES}})
        support = 1.0 if result.status.value in ("succeeded", "reused") else 0.5
        return ModuleResult(
            module_id=self._manifest.identity.module_id,
            module_version=self._manifest.identity.module_version,
            status=ModuleStatus.SUCCESS,
            artifacts=(),
            diagnostics={
                "result": result.to_dict(),
                "capability_support": {capability: support
                                       for capability in ORE_AGENTIC_CAPABILITIES},
            },
            warnings=tuple(result.limitations),
            provenance={"workflow_id": result.workflow_id,
                        "plan_hash": result.plan_hash,
                        "request_id": request.request_id},
            processing_metadata={"operation_steps": len(result.trace),
                                 "outcome": result.outcome_status.value})


def register_ore_agentic_module(registry: ModuleRegistry, *,
                                engine: AgenticEngine | None = None,
                                replace: bool = False) -> AgenticModuleBackend:
    """Register contracts, manifest and backend with a ModuleRegistry."""
    if not isinstance(registry, ModuleRegistry):
        raise TypeError("registry must be a ModuleRegistry")
    register_ore_agentic_contracts(registry.contract_catalog)
    manifest = build_ore_agentic_manifest()
    existing = registry.get_manifest(ORE_AGENTIC_MODULE_ID)
    if existing is None:
        registry.register(manifest)
    elif existing != manifest and not replace:
        raise ValueError(
            "a different ore_agentic manifest is already registered; pass "
            "replace=True to update it")
    elif existing != manifest and replace:
        registry.register(manifest, replace=True)
    if registry.get_backend(ORE_AGENTIC_MODULE_ID) is not None and not replace:
        raise ValueError("the ore_agentic module already has a bound backend")
    backend = AgenticModuleBackend(manifest, engine)
    if replace:
        registry.unbind_backend(ORE_AGENTIC_MODULE_ID)
    registry.bind_backend(ORE_AGENTIC_MODULE_ID, backend)
    return backend


def worker_manifest_from_dict(value: Mapping[str, Any]) -> WorkerManifest:
    """Rebuild a WorkerManifest from its serialized form (registry interop)."""
    from .registry.agents import manifest_from_dict

    return manifest_from_dict(value)
