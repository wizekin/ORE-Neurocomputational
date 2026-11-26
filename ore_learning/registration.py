"""Central registration of the ore_learning module with the ORE module registry.

Follows the repository's ``ore_usg/registration.py`` and ``ore_temporal/registration.py``
pattern: register contract definitions first, build a validated ``ModuleManifest``,
then register the manifest and bind a backend satisfying ``OREModuleProtocol``.
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
from .contracts import CapabilityRequest, CognitiveTaskRecord
from .pipeline import LearnedModelEngine

__all__ = (
    "ORE_LEARNING_MODULE_ID",
    "ORE_LEARNING_CAPABILITIES",
    "ORE_LEARNING_CONTRACTS",
    "LearningModuleBackend",
    "build_ore_learning_manifest",
    "register_ore_learning_contracts",
    "register_ore_learning_module",
)

ORE_LEARNING_MODULE_ID = "ore.replaceable_learned_models"

ORE_LEARNING_CONTRACTS = (
    ("ORE.LEARNING.TASK", "1.0.0"),
    ("ORE.LEARNING.CAPABILITY_REQUEST", "1.0.0"),
    ("ORE.LEARNING.RESULT", "1.0.0"),
)

ORE_LEARNING_CAPABILITIES: tuple[str, ...] = (
    "learning.registry_admission",
    "learning.capability_routing",
    "learning.typed_inference",
    "learning.process_validation",
    "learning.lifecycle_governance",
)


def register_ore_learning_contracts(catalog: ContractCatalog) -> tuple[ContractDefinition, ...]:
    """Register the learning subsystem's typed contracts in the shared catalog."""
    if not isinstance(catalog, ContractCatalog):
        raise TypeError("catalog must be a ContractCatalog")
    registered: list[ContractDefinition] = []
    for contract_id, version in ORE_LEARNING_CONTRACTS:
        existing = catalog.get_contract(contract_id, version)
        if existing is not None:
            registered.append(existing)
            continue
        registered.append(catalog.register_model(
            contract_id=contract_id, version=version, model_type=dict,
            required_fields=(),
            optional_fields=(),
            description=f"Point 14 RLMS contract {contract_id} at {version}"))
    return tuple(registered)


def build_ore_learning_manifest() -> ModuleManifest:
    """Build the validated module manifest for the learning subsystem."""
    identity = ModuleIdentity(
        module_id=ORE_LEARNING_MODULE_ID,
        module_version=SOFTWARE_VERSIONS["ore_learning"],
        module_name="ORE Replaceable Learned-Model Subsystem (Point 14 RLMS)",
        package_path="ore_learning",
        biological_reference=(
            "Specialized learning architectures (transfer learning, MoE routing, "
            "generative models, PRMs) as replaceable engineering capabilities; no "
            "model is an authority for semantic truth, reasoning policy, or "
            "durable knowledge"),
    )
    input_ports = (
        InputPort("task", "ORE.LEARNING.TASK", ("1.0.0",), required=True,
                  expected_data_type="object", multiple=False),
        InputPort("capability_request", "ORE.LEARNING.CAPABILITY_REQUEST",
                  ("1.0.0",), required=True, expected_data_type="object",
                  multiple=False),
    )
    output_ports = (
        OutputPort("learning_result", "ORE.LEARNING.RESULT", "1.0.0", "object",
                   "typed learning result with selection, proposals, validation, "
                   "rewards, and audit trail"),
        OutputPort("lifecycle_events", "ORE.LEARNING.RESULT", "1.0.0", "object",
                   "append-only lifecycle events linked to tasks and models",
                   availability="always"),
    )
    return ModuleManifest(
        identity=identity,
        responsibilities=(
            "translate cognitive tasks into typed capability requests",
            "admit, rank, and route replaceable learned models under hard policy gates",
            "execute specialists behind typed adapters and CROS resource limits",
            "validate process steps deterministically; PRM and RLVR stay signals",
            "monitor drift, quarantine, roll back, and retire with full lineage",
        ),
        input_ports=input_ports,
        output_ports=output_ports,
        capabilities=ORE_LEARNING_CAPABILITIES,
        dependencies=(
            ModuleDependency("ore.workspace", required=False,
                             purpose="shared cognitive workspace publication"),
        ),
        execution_modes=("FUNCTIONAL", "INFER", "VALIDATE"),
        limitations=(
            "learned models only propose; independent validation gates verification",
            "confidence is a proposal signal and never proof",
            "federated learning, TinyML, GANs, diffusion and QUBO remain optional "
            "research adapters until a workload justifies them",
        ),
    )


class LearningModuleBackend:
    """Adapt the LearnedModelEngine to the ORE module protocol (describe / execute)."""

    def __init__(self, manifest: ModuleManifest,
                 engine: LearnedModelEngine | None = None) -> None:
        """Store the manifest and an optional preconfigured engine."""
        self._manifest = manifest
        self._engine = engine or LearnedModelEngine()

    def describe(self) -> ModuleManifest:
        """Return exactly the registered manifest."""
        return self._manifest

    def execute(self, inputs: Mapping[str, Any], context: RunContext) -> ModuleResult:
        """Run the 56-step pipeline over the supplied task and capability request."""
        task = inputs.get("task")
        request = inputs.get("capability_request")
        if not isinstance(task, CognitiveTaskRecord) or \
                not isinstance(request, CapabilityRequest):
            return ModuleResult(
                module_id=self._manifest.identity.module_id,
                module_version=self._manifest.identity.module_version,
                status=ModuleStatus.ABSTAINED,
                warnings=("inputs must be a CognitiveTaskRecord and a "
                          "CapabilityRequest",))
        try:
            result = self._engine.run(task, request,
                                      inputs=inputs.get("adapter_inputs"),
                                      modality_payloads=inputs.get("modality_payloads"))
        except (ValueError, TypeError) as exc:
            return ModuleResult(
                module_id=self._manifest.identity.module_id,
                module_version=self._manifest.identity.module_version,
                status=ModuleStatus.FAILED,
                diagnostics={"error": str(exc), "capability_support": {
                    capability: 0.0 for capability in ORE_LEARNING_CAPABILITIES}})
        support = 1.0 if result.decision == "completed" else 0.5
        return ModuleResult(
            module_id=self._manifest.identity.module_id,
            module_version=self._manifest.identity.module_version,
            status=ModuleStatus.SUCCESS,
            artifacts=(),
            diagnostics={
                "result": result.to_dict(),
                "capability_support": {capability: support
                                       for capability in ORE_LEARNING_CAPABILITIES},
            },
            warnings=tuple(result.limitations),
            provenance={"task_id": result.task_id,
                        "selected": ([manifest.identity
                                      for manifest in result.selection.selected]
                                     if result.selection is not None else [])},
            processing_metadata={"operation_steps": len(result.operation_trace),
                                 "decision": result.decision})


def register_ore_learning_module(registry: ModuleRegistry, *,
                                 engine: LearnedModelEngine | None = None,
                                 replace: bool = False) -> LearningModuleBackend:
    """Register contracts, manifest, and backend with a ModuleRegistry."""
    if not isinstance(registry, ModuleRegistry):
        raise TypeError("registry must be a ModuleRegistry")
    register_ore_learning_contracts(registry.contract_catalog)
    manifest = build_ore_learning_manifest()
    existing = registry.get_manifest(ORE_LEARNING_MODULE_ID)
    if existing is None:
        registry.register(manifest)
    elif existing != manifest and not replace:
        raise ValueError(
            "a different ore_learning manifest is already registered; pass "
            "replace=True to update it")
    elif existing != manifest and replace:
        registry.register(manifest, replace=True)
    if registry.get_backend(ORE_LEARNING_MODULE_ID) is not None and not replace:
        raise ValueError("the ore_learning module already has a bound backend")
    backend = LearningModuleBackend(manifest, engine)
    if replace:
        registry.unbind_backend(ORE_LEARNING_MODULE_ID)
    registry.bind_backend(ORE_LEARNING_MODULE_ID, backend)
    return backend
