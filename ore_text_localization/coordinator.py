"""Typed adapter that registers the sensory-language pathway with Task 3's coordinator."""
from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Mapping
from uuid import uuid4

from .contracts import TextImageRequest
from .pathway import VisualLanguagePathway, VisualLanguagePathwayResult
from .registry import build_visual_language_manifest

if TYPE_CHECKING:
    from ore_contracts import (
        CognitiveTask,
        CognitiveTaskCoordinator,
        CognitiveTaskResult,
        ContractArtifact,
        ContractCatalog,
        ContractDefinition,
        ModuleManifest,
        ModuleRegistry,
        ModuleResult,
        RunContext,
    )


IMAGE_CONTRACT_ID = "ore.text_image_request"
PATHWAY_RESULT_CONTRACT_ID = "ore.visual_language_pathway_result"
CONTRACT_VERSION = "1.0.0"
PATHWAY_MODULE_ID = "language.visual_pathway"


def create_text_image_artifact(
    request: TextImageRequest,
    *,
    run_id: str | None = None,
    producer_id: str = "ore_text_localization.input_adapter",
) -> ContractArtifact:
    """Wrap an observed image request in the exact typed contract consumed by Task 3."""
    from ore_contracts import ContractEnvelope, ContractArtifact, EvidenceKind, sha256_json, utc_now_iso
    from ore_contracts.module_artifacts import content_hash

    if not isinstance(request, TextImageRequest):
        raise TypeError("request must be a TextImageRequest")
    if not isinstance(producer_id, str) or not producer_id.strip():
        raise ValueError("producer_id must be non-empty")
    config_hash = sha256_json({
        "contract_id": IMAGE_CONTRACT_ID,
        "schema_version": CONTRACT_VERSION,
        "producer_id": producer_id,
    })
    envelope = ContractEnvelope(
        contract_id=IMAGE_CONTRACT_ID,
        schema_version=CONTRACT_VERSION,
        run_id=run_id or f"external_image_{uuid4().hex}",
        producer_id=producer_id,
        config_hash=config_hash,
        created_at=utc_now_iso(),
        evidence_kind=EvidenceKind.OBSERVED,
        sample_id=request.sequence_id,
        data_hash=content_hash(request),
    )
    return ContractArtifact(
        envelope=envelope,
        payload=request,
        metadata={
            "sequence_id": request.sequence_id,
            "source_id": request.source_id,
            "image_shape": list(request.image.shape),
            "script_profile": dict(request.script_profile),
        },
    )


def _register_contract_if_absent(
    catalog: ContractCatalog,
    definition: ContractDefinition,
) -> None:
    """Register one exact contract version or reject an incompatible existing definition."""
    current = catalog.get_contract(definition.contract_id, definition.version)
    if current is None:
        catalog.register_contract(definition)
        return
    if (
        current.model_type != definition.model_type
        or current.required_fields != definition.required_fields
    ):
        raise ValueError(
            f"contract {definition.contract_id}@{definition.version} is already registered "
            "with an incompatible payload model or required-field schema"
        )


class VisualLanguageModuleAdapter:
    """Implement the ORE module protocol so Task 3 can plan, execute and validate Task 4."""

    def __init__(
        self,
        pathway: VisualLanguagePathway,
        manifest: ModuleManifest | None = None,
    ) -> None:
        """Bind one reusable pathway instance to its stable typed module manifest."""
        if not isinstance(pathway, VisualLanguagePathway):
            raise TypeError("pathway must be a VisualLanguagePathway")
        self.pathway = pathway
        self.manifest = manifest or build_visual_language_manifest()

    def describe(self) -> ModuleManifest:
        """Return the exact manifest already registered in Task 3's ModuleRegistry."""
        return self.manifest

    def execute(
        self,
        inputs: Mapping[str, Any],
        context: RunContext,
    ) -> ModuleResult:
        """Run the pathway from one validated image artifact and return a typed result artifact."""
        from ore_contracts import (
            ContractEnvelope,
            ContractArtifact,
            EvidenceKind,
            ModuleResult,
            ModuleStatus,
            RunContext,
            sha256_json,
            utc_now_iso,
        )
        from ore_contracts.module_artifacts import content_hash

        module_id = self.manifest.identity.module_id
        module_version = self.manifest.identity.module_version
        if not isinstance(context, RunContext):
            raise TypeError("context must be an ore_contracts.RunContext")
        input_artifact = inputs.get("raw_text_image")
        if not isinstance(input_artifact, ContractArtifact):
            return ModuleResult(
                module_id, module_version, ModuleStatus.UNAVAILABLE,
                diagnostics={"input_error": "raw_text_image must be a ContractArtifact"},
            )
        request = input_artifact.payload
        if not isinstance(request, TextImageRequest):
            return ModuleResult(
                module_id, module_version, ModuleStatus.UNAVAILABLE,
                diagnostics={"input_error": "artifact payload must be a TextImageRequest"},
            )

        result: VisualLanguagePathwayResult = self.pathway.process(
            request,
            context={
                "coordinator_run_id": context.run_id,
                "execution_mode": context.mode.value,
                "deterministic_seed": context.deterministic_seed,
            },
        )
        localizer_config = asdict(self.pathway.localizer.config)
        envelope = ContractEnvelope(
            contract_id=PATHWAY_RESULT_CONTRACT_ID,
            schema_version=CONTRACT_VERSION,
            run_id=context.run_id,
            producer_id=module_id,
            config_hash=sha256_json({
                "module_id": module_id,
                "module_version": module_version,
                "localization_config": localizer_config,
            }),
            created_at=utc_now_iso(),
            evidence_kind=EvidenceKind.DERIVED,
            parent_artifact_ids=(input_artifact.envelope.fingerprint(),),
            sample_id=request.sequence_id,
            data_hash=content_hash(result),
        )
        output_artifact = ContractArtifact(
            envelope=envelope,
            payload=result,
            metadata={
                "sequence_id": request.sequence_id,
                "pathway_status": result.status,
                "input_artifact_id": input_artifact.envelope.fingerprint(),
            },
        )
        statuses = {
            "COMPLETED": ModuleStatus.SUCCESS,
            "PARTIAL": ModuleStatus.PARTIAL,
            "UNAVAILABLE": ModuleStatus.ABSTAINED,
        }
        status = statuses.get(result.status, ModuleStatus.FAILED)
        limitations = result.integration.get("limitations", ())
        return ModuleResult(
            module_id,
            module_version,
            status,
            artifacts=(output_artifact,),
            diagnostics={
                "pathway_status": result.status,
                "integration": dict(result.integration),
                "localization": dict(result.localization.get("diagnostics", {})),
            },
            warnings=tuple(str(item) for item in limitations),
            provenance={
                "source": "ore_text_localization.VisualLanguageModuleAdapter",
                "input_artifact_id": input_artifact.envelope.fingerprint(),
                "rules_applied": (
                    "raw_image_localization",
                    "n9_visual_language_processing",
                    "working_memory_handoff",
                ),
            },
            processing_metadata={
                "sequence_id": request.sequence_id,
                "execution_mode": context.mode.value,
                "input_artifact_id": input_artifact.envelope.fingerprint(),
            },
        )


def register_visual_language_module(
    catalog: ContractCatalog,
    registry: ModuleRegistry,
    pathway: VisualLanguagePathway,
    *,
    replace: bool = False,
) -> VisualLanguageModuleAdapter:
    """Register contracts, manifest and executable Task 4 backend in Task 3's registry."""
    from ore_contracts import ContractDefinition, ContractCatalog, ModuleRegistry

    if not isinstance(catalog, ContractCatalog):
        raise TypeError("catalog must be an ore_contracts.ContractCatalog")
    if not isinstance(registry, ModuleRegistry) or registry.contract_catalog is not catalog:
        raise ValueError("registry must be a ModuleRegistry bound to the supplied catalog")

    input_definition = ContractDefinition(
        contract_id=IMAGE_CONTRACT_ID,
        version=CONTRACT_VERSION,
        model_type=TextImageRequest,
        required_fields=("sequence_id", "image"),
        description="Observed text-bearing image with explicit or default script/layout metadata.",
    )
    output_definition = ContractDefinition(
        contract_id=PATHWAY_RESULT_CONTRACT_ID,
        version=CONTRACT_VERSION,
        model_type=VisualLanguagePathwayResult,
        required_fields=("sequence_id", "status", "localization", "integration"),
        description="Structured output of visual localization, N9 language processing and working-memory handoff.",
        validator=lambda value: value.status in {"COMPLETED", "PARTIAL", "UNAVAILABLE"},
    )
    _register_contract_if_absent(catalog, input_definition)
    _register_contract_if_absent(catalog, output_definition)

    manifest = build_visual_language_manifest()
    existing_manifest = registry.get_manifest(manifest.identity.module_id)
    existing_backend = registry.get_backend(manifest.identity.module_id)
    if existing_manifest is not None and existing_manifest != manifest:
        if not replace:
            raise ValueError(f"incompatible manifest already registered: {manifest.identity.module_id}")
        if existing_backend is not None:
            registry.unbind_backend(manifest.identity.module_id)
        registry.register(manifest, replace=True)
    elif existing_manifest is None:
        registry.register(manifest)
    elif existing_backend is not None:
        if isinstance(existing_backend, VisualLanguageModuleAdapter) and existing_backend.pathway is pathway:
            return existing_backend
        if not replace:
            raise ValueError(f"a backend is already bound: {manifest.identity.module_id}")
        registry.unbind_backend(manifest.identity.module_id)

    adapter = VisualLanguageModuleAdapter(pathway, manifest)
    registry.bind_backend(manifest.identity.module_id, adapter)
    return adapter


def build_visual_language_task(
    request: TextImageRequest,
    *,
    priority: float = 0.7,
    max_replans: int = 1,
    task_id: str | None = None,
    input_run_id: str | None = None,
) -> CognitiveTask:
    """Create a Task 3 CognitiveTask with a validated image artifact and Task 4 capability."""
    from ore_contracts import CognitiveTask

    artifact = create_text_image_artifact(request, run_id=input_run_id)
    metadata = {
        "input_text": f"visual text image {request.sequence_id}",
        "source_id": request.source_id,
        "task_family": "visual_language_comprehension",
    }
    arguments: dict[str, Any] = {
        "goal": f"Localize and understand the text in image {request.sequence_id}",
        "required_capabilities": ("language.visual_pathway",),
        "input_artifacts": {"raw_text_image": artifact},
        "expected_outputs": (PATHWAY_RESULT_CONTRACT_ID,),
        "execution_mode": "FUNCTIONAL",
        "priority": priority,
        "max_replans": max_replans,
        "metadata": metadata,
    }
    if task_id is not None:
        arguments["task_id"] = task_id
    return CognitiveTask(**arguments)


def coordinate_visual_language_input(
    coordinator: CognitiveTaskCoordinator,
    request: TextImageRequest,
    **task_options: Any,
) -> CognitiveTaskResult:
    """Submit a typed text-image task to an already configured CognitiveTaskCoordinator."""
    if not callable(getattr(coordinator, "coordinate", None)):
        raise TypeError("coordinator must expose coordinate(CognitiveTask)")
    task = build_visual_language_task(request, **task_options)
    return coordinator.coordinate(task)


__all__ = [
    "PATHWAY_MODULE_ID",
    "PATHWAY_RESULT_CONTRACT_ID",
    "IMAGE_CONTRACT_ID",
    "VisualLanguageModuleAdapter",
    "build_visual_language_task",
    "coordinate_visual_language_input",
    "create_text_image_artifact",
    "register_visual_language_module",
]
