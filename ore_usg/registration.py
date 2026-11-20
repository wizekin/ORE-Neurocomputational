"""Register the USG lexical import as a normal, versioned ORE module."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ore_contracts import (
    ContractEnvelope, EvidenceKind, make_envelope, sha256_json, utc_now_iso,
)
from ore_contracts.contracts_registry import ContractCatalog
from ore_contracts.module_adapters import CallableModuleAdapter
from ore_contracts.module_artifacts import ContractArtifact, content_hash
from ore_contracts.module_manifest import (
    InputPort, ModuleIdentity, ModuleManifest, OutputPort,
)
from ore_contracts.module_registry import ModuleRegistry

from .contracts import (
    ORE_USG_BUNDLE_CONTRACT,
    ORE_USG_LEXICON_CONTRACT,
    ORE_USG_KNOWLEDGE_CONTRACT,
    USG_KNOWLEDGE_SCHEMA_VERSION,
    USGLexiconResource,
)
from .lexical import USGKnowledgeLexicalAdapter
from .knowledge import USGKnowledgeAdapter, USGKnowledgeResource
from .validation import validate_knowledge_bundle


USG_LEXICAL_MODULE_ID = "ore.usg_lexical_import"
USG_LEXICAL_MODULE_VERSION = "0.1.0"
USG_KNOWLEDGE_MODULE_ID = "ore.usg_knowledge_import"
USG_KNOWLEDGE_MODULE_VERSION = "0.1.0"
USG_KNOWLEDGE_CAPABILITIES = (
    "language.lexical_resource.import",
    "language.semantic_resource.import",
    "language.semantic_structure.import",
    "language.knowledge.provenance",
)


USG_LEXICAL_CAPABILITIES = (
    "language.lexical_resource.import",
    "language.lexical_resource.provenance",
)


def _ensure_model_contract(
    catalog: ContractCatalog,
    contract_id: str,
    version: str,
    model_type: type,
    *,
    required_fields: tuple[str, ...],
    description: str,
    validator: Any = None,
) -> None:
    """Register a model-backed contract once, rejecting incompatible pre-registration."""
    existing = catalog.get_contract(contract_id, version)
    if existing is not None:
        if existing.model_type is not model_type or existing.required_fields != required_fields:
            raise ValueError(f"{contract_id}@{version} is already registered with an incompatible model contract")
        return
    catalog.register_model(
        contract_id, version, model_type,
        required_fields=required_fields,
        description=description,
        validator=validator,
    )


def make_usg_bundle_artifact(
    bundle: Mapping[str, Any],
    *,
    run_id: str,
    producer_id: str = "usg.workbench.knowledge_provider",
    config: Mapping[str, Any] | None = None,
) -> ContractArtifact:
    """Wrap a detached USG bundle in the exact ORE input contract version (0.1.0)."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be non-empty")
    if not isinstance(producer_id, str) or not producer_id.strip():
        raise ValueError("producer_id must be non-empty")
    errors = validate_knowledge_bundle(bundle)
    if errors:
        raise ValueError("USG knowledge bundle is invalid: " + "; ".join(errors))
    payload = deepcopy(dict(bundle))
    configuration = dict(config or {})
    envelope = ContractEnvelope(
        contract_id=ORE_USG_BUNDLE_CONTRACT,
        schema_version=USG_KNOWLEDGE_SCHEMA_VERSION,
        run_id=run_id,
        producer_id=producer_id,
        config_hash=sha256_json(configuration),
        created_at=utc_now_iso(),
        evidence_kind=EvidenceKind.EXTERNAL_REFERENCE,
        model_version=USG_KNOWLEDGE_SCHEMA_VERSION,
        data_hash=content_hash(payload),
    )
    return ContractArtifact(
        envelope,
        payload,
        metadata={
            "usg_contract": payload["contract"],
            "usg_bundle_id": payload["bundle_id"],
            "read_only": True,
            "source_database_mutation_performed": False,
        },
    )


def register_usg_contracts(catalog: ContractCatalog) -> tuple[Any, ...]:
    """Register the inbound USG bundle and outbound ORE lexicon payload contracts."""
    if not isinstance(catalog, ContractCatalog):
        raise TypeError("catalog must be a ContractCatalog")
    _ensure_model_contract(
        catalog,
        ORE_USG_BUNDLE_CONTRACT,
        USG_KNOWLEDGE_SCHEMA_VERSION,
        dict,
        required_fields=("contract", "namespace", "bundle_id", "records", "sources"),
        description="Read-only USG-KNOWLEDGE/0.1.0 input bundle; structural and reference validation applies.",
        validator=lambda value: not validate_knowledge_bundle(value),
    )
    _ensure_model_contract(
        catalog,
        ORE_USG_LEXICON_CONTRACT,
        "1.0.0",
        USGLexiconResource,
        required_fields=("entries", "bundle_id", "bundle_digest"),
        description="ORE lexical entries imported from one immutable USG knowledge bundle, with provenance report.",
    )
    _ensure_model_contract(
        catalog,
        ORE_USG_KNOWLEDGE_CONTRACT,
        "1.0.0",
        USGKnowledgeResource,
        required_fields=("lexical", "semantic", "structures", "namespace", "bundle_id", "bundle_digest"),
        description="Unified read-only ORE lexical, semantic, structure, evidence, context, and provenance resource.",
    )
    return tuple(item for item in (
        catalog.get_contract(ORE_USG_BUNDLE_CONTRACT, USG_KNOWLEDGE_SCHEMA_VERSION),
        catalog.get_contract(ORE_USG_LEXICON_CONTRACT, "1.0.0"),
        catalog.get_contract(ORE_USG_KNOWLEDGE_CONTRACT, "1.0.0"),
    ) if item is not None)


def build_usg_lexical_manifest() -> ModuleManifest:
    """Describe the lexical importer without binding it to a USG server or database."""
    return ModuleManifest(
        identity=ModuleIdentity(
            USG_LEXICAL_MODULE_ID,
            USG_LEXICAL_MODULE_VERSION,
            "USG Knowledge Lexical Import",
            package_path="ore_usg",
        ),
        responsibilities=(
            "validate a supplied USG-KNOWLEDGE/0.1.0 bundle",
            "convert eligible lexical forms, lemmas, and senses to ORE lexical records",
            "preserve lifecycle state, source rights, evidence references, and bundle digest",
        ),
        input_ports=(
            InputPort(
                "knowledge_bundle",
                ORE_USG_BUNDLE_CONTRACT,
                (USG_KNOWLEDGE_SCHEMA_VERSION,),
                expected_data_type="builtins.dict",
            ),
        ),
        output_ports=(
            OutputPort(
                "lexicon",
                ORE_USG_LEXICON_CONTRACT,
                "1.0.0",
                "ore_usg.contracts.USGLexiconResource",
                "Read-only ORE lexical resource and import report.",
            ),
        ),
        capabilities=USG_LEXICAL_CAPABILITIES,
        execution_modes=("FUNCTIONAL", "INFER"),
        limitations=(
            "No USG database writes or network calls are performed.",
            "Candidate records are excluded unless explicitly enabled in the adapter.",
            "Unresolved, deprecated, and rejected records are never imported.",
            "Definitions are glosses; numeric semantic features and morphology are not inferred.",
            "Rights metadata is retained; this adapter does not grant redistribution rights.",
        ),
    )


def register_usg_lexical_module(
    registry: ModuleRegistry,
    *,
    adapter: USGKnowledgeLexicalAdapter | None = None,
) -> CallableModuleAdapter:
    """Register and bind the read-only USG lexical importer in an existing ORE registry."""
    if not isinstance(registry, ModuleRegistry):
        raise TypeError("registry must be a ModuleRegistry")
    catalog = registry.contract_catalog
    register_usg_contracts(catalog)
    manifest = build_usg_lexical_manifest()
    existing = registry.get_manifest(USG_LEXICAL_MODULE_ID)
    if existing is None:
        registry.register(manifest)
    elif existing != manifest:
        raise ValueError("an incompatible USG lexical module manifest is already registered")
    if registry.get_backend(USG_LEXICAL_MODULE_ID) is not None:
        raise ValueError("the USG lexical module already has a bound backend")

    lexical_adapter = adapter or USGKnowledgeLexicalAdapter()

    def execute(inputs: dict[str, Any], context: Any) -> tuple[ContractArtifact, ...]:
        """Convert one validated bundle artifact into an immutable ORE lexical resource."""
        source = inputs.get("knowledge_bundle")
        if not isinstance(source, ContractArtifact):
            raise TypeError("knowledge_bundle input must be a ContractArtifact")
        if source.envelope.contract_id != ORE_USG_BUNDLE_CONTRACT:
            raise ValueError("knowledge_bundle input has the wrong contract ID")
        if source.envelope.schema_version != USG_KNOWLEDGE_SCHEMA_VERSION:
            raise ValueError("knowledge_bundle input has an unsupported contract version")
        validation = catalog.validate_artifact(source)
        if not validation.valid:
            raise ValueError("knowledge_bundle artifact is invalid: " + "; ".join(validation.errors))

        resource = lexical_adapter.convert(source.payload)
        envelope = make_envelope(
            ORE_USG_LEXICON_CONTRACT,
            context.run_id,
            USG_LEXICAL_MODULE_ID,
            config={
                "adapter_version": lexical_adapter.version,
                "include_candidate_states": lexical_adapter.include_candidate_states,
            },
            evidence_kind=EvidenceKind.DERIVED,
            parent_artifact_ids=(source.payload_hash,),
            data_hash=source.payload_hash,
            model_version=resource.adapter_version,
        )
        artifact = ContractArtifact(
            envelope,
            resource,
            metadata={
                "usg_bundle_id": resource.bundle_id,
                "usg_namespace": resource.namespace,
                "usg_bundle_digest": resource.bundle_digest,
                "adapter_version": resource.adapter_version,
                "candidate_states_enabled": lexical_adapter.include_candidate_states,
                "runtime_mutation_performed": False,
                "source_database_mutation_performed": False,
            },
        )
        return (artifact,)

    backend = CallableModuleAdapter(manifest, execute)
    binding = registry.bind_backend(USG_LEXICAL_MODULE_ID, backend)
    if binding.module_id != USG_LEXICAL_MODULE_ID:
        raise RuntimeError("ORE registry returned an unexpected module binding")
    return backend


def build_usg_knowledge_manifest() -> ModuleManifest:
    """Describe the unified USG knowledge importer as one typed ORE module."""
    return ModuleManifest(
        identity=ModuleIdentity(
            USG_KNOWLEDGE_MODULE_ID,
            USG_KNOWLEDGE_MODULE_VERSION,
            "USG Unified Knowledge Import",
            package_path="ore_usg",
        ),
        responsibilities=(
            "validate one supplied USG-KNOWLEDGE/0.1.0 bundle",
            "import lexical forms, lemmas, and senses",
            "import concepts, semantic classes, predicate frames, and explicit relations",
            "preserve source rights, evidence, contexts, lifecycle decisions, and bundle identity",
        ),
        input_ports=(
            InputPort(
                "knowledge_bundle",
                ORE_USG_BUNDLE_CONTRACT,
                (USG_KNOWLEDGE_SCHEMA_VERSION,),
                expected_data_type="builtins.dict",
            ),
        ),
        output_ports=(
            OutputPort(
                "knowledge",
                ORE_USG_KNOWLEDGE_CONTRACT,
                "1.0.0",
                "ore_usg.knowledge.USGKnowledgeResource",
                "Unified read-only USG-derived ORE knowledge snapshot.",
            ),
        ),
        capabilities=USG_KNOWLEDGE_CAPABILITIES,
        execution_modes=("FUNCTIONAL", "INFER"),
        limitations=(
            "No USG database writes or network calls are performed.",
            "Candidate records are excluded unless explicitly enabled.",
            "Unresolved, deprecated, and rejected records are not imported as active knowledge.",
            "Semantic features are not inferred from labels or definitions.",
            "Predicate-frame roles are preserved as data and are not yet executable constraints.",
            "Relations are imported only when both endpoints are eligible records.",
            "Source licensing metadata does not grant redistribution rights.",
        ),
    )


def register_usg_knowledge_module(
    registry: ModuleRegistry,
    *,
    adapter: USGKnowledgeAdapter | None = None,
) -> CallableModuleAdapter:
    """Register and bind the unified read-only USG knowledge importer."""
    if not isinstance(registry, ModuleRegistry):
        raise TypeError("registry must be a ModuleRegistry")
    catalog = registry.contract_catalog
    register_usg_contracts(catalog)
    manifest = build_usg_knowledge_manifest()
    existing = registry.get_manifest(USG_KNOWLEDGE_MODULE_ID)
    if existing is None:
        registry.register(manifest)
    elif existing != manifest:
        raise ValueError("an incompatible unified USG knowledge module manifest is already registered")
    if registry.get_backend(USG_KNOWLEDGE_MODULE_ID) is not None:
        raise ValueError("the unified USG knowledge module already has a bound backend")
    knowledge_adapter = adapter or USGKnowledgeAdapter()

    def execute(inputs: dict[str, Any], context: Any) -> tuple[ContractArtifact, ...]:
        """Produce one linked resource artifact from a validated bundle artifact."""
        source = inputs.get("knowledge_bundle")
        if not isinstance(source, ContractArtifact):
            raise TypeError("knowledge_bundle input must be a ContractArtifact")
        if source.envelope.contract_id != ORE_USG_BUNDLE_CONTRACT:
            raise ValueError("knowledge_bundle input has the wrong contract ID")
        if source.envelope.schema_version != USG_KNOWLEDGE_SCHEMA_VERSION:
            raise ValueError("knowledge_bundle input has an unsupported contract version")
        validation = catalog.validate_artifact(source)
        if not validation.valid:
            raise ValueError("knowledge_bundle artifact is invalid: " + "; ".join(validation.errors))
        resource = knowledge_adapter.convert(source.payload)
        envelope = make_envelope(
            ORE_USG_KNOWLEDGE_CONTRACT,
            context.run_id,
            USG_KNOWLEDGE_MODULE_ID,
            config={"include_candidate_states": knowledge_adapter.lexical_adapter.include_candidate_states},
            evidence_kind=EvidenceKind.DERIVED,
            parent_artifact_ids=(source.payload_hash,),
            data_hash=source.payload_hash,
            model_version=USG_KNOWLEDGE_MODULE_VERSION,
        )
        return (ContractArtifact(
            envelope,
            resource,
            metadata={
                "usg_bundle_id": resource.bundle_id,
                "usg_namespace": resource.namespace,
                "usg_bundle_digest": resource.bundle_digest,
                "candidate_states_enabled": knowledge_adapter.lexical_adapter.include_candidate_states,
                "runtime_mutation_performed": False,
                "source_database_mutation_performed": False,
            },
        ),)

    backend = CallableModuleAdapter(manifest, execute)
    binding = registry.bind_backend(USG_KNOWLEDGE_MODULE_ID, backend)
    if binding.module_id != USG_KNOWLEDGE_MODULE_ID:
        raise RuntimeError("ORE registry returned an unexpected unified USG module binding")
    return backend


__all__ = [
    "USG_KNOWLEDGE_MODULE_ID",
    "USG_KNOWLEDGE_MODULE_VERSION",
    "USG_KNOWLEDGE_CAPABILITIES",
    "build_usg_knowledge_manifest",
    "register_usg_knowledge_module",
    "USG_LEXICAL_MODULE_ID",
    "USG_LEXICAL_MODULE_VERSION",
    "USG_LEXICAL_CAPABILITIES",
    "register_usg_contracts",
    "make_usg_bundle_artifact",
    "build_usg_lexical_manifest",
    "register_usg_lexical_module",
]
