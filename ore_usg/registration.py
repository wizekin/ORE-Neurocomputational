"""Register versioned read-only USG knowledge import modules in ORE."""
from __future__ import annotations

from copy import deepcopy
import time
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
from ore_contracts.module_protocol import ModuleResult, ModuleStatus

from .contracts import (
    ORE_USG_BUNDLE_CONTRACT,
    ORE_USG_LEXICON_CONTRACT,
    ORE_USG_KNOWLEDGE_CONTRACT,
    ORE_USG_QUERY_CONTRACT,
    ORE_USG_RETRIEVAL_CONTRACT,
    USG_KNOWLEDGE_SCHEMA_VERSION,
    USGLexiconResource,
)
from .lexical import USGKnowledgeLexicalAdapter
from .knowledge import USGKnowledgeAdapter, USGKnowledgeResource
from .access import USGKnowledgeAccess, USGKnowledgeQuery, USGKnowledgeQueryResult
from .validation import validate_knowledge_bundle
from .observability import USGSQLiteObservability


USG_LEXICAL_MODULE_ID = "ore.usg_lexical_import"
USG_LEXICAL_MODULE_VERSION = "0.1.0"
USG_KNOWLEDGE_MODULE_ID = "ore.usg_knowledge_import"
USG_KNOWLEDGE_MODULE_VERSION = "0.1.0"
USG_KNOWLEDGE_ACCESS_MODULE_ID = "ore.usg_knowledge_access"
USG_KNOWLEDGE_ACCESS_MODULE_VERSION = "0.1.0"
USG_KNOWLEDGE_ACCESS_CAPABILITIES = (
    "language.knowledge.exact_lookup",
    "language.knowledge.related_lookup",
    "language.knowledge.contextual_lookup",
    "language.lexical.meanings.retrieve",
    "language.semantic.concepts.retrieve",
    "language.semantic.relations.retrieve",
    "language.predicate.frames.retrieve",
    "language.knowledge.provenance.retrieve",
)
USG_KNOWLEDGE_CAPABILITIES = (
    "language.lexical_resource.import",
    "language.semantic_resource.import",
    "language.semantic_structure.import",
    "language.semantic_categories.import",
    "language.semantic_relations.import",
    "language.predicate_frames.import",
    "language.lexical_semantics.link",
    "language.morphology.link",
    "language.context.import",
    "language.evidence.provenance",
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
    """Register inbound USG bundle and outbound ORE lexical and unified knowledge contracts."""
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
    _ensure_model_contract(
        catalog,
        ORE_USG_QUERY_CONTRACT,
        "1.0.0",
        USGKnowledgeQuery,
        required_fields=("mode",),
        description="Validated request for exact, direct-related, or contextual retrieval from one ORE USG snapshot.",
    )
    _ensure_model_contract(
        catalog,
        ORE_USG_RETRIEVAL_CONTRACT,
        "1.0.0",
        USGKnowledgeQueryResult,
        required_fields=("mode", "bundle_id", "bundle_digest", "candidates"),
        description="Read-only source-backed USG retrieval result with explicit uncertainty and lineage.",
    )
    return tuple(item for item in (
        catalog.get_contract(ORE_USG_BUNDLE_CONTRACT, USG_KNOWLEDGE_SCHEMA_VERSION),
        catalog.get_contract(ORE_USG_LEXICON_CONTRACT, "1.0.0"),
        catalog.get_contract(ORE_USG_KNOWLEDGE_CONTRACT, "1.0.0"),
        catalog.get_contract(ORE_USG_QUERY_CONTRACT, "1.0.0"),
        catalog.get_contract(ORE_USG_RETRIEVAL_CONTRACT, "1.0.0"),
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




def make_usg_knowledge_query_artifact(
    request: USGKnowledgeQuery,
    *,
    run_id: str,
    producer_id: str = "ore.knowledge_client",
) -> ContractArtifact:
    """Wrap a typed retrieval request in an ORE contract artifact."""
    if not isinstance(request, USGKnowledgeQuery):
        raise TypeError("request must be a USGKnowledgeQuery")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be non-empty")
    if not isinstance(producer_id, str) or not producer_id.strip():
        raise ValueError("producer_id must be non-empty")
    return ContractArtifact(
        ContractEnvelope(
            contract_id=ORE_USG_QUERY_CONTRACT,
            schema_version="1.0.0",
            run_id=run_id,
            producer_id=producer_id,
            config_hash=sha256_json({"mode": request.mode}),
            created_at=utc_now_iso(),
            evidence_kind=EvidenceKind.EXTERNAL_REFERENCE,
            model_version=USG_KNOWLEDGE_ACCESS_MODULE_VERSION,
            data_hash=content_hash(request),
        ),
        request,
        metadata={"read_only": True, "query_mode": request.mode},
    )


def build_usg_knowledge_access_manifest() -> ModuleManifest:
    """Declare the typed request/response contract for ORE's shared USG knowledge API."""
    return ModuleManifest(
        identity=ModuleIdentity(
            USG_KNOWLEDGE_ACCESS_MODULE_ID,
            USG_KNOWLEDGE_ACCESS_MODULE_VERSION,
            "USG Knowledge Access",
            package_path="ore_usg",
        ),
        responsibilities=(
            "provide exact lexical, semantic class, concept, and predicate-frame lookup",
            "retrieve directly related concepts only through explicit directed source assertions",
            "retrieve contextual associations only through explicit USG context assignments",
            "return evidence statuses, source rights, source references, and snapshot lineage",
            "expose unavailable domains and ambiguous/unresolved states without selecting a winner",
        ),
        input_ports=(
            InputPort(
                "knowledge",
                ORE_USG_KNOWLEDGE_CONTRACT,
                ("1.0.0",),
                expected_data_type="ore_usg.knowledge.USGKnowledgeResource",
            ),
            InputPort(
                "request",
                ORE_USG_QUERY_CONTRACT,
                ("1.0.0",),
                expected_data_type="ore_usg.access.USGKnowledgeQuery",
            ),
        ),
        output_ports=(
            OutputPort(
                "retrieval",
                ORE_USG_RETRIEVAL_CONTRACT,
                "1.0.0",
                "ore_usg.access.USGKnowledgeQueryResult",
                "Controlled retrieval results with provenance, uncertainty, and confidence availability.",
            ),
        ),
        capabilities=USG_KNOWLEDGE_ACCESS_CAPABILITIES,
        execution_modes=("FUNCTIONAL", "INFER"),
        limitations=(
            "Exact lookup uses conservative case-insensitive equality; no stemming or fuzzy substitution is applied.",
            "Related-concept lookup is one-hop and follows only explicit source-backed relations, memberships, and lexical-concept links.",
            "Contextual lookup follows explicit context assignments and does not infer context from text.",
            "No transitive, inverse, causal, or unasserted relation is generated.",
            "No calibrated confidence is present in the current USG contract; confidence_value remains None.",
            "Evidence support status is not a verified fact or an independently adjudicated truth value.",
            "Missing domains return empty results with explicit availability metadata instead of blocking unrelated domains.",
        ),
    )


class USGKnowledgeAccessModuleBackend:
    """Registry backend for controlled read-only queries against a supplied USG snapshot."""

    def __init__(
        self, manifest: ModuleManifest, catalog: ContractCatalog,
        observability: USGSQLiteObservability | None = None,
    ) -> None:
        self.manifest = manifest
        self.catalog = catalog
        self.observability = observability

    def describe(self) -> ModuleManifest:
        return self.manifest

    def execute(self, inputs: Mapping[str, Any], context: Any) -> ModuleResult:
        """Run retrieval and optionally persist execution and artifact lineage."""
        if self.observability is not None:
            return _execute_observed(
                self.observability, USG_KNOWLEDGE_ACCESS_MODULE_ID,
                USG_KNOWLEDGE_ACCESS_MODULE_VERSION, inputs, context,
                lambda: self._execute(inputs, context),
            )
        return self._execute(inputs, context)

    def _execute(self, inputs: Mapping[str, Any], context: Any) -> ModuleResult:
        """Validate both parent artifacts, retrieve, then emit a two-parent result artifact."""
        try:
            resource_artifact = inputs.get("knowledge")
            request_artifact = inputs.get("request")
            if not isinstance(resource_artifact, ContractArtifact):
                return ModuleResult(
                    USG_KNOWLEDGE_ACCESS_MODULE_ID, USG_KNOWLEDGE_ACCESS_MODULE_VERSION,
                    ModuleStatus.UNAVAILABLE,
                    diagnostics={"error": "knowledge input must be a ContractArtifact"},
                )
            if not isinstance(request_artifact, ContractArtifact):
                return ModuleResult(
                    USG_KNOWLEDGE_ACCESS_MODULE_ID, USG_KNOWLEDGE_ACCESS_MODULE_VERSION,
                    ModuleStatus.UNAVAILABLE,
                    diagnostics={"error": "request input must be a ContractArtifact"},
                )
            expected = (
                (resource_artifact, ORE_USG_KNOWLEDGE_CONTRACT, "1.0.0", USGKnowledgeResource),
                (request_artifact, ORE_USG_QUERY_CONTRACT, "1.0.0", USGKnowledgeQuery),
            )
            validation_errors: list[str] = []
            for artifact, contract_id, version, payload_type in expected:
                if artifact.envelope.contract_id != contract_id:
                    validation_errors.append(
                        f"expected contract {contract_id}, got {artifact.envelope.contract_id}"
                    )
                    continue
                if artifact.envelope.schema_version != version:
                    validation_errors.append(f"{contract_id} requires schema version {version}")
                    continue
                validation = self.catalog.validate_artifact(artifact)
                if not validation.valid:
                    validation_errors.extend(validation.errors)
                if not isinstance(artifact.payload, payload_type):
                    validation_errors.append(
                        f"{contract_id} payload must be {payload_type.__module__}.{payload_type.__qualname__}"
                    )
            if validation_errors:
                return ModuleResult(
                    USG_KNOWLEDGE_ACCESS_MODULE_ID, USG_KNOWLEDGE_ACCESS_MODULE_VERSION,
                    ModuleStatus.UNAVAILABLE,
                    diagnostics={"input_validation_errors": validation_errors},
                )

            request = request_artifact.payload
            access = USGKnowledgeAccess(resource_artifact.payload)
            result = access.query(request)
            parent_hashes = (resource_artifact.payload_hash, request_artifact.payload_hash)
            envelope = make_envelope(
                ORE_USG_RETRIEVAL_CONTRACT,
                context.run_id,
                USG_KNOWLEDGE_ACCESS_MODULE_ID,
                config={"mode": request.mode, "max_results": request.max_results},
                evidence_kind=EvidenceKind.DERIVED,
                parent_artifact_ids=parent_hashes,
                data_hash=content_hash(result),
                model_version=USG_KNOWLEDGE_ACCESS_MODULE_VERSION,
            )
            artifact = ContractArtifact(
                envelope,
                result,
                metadata={
                    "read_only": True,
                    "bundle_id": result.bundle_id,
                    "bundle_digest": result.bundle_digest,
                    "query_mode": result.mode,
                    "candidate_count": len(result.candidates),
                    "uncertainty_present": result.uncertainty_present,
                    "confidence_available": result.confidence_available,
                    "unavailable_domains": list(result.unavailable_domains),
                },
            )
            return ModuleResult(
                USG_KNOWLEDGE_ACCESS_MODULE_ID,
                USG_KNOWLEDGE_ACCESS_MODULE_VERSION,
                ModuleStatus.SUCCESS,
                artifacts=(artifact,),
                diagnostics={
                    "output_count": 1,
                    "candidate_count": len(result.candidates),
                    "unavailable_domains": list(result.unavailable_domains),
                    "uncertainty_present": result.uncertainty_present,
                    "confidence_available": result.confidence_available,
                    "capability_support": {
                        capability: 1.0 for capability in USG_KNOWLEDGE_ACCESS_CAPABILITIES
                    },
                    "capability_support_basis": (
                        "successful typed retrieval only; not truth, confidence, or knowledge completeness"
                    ),
                },
                provenance={
                    "read_only": True,
                    "knowledge_artifact_hash": resource_artifact.payload_hash,
                    "query_artifact_hash": request_artifact.payload_hash,
                    "parent_artifact_hashes": list(parent_hashes),
                    "bundle_id": result.bundle_id,
                    "bundle_digest": result.bundle_digest,
                },
            )
        except Exception as exc:
            return ModuleResult(
                USG_KNOWLEDGE_ACCESS_MODULE_ID,
                USG_KNOWLEDGE_ACCESS_MODULE_VERSION,
                ModuleStatus.FAILED,
                diagnostics={"exception_type": type(exc).__name__, "error": str(exc)},
                warnings=("USG knowledge retrieval failed; no output artifact was emitted.",),
                provenance={"read_only": True},
            )


def register_usg_knowledge_access_module(
    registry: ModuleRegistry,
    *,
    observability: USGSQLiteObservability | None = None,
) -> USGKnowledgeAccessModuleBackend:
    """Register and bind the controlled USG knowledge query module."""
    if not isinstance(registry, ModuleRegistry):
        raise TypeError("registry must be a ModuleRegistry")
    catalog = registry.contract_catalog
    register_usg_contracts(catalog)
    manifest = build_usg_knowledge_access_manifest()
    existing = registry.get_manifest(USG_KNOWLEDGE_ACCESS_MODULE_ID)
    if existing is None:
        registry.register(manifest)
    elif existing != manifest:
        raise ValueError("an incompatible USG knowledge-access manifest is already registered")
    backend = registry.get_backend(USG_KNOWLEDGE_ACCESS_MODULE_ID)
    if backend is not None:
        if not isinstance(backend, USGKnowledgeAccessModuleBackend):
            raise ValueError("the USG knowledge-access module has an incompatible backend")
        if observability is not None:
            backend.observability = observability
        return backend
    backend = USGKnowledgeAccessModuleBackend(manifest, catalog, observability=observability)
    binding = registry.bind_backend(USG_KNOWLEDGE_ACCESS_MODULE_ID, backend)
    if binding.module_id != USG_KNOWLEDGE_ACCESS_MODULE_ID:
        raise RuntimeError("ORE registry returned an unexpected USG knowledge-access binding")
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
            "link lexical entries and meanings to source-backed semantic concepts",
            "link surface forms to lemmas only through explicit source relations",
            "expose typed context, evidence, source licensing, and redistribution metadata",
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
            "Candidate records are excluded from active inventories unless explicitly enabled.",
            "Unresolved records and assertions are retained as inactive knowledge when source rights permit; deprecated and rejected records are excluded.",
            "Semantic features are not inferred from labels or definitions.",
            "Predicate roles are executable only when the ORE frame contract can represent them without dropping declared constraints; all other fields remain preserved.",
            "Relations and mappings retain their original direction, predicate, evidence, context, and provenance; no inverse or transitive edges are inferred.",
            "Category membership conflicts are represented from explicit contradictory evidence rather than resolved automatically.",
            "Lexical-concept links use exact lexical-sense identity or explicit USG relation/mapping records; ambiguous alternatives remain separate.",
            "Morphological form-to-lemma links require a recognized explicit directed USG relation; missing and ambiguous forms are not guessed.",
            "Context and evidence values remain descriptive metadata; no numeric activation feature or verified fact is invented.",
            "Evidence local-use eligibility and redistribution restrictions remain tied to source provenance.",
            "Source licensing metadata does not grant redistribution rights.",
        ),
    )




def _execute_observed(
    observer: USGSQLiteObservability,
    module_id: str,
    module_version: str,
    inputs: Mapping[str, Any],
    context: Any,
    execute: Any,
) -> ModuleResult:
    """Run a backend and persist its event plus input/output artifact lineage."""
    started = time.perf_counter()
    run_id = getattr(context, "run_id", None)
    input_artifacts = [
        value for value in inputs.values() if isinstance(value, ContractArtifact)
    ]
    source_artifact = next(
        (
            artifact for artifact in input_artifacts
            if artifact.envelope.contract_id == ORE_USG_BUNDLE_CONTRACT
            and isinstance(artifact.payload, Mapping)
        ),
        None,
    )
    result: ModuleResult | None = None
    failure: Exception | None = None
    try:
        result = execute()
        status_value = getattr(result.status, "value", str(result.status))
    except Exception as exc:
        failure = exc
        status_value = "FAILED"
    duration_ms = (time.perf_counter() - started) * 1000.0

    source_ids: tuple[str, ...] = ()
    source_namespace = None
    source_bundle_id = None
    source_bundle_digest = None
    if source_artifact is not None:
        bundle = source_artifact.payload
        for key in ("records", "relations", "mappings", "evidence", "contexts", "sources"):
            for item in bundle.get(key, ()) if isinstance(bundle.get(key, ()), list) else ():
                if isinstance(item, Mapping) and item.get("id") is not None:
                    source_ids += (str(item["id"]),)
        source_namespace = str(bundle.get("namespace") or "") or None
        source_bundle_id = str(bundle.get("bundle_id") or "") or None
        source_bundle_digest = str(bundle.get("bundle_digest") or "") or None
    output_artifacts = tuple(getattr(result, "artifacts", ()) or ()) if result is not None else ()
    for artifact in output_artifacts:
        envelope = artifact.envelope
        metadata = dict(getattr(artifact, "metadata", {}) or {})
        source_bundle_digest = str(
            metadata.get("usg_bundle_digest") or metadata.get("bundle_digest")
            or source_bundle_digest or ""
        ) or None
        observer.record_artifact_lineage(
            artifact_id=artifact.payload_hash,
            payload_hash=artifact.payload_hash,
            contract_id=envelope.contract_id,
            schema_version=envelope.schema_version,
            producer_module=module_id,
            producer_version=module_version,
            parent_artifact_ids=tuple(getattr(envelope, "parent_artifact_ids", ()) or ()),
            run_id=run_id,
            source_namespace=str(metadata.get("usg_namespace") or source_namespace or "") or None,
            source_bundle_id=str(metadata.get("usg_bundle_id") or metadata.get("bundle_id") or source_bundle_id or "") or None,
            source_bundle_digest=str(metadata.get("usg_bundle_digest") or metadata.get("bundle_digest") or source_bundle_digest or "") or None,
            source_record_ids=source_ids,
        )

    if source_artifact is not None:
        observer.record_source_bundle(
            artifact_id=source_artifact.payload_hash,
            payload_hash=source_artifact.payload_hash,
            contract_id=source_artifact.envelope.contract_id,
            schema_version=source_artifact.envelope.schema_version,
            producer_module="usg.source_export",
            producer_version="external",
            bundle=source_artifact.payload,
            bundle_digest=source_bundle_digest or source_artifact.payload_hash,
        )
        source_bundle_digest = source_bundle_digest or source_artifact.payload_hash

    diagnostics = dict(getattr(result, "diagnostics", {}) or {}) if result is not None else {}
    if failure is not None:
        diagnostics.update({"exception_type": type(failure).__name__, "error": str(failure)})
    observer.record_module_execution(
        module_id=module_id,
        module_version=module_version,
        operation="execute",
        status=status_value,
        duration_ms=duration_ms,
        run_id=run_id,
        input_artifact_ids=tuple(item.payload_hash for item in input_artifacts),
        output_artifact_ids=tuple(item.payload_hash for item in output_artifacts),
        bundle_id=source_bundle_id,
        bundle_digest=source_bundle_digest,
        metadata=diagnostics,
    )
    if failure is not None:
        raise failure
    assert result is not None
    return result


class USGKnowledgeModuleBackend:
    """Protocol backend that reports operational capability completion to adaptive ORE."""

    def __init__(
        self,
        manifest: ModuleManifest,
        catalog: ContractCatalog,
        adapter: USGKnowledgeAdapter,
        observability: USGSQLiteObservability | None = None,
    ) -> None:
        """Bind the exact manifest, validator, adapter, and optional SQLite observer."""
        self.manifest = manifest
        self.catalog = catalog
        self.adapter = adapter
        self.observability = observability

    def describe(self) -> ModuleManifest:
        """Return the module manifest used for registry compatibility checks."""
        return self.manifest

    def execute(self, inputs: Mapping[str, Any], context: Any) -> ModuleResult:
        """Run the importer and optionally persist execution and artifact lineage."""
        if self.observability is not None:
            return _execute_observed(
                self.observability, USG_KNOWLEDGE_MODULE_ID, USG_KNOWLEDGE_MODULE_VERSION,
                inputs, context, lambda: self._execute(inputs, context),
            )
        return self._execute(inputs, context)

    def _execute(self, inputs: Mapping[str, Any], context: Any) -> ModuleResult:
        """Validate one bundle, emit a typed resource, and report operational support."""
        try:
            source = inputs.get("knowledge_bundle")
            if not isinstance(source, ContractArtifact):
                return ModuleResult(
                    USG_KNOWLEDGE_MODULE_ID, USG_KNOWLEDGE_MODULE_VERSION,
                    ModuleStatus.UNAVAILABLE,
                    diagnostics={"error": "knowledge_bundle input must be a ContractArtifact"},
                )
            if source.envelope.contract_id != ORE_USG_BUNDLE_CONTRACT:
                return ModuleResult(
                    USG_KNOWLEDGE_MODULE_ID, USG_KNOWLEDGE_MODULE_VERSION,
                    ModuleStatus.UNAVAILABLE,
                    diagnostics={"error": "knowledge_bundle input has the wrong contract ID"},
                )
            if source.envelope.schema_version != USG_KNOWLEDGE_SCHEMA_VERSION:
                return ModuleResult(
                    USG_KNOWLEDGE_MODULE_ID, USG_KNOWLEDGE_MODULE_VERSION,
                    ModuleStatus.UNAVAILABLE,
                    diagnostics={"error": "knowledge_bundle input has an unsupported contract version"},
                )
            validation = self.catalog.validate_artifact(source)
            if not validation.valid:
                return ModuleResult(
                    USG_KNOWLEDGE_MODULE_ID, USG_KNOWLEDGE_MODULE_VERSION,
                    ModuleStatus.UNAVAILABLE,
                    diagnostics={"input_validation_errors": list(validation.errors)},
                )
            resource = self.adapter.convert(source.payload)
            envelope = make_envelope(
                ORE_USG_KNOWLEDGE_CONTRACT,
                context.run_id,
                USG_KNOWLEDGE_MODULE_ID,
                config={"include_candidate_states": self.adapter.lexical_adapter.include_candidate_states},
                evidence_kind=EvidenceKind.DERIVED,
                parent_artifact_ids=(source.payload_hash,),
                data_hash=source.payload_hash,
                model_version=USG_KNOWLEDGE_MODULE_VERSION,
            )
            artifact = ContractArtifact(
                envelope,
                resource,
                metadata={
                    "usg_bundle_id": resource.bundle_id,
                    "usg_namespace": resource.namespace,
                    "usg_bundle_digest": resource.bundle_digest,
                    "candidate_states_enabled": self.adapter.lexical_adapter.include_candidate_states,
                    "runtime_mutation_performed": False,
                    "source_database_mutation_performed": False,
                },
            )
            return ModuleResult(
                USG_KNOWLEDGE_MODULE_ID,
                USG_KNOWLEDGE_MODULE_VERSION,
                ModuleStatus.SUCCESS,
                artifacts=(artifact,),
                diagnostics={
                    "output_count": 1,
                    "capability_support": {
                        capability: 1.0 for capability in USG_KNOWLEDGE_CAPABILITIES
                    },
                    "capability_support_basis": (
                        "successful typed import and output validation; not semantic truth or completeness"
                    ),
                    "bundle_digest": resource.bundle_digest,
                },
                provenance={
                    "source_artifact_hash": source.payload_hash,
                    "read_only": True,
                    "source_database_mutation_performed": False,
                },
            )
        except Exception as exc:
            return ModuleResult(
                USG_KNOWLEDGE_MODULE_ID,
                USG_KNOWLEDGE_MODULE_VERSION,
                ModuleStatus.FAILED,
                diagnostics={"exception_type": type(exc).__name__, "error": str(exc)},
                warnings=("Unified USG import failed; no output artifact was emitted.",),
                provenance={"read_only": True},
            )


def register_usg_knowledge_module(
    registry: ModuleRegistry,
    *,
    adapter: USGKnowledgeAdapter | None = None,
    observability: USGSQLiteObservability | None = None,
) -> USGKnowledgeModuleBackend:
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
    backend = USGKnowledgeModuleBackend(
        manifest, catalog, adapter or USGKnowledgeAdapter(), observability=observability,
    )
    binding = registry.bind_backend(USG_KNOWLEDGE_MODULE_ID, backend)
    if binding.module_id != USG_KNOWLEDGE_MODULE_ID:
        raise RuntimeError("ORE registry returned an unexpected unified USG module binding")
    register_usg_knowledge_access_module(registry, observability=observability)
    return backend


__all__ = [
    "USGKnowledgeModuleBackend",
    "USGKnowledgeAccessModuleBackend",
    "USG_KNOWLEDGE_ACCESS_MODULE_ID",
    "USG_KNOWLEDGE_ACCESS_MODULE_VERSION",
    "USG_KNOWLEDGE_ACCESS_CAPABILITIES",
    "build_usg_knowledge_access_manifest",
    "register_usg_knowledge_access_module",
    "make_usg_knowledge_query_artifact",
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
