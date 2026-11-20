# ORE USG adapters

## Initial delivery: read-only lexical import

The initial package lives in ore_usg and consumes decoded USG-KNOWLEDGE/0.1.0
bundle mappings. It does not connect to, open, or write the USG SQLite
repository. The caller supplies a snapshot.

The main entry point is USGKnowledgeLexicalAdapter:

    from ore_usg import USGKnowledgeLexicalAdapter

    resource = USGKnowledgeLexicalAdapter().convert(knowledge_bundle)
    ore_lexicon = resource.entries

The output entries are ordinary ore_lexical_context LexicalEntry records and
can be passed to LexicalAccessProcessor or LexicalContextProcessor. The
USGLexiconResource also retains the bundle ID, namespace, SHA-256 digest,
adapter version, source-rights summary (including license and source locator),
evidence summaries (including contradictions), context summaries, imported
record IDs, exclusions, and warnings. Evidence text itself is not copied into
the report; stable locators and digests remain available where supplied.

## Lifecycle and rights policy

By default, only records in ACCEPTED or VALIDATED_FOR_PROFILE states are
eligible. CANDIDATE records can be imported only by constructing the adapter
with include_candidate_states=True. UNRESOLVED, DEPRECATED, and REJECTED
records are never imported as active lexical entries.

A record is excluded when any contributing source has rights_status other
than KNOWN_PERMITTED or INTERNAL, or when the record has no source/evidence
provenance. Rights and redistribution fields remain in the import report.
This internal adapter does not grant redistribution rights.

LEXICAL_SENSE records attach to their referenced LEMMA only when the languages
match. Every eligible sense becomes a separate LexicalMeaning so polysemy is
preserved. Evidence support/contradiction states and context descriptors remain
available in the resource report; the adapter does not select a winner when
evidence conflicts. Definitions are
copied into gloss fields only; they are not converted into numeric semantic
features. ORE's neutral frequency default is used when USG provides no
positive finite frequency. This default is not an observed frequency.

LEXICAL_FORM records become surface-form-only entries because the base
USG-KNOWLEDGE/0.1.0 record kind does not require a lemma link. The adapter
does not guess a lemma, inflection, or semantic sense. Morphology and
form-to-lemma mapping are separate future adapters.

## ORE module registration

The importer can also be registered in the ORE module registry:

    from ore_contracts import ContractCatalog, ModuleRegistry
    from ore_usg import make_usg_bundle_artifact, register_usg_lexical_module

    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    register_usg_lexical_module(registry)
    bundle_artifact = make_usg_bundle_artifact(
        knowledge_bundle,
        run_id="example-run",
    )

make_usg_bundle_artifact creates a detached ContractArtifact with the exact
USG input contract version (0.1.0); the general ORE envelope helper defaults
to 1.0.0 and should not be used for this input. The importer registers
ORE.USG.KNOWLEDGE_BUNDLE@0.1.0 as its input contract and ORE.USG.LEXICON@1.0.0
as its output contract. The resulting module ID is
ore.usg_lexical_import. Its backend accepts one ContractArtifact at the
knowledge_bundle input and emits one provenance-tracked ContractArtifact.

## Validation and limitations

validate_knowledge_bundle implements dependency-light structural and
cross-reference checks for the USG knowledge contract fields needed by this
adapter. It does not replace the complete Draft 2020-12 JSON-Schema validator
maintained in USG-Workbench and does not assess semantic truth.

The initial scope deliberately excludes concept mappings, semantic classes,
predicate frames, morphology rules, relations, and semantic-document import.
Those should be added in later adapters with their own explicit compatibility
rules. No USG resource is promoted, edited, or written back by this package.

Run the focused tests:

    python -m pytest tests/test_usg_lexical_adapter.py


## Read-only bundle discovery and loading

The package also exposes a deterministic JSON loader:

    from ore_usg import load_knowledge_bundle

    snapshot = load_knowledge_bundle("exports/knowledge_bundle.json")
    lexical_resource = USGKnowledgeLexicalAdapter().convert(snapshot.bundle)

A directory may be supplied when it contains exactly one of
`knowledge_bundle.json`, `usg_knowledge_bundle.json`, or `bundle.json`.
If more than one conventional filename is present, discovery fails and asks
the caller to select a file explicitly. The loader rejects duplicate JSON
keys, non-finite JSON constants, invalid bundle structure, digest mismatches,
and files larger than the configured limit. It returns a detached snapshot
with the canonical SHA-256 digest and source path. The loader never connects
to or writes to the USG database.

    snapshot = load_knowledge_bundle(
        "exports",
        expected_digest="<expected-canonical-sha256>",
        max_bytes=64 * 1024 * 1024,
    )

The canonical digest is calculated over sorted, compact UTF-8 JSON, matching
the lexical adapter's bundle fingerprint. This is an integrity check, not a
digital signature or proof of source authenticity.


## Shared import policy and semantic concept adapter

`USGImportPolicy` centralizes the lifecycle-state and source-rights rules used
by lexical and semantic imports. Accepted and profile-validated records are
eligible by default; candidate records require explicit opt-in. Local import
requires every direct or evidence-derived source to declare
`KNOWN_PERMITTED` or `INTERNAL` rights.

The first semantic adapter is available through:

    from ore_usg import USGConceptAdapter

    semantic_resource = USGConceptAdapter().convert(snapshot.bundle)

It imports eligible `CONCEPT` records as ORE `SemanticConcept` objects and
preserves eligible USG mappings as separate typed `USGConceptMapping`
records. Mapping endpoints must both be imported concepts. The adapter does
not merge concepts, even for `EQUIVALENT` mappings, and does not infer
semantic or modality feature values from labels or definitions. The import
report records imported IDs, exclusions, and explicit limitations.

This adapter does not yet import `SEMANTIC_CLASS` or `PREDICATE_FRAME`
records, nor does it register a separate semantic module backend in ORE's
module registry.


## Semantic classes, predicate frames, and cross-resource relations

`USGSemanticStructureAdapter` imports `SEMANTIC_CLASS` and
`PREDICATE_FRAME` records using the same lifecycle and source-rights policy.
Class and frame payloads are retained as canonical JSON so additional USG
fields are not discarded. Predicate-frame roles are preserved verbatim as
canonical JSON; they are not yet compiled into executable ORE argument
validation or inference rules.

Explicit directed USG relations are retained when both endpoints are active,
rights-permitted records supported by the lexical, concept, or structure
adapters. The relation's predicate, direction, lifecycle, evidence IDs, and
context IDs are preserved. No inverse, transitive, or inferred relations are
generated.

## Unified knowledge resource

The current adapters can be coordinated through one read-only entry point:

    from ore_usg import USGKnowledgeAdapter

    importer = USGKnowledgeAdapter()
    snapshot, knowledge = importer.load_and_convert(
        "exports/knowledge_bundle.json",
        expected_digest="<canonical-sha256>",
    )

The returned `USGKnowledgeResource` contains the lexical resource, semantic
concept resource, semantic structures, and cross-resource relations. It also
exposes the source, evidence, and context summaries retained in the lexical
import report. All component resources must share the same namespace, bundle
ID, and digest; construction rejects mixed snapshots.

This is a unified data-access resource, not yet a registered multi-output ORE
module. The current lexical module registration remains available separately.


## ORE module-registry integration

For ORE workflows that need one coordinated output artifact, register the
unified importer:

    from ore_contracts import ContractCatalog
    from ore_contracts.module_registry import ModuleRegistry
    from ore_usg import register_usg_knowledge_module

    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    register_usg_knowledge_module(registry)

The module ID is `ore.usg_knowledge_import`; its output contract is
`ORE.USG.KNOWLEDGE@1.0.0`. It consumes the same versioned
`ORE.USG.KNOWLEDGE_BUNDLE@0.1.0` input as the lexical importer and emits one
`USGKnowledgeResource` artifact with the input payload hash in its lineage.
The existing lexical-only module remains available for consumers that need
only an ORE lexicon.

The unified resource currently exposes linked data and provenance. It does
not yet implement runtime retrieval APIs, connect to the Main Brain
Coordinator automatically, or compile predicate frames into executable
reasoning operations.


## Deterministic retrieval API

The unified resource can be queried without fuzzy matching or invented
confidence values:

    from ore_usg import USGKnowledgeIndex

    index = USGKnowledgeIndex(knowledge)
    result = index.lookup("bank", kind="LEXICAL_ENTRY", language="en")
    for item in result.items:
        print(item.record_id, item.label, item.source_reference)

Supported exact lookup kinds are `LEXICAL_ENTRY`, `SEMANTIC_CONCEPT`,
`SEMANTIC_CLASS`, and `PREDICATE_FRAME`. Matching trims surrounding
whitespace and applies Unicode case-folding only; it does not stem, tokenize,
guess synonyms, or perform fuzzy similarity. Results retain source references
and meaning/frame details. `relations_for(record_id)` returns only explicit
directed relations incident to the supplied namespaced ORE record ID.
`frames_by_key(frame_key)` returns all exact versions; it does not choose a
preferred frame version.

This index is a deterministic retrieval layer. It does not yet provide
context-weighted ranking, learned confidence, or coordinator-driven routing.
