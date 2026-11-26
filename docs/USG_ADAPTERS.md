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
preferred frame version. Exact `context_by_id`, `evidence_by_id`, and
`source_by_id` methods expose preserved context, evidence-support status, and
source-rights summaries without changing or re-evaluating them.

This index is a deterministic retrieval layer. It does not yet provide
context-weighted ranking, learned confidence, or coordinator-driven routing.


## Adaptive coordinator integration

The unified importer declares the four capabilities
`language.lexical_resource.import`,
`language.semantic_resource.import`,
`language.semantic_structure.import`, and
`language.knowledge.provenance`. Its protocol backend reports operational
completion for these capabilities only after it validates the input and
emits the typed output artifact. The diagnostic field explicitly states that
this is import completion, not semantic truth, factual confidence, or knowledge
coverage.

A bounded task profile and task can be built and submitted to the existing
adaptive coordinator:

    from ore_usg import coordinate_usg_knowledge_import

    result = coordinate_usg_knowledge_import(
        adaptive_coordinator,
        bundle_artifact,
        task_id="import-usg-knowledge-001",
        context=run_context,
    )

The helper requests all four capabilities, supplies the bundle artifact to
`ore.usg_knowledge_import`, expects `ORE.USG.KNOWLEDGE`, and defaults to
one module call with no replanning. The adaptive coordinator still decides
runnability from the registry, backend binding, contract compatibility, and
resource budget. This is an optional recruitment path; it does not alter
canonical N9 processing or cause USG to be loaded for unrelated tasks.


## Direct ORE processor bridges

The unified resource can initialize the existing ORE lexical-access and
semantic-composition processors without replacing their algorithms:

    from ore_usg import (
        build_lexical_access_processor,
        build_semantic_composition_processor,
    )

    lexical_processor = build_lexical_access_processor(knowledge)
    semantic_processor = build_semantic_composition_processor(knowledge)

The lexical processor receives the imported `LexicalEntry` inventory. The
semantic-composition processor receives imported `SemanticConcept` records
and predicate frames converted only from explicit role declarations. Frame
conversion rejects unsupported role structures rather than silently using
ORE's default roles. ORE semantic feature maps remain empty unless USG supplied
an explicit representation supported by the destination contract; labels and
definitions are not converted into invented numeric features.

These helpers create processors; they do not run a request or mutate either
inventory. The caller remains responsible for supplying the task's request,
syntax, and contextual evidence.



## Phase 2 — Semantic knowledge integration (roadmap items 4–7)

This section supersedes the earlier "not yet imported" and "not yet compiled"
status notes above. The current implementation adds a complete, typed semantic
knowledge projection alongside the original Phase 1 import records.

### Unified semantic-knowledge API

    from ore_usg import USGKnowledgeAdapter

    snapshot, knowledge = USGKnowledgeAdapter().load_and_convert(
        "exports/knowledge_bundle.json",
    )
    semantic = knowledge.semantic_knowledge

The original `knowledge.semantic`, `knowledge.structures`, and
`knowledge.concepts` APIs remain available for compatibility.
`knowledge.semantic_knowledge` is the complete semantic resource. It
contains the distinct ORE semantic concepts, source-linked category records,
membership assertions, typed relations, and preserved predicate structures.
Its namespace, bundle ID, and digest must match the same immutable source
snapshot as all Phase 1 resources.

### 4. Sense-to-concept adapter

`USGSemanticKnowledgeAdapter` converts both `LEXICAL_SENSE` and
`CONCEPT` records into ORE `SemanticConcept` objects. Every source record
gets its own version-scoped ORE identifier; two senses belonging to one lemma
remain two separate concept records. A sense also carries its referenced lemma as an alias and its
explicit part-of-speech when the lemma passes the same lifecycle and rights policy, allowing ORE to
retrieve multiple distinct sense concepts for one surface word. Each concept retains its definition,
language, source locator, lifecycle state, source IDs, evidence IDs, context
IDs, and canonical source payload in the companion
`USGSemanticConceptRecord`.

    semantic.concepts
    semantic.concept_records
    semantic.active_concepts

`active_concepts` excludes unresolved records and candidate records unless
the caller opted into candidates. Feature maps remain empty unless supplied
explicitly in a supported format; labels and definitions are not converted
into invented semantic activations. Equivalent, overlapping, related, and
unresolved mappings remain separate directed knowledge relations. None of
these mapping types merges concept identities.

### 5. Semantic categories and memberships

Every eligible `SEMANTIC_CLASS` becomes an ORE `SemanticCategory`, with its
original class key, definition, lifecycle state, provenance, and payload.
Parent keys are resolved only when they identify exactly one imported
category. Missing or ambiguous parent keys stay in
`unresolved_parent_keys`; no parent edge is guessed.

    semantic.categories
    semantic.memberships

Membership assertions are created only from explicit relation predicates
(`INSTANCE_OF`, `CLASSIFIED_AS`, `MEMBER_OF`, or `HAS_CLASS`) whose
endpoints identify exactly one category and one member. Both the normalized
member/category identifiers and the original directed subject/object
orientation are retained. Membership status explicitly records
`ACTIVE`, `CANDIDATE`, `UNRESOLVED`, `CONFLICTED`, `CONTRADICTED`,
or `UNASSESSED`. An assertion is marked conflicted only when its own
evidence contains both `SUPPORTS` and `CONTRADICTS`; multiple memberships
alone are not treated as contradictions.

### 6. Concept relations

Source `relations` and `mappings` are converted into typed ORE
`SemanticRelation` records:

    semantic.relations

Each relation preserves its source subject, object, direction, raw predicate
or mapping type, lifecycle state, evidence/context IDs, source reference, and
source kind. A conservative exact-label registry classifies known predicates
as `EQUIVALENCE`, `HIERARCHY`, `ASSOCIATION`, `UNRESOLVED`, or
`OTHER`; the original predicate string is never replaced.

All USG relation rows in this contract are directed. Equivalent relations
therefore remain directed, source-backed assertions; they do not union,
collapse, or overwrite the endpoint concept records. Hierarchy parent IDs are
projected only from explicit `parent_class_keys` or explicit hierarchy
relations. No inverse, transitive, synonym, or other relationship is
fabricated. Unrecognized predicates remain `OTHER`, available for later
interpretation without losing the source label.

### 7. Predicate-frame adapter

Every eligible or unresolved `PREDICATE_FRAME` is retained as a
`USGPredicateStructure`. Its source frame ID/version, lifecycle state,
roles, role descriptions, required flags, role constraints, semantic
relationships, complete payload, and provenance remain available.

    semantic.predicate_frames
    semantic.executable_frames

The adapter creates a native ORE `PredicateFrame` only when its roles,
required-role declarations, and numeric selectional preferences can be
represented by the current ORE contract without discarding declared
semantics. Custom constraints, role relationships, malformed/incomplete
roles, explicit `complete: false`, or `resolved: false` prevent the
executable projection. These records are not discarded: their raw fields
and unresolved-field explanations remain in `USGPredicateStructure`.
Unresolved and inactive frames are never included in
`executable_frames`.

The standard processor bridge now uses the complete active semantic-concept
inventory and only safely executable predicate frames:

    from ore_usg import build_semantic_composition_processor

    processor = build_semantic_composition_processor(knowledge)

The categories, memberships, and relation records remain available through
the same immutable `semantic_knowledge` resource for ORE components that
consume structured semantic knowledge. The existing nine-stage composition
algorithm is not replaced or rewritten by this adapter.

### Focused adapters

For applications that need one part of Phase 2 independently, the package
also exports `USGSenseToConceptAdapter`,
`USGSemanticCategoryAdapter`, `USGConceptRelationAdapter`, and
`USGPredicateFrameAdapter`. They use the same validation, lifecycle, and
source-rights rules as the unified adapter.

### Local verification

The Phase 2 regression tests are stored with the repository for local use.
They are intentionally not executed by this integration workflow. To run
them after downloading the repository:

    python -m pip install -e ".[dev]"
    python -m pytest tests/test_usg_semantic_knowledge_adapter.py -q

The full local suite remains available with `python -m pytest -q`.



## Phase 3 — Connect language and meaning (roadmap items 8–11)

The unified importer now assembles a typed cross-resource layer at
`knowledge.language_integration` and projects stable link information onto
the ordinary ORE lexical entries and meanings. The older Phase 1/2 report
properties remain available.

### 8. Link lexical entries and meanings to semantic concepts

    semantic_links = knowledge.lexical_concept_links
    link_groups = knowledge.lexical_concept_groups

A `LEXICAL_SENSE` record that produced both an ORE `LexicalMeaning` and an
ORE `SemanticConcept` receives an identity link using the same version-scoped
USG record ID. Additional links are created only when an explicit USG
`relations` or `mappings` row connects a lexical record to a semantic
`LEXICAL_SENSE` or `CONCEPT` record.

Links are separate source-backed directed records. One lexical meaning may
link to multiple concepts, and multiple meanings/entries may link to one
concept. The importer does not select one preferred target or merge concepts.
Each link keeps the original USG subject and object IDs, predicate or mapping
type, source kind, lifecycle state, evidence/context IDs, and source locator.

A missing link is represented by an explicit `MISSING` record. Entry-level
and meaning-level link groups distinguish `LINKED`, `AMBIGUOUS`,
`MISSING`, `CONFLICTED`, and `UNRESOLVED` states, while individual links
also retain `CANDIDATE`, `CONTRADICTED`, `NEUTRAL`, and `UNASSESSED`
evidence states where applicable.

The imported `LexicalEntry` and `LexicalMeaning` records expose
`semantic_concept_ids` and `semantic_link_status`, so existing ORE lexical
processors can read their concept candidates without a new call signature.
The complete link edges and groups remain available from
`knowledge.language_integration`.

### 9. Explicit surface-form and lemma morphology

    morphology = knowledge.morphology_links

A form-to-lemma link is created only if a USG relation directly connects a
`LEXICAL_FORM` record and a `LEMMA` record with a recognized predicate and
direction. Recognized form-to-lemma predicates are `FORM_OF`,
`INFLECTED_FORM_OF`, `SURFACE_FORM_OF`, `MORPHOLOGICAL_FORM_OF`, and
`DERIVED_FORM_OF`. Recognized reverse predicates include `HAS_FORM`,
`HAS_INFLECTED_FORM`, `HAS_SURFACE_FORM`, and
`HAS_MORPHOLOGICAL_FORM`.

Unrecognized relation labels between a form and a lemma remain unresolved;
they do not produce an executable lemma selection. A form with no explicit
USG link receives a `MISSING` status. If a form is explicitly related to
more than one lemma, all alternatives remain present and the lexical entry's
morphology status is `AMBIGUOUS`; no arbitrary lemma is chosen.

Language, script and extra grammatical payload fields are preserved. ORE
`LexicalEntry` records expose `lemma_entry_id`,
`morphology_link_status`, `script`,
`grammatical_information_json`, and
`morphology_source_reference`. `lemma_entry_id` is populated only for a
single unambiguous linked lemma.

This phase does not infer pluralization, conjugation, derivation, spelling
variants, or other morphological relations from string similarity.

### 10. Contextual knowledge

    contexts = knowledge.contextual_records
    sense_contexts = knowledge.contexts_for(meaning_id)

USG context rows become typed ORE `ContextualRecord` records retaining their
stable context ID, context type, value, language profile, and scope.
`ContextAssignment` records preserve each explicit `context_ids` attachment
to a lexical, semantic, relation, or mapping record. Context values are not
turned into invented numeric feature activations.

ORE lexical-access, lexical-context, semantic-cognition, and
semantic-composition requests now accept typed `contextual_records`,
`evidence_records`, and `source_provenance` fields. Processor results
retain these inputs in their metadata. This makes the source-backed context
available to the processing and evaluation stages without silently changing
the existing numerical context-feature algorithms. A later phase can define
validated mappings from specific context types into those numerical feature
spaces.

### 11. Evidence and provenance

    evidence = knowledge.evidence_records
    sources = knowledge.source_provenance
    sense_evidence = knowledge.evidence_for(meaning_id)

Every evidence summary becomes an `EvidenceRecord` with its evidence type,
locator/digest, capture time, transformation IDs and one of
`SUPPORTS`, `CONTRADICTS`, `NEUTRAL`, or `UNASSESSED`.
`EvidenceAttachment` records link evidence to the exact record, relation,
or mapping that cites it.

Evidence records also carry the associated source's rights status,
redistribution status, license ID and local-reasoning eligibility. A source
with restricted/unknown/prohibited local-use rights cannot silently make its
evidence locally usable. The separate `SourceProvenance` record preserves
provider, source type, attribution, locator, retrieval date, content digest,
license, rights and redistribution status. The `redistributable` property
is true only when both source rights and redistribution explicitly permit it.

An imported evidence summary always has `verified_fact=False` and the
interpretation marker `EVIDENCE_SUMMARY_NOT_VERIFIED_FACT`. ORE request and
result metadata use the same explicit `evidence_is_verified_fact=False`
marker. Evidence is available for reasoning and evaluation, but it is not
upgraded into a verified fact, and this adapter does not adjudicate source
contradictions.

### Retrieval and local verification

The exact `USGKnowledgeIndex` adds:

    index.lexical_concept_links_for(entry_id)
    index.concepts_for_lexical_entry(entry_id)
    index.morphology_for_form(form_entry_id)
    index.contexts_for(record_id)
    index.evidence_for(record_id)
    index.source_provenance_by_id(source_id)

Run the focused Phase 3 regression tests locally:

    python -m pip install -e ".[dev]"
    python -m pytest tests/test_usg_language_integration.py -q

The full local suite remains available with `python -m pytest -q`.
No automated tests are run by the Phase 3 integration workflow.
