# Phase 2 — USG semantic knowledge integration

**Repository:** `wizekin/ORE-Neurocomputational`  
**Phase objective:** connect USG concepts, semantic classes, explicit relations, concept mappings, and predicate frames to ORE sentence-level semantic composition.

## Implementation

Phase 1 supplies validated, read-only USG bundle loading, a unified snapshot, lifecycle/rights filtering, ORE concept records, semantic classes, predicate frames, and explicit relation/mapping records.

Phase 2 adds `build_usg_semantic_composition_processor`, which wraps ORE's existing `SemanticCompositionProcessor`:

- Imported USG concepts are the processor's concept inventory.
- Explicitly converted USG predicate frames are the processor's predicate-frame inventory.
- Exact frame-key/token matches are exposed in the sentence graph with frame version, role inventory, required roles, selectional preferences, and provenance.
- Selected concepts and exact surface-form matches ground USG concept, lexical-entry, and lexical-sense nodes to sentence token nodes.
- Semantic-class nodes are attached only when an explicit imported relation links the class to a record grounded in the current sentence.
- Explicit relation edges are attached only when both endpoints are present in the sentence-local knowledge graph. Predicate and direction are preserved.
- Explicit concept-mapping edges are attached only when both mapped concepts are selected in the current sentence.
- The enriched graph is validated again for dangling references.

The overlay appears at `result.sentence_representation["usg_knowledge_graph"]`; its nodes and edges are also included in the parent sentence graph.

## Safety and provenance invariants

- The source bundle and imported resources are never mutated.
- Candidate/restricted records remain excluded by the existing import policy unless the caller explicitly enables candidate states where permitted.
- Labels and definitions are not converted into invented semantic or modality feature values.
- No inverse, transitive, or other inferred relations are generated.
- Explicit USG relations are represented as source-backed knowledge, not as experimentally verified truth about the sentence.
- The existing nine-stage ORE semantic-composition algorithm remains the execution engine.

## Acceptance tests

`tests/test_usg_processor_bridges.py` now exercises the USG-aware composition path with a fixture that includes a selected concept, a semantic class, an explicit `INSTANCE_OF` relation, a typed concept mapping, and a versioned predicate frame. It checks that the expected graph nodes and edges appear, graph references remain valid, provenance identifies the same bundle digest, and the input bundle is unchanged.

A second test confirms that unrelated relations are not copied into a sentence's local knowledge graph.

Run:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_usg_processor_bridges.py -q
python -m pytest -q
```

## Scope boundary

Phase 2 makes the imported resources available to ORE's sentence-level semantic graph. It does not add learned semantic feature vectors, infer missing relations, resolve all ambiguities, or establish empirical accuracy. Those require separate evidence, model design, and evaluation phases.
