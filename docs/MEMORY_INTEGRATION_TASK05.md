# Task 5 — Connect memory to every relevant cognitive structure

## Purpose

The `ore_memory_integration` package connects existing ORE memory stores to the versioned shared workspace. It does not replace the episodic/semantic store in Point 7, the lexical memory in `ore_semantic`, the working-memory processor, schema learners, analogical transfer, or reconstructive-memory processor.

The system has three distinct layers:

1. **Temporary state:** `ore_working_memory` maintains the current input, partial structures, active goals, unresolved dependencies and short-lived references.
2. **Long-term records:** Point 7's `Point7MemoryStore` retains episodic traces, semantic candidates and outcome feedback in its own SQLite database. The lexical form memory in `ore_semantic` remains a complementary specialist store.
3. **Communication and integration:** `ore_workspace` publishes normalized memory evidence, versions updates and dispatches changes only to subscribed cognitive modules.

The logger's `ore_logger.sqlite3` is not used as the long-term memory database. It remains the annotation and execution-observation catalogue.

## Processing sequence

```text
current input + goal
        |
        v
initial WorkingMemoryProcessor pass
        |
        v
construct lexical / event / context / goal / temporal cues
        |
        v
read-only episodic + semantic retrieval from Point7MemoryStore
        |
        v
rank evidence and preserve source / verification metadata
        |
        v
publish MemoryEvidence to SharedWorkspace
        |
        +--> semantic processor
        +--> schema formation
        +--> schema-guided inference
        +--> analogical transfer
        +--> reconstructive memory
        +--> cognitive control and discourse
        +--> working-memory overload monitoring
        |
        v
select bounded references using WorkingMemoryPolicy
        |
        v
one memory-informed working-memory pass
        |
        v
optional, explicit encoding of caller-supplied observed content
```

The recall adapters read `episodes()` and `semantic_records()` directly. They deliberately do not call `Point7MemoryDecisionProcessor.process()` as a retrieval shortcut because that pipeline can encode before it reaches its retrieval stage. Records with the current `sequence_id` are excluded. Optional new episode encoding is considered only after recall and the memory-informed working-memory pass.

## Mathematical operations

### Episodic ranking

For episode (m) and query (q):

[
R_e(m,q)=rac{w_lL+w_rR+w_cC+w_gG+w_tT+w_sS}{w_l+w_r+w_c+w_g+w_t+w_s}.
]

- (L): transparent lexical Jaccard overlap.
- (R): overlap between caller-supplied event cues and stored role/relation structure.
- (C): context-cue compatibility.
- (G): goal compatibility.
- (T): temporal compatibility, using exponential decay when both timestamps are available and a neutral value otherwise.
- (S): stored trace strength.

### Semantic ranking

[
R_s(m,q)=rac{v_cK+v_rR+v_gG+v_qQ+v_xX}{v_c+v_r+v_g+v_q+v_x}.
]

Here (K) is concept overlap, (R) relation overlap, (G) goal compatibility, (Q) verification/source quality, and (X) context compatibility. Pending semantic candidates can be retrieved for review but remain separate from verified semantic knowledge. Retrieval scores rank records; they are not truth probabilities.

### Working-memory retention

The adapter reuses the workspace's normalized retention policy:

[
P_i=rac{0.35G_i+0.25R_i+0.25C_i+0.15U_i}{0.35+0.25+0.25+0.15}.
]

Only bounded references and supporting metadata are passed downstream; long-term records are not copied into temporary buffers.

### Independence-aware evidence

For evidence contributions (E_i(h)), records with the same source-independence group are not added as if independent. The implementation retains one strongest contribution per group before scoring:

[
S(h)=sum_{gin G}w_gE_g(h)-lambda C(h)-mu U(h).
]

(C(h)) is an explicit conflict count and (U(h)) an unresolved-evidence count. This is an inspectable evidence score, not a calibrated posterior probability. Candidate selection remains the responsibility of the receiving cognitive module and its task-specific thresholds.

### Bounded feedback

The coordinator performs at most two working-memory passes: the initial pass and one memory-informed pass. Workspace callbacks only construct handoffs; they do not republish findings. This avoids a recursive notification loop.

## API

```python
from ore_language_comprehension.point07 import Point7MemoryStore
from ore_memory_integration import (
    MemoryIntegrationCoordinator,
    MemoryIntegrationConfig,
)

store = Point7MemoryStore("ore_point7_memory.sqlite3")

with MemoryIntegrationCoordinator(
    store,
    config=MemoryIntegrationConfig(max_retrieval_items=12, max_active_memories=8),
) as memory:
    result = memory.process(
        "Lina adopted a puppy at the animal shelter.",
        goal="recall adoption event",
        sequence_id="conversation-001",
        event_cues={"agent": "Lina", "predicate": "adopted", "patient": "puppy"},
        context_cues={"place": "animal shelter"},
    )

    print(result.memory_batch.to_dict())
    print(result.handoffs.keys())
    print(result.workspace_snapshot["state"]["retrieved_memories"])

store.close()
```

The coordinator accepts caller-owned stores/processors and does not close them. When no store is supplied, it creates a separate persistent `ore_memory.sqlite3`; close the coordinator to release resources it owns. Its logger annotations are registered in `ore_logger.annotations`, honoring `ORE_LOGGER_DB` when set.

## Module routing

The workspace subscription matrix now connects working memory, semantic processing, schema formation, schema-guided inference, analogy, reconstructive memory, cognitive control, discourse and overload monitoring to relevant changes in `retrieved_memories` and `working_memory`.

Each route produces a `MemoryHandoff` with memory IDs, workspace revision and source-preserving payload. Optional `module_handlers` let an application bind the handoff to a particular running processor. This is intentional: several specialist packages require typed inputs (for example, `EpisodeInput`, `CaseEvidence` or `SituationGraph`), so the adapter does not fabricate domain entities or silently coerce incomplete records into those types.

## Provenance and limitations

- Episodic recall preserves source, confidence, time, context, relations and parent memory IDs.
- Accepted semantic knowledge is exposed separately from pending claims.
- A derived schema, analogy or reconstruction retains its original independence group; it cannot become independent corroboration just by passing through another module.
- Missing relations and event details remain unknown. Pattern completion is not an authorization to invent stored facts.
- Similarity uses deterministic token overlap, not a learned embedding or a validated semantic model.
- The numeric coefficients are transparent prototype defaults, not fitted biological constants.
- The integration layer publishes interoperable evidence envelopes; specialist modules still need caller-provided handlers to run their full typed learning/inference APIs.
- No claim is made that this software is a biologically faithful simulation or a complete memory/AGI system.

## Validation

Run:

```bash
python -m pytest -q tests/test_memory_workspace_integration.py
python -m pytest -q
```

The new tests cover same-sequence exclusion, verification separation, source independence, explicit encoding, workspace publication, module routing and SQLite annotations.
