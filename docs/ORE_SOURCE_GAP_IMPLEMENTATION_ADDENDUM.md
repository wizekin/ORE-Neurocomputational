# ORE Source Gap Implementation Addendum

**Branch:** `implement-eight-source-gaps`  
**Base:** `main` at `895207eb8b6b9ab4f2977dc6db0acfb6e4089d80`  
**Scope:** Only gaps and incomplete source-to-code links identified in the eight source specifications and the baseline crosswalk. Existing aligned subsystems and stable operation IDs were preserved.

This addendum supplements `ORE_SOURCE_TO_CODE_IMPLEMENTATION_CROSSWALK.md`. It records what was added on this branch; it does not claim that software analogues prove the corresponding human neurobiology.

## Implementation summary by source

| Source subsystem | Changes on this branch | Source traceability / status |
|---|---|---|
| Working memory and cognitive control | Added explicit semantic prediction-error evaluation, accepted-interpretation consolidation, and an update-permission gate. Added source mapping while preserving WMCC-001..040. | 39 source operations map to canonical WMCC IDs or WMX-001..003 extensions. |
| Semantic composition | No rewrite or duplicate implementation. | Existing nine-stage subsystem remains the source-aligned implementation. |
| Syntactic parsing | Added resolved-dependency-memory release and nested/competing filler-gap handling. | Stage 9 now exposes SYN-057 and SYN-058; the canonical parser trace has 58 operations. |
| Lexical access and morphology | Added explicit caller-supplied derivational/inflectional affix classes, separate retrieval collections and separate morphology support values. | Existing BLA-001..036 IDs remain stable; unclassified affixes remain unspecified rather than guessed. |
| Contextual prediction / orthography | Retained observed transition counts; added 38-step source mapping, per-operation intermediate-output evidence and a validated residual/uncertainty feedback path for a subsequent processing cycle. | MPO-001..040 remain stable; each operation now has source-step IDs and operation-specific output evidence. |
| Semantic and lexical memory | Added a separate SQLite memory store for episodes and prototypes; connected explicit learning, episodic recall, pattern completion, semantic feedback, prototype consolidation, ordered phonological prototypes and generalization into semantic retrieval. | SM-001..016 map source steps 29..44; all 44 source step headings are catalogued. |
| Integrated word recognition and semantic prediction | No rewrite of the existing 40-operation subsystem. | Existing LC-001..040 implementation remains the source-aligned path. |
| Integrated reading | Added acoustic framing and heuristic speech-interval proposals, a pluggable speech-output adapter, and discrete next-reading-unit attention planning. | Implements adapters for IRA-OP-039, 046, 060 and 061 without claiming ASR, phoneme decoding, TTS synthesis or eye tracking. |

## New and changed implementation files

### Working memory
- `ore_working_memory/source_mapping.py` — 39 source operations mapped to repository operations.
- `ore_working_memory/pipeline.py` — WMX-001 semantic prediction error, WMX-002 accepted-interpretation consolidation and WMX-003 task-update gate; per-operation source IDs are included in trace rows.
- `ore_working_memory/__init__.py` — exports the source map and extension registry.
- `tests/test_working_memory.py` — tests source mapping, explicit prediction-error evidence, abstention-aware consolidation and task-update permission.

The existing WMCC registry remains at 40 operations. WMX extensions are separately namespaced so existing operation IDs and downstream consumers do not shift. Prediction error is marked `not_observed` when no prediction evidence was provided. Consolidation is an active working-memory record only; it is not durable episodic learning.

### Syntax
- `ore_syntax/registry.py` — Stage 9 now includes SYN-057 and SYN-058.
- `ore_syntax/pipeline.py` — releases only strongly resolved dependency retrieval records; retains weak/open records; marks nested intervals and groups shared-filler/shared-head competition while preserving candidate gap sites.
- `tests/test_syntax.py` — covers the two new operations and their outputs.

The release threshold is an explicit software policy parameter (currently 0.75), not a measured biological threshold.

### Lexical access and morphology
- `ore_lexical_access/pipeline.py` — accepts `morpheme_affix_categories`, validates the labels against caller-supplied prefix/suffix entries, preserves separate derivational and inflectional collections and reports class-specific support.
- `tests/test_lexical_access.py` — covers separated classes and the no-guessing behavior for unspecified affixes.

The package does not infer a universal derivational/inflectional label from spelling alone. The inventory owner must supply the classification.

### Predictive orthography
- `ore_n3/transitions.py` — retains unigram and bigram observation counts and the smoothing parameter alongside fitted probabilities.
- `ore_predictive_orthography/source_mapping.py` — maps all 38 source steps (1.1 through 7.7) to MPO operations.
- `ore_predictive_orthography/pipeline.py` — replaces stage-summary replay in operation rows with operation-specific evidence summaries and implements `feedback_for_next_cycle()` plus validated `process(feedback=...)`.
- `tests/test_predictive_orthography.py` — tests trace evidence, source coverage, feedback range/shape and second-cycle consumption.

This processor still starts from an already segmented and ordered `N3Sequence`; it does not become a raw-image character-segmentation pipeline through these changes.

### Semantic and lexical memory
- `ore_semantic/memory.py` — adds `SemanticMemoryStore`, SQLite episode/prototype/ledger tables, pattern separation, pattern completion, recall, semantic-feedback logging, ordered phonological prototype consolidation and prototype generalization.
- `ore_semantic/source_mapping.py` — catalogs all 44 source steps and links source steps 29..44 to SM-001..016.
- `ore_semantic/pipeline.py` — optionally uses stored episodes and consolidated prototype matches during inference; records memory operation trace and prototype evidence separately.
- `ore_semantic/__init__.py` — exports the memory store, prototype evidence contract and source map.
- `tests/test_semantic_memory.py` — covers persistence, separation, completion, prototype consolidation, generalization and semantic-pipeline integration.

Learning remains explicit through `learn_word_form_episode()` and `consolidate_memory()`. Ordinary recognition reads but does not mutate episodic memory or prototypes.

### Integrated reading
- `ore_reading/modalities.py` — implements overlapping-window acoustic feature extraction, conservative energy-based speech-time segmentation hypotheses, priority-based next-unit planning and a caller-supplied speech-synthesis backend contract.
- `ore_reading/__init__.py` — exports the adapters and contracts.
- `tests/test_reading_modalities.py` — covers feature extraction, interval proposals, invalid inputs, attention planning and backend dispatch.

The speech segmentation output is a timing hypothesis, not a transcript. The speech-output adapter dispatches only when phoneme sequences and a compatible backend are supplied. The attention controller selects a discrete unit and does not measure gaze or execute eye movements.

## Explicitly unchanged subsystems

The source-aligned semantic-composition subsystem and the integrated lexical-context / word-recognition subsystem were not rewritten. They were already classified in the baseline audit as strong matches. Rewriting them would have introduced duplication rather than closing an identified gap.

## Verification boundaries

- New and modified tests have been added under `tests/`; the repository's GitHub Actions workflows install `.[dev]` and run `python -m pytest`.
- Test execution status must be read from the pull request's checks. The presence of a test file is not evidence that it passed.
- The LIF reference tracks remain computational analogies. This branch adds no claim of anatomical validation, calibrated neural parameters, or agreement with neural recordings.
- Raw visual OCR segmentation, full ASR/phoneme decoding, an actual TTS engine, and physical eye-movement control remain outside the implemented adapters.
- The branch is intentionally not merged into `main`; review and passing CI are the merge gates.
