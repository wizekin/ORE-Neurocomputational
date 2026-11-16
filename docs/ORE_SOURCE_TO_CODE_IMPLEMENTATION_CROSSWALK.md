# ORE Official Source-to-Code Implementation Crosswalk

**Document ID:** ORE-AUDIT-XWALK-001  
**Status:** Initial static-audit baseline  
**Audit date:** 2026-11-16  
**Repository:** [wizekin/ORE-Neurocomputational](https://github.com/wizekin/ORE-Neurocomputational)  
**Branch audited:** \`main\`  
**Observed repository tree:** \`c9f58942c12166e70465512d7588b5c17dd6d077\`  
**Purpose:** Establish a traceable mapping from the eight specified Google Drive algorithm documents to the corresponding ORE registries, functional implementation paths, biological-reference paths, and test evidence.

---

## 1. Executive determination

The repository contains corresponding implementation packages for all eight source topics. However, the source algorithms and current repository registries are not universally identical. Some modules preserve the source stages but expand or merge operations; some registries are generated from stage summaries rather than recording a distinct computation for every listed operation; and some source capabilities are represented only as constrained approximations.

This document therefore does **not** use a matching operation count or registry entry as sufficient proof of implementation. The status of each mapping is based on the source descriptions, the repository's current code paths, the registry/trace design, and the tests present in the repository at the audit baseline.

### 1.1 Source-to-repository status summary

| # | Source document | Source structure | Repository package / registry | Determination |
|---:|---|---:|---|---|
| 1 | Working Memory and Cognitive Control | 39 operations / 5 stages | \`ore_working_memory\`, WMCC-001..040 | **PARTIAL / RESTRUCTURED** |
| 2 | Brain-Inspired Semantic Composition | 9 named stages; detailed stage operations | \`ore_semantic_composition\`, SMC-001..054 | **STRONG STAGE-LEVEL MATCH** |
| 3 | Syntactic Parsing and Structure-Building | 55 operations / 9 stages | \`ore_syntax\`, SYN-001..056 | **PARTIAL / RESTRUCTURED** |
| 4 | Lexical Access and Morphological Processing | 29 operations / 7 stages | \`ore_lexical_access\`, BLA-001..036 | **PARTIAL / EXPANDED** |
| 5 | Contextual Prediction in Orthographic Processing | 38 numbered steps when final Stage 7.7 is included | \`ore_predictive_orthography\`, MPO-001..040 | **PARTIAL; TRACE GRANULARITY GAP** |
| 6 | Human Semantic and Lexical Memory | 44 steps / 7 mechanisms | \`ore_semantic\`, SC-001..044 | **PARTIAL; MEMORY ACQUISITION GAP** |
| 7 | Integrated Word Recognition and Semantic Prediction | 40 operations / 8 subsystems | \`ore_lexical_context\`, LC-001..040 | **STRONG OPERATION-LEVEL MATCH** |
| 8 | Integrated Reading Algorithm — Formal Operation Registry | 62 operations / 8 groups | \`ore_reading\`, IR-001..072 | **PARTIAL / REVISED REGISTRY** |

**Correction to the preliminary audit:** the contextual-prediction document contains 37 headings formatted as “Step 1.x–7.x” plus a final heading “Stage 7.7 — Return diagnostics to the prediction system.” The full source sequence is therefore 38 numbered operations, not 37. The integrated-reading document contains 62 unique IRA-OP identifiers. The semantic-composition source defines nine named stages but does not provide a separate stable, consecutively numbered operation registry in the same form as the other files; SMC-001..054 is the repository's executable decomposition of those nine stages.

### 1.2 Status definitions

| Status | Meaning |
|---|---|
| **I — Implemented** | A functional computation corresponding to the source responsibility is present in the package's execution path. This does not establish scientific or behavioral validity. |
| **M — Merged / many-to-one** | The source responsibility is implemented together with another responsibility, or multiple source operations map to one repository operation. |
| **P — Partial / constrained** | A related computation exists, but important source inputs, distinctions, state changes, or behaviors are absent or approximated. |
| **T — Trace / metadata only** | A registry or trace row exists, but the evidence inspected does not show a distinct operation-specific computation and output for that row. |
| **N — Not found** | No corresponding implementation was identified in the audited package path. |
| **X — Extension** | Repository behavior beyond the cited source requirement; useful, but not evidence that the source requirement itself is satisfied. |

A source step can map to more than one repository operation. One repository operation can also cover several source steps. The crosswalk records these cases rather than forcing artificial one-to-one matches.

---

## 2. Audit scope, method, and boundaries

### 2.1 Source documents inspected

Each source file was located in the connected Google Drive account and retrieved as text.

| Source | Google Drive file |
|---|---|
| ORE Neurocomputational Architecture - Working Memory and Cognitive Control.txt | [Open source](https://drive.google.com/file/d/1WQoZMvhEHyPbcEAhtSit9ljmOax00w4b/view) |
| Master Algorithm - Brain-Inspired Semantic Composition.txt | [Open source](https://drive.google.com/file/d/1EkbfTURhR2K0n7syj_FLb8hz6vzwzA4L/view) |
| Master Algorithm - Syntactic Parsing and Structure-Building in the Human Brain.txt | [Open source](https://drive.google.com/file/d/18ms6EbK8nPlZNKnAVzfVl1I3E1y7zQNB/view) |
| Master Algorithm - Lexical Access and Morphological Processing in the Human Brain.txt | [Open source](https://drive.google.com/file/d/10kUKpRhnxR8VIzBILbmZsBTsoPofW9ck/view) |
| Master Algorithm - Contextual Prediction in Human Orthographic Processing.txt | [Open source](https://drive.google.com/file/d/1EmSbo2z5OtF_e35EtZe6gbJ_Q3qiIkVn/view) |
| Human Semantic and Lexical Memory Algorithm.txt | [Open source](https://drive.google.com/file/d/1yGsoTezrHe2b_oFIKBTIJE97FN0UE26N/view) |
| Integrated Word Recognition and Semantic Prediction Algorithm.txt | [Open source](https://drive.google.com/file/d/1IU4BQwpqSnKBAVEill5FESUDY2Uxu6Bo/view) |
| Integrated Reading Algorithm — Formal Operation Registry.txt | [Open source](https://drive.google.com/file/d/1cfHkGjRavIzPrXpANM-2Gtq73ZvqAlb5/view) |

### 2.2 Evidence inspected

The audit inspected repository file paths, the corresponding registry modules, processing pipelines, relevant biological-reference modules, architecture documents, and focused test files. The main code paths are identified below.

**This is a static audit.** The test files were inspected, but the test suite was not executed as part of this audit. A listed test is evidence that a test case exists, not evidence that it passed in the current environment. No claim of live CI success is made here.

The audit does not establish:
- equivalence between an algorithmic abstraction and actual human neural computation;
- empirical accuracy on independent reading, language, EEG/MEG, fMRI, or behavioral datasets;
- correctness of every possible input or language;
- that all registry rows correspond to separate, independently executed functions;
- that optional biological-reference simulation is enabled in every run.

### 2.3 Shared implementation principle

ORE generally separates a **functional processing path** from an optional **biological-reference path**. The latter uses simplified neural-population models such as current-based leaky integrate-and-fire (LIF) systems. Those simulations are computational analogies, not anatomically fitted or experimentally validated reproductions of the human brain.

---

## 3. Source 1 — Working Memory and Cognitive Control

**Source:** [Google Drive document](https://drive.google.com/file/d/1WQoZMvhEHyPbcEAhtSit9ljmOax00w4b/view)  
**Package:** [\`ore_working_memory/\`](https://github.com/wizekin/ORE-Neurocomputational/tree/main/ore_working_memory)  
**Registry:** [\`ore_working_memory/registry.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/ore_working_memory/registry.py)  
**Functional entry point:** \`WorkingMemoryProcessor.process()\`  
**Main implementation:** [\`ore_working_memory/pipeline.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/ore_working_memory/pipeline.py)  
**Biological reference:** \`WorkingMemoryLIFSimulator\` in \`ore_working_memory/biology.py\`  
**Tests:** [\`tests/test_working_memory.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/tests/test_working_memory.py)

The source specifies five stages with 39 operations: 7 / 7 / 7 / 9 / 9. The repository executes a revised 40-operation sequence with 8 / 6 / 7 / 9 / 10 operations. The principal processing concepts are present, but the source operation sequence is not preserved exactly.

| Source ID* | Source operation | Repository mapping | Code path | Status |
|---|---|---|---|---|
| WM-S1-01 | Receive processed linguistic information | WMCC-002 | \`_op_receive_linguistic_information\` | I |
| WM-S1-02 | Encode information into temporary memory representations | WMCC-003 | \`_op_tokenize_and_encode\` | I |
| WM-S1-03 | Maintain temporary linguistic activation | WMCC-006, WMCC-007 | \`_op_calculate_buffer_activation\`, \`_op_decay_and_capacity\` | M |
| WM-S1-04 | Organize memory according to functional buffers | WMCC-004, WMCC-005 | \`_op_allocate_linguistic_buffers\`, \`_op_chunk_stored_information\` | M |
| WM-S1-05 | Prioritize important information | WMCC-007 | \`_op_decay_and_capacity\` | I |
| WM-S1-06 | Retrieve information from working memory | WMCC-008 | \`_op_retrieve_context_cues\` | I |
| WM-S1-07 | Update and refresh working memory continuously | WMCC-006, WMCC-007 | \`_op_rehearse_active_items\`, \`_op_calculate_buffer_activation\`, \`_op_decay_and_capacity\` | M |
| WM-S2-01 | Detect incomplete syntactic dependencies | WMCC-009 | \`_op_extract_incomplete_structures\` | I |
| WM-S2-02 | Register pending grammatical expectations | WMCC-010 | \`_op_score_slot_retrieval_cues\` | I |
| WM-S2-03 | Maintain unresolved dependencies | WMCC-011 | \`_op_track_unresolved_dependencies\` | I |
| WM-S2-04 | Retrieve the appropriate grammatical element | WMCC-008, WMCC-010 | \`_op_retrieve_context_cues\`, \`_op_score_slot_retrieval_cues\` | M |
| WM-S2-05 | Bind the retrieved element to the new structure | WMCC-013 | \`_op_bind_supported_slots\` | I |
| WM-S2-06 | Detect structural conflicts | WMCC-017, WMCC-018 | \`_op_measure_conflict\`, \`_op_allocate_control\` | M |
| WM-S2-07 | Synchronize syntactic memory with semantic memory | WMCC-014, WMCC-021 | \`_op_emit_incomplete_state\`, \`_op_coordinate_subsystems\` | M |
| WM-S3-01 | Load the current cognitive objective | WMCC-015 | \`_op_maintain_cognitive_goals\` | I |
| WM-S3-02 | Monitor processing conflicts | WMCC-016, WMCC-017 | \`_op_monitor_sentence_progress\`, \`_op_measure_conflict\` | M |
| WM-S3-03 | Estimate required control strength | WMCC-018 | \`_op_allocate_control\` | I |
| WM-S3-04 | Identify which information should receive executive support | WMCC-019 | \`_op_evaluate_goal_fit\`, \`_op_select_control_focus\`, \`_op_regulate_competing_representations\` | M |
| WM-S3-05 | Protect relevant information from interference | WMCC-019, WMCC-035 | \`_op_apply_top_down_bias\`, \`_op_manage_task_interference\` | P |
| WM-S3-06 | Coordinate different processing systems | WMCC-021 | \`_op_coordinate_subsystems\` | I |
| WM-S3-07 | Evaluate the consequences of the current interpretation | WMCC-020, WMCC-030 | \`_op_determine_interpretation_acceptance\`, \`_op_revise_interpretations\` | M |
| WM-S4-01 | Generate semantic interpretation candidates | WMCC-023 | \`_op_prepare_interpretations\` | I |
| WM-S4-02 | Collect evidence for each candidate | WMCC-024, WMCC-025, WMCC-026 | \`_op_retrieve_supporting_semantic_knowledge\`, \`_op_evaluate_grammatical_compatibility\`, \`_op_evaluate_contextual_compatibility\` | M |
| WM-S4-03 | Calculate candidate scores | WMCC-027 | \`_op_integrate_semantic_evidence\` | I |
| WM-S4-04 | Calculate probability distributions over interpretations | WMCC-028 | \`_op_normalize_posteriors\` | I |
| WM-S4-05 | Perform competitive activation | WMCC-028 | \`_op_run_recurrent_competition\` | I |
| WM-S4-06 | Integrate contextual evidence | WMCC-026, WMCC-027 | \`_op_evaluate_contextual_compatibility\`, \`_op_integrate_semantic_evidence\` | M |
| WM-S4-07 | Evaluate semantic prediction errors | WMCC-029, WMCC-030 | \`_op_measure_uncertainty\`, \`_op_revise_interpretations\` | P |
| WM-S4-08 | Select, preserve or revise the interpretation | WMCC-029, WMCC-030 | \`_op_apply_abstention\`, \`_op_revise_interpretations\` | M |
| WM-S4-09 | Consolidate the selected interpretation | WMCC-039, WMCC-040 | \`_op_integrate_cognitive_state\`, \`_op_evaluate_processing_completion\` | P |
| WM-S5-01 | Maintain the current task representation | WMCC-031 | \`_op_encode_task_sets\` | I |
| WM-S5-02 | Detect task-changing signals | WMCC-032 | \`_op_detect_task_switches\` | I |
| WM-S5-03 | Evaluate whether switching is necessary | WMCC-032 | \`_op_calculate_switch_costs\` | M |
| WM-S5-04 | Retrieve the new task set | WMCC-034 | \`_op_reconfigure_task_control\` | I |
| WM-S5-05 | Open the working-memory updating gate | WMCC-034 | \`_op_reconfigure_task_control\` | P |
| WM-S5-06 | Preserve unfinished cognitive work | WMCC-033 | \`_op_preserve_suspended_task_states\` | I |
| WM-S5-07 | Reconfigure processing priorities | WMCC-034, WMCC-035 | \`_op_reconfigure_task_control\`, \`_op_manage_task_interference\` | M |
| WM-S5-08 | Re-establish the previous task when appropriate | WMCC-037 | \`_op_resume_suspended_tasks\` | I |
| WM-S5-09 | Learn from previous switching demands | WMCC-038 | \`_op_learn_task_switching_history\` | P |

\*The WM-Sx-Oy identifiers are crosswalk-local labels. The source document numbers these operations within its five stage sections; it does not assign WMCC identifiers.

**Additional repository operations:** WMCC-001 initializes the processing environment; WMCC-022 identifies ambiguity; WMCC-036 monitors the new task and decides when to return; WMCC-039 integrates the full cognitive state; WMCC-040 evaluates completion. These are useful repository extensions or decompositions, not additional source operations.

**Important limitations:** the source's explicit selected-interpretation consolidation is not shown as a distinct durable long-term consolidation mechanism. The source's semantic prediction-error operation is only partially represented by uncertainty measurement and revision. WMCC trace integrity is tested, but that does not prove exact preservation of the original 39-operation sequence.

---

## 4. Source 2 — Brain-Inspired Semantic Composition

**Source:** [Google Drive document](https://drive.google.com/file/d/1EkbfTURhR2K0n7syj_FLb8hz6vzwzA4L/view)  
**Package:** [\`ore_semantic_composition/\`](https://github.com/wizekin/ORE-Neurocomputational/tree/main/ore_semantic_composition)  
**Registry:** [\`ore_semantic_composition/registry.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/ore_semantic_composition/registry.py)  
**Functional entry point:** \`SemanticCompositionProcessor.process()\`  
**Main implementation:** [\`ore_semantic_composition/pipeline.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/ore_semantic_composition/pipeline.py)  
**Biological reference:** \`SemanticCompositionLIFSimulator\` in \`ore_semantic_composition/biology.py\`  
**Tests:** [\`tests/test_semantic_composition.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/tests/test_semantic_composition.py)

The source defines nine named processing stages. The repository implements nine correspondingly named stages and decomposes them into 54 executable-registry entries, six per stage. The source document does not supply a separate 54-item stable ID list, so the mapping below is explicitly **stage-level**, not a claim that source operation IDs exist.

| Source stage | Repository operations | Functional stage handler | Main output | Status |
|---|---|---|---|---|
| Stage 1 — Retrieve individual word concepts | SMC-001..006 | \`_stage_one_retrieve_concepts()\` | Token-linked concept candidates, feature maps, context fit, concept posteriors | I |
| Stage 2 — Combine word meanings | SMC-007..012 | \`_stage_two_combine_meanings()\` | Token semantic nodes, pooled features, sentence composition workspace | I |
| Stage 3 — Predicate-argument composition | SMC-013..018 | \`_stage_three_predicate_arguments()\` | Predicate candidates, frames, argument slots, argument compatibility | I |
| Stage 4 — Assign thematic roles | SMC-019..024 | \`_stage_four_thematic_roles()\` | Agent, patient, theme and other role candidates with evidence | I |
| Stage 5 — Construct propositions | SMC-025..030 | \`_stage_five_propositions()\` | Proposition shells, entity references, event attributes and completeness | I |
| Stage 6 — Represent negation and modality | SMC-031..036 | \`_stage_six_operators()\` | Negation and modal operators, scope and proposition polarity | I |
| Stage 7 — Represent quantifiers and their scope | SMC-037..042 | \`_stage_seven_quantifiers()\` | Quantifier domains, variable bindings, scope alternatives and ambiguity | I |
| Stage 8 — Represent cause and effect | SMC-043..048 | \`_stage_eight_causality()\` | Cause/effect candidates and directed causal graph | I |
| Stage 9 — Generate sentence-level semantic representation | SMC-049..054 | \`_stage_nine_sentence_representation()\` | Typed sentence-semantic graph, global uncertainty and validated serialized result | I |

**Operation-level registry contents:** SMC-001..054 are defined in the repository registry. They cover token normalization and concept retrieval; semantic workspace construction and feature pooling; predicate frames and ordered argument slots; role evidence and role selection; proposition construction and completeness; negation/modality detection and scope; quantifier domains and scope alternatives; causal marker detection, clause alignment and relation construction; and final graph assembly, uncertainty and validation.

**Test evidence:** the test file includes explicit registry/trace count checks, predicate-argument and role tests, negation/modality/quantifier tests, causal-marker tests, and biological-reference unit/operation-state checks.

**Boundary:** the package depends on caller-supplied concept inventories, optional predicate frames, and optional upstream syntax results. It is a constrained semantic-composition processor, not an unrestricted world-knowledge engine. Its LIF reference is not an empirically validated neural simulation.

**Determination:** strong functional correspondence at the nine-stage level. A stricter source-operation crosswalk would require the source document to be normalized into stable individually numbered operations first.

---

## 5. Source 3 — Syntactic Parsing and Structure-Building

**Source:** [Google Drive document](https://drive.google.com/file/d/18ms6EbK8nPlZNKnAVzfVl1I3E1y7zQNB/view)  
**Package:** [\`ore_syntax/\`](https://github.com/wizekin/ORE-Neurocomputational/tree/main/ore_syntax)  
**Registry:** [\`ore_syntax/registry.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/ore_syntax/registry.py)  
**Functional entry point:** \`SyntaxProcessor.process()\`  
**Main implementation:** [\`ore_syntax/pipeline.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/ore_syntax/pipeline.py)  
**Biological reference:** \`SyntaxLIFSimulator\` in \`ore_syntax/biology.py\`  
**Tests:** [\`tests/test_syntax.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/tests/test_syntax.py)

The source contains 55 operations across nine stages. The repository contains 56 SYN operations, distributed 7 / 6 / 7 / 6 / 5 / 6 / 6 / 7 / 6. The following maps each source operation to the corresponding repository stage range. Multiple source responsibilities can share one implementation; the stage helper, not an independently named function for every source operation, performs the stage work.

| Source ID | Source operation | Repository mapping | Stage handler | Status |
|---|---|---|---|---|
| SYN-S1-1 | Activate possible grammatical categories | SYN-002..003 | \`_pos_identification()\` | I |
| SYN-S1-2 | Retrieve grammatical information | SYN-002 | \`_pos_identification()\` | I |
| SYN-S1-3 | Integrate surrounding grammatical evidence | SYN-004..005 | \`_pos_identification()\` | I |
| SYN-S1-4 | Resolve category competition | SYN-006..007 | \`_pos_identification()\` | I |
| SYN-S1-5 | Update grammatical expectations | SYN-004..007 | \`_pos_identification()\` | M |
| SYN-S2-1 | Detect possible phrase beginnings | SYN-008 | \`_phrase_detection()\` | I |
| SYN-S2-2 | Detect relationships between neighboring words | SYN-009..010 | \`_phrase_detection()\` | I |
| SYN-S2-3 | Detect phrase boundaries | SYN-008, SYN-013 | \`_phrase_detection()\` | I |
| SYN-S2-4 | Combine words into preliminary phrases | SYN-010..012 | \`_phrase_detection()\` | I |
| SYN-S2-5 | Update phrase membership | SYN-013 | \`_phrase_detection()\` | I |
| SYN-S3-1 | Identify possible grammatical heads | SYN-014 | \`_dependency_parse()\` | I |
| SYN-S3-2 | Detect potential head-dependent relationships | SYN-015 | \`_dependency_parse()\` | I |
| SYN-S3-3 | Assign dependency labels | SYN-016 | \`_dependency_parse()\` | I |
| SYN-S3-4 | Evaluate competing dependency links | SYN-017 | \`_dependency_parse()\` | I |
| SYN-S3-5 | Enforce structural consistency | SYN-018..020 | \`_dependency_parse()\`, cycle repair and validation | I |
| SYN-S4-1 | Detect possible constituent structures | SYN-021 | \`_constituency_parse()\` | I |
| SYN-S4-2 | Combine smaller constituents into larger constituents | SYN-022..023 | \`_constituency_parse()\` | I |
| SYN-S4-3 | Build nested structures | SYN-024 | \`_constituency_parse()\` | I |
| SYN-S4-4 | Combine grammatical rules with word evidence | SYN-025..026 | \`_constituency_parse()\` | I |
| SYN-S5-1 | Create structural levels | SYN-027 | \`_hierarchy_stage()\` | I |
| SYN-S5-2 | Establish parent-child relationships | SYN-028 | \`_build_hierarchy()\` | I |
| SYN-S5-3 | Bind grammatical relationships to structural positions | SYN-029 | \`_build_hierarchy()\` | I |
| SYN-S5-4 | Perform structural unification | SYN-029..030 | \`_hierarchy_stage()\` | M |
| SYN-S5-5 | Preserve structural uncertainty | SYN-031 | \`_hierarchy_stage()\` | I |
| SYN-S6-1 | Identify the main predicate | SYN-032 | \`_subject_object_assignment()\` | I |
| SYN-S6-2 | Identify possible subjects | SYN-033..034 | \`_subject_object_assignment()\` | I |
| SYN-S6-3 | Identify possible direct objects | SYN-035 | \`_subject_object_assignment()\` | I |
| SYN-S6-4 | Identify indirect objects and additional arguments | SYN-035 | \`_subject_object_assignment()\` | M |
| SYN-S6-5 | Verify grammatical assignments | SYN-036..037 | \`_subject_object_assignment()\` | I |
| SYN-S6-6 | Distinguish grammatical subject from semantic actor | SYN-036..037; SYN-038..043 | \`_subject_object_assignment()\`, \`_thematic_roles()\` | M |
| SYN-S7-1 | Retrieve the verb's semantic argument structure | SYN-038 | \`_thematic_roles()\` | I |
| SYN-S7-2 | Identify the possible agent | SYN-039..041 | \`_thematic_roles()\` | I |
| SYN-S7-3 | Identify the patient or theme | SYN-040..041 | \`_thematic_roles()\` | I |
| SYN-S7-4 | Identify additional thematic roles | SYN-041 | \`_thematic_roles()\` | I |
| SYN-S7-5 | Map grammatical functions to thematic roles | SYN-041 | \`_thematic_roles()\` | I |
| SYN-S7-6 | Calculate thematic compatibility | SYN-040..042 | \`_thematic_roles()\` | I |
| SYN-S8-1 | Detect structural uncertainty | SYN-043 | \`_resolve_ambiguity()\` | I |
| SYN-S8-2 | Generate alternative parses | SYN-043 | \`_resolve_ambiguity()\` | I |
| SYN-S8-3 | Evaluate grammatical compatibility | SYN-044 | \`_resolve_ambiguity()\` | I |
| SYN-S8-4 | Evaluate semantic plausibility | SYN-045 | \`_resolve_ambiguity()\` | I |
| SYN-S8-5 | Evaluate contextual and probabilistic expectations | SYN-044..045 | \`_resolve_ambiguity()\` | M |
| SYN-S8-6 | Integrate all evidence | SYN-046 | \`_resolve_ambiguity()\` | I |
| SYN-S8-7 | Detect interpretation errors | SYN-047..048 | \`_resolve_ambiguity()\` | P |
| SYN-S8-8 | Reconstruct the affected structure | SYN-047..050 plus reconciliation helpers | \`process()\`, \`_phrase_detection_reconcile()\`, \`_dependency_parse_reconcile()\`, \`_constituency_parse_reconcile()\` | I |
| SYN-S8-9 | Update predictions | SYN-047..050 | \`_resolve_ambiguity()\` and reconciliation helpers | M |
| SYN-S9-1 | Detect a dependency-triggering construction | SYN-051 | \`_long_distance_dependencies()\` | I |
| SYN-S9-2 | Register the unresolved dependency | SYN-052 | \`_long_distance_dependencies()\` | I |
| SYN-S9-3 | Maintain the dependency during intervening processing | SYN-053 | \`_long_distance_dependencies()\` | I |
| SYN-S9-4 | Predict the expected grammatical position | SYN-053..054 | \`_long_distance_dependencies()\` | I |
| SYN-S9-5 | Detect a possible dependency-completion point | SYN-054 | \`_long_distance_dependencies()\` | I |
| SYN-S9-6 | Retrieve the appropriate earlier element | SYN-054 | \`_long_distance_dependencies()\` | I |
| SYN-S9-7 | Reconnect the distant grammatical elements | SYN-055 | \`_long_distance_dependencies()\` | I |
| SYN-S9-8 | Check whether the dependency is correctly resolved | SYN-056 | \`_long_distance_dependencies()\` | I |
| SYN-S9-9 | Release unnecessary working-memory information | No distinct release operation identified | \`_long_distance_dependencies()\` returns dependency state | P |
| SYN-S9-10 | Manage nested and competing dependencies | SYN-053..056 | \`_long_distance_dependencies()\` | P |

**Test evidence:** registry order/count; active-sentence POS, phrases, dependency and constituency structure; subject/object and thematic roles; passive voice; wh-filler remote dependency; lexical evidence input; and optional LIF trace.

**Limitations:** the parser is described as a compact English-oriented rule/heuristic prototype, not a treebank-trained general parser. The final source operations concerning active release of obsolete working-memory items and systematic handling of nested/competing dependencies are not established as complete by the inspected code/tests.

---

## 6. Source 4 — Lexical Access and Morphological Processing

**Source:** [Google Drive document](https://drive.google.com/file/d/10kUKpRhnxR8VIzBILbmZsBTsoPofW9ck/view)  
**Package:** [\`ore_lexical_access/\`](https://github.com/wizekin/ORE-Neurocomputational/tree/main/ore_lexical_access)  
**Registry:** [\`ore_lexical_access/registry.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/ore_lexical_access/registry.py)  
**Functional entry point:** \`LexicalAccessProcessor.process()\`  
**Main implementation:** [\`ore_lexical_access/pipeline.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/ore_lexical_access/pipeline.py)  
**Biological reference:** \`LexicalAccessLIFSimulator\` in \`ore_lexical_access/biology.py\`  
**Tests:** [\`tests/test_lexical_access.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/tests/test_lexical_access.py)

The source has 29 operations. The repository expands the same seven broad stages into 36 BLA operations. The mapping is therefore many-to-one or one-to-many rather than identifier-equivalent.

| Source ID | Source operation | Repository mapping | Code path | Status |
|---|---|---|---|---|
| LEX-S1-1 | Encode the incoming word pattern | BLA-001 | Input normalization in \`process()\` | I |
| LEX-S1-2 | Search the stored lexical representations | BLA-002 | \`_identity_candidates()\` | I |
| LEX-S1-3 | Identify the best-fitting lexical identity | BLA-003..005 | \`_identity_candidates()\`, \`_confident_choice()\` | I |
| LEX-S2-1 | Activate the semantic representation | BLA-006..007 | \`_meaning_candidates()\` | I |
| LEX-S2-2 | Retrieve possible word senses | BLA-006, BLA-010 | \`_meaning_candidates()\` | I |
| LEX-S2-3 | Construct a contextual semantic representation | BLA-008..010 | \`_meaning_candidates()\` and context-feature scoring | I |
| LEX-S3-1 | Activate the grammatical representation | BLA-011 | \`_category_candidates()\` | I |
| LEX-S3-2 | Evaluate grammatical compatibility | BLA-012..014 | \`_category_candidates()\` | I |
| LEX-S3-3 | Retrieve additional grammatical properties | BLA-011..014 | \`_category_candidates()\` | P |
| LEX-S4-1 | Detect possible morphological boundaries | BLA-015..016 | \`_decompositions()\` | I |
| LEX-S4-2 | Detect possible roots | BLA-017..018 | \`_decompositions()\` | I |
| LEX-S4-3 | Detect possible derivational structures | BLA-015..019 | \`_decompositions()\` | P |
| LEX-S4-4 | Detect possible inflectional structures | BLA-015..019 | \`_decompositions()\` | P |
| LEX-S4-5 | Construct possible morphological parses | BLA-016..019 | \`_decompositions()\` | I |
| LEX-S5-1 | Retrieve the root or stem representation | BLA-020 | \`_retrieve_components()\` | I |
| LEX-S5-2 | Retrieve derivational affix representations | BLA-021 | \`_retrieve_components()\` | P |
| LEX-S5-3 | Retrieve inflectional affix representations | BLA-021 | \`_retrieve_components()\` | P |
| LEX-S6-1 | Compare possible morphological structures | BLA-024, BLA-027 | \`_score_morphology()\` | I |
| LEX-S6-2 | Evaluate morphological validity | BLA-025, BLA-027 | \`_score_morphology()\` | I |
| LEX-S6-3 | Evaluate semantic compatibility | BLA-026 | \`_score_morphology()\` | I |
| LEX-S6-4 | Resolve competition between alternatives | BLA-027..028 | \`_score_morphology()\`, \`_confident_choice()\` | I |
| LEX-S6-5 | Determine whether the whole word or its components dominate | BLA-019, BLA-027..028 | \`_decompositions()\`, \`_score_morphology()\` | I |
| LEX-S7-1 | Generate initial lexical candidates | BLA-029 | \`_incremental_activation()\` | I |
| LEX-S7-2 | Update candidates as more information arrives | BLA-030, BLA-035..036 | \`_incremental_activation()\` | I |
| LEX-S7-3 | Activate morphological candidates | BLA-034..035 | \`_incremental_activation()\` | I |
| LEX-S7-4 | Allow semantic and grammatical feedback | BLA-034 | \`_incremental_activation()\` | I |
| LEX-S7-5 | Apply activation decay and competition | BLA-033..035 | \`_incremental_activation()\` | I |
| LEX-S7-6 | Select the leading lexical interpretation | BLA-036 | \`_confident_choice()\`, final candidate readout | I |
| LEX-S7-7 | Continue updating after recognition | BLA-035..036 | Final incremental candidate state | P |

**Material differences:** the repository uses generic supplied prefix/root/suffix inventories and does not preserve a full separate operation sequence for derivational versus inflectional affixes. The source's request to continue updating after recognition is not clearly established as a distinct post-recognition recurrent process; the implemented incremental trajectory updates as the input characters arrive.

**Test evidence:** exact seven-stage/36-operation registry, homographic meaning alternatives, category use, stem/affix retrieval, morphological composition, abstention, per-character activation, unknown words, and optional seven-population biological-reference dynamics.

---

## 7. Source 5 — Contextual Prediction in Human Orthographic Processing

**Source:** [Google Drive document](https://drive.google.com/file/d/1EmSbo2z5OtF_e35EtZe6gbJ_Q3qiIkVn/view)  
**Package:** [\`ore_predictive_orthography/\`](https://github.com/wizekin/ORE-Neurocomputational/tree/main/ore_predictive_orthography)  
**Registry:** [\`ore_predictive_orthography/registry.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/ore_predictive_orthography/registry.py)  
**Functional entry point:** \`PredictiveOrthographyProcessor.process()\`  
**Main implementation:** [\`ore_predictive_orthography/pipeline.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/ore_predictive_orthography/pipeline.py)  
**Biological reference:** \`simulate_lif_reference()\` in \`ore_predictive_orthography/biology.py\`  
**Tests:** [\`tests/test_predictive_orthography.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/tests/test_predictive_orthography.py)

The source has 38 numbered operations after including the final “Stage 7.7 — Return diagnostics to the prediction system.” The repository has 40 MPO operations. Importantly, the repository's operation trace is assembled after stage-level calculations: each MPO record is populated from its stage's summary and output-key list. The inspected code does not show each of the 40 MPO entries being invoked as its own independent operation with a unique intermediate result.

| Source ID | Source operation | Repository mapping | Functional implementation path | Status |
|---|---|---|---|---|
| ORTH-S1-1 | Identify character positions | MPO-001..002 | Input sequence is already segmented and ordered before N3 recognition | P |
| ORTH-S1-2 | Extract visual features | MPO-002 | N2 feature descriptors through \`ore_n3.visual.extract_patch_features\` | P |
| ORTH-S1-3 | Generate candidate grapheme identities | MPO-005..007 | \`VisualPopulation.fit()\`, \`VisualPopulation.probabilities()\` | I |
| ORTH-S1-4 | Accumulate sensory evidence over time | MPO-006..008 | N3 population evidence and unknown-class probability | P |
| ORTH-S2-1 | Retrieve learned orthographic statistics | MPO-009, MPO-015 | \`OrthographicTransitionModel.fit()\` | I |
| ORTH-S2-2 | Calculate transition strength | MPO-010..013 | Count/smooth/normalize adjacent grapheme transitions | I |
| ORTH-S2-3 | Construct sequence-level predictions | MPO-012..014 | Unigram and forward/reverse transition priors | I |
| ORTH-S2-4 | Convert transition probabilities into neural support | MPO-014..015 | Contextual-prior readout and transition-memory fingerprint | M |
| ORTH-S3-1 | Retrieve neighboring grapheme beliefs | MPO-016 | N3 posterior beliefs | I |
| ORTH-S3-2 | Generate predictions from the left neighbor | MPO-017 | \`predict_from_left()\` | I |
| ORTH-S3-3 | Generate predictions from the right neighbor | MPO-018 | \`predict_from_right()\` | I |
| ORTH-S3-4 | Combine neighboring evidence | MPO-019 | \`_neighbor_messages()\` | I |
| ORTH-S3-5 | Transmit contextual information to grapheme populations | MPO-019..020 | Contextual prior and N4 input | M |
| ORTH-S3-6 | Preserve uncertainty in neighboring positions | MPO-020, MPO-036..040 | Boundary availability, posterior alternatives and output diagnostics | P |
| ORTH-S4-1 | Estimate attentional distribution | MPO-021..027 | \`N4Processor.process()\` | I |
| ORTH-S4-2 | Estimate sensory uncertainty | MPO-021..022 | \`normalized_entropy()\` | I |
| ORTH-S4-3 | Calculate sensory precision | MPO-028 | N4 sensory-precision calculation | I |
| ORTH-S4-4 | Modulate precision through attention | MPO-026..028 | \`N4Processor.process()\` | I |
| ORTH-S4-5 | Modulate the influence of visual prediction errors | MPO-023..025, MPO-028 | Error/salience and attention-gain calculations | M |
| ORTH-S5-1 | Retrieve the prior distribution | MPO-012..015, MPO-029 | N3 transition prior and N4 prior precision | M |
| ORTH-S5-2 | Estimate contextual reliability | MPO-029 | N4 prior-precision calculation | P |
| ORTH-S5-3 | Estimate prior precision | MPO-029 | N4 prior-precision calculation | I |
| ORTH-S5-4 | Adjust prior influence | MPO-030 | Top-down relevance modulation | I |
| ORTH-S5-5 | Balance sensory and prior reliability | MPO-031..032 | Separate precision channels and validation | I |
| ORTH-S6-1 | Construct the initial posterior distribution | MPO-033..035 | N4 attended belief initialization and posterior proposal | I |
| ORTH-S6-2 | Compare the new posterior with the previous beliefs | MPO-036 | Damped belief-update iteration trace | I |
| ORTH-S6-3 | Apply recurrent feedback | MPO-036..037 | Damped posterior update | I |
| ORTH-S6-4 | Control the magnitude of posterior updates | MPO-036 | Configured damping coefficient | I |
| ORTH-S6-5 | Repeat the inference process | MPO-037 | N4 convergence/iteration stopping logic | I |
| ORTH-S6-6 | Retain alternative interpretations when necessary | MPO-035..037 | Posterior alternatives and abstention diagnostics | I |
| ORTH-S6-7 | Generate an updated orthographic representation | MPO-037 | Posterior beliefs in result object | P |
| ORTH-S7-1 | Reconstruct the expected visual feature representation | MPO-039 | Feature-prototype reconstruction | I |
| ORTH-S7-2 | Calculate feature-space prediction residuals | MPO-039 | Actual-minus-predicted feature residuals | I |
| ORTH-S7-3 | Calculate precision-weighted residuals | MPO-039 | Attended/contextual residual diagnostics | P |
| ORTH-S7-4 | Calculate divergence between prior predictions and posterior beliefs | MPO-038 | Symmetric KL / posterior-prediction divergence | I |
| ORTH-S7-5 | Calculate uncertainty and confidence | MPO-038..040 | Entropy, posterior margin and diagnostics | I |
| ORTH-S7-6 | Diagnose the source of prediction failure | MPO-040 | Combined output diagnostics | P |
| ORTH-S7-7 | Return diagnostics to the prediction system | MPO-040 | Result diagnostics; no separate feedback-cycle operation identified | P |

**Critical input boundary:** the processor requires an already segmented and ordered N3 sequence. It uses N2-derived visual descriptors, trained grapheme prototypes, transition statistics, and N4 attention/precision processing. It is not by itself a raw-image character-segmentation or complete OCR pipeline.

**Critical trace boundary:** MPO-001..040 are emitted from registry metadata and stage summaries after the computations. Treat this as a registry-completeness trace, not proof that every MPO row corresponds to an individually executed intermediate operation.

**Test evidence:** registry count/order, seven stage summaries, 40 trace entries, N3/N4 validity, posterior shapes, left/right-neighbor boundary behavior, residual arrays, optional LIF population simulation, and rejection of an unfitted upstream N3 processor.

---

## 8. Source 6 — Human Semantic and Lexical Memory

**Source:** [Google Drive document](https://drive.google.com/file/d/1yGsoTezrHe2b_oFIKBTIJE97FN0UE26N/view)  
**Package:** [\`ore_semantic/\`](https://github.com/wizekin/ORE-Neurocomputational/tree/main/ore_semantic)  
**Registry:** [\`ore_semantic/registry.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/ore_semantic/registry.py)  
**Functional entry point:** \`SemanticCognitionProcessor.process()\`  
**Main implementation:** [\`ore_semantic/pipeline.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/ore_semantic/pipeline.py)  
**Biological reference:** \`SemanticLIFSimulator\` in \`ore_semantic/biology.py\`  
**Tests:** [\`tests/test_semantic_cognition.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/tests/test_semantic_cognition.py)

Both source and repository contain 44 steps/operations, but the order and internal grouping differ. A matching count does not indicate one-to-one identity. The key limitation is that the repository retrieves caller-supplied episodic traces and derives read-only prototypes; it does not demonstrate a complete episodic acquisition, durable storage, consolidation, or experience-driven prototype-learning lifecycle.

| Source step | Source responsibility | Repository mapping | Implementation path / evidence | Status |
|---:|---|---|---|---|
| 1 | Receive incoming lexical or conceptual activation | SC-007..008 | \`_retrieve_candidates()\` | I |
| 2 | Activate the appropriate ATL neuronal populations | SC-027 | \`_hub_features()\`, ATL drive in LIF simulator | I |
| 3 | Integrate information arriving from multiple modalities | SC-013..020, SC-028 | \`_spoke_activations()\`, hub integration | I |
| 4 | Construct a coherent conceptual representation | SC-029..031 | \`_hub_features()\`, hub readout | I |
| 5 | Complete an incomplete conceptual representation | SC-030 | Concept-feature completion | I |
| 6 | Identify the modalities associated with the concept | SC-013..020 | Modality evidence channels supplied through the request | P |
| 7 | Activate the modality-specific feature populations | SC-013..019 | \`_spoke_activations()\` | I |
| 8 | Retrieve modality-specific features | SC-013..019, SC-021 | Caller-supplied feature maps and spoke activation | I |
| 9 | Exchange information between spokes and the ATL | SC-028, SC-032 | Hub convergence and feedback | I |
| 10 | Weight the reliability of each modality | SC-020 | Spoke reliability calculation | I |
| 11 | Activate the individual semantic features | SC-021 | Distributed feature initialization | I |
| 12 | Activate associated semantic features | SC-022 | Feature propagation | I |
| 13 | Combine information from multiple semantic pathways | SC-023 | Feature evidence integration | I |
| 14 | Calculate semantic relationships and similarities | SC-023..024 | Feature similarity/competition | I |
| 15 | Maintain distributed activation and competition | SC-024..026 | Feature competition and prediction-error state | I |
| 16 | Receive the current task or contextual requirement | SC-005 | Request task and context fields | I |
| 17 | Estimate semantic control demand | SC-033 | Control-demand calculation | I |
| 18 | Identify the task-relevant features | SC-034 | Goal/task relevance | I |
| 19 | Amplify the relevant features | SC-037..038 | Context reweighting and controlled retrieval cycles | M |
| 20 | Suppress irrelevant or competing features | SC-036 | Lateral inhibitory control | I |
| 21 | Send top-down control signals to semantic representations | SC-037..038 | Controlled contextual reweighting/retrieval | M |
| 22 | Repeat semantic activation under control | SC-038 | Controlled retrieval cycles | I |
| 23 | Generate candidate concepts | SC-007..008, SC-035 | Lexical candidate and competing-interpretation generation | M |
| 24 | Calculate the evidence supporting each candidate | SC-013..039 | Spoke, feature, hub, control and candidate evidence | M |
| 25 | Calculate the probability of each candidate | SC-035, SC-039 | Candidate competition and ambiguity/selection-margin computation | P |
| 26 | Resolve ambiguous meanings | SC-039, SC-043 | Ambiguity measurement and selection/abstention | I |
| 27 | Select the most appropriate concept | SC-043 | \`_candidate_evidence()\`, selection gate | I |
| 28 | Construct the final semantic representation | SC-031, SC-044 | Hub readout and result assembly | I |
| 29 | Capture the current word-form episode | No episodic write operation identified | Request accepts caller-provided episodes; no new episode capture found | N |
| 30 | Encode the episode into a memory representation | No episodic encoder/write path identified | Episodic traces are consumed as input | N |
| 31 | Separate similar episodes | SC-010..012 provide matching/weighting, not explicit durable pattern-separation storage | Episodic matching path | P |
| 32 | Bind the episode's different components | SC-010..012 retrieve supplied bound records; no write-time binding path identified | Episodic retrieval | P |
| 33 | Store an episodic memory trace | No durable episode-store operation identified | No repository episode persistence path found in this package | N |
| 34 | Retrieve matching episodic traces | SC-010..012 | Episode retrieval, surface matching, context/recency weighting | I |
| 35 | Perform pattern completion | SC-011, SC-030 | Similarity-based episode matching plus concept completion; not a full episodic pattern-completion system | P |
| 36 | Send retrieved episodic information back into semantic processing | SC-009..012, SC-032 | Phonological/episode cues feed lexical candidates and hub feedback | M |
| 37 | Collect multiple word-form experiences | SC-040..042 operate on available concept/prototype inventory, not a growing episodic experience stream | \`_build_prototypes()\` | P |
| 38 | Identify common features across episodes | SC-041 | Frequency-weighted prototype feature structure from supplied exemplars | P |
| 39 | Build an abstract orthographic representation | SC-040..042 | Abstract prototype retrieval/feature structure; no separately verified orthographic episode-learning model | P |
| 40 | Build an abstract phonological representation | SC-009, SC-040..042 | Phonological episode cues and abstract prototypes; distinct learned phonological prototype not established | P |
| 41 | Construct the abstract lexical identity | SC-040..042 | Prototype records and concept-to-prototype fit | P |
| 42 | Strengthen the prototype through experience | No online prototype-strengthening path identified | Prototype path is read-only during inference | N |
| 43 | Consolidate the abstract representation | No durable consolidation/update operation identified | Prototype records are derived, not consolidated by a memory-writing process | N |
| 44 | Generalize across new experiences | SC-042 offers bounded exemplar-to-prototype comparison; it does not update the prototype from new experience | \`_prototype_fit()\` | P |

**Biological-reference coverage:** the repository defines populations for lexical access, episodic word form, visual/auditory/action/somatosensory/chemosensory/affective/spatiotemporal spokes, distributed features, ATL, IFG control and abstract prototypes. It records 44 biological snapshots when enabled. Those populations model the architecture; they do not compensate for missing functional episode-writing or consolidation behavior.

**Determination:** the seven mechanisms appear in the architecture and the functional retrieval/control path, but the source's episodic-memory acquisition and experience-driven abstract-prototype learning lifecycle is incomplete. This source must remain **PARTIAL**, not fully implemented.

---

## 9. Source 7 — Integrated Word Recognition and Semantic Prediction

**Source:** [Google Drive document](https://drive.google.com/file/d/1IU4BQwpqSnKBAVEill5FESUDY2Uxu6Bo/view)  
**Package:** [\`ore_lexical_context/\`](https://github.com/wizekin/ORE-Neurocomputational/tree/main/ore_lexical_context)  
**Registry:** [\`ore_lexical_context/registry.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/ore_lexical_context/registry.py)  
**Functional entry point:** \`LexicalContextProcessor.process()\`  
**Main implementation:** [\`ore_lexical_context/pipeline.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/ore_lexical_context/pipeline.py)  
**Biological reference:** \`LexicalContextLIFSimulator\` in \`ore_lexical_context/biology.py\`  
**Tests:** [\`tests/test_lexical_context.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/tests/test_lexical_context.py)

This is the closest operation-number match: source IDs 001..040 map by suffix to repository IDs LC-001..040. The source subsystem headings and repository subsystem labels differ, but the 40-operation sequence is represented in the registry and the processing code emits an operation mark for each step for each token.

| Source ID | Source operation | Repository ID | Code path | Status |
|---|---|---|---|---|
| WF-001 | Retrieve accumulated word exposure | LC-001 | \`process()\`, exposure/session state | I |
| WF-002 | Calculate logarithmic frequency strength | LC-002 | Lexical frequency calculation in \`process()\` | I |
| WF-003 | Construct the lexical resting activation | LC-003 | Lexical activation calculation | I |
| WF-004 | Adjust lexical activation thresholds | LC-004 | Activation/adaptation calculation | I |
| WF-005 | Generate frequency-conditioned activation | LC-005 | Lexical activation calculation | I |
| RP-006 | Retrieve recent lexical activation history | LC-006 | Session activation history | I |
| RP-007 | Calculate stimulus similarity | LC-007 | \`_levenshtein()\` / similarity calculation | I |
| RP-008 | Calculate temporal decay | LC-008 | Time-decay calculation | I |
| RP-009 | Generate repetition-dependent facilitation | LC-009 | Repetition priming calculation | I |
| RP-010 | Apply repetition-related neural adaptation | LC-010 | Adaptation state calculation | I |
| MD-011 | Detect possible morphological boundaries | LC-011 | \`_morph_splits()\` | I |
| MD-012 | Evaluate stem and affix candidates | LC-012 | Morphological candidate scoring | I |
| MD-013 | Retrieve morpheme-associated meanings | LC-013 | Morpheme feature lookup | I |
| MD-014 | Evaluate morphological composition | LC-014 | Feature composition | I |
| MD-015 | Integrate whole-word and decomposed representations | LC-015 | Whole-form/morpheme integration | I |
| SSC-016 | Generate candidate grapheme-to-phoneme correspondences | LC-016 | \`_pronunciations()\` | I |
| SSC-017 | Calculate mapping consistency | LC-017 | G2P coverage/reliability | I |
| SSC-018 | Activate competing pronunciation representations | LC-018 | Pronunciation candidate generation | I |
| SSC-019 | Weight pronunciation alternatives | LC-019 | Pronunciation alternative weighting | I |
| SSC-020 | Send phonological evidence into lexical recognition | LC-020 | Phonological feedback into lexical candidates | I |
| ONC-021 | Generate orthographic neighbors | LC-021 | Neighbor search | I |
| ONC-022 | Calculate neighborhood strength | LC-022 | Frequency-weighted neighborhood strength | I |
| ONC-023 | Calculate shared orthographic activation | LC-023 | N-gram/Dice overlap | I |
| ONC-024 | Apply competitive inhibition | LC-024 | Similarity-weighted neighbor suppression | I |
| ONC-025 | Settle the lexical competition | LC-025 | Damped recurrent competition | I |
| PHD-026 | Detect semantic ambiguity | LC-026 | Meaning-candidate enumeration | I |
| PHD-027 | Generate prior probabilities for meanings | LC-027 | Sense-prior calculation | I |
| PHD-028 | Retrieve meaning-specific semantic features | LC-028 | Meaning feature retrieval | I |
| PHD-029 | Calculate contextual compatibility | LC-029 | Context-feature compatibility | I |
| PHD-030 | Select or retain competing meanings | LC-030 | Posterior and abstention gate | I |
| CWP-031 | Construct the current sentence context | LC-031 | Context accumulation across tokens | I |
| CWP-032 | Generate upcoming word predictions | LC-032 | Transition/feature-based next-word prediction | I |
| CWP-033 | Calculate contextual predictability | LC-033 | Predictability/surprisal calculation | I |
| CWP-034 | Pre-activate expected lexical and semantic features | LC-034 | Predicted-feature pooling | I |
| CWP-035 | Modulate predictions according to context reliability | LC-035 | Predictive reliability calculation | I |
| SPE-036 | Construct the incoming semantic representation | LC-036 | Incoming semantic feature construction | I |
| SPE-037 | Calculate the semantic prediction residual | LC-037 | Feature-wise semantic residual | I |
| SPE-038 | Calculate precision-weighted semantic error | LC-038 | Precision-weighted residual | I |
| SPE-039 | Update the semantic interpretation | LC-039 | Posterior semantic update | I |
| SPE-040 | Update the prediction system for subsequent words | LC-040 | Sentence-state/next-token prediction update | I |

**Trace behavior:** the code's local \`mark(step, ...)\` function selects the registry entry by step number, appends an operation trace record, and advances the optional biological simulator. The focused tests assert a 40-row trace and biological operation trace.

**Boundary:** lexical entries, morphemes, pronunciations and semantic features are supplied as inventories. The implementation supports controlled inference and exposure history; it should not be interpreted as automatic acquisition of a complete language lexicon from unrestricted experience.

**Determination:** strong functional and operation-trace correspondence to the source's 40 operations. This is not proof of calibrated human psycholinguistic behavior.

---

## 10. Source 8 — Integrated Reading Algorithm: Formal Operation Registry

**Source:** [Google Drive document](https://drive.google.com/file/d/1cfHkGjRavIzPrXpANM-2Gtq73ZvqAlb5/view)  
**Package:** [\`ore_reading/\`](https://github.com/wizekin/ORE-Neurocomputational/tree/main/ore_reading)  
**Registry:** [\`ore_reading/registry.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/ore_reading/registry.py)  
**Functional entry point:** \`IntegratedReadingProcessor.process()\`  
**Main implementation:** [\`ore_reading/pipeline.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/ore_reading/pipeline.py)  
**Biological reference:** \`BiologicalReadingSimulator\` in \`ore_reading/biology.py\`  
**Tests:** [\`tests/test_integrated_reading.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/tests/test_integrated_reading.py)  
**Repository architecture document:** [\`docs/INTEGRATED_READING_OPERATION_REGISTRY.md\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/docs/INTEGRATED_READING_OPERATION_REGISTRY.md)

The Google Drive document defines IRA-OP-001..062, in eight operation groups. The repository defines IR-001..072, in nine groups. The following table maps the source operation to the closest repository operation(s). Where the source expects raw visual or auditory signals but the repository accepts text/tokens or generated phonological sequences, the mapping is marked partial or missing rather than equivalent.

| Source ID | Source operation | Repository mapping | Code path / limitation | Status |
|---|---|---|---|---|
| IRA-OP-001 | Initialize Reading Transaction | IR-001 | \`process()\` request initialization | I |
| IRA-OP-002 | Determine Input Modality and Reading Task | IR-001, IR-007 | Request contract, script/direction metadata; modality is limited by supported input contract | P |
| IRA-OP-003 | Acquire Input Signal | IR-001 | Receives text or caller-tokenized words; no direct raw image/audio acquisition | P |
| IRA-OP-004 | Normalize and Organize Input | IR-002..008 | Unicode, token, punctuation, grapheme and position encoding | I |
| IRA-OP-005 | Extract Visual Features | IR-006 | Symbolic orthographic evidence, not raw visual feature extraction | P |
| IRA-OP-006 | Detect Graphemic Features | IR-005..006 | Unicode grapheme extraction and symbolic feature evidence | P |
| IRA-OP-007 | Activate Letter Identity Units | IR-013 | Candidate bottom-up orthographic drive; not an independently measured letter-population layer | P |
| IRA-OP-008 | Encode Letter Position | IR-008 | Position-coded orthographic form | I |
| IRA-OP-009 | Segment and Buffer Grapheme Sequences | IR-003..005, IR-008 | Token/grapheme preparation and position coding | I |
| IRA-OP-010 | Initialize Interactive Activation Network | IR-015 | Interactive activation initialization | I |
| IRA-OP-011 | Propagate Bottom-Up Feature Activation | IR-013 | Letter-level bottom-up drive | I |
| IRA-OP-012 | Perform Letter-Level Competition | IR-016 | Competition is represented mainly at candidate/word level; a distinct letter-unit competition layer is not established | P |
| IRA-OP-013 | Propagate Letter Activation to Word Units | IR-013..016 | Candidate activation and recurrence | M |
| IRA-OP-014 | Perform Top-Down Word-to-Letter Feedback | IR-016 | Candidate feedback in interactive settling | I |
| IRA-OP-015 | Perform Word-Level Lateral Inhibition | IR-016 | Recurrent excitation and competition | I |
| IRA-OP-016 | Integrate Contextual Bias Into Lexical Activation | IR-014, IR-044 | Frequency/context priors in lexical and Bayesian stages | M |
| IRA-OP-017 | Settle Interactive Activation | IR-016 | Recurrent settling | I |
| IRA-OP-018 | Search the Orthographic Lexicon | IR-010..012 | Indexing, exact and approximate retrieval | I |
| IRA-OP-019 | Identify Lexical Word | IR-011..012, IR-063 | Candidate identification and final selection | I |
| IRA-OP-020 | Dispatch Information to Reading Routes | IR-017..018 | Lexical lookup and sublexical rule preparation | I |
| IRA-OP-021 | Match the Complete Word in the Lexical Route | IR-017 | Lexical phonology lookup and exact word candidates | I |
| IRA-OP-022 | Retrieve Stored Phonological Representation | IR-017 | Stored lexical pronunciation lookup | I |
| IRA-OP-023 | Retrieve Semantic Information From Lexical Representation | IR-056 | Meaning readout from lexical/connectionist representation | I |
| IRA-OP-024 | Segment Orthographic Input for Nonlexical Conversion | IR-019 | Grapheme chunking | I |
| IRA-OP-025 | Retrieve Grapheme-to-Phoneme Correspondence Rules | IR-018 | G2P rule preparation | I |
| IRA-OP-026 | Align Graphemes With Phonemes | IR-019..020 | Chunking and conversion | I |
| IRA-OP-027 | Generate Phonological Candidates | IR-020 | G2P candidate output | I |
| IRA-OP-028 | Apply Phonological and Orthographic Constraints | IR-021 | Route compatibility scoring | I |
| IRA-OP-029 | Integrate Lexical and Nonlexical Route Evidence | IR-022..023 | Route gain and evidence integration | I |
| IRA-OP-030 | Select Task-Appropriate Reading Representation | IR-023..024 | Route selection and pronunciation uncertainty | I |
| IRA-OP-031 | Encode Distributed Orthographic Representation | IR-049 | Orthographic vector | I |
| IRA-OP-032 | Compute Hidden-Layer Activation | IR-055 | Triangular recurrent settling; not a separately trained deep hidden-layer model | P |
| IRA-OP-033 | Compute Orthography-to-Phonology Mapping | IR-052 | O-P association weights | I |
| IRA-OP-034 | Compute Orthography-to-Semantics Mapping | IR-053 | O-S association weights | I |
| IRA-OP-035 | Compute Semantics-to-Phonology Mapping | IR-054 | S-P association weights | I |
| IRA-OP-036 | Perform Distributed Activation Settling | IR-055 | Triangular recurrent settling | I |
| IRA-OP-037 | Read Out Connectionist Predictions | IR-056 | Meaning activation readout | I |
| IRA-OP-038 | Update Connectionist Learning Weights | IR-052..055 | Association matrices are induced from supplied lexicon; online supervised weight learning is not established by this path | P |
| IRA-OP-039 | Extract Temporal Acoustic Information | No direct IR equivalent | \`ReadingRequest\` is text/token based; no raw continuous audio waveform enters \`ore_reading\` | N |
| IRA-OP-040 | Activate TRACE Feature Units | IR-033 | TRACE-style feature evidence is built from generated phonological input, not extracted acoustic features | P |
| IRA-OP-041 | Activate TRACE Phoneme Units | IR-034 | Phoneme candidate activation | P |
| IRA-OP-042 | Activate TRACE Word Units and Feedback | IR-035..037 | Word candidates, top-down feedback and competition | P |
| IRA-OP-043 | Initialize Cohort Candidate Set | IR-038 | Cohort initialization | P |
| IRA-OP-044 | Update and Restrict the Cohort | IR-039 | Incremental cohort narrowing on generated phoneme sequence | P |
| IRA-OP-045 | Select the Spoken Lexical Candidate | IR-040, IR-063 | Recognition-point readout plus final lexical decision; this is not full acoustic spoken-word recognition | P |
| IRA-OP-046 | Generate Continuous-Speech Segmentation Hypotheses | No direct IR equivalent | No raw continuous speech input/segmentation in \`ore_reading\` | N |
| IRA-OP-047 | Calculate Word-Input Likelihood | IR-041..042 | Candidate generation and phonological noisy-channel likelihood | P |
| IRA-OP-048 | Calculate Lexical Prior Probability | IR-043 | Frequency-based lexical prior | I |
| IRA-OP-049 | Calculate Bayesian Posterior Probability | IR-044..045 | Context prior and Bayesian posterior update | I |
| IRA-OP-050 | Perform Bayesian Word and Segmentation Competition | IR-046 | Shortlist-B-style noise competition | I |
| IRA-OP-051 | Commit Spoken-Word Recognition Result | IR-047..048 | Posterior calibration and abstention; commits a generated phonological candidate rather than an audio-decoded spoken word | P |
| IRA-OP-052 | Consolidate Lexical Identity | IR-057, IR-063 | Sequence-level lexical integration and final decision; durable consolidation is not established | P |
| IRA-OP-053 | Activate Semantic Representations | IR-056 | Connectionist meaning readout | I |
| IRA-OP-054 | Integrate Sentence Context | IR-058, IR-061 | Neighbor context and situation/context state update | I |
| IRA-OP-055 | Resolve Semantic Ambiguity | IR-059, IR-063 | Sense resolution and final decision/abstention | I |
| IRA-OP-056 | Integrate Words Into Syntactic Structure | IR-060 | Typed relation proposals from explicit lexical categories; not a full syntactic parse | P |
| IRA-OP-057 | Update Working Memory and Sentence Meaning | IR-057..061 | Sequence/context state; this is not a replacement for the separate \`ore_working_memory\` processor | P |
| IRA-OP-058 | Construct Task-Specific Reading Result | IR-063..064, IR-072 | Final decision, explanation and structured result | I |
| IRA-OP-059 | Prepare Phonological Output | IR-025..032 | Phonological feature/sequence assembly | I |
| IRA-OP-060 | Produce Overt Speech or Silent Reading Output | IR-032, IR-072 | Returns phonological/decoded result; no text-to-speech or overt speech-production engine identified | P |
| IRA-OP-061 | Direct Attention to the Next Reading Unit | No direct IR equivalent | No eye-movement or next-fixation controller identified in this package | N |
| IRA-OP-062 | Monitor Processing and Initiate Reanalysis | IR-071, IR-063 | Integrity/trace validation and final abstention; full cognitive error-triggered reanalysis is not established | P |

### 10.1 Repository-only extensions beyond the 62-operation source registry

The repository adds IR-065..070 for functional-to-neural drive mapping, LIF population dynamics, neural-state measurement and prediction-error comparison. It also uses IR-071..072 for registry integrity and structured result/scope reporting. These are useful extensions but must not be confused with source IRA-OP-001..062.

### 10.2 Integrated reading limitations

The repository's own architecture documentation states that the package begins with text or caller-tokenized orthographic words and does not replace upstream image segmentation. It can receive upstream N6 output through an adapter. The auditory TRACE/Cohort/Shortlist-inspired operations work on phonological material generated in the reading path; they do not establish a full raw-audio continuous-speech recognition path in \`ore_reading\`. A separate auditory-capable sensory-perception package exists in ORE, but this audit did not find an integrated raw-audio adapter from that package into \`ore_reading\`.

---

## 11. Cross-package evidence and test index

These focused tests exist in the audited repository tree. Their presence was inspected; they were **not executed** for this crosswalk.

| Package | Test file | Test coverage identified |
|---|---|---|
| Working memory | [\`tests/test_working_memory.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/tests/test_working_memory.py) | 40-operation trace, bounded buffers, incomplete syntax, unresolved dependencies, semantic ambiguity/abstention, control, task switching, serialization and SQLite annotation catalog |
| Semantic composition | [\`tests/test_semantic_composition.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/tests/test_semantic_composition.py) | 9 stages/54 operations, predicate arguments, roles, propositions, negation, modality, quantifiers, causality, biological units |
| Syntax | [\`tests/test_syntax.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/tests/test_syntax.py) | 9 stages/56 operations, parse structure, argument roles, passive voice, long-distance dependencies, biological reference |
| Lexical access | [\`tests/test_lexical_access.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/tests/test_lexical_access.py) | 7 stages/36 operations, homographs, categories, morphology, incremental activation, uncertainty, LIF reference |
| Predictive orthography | [\`tests/test_predictive_orthography.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/tests/test_predictive_orthography.py) | 7 stages/40 registry entries, N3/N4 integration, neighbor prediction, posterior, residual and LIF diagnostics |
| Semantic/lexical memory | [\`tests/test_semantic_cognition.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/tests/test_semantic_cognition.py) | 44 operations, episodic retrieval from supplied episodes, modality evidence, prototype/selection behavior, LIF snapshots and adapters |
| Integrated word recognition | [\`tests/test_lexical_context.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/tests/test_lexical_context.py) | 40 operations, biological trace, ambiguity, morphology, G2P competition, repetition/exposure, adapter |
| Integrated reading | [\`tests/test_integrated_reading.py\`](https://github.com/wizekin/ORE-Neurocomputational/blob/main/tests/test_integrated_reading.py) | 72 operations, functional/biological pipeline, lexical/sublexical routes, posterior, LIF units, validation and N6 adapter |

---

## 12. Implementation gaps requiring follow-up

The following work is recommended before describing all eight source algorithms as fully applied.

| Priority | Package | Required follow-up |
|---|---|---|
| P0 | \`ore_predictive_orthography\` | Replace stage-summary replication of 40 registry rows with genuine operation-level trace records, each attached to the computation that produced its output. Preserve the 38 source steps in an explicit source-to-MPO mapping. |
| P0 | \`ore_semantic\` | Implement episodic word-form capture, encoding, pattern separation/binding, durable storage, retrieval, consolidation and experience-driven prototype strengthening as separate, testable operations. Keep read-only inference distinct from memory writes. |
| P0 | \`ore_reading\` | Decide whether raw auditory input and continuous-speech segmentation belong inside \`ore_reading\` or are supplied by an upstream auditory adapter. Implement and test an explicit interface if full 62-operation source coverage is required. |
| P1 | \`ore_working_memory\` | Add an explicit source-39-operation registry or a versioned crosswalk from the 39 source steps to WMCC-001..040. Separate semantic prediction-error evaluation and interpretation consolidation where those are source requirements. |
| P1 | \`ore_syntax\` | Add explicit unresolved-memory release and nested/competing long-distance dependency handling, or mark those source responsibilities as intentionally unsupported. |
| P1 | \`ore_lexical_access\` | Represent derivational and inflectional morphology separately where required by the source, and distinguish word-class retrieval from other grammatical features. |
| P1 | \`ore_predictive_orthography\` | Document the N2/N3 segmentation boundary in the cross-package contract and connect output diagnostics to a tested next-cycle feedback consumer if recurrent system-level prediction is intended. |
| P1 | \`ore_reading\` | Decide whether overt speech generation and next-reading-unit attention control are in scope. If not, mark these two source operations as excluded at the package boundary. |
| P2 | All packages | Add source-operation IDs to each runtime trace row so that source-to-code coverage can be verified automatically. |
| P2 | All packages | Add tests that fail when a source operation has no mapping, when a mapped code path is removed, or when an operation is only represented by a placeholder/summary trace. |
| P2 | All packages | Run the complete test suite and preserve CI run/commit evidence alongside this crosswalk. |

---

## 13. Recommended traceability contract for future ORE modules

Each source operation should have a stable crosswalk record with, at minimum:

| Field | Requirement |
|---|---|
| \`source_document_id\` | Stable identifier for the exact source document/version |
| \`source_operation_id\` | Original source stage/operation identifier; do not silently renumber it |
| \`source_operation_name\` | Original operation name |
| \`repository_operation_id\` | Stable repository registry ID |
| \`implementation_symbol\` | Fully qualified module/class/function responsible for the computation |
| \`functional_status\` | Implemented, merged, partial, trace-only, or missing |
| \`inputs_observed\` | Actual inputs used by the operation |
| \`outputs_produced\` | Actual outputs produced by the operation |
| \`mathematical_operation\` | Formula or numerical procedure actually evaluated |
| \`biological_reference_status\` | Simulated, disabled, conceptual-only, or not applicable |
| \`test_ids\` | Tests that exercise the behavior |
| \`evidence_reference\` | Source path/line range, trace entry or test assertion supporting the mapping |

**Trace invariant:** a registry entry is not sufficient evidence of execution. A functional operation trace should be emitted at the operation's actual execution point and should contain its inputs (or safe input summary), output, status, provenance and any abstention/error state. A stage-level summary must not be copied into every operation row and labeled as if each operation had independently executed.

**Biological invariant:** simulated membrane voltage, current, firing rate, spikes and time must be reported separately from functional decision scores. A biological-reference step marked disabled must remain visibly disabled; it must not be represented as executed.

**Source-version invariant:** when a Drive source file changes, the crosswalk should record the new source version and identify affected source operation IDs. A changed source should not silently inherit an old implementation-complete status.

---

## 14. Final audit conclusion

At this audit baseline, ORE has implementation packages corresponding to all eight specified source documents. The strongest operation-level correspondence is \`ore_lexical_context\` to the 40-operation integrated word-recognition source. The semantic-composition package has strong alignment at the nine-stage level. The other packages have meaningful functional coverage but also important registry differences, scope boundaries, or missing operations.

The most consequential gaps are:

1. Predictive orthography's 40 registry rows are currently generated from stage summaries rather than separately evidenced operation execution.
2. Semantic/lexical memory does not yet implement the full episodic memory write-and-consolidation lifecycle or experience-driven prototype strengthening.
3. Integrated reading does not implement raw continuous-audio acquisition/segmentation, eye/fixation-style next-unit attention, or overt speech generation inside \`ore_reading\`.
4. Several repository packages use revised operation counts and ordering; source-level traceability is not yet automatically enforced.

**Certification statement:** this document is an initial static source-to-code crosswalk. It is not a certification that all source operations are fully implemented, that all tests pass, or that the models reproduce human neurobiology. The statuses should be upgraded only after the missing behavior is implemented and operation-level tests and execution traces demonstrate it.

