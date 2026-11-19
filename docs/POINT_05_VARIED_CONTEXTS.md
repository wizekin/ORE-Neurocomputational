# Point 05 — Varied Contexts and Interleaved Learning

## Scope and scientific boundary

The ore_varied_contexts package implements the 23 stages in Point_05_Varied_Contexts_Brain_Algorithm.md.

It keeps three learning dimensions separate:

1. **Contextual variation** changes the setting, wording, visual form, modality, or other situational cues while retaining the concept's task-relevant structure.
2. **Interleaving** alternates related concepts or procedures when their comparison can teach a useful distinction.
3. **Spacing** separates encounters over time. Point 05 records elapsed-time metadata but does not replace Point 03's calendar scheduler or infer spacing simply from an interleaved order.

The implementation does not equate useful variation with randomness. It also preserves individual episode records while estimating shared concept-level features, so abstraction does not delete information about particular examples.

All equations are candidate computational models. The biological classes are functional surrogates, not neuron-level simulations or measurements of human neural activity.

## Package layout

- contracts.py: validated concept definitions, examples, context cues, observations, learner state, configuration, requests, and results.
- functional.py: feature distances and similarities, prototypes, within-category variability, context inference, boundary-condition checks, interleaving eligibility and scheduling, concept selection, contrastive analysis, cross-modality comparisons, transfer, diagnosis, adaptive difficulty, and flexibility metrics.
- biology.py: recurrent population activity, hippocampal-style episode/context binding and partial-cue completion, cortical concept prototypes, modality-specific association pathways, prefrontal task control, medial-frontal-inspired error monitoring, and a neuromodulatory plasticity gate.
- registry.py: the 23-stage P05-01 through P05-23 operation registry.
- pipeline.py: the 23-stage coordinator, shared SQLite annotation registration, and ordered operation-event recorder.
- annotations.py: Point 05 entry point for the repository AST annotation catalogue.
- examples/point05_varied_contexts.py: a runnable mathematical problem-type example.
- tests/test_varied_contexts_point05.py: functional, biological, pipeline, and SQLite tests.

## The 23 implemented stages

| Stage | Operation | Main implementation |
|---|---|---|
| 1 | Identify target concepts | Typed target definitions, domains, features, relations, and conditions |
| 2 | Retrieve prerequisites | Readiness and prerequisite-gap assessment |
| 3 | Build informative examples | Concept prototypes, feature means, within-category variance, and episode counts |
| 4 | Separate context from concept features | Context-dimension coverage and condition metadata |
| 5 | Establish anchor context | Baseline selection against a clear example |
| 6 | Vary context while preserving the relation | Multiple context-linked examples with a stable target identity |
| 7 | Infer current context | Smoothed Bayesian-style context inference and uncertainty |
| 8 | Encode episode and concept | Individual hippocampal-style episode binding plus a separate cortical prototype |
| 9 | Detect invariants and discriminative features | Within-category variance, between-category distances, and diagnostic feature ranks |
| 10 | Change modality or representation | Cross-modality feature and relation consistency reports |
| 11 | Decide whether to interleave | Relatedness, prerequisite structure, observed confusion, and learnable contrast |
| 12 | Build a schedule | Balanced deterministic ordering with switches only among eligible pairs |
| 13 | Execute trials | Candidate concept scoring, selection, confidence, context match, and effort |
| 14 | Compare alternatives | Contrastive margins and pairwise confusion counts |
| 15 | Adapt variation and complexity | New-context, task-switch, and unknown-feature cost estimates |
| 16 | Retrieve through different cues | Modality, representation, and context cue routes |
| 17 | Test near transfer | Unseen examples in a familiar domain |
| 18 | Test far transfer | Relation preservation and incompatible-analogy penalties |
| 19 | Restore meaningful context | Explicit boundary-condition and exception checks |
| 20 | Consolidate concept structure | Bounded prototype updates while retaining episode records |
| 21 | Diagnose errors | Prerequisite, context, feature, cue/modality, and overload categories |
| 22 | Adapt the next practice set | Bounded difficulty, variation, and concept-priority updates |
| 23 | Evaluate flexibility | Separate original/changed context, modality, discrimination, near/far transfer, and delayed measures |

## Mathematical operations represented

- **Feature decomposition:** concept features and context cues are kept in separate fields; both can influence a decision when the concept has valid boundary conditions.
- **Similarity and distance:** cosine similarity and normalized Euclidean distance over named features.
- **Prototype formation:** feature-wise means and variances for each concept.
- **Context inference:** smoothed categorical likelihoods and priors with entropy-based uncertainty.
- **Concept selection:** weighted feature compatibility and stable softmax probabilities.
- **Discrimination:** between-prototype distance, diagnostic feature differences, pairwise margins, and confusion matrices.
- **Cross-modality consistency:** similarity of task-relevant feature summaries and preservation of shared relations.
- **Interleaving:** a relatedness/confusion/learnable-contrast eligibility rule plus balanced schedule scoring; maximum switching is not the objective.
- **Task complexity:** weighted count of novel contextual dimensions, concept switches, and unseen feature dimensions.
- **Adaptive variation:** bounded update from recent accuracy, target accuracy, and effort.
- **Transfer:** near-transfer accuracy, far-transfer relation mapping, incompatible relation penalties, and training-to-transfer generalization gaps.
- **Context flexibility:** mean accuracy across contexts and the original-versus-changed context robustness gap.
- **Discrimination evaluation:** per-class precision, recall, F1, and balanced accuracy.
- **Prototype updating:** an exponential update of feature means with a corresponding running variance estimate.
- **Biological plasticity gate:** a bounded learning signal influenced by error discrepancy, relevance, and verified evidence.

The feature values, thresholds, weights, and default learning rates are engineering parameters, not universal psychological constants. They need calibration against an explicit task, learner, and evaluation set.

## Biological computational surrogates

### Hippocampal context memory

Stores an episode vector together with its context cues and relation metadata. A partial context cue can reactivate a stored episode using similarity-based pattern completion. The model preserves separate episode records rather than reducing every example to one prototype.

### Distributed cortical concept representation

Maintains feature prototypes at the concept level. Verified examples update the prototype through a bounded exponential rule. The module does not overwrite or delete episodic details.

### Modality-specific association pathways

Maintains separate input populations for modalities such as text, diagrams, or audio and reports similarity of their modeled task-relevant activity. This is a simplified shared-representation model, not a biological sensory decoder.

### Prefrontal task control

Represents maintenance of the selected concept, candidate selection, task-set change, and an illustrative switching cost. It does not claim that the prefrontal cortex uses this exact score.

### Medial-frontal error monitoring and neuromodulatory gate

Track discrepancy between expected and observed behavior and compute a bounded learning gate. Gate values are not neurotransmitter concentrations. The model labels the evidence source and only treats feedback as verified when the caller explicitly marks it as such.

## Safeguards

- Context variation and interleaving are separate outputs and separately tunable.
- The schedule only uses examples with training status; novel near/far and delayed examples remain evaluation evidence.
- Relatedness alone is insufficient: interleaving also requires shared domain/prerequisite structure and a learnable contrast or observed confusion.
- When prerequisites are weak, the readiness report recommends an anchor or initial blocked practice rather than immediate interleaving.
- A concept can remain context-dependent. Boundary conditions are retained and checked before applying the rule.
- Prototype abstraction does not remove the episode records.
- Far-transfer scoring distinguishes preserved relations from incompatible ones and does not reward superficial word overlap.
- Delayed observations are counted separately from ordinary or immediate trials.
- A model-generated prediction is not confused with an observed learner response. Evaluation metrics use supplied observations.
- The software's neural activity, plasticity, and context-memory variables are explicitly labeled computational surrogates.

## SQLite annotations and operation logging

The register_annotations(database_path) function uses the shared ore_logger.catalog.sync_repository AST indexer with include_paths set to the ore_varied_contexts package. It registers package modules, every class, and every function in the repository-wide annotation table.

The record_result_to_sqlite(result, database_path) function registers those definitions, creates a logger run, and writes exactly 23 ordered operation events. Each event stores the operation identifier, source step, mathematical form, functional output, biological diagnostic output, and evidence-level caution.

## Usage overview

Create ConceptDefinition objects with defining feature weights and any required boundary conditions. Create LearningExample objects with concept features, context features, modality, representation, and relations. Add TrialObservation objects for actual learner choices; a generated classifier prediction is not treated as an observed learner answer.

Pass those objects to VariedContextsProcessor.process using a VariedContextsRequest. The result includes the 23-stage trace, concept profiles, context inference, interleaving decision, schedule, transfer report, flexibility report, biological diagnostics, and updated learner state.

For reproducible execution, run examples/point05_varied_contexts.py. For focused validation, run:

    python -m pytest -q tests/test_varied_contexts_point05.py

The full repository test suite remains the integration check.
