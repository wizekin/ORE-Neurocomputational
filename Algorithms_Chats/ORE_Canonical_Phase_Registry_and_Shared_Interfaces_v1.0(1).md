# ORE Canonical Phase Registry and Shared Interfaces

**Specification ID:** `ORE-ARCH-REGISTRY-001`  
**Version:** 1.0.0  
**Date:** 14 November 2026  
**Status:** Canonical architecture specification; implementation of the contracts is a separate work item  
**Applies to:** `ORE_Master_Algorithm_v1.0` (functional A–J) and the dedicated biological-development track (N1–N9)

---

## 0. Decision summary

This document establishes two coordinated but separately named architecture tracks:

1. **Functional ORE (`F-A` through `F-J`)** — the orthographic learning and recognition algorithms defined by `ORE_Master_Algorithm_v1.0.md`.
2. **Biological ORE (`B-N1` through `B-N9`)** — the dedicated neural-development sequence defined by the current biological-development roadmap.

The tracks share versioned data contracts, evidence provenance, model lineage and evaluation protocols. They do **not** share an assumption that their internal algorithms are identical.

The biological phase register is canonical as follows:

| Canonical ID | Canonical name | Sole responsibility |
|---|---|---|
| `B-N1` | Neural simulation foundations | Define and simulate neuron, synapse, population and time-dependent state dynamics |
| `B-N2` | Biological visual processing | Transform visual input into retinal/LGN-inspired and V1-like responses |
| `B-N3` | Hierarchical visual architecture | Build progressively more abstract visual representations |
| `B-N4` | Neural orthographic representations | Encode grapheme identity, absolute/ordinal position and character sequences in neural populations |
| `B-N5` | Neural connectivity and recurrence | Define local/long-range, excitatory/inhibitory, feed-forward and feedback interactions |
| `B-N6` | Biological learning and memory | Implement and test experience-dependent plasticity, homeostasis and retention |
| `B-N7` | Predictive and contextual processing | Generate predictions, compare them with evidence, represent uncertainty and modulate precision |
| `B-N8` | Brain-data validation | Test behavioral, representational, temporal and connectivity correspondence against independent evidence |
| `B-N9` | Integrated neurocomputational ORE | Coordinate the functional and biological engines, experiments, model versions and release decisions |

**No phase number is reused for a different responsibility.** In particular, N8 is validation, not lexical-semantic lookup; N9 is the system integration/controller, not an additional independent recognition engine.

---

## 1. Authority, precedence and naming policy

### 1.1 Canonical sources

The following documents define the two parent tracks:

- `ORE_Master_Algorithm_v1.0.md` — functional A–J algorithm and contracts.
- The current dedicated biological-development specification — N1–N9 phase definitions and acceptance requirements stated in the current roadmap.

The earlier `ORE_NEUROCOMPUTATIONAL_DEVELOPMENT_ROADMAP_v1.0.md` and the earlier `ORE_N9_Integrated_Neurocomputational_Algorithm_v1.0.md` are retained as historical design records. They are not authoritative for the current biological phase numbering.

### 1.2 Superseded numbering

The earlier N9 proposal used a different N1–N8 allocation: early visual processing, visual organization, recurrent prediction, attention, plasticity, sequence/lexical access, contextual inference and lexical-semantic access. That allocation conflicts with the dedicated track now established here.

The following migration rule applies:

- Preserve useful algorithms and research ideas from the earlier proposal.
- Reassign them by **function**, not by their old phase number.
- Do not rename the current canonical phases to fit the older proposal.
- Keep phonology, syntax, proposition composition, discourse and situation-model processing outside the orthographic core unless separately approved as a downstream extension. They must not be silently inserted into `B-N8` or `B-N9`.

### 1.3 IDs and versioning

Use explicit namespaces in code, files, tests and experiment reports:

- `F-A` … `F-J`: functional stages.
- `B-N1` … `B-N9`: biological stages.
- `IF-*`: shared interface contracts.
- `EXP-*`: experiment identifiers.
- `VAL-*`: acceptance and validation gates.

A phase's operation count is metadata belonging to a particular algorithm revision. It is not part of the phase ID and must not be used as a stable identifier. This avoids renaming a phase whenever its operation list expands.

### 1.4 Evidence/status vocabulary

Every phase and artifact must distinguish these statuses:

- `SPECIFIED` — responsibilities, contracts and acceptance criteria are written.
- `IMPLEMENTED` — executable code exists for the declared scope.
- `SOFTWARE_TESTED` — specified software tests pass.
- `TASK_VALIDATED` — held-out task performance meets preregistered criteria.
- `BIOLOGICALLY_SUPPORTED` — independent biological evidence supports the declared correspondence.
- `PENDING` / `NOT_TESTED` / `FAILED` — use explicitly; never treat absence of evidence as a pass.

These statuses are independent. For example, `SOFTWARE_TESTED` does not imply `TASK_VALIDATED` or `BIOLOGICALLY_SUPPORTED`.

---

## 2. Canonical functional registry: F-A through F-J

The functional engine remains the reference implementation of the orthographic task. The biological engine may provide alternative or additional evidence, but must not silently replace these responsibilities.

| ID | Master stage | Owns | Consumes | Produces | Boundary rule |
|---|---|---|---|---|---|
| `F-A` | Initialize and validate | Configuration, script profile, run state, compatibility checks | Config, `ScriptProfile`, optional checkpoint | `RunContext`, validated profile and model references | No silent profile repair or incompatible checkpoint loading |
| `F-B` | Prepare learning data | Source registration, hashes, labels, split policy, safe transformations and exposure events | Raw samples and source metadata | `DatasetManifest`, patch samples, labels and exposure events | Test data never updates weights, pattern statistics or calibration |
| `F-C` | Learn visual and character representations | Functional visual encoder, identity head, view invariance and `UNK` mass | `VisualObservation` / `VisualPatch` batches | `VisualFeatureBundle`, `GraphemeEvidence` | Preserve alternatives and raw visual evidence |
| `F-D` | Learn position and order representations | Absolute A, relative/open-bigram B and overlap O backends | Grapheme evidence and positions | `PositionCodeBundle` | Each backend remains independently runnable and ablatable |
| `F-E` | Learn orthographic regularities | Soft transitions, candidate chunks, MDL promotion and exposure memory | Verified training exposure events | `OrthographicMemorySnapshot` | No whole-word lookup masquerading as sublexical generalization |
| `F-F` | Train, evaluate and freeze | Functional losses, optimizer, validation, calibration and immutable checkpoints | Training batches, F-C/F-D outputs, F-E memory | Functional `ModelBundle`, training report | No updates from validation/test data |
| `F-G` | Recognize a visual string | Candidate generation, beam search, evidence scoring, ranking and abstention | Frozen model, visual/position/orthographic evidence | `SequenceHypothesisSet` | Keep visual and orthographic scores separate; no semantic correction |
| `F-H` | Return result and trace | Structured output, trace serialization and provenance | Recognition hypotheses and representations | `ORERecognitionResult`, trace and manifest | Do not discard alternative channels or uncertainty |
| `F-I` | Optional incremental learning | Permission, evidence policy, child-model update, regression and rollback | Explicit learning request and verified examples | New model child or rollback report | Recognition is read-only; parent model is immutable |
| `F-J` | Acceptance and release | Software gates, task experiments, reproducibility and release decision | Artifacts, experiments, manifests and reports | `EvaluationReport`, release status | `NOT_TESTED` is never `PASS` |

### Functional engine boundary

The Master Algorithm's primary result is an orthographic interpretation: visual evidence, uncertain grapheme identities, position/order channels, learned sublexical patterns, ranked sequence hypotheses and trace. It does not become a sentence-understanding system merely because a downstream language module is connected.

---

## 3. Canonical biological registry: B-N1 through B-N9

### `B-N1` — Neural simulation foundations

**Owns:** neuron state, membrane dynamics, firing, synaptic transmission, population state and simulation time.  
**Consumes:** model configuration, population definitions, connectivity definitions and stimulus/current inputs.  
**Produces:** time-indexed neural state snapshots, spike/event records, population summaries and simulator diagnostics.  
**Must not own:** visual feature interpretation, grapheme labels, orthographic chunks or final recognition decisions.  
**Acceptance:** independently tested neuron and population dynamics, numerical stability, deterministic/reproducible runs under fixed settings and documented units/time-step behavior.

### `B-N2` — Biological visual processing

**Owns:** retinal/LGN-inspired preprocessing and V1-like response mechanisms, including contrast sensitivity, orientation selectivity, spatial-frequency responses and retinotopic/spatial organization.  
**Consumes:** raw visual observations and the `B-N1` simulation interface.  
**Produces:** multichannel early visual response maps, response reliability and receptive-field/filter metadata.  
**Must not own:** character identity resolution, whole-string recognition or lexical correction.  
**Acceptance:** response properties verified against predeclared visual experiments; report which mechanisms are simplified or absent.

### `B-N3` — Hierarchical visual architecture

**Owns:** intermediate and higher-level visual populations, feature composition, increasing abstraction and tested visual invariance.  
**Consumes:** `B-N2` activity/features and `B-N1` time/state services.  
**Produces:** layer-indexed visual representations, spatial maps, abstraction/invariance diagnostics and uncertainty.  
**Must not own:** final grapheme decoding or orthographic transition memory.  
**Acceptance:** evidence of increasing feature abstraction and measurable invariance without unacceptable loss of identity discrimination.

### `B-N4` — Neural orthographic representations

**Owns:** distributed grapheme identity populations, identity/position conjunctions, absolute and ordinal position coding, and neural sequence representations.  
**Consumes:** `B-N3` visual representations, `ScriptProfile`, segmentation alternatives and `B-N1` state.  
**Produces:** neural grapheme activity, `C/A/B/O`-compatible position codes, order hypotheses and uncertainty.  
**Must not own:** the functional A/B/O benchmark definitions or final candidate ranking policy; those remain in `F-D`/`F-G`.  
**Acceptance:** character and sequence recognition, controlled position/order tests and independent generalization beyond fixed character prototypes.

### `B-N5` — Neural connectivity and recurrence

**Owns:** excitatory/inhibitory interactions, local and long-range connections, recurrent loops, feedback pathways and network stability.  
**Consumes:** neural populations from `B-N1`–`B-N4`, connectivity/configuration and bounded recurrent task inputs.  
**Produces:** recurrent state trajectories, connection-level diagnostics, feedback effects and convergence/stability reports.  
**Must not own:** learned orthographic priors by itself; it transports/interacts with information supplied by memory and predictive modules.  
**Acceptance:** reproducible network dynamics and experimentally testable feedback effects against feed-forward controls.

### `B-N6` — Biological learning and memory

**Owns:** Hebbian mechanisms, STDP, homeostatic regulation, experience-dependent specialization, retention and forgetting/stability measurements.  
**Consumes:** authorized exposure events, neural activity, synaptic state and parent model snapshot.  
**Produces:** detached child neural model/memory state, plasticity traces, exposure lineage and retention reports.  
**Must not own:** unverified labels, automatic learning during inference or in-place mutation of accepted parent models.  
**Acceptance:** learning and retention demonstrated under controlled exposure and independent tests, without relying solely on fixed character prototypes.

### `B-N7` — Predictive and contextual processing

**Owns:** candidate predictive coding mechanisms, top-down predictions, bottom-up residuals, uncertainty, precision modulation and alternative recurrent-model comparisons.  
**Consumes:** visual/orthographic beliefs, learned priors/memory, recurrent state and explicitly supplied context.  
**Produces:** prediction distributions, prediction errors, precision/reliability signals, revised beliefs and conflict diagnostics.  
**Must not own:** source pixels or measured visual evidence; predictions must never overwrite observations.  
**Acceptance:** predictions and errors tested under independently specified conditions; comparison to feed-forward and alternative recurrent models; false-correction risk measured.

### `B-N8` — Brain-data validation

**Owns:** independent comparison of behavioral outcomes, neural representational structure, temporal activity and connectivity against empirical evidence.  
**Consumes:** preregistered experiment specifications, model predictions/traces, rights-cleared datasets and independent human/neural reference measurements.  
**Produces:** validation reports, effect estimates, uncertainty intervals, replication artifacts and explicit limitations.  
**Must not own:** online recognition, training labels or runtime model mutation. It is an evaluation work package, not a mandatory inference step.  
**Acceptance:** reproducible results with held-out evidence and independent audit; report task performance and biological correspondence as separate conclusions.

### `B-N9` — Integrated neurocomputational ORE

**Owns:** orchestration across `F-A…F-J` and `B-N1…B-N8`, dependency scheduling, backend selection, cross-module contracts, evidence ledger, model lineage and system-level acceptance.  
**Consumes:** version-compatible functional and biological components plus a task/run request.  
**Produces:** integrated result, complete trace, selected/alternative backend comparison, model lineage and separate software/task/biological statuses.  
**Must not own:** a hidden replacement recognizer, unbounded recurrence or a claim of brain equivalence.  
**Acceptance:** complete regression and contract suite, reproducible integrated runs, backward compatibility with functional ORE, and independently evaluated biological correspondence where claimed.

---

## 4. Canonical crosswalk: functional stages to biological phases

The crosswalk states where collaboration is allowed. It does not mean one phase replaces another.

| Functional stage | Primary biological counterpart(s) | Interface / exchange | Integration rule |
|---|---|---|---|
| `F-A` Initialize | `B-N1`, all phases | `IF-01 RunContext`, `IF-02 ScriptProfile` | Shared configuration/profile validation; independent model configs remain namespaced |
| `F-B` Data preparation | `B-N2`, `B-N3`, `B-N8` | `IF-03 VisualObservation`, `IF-04 DatasetManifest` | Same stimuli and split hashes can be used; test split stays sealed |
| `F-C` Visual/character learning | `B-N2`, `B-N3` | `IF-05 VisualFeatureBundle`, `IF-06 GraphemeEvidence` | Functional encoder and biological pathway are alternative/parallel backends, not silently stacked duplicates |
| `F-D` Position/order | `B-N4` | `IF-07 PositionCodeBundle` | Compare A/B/O mathematical backends with neural population implementations on identical tasks |
| `F-E` Orthographic memory | `B-N6`, `B-N7` | `IF-08 OrthographicMemorySnapshot`, `IF-10 PredictionState` | Functional transition/chunk memory and biological plasticity remain separately stored and evaluated |
| `F-F` Functional training | `B-N1`, `B-N6` | `IF-09 LearningRequest`, `IF-11 ModelBundleRef` | Conventional optimization and neural plasticity are competing learning backends, not one conflated update rule |
| `F-G` Recognition | `B-N2…B-N7` | `IF-06`–`IF-10`, `IF-12 SequenceHypothesisSet` | Candidate scores retain channel identities; neural output is additional evidence, not an invisible correction |
| `F-H` Output/trace | All producing phases | `IF-13 ORERecognitionResult`, `IF-14 TraceEvent` | One canonical result references source traces from both engines |
| `F-I` Incremental learning | `B-N6` | `IF-09 LearningRequest`, `IF-15 LearningResult` | Explicit authorization, detached child checkpoint, regression and rollback |
| `F-J` Acceptance | `B-N8`, `B-N9` | `IF-16 ExperimentSpec`, `IF-17 EvaluationReport` | Software, OCR/task and biological gates remain separate |

### Non-equivalence rule

A biological population response may provide evidence corresponding to a functional feature, but must not be copied into a functional field unless the adapter records its transformation, units, producer and provenance. For example, neural firing rate, membrane potential, spike count and a softmax character probability are different quantities.

---

## 5. Shared interface contracts

All shared objects are versioned, typed and traceable. The contracts below define the minimum exchange format; implementations may add optional fields without changing existing meanings.

### 5.1 Common envelope — `IF-00 ContractEnvelope`

Every cross-module artifact must carry:

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `contract_id` | string | yes | Stable interface identifier, e.g. `IF-06` |
| `schema_version` | string | yes | Semantic version of the data contract |
| `run_id` | string | yes | One coordinated run identifier |
| `sample_id` | string/null | where applicable | Stable stimulus/sample identity |
| `producer_id` | string | yes | Exact producing module/backend and version |
| `model_version` | string/null | where applicable | Model/checkpoint version |
| `config_hash` | SHA-256 string | yes | Canonical configuration fingerprint |
| `script_profile_id` | string/null | where applicable | Script/profile used for interpretation |
| `data_hash` | SHA-256 string/null | where applicable | Source/sample/dataset fingerprint |
| `created_at` | UTC ISO-8601 string | yes | Artifact creation time |
| `evidence_kind` | enum | yes | `OBSERVED`, `DERIVED`, `PREDICTED`, `LEARNED`, `CALIBRATED`, `EXTERNAL_REFERENCE` |
| `parent_artifact_ids` | list[string] | yes | Provenance lineage; empty only for source inputs |

The envelope describes provenance; it is not itself scientific evidence.

### 5.2 `IF-01 ScriptProfile`

Required fields:

- `profile_id`, `profile_version`, `script_id`;
- `reading_direction`: `LTR`, `RTL`, `TTB`, `BTT`, or a declared mixed-layout policy;
- `grapheme_segmentation_policy` and Unicode normalization policy;
- case, diacritic, joining, ligature and punctuation policies;
- logical reading-axis definition;
- script-specific geometry/augmentation constraints.

Physical image coordinates and logical reading order must remain separate. No component may assume left-to-right or one Unicode code point per grapheme.

### 5.3 `IF-02 RunContext`

Contains run ID, operation mode (`FUNCTIONAL`, `BIOLOGICAL`, `HYBRID`, `TRAIN`, `INFER`, `VALIDATE`), deterministic seed/settings, compute budget, module/backend selection, timeout limits and immutable model references.

### 5.4 `IF-03 VisualObservation`

Minimum fields:

- raw source reference and immutable source hash;
- pixel array or patch set, declared layout, channel order and numeric range;
- physical coordinate frame and image dimensions;
- optional missing/occlusion mask;
- segmentation hypotheses with patch IDs, bounding boxes, confidence and logical order alternatives;
- `ScriptProfile` reference and source/license metadata.

Do not destroy the raw input during normalization. Transformed arrays must identify their parent and transformation parameters.

### 5.4 `IF-04 DatasetManifest`

Contains dataset/version, rights/license/provenance, sample hashes, labels and label provenance, writer/font/source/string-family grouping, split assignments, transformation manifest, seed and leakage checks. The final test split is immutable after preregistration.

### 5.5 `IF-05 VisualFeatureBundle`

Common output for either functional visual features or biological visual processing:

- feature tensor(s) and explicit shape/layout/dtype;
- named feature levels and their spatial resolutions;
- source patch/region mapping;
- feature producer (`F-C`, `B-N2`, `B-N3`, or named backend);
- receptive-field/filter/normalization metadata where available;
- uncertainty/reliability and transformation lineage.

Functional encoder embeddings and biological activity maps may both implement this interface, but must retain `representation_kind` and `units` so they cannot be mistaken for one another.

### 5.6 `IF-06 GraphemeEvidence`

For each patch/region:

- stable patch ID and segmentation hypothesis ID;
- grapheme inventory/profile version;
- probability distribution over graphemes plus explicit `UNK`/unknown mass, when a probability is available;
- raw logits or activations separately from normalized probabilities;
- supporting visual feature IDs;
- uncertainty/entropy and reliability;
- producer/backend and calibration status.

A neural population activation is not automatically a calibrated posterior. An adapter must provide an explicit calibrated mapping before labeling it as probability.

### 5.7 `IF-07 PositionCodeBundle`

Required distinct channels:

- `C`: identity evidence;
- `A`: absolute/slot code;
- `B`: directed relative/open-bigram code, including gap descriptors;
- `O`: distributed-overlap/ordinal code;
- uncertainty and positional reliability;
- physical coordinates and logical/ordinal positions;
- backend, parameters and provenance for each code.

Every channel can be disabled, evaluated and ablated independently. Neural population codes may be attached as additional implementations of these channel semantics, but must not erase the functional reference activations.

### 5.8 `IF-08 OrthographicMemorySnapshot`

Contains functional transition counts/probabilities, chunk candidates and promoted chunks, lifetime/recent exposure, promotion evidence and memory version. If biological memory is also provided, it is stored in a different namespace and referenced through `biological_memory_ref`; do not serialize synaptic weights as if they were chunk counts.

### 5.9 `IF-09 LearningRequest`

Required fields:

- parent model fingerprint;
- training sample IDs and split authorization;
- independently verified targets or explicitly named unsupervised regime;
- reviewer/source provenance;
- update channels allowed (`functional_parameters`, `orthographic_memory`, `synapses`, `homeostasis`, etc.);
- explicit authorization flag and policy reference;
- regression suite and rollback target.

A failed gate returns a rejected request and leaves the parent unchanged.

### 5.10 `IF-10 PredictionState`

Contains separately:

- prediction target and level;
- predicted distribution/value and its generating model;
- observed evidence reference;
- residual/prediction-error definition and units;
- sensory and prior precision/reliability values with calibration status;
- pre-update and post-update beliefs;
- iteration number, damping/step parameters and stop reason;
- conflict diagnostics.

Predictions and errors are derived/predicted quantities, not observations. They must never overwrite the raw visual or label evidence.

### 5.11 `IF-11 ModelBundleRef`

Contains model family, version, immutable fingerprint, schema version, script-profile compatibility, parent fingerprint, training/data manifest references, code version and checkpoint location/reference. A functional model and a neural model are separate model families even when trained from the same samples.

### 5.12 `IF-12 SequenceHypothesisSet`

Each candidate contains:

- grapheme sequence and segmentation path;
- `S_vis`, `S_geometry`, `S_position`, `S_ortho`, `S_recurrent`, `S_predictive` where available;
- score definitions, weights and calibration status;
- neural activity/trace references, not opaque copied values;
- unknown symbols and alternatives;
- confidence, ambiguity/abstention status and out-of-distribution flags.

Raw scores from different modules must not be added blindly. Fusion weights or a learned combiner must be fitted on validation data and reported. Preserve all component scores even when a fused ranking is produced.

### 5.13 `IF-13 ORERecognitionResult`

Canonical result fields:

- status: `RESOLVED`, `AMBIGUOUS`, `PARTIAL`, `ABSTAINED`, `FAILED`;
- ordered candidate hypotheses;
- canonical `OrthographicRepresentation` with `C/A/B/O/N/U/M` preserved;
- optional neural-state and prediction references;
- confidence/calibration status;
- provenance and run manifest references;
- limitations and unsupported-channel declarations.

### 5.14 `IF-14 TraceEvent`

Append-only structured event with monotonic event index, simulation time (if applicable), wall-clock time, module ID, input artifact IDs, output artifact IDs, operation name, parameters, numerical diagnostics and status. An ordered trace is required for replay/debugging.

### 5.15 `IF-15 LearningResult`

Contains decision (`ACCEPTED`, `REJECTED`, `ROLLED_BACK`), parent and child fingerprints, changed parameter/memory references, tests executed, metrics before/after, regression failures and provenance.

### 5.16 `IF-16 ExperimentSpec`

Contains hypothesis, task, stimulus design, controlled conditions, baseline/comparison models, preregistration timestamp, train/validation/test grouping, seeds, dependent variables, exclusions, thresholds, analysis code version and intended scientific claim level.

### 5.17 `IF-17 EvaluationReport`

Contains separate sections for:

1. software/contract correctness;
2. orthographic task performance;
3. robustness/generalization/calibration;
4. computational cost and latency;
5. behavioral correspondence;
6. neural representational/temporal/connectivity correspondence;
7. failures, uncertainty and limitations.

Every metric records dataset/split, sample size, seeds, confidence interval, baseline, code/model fingerprints and analysis version. Missing evidence is `NOT_TESTED` or `PENDING`, not zero and not pass.

---

## 6. Neural simulation interface: `IF-18 NeuralStateSnapshot`

`B-N1` exposes a simulator-neutral snapshot so downstream modules are not tied to a specific integrator or simulator library.

Required fields:

- `simulation_time_ms`, `step_index`, `dt_ms`;
- population IDs and stable unit indices;
- optional membrane-potential arrays with declared units;
- spike/event records with unit IDs and timestamps;
- firing-rate summaries with declared bin/window;
- optional synaptic-current/conductance summaries with units;
- connectivity/model fingerprint and simulator version;
- numerical status and trace reference.

Rules:

1. The simulator declares whether it is rate-based, spiking, conductance-based or another model class.
2. Membrane potential, firing rate, spike count and activation probability are distinct quantities.
3. Missing state is represented as unavailable, not fabricated as zero.
4. Each numerical integration method and time step is configurable and recorded.
5. Stability, conservation or boundedness requirements are specific to the chosen neuron/synapse model and must be stated in its model card.

### Python protocol sketch

```python
from typing import Protocol, Sequence

class NeuralSimulator(Protocol):
    def initialize(self, config: "NeuralSimulationConfig") -> "NeuralStateSnapshot": ...

    def step(
        self,
        state: "NeuralStateSnapshot",
        inputs: Sequence["NeuralInputEvent"],
        dt_ms: float,
    ) -> "NeuralStateSnapshot": ...

    def reset(self, state: "NeuralStateSnapshot") -> "NeuralStateSnapshot": ...
```

The protocol is an interface contract; it does not prescribe a biological model or numerical solver.

---

## 7. Backend protocols and ownership

The implementation should use dependency injection and narrow protocols. One object should not own every algorithm.

```python
from typing import Protocol, Sequence

class FunctionalVisualBackend(Protocol):
    def encode(self, observation: "VisualObservation") -> "VisualFeatureBundle": ...

class BiologicalVisualBackend(Protocol):
    def process(
        self,
        observation: "VisualObservation",
        simulator: NeuralSimulator,
    ) -> "VisualFeatureBundle": ...

class OrthographicBackend(Protocol):
    def represent(
        self,
        features: "VisualFeatureBundle",
        evidence: Sequence["GraphemeEvidence"],
        profile: "ScriptProfile",
    ) -> "PositionCodeBundle": ...

class RecurrentBackend(Protocol):
    def refine(
        self,
        state: "RecurrentInferenceState",
        max_iterations: int,
    ) -> "RecurrentInferenceResult": ...

class PlasticityBackend(Protocol):
    def create_child(
        self,
        parent: "ModelBundleRef",
        request: "LearningRequest",
    ) -> "LearningResult": ...

class PredictiveBackend(Protocol):
    def infer(
        self,
        evidence: "EvidenceLedger",
        context: "ContextSnapshot | None",
    ) -> "PredictionState": ...

class SequenceDecoder(Protocol):
    def decode(
        self,
        evidence: "EvidenceLedger",
        representations: "PositionCodeBundle",
    ) -> "SequenceHypothesisSet": ...

class BiologicalValidator(Protocol):
    def evaluate(self, spec: "ExperimentSpec") -> "EvaluationReport": ...
```

The concrete dataclasses may live in a shared `ore_contracts` package. Do not make the core contracts depend on a particular neural library, OCR framework or experiment UI.

---

## 8. Runtime modes and execution policy

N9 supports three declared modes. They are experiment conditions, not three different definitions of ORE.

### 8.1 `FUNCTIONAL`

Run the Master A–J path using the functional visual/orthographic backends. This is the reference path for task correctness, speed and existing behavior.

### 8.2 `BIOLOGICAL`

Run B-N1 through B-N7 for the declared neural experiment. B-N8 evaluates saved model predictions/traces offline. It is not called as a recognition layer.

### 8.3 `HYBRID`

Run compatible functional and biological branches and exchange outputs only through the shared contracts. The controller may use either:

- parallel comparison (both branches see identical inputs; outputs are compared); or
- a declared cascade (one backend's output becomes a documented input to another).

Parallel comparison is the default for initial research because it reveals each branch's contribution. A cascade must state which fields are passed, how units are mapped and which evidence remains independent.

### 8.4 Dependency graph

Recommended high-level dependencies:

```text
F-A/F-B ──> VisualObservation + DatasetManifest
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
      F-C backend          B-N1 substrate
          │                   │
          │                 B-N2
          │                   │
          │                 B-N3
          │                   │
          └─────────┬─────────┘
                    ▼
             F-D / B-N4 position
                    │
       F-E memory ──┼── B-N6 memory
                    │
                 B-N5 recurrence
                    │
                 B-N7 prediction
                    │
                    ▼
                 F-G decode
                    │
                 F-H trace/result
                    │
            F-J + B-N8 evaluation
                    │
                 B-N9 governance
```

The graph is a dependency sketch, not a biological timing claim. N2/N3 and independent data/profile tasks may execute in parallel; N5/N7 loops must be bounded and traceable.

### 8.5 Resource policy

- Use a maximum recurrent iteration count and a wall-clock/compute budget.
- Record every pruning, early-exit and backend-selection decision.
- A missing module produces `CHANNEL_UNAVAILABLE`, not a fabricated substitute.
- Do not invoke expensive neural processing only on an uncalibrated confidence heuristic without testing the miss rate on difficult examples.
- Inference is immutable; learning is a separate transaction.

---

## 9. Evidence separation and fusion rules

Every evidence item must be labeled as one of:

- `OBSERVED` — source pixels, measured behavior or measured neural data;
- `DERIVED` — deterministic feature or geometric transform;
- `PREDICTED` — top-down expectation from a model;
- `LEARNED` — stored parameter, transition or memory state;
- `CALIBRATED` — probability transformed using validation-fitted calibration;
- `EXTERNAL_REFERENCE` — lexicon, corpus or external biological reference.

Mandatory rules:

1. Raw visual evidence is immutable.
2. Predicted character identity never overwrites observed patch evidence.
3. Neural firing rates are not probabilities unless a calibrated mapping is supplied.
4. Functional A/B/O codes and neural position activity are linked by adapters, not treated as numerically identical by default.
5. Visual, geometric, positional, orthographic, recurrent and predictive scores remain separately reported.
6. Fusion may occur only through a declared combiner whose parameters are fitted using training/validation data and frozen before final test evaluation.
7. Semantic/contextual plausibility cannot be used to claim that the visual source contained a character that the visual evidence does not support.
8. Every output retains alternatives, unknowns, conflicts and abstention states.

---

## 10. Learning, checkpoint and memory governance

All learning paths obey one transaction protocol:

1. Freeze and fingerprint the parent model(s).
2. Validate sample provenance, split and label/evidence policy.
3. Require explicit authorization for updates to persistent models.
4. Create a detached child model or memory snapshot.
5. Record changed parameters and learning mechanism (`gradient`, `Hebbian`, `STDP`, `homeostatic`, `count_update`, etc.) separately.
6. Run unit, regression, minimal-pair, invariance and generalization tests.
7. Compare to the parent on locked regression data.
8. Commit the child only if all declared gates pass; otherwise retain the parent and record rollback.
9. Never allow recognition or B-N8 validation to mutate training models.

Functional and biological memories use different namespaces and update semantics. They may be initialized from common verified exposure events, but their state is not merged into one opaque memory object.

---

## 11. Acceptance structure

N9 reports three independent gates:

| Gate | Owner | What it establishes | What it does not establish |
|---|---|---|---|
| `VAL-SOFTWARE` | F-J / B-N9 | Contract integrity, unit tests, regression, determinism, safe checkpoint behavior | Recognition accuracy or brain correspondence |
| `VAL-TASK` | F-J / B-N9 | Held-out OCR/orthographic performance, robustness, generalization, calibration and efficiency | Biological implementation |
| `VAL-BIOLOGICAL` | B-N8 | Independent behavioral/neural correspondence under preregistered evaluation | General OCR performance outside tested tasks |

A release can state `SOFTWARE_ACCEPTED` while `VAL-TASK` or `VAL-BIOLOGICAL` remains `PENDING`. It must not say “biologically validated” until the biological gate passes.

### Minimum shared experiment matrix

Every relevant experiment should include:

1. Functional feed-forward baseline (`F-C/F-D/F-G`).
2. Biological feed-forward condition (`B-N1…B-N4`).
3. Biological recurrence condition (`B-N5`).
4. Biological prediction/precision condition (`B-N7`).
5. Functional + biological hybrid condition.
6. Ablations of position channels, recurrence, plasticity, prediction and precision.

Use identical stimuli, splits and predeclared metrics. Run multiple fixed seeds where stochasticity is present. Report accuracy and discrimination together, plus calibration, abstention risk/coverage, latency and memory use. N8 additionally compares model outputs with independent human/neural measurements.

---

## 12. Migration of the earlier 120-operation N9 proposal

The earlier 120-operation proposal remains a useful source of design material, but its part labels are not the canonical phase register. Reuse its content as follows:

| Earlier proposal block | Retain as | Canonical owner(s) |
|---|---|---|
| Operations 1–12: input orchestration/state setup | Shared orchestration contracts | F-A, F-B, B-N9 |
| Operations 13–24: early visual processing/segmentation | Visual pathway and image-region adapter | B-N2, B-N3, F-B/F-C |
| Operations 25–36: position/population/sequence hypotheses | Orthographic representation interface | F-D, B-N4 |
| Operations 37–48: attention/precision | Precision/attention subsystem | B-N7, with B-N5 where recurrent interaction is tested |
| Operations 49–60: recurrent sequence decoding/lexical and phonological interfaces | Orthographic decoding; optional downstream interfaces kept separate | F-G, B-N5, B-N7; optional extension namespace for phonology |
| Operations 61–72: syntax and semantic composition | Out-of-core language extension | Not assigned to B-N1…B-N9 |
| Operations 73–84: discourse/situation integration | Optional context provider; only prediction interfaces cross into core | B-N7 context input; downstream extension owns discourse model |
| Operations 85–96: hierarchical feedback/conflict | Recurrent/predictive refinement | B-N5, B-N7, B-N9 |
| Operations 97–108: integration/decision/output | System controller and output governance | B-N9, F-H, F-J |
| Operations 109–120: learning/consolidation/acceptance | Guarded learning and separate validation gates | B-N6, F-I, F-J, B-N8, B-N9 |

No operation is deleted merely because its old phase number changed. Its function, dependencies, tests and provenance are carried forward under the new ownership.

---

## 13. Recommended package layout

```text
ore/
├── contracts/                  # Stable, framework-neutral schemas/protocols
│   ├── envelope.py
│   ├── script_profile.py
│   ├── observations.py
│   ├── visual.py
│   ├── orthography.py
│   ├── neural_state.py
│   ├── prediction.py
│   ├── learning.py
│   ├── results.py
│   └── evaluation.py
├── functional/                 # Existing A–J implementation/adapters
│   ├── stage_a/ ... stage_j/
│   └── adapters/
├── biological/                 # Canonical B-N1…B-N8 implementations
│   ├── n1_simulation/
│   ├── n2_biological_vision/
│   ├── n3_hierarchical_vision/
│   ├── n4_orthography/
│   ├── n5_connectivity/
│   ├── n6_plasticity_memory/
│   ├── n7_prediction/
│   └── n8_validation/
├── integration/                # B-N9 controller and cross-track routing
│   ├── controller.py
│   ├── dependency_graph.py
│   ├── evidence_ledger.py
│   └── backend_registry.py
├── experiments/
│   ├── specs/
│   ├── runners/
│   ├── metrics/
│   └── reports/
└── tests/
    ├── contracts/
    ├── functional_regression/
    ├── biological_unit/
    ├── integration/
    └── scientific_validation/
```

This is a recommended logical layout, not a statement that these files already exist in the repository.

---

## 14. First implementation backlog

Implement interfaces before implementing additional neural mechanisms.

| Priority | Work item | Deliverable | Acceptance test |
|---:|---|---|---|
| P0 | Freeze this registry | Registry and migration decision recorded | No duplicate phase ownership or numbering |
| P1 | Implement `IF-00`–`IF-04` | Envelope, script profile, run context, observations and manifests | Schema round-trip, hashes, RTL/LTR and leakage tests |
| P2 | Implement `IF-05`–`IF-08` | Visual, grapheme, position and orthographic memory contracts | Functional A–J outputs adapt without loss of fields |
| P3 | Implement `IF-18` | Neural state snapshot and simulator protocol | Unit/time consistency and deterministic one-step/sequence tests |
| P4 | Implement `IF-09`–`IF-11`, `IF-15` | Learning request, model bundle and transaction result | Parent immutability, authorization and rollback tests |
| P5 | Implement `IF-10`, `IF-12`–`IF-14` | Prediction, sequence, result and trace contracts | Evidence separation, uncertainty and trace replay tests |
| P6 | Implement `IF-16`–`IF-17` | Experiment and evaluation report | Preregistered split, status separation and provenance tests |
| P7 | Build adapters | Functional and biological branches use common contracts | Run the same fixture through both branches and compare traces |
| P8 | Build N9 controller | `FUNCTIONAL`, `BIOLOGICAL`, `HYBRID` modes | Regression, compatibility, bounded recurrence and immutability |

### First vertical-slice experiment

Before adding more features, implement one small end-to-end experiment:

1. Load a known, versioned, grayscale patch sequence under a declared script profile.
2. Run the functional F-C visual encoder.
3. Run a B-N1/B-N2 minimal neural visual path on the same input.
4. Adapt both outputs to `IF-05` while preserving their representation kinds and units.
5. Run F-D A/B/O position backends and a B-N4 neural position backend.
6. Produce F-G candidates with separate visual, position, orthographic and neural evidence.
7. Save F-H trace/results and run the corresponding F-J software/behavioral tests.
8. Do not claim biological validation until B-N8 has independent empirical evidence.

This vertical slice validates the interfaces before the project invests in a full multi-population simulation or complex recurrent language processing.

---

## 15. Non-goals and scientific guardrails

This registry does not assert that:

- the human brain uses the exact equations or software structures adopted here;
- a Gabor filter bank is equivalent to retina/LGN/V1 physiology;
- a prototype classifier is a measured neural population;
- a recurrent network is automatically predictive coding;
- Hebbian or STDP learning automatically improves recognition;
- matching an OCR output proves biological correspondence;
- adding more modules necessarily improves speed, accuracy or efficiency.

The purpose is to make these hypotheses explicit, modular, measurable and falsifiable.

---

## 16. Final canonical decision

As of this specification:

- `F-A…F-J` is the functional orthographic engine register.
- `B-N1…B-N9` is the sole authoritative biological-development register.
- `B-N8` is independent brain-data validation.
- `B-N9` is integration/orchestration and system governance.
- The older N9 120-operation proposal is retained as a historical algorithm source and migrated by function, not by its old phase numbers.
- The shared contracts in Sections 5–7 are the official integration boundary.
- Functional and biological results must remain separately traceable and independently testable.

**Next implementation target:** `P1` — create the framework-neutral typed contract package and adapters, beginning with `IF-00` through `IF-04`.
