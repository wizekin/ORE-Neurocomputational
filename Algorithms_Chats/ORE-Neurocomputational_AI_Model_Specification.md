# ORE-Neurocomputational — AI Model Repository Specification

**Document purpose:** A repository-level technical reference for AI coding agents, research assistants, and maintainers that need to understand, inspect, extend, test, or integrate ORE-Neurocomputational.

**Repository:** <https://github.com/wizekin/ORE-Neurocomputational>  
**Owner:** `wizekin`  
**Default branch inspected:** `main`  
**Snapshot inspected:** 2026-11-21  
**HEAD commit inspected:** `e23e0122ee50e6f641a8d4984811224499b7c39e`  
**Latest commit message at inspection:** `docs: add integrated cognitive architectures algorithm for point 02`  
**License:** MIT (`LICENSE`)  
**Document status:** Repository specification based on the inspected source tree, project documentation, selected public APIs, package metadata, tests, and GitHub Actions workflows. It is not a claim that every source line has been independently audited or every test has been executed.

---

## 1. Executive overview

ORE-Neurocomputational is a modular Python research and engineering repository for biologically motivated models of visual perception, orthographic and lexical processing, language comprehension, semantic cognition, memory, learning, cognitive control, and coordinated multi-module reasoning.

It is **not one monolithic neural network**. It is a collection of specialized packages with their own typed requests/results, processing pipelines, intermediate representations, mathematical operations, operation registries, diagnostics, tests, and—in selected packages—optional biological-reference simulations.

The repository contains two architectural lines that must be kept separate when reasoning about implementation status:

1. **Legacy-compatible ORE packages**, especially `ore_n1`–`ore_n6` and `ore_n9`, plus later domain packages such as `ore_discourse`, `ore_semantic`, `ore_syntax`, and `ore_working_memory`. These are working research prototypes with their own historical `N` naming and package-specific interfaces.
2. **The canonical ORE architecture**, defined by functional phases `F-A`–`F-J` and biological phases `B-N1`–`B-N9`, with framework-neutral contracts, adapters, module manifests, exact-version compatibility, controlled learning, experiment/evaluation contracts, and an injected canonical controller in `ore_contracts`.

The old package names are **not aliases** for the new canonical phase identifiers. In particular, `ore_n9.N9Processor` is not the canonical `B-N9` controller. The legacy N labels must not be used as evidence that a canonical B-N phase has been implemented.

The most important shared infrastructure is:

- `ore_contracts`: canonical contracts, module manifests, artifacts, registries, adapters, compatibility checks, task planning, resource coordination, controlled learning, and canonical controller interfaces.
- `ore_workspace`: versioned, permission-controlled, provenance-aware shared state and module communication.
- `ore_logger`: shared SQLite source annotations and execution-history/event logging.
- `experiment_workbench`: a Streamlit laboratory for executing actual component callables, inspecting traces and arrays, and chaining in-session Python result objects.
- `ore_usg`: read-only import and adaptation of versioned USG-KNOWLEDGE bundles into ORE lexical/semantic resources.

The intended engineering pattern is **specialized modules + explicit typed contracts + bounded coordination + preserved evidence/provenance + inspectable traces + independent validation**.

## 2. Repository snapshot and inventory

At the inspected `main` snapshot:

| Item | Observed count / value |
|---|---:|
| Git tree entries | 805 |
| Python files (`.py`) | 647 |
| Top-level `ore_*` packages | 66 |
| Markdown files under `docs/` | 72 |
| Python test files under `tests/` | 111 |
| Example Python files under `examples/` | 17, including `__init__.py` |
| Python requirement | `>=3.10` |
| Project version | `0.1.0` |
| Default branch | `main` |

Counts are for the exact snapshot above. They will change as files are added, removed, or reorganized.

### 2.1 Build metadata

`pyproject.toml` defines:

- Build backend: `setuptools.build_meta`
- Build requirements: `setuptools>=68`, `wheel`
- Distribution name: `ore-neurocomputational-n1`
- Version: `0.1.0`
- Python: `>=3.10`
- Core dependencies: `numpy>=1.24`, `scipy>=1.10`
- Development extras: `pytest>=7.4`, `ruff>=0.4`
- Pytest test root: `tests/`
- Pytest default option: `-q`
- Ruff line length: 100
- Ruff target: Python 3.10

The distribution name still contains `-n1`; do not assume that the project is limited to N1. The source repository has expanded substantially beyond the original visual-processing pipeline.

### 2.2 Top-level layout

```text
.
├── .github/workflows/             # CI and manual test workflows
├── docs/                          # architecture, algorithms, equations, APIs, caveats
├── examples/                      # runnable demonstrations and integration examples
├── experiment_workbench/          # Streamlit execution/inspection laboratory
├── ore_* /                        # 66 specialized ORE Python packages
├── tests/                         # package, contract, integration, and workspace tests
├── N2_ARCHITECTURE.md
├── ORE_WORKSPACE_ARCHITECTURE.md
├── README.md
├── LICENSE
├── pyproject.toml
└── sitecustomize.py
```

Package-level source files commonly include `contracts.py`, `config.py`, `pipeline.py`, `functional.py`, `biology.py`, `registry.py`, `annotations.py`, and `__init__.py`, but the layout is package-specific. Do not assume every package has every file or uses the same internal layering.

## 3. Core conceptual model

### 3.1 A module is a specialist, not an autonomous general intelligence

An ORE package typically models one bounded capability: e.g. early visual processing, grapheme prediction, lexical access, semantic integration, memory retrieval, or corrective feedback. A package's presence does not mean the overall system autonomously performs every upstream or downstream task.

A caller must supply the input data and upstream products required by the package. Examples include localized image patches, lexical inventories, verified labels, predicate frames, task relevance, discourse history, world knowledge, or external neural/behavioral measurements.

### 3.2 Preserve representational distinctions

Across the repository, the following distinctions are important and should not be collapsed by adapters or refactors:

- Raw observations vs. extracted features.
- Physical image coordinates vs. logical reading order.
- Grapheme evidence vs. lexical identity.
- Visual evidence vs. contextual predictions.
- Contextual priors vs. posterior beliefs.
- Activation values vs. calibrated probabilities.
- Literal proposition content vs. inferred/pragmatic content.
- Episodic evidence vs. semantic/schema-generated inference.
- Observed, derived, predicted, learned, calibrated, and external-reference evidence.
- Functional computations vs. simulated biological-reference activity.
- Recognition/inference vs. explicitly authorized learning.
- A missing/unavailable stage vs. a stage that executed and returned an empty result.

When uncertainty is material, preserve alternatives, unknown mass, unresolved fields, and abstention rather than forcing a single answer.

### 3.3 Functional and biological-reference tracks

Many packages offer a functional computational path and an optional biological-reference path. The functional path generally implements explicit equations, candidate activation, evidence integration, or state transitions. The biological-reference path may use current-based leaky integrate-and-fire (LIF), rate-coded population dynamics, or another simplified neural abstraction.

These simulations are research/engineering analogies. They are not automatically anatomically fitted, biologically validated, or literal models of human neuronal computation. Never claim physiological validation from successful Python tests or plausible-looking simulation traces.

### 3.4 Traceability is a design goal

Many packages expose stable operation IDs, stage records, iteration information, diagnostics, warnings, confidence/uncertainty fields, and provenance. The same operation registry may serve both documentation and runtime tracing, but the existence of a registry row alone does not prove that a distinct computation is executed for that row. Confirm the implementation path and tests before claiming completion.

## 4. The two ORE architecture namespaces

### 4.1 Legacy-compatible `ore_n*` pipeline

The older implementation line includes:

| Package | Role | Main public API / boundary |
|---|---|---|
| `ore_n1` | Early visual encoding: illumination adaptation, retinal ON/OFF, center-surround, LGN-like gain control, Gabor orientation responses, pooling, optional temporal state, diagnostics. | `N1Config`, `N1Processor`, `N1Result`, `TemporalDynamics`; 2-D grayscale input. |
| `ore_n2` | Visual organization: retinotopy, scale-space, contours, boundaries, border ownership, figure-ground, junctions, fragments, shape graph, crowding, tolerance, recurrent refinement. | `N2Config`, `N2Input`, `N2Result`, `N2Processor`; consumes N1 spatial feature maps. |
| `ore_n3` | Recurrent orthographic prediction: supervised N2 patch descriptors, grapheme prototypes, unigram/bigram statistics, bidirectional neighbor messages, precision-weighted inference, error/residual traces. | `N3Config`, `N3Processor`, `N3Sequence`, `N3Result`, labelled patch contracts. |
| `ore_n4` | Attention and precision gating: bottom-up salience, task/class relevance, competitive position attention, class gates, sensory/prior precision, bounded posterior update. | `N4Config`, `N4Input`, `N4Result`, `N4Processor`; consumes N3 result read-only. |
| `ore_n5` | Experience-dependent plasticity: fast/slow prototypes, eligibility traces, replay, homeostasis, bounded drift, transition memories. | `N5Config`, `N5Experience`, `N5LearningRequest`, `N5ModelState`, `N5Result`, `N5Processor`; explicit authorized learning. |
| `ore_n6` | Hierarchical orthographic/lexical access: fast/slow forward/reverse transitions, bidirectional sequence inference, word-boundary estimates, provenance-tracked lexicon search. | `N6Config`, `N6Input`, `N6LexiconEntry`, `N6Result`, `N6Processor`; consumes N4 posterior and immutable N5 model snapshot. |
| `ore_n9` | Integrated controller across available visual/orthographic and optional later language/semantic modules; 120-operation inspectable trace. | `N9Request`, `N9Patch`, `N9Input`, `N9Adapters`, `N9Config`, `N9Result`, `N9Processor`. |

Important boundaries:

- **N1:** input is a 2-D grayscale image. Floating arrays must be finite and in `[0,1]`; integer grayscale inputs are scaled by the dtype maximum (e.g. uint8 / 255). RGB conversion is not part of the N1 input contract. `N1Result` preserves intermediate arrays in `stage_outputs`; the feature vector is orientation-major, row-major pooled output. Reset temporal state between unrelated sequences.
- **N2:** consumes N1's preserved `stage_outputs["normalized_energy"]` with spatial layout `[orientation, height, width]`. It must not reconstruct the maps from N1's flattened vector. N2 does not label graphemes. Single-frame motion energy is unavailable/zero; a temporal sequence interface would be needed for true motion processing.
- **N3:** requires labelled reference patches to fit its visual prototype/grapheme model and optionally labelled grapheme sequences to learn transitions. Recognition accepts already-segmented patches in logical order. It does not segment a line/page, perform lexical retrieval, or apply sentence semantics. Unknown mass (`UNK`) is an explicit outcome.
- **N4:** takes a validated N3 result and creates a new attended posterior. It preserves N3 visual probabilities and contextual priors separately and does not modify N3 arrays/state.
- **N5:** learning is separate from inference. The model requires explicitly authorized training requests with independent label verification and provenance; it returns a separately fingerprinted child model rather than silently mutating the parent. N4 signals can modulate learning.
- **N6:** uses N4 posterior information and an immutable N5 model snapshot. Lexicon entries and their provenance are supplied by the caller. Missing lexical entries produce explicit fallback/unknown outcomes; N6 does not infer semantic meaning.
- **N9:** expects localized patches and their logical reading order. It does not silently invent a full-page character segmenter. A patch can be supplied as a grayscale image, N2 result, feature vector, or grapheme-probability map. It can invoke N1/N2 for images; fitted N3/N4 processors and adapters for N5/N6, syntax, N7 discourse, and N8 semantic components are optional. Missing adapters are recorded as `not_configured`/unavailable, not fabricated as having run. N3 training is explicit and is not performed from an unlabelled recognition request.

The README documents 12 N1 operations, 17 N2 operations, 29 N3 operations, 40 N4 operations, 52 N5 operations, 60 N6 operations, and 120 N9 operations. Use each package's architecture file and runtime trace for the exact current operation definitions.

The README's newer N9 description maps later functional responsibilities approximately as N7 discourse (`ore_discourse`) and N8 semantic/concept processing (`ore_semantic` and related adapters). The N9 package also exposes a syntax adapter. Do not infer that separate `ore_n7` or `ore_n8` packages exist; they do not appear in the inspected tree.

### 4.2 Canonical functional phases `F-A`–`F-J`

Defined in `ore_contracts/__init__.py`:

| ID | Canonical name | Responsibility |
|---|---|---|
| `F-A` | Initialize and validate | Configuration, script profile, run state, compatibility checks. |
| `F-B` | Prepare learning data | Source registration, hashes, labels, split policy, safe transformations. |
| `F-C` | Learn visual and character representations | Functional visual encoder, identity evidence, view invariance, unknown mass. |
| `F-D` | Learn position and order representations | Independent absolute, relative/open-bigram, and overlap position backends. |
| `F-E` | Learn orthographic regularities | Soft transitions, candidate chunks, promotion, exposure memory. |
| `F-F` | Train, evaluate and freeze | Functional optimization, validation, calibration, immutable checkpoints. |
| `F-G` | Recognize a visual string | Candidate generation, evidence scoring, ranking, abstention. |
| `F-H` | Return result and trace | Structured output, trace serialization, provenance. |
| `F-I` | Optional incremental learning | Authorized child-model update, regression, rollback. |
| `F-J` | Acceptance and release | Software gates, task experiments, reproducibility, release decision. |

### 4.3 Canonical biological phases `B-N1`–`B-N9`

| ID | Canonical responsibility |
|---|---|
| `B-N1` | Neural simulation foundations: neuron state, membrane dynamics, firing, synaptic transmission, populations, simulation time. |
| `B-N2` | Biological visual processing: retinal/LGN-inspired preprocessing and V1-like responses. |
| `B-N3` | Hierarchical visual architecture: intermediate/higher visual populations, feature composition, invariance. |
| `B-N4` | Neural orthographic representations: distributed grapheme identity, identity-position conjunctions, sequence codes. |
| `B-N5` | Neural connectivity and recurrence: excitatory/inhibitory, local/long-range, recurrence and feedback. |
| `B-N6` | Biological learning and memory: plasticity, homeostasis, retention, forgetting. |
| `B-N7` | Predictive and contextual processing: top-down prediction, bottom-up residual, uncertainty, precision modulation. |
| `B-N8` | Brain-data validation: independent behavioral, representational, temporal and connectivity validation. |
| `B-N9` | Integrated neurocomputational ORE: orchestration, dependencies, evidence ledger, lineage and release governance. |

**Implementation status boundary:** the canonical framework and interfaces are present, and an included rate-based B-N1 reference backend exists. The controller is dependency-injected and reports unavailable backends. The P2–P8 implementation notes explicitly state that canonical biological B-N2–B-N8 implementations and independent brain-data validation still require their own scientific specifications and evidence. Do not equate the legacy N1–N9 packages with canonical B-N1–B-N9.

### 4.4 Canonical interface family IF-00–IF-18

`ore_contracts` defines a shared type system. The main interface groups are:

| Contract | Purpose |
|---|---|
| IF-00 `ContractEnvelope` | Contract/schema identity, run and producer identity, SHA-256 config/data fingerprints, UTC timestamp, evidence kind, parent lineage. |
| IF-01 `ScriptProfile` | Script/language profile, grapheme segmentation, Unicode, writing direction (`LTR`, `RTL`, `TTB`, `BTT`, `MIXED`), logical axis and geometry policy. Physical and logical coordinates stay distinct. |
| IF-02 `RunContext` | Operation mode, deterministic seed, compute budget, backend selection, timeout limits, immutable model references. |
| IF-03 `VisualObservation` | Immutable source reference/hash, image geometry/range, pixel/patch reference, segmentation alternatives/masks, transformation provenance. |
| IF-04 `DatasetManifest` | Dataset/license/provenance, per-sample hashes, label provenance, writer/font/source/string-family groups, split assignment, seed, leakage checks and locked preregistered tests. |
| IF-05 `VisualFeatureBundle` | Typed visual feature tensors, layout/dtype/units, spatial resolution, source regions, uncertainty/reliability, transform lineage. |
| IF-06 `GraphemeEvidence` | Per-region grapheme identity evidence, explicit unknown probability mass, supporting feature IDs, calibration status and backend provenance. |
| IF-07 `PositionCodeBundle` | Explicit `C`, `A`, `B`, `O` channels for character identity and position/order encodings; unavailable channels must remain unavailable rather than fabricated. |
| IF-08 `OrthographicMemorySnapshot` | Functional orthographic memory snapshot, separate from biological synapses. |
| IF-09 | Learning request. |
| IF-10 | Prediction state. |
| IF-11 | Model bundle reference. |
| IF-12 | Sequence hypotheses/candidates. |
| IF-13 | Recognition result. |
| IF-14 | Append-only trace event. |
| IF-15 | Learning result / detached model proposal. |
| IF-16 | Preregistered experiment specification. |
| IF-17 | Separate evaluation report. |
| IF-18 | Simulator-neutral neural state/simulation interface and rate-based reference simulator. |

`OperationMode` includes `FUNCTIONAL`, `BIOLOGICAL`, `HYBRID`, `TRAIN`, `INFER`, and `VALIDATE`. `EvidenceKind` includes `OBSERVED`, `DERIVED`, `PREDICTED`, `LEARNED`, `CALIBRATED`, and `EXTERNAL_REFERENCE`.

`ArtifactStatus` distinguishes `SPECIFIED`, `IMPLEMENTED`, `SOFTWARE_TESTED`, `TASK_VALIDATED`, `BIOLOGICALLY_SUPPORTED`, `PENDING`, `NOT_TESTED`, and `FAILED`. These statuses are independent. A model must not infer task validation or biological support from the fact that code exists or tests pass.

## 5. Shared integration and coordination

### 5.1 `ore_contracts`: module identity, contracts, adapters, and planning

Important files:

- `module_manifest.py`: `ModuleIdentity`, `InputPort`, `OutputPort`, `ModuleDependency`, `ModuleManifest`.
- `module_protocol.py`: `OREModuleProtocol`, `ModuleStatus`, `ModuleResult`.
- `module_artifacts.py`: typed `ContractArtifact` and content hashing.
- `contracts_registry.py`: exact-version `ContractCatalog`, contract definitions and validators.
- `module_registry.py`: source inventory, manifest registration, backend binding, capability discovery and diagnostics.
- `module_compatibility.py`: directed producer/consumer compatibility, explicit converters and dependency checks.
- `module_adapters.py`: wrappers for callable/legacy processors without requiring inheritance rewrites.
- `module_logging.py`: annotation and runtime event logging through `ore_logger`.
- `coordinator.py`: dependency-aware cognitive task planning/execution.
- `resource_profiles.py` and `resource_coordination.py`: task profiles, module resource estimation, reliability/cost history and adaptive plans.
- `governance.py`: guarded learning transactions and model fingerprints.
- `integration.py`: canonical functional/biological/hybrid/validation controller and backend registry.
- `simulation.py`: included rate-based reference neural simulation support.

Universal module contract:

```python
class OREModuleProtocol(Protocol):
    def describe(self) -> ModuleManifest: ...
    def execute(
        self,
        inputs: Mapping[str, ContractArtifact | Sequence[ContractArtifact]],
        context: RunContext,
    ) -> ModuleResult: ...
```

A manifest declares a stable namespaced module ID, full semantic version, responsibilities, typed input/output ports, namespaced capabilities, dependencies, execution modes and limitations. A backend must be separately bound and validated. Registering a manifest does **not** mean a module is executable.

`ModuleResult` status values are distinct: `success`, `partial`, `abstained`, `unavailable`, and `failed`. Abstention is not an exception. A missing backend is not a successful empty result.

Compatibility is exact by default:

- Producer and consumer contract IDs must match.
- Producer schema version must be explicitly listed by the consumer.
- Both contract models and payload/artifact validation must agree.
- A version conversion requires an explicitly registered converter and validation of the converted target artifact.

No fuzzy schema matching, implicit coercion, or silent version interpolation is intended. The registry itself is not a scheduler or learning engine; task execution is handled by the coordinator, and durable learning is governed separately.

### 5.2 `CognitiveTaskCoordinator`

`ore_contracts.coordinator` builds a graph from declared capabilities, exact contract versions, dependencies and supplied artifacts. It represents selected/required modules, edges, topological execution layers, per-port input plans, missing capabilities/dependencies/inputs, optional gaps, cycle nodes and capability coverage. A plan is executable only when its mandatory constraints are satisfied. This layer delegates specialist computation to registered module backends; it does not replace them.

### 5.3 Adaptive resource coordination

`AdaptiveResourceCoordinator` estimates task/module requirements and builds execution plans. It can use observed reliability and measured cost history, persisted through SQLite in the existing logger database. Information gain in the current implementation is explicitly heuristic unless a calibrated model is supplied. Do not describe it as calibrated information-theoretic optimality by default. Repeated failures can open a circuit-breaker cooldown.

### 5.4 `ore_workspace`: shared cognitive state

`ore_workspace` is a separate shared-state communication subsystem, not a central reasoning engine and not the same thing as the module registry.

Core public components include:

- `WorkspaceManager`, `SharedWorkspace`, `WorkspaceStateManager`
- `WorkspaceSnapshot`, `WorkspaceState`, `WorkspaceRevision`
- `Finding`, `FindingPublisher`, `FindingTarget`, `Provenance`
- `PermissionPolicy`, default write permissions and subscription rules
- `WorkspaceEventBus`, `SubscriptionRegistry`, `ModuleScheduler`
- `MergePolicyRegistry`, `ConflictRegistry`, `ConflictRecord`
- `HypothesisScorer`, `GoalEvaluator`, `WorkingMemoryPolicy`
- `SQLiteWorkspaceObserver`, `register_workspace_annotations`

Lifecycle:

1. Create a workspace/revision-zero state from input and goal.
2. Modules read detached snapshots and compute independently.
3. A producer submits a typed finding with target, producer identity, provenance, dependencies, optional confidence and base revision.
4. State manager validates schema, write permission, revision, provenance and dependencies.
5. Missing prerequisites are deferred; malformed or unauthorized writes are rejected.
6. Field-specific merge policies preserve competing hypotheses; contradictions create explicit conflict records rather than overwriting one candidate.
7. Valid changes commit atomically into reconstructable revisions and publish bounded events.
8. Subscribers are notified independently; a subscriber failure does not stop other subscribers.
9. Scheduler supports weighted activation, synchronous/asynchronous handlers, timeouts and bounded processing rounds.
10. Goal evaluation combines weighted coverage with explicit critical requirements, so a high average cannot hide a missing mandatory representation.

The workspace uses idempotence/fingerprints, a dependency graph, explicit revisions and optimistic concurrency. Evidence scores/confidence fields are not automatically calibrated probabilities. The detailed contract and equations are in `ORE_WORKSPACE_ARCHITECTURE.md`.

### 5.5 `ore_logger`: SQLite annotations and execution events

`ore_logger` uses SQLite and defines the shared tables:

- `annotations`: reusable explanations for modules/classes/functions/methods, signatures, source locations, source hashes, source-analysis/manual origin and stale status.
- `source_files`: source-file mtime, size, SHA-256, indexing time and AST parse errors.
- `runs`: unique run ID, local/UTC start, finish, duration, script/module identity, status and error summary.
- `events`: ordered function/operation events with source line, timing, thread information, exception information and metadata; events reference annotation IDs instead of duplicating descriptions.
- `settings`: stored settings.

Public package exports include `LoggerDatabase` and `sync_repository`. Runtime, recorder, catalog, database and viewer modules provide the additional services.

The default workbench database path is `ore_logs/ore_logger.sqlite3`. The logger is an observability service, not the shared workspace/message bus and not a reasoning module.

### 5.6 `experiment_workbench`: run and inspect actual Python components

The workbench is a reusable Streamlit laboratory. It executes selected Python callables and captures actual Python return objects; component-authored traces; Python call/return/exception events; nesting, source locations and timing; NumPy array shape/dtype/range/statistics; diagnostics and warnings; and JSON experiment reports. Successful in-session Python outputs can be reused as upstream artifacts without JSON round-tripping the object.

Current documented components include `ore_contracts`, lexical systems, `ore_n1`–`ore_n6`, `ore_n9`, `ore_predictive_orthography`, `ore_reading`, and `ore_semantic`.

Run from repository root:

```bash
python -m pip install -e .
python -m pip install -r experiment_workbench/requirements.txt
streamlit run experiment_workbench/app.py
```

The workbench is a test/inspection harness, not an LLM. A text prompt cannot substitute for a required image, patch sequence, lexicon, verified training sample, or upstream result. Its Python profiling does not capture NumPy/SciPy C-level internals or work dispatched into other threads/processes. Function durations are inclusive and nested timings overlap.

**Security:** run trusted Python packages only. Package imports execute code. Keep the workbench local unless authentication, isolation, resource limits and uploaded-data controls are implemented. The documented observer does not store raw prompts, image pixels, complete arrays or Python argument values in event rows.

## 6. Language and visual-processing systems beyond N1–N9

### 6.1 `ore_sensory_perception`

Six-stage, 39-operation visual-and-auditory system. Stages 1–2 locate/segment text-like visual structures; stages 3–4 extract time-varying acoustic features; stage 5 estimates distributed phonetic-feature evidence; stage 6 updates/selects language-specific phoneme hypotheses. Visual and auditory paths remain distinct until explicit shared-language readout. The visual output is glyph/letter evidence, not a claim of general-purpose OCR.

### 6.2 `ore_brain_networks` and related visual neural packages

`ore_brain_networks` contains a 12-operation visual/orthographic/language/auditory/neural-measurement/causal-validation pipeline: Gabor filters, grapheme populations, recurrent LIF dynamics, visual hierarchy, VWFA hierarchy, IFG/MTG/ATL language networks, auditory dorsal/ventral pathways, EEG/MEG and fMRI forward models, virtual lesion/TMS analyses, and independent human-reading validation. Synthetic EEG/MEG and BOLD are proxies/toy models, not measured brain data.

`ore_vwfa` models a six-stage VWFA hierarchy. `ore_atl_amodal` provides an ATL-inspired amodal semantic hub. Both should be read alongside their dedicated architecture documents; neither should be represented as a literal validated cortical replica.

### 6.3 Orthographic, lexical and syntax systems

- `ore_predictive_orthography`: seven-stage, 40-operation master predictive orthography algorithm. Composes bottom-up grapheme evidence and bidirectional transition predictions with attention and precision gating; exposes classwise errors and distinct contextual/attended residuals.
- `ore_lexical_context`: 40-operation lexical context/prediction pipeline: word exposure/frequency, lexical activation/settling, morphology, pronunciation mapping, orthographic competition, context, upcoming-word prediction, surprisal, semantic residuals and precision-weighted updates.
- `ore_lexical_access`: seven-stage, 36-operation lexical/morphological access pipeline: identity, meaning, grammatical category, morphological decomposition, stem/affix retrieval, alternatives and incremental lexical activation. It reuses supplied, provenance-tracked lexical/morpheme data; homographs and competing parses are preserved, and the system can abstain.
- `ore_syntax`: nine-stage syntactic parsing pipeline for POS alternatives, phrases, dependency and constituency structure, hierarchical representation, thematic roles, ambiguity resolution and long-distance dependencies. It exposes parse alternatives and trace/diagnostics. It is a configurable research parser; inputs/lexical evidence matter.
- `ore_garden_path`: incremental ambiguity/garden-path research model with lexical-category alternatives, supplied grammar hypotheses, surprisal, parse competition, dependency load, disambiguation, repair and feedback-based adaptation. Without grammar candidates it returns a surface fallback/unresolved interpretation; it is not a general-purpose grammar.

### 6.4 Sentence, discourse and integrated comprehension

- `ore_semantic_composition`: nine stages, 54 operations to build predicate/argument structure, thematic roles, propositions, negation/modality, quantifier scope, causal relations and typed sentence semantic graphs. Consumes supplied semantic concepts and optional syntax result/predicate frames; preserves alternatives and can abstain.
- `ore_discourse`: eight stages, 39 operations for discourse state, consecutive proposition integration, speaker intention, indirect requests, irony, situation models, temporal relations and topic/narrative coherence. Caller-provided world knowledge, prosody, entities and common ground remain explicit inputs.
- `ore_working_memory`: five stages, 40 operations covering buffers, incomplete syntax, executive control, semantic resolution and task switching. It preserves chunks, unresolved dependencies, separate evidence channels, interpretation alternatives and checkpoints/resumption rather than filling gaps silently.
- `ore_comprehension`: top-down incremental comprehension architecture (see `docs/TOP_DOWN_INCREMENTAL_COMPREHENSION_38_OPERATION_ARCHITECTURE.md`).
- `ore_reading`: 72-operation integrated reading algorithm connecting orthographic lexicon and interactive activation, dual-route reading, grapheme-to-phoneme conversion, phonological assembly, TRACE-style word competition, Cohort narrowing, Shortlist-B-like Bayesian selection and a spelling-sound-meaning triangle. It accepts text/tokens or upstream recognized tokens; it does not OCR pixels or segment a full image. Functional and LIF reference branches are separate.
- `ore_language_comprehension`: 15-point language-comprehension collection. Points 1–9 are cognitive processing/integration and neural-architecture components; Points 10–15 are primarily research, methodological validation, network/temporal analysis, or profile workflows—not six more mandatory serial sentence-processing stages. Point adapters reuse established ORE subsystems.
- `ore_language_domain_reasoning`: language-specific networks and domain-general reasoning model (see `docs/LANGUAGE_SPECIFIC_NETWORKS_DOMAIN_GENERAL_REASONING_POINT15.md`).
- `ore_text_localization`: transforms printed-text images into ordered glyph patches, preserves segmentation alternatives and source coordinates, reuses N9 pathway components, hands supported interpretations to working memory, and registers with the task coordinator. It is an integration/segmentation layer, not a claim of complete end-to-end OCR for all scripts.

## 7. Semantic architecture variants

The repository intentionally retains several alternative semantic-cognition approaches rather than forcing them into one implementation:

| Package | Model focus |
|---|---|
| `ore_semantic` | 44-operation semantic access/concept selection, episodic word-form retrieval, modality spokes, distributed feature activation, ATL-inspired hub, IFG-inspired semantic control and abstract lexical prototypes. Evidence channels remain inspectable; ambiguity/low confidence can produce abstention. |
| `ore_hub_spoke` | 21-stage Hub-and-Spoke integration: seven modality spokes, reliability-weighted nonlinear hub integration, recurrence, feedback, cross-modal invariance, semantic control, experience learning and lesion experiments. |
| `ore_csc` | 18-stage / 78-operation Controlled Semantic Cognition model with semantic-control pipeline and optional LIF reference. It is separate from `ore_semantic` and `ore_hub_spoke`. |
| `ore_hybrid_embodied_amodal` | 19-stage hybrid embodied/amodal concept processing with modality-specific representations, ATL-like common information, residual separation, abstraction, spoke feedback, predictive correction, response choice and gated learning. |
| `ore_distributed_semantics` | 36-stage distributed semantic composition/integration system: lexical/concept activation, modality features, cross-modal association, bilateral ATL-like convergence, graph propagation, phrase composition, thematic roles, context, ambiguity, episodic/schema retrieval and learning. |
| `ore_distributed_only` | 21-stage distributed-only concept representation with separate visual, auditory, action, tactile, affective, linguistic, contextual and spatial populations; no central semantic vector. |
| `ore_grounded_integration` | Grounded semantic integration system; see `docs/GROUNDED_SEMANTIC_INTEGRATION_22_STAGE_ARCHITECTURE.md`. |
| `ore_commonality_concepts` | Commonality-based concept representation/abstraction; see its package code and annotations. |
| `ore_prototype_exemplar` | Prototype- and exemplar-based concept representation/selection; see its package code and annotations. |

`ore_semantic`, `ore_semantic_composition`, `ore_csc`, `ore_hub_spoke`, and `ore_distributed_semantics` have different scopes. Concept retrieval/selection, sentence-level composition, controlled semantic cognition, hub-and-spoke representation, and distributed semantic integration must not be treated as interchangeable APIs.

## 8. Memory, learning, cognitive control and reasoning systems

The following table is a navigation index. For each package, its package-level contracts, pipeline/functional implementation, registry, tests and matching document are the source of detailed behavior.

| Package | Main research/engineering responsibility |
|---|---|
| `ore_active_engagement` | Active engagement as a learning/retention mechanism; see `docs/ACTIVE_ENGAGEMENT_POINT01.md`. |
| `ore_adult_individual_differences` | Adult individual differences in learning/comprehension; see Point 17 document. |
| `ore_analogical_transfer` | Mapping relational structure between source and target situations and evaluating transfer. |
| `ore_attentional_failure` | Attentional failures and their effects on cognitive processing; Point 18. |
| `ore_cognitive_heuristics` | Heuristics and systematic judgment errors, verification and feedback; 17 operations. |
| `ore_cognitive_load` | Cognitive load and processing constraints; Point 06. |
| `ore_context_dependent_interpretation` | Context-sensitive interpretation and competing readings; Point 12. |
| `ore_confirmation_bias` | 14-operation analysis of belief effects on source selection, attention, encoding, interpretation, readout, updating and retrieval; includes accuracy-first mode. |
| `ore_corrective_learning` | 20-operation feedback/correction loop with evidence provenance, independent-source weighting, Bayesian updates, clarification/verification, retrieval/transfer tests and calibration. Requires caller-supplied likelihoods/reliability/evidence; does not fetch evidence or invent explanations. |
| `ore_cortical_learning` | Cortical/experience-dependent learning model; functional and optional biological reference paths. |
| `ore_curiosity_motivation` | 24-step curiosity/motivated-learning algorithm and delayed-learning evaluation. |
| `ore_developmental_language_cognition` | Developmental language/concept/executive cognition model; Point 16. |
| `ore_emotion_motivation` | Emotion, motivation, attention and memory interactions; Point 13. |
| `ore_feedback_seeking` | 18-step feedback-seeking/correction workflow with source evaluation, calibration and delayed-retention scheduling. |
| `ore_garden_path` | Incremental syntactic ambiguity, repair and good-enough interpretation. |
| `ore_hippocampal_encoding` | Hippocampal-style episodic encoding model. |
| `ore_memory_integration` | Connects Point 7 episodic/semantic stores with workspace, goal-sensitive working memory, semantic processing, schema learning, analogical transfer, reconstructive memory and cognitive control. Recall is read-only and excludes current sequence; pending semantic claims remain unverified; episode encoding is separately gated. |
| `ore_memory_transformation` | Memory transformation/consolidation across memory representations. |
| `ore_novelty_consolidation` | Novelty-sensitive consolidation. |
| `ore_prior_knowledge` | Prior knowledge activation and integration; Point 02. |
| `ore_reconstructive_memory` | 16-operation schema-driven reconstructive memory; separates event evidence from inferred/source-uncertain details, contradictions and unknowns. Includes source monitoring, schema competition, temporal/spatial/causal checks, confidence and provenance DAG. |
| `ore_schema_formation` | Schema formation from experience. |
| `ore_schema_guided_inference` | Schema-guided inference and knowledge application. |
| `ore_sleep_consolidation` | Sleep-related memory consolidation approximation. |
| `ore_social_conversational_learning` | Social interaction and conversational learning; Point 14. |
| `ore_spaced_repetition` | Spaced-repetition learning and scheduling; Point 03. |
| `ore_testing_effect` | Retrieval-practice/testing-effect learning; Point 04. |
| `ore_varied_contexts` | Varied-context learning and transfer; Point 05. |
| `ore_working_memory_overload` | 19-operation evidence-aware model of overload, unresolved dependencies, similarity interference and failure localization across encoding/maintenance/retrieval/binding/integration. |
| `ore_language_domain_reasoning` | Interaction between language-specific networks and domain-general reasoning. |
| `ore_comprehension` | Top-down incremental language comprehension architecture. |

### 8.1 Recurrent feedback integration — Task 06

`RecurrentFeedbackController` in `ore_corrective_learning` is an outer bounded coordination loop around Point 24. It can:

1. Freeze the current input and initial interpretation.
2. Run Point 24's initial hypothesis/evidence evaluation.
3. Measure available compatibility dimensions and detect a possible conflict.
4. Route targeted checks to confirmation-bias, heuristic, curiosity, feedback-seeking, memory and semantic modules where inputs/handlers are supplied.
5. Rank caller-supplied information actions using expected information gain minus cost.
6. Invoke one bounded action and validate the returned evidence.
7. Apply only the new validated evidence batch against the previous posterior, avoiding repeated evidence counting.
8. Optionally run a supplied N9 read-only reprocessing pass.
9. Stop when evidence/conflict/uncertainty gates are satisfied, no progress is made, or the iteration/action/cost budget is exhausted.

The controller does not invent actions, evidence, likelihoods, or missing module outputs. An unavailable component stays unavailable. Ordinary inference does not update persistent model parameters. A separate authorized learning transaction returns an unpromoted child model.

## 9. USG knowledge integration

`ore_usg` adapts versioned `USG-KNOWLEDGE/0.1.0` bundles to ORE resources. It is an **import/read-only adapter**, not the USG database or a full USG authoring system.

Documented capabilities include:

- Deterministic loading of local JSON bundles/files/directories and digest verification.
- Lexical forms, lemmas and senses.
- Concepts and typed mappings.
- Semantic classes and predicate frames.
- Explicit cross-resource relations.
- Unified `USGKnowledgeResource` preserving source, evidence, context, lifecycle and bundle provenance.
- Exact retrieval and module-registry integration.
- Bridges to build lexical access and semantic composition processors.

Default lifecycle/rights behavior:

- By default, import accepted or validated-for-profile records; candidate records are excluded unless policy explicitly includes them.
- Unresolved, deprecated and rejected records are not imported as active entries.
- Sources must have known permitted/internal rights status and the record must have source/evidence provenance.
- Polysemous senses remain separate; contradictory evidence is preserved rather than automatically resolved.
- The adapter does not infer lemma/morphology links that are absent from the source.
- No USG SQLite repository is opened or written; the caller supplies a bundle snapshot.
- Predicate frames remain preserved data; they are not automatically compiled into executable rules.

Primary documentation: `docs/USG_ADAPTERS.md`. Focused tests include `tests/test_usg_bundle_io.py`, `test_usg_knowledge_adapter.py`, `test_usg_lexical_adapter.py`, `test_usg_processor_bridges.py`, `test_usg_retrieval.py`, `test_usg_semantic_adapter.py`, and `test_usg_structure_adapter.py`.

## 10. Package inventory — all 66 `ore_*` packages

This inventory is grouped by concern, not by mandatory execution order.

### Visual, orthographic, language and integrated processing (24)

1. `ore_n1`
2. `ore_n2`
3. `ore_n3`
4. `ore_n4`
5. `ore_n5`
6. `ore_n6`
7. `ore_n9`
8. `ore_sensory_perception`
9. `ore_brain_networks`
10. `ore_vwfa`
11. `ore_atl_amodal`
12. `ore_bnbea`
13. `ore_predictive_orthography`
14. `ore_lexical_access`
15. `ore_lexical_context`
16. `ore_syntax`
17. `ore_semantic_composition`
18. `ore_discourse`
19. `ore_working_memory`
20. `ore_comprehension`
21. `ore_reading`
22. `ore_language_comprehension`
23. `ore_language_domain_reasoning`
24. `ore_text_localization`

### Semantic representation and control (9)

25. `ore_semantic`
26. `ore_hub_spoke`
27. `ore_csc`
28. `ore_hybrid_embodied_amodal`
29. `ore_distributed_semantics`
30. `ore_distributed_only`
31. `ore_grounded_integration`
32. `ore_commonality_concepts`
33. `ore_prototype_exemplar`

### Memory, learning, development, attention and reasoning (29)

34. `ore_active_engagement`
35. `ore_adult_individual_differences`
36. `ore_analogical_transfer`
37. `ore_attentional_failure`
38. `ore_cognitive_heuristics`
39. `ore_cognitive_load`
40. `ore_context_dependent_interpretation`
41. `ore_confirmation_bias`
42. `ore_corrective_learning`
43. `ore_cortical_learning`
44. `ore_curiosity_motivation`
45. `ore_developmental_language_cognition`
46. `ore_emotion_motivation`
47. `ore_feedback_seeking`
48. `ore_garden_path`
49. `ore_hippocampal_encoding`
50. `ore_memory_integration`
51. `ore_memory_transformation`
52. `ore_novelty_consolidation`
53. `ore_prior_knowledge`
54. `ore_reconstructive_memory`
55. `ore_schema_formation`
56. `ore_schema_guided_inference`
57. `ore_sleep_consolidation`
58. `ore_social_conversational_learning`
59. `ore_spaced_repetition`
60. `ore_testing_effect`
61. `ore_varied_contexts`
62. `ore_working_memory_overload`

### Shared infrastructure (4)

63. `ore_contracts`
64. `ore_workspace`
65. `ore_logger`
66. `ore_usg`

Supporting non-`ore_*` components: `examples/`, `experiment_workbench/`, `tests/`, `docs/`, and `sitecustomize.py`.

## 11. Running, testing and inspecting the repository

### 11.1 Create an environment and install

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 11.2 Test commands

Run the full suite:

```bash
python -m pytest -q
```

Run one focused package suite, for example:

```bash
python -m pytest tests/test_n1.py -q
python -m pytest tests/test_n3.py -q
python -m pytest tests/test_module_registry.py -q
python -m pytest tests/workspace/test_workspace.py -q
python -m pytest tests/test_usg_knowledge_adapter.py -q
```

Run Ruff if needed:

```bash
ruff check .
```

Use a focused test during development, then the relevant integration tests, then the full suite before claiming repository-wide compatibility. The user may prefer downloading the repository and running the complete suite locally because full-suite pytest execution can be time-consuming; this document does not assume a particular machine's test duration.

### 11.3 CI

Inspected workflows:

- `.github/workflows/python-tests.yml`: runs on pushes to `main` and `feature/**`, and pull requests to `main`; Ubuntu; Python 3.11; installs `-e ".[dev]"`; runs `python -m pytest -q`; cancels superseded runs in the same concurrency group.
- `.github/workflows/tests.yml`: manual `workflow_dispatch` test workflow; also installs `-e ".[dev]"` and runs `python -m pytest -q`.
- `.github/workflows/hybrid-embodied-amodal.yml`: package-specific workflow exists; inspect its current contents before relying on its trigger or gates.

No statement in this specification asserts that the workflows passed at the inspection time. Repository tree and workflow configuration were inspected; tests were not executed as part of this specification task.

## 12. AI agent operating protocol

This section is normative guidance for any model or coding agent changing this repository.

### 12.1 Start with repository evidence

Before changing code:

1. Confirm the branch and current commit.
2. Read the root `README.md` and the relevant architecture document under `docs/`.
3. Inspect the target package's `__init__.py`, contracts, config, pipeline, functional implementation, biological implementation (if present), and registry.
4. Read the focused tests and any integration tests that consume the package.
5. Search for callers/adapters before changing a public contract.
6. Distinguish legacy N identifiers from canonical F/B-N identifiers.

Never base a behavioral change only on package naming or on a README summary. The source code, public contracts, tests, and package-specific architecture document must agree.

### 12.2 Preserve backward compatibility

- Prefer additive changes, explicit adapters, and versioned contracts over rewriting legacy request/result objects.
- Do not rename `ore_n*` packages to canonical `B-N*` names.
- Do not change public field names, operation IDs, package exports, or serialized shapes without checking dependents and updating tests/docs.
- If a schema changes incompatibly, increment the contract version and provide an explicit converter where appropriate.
- Preserve current stable IDs; never recycle an old operation/module/contract ID for a new meaning.

### 12.3 Preserve evidence, provenance and uncertainty

- Never manufacture a source, data label, lexical entry, predicate frame, measurement, probability, or human-validation result.
- Keep `OBSERVED`, `DERIVED`, `PREDICTED`, `LEARNED`, `CALIBRATED`, and `EXTERNAL_REFERENCE` distinct.
- Carry source IDs, hashes, parent artifact IDs, dataset/split identity, model fingerprints and transformation lineage when the relevant contracts expose them.
- Do not call uncalibrated activation/confidence a probability.
- Retain competing candidates and unknown mass where supported.
- Use explicit `ABSTAINED`, `UNAVAILABLE`, `PARTIAL`, `FAILED`, `not_configured`, or fallback statuses instead of inventing success.
- Never convert missing evidence into a negative finding by default.

### 12.4 Keep inference and learning separate

- Normal recognition/inference should be read-only with respect to learned model parameters unless a package explicitly documents otherwise.
- Learning must use an explicit learning request, authorization, provenance/verification policy, parent fingerprint, detached proposed child and regression/acceptance gates where required.
- Do not silently train from unlabelled user input or promote a child model into durable state.
- Do not reuse evidence in recurrent correction as if it were new evidence; pass only the newly validated evidence batch to the next posterior update.

### 12.5 Keep functional and biological evidence separate

- Functional output and LIF/rate/current-based simulations are distinct tracks.
- Do not alter functional probabilities merely because a biological simulator ran.
- Do not claim that a toy EEG/MEG forward model, HRF-convolved BOLD proxy, model-only lesion, or simulated TMS is empirical brain evidence.
- Biological support requires independent reference measurements and validation criteria.

### 12.6 Test strategy

For every implementation change:

1. Add/adjust input-contract tests, including invalid/non-finite/shape/range/version cases.
2. Test the normal path and all explicit fallback/abstention/unavailable paths.
3. Test that upstream inputs/state are not mutated when the contract promises immutability.
4. Test determinism/reproducibility where seeds are supported.
5. Test operation ordering, stable trace IDs, diagnostics and provenance.
6. Test explicit learning authorization and parent/child fingerprint behavior for learning code.
7. Test integration adapters and compatibility/version failures.
8. Run the focused suite, related integration suites and the full suite.
9. Report exact commands, environment and test outcomes. Do not claim tests passed if they were only inspected.

### 12.7 Code design expectations

- Use typed dataclasses/contracts and validate inputs at boundaries.
- Prefer deterministic, explicit numerical operations with configurable parameters and visible defaults.
- Preserve pure/stateless functions where possible; isolate stateful processors and document reset semantics.
- Bound iterative loops, retries, recurrence, scheduler rounds and resource budgets.
- Use stable random seeds for reproducible experiments.
- Preserve array layout, dtype, units and coordinate conventions.
- Make errors actionable; use exceptions for malformed inputs and explicit result statuses for runtime unavailability/abstention.
- Update module docstrings, architecture docs, operation registries, examples and tests together.

### 12.8 Scientific claims and terminology

Use phrases such as “biologically motivated”, “inspired by”, “functional approximation”, “population-rate reference”, “current-based LIF reference”, or “computational hypothesis” where appropriate. Avoid “the brain does exactly this”, “human-validated”, “biologically accurate”, or “AGI” unless strong independent evidence actually supports that exact claim.

### 12.9 Safe repository changes

- Do not delete generated/user data, databases, model bundles or source artifacts without a clear task requirement.
- Do not run broad refactoring merely to normalize style across unrelated packages.
- Do not let one specialist package silently take ownership of another package's state.
- Do not let a logger become an inference dependency; logging failures should not corrupt the underlying cognitive result unless explicitly designed and tested.
- Do not expose the Streamlit workbench publicly without authentication, isolation and execution/resource controls.

## 13. Known documentation/versioning caveats to verify during future work

The inspected repository contains a few documentation/status mismatches. AI agents should resolve these against the current implementation before modifying or documenting the affected area:

1. **Legacy vs canonical N labels:** older packages `ore_n1`–`ore_n9` predate the canonical F/B-N register. The architecture docs explicitly say these labels overlap in name but are not aliases.
2. **Historical P1 status text:** `docs/CANONICAL_PHASE_REGISTRY_AND_P1_CONTRACTS.md` contains an earlier P1-scope paragraph that describes P2–P8 as backlog. The later `docs/CANONICAL_ORE_P2_P8_IMPLEMENTATION.md` documents P2–P8 deliveries. Use the later implementation note and current code for current status; retain the historical P1 document as a baseline, not the sole current-status authority.
3. **Syntax operation count:** the README's syntax heading says 56 operations, while a language-comprehension Point 3 passage refers to 58 lower-level syntax operations. The source-to-code crosswalk records 56 repository operations against a 55-operation source. Inspect `ore_syntax/registry.py`, runtime trace, and focused tests before repeating a precise count.
4. **README opening title:** the README begins with an N1–N4 title even though later sections and packages document N5, N6, N9, canonical contracts, and many cognitive modules. The opening title is historical/narrow, not a complete project inventory.
5. **Legacy N7/N8 mapping:** the N9 docs describe N7/N8 responsibilities via `ore_discourse` and `ore_semantic`/semantic adapters; no top-level `ore_n7` or `ore_n8` package exists in the inspected tree.
6. **Research audit boundaries:** `docs/ORE_SOURCE_TO_CODE_IMPLEMENTATION_CROSSWALK.md` identifies partial/restructured mappings and explicitly says its baseline was a static audit, not a test run or proof of biological validity. Treat its I/M/P/T/N/X statuses as audit evidence, not as universal implementation proof.
7. **Packaging name:** distribution metadata is still `ore-neurocomputational-n1`; this is not the same as the repository's full scope.
8. **Workbench install:** `experiment_workbench` has its own `requirements.txt` and README installation steps; do not assume Streamlit is installed by the base project extras.

When updating these topics, edit the authoritative package docs and tests, then update README and this specification to keep the status synchronized.

## 14. Recommended documentation reading order for new AI models

For an agent that needs a fast but accurate mental model, read in this order:

1. `README.md` — current overall feature inventory and example commands.
2. `N2_ARCHITECTURE.md` — an example of a self-contained stage-level architecture specification.
3. `ORE_WORKSPACE_ARCHITECTURE.md` — shared-state/message coordination semantics.
4. `docs/CANONICAL_PHASE_REGISTRY_AND_P1_CONTRACTS.md` — phase namespaces and original P1 contracts.
5. `docs/CANONICAL_ORE_P2_P8_IMPLEMENTATION.md` — subsequent canonical contracts and controller scope.
6. `docs/ORE_MODULE_COMMUNICATION_REGISTRY.md` — module manifests, adapters, exact compatibility and status semantics.
7. Target package's architecture document and `__init__.py`.
8. Target package's contracts/config/pipeline/functional/biology/registry files.
9. Focused tests and integration tests.
10. `docs/ORE_SOURCE_TO_CODE_IMPLEMENTATION_CROSSWALK.md` and `docs/ORE_SOURCE_GAP_IMPLEMENTATION_ADDENDUM.md` when comparing implementation against the original source algorithms.

## 15. Important architecture and audit documents

The following files are high-value starting points. The docs directory contained 72 Markdown files at the inspected snapshot.

### Repository-wide architecture, crosswalk and governance

- `N2_ARCHITECTURE.md`
- `ORE_WORKSPACE_ARCHITECTURE.md`
- `docs/ORE_MODULE_COMMUNICATION_REGISTRY.md`
- `docs/ORE_SOURCE_TO_CODE_IMPLEMENTATION_CROSSWALK.md`
- `docs/ORE_SOURCE_GAP_IMPLEMENTATION_ADDENDUM.md`
- `docs/CANONICAL_PHASE_REGISTRY_AND_P1_CONTRACTS.md`
- `docs/CANONICAL_ORE_P2_P8_IMPLEMENTATION.md`

### Core visual, orthographic and language pipeline

- `docs/N3_ARCHITECTURE.md`
- `docs/N4_ARCHITECTURE.md`
- `docs/N5_ARCHITECTURE.md`
- `docs/N6_ARCHITECTURE.md`
- `docs/N9_ARCHITECTURE.md`
- `docs/MASTER_PREDICTIVE_ORTHOGRAPHY_7_STAGE_ALGORITHM.md`
- `docs/LEXICAL_ACCESS_7_STAGE_FINAL_ARCHITECTURE.md`
- `docs/LEXICAL_CONTEXT_40_OPERATION_REGISTRY.md`
- `docs/SENSORY_PERCEPTUAL_6_STAGE_ARCHITECTURE.md`
- `docs/VWFA_SIX_STAGE_ARCHITECTURE.md`
- `docs/SYNTAX_PARSING_ARCHITECTURE.md`
- `docs/SEMANTIC_COMPOSITION_9_STAGE_ARCHITECTURE.md`
- `docs/DISCOURSE_EIGHT_STAGE_ARCHITECTURE.md`
- `docs/WORKING_MEMORY_COGNITIVE_CONTROL_ARCHITECTURE.md`
- `docs/INTEGRATED_READING_OPERATION_REGISTRY.md`
- `docs/LANGUAGE_SPECIFIC_NETWORKS_DOMAIN_GENERAL_REASONING_POINT15.md`

### Semantic and memory architectures

- `docs/ATL_AMODAL_SEMANTIC_19_STAGE_ARCHITECTURE.md`
- `docs/HUB_AND_SPOKE_21_STAGE_ARCHITECTURE.md`
- `docs/CSC_18_STAGE_ARCHITECTURE.md`
- `docs/HYBRID_EMBODIED_AMODAL_19_STAGE_ARCHITECTURE.md`
- `docs/DISTRIBUTED_NEURAL_SEMANTIC_36_STAGE_ARCHITECTURE.md`
- `docs/DISTRIBUTED_ONLY_SEMANTIC_ARCHITECTURE.md`
- `docs/GROUNDED_SEMANTIC_INTEGRATION_22_STAGE_ARCHITECTURE.md`
- `docs/SEMANTIC_COGNITION_44_OPERATION_REGISTRY.md`
- `docs/RECONSTRUCTIVE_MEMORY_POINT20.md`
- `docs/WORKING_MEMORY_OVERLOAD_POINT19.md`
- `docs/MEMORY_INTEGRATION_TASK05.md`
- `docs/RECURRENT_FEEDBACK_TASK06.md`
- `docs/USG_ADAPTERS.md`

### Research-point specifications

- `docs/LANGUAGE_COMPREHENSION_POINT_01.md` through `docs/LANGUAGE_COMPREHENSION_POINT_15.md`
- `docs/ACTIVE_ENGAGEMENT_POINT01.md`
- `docs/POINT_02_PRIOR_KNOWLEDGE.md`
- `docs/POINT_03_SPACED_REPETITION.md`
- `docs/POINT_04_TESTING_EFFECT.md`
- `docs/POINT_05_VARIED_CONTEXTS.md`
- `docs/COGNITIVE_LOAD_POINT06.md`
- `docs/FEEDBACK_SEEKING_POINT07.md`
- `docs/CURIOSITY_MOTIVATION_POINT08.md`
- Schema formation and schema-guided inference: inspect `ore_schema_formation/`, `ore_schema_guided_inference/`, and their focused tests; do not assume a dedicated architecture document exists unless it is present in the checked-out `docs/` tree.
- `docs/CONFIRMATION_BIAS_POINT21.md`
- `docs/COGNITIVE_HEURISTICS_POINT22.md`
- `docs/GARDEN_PATH_POINT23.md`
- `docs/CORRECTIVE_LEARNING_POINT24.md`

## 16. Practical integration examples

### 16.1 N1 → N2

```python
import numpy as np
from ore_n1 import N1Processor
from ore_n2 import N2Processor

image = np.random.default_rng(7).random((128, 128), dtype=np.float32)
n1_result = N1Processor().process(image)
n2_result = N2Processor().process(n1_result)

print(n1_result.feature_vector.shape)
print(n2_result.structured_output)
```

For actual input, follow N1's normalized grayscale contract; random synthetic data is only a software example.

### 16.2 Train then recognize with N3

N3 requires labelled references. The exact fit/recognize signatures are defined in `ore_n3` contracts/docs; do not fabricate training labels. A conceptual lifecycle is:

```python
from ore_n3 import N3Processor

processor = N3Processor()
processor.fit(reference_patches=labelled_training_patches)
result = processor.recognize(sequence)
```

Check the actual function signature before using this sketch; it communicates the separation between fitting and inference, not a guarantee that every version accepts exactly these keyword names.

### 16.3 N4 as a read-only attention stage

```python
from ore_n4 import N4Processor

# Construct the required N4Input using the fields in the checked-out
# ore_n4/contracts.py, then pass that object to process().
n4_result = N4Processor().process(n4_input)
```

This is an integration pattern rather than a complete constructor example. Use the actual `N4Input` signature in the checked-out version. Supply task relevance/seed attention only when genuinely available. N4 must not mutate the N3 result.

### 16.4 Module-registry adapter path

Register the contract model and manifest; then bind a backend and validate compatibility. A manifest by itself is discoverable metadata, not a runnable module.

```python
from ore_contracts import (
    ContractCatalog, ModuleRegistry, ModuleManifest,
    ModuleIdentity, InputPort, OutputPort,
)

catalog = ContractCatalog()
# Register exact contract IDs, versions and payload models before using them.
registry = ModuleRegistry(catalog)
manifest = ModuleManifest(
    identity=ModuleIdentity(
        module_id="ore.example.processor",
        module_version="1.0.0",
        module_name="Example processor",
        package_path="ore_example",
    ),
    responsibilities=("process a typed input",),
    input_ports=(InputPort("input", "ORE.EXAMPLE.INPUT", ("1.0.0",)),),
    output_ports=(OutputPort(
        "output", "ORE.EXAMPLE.OUTPUT", "1.0.0", "builtins.dict", "result"
    ),),
    capabilities=("example.processing",),
    execution_modes=("FUNCTIONAL",),
)
registry.register(manifest)
# Bind a validated backend separately; see docs/ORE_MODULE_COMMUNICATION_REGISTRY.md.
```

### 16.5 Shared workspace publication

```python
from ore_workspace import FindingPublisher, WorkspaceManager

with WorkspaceManager() as manager:
    workspace = manager.create_workspace("The boy opened the door.")
    publisher = FindingPublisher(workspace, "lexical_processor")
    result = publisher.publish(
        "lexical_hypotheses",
        {"candidate": "PERSON", "word": "boy"},
        key="boy",
    )
    print(result.decision, workspace.snapshot().revision)
```

The default permissions/subscriptions are defined in the workspace package. Do not bypass the state manager to mutate another module's data.

---

## 17. Summary for AI coding agents

Treat ORE as a **large, modular, evidence-aware cognitive research codebase**. The most important practical rules are:

1. Distinguish the legacy `ore_n*` pipeline from canonical `F-*`/`B-N*` architecture.
2. Read the target package's actual contracts, pipeline, docs and tests before coding.
3. Preserve typed interfaces, exact versions, provenance, uncertainty, immutable upstream artifacts and explicit statuses.
4. Keep functional computation, biological-reference simulation, task validation and biological support as separate claims.
5. Keep ordinary inference read-only; learning requires explicit authorization and guarded detached model updates.
6. Use `ore_contracts` for module interfaces/planning, `ore_workspace` for shared state, and `ore_logger` for observability—do not conflate them.
7. Do not fabricate missing modules, labels, data, adapter outputs, or evidence.
8. Run focused tests and report exact test outcomes; do not claim tests passed when merely inspecting them.
9. Update docs, registries, public exports and tests whenever a contract or operation changes.
10. Treat this document as a navigation/specification baseline for the inspected commit, then re-check `main` before every implementation task because the repository evolves.

