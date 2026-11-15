# ORE Neurocomputational — N1, N2, N3, and N4

A modular Python implementation of the ORE N1 early-visual-processing, N2 visual-organization, N3 recurrent orthographic-prediction, and N4 attention/precision-gating research pipelines. It exposes intermediate representations, configurable computational approximations, uncertainty, recurrent state, and per-stage diagnostics.

> Scientific scope: this is a biologically motivated computational approximation. It is not a cellular simulation, a fitted model of a particular species, or evidence that ORE reproduces the complete human visual system. Default parameters are engineering starting points and require calibration against empirical data before physiological claims are made.

## The 12 operations

| # | Operation | Main computation | Main output |
|---:|---|---|---|
| 1 | Photoreceptor adaptation | Gaussian local-illumination estimate and logarithmic response compression | photoreceptor response, adaptation field |
| 2 | Retinal ON/OFF processing | Local reference subtraction and positive/negative rectification | retinal ON and OFF |
| 3 | Center-surround receptive fields | Center-minus-surround Difference of Gaussians | ganglion ON and OFF |
| 4 | LGN contrast gain control | Local RMS estimate and divisive gain | LGN ON/OFF and gain map |
| 5 | V1 orientation-selective populations | Even/odd Gabor filter bank over configured orientations | V1 linear responses |
| 6 | Simple-cell-like responses | Phase-specific ON/OFF half-wave rectification | simple-cell response bank |
| 7 | Complex-cell-like quadrature energy | Square-root sum of even/odd phase energies | complex-cell energy |
| 8 | Population divisive normalization | Local pooled activity across orientation channels | normalized energy |
| 9 | Spatial pooling | Adaptive non-overlapping mean pooling | pooled orientation maps |
| 10 | Feature-vector construction | Orientation-major flattening of pooled maps | feature vector |
| 11 | Optional temporal neural dynamics | Leaky state integration with an exponential time constant | temporal activity |
| 12 | Diagnostics and numerical validation | Shapes, ranges, moments, timings, and finite-value checks | diagnostics report |

Operation 11 is applied between the normalized V1 population output (operation 8) and spatial readout (operation 9). When disabled, it is an identity operation and processing is frame-independent.

## Installation

Python 3.10+ is required.

    python -m venv .venv
    python -m pip install -e ".[dev]"

Activate the environment first. On Windows PowerShell use .venv\Scripts\Activate.ps1; on macOS/Linux use source .venv/bin/activate.

## Quick start

    import numpy as np
    from ore_n1 import N1Config, N1Processor

    image = np.random.default_rng(7).random((128, 128), dtype=np.float32)
    result = N1Processor(N1Config()).process(image)

    print(result.feature_vector.shape)
    print(result.metadata)
    print(result.diagnostics["valid"])
    retinal_map = result.stage_outputs["retinal_on"]

Inputs are two-dimensional grayscale images. Floating inputs must be finite and in [0, 1]. Integer grayscale inputs are scaled by their dtype maximum (uint8 is divided by 255). RGB conversion is deliberately outside the first N1 input contract so color handling cannot silently alter the grayscale signal.

Run the included synthetic experiment:

    python -m ore_n1
    python -m ore_n1 --output n1_demo.npz
    python -m pytest

## Mathematical sketch

Let I(x,y) be the normalized input image and G_sigma * I denote Gaussian smoothing.

1. Photoreceptor approximation: A = G_sigma_a * I; P = log(1 + I / (A + epsilon_a)).
2. ON/OFF pathways: C = P - G_sigma_b * P; ON = max(C, 0); OFF = max(-C, 0).
3. Center-surround: for either pathway X, R = max(G_sigma_c * X - G_sigma_s * X, 0), with sigma_s greater than sigma_c.
4. LGN gain: E = ON² + OFF²; g = 1 / (sqrt(G_sigma_l * E + epsilon) + beta)^gamma; multiply each pathway by g.
5. Orientation population: convolve the combined afferent drive with even and odd Gabor kernels at each orientation theta.
6. Simple-cell approximation: preserve positive and negative half-wave responses for each orientation and phase.
7. Complex-cell approximation: Q_theta = sqrt(E_even,theta² + E_odd,theta² + epsilon). Quadrature energy reduces sensitivity to carrier phase.
8. Divisive normalization: N_theta = Q_theta / (epsilon_n^p + G_sigma_n * sum_phi(Q_phi^p))^(1/p).
9. Spatial pooling: average each normalized orientation map over each cell of a fixed grid-height by grid-width partition.
10. Feature vector: concatenate pooled values in orientation-major, row-major order.
11. Temporal dynamics: state <- state + (1 - exp(-dt/tau)) * (input - state).
12. Validation: ensure each stage is finite and non-empty; record shape, dtype, minimum, maximum, mean, standard deviation, and elapsed time.

The equations are explicit so each approximation can be replaced, measured, or tested independently. Gaussian boundary handling uses reflection. Gabor kernels are zero-mean and unit-L2 normalized.

## Output contract

N1Result contains:

- feature_vector: one-dimensional float32 vector with length orientation_count × grid_height × grid_width.
- pooled_features: array shaped [orientation, grid_y, grid_x].
- stage_outputs: dictionary retaining intermediate arrays for analysis and debugging.
- metadata: input shape, orientation angles, pooling grid, feature dimension, temporal setting, and model-scope declaration.
- diagnostics: JSON-serializable dictionary with validity, stage count, warnings, and numerical records.

The numerical stage functions are mostly pure functions in ore_n1/stages.py. N1Processor owns orchestration and optional temporal state. Call reset_temporal_state() between unrelated video sequences, or pass reset_temporal=True for the first frame of a sequence.

## Validation and limitations

The software tests check input contracts, constant-field behavior, feature dimensions, finite outputs, phase-tolerant quadrature energy at a population scale, temporal-state behavior, and configuration validation. Synthetic tests are software sanity checks, not physiological validation.

A research-grade validation phase should compare responses to controlled gratings, edges, contrast series, natural-image corpora, and published measurements. Parameter fitting should use training data, with final performance assessed on held-out data and independent replication.

## Repository layout

- ore_n1/config.py — validated configuration
- ore_n1/filters.py — even/odd Gabor filter bank
- ore_n1/stages.py — numerical functions for operations 1–10
- ore_n1/temporal.py — operation 11 state integrator
- ore_n1/diagnostics.py — operation 12 recording and validation
- ore_n1/pipeline.py — end-to-end orchestration and result contract
- ore_n1/synthetic.py — reproducible visual stimuli
- ore_n1/__main__.py — runnable demonstration
- tests/ — automated software tests
- examples/ — example integration script

## License

MIT. See LICENSE.


## N2 biological visual processing (17 operations)

The repository now includes an independent `ore_n2` package implementing the
17-step N2 visual-organization pipeline. N2 consumes the preserved
`normalized_energy` spatial maps from N1; it does not reconstruct maps from the
flattened N1 feature vector and does not modify N1. See
[`N2_ARCHITECTURE.md`](N2_ARCHITECTURE.md) for the ordered algorithm, equations,
interfaces, and scientific limitations.

Quick start:

```python
from ore_n1 import N1Processor
from ore_n2 import N2Processor

n1_result = N1Processor().process(grayscale_image)
n2_result = N2Processor().process(n1_result)
print(n2_result.structured_output)
print(n2_result.diagnostics["valid"])
```

Install with `python -m pip install -e ".[dev]"` and run the complete suite
with `python -m pytest`. N2 outputs remain visual-structural representations;
grapheme identity and lexical semantics are deliberately outside this phase.


## N3 recurrent orthographic prediction and error (29 operations)

The repository now includes an independent `ore_n3` package. It learns labelled
N2 patch descriptors and smoothed grapheme unigram/bigram transitions, then
combines visual candidate beliefs with left- and right-neighbor predictions
using precision-weighted, damped inference. It preserves alternatives,
classwise prediction-error terms, feature-space residuals, and convergence
diagnostics. N3 expects one already-segmented N2 result per patch; it does not
perform segmentation or semantic correction.

See [docs/N3_ARCHITECTURE.md](docs/N3_ARCHITECTURE.md) for the exact 29
operations, equations, contracts, API, and scientific limitations.

Quick start:

```python
from ore_n3 import N3Config, LabeledN2Patch, N3Patch, N3Sequence, N3Processor

n3 = N3Processor(N3Config()).fit(
    labelled_reference_patches,
    grapheme_sequences=[("a", "b", "a"), ("b", "a", "b")],
)
result = n3.process(N3Sequence("sample", [
    N3Patch("p1", n2_result=n2_patch_1, segmentation_confidence=0.9),
    N3Patch("p2", n2_result=n2_patch_2, segmentation_confidence=0.8),
]))
print(result.candidates)
```

Install and run all tests with `python -m pip install -e ".[dev]"` followed by
`python -m pytest`. N3 is an algorithmic research approximation, not a
validated neural circuit model.


## N4 attention and precision gating (40 operations)

The repository includes an independent **ore_n4** package that consumes an N3 result without modifying it. N4 derives bottom-up salience from visual uncertainty, N3 classwise prediction error, and visual/prior disagreement; optionally integrates task and class goals; allocates competitive attention across sequence positions; computes class-selective attention gates; and modulates sensory/prior precision during bounded belief updates.

The stage preserves N3 visual probabilities and contextual priors as separate channels. Its output contains the attention allocation, bottom-up and top-down signals, class gates, precision vectors, attended posterior, ranked alternatives, diagnostics, and a trace of all 40 operations.

See [docs/N4_ARCHITECTURE.md](docs/N4_ARCHITECTURE.md) for the operation-by-operation algorithm, equations, API, tests, and scientific limitations.

Quick start:

    from ore_n4 import N4Config, N4Input, N4Processor

    result = N4Processor(N4Config()).process(
        N4Input(n3_result, task_relevance=[0.0, 1.0, 0.0])
    )
    print(result.candidates)
    print(result.diagnostics["valid"])

Run focused N4 tests with:

    python -m pytest tests/test_n4.py -q

Run the full repository suite with:

    python -m pip install -e ".[dev]"
    python -m pytest

N4 is an algorithmic research approximation. Its attention and precision parameters are engineering hypotheses, not measured neural constants; software tests do not establish biological validity.


## N9 integrated neurocomputational ORE (120 operations)

The additive \`ore_n9\` package coordinates the available N1-N4 processors and
provides explicit adapter contracts for the later lexical, contextual and
semantic components. It records an inspectable 120-operation trace, preserves
visual probabilities separately from top-down posterior beliefs, bounds
recurrent feedback, and abstains when evidence is insufficient.

N9 expects already-localized patches in logical reading order. It does not
silently invent a character segmenter. Each patch may contain a grayscale
image, an N2 result, a feature vector, or caller-supplied grapheme probabilities.

\`\`\`python
from ore_n9 import N9Patch, N9Processor, N9Request

request = N9Request(
    sequence_id="sample-001",
    patches=[
        N9Patch("g0", grapheme_probabilities={"a": 0.9, "o": 0.1}),
        N9Patch("g1", grapheme_probabilities={"b": 0.8, "d": 0.2}),
    ],
    word_spans=[(0, 2)],
    script_profile={"script": "Latin", "reading_direction": "ltr"},
)
result = N9Processor().process(request)
print(result.decision)
print(len(result.operation_trace))  # 120
\`\`\`

For image patches, provide each already-localized grayscale crop as
\`N9Patch("id", image=crop)\`; N9 runs the repository's N1 and N2 pipelines.
To use learned grapheme recognition, pass a fitted \`N3Processor\` through
\`N9Adapters(n3=...)\`; N9 then applies N4 attention/precision gating. N5-N8
are optional, explicitly injected adapters and are marked unavailable when
not configured. N6 resolves lexical hypotheses, the syntax adapter parses
provided grammatical evidence, N7 supplies discourse/context predictions,
and N8 composes typed semantic graphs and propositions.

\`\`\`python
from ore_n9 import N9Adapters, N9Processor

processor = N9Processor(adapters=N9Adapters(
    n3=fitted_n3_processor,
    n6=lexical_adapter,
    syntax=syntax_adapter,
    n7=context_adapter,
    n8=semantic_adapter,
))
\`\`\`

Learning is disabled unless \`learning_authorized=True\`. An N5 adapter is
deep-copied before a child-model proposal; the active parent model is checked
for mutation. Software tests do not establish real-corpus accuracy or
biological validity. See [docs/N9_ARCHITECTURE.md](docs/N9_ARCHITECTURE.md).


## N5 — 52-operation experience-dependent plasticity

N5 is a distinct, explicitly authorized learning stage after N4. It requires independently verified labels and provenance, preserves the parent N3-derived model, and returns a separately fingerprinted child model. Its 52 traced operations include fast/slow Oja-style prototype adaptation, eligibility traces, replay, homeostatic gain control, bounded drift, and fast/slow transition memories. N4 attention and precision signals can modulate learning. Recognition remains read-only; learning is a separate operation.

See [docs/N5_ARCHITECTURE.md](docs/N5_ARCHITECTURE.md) and the `ore_n5` package.

## N6 — 60-operation hierarchical orthographic access

N6 consumes N4 posteriors and an immutable N5 model snapshot. It combines fast/slow forward and reverse transition memories, performs bidirectional sequence inference, estimates word boundaries, and searches a caller-supplied provenance-tracked orthographic lexicon. It preserves ranked alternatives and uses explicit fallback tokens when lexical entries are unavailable. N6 does not modify N4 or N5 and does not infer word meaning.

See [docs/N6_ARCHITECTURE.md](docs/N6_ARCHITECTURE.md) and the `ore_n6` package.


## Canonical ORE phase registry and shared contracts

The canonical architecture now distinguishes the functional stages `F-A`–`F-J`
from the biological phases `B-N1`–`B-N9`. The new framework-neutral
`ore_contracts` package implements the P0 registry freeze and P1 interfaces
`IF-00`–`IF-04` (provenance envelope, script profile, run context, visual
observation and dataset manifest).

The older `ore_n1`–`ore_n9` packages are retained unchanged for compatibility.
Their historical N labels are not aliases for the canonical biological phase
IDs. In particular, canonical `B-N8` is independent brain-data validation and
canonical `B-N9` is system integration/governance.

See [the canonical registry and P1 contract notes](docs/CANONICAL_PHASE_REGISTRY_AND_P1_CONTRACTS.md).
Run the P1 tests with `python -m pytest tests/test_canonical_contracts.py -q`.


## Canonical ORE shared interfaces — P2 through P8

The canonical registry is now extended with shared interfaces **IF-05–IF-18**, a rate-based IF-18 neural simulation reference, guarded learning transactions, preregistered multi-seed experiment/evaluation contracts, adapters for legacy module outputs, and the dependency-injected B-N9 controller.

See [the P2–P8 implementation notes](docs/CANONICAL_ORE_P2_P8_IMPLEMENTATION.md) and run the complete suite with `python -m pytest`.

Important scope boundary: the existing `ore_n1`–`ore_n9` modules remain legacy-compatible packages, not aliases for canonical `B-N1`–`B-N9`. The controller reports missing canonical biological backends as unavailable. The included B-N1 rate simulator is a computational reference, not a validated cellular model; canonical B-N2–B-N8 biological implementations and independent brain-data validation remain future scientific work.

## Integrated Reading Algorithm — 72-operation registry

The additive `ore_reading` package implements a 72-operation integrated reading pipeline. It connects orthographic lexicon access and interactive activation with dual-route reading, grapheme-to-phoneme conversion, phonological assembly, TRACE-style word-form competition, Cohort narrowing, Shortlist-B-like Bayesian selection, and a connectionist spelling-sound-meaning model.

It keeps the functional algorithm separate from a biological reference branch using current-based leaky integrate-and-fire populations. The biological branch exposes membrane potential (mV), synaptic current (pA), spike counts, firing rates (Hz), and simulated time (ms); it is not a fitted or validated reproduction of human reading circuitry.

See [the 72-operation registry and mathematical specification](docs/INTEGRATED_READING_OPERATION_REGISTRY.md).

Quick start:

```python
from ore_reading import IntegratedReadingProcessor, ReadingLexiconEntry, ReadingRequest

lexicon = [
    ReadingLexiconEntry(
        "word-read", "read", ("R", "IY", "D"), 40.0,
        {"READING": 1.0, "KNOWLEDGE": 0.7}, "verb",
        verified_by="reviewer", source_reference="dataset://read",
    ),
]
result = IntegratedReadingProcessor(lexicon).process(
    ReadingRequest("sample-001", text="read")
)
print(result.tokens[0]["pronunciation"])
print(len(result.operation_trace))  # 72
print(result.biological["layers"]["orthographic"]["mean_firing_rate_hz"])
```

For an upstream N6 result, use `reading_request_from_n6_result(n6_result)`; the adapter preserves N6 token order and does not modify N6 output. Input is text or already-tokenized orthography: image OCR and character segmentation remain upstream responsibilities. The default G2P rules are an English-oriented demonstration; language-specific rules and lexicons must be supplied for other writing systems.

Run focused tests with `python -m pytest tests/test_integrated_reading.py -q` and the complete repository suite with `python -m pytest`.


## Semantic Cognition — 44-operation registry

The additive `ore_semantic` package implements a 44-operation semantic access and concept-selection subsystem. It adds episodic word-form retrieval, modality-specific semantic spokes, distributed feature activation, an ATL-inspired transmodal hub, IFG-inspired semantic control, and read-only abstract lexical prototypes.

Functional evidence channels (lexical similarity, frequency, episodic retrieval, modality fit, context fit, hub fit and prototype fit) remain separately inspectable. Selection uses explicit confidence and ambiguity gates and preserves alternatives when it abstains.

A second, distinct execution track runs current-based leaky integrate-and-fire population references for lexical, episodic, modality-spoke, distributed-feature, ATL-hub, IFG-control and abstract-prototype populations. It reports voltage (mV), current (pA), firing rate (Hz) and time (ms). This is a computational reference, not a fitted or validated reproduction of the human semantic system.

See [the 44-operation semantic cognition registry](docs/SEMANTIC_COGNITION_44_OPERATION_REGISTRY.md).

Quick start:

```python
from ore_semantic import SemanticConcept, SemanticCognitionProcessor, SemanticRequest

concepts = [
    SemanticConcept(
        concept_id="concept-apple",
        label="apple",
        aliases=("fruit",),
        semantic_features={"FOOD": 1.0, "PLANT": 0.8},
        lexical_category="noun",
        frequency=30.0,
        verified_by="reviewer",
        source_reference="dataset://apple",
    ),
]
result = SemanticCognitionProcessor(concepts).process(
    SemanticRequest("sample-001", text="apple", context={"FOOD": 0.8})
)
print(result.tokens[0]["selected_concept"])
print(len(result.operation_trace))  # 44
```

To connect semantic concept access to the existing N9 semantic adapter slot, inject `SemanticN8Adapter(SemanticCognitionProcessor(concepts))` through `N9Adapters(n8=...)`. The adapter returns concept nodes only; it deliberately does not fabricate syntax, propositions or semantic relation edges.

Run focused checks with `python -m pytest tests/test_semantic_cognition.py -q` and all repository tests with `python -m pytest`.


## Lexical Context and Predictive Processing — 40-operation registry

The additive `ore_lexical_context` package implements a 40-step lexical access, morphology, pronunciation mapping, orthographic competition, semantic ambiguity and predictive-processing pipeline. It follows the stages from accumulated word exposure and logarithmic frequency through lexical activation and settling, to sentence context, upcoming-word prediction, surprisal, semantic residuals and precision-weighted updating.

Every operation emits an inspectable trace row with a stable `LC-001`–`LC-040` ID and a mathematical form. The package also provides typed lexical/meaning/morpheme/G2P contracts, an adapter from `ore_reading` outputs, a runnable demonstration and focused tests.

A separate current-based LIF reference advances orthographic, lexical, morphological, phonological, semantic and context-predictive populations for each operation. The model reports voltage (mV), current (pA), firing rate (Hz) and time (ms). It is a generic, configurable biological hypothesis—not a fitted or empirically validated simulation of the human language system. Morphology and G2P candidates must be supplied through the inventories; no universal rules are fabricated.

See [the 40-operation lexical context registry](docs/LEXICAL_CONTEXT_40_OPERATION_REGISTRY.md).

Quick start:

```python
from ore_lexical_context import LexicalContextProcessor, LexicalContextRequest, LexicalEntry, LexicalMeaning

lexicon = [LexicalEntry("bank", "bank", frequency=25, meanings=(
    LexicalMeaning("river-bank", {"RIVER": 1.0, "LAND": 0.7}),
    LexicalMeaning("financial-bank", {"MONEY": 1.0, "INSTITUTION": 0.7}),
))]
result = LexicalContextProcessor(lexicon).process(
    LexicalContextRequest("sample-001", text="bank", context_features={"RIVER": 0.9})
)
print(result.tokens[0]["selected_meaning"])
print(len(result.operation_trace))  # 40 operations for one token
```

Run the focused checks with `python -m pytest tests/test_lexical_context.py -q`; run the full repository suite with `python -m pytest`.

## Master Predictive Orthography — seven stages / 40 operations

The additive `ore_predictive_orthography` package composes N3's bottom-up grapheme evidence and bidirectional transition predictions with N4's competitive attention, sensory/prior precision and attended posterior inference. It exposes the seven requested master stages as a 40-operation registry (`MPO-001`–`MPO-040`), plus explicit left/right neighbor messages, classwise prediction errors and both contextual-prior and attended-posterior feature-space residuals.

A parallel biological reference advances seven connected current-based leaky-integrate-and-fire populations for character evidence, transition memory, neighbor prediction, attention/sensory gain, prior precision, posterior settling and prediction-error diagnostics. It reports voltage, current, spikes, firing rates and time. This neural track is an explicit, configurable research hypothesis and does not alter the functional posterior.

See [the seven-stage master algorithm](docs/MASTER_PREDICTIVE_ORTHOGRAPHY_7_STAGE_ALGORITHM.md). Run focused tests with `python -m pytest tests/test_predictive_orthography.py -q` and the complete suite with `python -m pytest`.

The processor expects already-segmented patches in logical reading order and a fitted, labelled N3 model. It does not replace OCR segmentation, and its LIF model is not an anatomically fitted or empirically validated simulation of human reading.



## Lexical access and morphology — seven-stage architecture

The additive `ore_lexical_access` package implements seven stages and a 36-operation trace: stored word identity, word meaning, grammatical category, morphological decomposition, stem/affix retrieval, morphological alternative selection, and incremental lexical-candidate activation. It reuses the read-only lexical, meaning and morpheme records in `ore_lexical_context`; exact lexical inventories, meanings, roots and affixes must be supplied with provenance. Competing homographs and morphological parses are retained, and selection abstains when confidence or the top-two margin is insufficient.

A separate seven-population recurrent current-based leaky integrate-and-fire reference reports membrane voltage, current, spikes, firing rate and simulation time. It is a configurable computational hypothesis, not an anatomically fitted or empirically validated human-brain simulation.

See [the seven-stage lexical access architecture](docs/LEXICAL_ACCESS_7_STAGE_FINAL_ARCHITECTURE.md). Run the focused tests with `python -m pytest tests/test_lexical_access.py -q`; run the package demo with `python -m ore_lexical_access`.


## ORE Experiment Workbench

Run and inspect ORE components through a Streamlit chat-style laboratory. It captures component stage traces, package-scoped Python call timings, numerical summaries for input/output arrays, validation flags, warnings, and full JSON experiment reports. Successful Python outputs can be reused as upstream artifacts for chained experiments.

See [experiment_workbench/README.md](experiment_workbench/README.md) for installation, supported component workflows, structured inputs, and limitations.

Run from the repository root:

```bash
python -m pip install -e .
python -m pip install -r experiment_workbench/requirements.txt
streamlit run experiment_workbench/app.py
```


## ORE syntax parsing — nine stages, 56 operations

The new independent `ore_syntax` package implements part-of-speech identification,
phrase detection, dependency parsing, constituency parsing, hierarchical
syntax representation, subject/object assignment, agent/patient thematic roles,
syntactic ambiguity resolution, and long-distance dependency processing.

It exposes separate functional and biological-reference tracks, a numbered
56-operation registry, detailed per-operation traces, ranked parse alternatives,
argument/role evidence, enhanced filler-gap links, and JSON-safe diagnostics.
It can consume token-indexed lexical evidence from upstream ORE lexical access
and semantic features from the caller without mutating those inventories.

See [docs/SYNTAX_PARSING_ARCHITECTURE.md](docs/SYNTAX_PARSING_ARCHITECTURE.md)
for the full ordered algorithm, equations, API and scientific limitations.

Quick start:

```python
from ore_syntax import SyntaxConfig, SyntaxProcessor, SyntaxRequest

result = SyntaxProcessor(config=SyntaxConfig()).process(
    SyntaxRequest("sample-001", text="The cat eats fish.", language="en")
)
print(result.selected_pos)
print(result.dependency_arcs)
print(result.operation_trace)  # 56 operations
```

Run `python -m ore_syntax` for a demonstration and
`python -m pytest tests/test_syntax.py -q` for focused tests. The parser is a
compact English-oriented research prototype, not a treebank-trained parser.
Its scores are not calibrated probabilities, and its LIF population model is
a computational analogy rather than a validated reproduction of neural syntax.


## ORE Execution Logger — SQLite annotations and run history

The repository includes an automatic, repository-wide execution logger and a Streamlit observatory. The logger scans all Python source files and indexes module, class, method, nested-function, function and asynchronous-function definitions in a SQLite annotation catalogue.

- Existing Python docstrings are used as the initial explanation where available.
- Definitions without docstrings receive a source-based English description generated from their name, signature, return annotation, helper calls and control-flow structure. These are marked as source analysis, not as manually verified documentation.
- Each execution receives a unique run ID and local/UTC date-time labels.
- Function calls, returns, class entry, exceptions, source locations and durations are recorded as event rows.
- Events store a foreign-key reference to an annotation. The explanation is stored once in the annotations table rather than copied into each event.
- The viewer supports searching and manually editing annotations. Manual edits are preserved across catalogue refreshes; a source change marks the annotation for review.
- Input argument values, raw prompts, image pixels and full numerical arrays are not recorded.

Install the repository and the optional Streamlit viewer dependency:

    python -m pip install -e .
    python -m pip install -r experiment_workbench/requirements.txt

Run the viewer:

    streamlit run ore_logger/viewer.py

The default database is created automatically at `ore_logs/ore_logger.sqlite3` the first time a repository script is run. The directory is ignored by Git. Use `ORE_LOGGER_DB` to choose another database location, `ORE_LOG_MAX_EVENTS` to change the per-script event cap (default 200,000), or `ORE_AUTO_LOG=0` to disable automatic tracing for a process.

The existing Experiment Workbench also creates a separate timestamped SQLite run for each component execution. See [the Workbench guide](experiment_workbench/README.md).


## Sentence-level semantic composition — nine stages / 54 operations

The additive ore_semantic_composition package continues after individual word-concept access. It composes word meanings, predicate-argument structures, thematic roles, propositions, negation/modality, quantifier scope, causal relations, and a typed sentence-level semantic graph. It defines 54 inspectable operations (SMC-001 through SMC-054) across the nine requested stages.

The functional track accepts the existing ore_semantic.SemanticConcept inventory and optional ore_syntax.SyntaxResult, preserving alternatives and abstaining when confidence or margin is insufficient. Optional caller-supplied PredicateFrame records make valency and selectional preferences explicit. A separate nine-population current-based LIF reference reports voltage, current, firing rate, spikes, and simulated time; it is a computational hypothesis, not a validated reproduction of human semantic processing.

See docs/SEMANTIC_COMPOSITION_9_STAGE_ARCHITECTURE.md for the full stage sequence, operation registry, mathematical models, biological-reference track, API, and limitations.

Run python -m ore_semantic_composition for the bundled demonstration and python -m pytest tests/test_semantic_composition.py -q for focused tests. The existing ORE SQLite logger indexes every new class/function docstring into its annotation catalogue when an ORE script starts; its local database remains ore_logs/ore_logger.sqlite3.


## ORE discourse integration — eight stages / 39 operations

The additive `ore_discourse` package implements the eight discourse stages:
maintaining discourse context; integrating consecutive sentence propositions;
inferring speaker intention; interpreting indirect requests; evaluating irony;
constructing situation models; tracking temporal discourse relations; and
maintaining topic/narrative coherence. It exposes a stable `DC-001`–`DC-039`
operation registry, mathematical sketches, stage outputs and a complete trace.

The functional track keeps caller-provided entities, common ground, beliefs,
temporal metadata and prosody separate from surface-based hypotheses. It uses
bounded heuristics for English examples, preserves alternative interpretations,
and does not invent unsupported world facts. The biological-reference track
advances coupled current-based LIF populations for discourse memory,
proposition binding, speaker modeling, pragmatic control, incongruity
monitoring, situation modeling, temporal order and narrative integration.

The LIF reference reports voltage (mV), current (pA), firing rate (Hz), spikes
and simulated time (ms). Its population assignments and parameters are
computational hypotheses, not a validated reproduction of human language
circuitry. Passing software tests does not establish behavioral or biological
validity.

Quick start:

```python
from ore_discourse import DiscourseProcessor, DiscourseRequest

request = DiscourseRequest(
    sequence_id="dialogue-001",
    sentences=[
        "The deployment failed yesterday.",
        "Great work on the release!",
        "Could you inspect the service logs?",
        "Then the engineer restarted the service.",
    ],
    context={"prior_turns": ["The team reported a failed deployment."]},
    prosody={1: {"intonation_mismatch": 0.7}},
)
result = DiscourseProcessor().process(request)
print(result.coherence["score"])
print(result.indirect_requests)
print(result.irony)
print(len(result.operation_trace))  # 39
```

Use the new ORE N7 adapter inside the existing N9 controller:

```python
from ore_discourse import DiscourseN7Adapter
from ore_n9 import N9Adapters, N9Processor

adapters = N9Adapters(n7=DiscourseN7Adapter())
n9 = N9Processor(adapters=adapters)
result = n9.process(request_for_n9)
```

Run the demonstration and tests:

```bash
python -m ore_discourse
python -m pytest tests/test_discourse.py -q
python -m pytest
```

See [docs/DISCOURSE_EIGHT_STAGE_ARCHITECTURE.md](docs/DISCOURSE_EIGHT_STAGE_ARCHITECTURE.md)
for all 39 registered operations, equations, interfaces and scientific limitations.


## ORE working memory and cognitive control — five stages / 40 operations

The additive ore_working_memory package implements the exact **40-operation WMCC-001–WMCC-040 registry** from the uploaded Brain Language Algorithm.md. Its five stages have the original distribution: **8 / 6 / 7 / 9 / 10 operations** for working-memory buffers, incomplete syntax, executive control, semantic interpretation resolution, and task switching.

Every operation has a functional implementation, a mathematical specification, a qualified biological-reference note, a stable trace row, and a corresponding optional LIF population update. The functional path preserves chunk and buffer references, unfinished grammatical dependencies, separate evidence channels, alternative semantic readings, explicit revision evidence, task checkpoints/resumption, and completion status. Supplied evidence is kept distinct from neutral defaults; the prototype does not invent missing syntax or world knowledge.

The optional current-based leaky integrate-and-fire reference advances eight coupled working-memory, lexical, syntactic, semantic, attention, prefrontal-control, conflict-monitoring, and task-set populations once per operation. It reports voltage, current, firing rate, spikes, and simulated time. This is a computational hypothesis, not a validated anatomical or physiological model.

Quick start:

    from ore_working_memory import WorkingMemoryProcessor, WorkingMemoryRequest

    request = WorkingMemoryRequest(
        sequence_id="reading-001",
        text="I saw the man with the telescope. He was holding it.",
        interpretations=[
            {"hypothesis_id": "instrument", "interpretation": "I used the telescope", "semantic_score": 0.7},
            {"hypothesis_id": "possession", "interpretation": "The man had the telescope", "semantic_score": 0.72},
        ],
        task_sequence=["reading", "reference_retrieval", "semantic_resolution"],
    )
    result = WorkingMemoryProcessor().process(request)
    print(result.semantic_resolution)
    print(result.task_switching)
    print(len(result.operation_trace))  # 40

Run python -m ore_working_memory for the demonstration, python -m pytest tests/test_working_memory.py -q for focused tests, and python -m pytest for the complete suite. See docs/WORKING_MEMORY_COGNITIVE_CONTROL_ARCHITECTURE.md for the complete 40-operation registry, equations, biological-reference descriptions, API details, logger integration, and limitations.

The ORE logger indexes every new class/function docstring in its SQLite annotation catalogue when an ORE Python entry point is launched with automatic logging enabled; the default local database is ore_logs/ore_logger.sqlite3.