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

The additive `ore_n9` package coordinates the available N1-N4 processors and
provides explicit adapter contracts for the later lexical, contextual and
semantic components. It records an inspectable 120-operation trace, preserves
visual probabilities separately from top-down posterior beliefs, bounds
recurrent feedback, and abstains when evidence is insufficient.

N9 expects already-localized patches in logical reading order. It does not
silently invent a character segmenter. Each patch may contain a grayscale
image, an N2 result, a feature vector, or caller-supplied grapheme probabilities.

```python
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
```

For image patches, provide each already-localized grayscale crop as
`N9Patch("id", image=crop)`; N9 runs the repository's N1 and N2 pipelines.
To use learned grapheme recognition, pass a fitted `N3Processor` through
`N9Adapters(n3=...)`; N9 then applies N4 attention/precision gating. N5-N8
are optional, explicitly injected adapters and are marked unavailable when
not configured. N6 resolves lexical hypotheses, the syntax adapter parses
provided grammatical evidence, N7 supplies discourse/context predictions,
and N8 composes typed semantic graphs and propositions.

```python
from ore_n9 import N9Adapters, N9Processor

processor = N9Processor(adapters=N9Adapters(
    n3=fitted_n3_processor,
    n6=lexical_adapter,
    syntax=syntax_adapter,
    n7=context_adapter,
    n8=semantic_adapter,
))
```


### Ready-made language adapter stack

The `ore_n9.adapters` module connects the existing 36-operation lexical
access processor, 58-operation syntax processor, 54-operation sentence semantic
composition processor, and the existing 39-operation N7 discourse adapter.
The sentence semantic adapter is distinct from the concept-only
`SemanticN8Adapter`: it returns predicate/argument propositions and a typed
sentence graph. All processors still require caller-supplied lexical, concept,
and predicate-frame inventories appropriate to the target language/domain.

```python
from ore_n9 import N9Adapters, N9Processor, build_n9_language_adapters

adapters = build_n9_language_adapters(
    lexical_access=lexical_access_processor,
    syntax=syntax_processor,
    semantic_composition=semantic_composition_processor,
    discourse=discourse_processor,
    base_adapters=N9Adapters(n3=fitted_n3_processor),
)
result = N9Processor(adapters=adapters).process(request)
print(result.propositions)
print(result.context_model)
```

The factory preserves supplied N1-N5 adapters, creates N6/N7/N8 and syntax
bridges, and does not create a lexicon or invent world knowledge. The focused
end-to-end test compares "musician tunes instrument" with "musician tunes
microwave" using an explicit predicate selectional-preference frame.

Learning is disabled unless `learning_authorized=True`. An N5 adapter is
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

## Master Biological Neurocomputational ORE — BNBEA

The additive ore_bnbea package implements the nine-stage biological/functional master sequence. It composes the existing N1–N9 functional pipeline with a separate conductance-based LIF network, recurrent excitatory/inhibitory projections, bounded Hebbian/STDP-style plasticity, two-timescale associative memory, Bayesian contextual prediction, and optional EEG/MEG/fMRI/behavioral validation.

The detailed per-stage lists produce **139 master operations** (11/12/12/14/13/16/18/19/24). Each operation has a stable ID and a SQLite annotation. N9's existing 120-operation trace is preserved separately.

Quick start:

    from ore_bnbea import BNBEAConfig, BNBEAProcessor, BNBEARequest
    from ore_n9 import N9Patch, N9Request

    request = N9Request(
        sequence_id="reading-001",
        patches=[
            N9Patch("p0", grapheme_probabilities={"a": 0.9, "o": 0.1}),
            N9Patch("p1", grapheme_probabilities={"b": 0.8, "d": 0.2}),
        ],
        script_profile={"script": "Latin"},
    )
    result = BNBEAProcessor(
        config=BNBEAConfig(simulation_steps=24, neurons_per_population=12)
    ).process(BNBEARequest(n9_request=request))
    print(len(result.operation_trace))  # 139

The default logger database remains ore_logs/ore_logger.sqlite3. BNBEA indexes every repository Python class/function/method docstring and writes all detailed stage-operation annotations and execution events to the same SQLite database.

Install and test with:

    python -m pip install -e ".[dev]"
    python -m pytest tests/test_bnbea.py -q
    python -m pytest

See docs/BNBEA_IMPLEMENTATION.md for the equations, stage boundaries, operation registry, API, logger contract, and scientific limitations. Software tests validate code behavior; they do not establish biological equivalence. Stage 8 remains not_tested until real independent brain/behavior datasets are supplied.
## Brain Networks and Biological Evidence Algorithm — 12 operations

The additive `ore_brain_networks` package implements the 12-operation visual, orthographic, language-network, auditory, neural-measurement and causal-validation sequence. It contains explicit Gabor filterbanks, grapheme population prototypes, recurrent LIF dynamics, hierarchical visual populations, a VWFA hierarchy, an IFG/MTG/ATL network, auditory dorsal/ventral representations, EEG/MEG and fMRI forward models, virtual lesion and TMS analyses, and independent human-reading validation.

Functional representations, simulated biological state, and experimental predictions are kept in separate output tracks. The LIF reference reports membrane voltage, current, firing rate, spikes and simulation time. Synthetic EEG/MEG projections are marked as toy lead fields; BOLD is an HRF-convolved proxy. Lesions and TMS are model-only interventions, and human validation waits for independent measurements rather than inventing empirical results.

See [docs/BNBEA_12_OPERATION_ARCHITECTURE.md](docs/BNBEA_12_OPERATION_ARCHITECTURE.md) for the ordered BNBEA-001–BNBEA-012 registry, mathematical operations, biological qualifications, API and scientific limitations.

Quick start:

```python
import numpy as np
from ore_brain_networks import BNBEAProcessor, BNBEARequest, LexicalEntry

request = BNBEARequest(
    sample_id="reading-001",
    visual_image=np.zeros((64, 96)),
    orthographic_sequence=("c", "a", "t"),
    lexicon=(LexicalEntry("cat", frequency=30.0),),
    semantic_features={"ANIMAL": 1.0},
)
result = BNBEAProcessor().process(request)
print(len(result.operation_trace))  # 12 operations
```

Run `python -m ore_brain_networks` for a deterministic demonstration and `python -m pytest tests/test_brain_networks.py -q` for focused tests. The existing ORE SQLite logger stores class and callable docstrings in `ore_logs/ore_logger.sqlite3` when the package initializes its annotation catalogue. Passing software tests does not establish biological validity.


## Integrated sensory and perceptual processing — six stages / 39 operations

The additive `ore_sensory_perception` package implements the six-stage algorithm from
`Integrated Brain Sensory and Perceptual Processing Algorithm.txt` as a 39-operation
visual-and-auditory pipeline. Stages 1–2 detect candidate text regions and segment
character-like visual structures; Stages 3–4 process sound and extract time-varying
acoustic features; Stage 5 estimates distributed phonetic-feature evidence; Stage 6
maintains, updates and selects language-specific phoneme hypotheses.

The visual and auditory modalities remain separate until an explicit shared-language
readout. The visual path produces glyph/letter hypotheses rather than claiming general OCR.
It requires caller-supplied grapheme prototypes and orthographic priors to assign labels.
The auditory path accepts a waveform and sample rate; a caller may supply a language- and
dialect-specific `PhonemeSpec` inventory, speaker baselines and phonetic context. An
English-oriented fallback inventory is provided only for demonstration. Uncertain decisions
are preserved and the decoder can abstain rather than force a phoneme label.

All 39 operations have stable `ISP-001`–`ISP-039` identifiers, per-operation mathematical
descriptions, biological analogues, and ordered functional traces. A separate configurable,
recurrent 16-population current-based LIF reference model reports membrane voltage, current,
spikes, firing rates and simulated time. This is a computational biological analogue, not
a literal brain simulation or measured EEG/MEG/fMRI/electrophysiology.

### Quick start

```python
import numpy as np
from ore_sensory_perception import SensoryPerceptionProcessor, SensoryRequest

image = np.ones((80, 240), dtype=float)
image[20:25, 15:220:8] = 0.0
time = np.arange(8000) / 16000.0
audio = 0.4 * np.sin(2 * np.pi * 180 * time)

request = SensoryRequest(
    sample_id="example-001",
    visual_image=image,
    auditory_waveform=audio,
    sample_rate_hz=16000,
)
result = SensoryPerceptionProcessor().process(request)
print(len(result.operation_trace))  # 39
print(len(result.biological_trace))  # 39 when simulation is enabled
```

The default annotation registration synchronizes Python module, class, function and method
docstrings into the repository-wide `ore_logger.annotations` table in
`ore_logs/ore_logger.sqlite3`. A custom path can be supplied for isolated tests, and
`ORE_LOGGER_DB` overrides the default database location. No sensory-specific database is
created. The execution logger records symbol metadata and stage summaries; it does not
persist raw images or audio arrays as event metadata.

Run `python -m ore_sensory_perception` for a synthetic smoke demonstration and
`python -m pytest tests/test_sensory_perception.py -q` for focused tests. See
`docs/SENSORY_PERCEPTUAL_6_STAGE_ARCHITECTURE.md` for the complete registry, operation
equations, biological qualifications, API, and validation boundaries.


## 21-stage Hub-and-Spoke Semantic Concept Integration

The additive `ore_hub_spoke` package implements the 21-stage semantic integration algorithm with separate functional and current-based LIF reference tracks. It includes seven modality spokes, reliability-weighted nonlinear hub integration, recurrent stabilization, feedback, cross-modal invariance, semantic control, explicit experience learning, and distinct hub/spoke/connection lesion experiments. It reuses `ore_semantic.SemanticConcept` and the existing SQLite logger.

See [the 21-stage architecture and API](docs/HUB_AND_SPOKE_21_STAGE_ARCHITECTURE.md). Run the focused suite with `python -m pytest tests/test_hub_spoke.py -q`, or launch the demonstration with `python -m ore_hub_spoke`.


## Controlled Semantic Cognition (CSC) — 18 stages

The additive `ore_csc/` package implements an 18-stage Controlled Semantic Cognition algorithm with a functional semantic-control pipeline and an optional current-based LIF population reference. It uses the existing provenance-tracked `SemanticConcept` contract from `ore_semantic` and keeps CSC separate from the existing 44-operation semantic-cognition and 21-stage hub-and-spoke packages.

The CSC stages cover sensory/linguistic input, parallel candidate activation, distributed semantic memory, ATL integration, context and task representation, control-demand estimation, semantic-control recruitment, targeted retrieval, feature amplification/inhibition, weak-knowledge retrieval, candidate evaluation, competition/selection, monitoring and feedback, semantic integration, response generation, and authorized adaptation. The registry enumerates 78 operations with mathematical equations, functional responsibilities, and biological-reference notes.

Run the demonstration from the repository root:

    python -m ore_csc

The demo selects between financial and river meanings of “bank”, indexes all CSC class/function docstrings in the shared `ore_logger` SQLite annotation catalogue, and records the operation/stage trace through `SQLiteRunRecorder`. The documentation is in `docs/CSC_18_STAGE_ARCHITECTURE.md`.

The functional and LIF tracks are transparent computational hypotheses, not a fitted or validated reproduction of human neural activity. Learning is preview-only unless the caller supplies explicit feedback and authorizes an update.

## Hybrid embodied–amodal concept formation — 19 stages

The additive `ore_hybrid_embodied_amodal` package implements the 19-stage hybrid concept-processing specification. It retains modality-specific embodied representations (visual, auditory, motor, somatosensory, affective, linguistic and related channels) while an ATL-inspired hub estimates cross-modal common information. It also preserves route-specific residuals, performs task/context abstraction, top-down spoke reactivation, predictive error correction, response selection, and provenance-gated learning.

Every stage has a stable `HEC-01.01`–`HEC-01.19` ID. Every registered sub-operation has its own stable ID, equation, functional responsibility, and biological-reference note. A separate optional current-based LIF reference reports voltage (mV), current (pA), firing rate (Hz), spikes, and simulated time (ms). These are transparent computational analogues, not a fitted or validated reproduction of human neural activity.

Quick start:

```python
from ore_hybrid_embodied_amodal import (
    HybridConcept, HybridEmbodiedAmodalProcessor, HybridRequest,
)

concept = HybridConcept(
    concept_id="guitar",
    label="guitar",
    category="musical instrument",
    features={"instrument": 1.0, "strings": 0.9, "music": 0.8},
    modality_features={
        "visual": {"strings": 0.9},
        "auditory": {"music": 0.8},
        "motor": {"strumming": 0.9},
    },
    verified_by="reviewer",
    source_reference="dataset://concepts/guitar",
)
processor = HybridEmbodiedAmodalProcessor([concept])
result = processor.process(HybridRequest(
    sequence_id="example-001",
    modality_inputs={
        "visual": {"strings": 0.9},
        "auditory": {"music": 0.8},
        "motor": {"strumming": 0.9},
    },
    task="recognition",
))
print(result.selected_concept)
print(len(result.stage_trace))       # 19
print(len(result.operation_trace))   # complete sub-operation trace
```

The package consumes named feature maps from upstream perception; raw image/audio transduction is outside its input contract. Learning is preview-only unless explicitly authorized with reviewer and source provenance.

The ORE SQLite logger stores module/class/function/method docstrings in `ore_logs/ore_logger.sqlite3`. Use `index_hybrid_annotations(".")` to refresh the symbol catalogue and `record_hybrid_result(result, ".")` to persist all operation explanations and the 19-stage run trace.

See [docs/HYBRID_EMBODIED_AMODAL_19_STAGE_ARCHITECTURE.md](docs/HYBRID_EMBODIED_AMODAL_19_STAGE_ARCHITECTURE.md) and run:

```bash
python -m ore_hybrid_embodied_amodal
python -m pytest tests/test_hybrid_embodied_amodal.py -q
python -m pytest
```


## Distributed Neural Semantic Composition and Integration — 36 stages

The additive `ore_distributed_semantics` package implements the 36-stage
algorithm described in the source document. It includes lexical and conceptual
activation, modality-specific feature retrieval, cross-modal association,
bilateral ATL-style semantic convergence, abstraction, associative graph
propagation, phrase composition, thematic roles, selectional compatibility,
contextual prediction, ambiguity control, AG/posterior-temporal integration,
prediction error, an explicitly labeled N400 difficulty proxy, reanalysis,
episodic and schema retrieval, discourse integration, semantic learning,
developmental/aging and multilingual adaptation, five impairment simulations,
and assessment.

The registry provides **36 stages and 108 mathematically described operations**
with stable identifiers (DS-01 through DS-36). Functional semantic processing
and a separate 14-population current-based LIF reference simulation produce
distinct traces. Every package class and function has a docstring for the
existing SQLite annotation catalogue.

Quick start:

    from ore_semantic.contracts import SemanticConcept
    from ore_distributed_semantics import (
        DistributedSemanticProcessor,
        DistributedSemanticRequest,
    )

    concept = SemanticConcept(
        concept_id="violin",
        label="violin",
        semantic_features={"ARTIFACT": 1.0, "MUSICAL_INSTRUMENT": 1.0},
        verified_by="example",
        source_reference="example://violin",
    )
    processor = DistributedSemanticProcessor([concept])
    result = processor.process(
        DistributedSemanticRequest(sequence_id="demo-001", text="violin")
    )
    print(len(result.stage_trace))      # 36
    print(len(result.operation_trace))  # 108

Run the demonstration with `python -m ore_distributed_semantics` and the
focused tests with `python -m pytest tests/test_distributed_semantics.py -q`.
The package reuses `ore_logger/ore_logs/ore_logger.sqlite3` and its standard
`annotations`, `runs`, and `events` tables; no separate database is created.
The demonstration indexes symbol docstrings and records the operation/stage
trace. See [the complete architecture, equations, biological references,
API, and validation boundaries](docs/DISTRIBUTED_NEURAL_SEMANTIC_36_STAGE_ARCHITECTURE.md).

Scientific qualification: vector, graph, probabilistic and LIF calculations
are explicit computational approximations, not claims about literal neuronal
arithmetic. The simulated LIF activity is not human neural recording; the N400
signal is an integration-difficulty proxy, not measured EEG. Learning updates
are preview-only unless `allow_learning=True` is explicitly set, and the
impairment stages are model interventions rather than clinical predictions.

## 21-stage Distributed-Only Semantic Representation

The additive ore_distributed_only package implements the distributed-only
alternative to the Hub-and-Spoke model. It represents concepts as separate
visual, auditory, action, tactile, affective, linguistic, contextual, and
spatial population patterns, with recurrent inter-population connectivity and
no central semantic storage vector. Its 21 ordered stages cover signal and
feature encoding, associative reactivation, action/affect/episodic retrieval,
recurrent excitation/inhibition, timing coordination, distributed cell
assemblies, pattern completion, category competition, context/task modulation,
learning, redundancy, and network-lesion analysis.

The package also runs a separate current-based leaky-integrate-and-fire
population reference at every stage. Each registered sub-operation includes
its mathematical expression and biological reference. The existing
ore_logger AST catalogue stores package module, class, and function
annotations, while operation and stage execution traces are written to the
shared ore_logs/ore_logger.sqlite3 database.

See [the distributed-only architecture and API](docs/DISTRIBUTED_ONLY_SEMANTIC_ARCHITECTURE.md).
Run focused tests with python -m pytest tests/test_distributed_only.py -q.




## Language comprehension — Point 1: Perceptual processing (12 steps)

The additive `ore_language_comprehension` package implements the first point from
`Language_Comprehension_Brain_Algorithms_Parts_01-15.md`: transform available auditory
and/or visual input into ordered, uncertainty-aware phonetic and orthographic form
hypotheses for downstream lexical access.

It exposes a source-level 12-step functional trace (`LC-P01-01`–`LC-P01-12`) and
reuses the existing `ore_sensory_perception` package for its 39 low-level visual and
auditory operations (`ISP-001`–`ISP-039`). When enabled, the 16-population LIF
reference simulation is retained as a separate biological trace. Functional outputs,
simulated voltage/current/firing-rate summaries, and actual human measurements are
never conflated.

Auditory phonetic segments and visual glyph candidates remain separate unless explicit
cross-modal alignment is provided by the caller. Label assignment requires the relevant
caller-supplied phoneme inventory or grapheme prototypes. Point 1 does not retrieve
lexical meaning, construct syntax, compose propositions, or infer speaker intent.

Quick start:

```python
import numpy as np
from ore_sensory_perception import PhonemeSpec, SensoryConfig, SensoryRequest
from ore_language_comprehension import Point1PerceptualProcessor

sample_rate = 16000.0
time = np.arange(4800) / sample_rate
request = SensoryRequest(
    sample_id="example-001",
    auditory_waveform=0.3 * np.sin(2 * np.pi * 180 * time),
    sample_rate_hz=sample_rate,
    phoneme_inventory=(
        PhonemeSpec("a", "vowel", voiced=True, vowel_height="open"),
        PhonemeSpec("m", "nasal", place="bilabial", voiced=True),
    ),
)
result = Point1PerceptualProcessor(
    config=SensoryConfig(simulate_biology=True)
).process(request)
print(len(result.functional_trace))  # 12
print(len(result.sensory_operation_trace))  # 39
```

Processor initialization registers the package and sensory dependency class/function
docstrings in the existing `ore_logger.annotations` SQLite catalogue
(default: `ore_logs/ore_logger.sqlite3`; override with `ORE_LOGGER_DB` or
`database_path`). It does not create a separate annotation database.

Run `python -m ore_language_comprehension` for the demonstration and
`python -m pytest tests/test_language_comprehension_point01.py -q` for focused tests.
See [the Point 1 architecture and mathematical/biological qualifications]
(docs/LANGUAGE_COMPREHENSION_POINT_01.md).


## Language comprehension — Point 2: Lexical access (12 steps)

The additive `ore_language_comprehension` package now implements the second source point:
retrieve word-level lexical entries, meanings, grammatical properties and possible senses from
the uncertain spoken/written forms produced by Point 1.

`Point2LexicalAccessProcessor` exposes **12 source-level steps**
(`LC-P02-01`–`LC-P02-12`) and reuses `ore_lexical_access` for its established
seven-stage, 36-operation lexical engine (`BLA-001`–`BLA-036)). It also aggregates that
engine's seven-population current-based LIF reference activity into a separate 12-step
biological trace. Functional outputs and simulated activity are kept distinct from human
neural measurements.

Visual glyph alternatives are beam-composed into orthographic form candidates. Auditory
phoneme alternatives are matched only to pronunciations explicitly supplied in
`LexicalEntry.phonemes`; the implementation does not invent pronunciations from spelling.
Both routes can converge on the same lexical entry ID. Word-level semantic features, morphology,
lexical category, stored grammatical properties, homonymy and polysemy alternatives are returned
without constructing sentence syntax or predicate-argument structure.

The processor accepts the Point 1 result directly:

```python
from ore_language_comprehension import Point2LexicalAccessProcessor, Point2Request
from ore_lexical_context import LexicalEntry, LexicalMeaning

lexicon = (
    LexicalEntry(
        "cat", "cat", frequency=24, phonemes=("k", "æ", "t"),
        meanings=(LexicalMeaning("feline", {"ANIMAL": 1.0, "FELINE": 1.0}),),
        lexical_category="noun", language="en",
        verified_by="your-lexicon-source",
        source_reference="dataset://your-lexicon/cat",
    ),
)
request = Point2Request.from_point1_result(
    "reading-001", point1_result, lexicon, language="en"
)
result = Point2LexicalAccessProcessor().process(request)
print(len(result.functional_trace))  # 12
print(result.lexical_candidates)
```

On initialization, Point 2 registers docstrings for the language-comprehension package,
`ore_lexical_access`, and `ore_lexical_context` in the existing
`ore_logger.annotations` SQLite table. No separate Point 2 annotation database is created.

Run `python -m pytest tests/test_language_comprehension_point02.py -q` for the focused
acceptance suite. See
[the Point 2 architecture, equations, biological reference and API](docs/LANGUAGE_COMPREHENSION_POINT_02.md).

## Language comprehension — Point 3: Syntactic parsing (12 steps)

The additive `ore_language_comprehension` package now adapts the third source point to ORE's
existing `ore_syntax` subsystem. `Point3SyntacticParser` runs the 12 source-level steps
(`LC-P03-01`–`LC-P03-12)) while reusing the syntax engine's nine stages and 58 lower-level
operations for POS candidates, phrase grouping, dependency relations, constituency alternatives,
hierarchy, ambiguity revision, and long-distance dependencies.

The adapter accepts one Point 2 lexical result per token through
`Point3Request.from_point2_results`, or accepts explicit lexical items with category alternatives,
morphology, inherited confidence, and optional boundary/prosodic cues. It adds explicit
morphosyntactic feature checks and graded agreement penalties when the required features were
supplied. Missing features remain unknown. A separate optional LIF reference track aggregates the
existing syntax-network operations into 12 source-level records.

Point 3 exposes only grammatical structure for the next semantic-composition stage. It does not
expose predicate-argument semantic interpretations, thematic/event roles, a sentence proposition,
or discourse/pragmatic meaning. The compact syntax grammar is English-oriented, and model scores
are not calibrated human probabilities.

On initialization, Point 3 registers all `ore_language_comprehension` and `ore_syntax` class and
function docstrings in the existing `ore_logger.annotations` SQLite table. It uses the same
`ore_logs/ore_logger.sqlite3` database; it does not create a new annotation database.

Run `python -m pytest tests/test_language_comprehension_point03.py -q` for the focused acceptance
suite. See [the Point 3 architecture, equations, biological track and API]
(docs/LANGUAGE_COMPREHENSION_POINT_03.md).

## Language comprehension — Point 4: Semantic composition (14 steps)

The additive `ore_language_comprehension` package now implements the fourth source point as a
14-step adapter over the existing `ore_semantic_composition` nine-stage / 54-operation engine.
`Point4SemanticComposer` consumes Point 3 syntactic dependencies and the Point 2 semantic sense
candidates embedded in the Point 3 lexical records. It creates sentence-level event propositions,
entity/event variables, predicate argument and thematic-role links, modifier/property relations,
negation/modality/quantifier scopes, semantic compatibility reports, competing readings, and a
validated proposition graph.

The adapter uses caller-supplied or explicitly embedded predicate frames where possible. It does
not equate grammatical subject with Agent or object with Patient as a universal rule. Missing
semantic features remain missing rather than being invented. If no explicit predicate frame exists,
the underlying semantic engine uses a documented generic fallback, so role conclusions should be
treated as provisional.

Point 4 recomputes each available word prefix through the existing composition engine and records
the evolving partial proposition, new/revised events, modifier links, and open argument slots. This
is an inspectable incremental approximation rather than a persistent neural-state simulation. The
existing nine-population LIF reference can be aggregated into 14 source-level biological records;
it is kept separate from the functional proposition output and is not measured human neural data.

Point 4 registers its class and function docstrings in the shared `ore_logger.annotations` SQLite
table, indexing `ore_language_comprehension`, `ore_semantic_composition`, `ore_semantic`, and
`ore_syntax`. It reuses `ore_logs/ore_logger.sqlite3`; no separate annotation database is created.

The public output intentionally stops at literal sentence meaning. Discourse history, physical
situation, world-knowledge inference, speaker intention, sarcasm and irony remain for Point 5.

Run `python -m pytest tests/test_language_comprehension_point04.py -q` for the focused acceptance
suite. See [the Point 4 architecture, mathematical operations, biological-reference mapping and API]
(docs/LANGUAGE_COMPREHENSION_POINT_04.md).


## Language comprehension — Point 5: Contextual and pragmatic integration (20 steps)

Point 5 implements the fifth source point as a 20-step adapter over ORE's established
ore_discourse eight-stage / 39-operation processor. Point5ContextualPragmaticIntegrator
accepts the literal proposition and unresolved alternatives from Point 4 together with
explicit discourse history, entities, common ground, speaker model, scene evidence, world
knowledge, prosody and conversation goals.

The adapter preserves literal meaning separately from discourse links, coreference,
situation/event structure, temporal relations, coherence, speaker-intention hypotheses,
indirect speech acts and irony hypotheses. It maps the 39 underlying operations onto 20
source-level functional and biological-reference records. The existing coupled nine-population
current-based LIF simulator is reused; it is a deterministic computational reference, not
measured or anatomically fitted human neural data.

The inference policy is conservative: caller-supplied evidence takes precedence, unsupported
facts are not invented, and pragmatic interpretations remain defeasible. Missing context is
left unknown. The compact discourse engine uses transparent text heuristics for some
operations, so its output is an inspectable research approximation rather than a validated
model of all human pragmatic comprehension.

Point 5 registers ore_language_comprehension and ore_discourse definitions in the existing
ore_logger.annotations SQLite table (ore_logs/ore_logger.sqlite3); it does not create a
separate annotation database.

Quick start:

```python
from ore_discourse import DiscourseConfig
from ore_language_comprehension import (
    Point5Request, Point5ContextualPragmaticIntegrator,
)

request = Point5Request(
    sequence_id="dialogue-001",
    sentences=["The deployment failed.", "Great work on the release!"],
    prior_turns=["The team expected the release to succeed."],
    prosody={1: {"intonation_mismatch": 0.8, "exaggeration": 0.6}},
)
result = Point5ContextualPragmaticIntegrator(
    DiscourseConfig(simulate_biology=False)
).process(request)

print(len(result.functional_trace))  # 20
print(result.enriched_interpretation["irony_hypotheses"])
print(result.handoff["target"])  # Point 6
```

To connect the actual Point 4 output, use
Point5Request.from_point4_result(sequence_id, point4_result, sentences=[...], ...).
The sentence text is required because Point 4's structured proposition is not a substitute
for the surface discourse history.

Run the focused acceptance suite:

```bash
python -m pytest tests/test_language_comprehension_point05.py -q
```

See docs/LANGUAGE_COMPREHENSION_POINT_05.md for architecture, equations, biological mapping
and limitations.


## Language comprehension — Point 6: Propositional working-memory representation and maintenance (18 steps)

Point 6 adds an 18-step source-level adapter over ORE's established
ore_working_memory five-stage / 40-operation processor. Point6PropositionalWorkingMemory
accepts the contextual interpretation from Point 5, preserves its literal and inferred
content separately, and passes the available proposition, situation, task and uncertainty
evidence into bounded working-memory buffers and cognitive-control operations.

The 18-step trace maps the 40 canonical WMCC operations in order. It also exposes the
working-memory engine's two additive source-extension records for semantic prediction error
and temporary interpretation consolidation. The adapter reports proposition candidates,
entity/event links, role and operator evidence, provenance, discourse/situation relations,
buffer activation and rehearsal, interference estimates, unresolved structures, semantic
competition, control demand and task state.

The result is explicitly temporary: no durable episodic or semantic memory write is performed.
The implementation is per-request rather than a persistent streaming state across separate
calls. Its eight-population current-based LIF reference is a computational analogy, not a
measured or anatomically fitted human neural model.

Point 6 registers ore_language_comprehension and ore_working_memory definitions in the
existing ore_logger.annotations SQLite table (ore_logs/ore_logger.sqlite3), not a new database.

Quick start:

```python
from ore_working_memory import WorkingMemoryConfig
from ore_language_comprehension import (
    Point6Request, Point6PropositionalWorkingMemory,
)

request = Point6Request(
    sequence_id="working-memory-001",
    text="Maya said that Omar had left.",
    partial_structures=[{
        "structure_id": "embedded-clause",
        "label": "embedded proposition",
        "expected_slots": ["content"],
        "filled_slots": ["content"],
    }],
)
result = Point6PropositionalWorkingMemory(
    WorkingMemoryConfig(simulate_biology=False)
).process(request)

print(len(result.functional_trace))  # 18
print(result.working_memory_state["buffers"].keys())
print(result.handoff["durable_storage_performed"])  # False
```

To consume Point 5 directly, use
Point6Request.from_point5_result(sequence_id, point5_result). It reads the surface sentences
from the Point 5 discourse result and carries forward the literal proposition, discourse
links, coreference, situation model, common ground, unresolved alternatives and task context.
An explicit surface text may be supplied when the Point 5 result has no sentence list.

Run the focused acceptance suite:

```bash
python -m pytest tests/test_language_comprehension_point06.py -q
```

See docs/LANGUAGE_COMPREHENSION_POINT_06.md for the 18 steps, mathematical models,
biological-reference mapping, API and limitations.


## Language comprehension Point 7 — long-term memory and decision use

The additive Point 7 implementation receives the structured proposition from Point 6 and
separates episodic encoding, semantic-knowledge candidates, retrieval, question answering,
and action planning. It exposes 28 ordered operations, preserves component-level provenance
and confidence, and abstains when evidence is insufficient.

Episodic traces and semantic candidates use separate SQLite tables through
`Point7MemoryStore`. The default is process-local `:memory:`; provide a database path to
persist records across restarts. This memory database is separate from the shared
`ore_logger.annotations` catalogue.

The optional ten-population current-based LIF reference includes hippocampal-binding,
entorhinal, perirhinal, parahippocampal, neocortical-semantic, medial-prefrontal,
dorsolateral-prefrontal, anterior-cingulate, orbitofrontal, and basal-ganglia functional
populations. It is a simplified biological analogy, not a fitted or validated human circuit.

See [docs/LANGUAGE_COMPREHENSION_POINT_07.md](docs/LANGUAGE_COMPREHENSION_POINT_07.md)
for the 28-step algorithm, equations, scientific qualifications, and API examples.

Run the focused Point 7 tests:

```bash
python -m pytest tests/test_language_comprehension_point07.py -q
```


## Language comprehension Point 8 — recurrent cross-stage integration (18 steps)

Point 8 is a system-level coordination layer across Points 1–7, not a compulsory
eighth stage that runs after memory and decision use. `Point8CrossStageIntegrator`
maintains a recurrent state across incoming increments and coordinates supplied
sensory, lexical, syntactic, semantic, discourse, working-memory, and memory outputs.

Its 18 source-level operations include multi-level prediction, typed prediction
errors, bounded attentional allocation, evidence-supported local reanalysis,
conditional pragmatic inference, working-memory accessibility, interpretation
adequacy checks, revisable proposition finalization, downstream routing, later
discourse revision, memory handoff, and outcome-based strategy adjustment.
It preserves literal and inferred content separately and does not inflate
confidence merely because a candidate was activated repeatedly.

The optional ten-population LIF reference is a functional analogy, not measured
or anatomically fitted human data. Point 8 coordinates the earlier processors;
it does not replace them or automatically write to durable memory. Episodic
and semantic memory requests are packaged for Point 7.

See [docs/LANGUAGE_COMPREHENSION_POINT_08.md](docs/LANGUAGE_COMPREHENSION_POINT_08.md)
for all 18 steps, equations, feedback loops, biological qualifications, and API examples.

Run the focused acceptance suite:

```bash
python -m pytest tests/test_language_comprehension_point08.py -q
```


## Language comprehension Point 9 — neural architecture and functional division of labour

Point 9 is a distributed network-level model, not another serial linguistic stage.
`Point9NeuralArchitectureCoordinator` coordinates perceptual, language, conceptual
knowledge, working-memory, mentalizing, executive-control, and episodic/contextual
systems through a sparse functional communication graph and recurrent state updates.
It keeps literal propositions separate from inferred mental-state candidates and
conditionally recruits social and executive support according to task demands.

The twenty operations provide an inspectable trace of evidence intake, system drives,
recurrent integration, typed prediction-error routing, contextual and working-memory
support, interpretation sufficiency, downstream handoffs, and diagnostics. Optional
individual-localizer and time-series inputs are used only when actually supplied;
the implementation does not infer neural localization from demographic information.
The optional seven-population LIF reference is a simulation, not measured human data.

See [docs/LANGUAGE_COMPREHENSION_POINT_09.md](docs/LANGUAGE_COMPREHENSION_POINT_09.md)
for all operations, mathematical approximations, API examples, and scientific limits.

Run the focused acceptance suite:

```bash
python -m pytest tests/test_language_comprehension_point09.py -q
```


## Language comprehension research algorithms — Point 10

Point 10 validates language-comprehension mechanisms using supplied behavioral, fMRI, EEG/ERP, MEG, lesion, stimulation, and computational evidence. It is a scientific analysis workflow, not another sentence-processing stage or a brain simulation. It records 15 operations from operationalizing a hypothesis to producing a calibrated, falsifiable report, with method-specific limitations and safeguards against double-counting dependent datasets.

See [Point 10 documentation](docs/LANGUAGE_COMPREHENSION_POINT_10.md) and run:

    python -m pytest tests/test_language_comprehension_point10.py


## Language comprehension research algorithms — Point 11

Point 11 separates core language-selective computations from sensory, conceptual, mentalizing, multiple-demand, and memory contributions while allowing conditional information exchange. It preserves literal-versus-inferred provenance, individual localizer information, task-dependent right-hemisphere contributions, and optional LIF reference dynamics. This is a functional architecture approximation, not a literal neural-routing claim.

See [Point 11 documentation](docs/LANGUAGE_COMPREHENSION_POINT_11.md) and run:

    python -m pytest tests/test_language_comprehension_point11.py


## Language comprehension research algorithms — Point 12

Point 12 is a 25-operation researcher-facing workflow for methodological evaluation, competing-hypothesis testing, confound auditing, measurement interpretation, model comparison, and calibrated reporting. It keeps statistical procedures separate from biological neural operations and refuses to rank hypotheses without comparable supplied likelihoods.

See [Point 12 documentation](docs/LANGUAGE_COMPREHENSION_POINT_12.md) and run:

    python -m pytest tests/test_language_comprehension_point12.py -q


## Language comprehension research algorithms — Point 14

Point 14 implements a 19-step research profile for clinically and practically relevant
language-comprehension patterns. It separates literal from inferred meaning, social from
nonsocial inference, discourse coherence, cue use, working-memory/executive load, and
component-specific longitudinal change. Optional task families cover right-hemisphere
injury, autism-related pragmatic tasks, psychosis-spectrum discourse, neurodegenerative
profiles, and aphasia-related language/concept contrasts.

The processor produces a transparent, uncertainty-aware functional profile. It does not
diagnose, infer a fixed deficit from a group label, or map one symptom to one brain region.
Possible biological implementations are described separately from analyst-side statistics.

See [Point 14 documentation](docs/LANGUAGE_COMPREHENSION_POINT_14.md) and run:

    python -m pytest tests/test_language_comprehension_point14.py -q


## Language comprehension — Point 15: temporal, network, and causal research

The source-ordered 25-stage Point 15 research workflow investigates unresolved
mechanisms rather than adding another cognitive processing stage. It organizes
falsifiable hypotheses, experimental conditions, event timing, multimodal neural
measurements, temporal overlap, candidate network connectivity, ethical causal
perturbation evidence, cross-population validity, and held-out prediction.

See [docs/LANGUAGE_COMPREHENSION_POINT_15.md](docs/LANGUAGE_COMPREHENSION_POINT_15.md)
for the data contract, mathematical operations, biological-reference track, tests,
and scientific limitations. The implementation is available in
`ore_language_comprehension.point15` and exports `Point15Request`,
`Point15MechanisticResearchProcessor`, and `register_point15_annotations`.

The research processor does not diagnose, infer unmeasured neural states, treat
connectivity as causality, or claim that statistical abstractions are literal
neuronal computations. Annotation registration reuses the shared
`ore_logger.annotations` SQLite table.

## Working-memory overload and information loss — Point 19

Point 19 adds a nineteen-operation, evidence-aware working-memory model in
`ore_working_memory_overload`. It maintains unresolved dependencies, estimates relative task
load and similarity-based interference, retrieves earlier representations by cue, distinguishes
encoding/maintenance/retrieval/binding/integration failures, and retains unresolved information
instead of silently filling gaps. The implementation combines a functional track with an
optional distributed rate-coded biological-reference network. The latter is a computational
analogy, not measured or anatomically fitted neural activity.

The package registers every class, function, and method in the shared
`ore_logger.annotations` SQLite catalogue and can log the full nineteen-operation trace through
the existing run/event tables. It reuses `ore_logs/ore_logger.sqlite3`.

See [Point 19 architecture, equations, biology, and API](docs/WORKING_MEMORY_OVERLOAD_POINT19.md).
Run the focused tests with:

```bash
python -m pytest tests/test_working_memory_overload_point19.py -q
```

## Schema-driven inference and reconstructive memory — Point 20

Point 20 implements a sixteen-operation reconstruction workflow in
`ore_reconstructive_memory`. It separates event-specific evidence from schema-generated
inferences, source-uncertain details, later suggestions, contradictions, and unknowns. It includes
cue-based episodic reactivation, multi-schema competition, Bayesian evidence/prior combination,
temporal/spatial/causal constraints, source monitoring, familiarity-versus-recollection checks,
competing-episode search, calibrated detail confidence, expected-loss reporting, provenance DAGs,
independent feedback updates, and separately gated schema learning.

The optional biological-reference network models recurrent interactions among hippocampal,
medial/lateral prefrontal, parietal, sensory association, temporal semantic, amygdala/salience, and
distributed event-integration populations. These are coarse functional reference signals rather
than measured neural activity or literal neuronal computations.

Every Point 20 class, function, and method is registered in the shared
`ore_logger.annotations` SQLite catalogue. The result logger writes all 16 operation traces
through the existing run/event tables.

See [Point 20 architecture, equations, API, and limitations](docs/RECONSTRUCTIVE_MEMORY_POINT20.md).
Run the focused tests with:

```bash
python -m pytest tests/test_reconstructive_memory_point20.py -q
```


## Confirmation bias and belief-consistent memory — Point 21

Point 21 adds a fourteen-operation model of how prior beliefs can influence information-source
selection, attention, evidence encoding, ambiguous interpretation, evidence readout, confidence
updating, and later memory retrieval. These stages are measured independently: a bias at one stage
does not imply that every later stage is biased.

The `ore_confirmation_bias` package provides typed competing-hypothesis, evidence-source, and
observation contracts; a functional inference pipeline; a distributed recurrent biological-reference
network; and an auditable operation registry. Source provenance and original observations remain
separate from interpretations and updated beliefs. Accuracy-first mode disables the modeled
belief-congruence preferences while retaining priors, source reliability, diagnosticity, and
measurement quality.

Every Point 21 class, function, and method is registered in the shared
`ore_logger.annotations` SQLite catalogue. `record_result_to_sqlite()` records all fourteen
operation outputs through the existing run/event tables.

See [Point 21 architecture, equations, API, and limitations](docs/CONFIRMATION_BIAS_POINT21.md).
Run the focused tests with:

```bash
python -m pytest tests/test_confirmation_bias_point21.py -q
```


## Cognitive heuristics and systematic reasoning errors — Point 22

Point 22 implements a seventeen-operation model of fast judgment shortcuts and their
failure conditions. It separates availability from real-world frequency, prototype similarity
from probability, fluency from independent evidence, numeric anchors from evidence-based
estimates, and correlation from causal identification. The algorithm produces an initial
intuitive judgment and a distinct verification result, quantifies sampling uncertainty, evaluates
whether a heuristic fits the environment, and records feedback without attributing an error to
a cue unless cue-level feedback is supplied.

The ore_cognitive_heuristics package includes a functional algorithm, an optional distributed
population-rate biological-reference track, typed contracts, an auditable operation registry, and
SQLite annotation/operation logging through the shared ore_logger tables. The biological track
is an engineering analogy rather than a validated neural simulation.

See docs/COGNITIVE_HEURISTICS_POINT22.md for architecture, equations, API, and limitations.
Run the focused tests with:

~~~bash
python -m pytest tests/test_cognitive_heuristics_point22.py -q
~~~


## Garden-path sentences and syntactic ambiguity — Point 23

Point 23 implements a seventeen-operation incremental sentence-parsing and recovery model in
the ore_garden_path package. It records lexical-category alternatives, grammar-supplied parse
hypotheses, token surprisal, parse competition, working-memory dependencies, disambiguating
conflict, revision choice points, thematic-role repair, partial/good-enough interpretations, and
adaptation from explicit feedback.

The processor accepts language- and modality-specific lexical candidates and parse hypotheses.
It is a configurable parser/reanalysis research prototype, not a general-purpose grammar: when
no grammar candidates are supplied, it emits a surface-only fallback and an unresolved result.
Its optional distributed frontotemporal population-rate track is an engineering abstraction,
not a validated neural simulation.

See docs/GARDEN_PATH_POINT23.md for architecture, equations, API, tests, and limitations.
Run the focused tests with:

~~~bash
python -m pytest tests/test_garden_path_point23.py -q
~~~


## Point 24 — Feedback, clarification, and misconception correction

The ore_corrective_learning package implements a twenty-operation feedback and correction loop. It compares predictions with observations, classifies possible error sources, chooses clarification or verification actions using expected information gain, evaluates evidence provenance and independence, updates supplied hypotheses with Bayesian evidence weighting, and tests the revised representation through retrieval and transfer. It also tracks confidence calibration, consolidation strength, and possible belief regression.

The implementation does not fetch evidence or invent candidate explanations. Evidence likelihoods, source reliability, independence groups, and any ground-truth labels must be supplied by the caller. Its optional distributed population-rate biological reference is an illustrative systems-level abstraction, not a validated neural simulation.

See docs/CORRECTIVE_LEARNING_POINT24.md for the full algorithm, equations, API, scientific limits, and source references. Run the focused tests with:

~~~bash
python -m pytest tests/test_corrective_learning_point24.py -q
~~~


## Point 07 — Feedback Seeking

The `ore_feedback_seeking` package implements the 18-step Point 07 feedback-correction algorithm in separate functional and distributed biological-reference traces. It includes Bayesian feedback-source evaluation, confidence calibration, targeted correction, retrieval and transfer tests, delayed-retention scheduling, and shared SQLite annotation/event logging. See [`docs/FEEDBACK_SEEKING_POINT07.md`](docs/FEEDBACK_SEEKING_POINT07.md) for the mathematics, API, safeguards, and biological-model limitations.


## Point 08 — Stay Motivated and Curious

The `ore_curiosity_motivation` package implements the 24-step curiosity and motivated-learning algorithm with distinct functional and distributed biological-reference tracks, mathematical operation registry, delayed-learning evaluation, and shared SQLite annotation/event logging. See [`docs/CURIOSITY_MOTIVATION_POINT08.md`](docs/CURIOSITY_MOTIVATION_POINT08.md) for the API, equations, scientific caveats, and validation instructions.

## Task 04 — Connect sensory processing to language and semantic understanding

The `ore_text_localization` package turns raw printed-text images into ordered glyph patches, reuses the existing N9 visual-language pathway, hands supported interpretations to working memory, and registers as a typed module in Task 3's `CognitiveTaskCoordinator`. It preserves segmentation alternatives and source coordinates, and records annotations in `ore_logger.annotations`. See [`docs/SENSORY_LANGUAGE_INTEGRATION_TASK04.md`](docs/SENSORY_LANGUAGE_INTEGRATION_TASK04.md) for the mathematical operations, API, coordinator registration and limitations.
