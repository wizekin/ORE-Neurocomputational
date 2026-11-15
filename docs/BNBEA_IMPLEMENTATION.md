# BNBEA — Master Biological Neurocomputational ORE

**Package:** ore_bnbea/  
**Input:** an ore_n9.N9Request, optional context priors, and optional independent brain/behavior measurements  
**Output:** a functional N9 result, a separate recurrent neural simulation, predictive-error metrics, validation report, 139 detailed master-operation records, and SQLite-linked annotations  
**Status:** software research prototype; empirical, behavioral, and biological acceptance remain separate and pending

## 1. Purpose and boundaries

BNBEA is an additive master layer over the existing ORE repository. It implements the nine-stage sequence in the library file without replacing N1–N9:

- The **functional track** invokes the repository's N9 controller. N9 already coordinates N1 early vision, N2 visual organization, optional fitted N3/N4, and explicitly injected N5–N8/syntax adapters.
- The **biological track** runs a reduced conductance-based leaky integrate-and-fire (LIF) network with specialized populations, delayed synaptic projections, excitatory/inhibitory interactions, recurrent feedback, bounded plasticity, and two-timescale associative memory.
- The **predictive track** independently combines N9 likelihoods with caller-supplied context priors and calculates precision-weighted error, posterior uncertainty, surprisal, and information gain.
- The **validation track** computes EEG/MEG forward and inverse model quantities, ERP/STFT summaries, HRF/BOLD convolution, GLM fits, representational similarity, lesion/ablation effects, behavioral metrics, and bootstrap intervals only when the corresponding real data are supplied.

These tracks remain distinct. A software posterior is not a neuronal firing rate; a simulated population is not an anatomical region; and a close fit to one dataset is not proof that the implementation reproduces the human brain.

N9 expects already-localized text patches in logical reading order. A full line/page image must be segmented into candidate patches upstream. Each image patch is grayscale and at least 4 × 4 pixels. N9 runs N1 and N2 for image patches and records not_applicable or upstream_provided when pixel-level processing is not available.

## 2. Nine stages and detailed operation registry

The detailed operation lists in the source file yield the following counts:

| Stage | Code | Main function | Operations |
|---:|---|---|---:|
| 1 | NFS | Neural simulation foundations | 11 |
| 2 | BVP | Biological visual processing | 12 |
| 3 | HVA | Hierarchical visual architecture | 12 |
| 4 | NOR | Neural orthographic representations | 14 |
| 5 | NCR | Neural connectivity and recurrence | 13 |
| 6 | BLM | Biological learning and memory | 16 |
| 7 | PCP | Predictive and contextual processing | 18 |
| 8 | BDV | Brain-data validation | 19 |
| 9 | INO | Integrated neurocomputational ORE | 24 |
|  |  | **Total** | **139** |

The uploaded file has conflicting summary arithmetic (including a stated 135-operation total and an earlier 8/8 allocation for visual stages). The implementation follows the detailed numbered stage lists: 11 + 12 + 12 + 14 + 13 + 16 + 18 + 19 + 24 = **139**. Each record has a stable ID such as NFS-001 or BDV-019, an explanation, a biological reference, a mathematical-operation description, and an execution status.

### Stage 1 — Neural simulation foundations

The biological engine defines a point-neuron state (membrane voltage, capacitance, leak, threshold, reset, refractory timer), directed synaptic projections, population activity, and a discrete time engine. Excitatory and inhibitory synapses are distinct conductance channels. It tracks spike rasters and estimates firing rate from emitted spikes.

The passive membrane equation is

\[
C_m \frac{dV}{dt}
= g_L(E_L - V) + I_{\mathrm{exc}} + I_{\mathrm{inh}} + I_{\mathrm{ext}}.
\]

The implementation uses explicit Euler integration:

\[
V_{t+\Delta t}=V_t+\frac{\Delta t}{C_m}
\left[g_L(E_L-V_t)+I_{\mathrm{syn}}+I_{\mathrm{ext}}\right].
\]

A threshold crossing emits a discrete spike, resets the membrane, and activates a refractory interval. Synaptic conductance decays exponentially, \(g(t+\Delta t)=g(t)e^{-\Delta t/\tau_s}\), before new presynaptic events are added.

### Stage 2 — Biological visual processing

For image patches, BNBEA reuses N1 and its existing retinal/LGN/V1 implementation through N9. The available mathematical operations include local Gaussian response, light adaptation, ON/OFF rectification, center-surround difference-of-Gaussians, orientation-selective Gabor filters, quadrature energy, and divisive population normalization. ore_bnbea.math_ops also exposes reusable DoG, Gabor, orientation-energy, and divisive-normalization functions for controlled experiments.

### Stage 3 — Hierarchical visual architecture

N9 reuses N2 for multiscale receptive fields, contours, boundary cues, figure-ground organization, junction/curvature evidence, shape fragments, and structural features. N9 also records patch grouping and alignment hypotheses. The master layer preserves these as visual evidence and drives the separate visual-hierarchy population. It does not claim that the reduced nine-node circuit is a detailed model of the ventral or dorsal anatomy.

### Stage 4 — Neural orthographic representations

The functional orthographic posterior, sequence hypotheses, and lexical/syntactic/semantic outputs come from N9 and the optional N3–N8 adapters. The package provides additional mathematical primitives for prototype similarity (including Mahalanobis distance), entropy, ordinal position codes, ordered n-grams, cosine similarity, and edit distance. Ambiguous posterior mass remains available; the master wrapper does not force an uncertain candidate into a single correct label.

### Stage 5 — Neural connectivity and recurrence

The default network has nine computational populations: sensory/retina–LGN–V1, visual hierarchy, orthographic/VWFA-like, language association, associative memory, predictive context, executive control, validation observer, and ORE integrator. Its directed projections have a weight matrix, synaptic type, decay time constant, delay queue, conductance state, and eligibility traces.

For a target population j, synaptic current is approximated by

\[
I_j^{\mathrm{syn}} =
\sum_i g_{ij}(t)(E_{\mathrm{rev}}-V_j).
\]

Feedforward, feedback, and lateral pathways are explicit; inhibitory projections contribute their own reversal potential. The simulator reports network activity, weight bounds, and a phase-coherence proxy when there is sufficient oscillatory activity to estimate phase.

### Stage 6 — Biological learning and memory

The biological graph supports co-activation/Hebbian updates, pre/post eligibility traces, timing-dependent potentiation and depression, weight clipping, and activity-dependent homeostatic scaling. A compact associative store keeps fast traces separate from slower consolidated traces and supports cosine retrieval, replay, consolidation, and fast-trace decay.

The STDP-style update uses

\[
\Delta W =
A_+\,\mathbf{s}_{post}\mathbf{e}_{pre}^{T}
-
A_-\,\mathbf{e}_{post}\mathbf{s}_{pre}^{T},
\quad
\mathbf{e}(t+\Delta t)=
e^{-\Delta t/\tau_e}\mathbf{e}(t)+\mathbf{s}(t).
\]

This is an engineering approximation of timing-sensitive plasticity, not a claim that biological synapses use the same implementation or fitted constants. Persistent N5 model learning remains governed by N5's existing explicit authorization and independent-verification contracts. The BNBEA local STDP update changes only the per-run simulated network.

### Stage 7 — Predictive and contextual processing

The predictive layer normalizes caller-supplied priors, combines them with N9 posterior likelihoods, and computes residual and uncertainty diagnostics. For class c,

\[
P(c\mid x,\mathcal{C}) =
\frac{P(x\mid c)P(c\mid\mathcal{C})}
{\sum_k P(x\mid k)P(k\mid\mathcal{C})}.
\]

The prediction error is \(\epsilon=x-\hat{x}\), precision is approximated by inverse variance \(\Pi=(\sigma^2+\varepsilon)^{-1}\), and the quadratic error proxy is

\[
F_{\mathrm{err}}=\frac{1}{2}\sum_i \Pi_i\epsilon_i^2.
\]

The system also records entropy, surprisal \(-\log P(x)\), and entropy reduction as an information-gain proxy. Context priors must be supplied; the model does not silently invent world knowledge.

### Stage 8 — Brain-data validation

Stage 8 is measurement-driven and optional. Supported operations include:

- EEG/MEG forward projection \(Y=XL^T\) and regularized inverse estimation \(\hat X=YL(L^TL+\lambda I)^{-1}\).
- ERP trial averaging, STFT power, HRF convolution for a BOLD proxy, and ridge GLM fitting.
- Euclidean representational dissimilarity matrices and correlation of their upper triangles.
- Human-label accuracy, MAE/RMSE, held-out accuracy, bootstrap percentile intervals, lesion deltas, and ablation deltas.

If no experimental data are supplied, Stage 8 is not_tested; it does not generate synthetic brain observations and label them as evidence. Even when measurements are computed, the default evidence status remains inconclusive until pre-registered acceptance criteria, appropriate controls, real datasets, and independent replication are available.

### Stage 9 — Integrated neurocomputational ORE

The controller assembles the functional N9 result, simulated population state, plasticity and memory metrics, contextual posterior, validation results, and full operation trace. It records adapter availability and whether persistent learning was authorized. The final report preserves channel-specific evidence and explicitly separates software completion from biological acceptance.

## 3. Python API

~~~python
from ore_bnbea import BNBEAConfig, BNBEAProcessor, BNBEARequest
from ore_n9 import N9Patch, N9Request

n9_request = N9Request(
    sequence_id="reading-001",
    patches=[
        N9Patch("letter-0", grapheme_probabilities={"a": 0.9, "o": 0.1}),
        N9Patch("letter-1", grapheme_probabilities={"b": 0.8, "d": 0.2}),
    ],
    script_profile={"script": "Latin", "reading_direction": "ltr"},
    context={"document_topic": "example"},
)

processor = BNBEAProcessor(
    config=BNBEAConfig(simulation_steps=24, neurons_per_population=12),
)
result = processor.process(BNBEARequest(
    n9_request=n9_request,
    context_priors={"a": 0.8, "o": 0.2, "b": 0.8, "d": 0.2},
))

print(len(result.operation_trace))                   # 139
print(len(result.functional_result.operation_trace)) # 120
print(result.biological_state["network"]["projection_count"])
print(result.validation["evidence_status"])          # inconclusive until assessed
~~~

Use N9Adapters to provide fitted N3/N4 and optional N5/N6/N7/N8/syntax adapters. A configured N3 processor must be fitted before N9 inference. The existing N5 interface continues to require independent verification and explicit user authorization before persistent model updates.

BrainValidationInput accepts the optional experimental arrays. Its documented shapes are [time, source] for source activity, [sensor, source] for lead fields, [trial, time, sensor] for trials, and [observation, regressor] for GLM design matrices. Missing modalities are retained as not_tested.

## 4. SQLite annotations and execution events

When BNBEAProcessor is initialized, it runs ore_logger.catalog.sync_repository over the repository. Every Python class, function, and method with a source definition is indexed using its docstring, signature, source path, and source hash. It then inserts all 139 stage-operation explanations into the existing annotations table.

The default database is:

    ore_logs/ore_logger.sqlite3

A custom location can be provided with BNBEAProcessor(database_path=...). The run is added to runs; each master operation and each N9 sub-operation is stored in events with a foreign-key link to its annotation. Event metadata includes stage/operation IDs and compact summaries, not raw images or large sensor arrays. SQLite files remain local and are excluded from Git.

## 5. Run tests

~~~bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_bnbea.py -q
python -m pytest
~~~

The tests verify the operation registry, LIF/refractory dynamics, recurrent graph and weight constraints, visual/orthographic/Bayesian mathematics, experimental-validation functions, N9 integration, and SQLite annotations/events.

## 6. Scientific limitations

The point-neuron network is intentionally small and does not include realistic morphologies, ion-channel dynamics, cell-type diversity, anatomical connectomes, conduction distributions, detailed neurotransmitter kinetics, or calibrated physiological parameters. The functional pipeline is a deterministic software architecture with biologically motivated components, not a simulation of all human reading or comprehension. N9 adapters that are absent remain unavailable rather than being fabricated. Behavioral and neural correspondence must be tested against independent, appropriately sampled datasets; software tests establish code behavior only.
