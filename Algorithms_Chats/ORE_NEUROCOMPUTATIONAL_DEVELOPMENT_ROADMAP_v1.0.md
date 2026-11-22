# ORE Neurocomputational Development Roadmap v1.0

**Date:** 13 November 2026  
**Baseline:** ORE A–J + Experiment Workbench 0.8.0, Stage A computational activation  
**Roadmap status:** Neurocomputational software prototype phases N0–N7 implemented; N8 scientific validation pending  
**Compatibility rule:** This roadmap is an additive research layer. It does not replace, patch, or silently alter the accepted ORE A–J implementation.

## 1. Executive result

ORE A–J is the existing orthographic recognition pipeline. The neurocomputational extension adds an inspectable experimental model of several candidate mechanisms that may contribute to visual word processing:

1. Early oriented visual filtering and local contrast normalization.
2. Distributed grapheme-selective population activity.
3. Recurrent orthographic prediction and prediction-error signals.
4. Attention/precision modulation.
5. Explicit, consent-gated experience-dependent plasticity.
6. Mechanism ablations, traces, and reproducible model bundles.
7. A bridge from the supplied Stage A sample format.

The implementation is deliberately modest. It is a **computational analogy and research prototype**, not a simulation of the full reading network, a validated neural circuit model, or evidence that ORE reproduces the brain. In particular, its feedback loop operates over candidate grapheme beliefs and prototype feature vectors; it is not yet a learned pixel-generative cortical hierarchy.

## 2. Relationship to the existing A–J system

| Existing ORE stage | Existing responsibility | Neuro extension relationship |
|---|---|---|
| A | Initialize, validate and execute deterministic patch baseline | N7 adapter accepts `StageASample` pixels; A is not changed |
| B | Prepare source/patch data, labels and leakage-aware splits | Future corpora should be prepared through B before neuro experiments |
| C | Train visual/character representation baseline | N1/N2 are independent alternative representations for controlled comparison |
| D | Compare absolute, relative and overlap position codes | N3 currently uses ordered sequence neighbors, not a replacement for D's A/B/O codes |
| E | Learn orthographic regularities/chunks | N3 uses smoothed grapheme transitions as a deliberately simple prior |
| F | Train/evaluate/freeze A–J model | Neuro bundle has its own independent checkpoint and manifest |
| G | Recognize from supplied pre-segmented patches | Neuro recognizer accepts pre-segmented patch sequences only |
| H | Serialize result/provenance | Neuro result JSON and bundle use separate schema and hashes |
| I | Verified incremental learning overlay | N5 is separate Oja-style plasticity; it does not update Stage I overlays |
| J | A–J software acceptance and scientific gate | Neuro tests are additional software evidence; they do not inherit J scientific acceptance |

## 3. Phase register and completion status

### N0 — Baseline audit, scope and isolation — COMPLETE

**Purpose:** Preserve the current A–J contract and prevent a research prototype from being misreported as a validated replacement.

**Implemented:**
- Built on the latest A–J 0.8.0 + Stage A activation package.
- Kept all new code in `ore_neuro/` with its own tests and artifacts.
- Kept recognition read-only; all model updates require a separate explicit operation.
- Recorded scientific limitations in every result and checkpoint.

**Exit evidence:** Baseline A–J suite passed 179 tests before the additive extension. The extension is tested independently and in the combined suite.

### N1 — Early visual processing — SOFTWARE COMPLETE

**Mechanism hypothesis:** Local oriented contrast filters provide a useful computational analogue of early visual feature extraction. A quadrature pair estimates phase-tolerant oriented energy; local normalization reduces sensitivity to broad illumination changes; pooling creates coarser spatial summaries.

**Implemented in:** `ore_neuro/core.py`, `EarlyVisualEncoder`.

**Algorithm:**
1. Validate a finite rectangular grayscale matrix in `[0,1]`.
2. Convert white-background intensity to ink strength.
3. Bilinearly resample to `image_size × image_size`.
4. Compute 3×3 local mean and variance.
5. Normalize local contrast: `I_n=(I-μ_local)/sqrt(max(σ²_local, ε))`.
6. For each configured wavelength and orientation, convolve with even/odd Gabor filters.
7. Compute quadrature energy `Eθ,λ=sqrt((I_n*g_even)^2+(I_n*g_odd)^2)`.
8. Pool each 2×2 cell by mean and maximum and L2-normalize the concatenated vector.
9. Return features and inspectable filter/energy diagnostics.

**Not implemented:** learned V1 receptive fields, retinal/LGN physiology, cortical layers, complex cell dynamics, color, foveation, eye movements, or image-region/character segmentation.

### N2 — Distributed orthographic population code — SOFTWARE COMPLETE

**Mechanism hypothesis:** A written patch may activate a distribution of candidate-selective units rather than an irreversible single identity at the first stage.

**Implemented in:** `OrthographicPopulationLayer`.

**Algorithm:**
1. Encode explicitly labeled reference glyphs with N1.
2. Group features by grapheme.
3. Compute and normalize one prototype vector per grapheme.
4. Compute cosine similarity between a new patch vector and each prototype.
5. Apply softmax temperature to create a class-activation distribution.
6. Preserve the entire distribution for later recurrent inference.
7. If the final maximum posterior is below the configured heuristic threshold, emit `<UNK>` instead of forcing a class.

**Important simplification:** One prototype per grapheme is not a measured population of neurons. There are no learned heterogeneous tuning curves, cell noise, population geometry measurements, or independent OOD calibration yet.

### N3 — Recurrent orthographic prediction and error — SOFTWARE COMPLETE

**Mechanism hypothesis:** Candidate identities can exchange information across neighboring positions. A prediction derived from learned orthographic regularities is combined with bottom-up visual evidence; the mismatch is retained for inspection.

**Implemented in:** `OrthographicTransitionModel` and `PredictiveCodingLayer`.

**Algorithm:**
1. Fit smoothed unigram and directed bigram counts from explicitly supplied grapheme sequences.
2. Convert counts into forward and reverse conditional distributions.
3. Initialize each patch belief from N2 visual probabilities.
4. For each iteration, predict each position from its own unigram prior and available left/right neighbor beliefs.
5. Compute sensory precision from segmentation confidence and attention.
6. Compute prior precision from predictive gain, visual entropy and segmentation uncertainty.
7. Combine log visual evidence and log prior; normalize by softmax.
8. Apply damping to the previous belief and proposed posterior.
9. Record prediction divergence, belief change and a diagnostic energy proxy.
10. Stop at configured convergence tolerance or the maximum step count.
11. Preserve both prior and posterior, their classwise divergence, and a feature-space residual between observed N1 features and the prior-weighted prototype prediction.

**Critical qualification:** The class-belief update and prototype-space residual are an engineering approximation. A stronger predictive-coding experiment must learn a generative model that predicts visual feature maps or pixels, computes layer-wise residuals, and tests whether feedback actually explains away redundant sensory input.

### N4 — Attention and precision gating — SOFTWARE COMPLETE

**Mechanism hypothesis:** Sensory evidence should have variable influence depending on signal reliability and attentional allocation.

**Implemented:** `NeuroPatch.segmentation_confidence`, `NeuroPatch.attention`, and the precision terms in `PredictiveCodingLayer`.

`visual_precision=(0.5+0.5·segmentation_confidence)·(1+attention_gain·attention)`.

`prior_precision=prediction_gain·(0.5+normalized_visual_entropy)·(2-segmentation_confidence)`.

Attention can be disabled as an ablation while leaving the visual evidence unchanged. These equations are configurable engineering choices, not biologically measured gain laws.

### N5 — Experience-dependent plasticity — SOFTWARE COMPLETE

**Mechanism hypothesis:** Experience may strengthen or modify candidate-selective representations and orthographic transition memories.

**Implemented:** `ORENeuroEngine.learn_verified(...)`.

**Algorithm:**
1. Require `independently_verified=True` and `user_authorized=True`.
2. Clone the current model; never mutate the frozen parent.
3. Apply a bounded Oja-style update to the labeled population prototype for each verified example: `Δw=ηy(x-yw)` followed by normalization.
4. Extend existing transition/unigram memory only with explicitly supplied verified sequence labels.
5. Increment model version and exposure counters.
6. Save the child as a new, separately fingerprinted model bundle.

**Not implemented:** continual-learning replay, synaptic consolidation, homeostatic plasticity, interference/forgetting controls, sleep-like consolidation, or multi-timescale plasticity. The Oja update is a simple algorithmic choice, not a claim about the exact learning rule in human reading circuits.

### N6 — Mechanism instrumentation, ablations and reproducibility — SOFTWARE COMPLETE

**Implemented:**
- Per-patch visual probabilities, contextual prior, posterior, classwise prediction-error terms, feature residual vector/norm, heuristic explained fraction, precision, entropy and decision.
- Per-iteration prediction divergence, mean belief change and energy proxy.
- Feed-forward/no-attention, recurrent/no-attention, and recurrent/attention conditions on the same frozen model and samples.
- Accuracy, negative log-likelihood, Brier score and descriptive 10-bin expected calibration error.
- Model weight SHA-256, manifest SHA-256 sidecar, model fingerprint, version and explicit scientific status.
- Bundle load checks for schema, file paths, checksums, array dimensions, finite values, probability normalization and model fingerprint.

**Interpretation:** These instruments establish observability and repeatability. An ablation effect in a small synthetic fixture is not evidence that the ablated mechanism is necessary in the brain.

### N7 — Stage A bridge and repeatable demo — SOFTWARE COMPLETE

**Implemented:** `from_stage_a_sample(...)` plus `examples/run_neurocomputational_demo.py`.

The demo takes the already-segmented patches in `examples/workbench/stage_a_request.example.json`, executes Stage A, fits N1/N2 from the explicitly labeled references, runs feed-forward and recurrent neuro inference, saves result JSON and a separate model bundle, then writes a controlled ablation smoke report.

**Data-integrity warning:** The bundled synthetic sample repeats the exact glyph matrices used as references. Consequently, this demo proves that the adapter and stages execute end-to-end; any resulting accuracy is a memorization smoke check and must not be quoted as held-out recognition performance.

### N8 — Independent neurocognitive and real-corpus validation — PENDING EXTERNAL EVIDENCE

This phase is intentionally not auto-completable by writing more code or passing synthetic tests. It requires independent data, preregistered methods and empirical comparison.

**Required work package:**
1. Assemble rights-cleared real image corpora with source/licence provenance for Latin and Arabic, including multiple writers, typefaces, sizes, layouts and acquisition conditions.
2. Separate train/validation/test by writer, font family, source document and string family; report exact grouping policy and hashes.
3. Include minimal pairs, transpositions, substitutions, insertions/deletions, spacing perturbations, blur, noise, occlusion, low contrast, skew and segmentation uncertainty.
4. Compare Stage A baseline, A–J Stage C/G, N1/N2 feed-forward, N3 recurrent, N4 precision-gated, and relevant conventional OCR baselines.
5. Run at least five fixed seeds for stochastic variants; publish confidence intervals and per-script/per-condition breakdowns.
6. Measure character error rate, word error rate when a word recognizer is included, top-k accuracy, abstention risk/coverage, NLL, Brier, ECE, robustness and latency.
7. Run ablations that independently remove recurrence, neighbor directions, learned transition priors, attention, local contrast normalization and pooling.
8. Test whether feedback improves performance on held-out cases without increasing systematic orthographic hallucination.
9. Seek human behavioral comparison (accuracy and response time) and, only where feasible, eye-tracking/EEG/fMRI or representational-similarity evidence from appropriately collected studies.
10. Have an independent reviewer audit data rights, leakage, analysis code, preregistration, failure cases and biological claims.

**Exit gate:** No “biologically validated” or “reproduces the brain” claim unless a preregistered model makes quantitative predictions that survive independent empirical tests against appropriate human/neural observations. Real-corpus OCR acceptance and neural-mechanism validation are separate gates.

## 4. Mathematical model summary

For patch `i`, N1 generates feature vector `x_i`; N2 produces visual class probabilities `q_i(c)`.

For transition counts `N(a,b)` and smoothing `α`:

`P(b|a) = (N(a,b)+α) / Σ_b'(N(a,b')+α)`.

`P(a|b)` is estimated from the same counts normalized by destination column.

For a current sequence belief matrix `B`, the contextual prior is an average of the smoothed unigram prior and available neighbor predictions. The update is:

`B_i^(t+1) = normalize((1-ρ)B_i^t + ρ·softmax(π_v,i log q_i + π_p,i log prior_i^t))`.

The classwise prediction-error decomposition is:

`e_i(c)=B_i(c)·(log B_i(c)-log prior_i(c))`.

The scalar divergence is the sum across classes. The model also reports the feature-space residual:

`r_i=x_i - Σ_c prior_i(c)·w_c`.

This last quantity is not a pixel-level cortical prediction error; it is a difference between a measured engineered feature vector and a prior-weighted class prototype.

## 5. Run instructions

From the package root:

```bash
python -m unittest tests.test_ore_neuro -v
python -m unittest discover -s tests -v
python examples/run_neurocomputational_demo.py --output neuro_demo_output
```

Main API:

```python
from ore_neuro import (
    LabeledGlyph, NeuroConfig, NeuroPatch, NeuroSequence, ORENeuroEngine,
    from_stage_a_sample, load_bundle,
)

engine = ORENeuroEngine(NeuroConfig())
engine.fit([LabeledGlyph("a", glyph_a), LabeledGlyph("b", glyph_b)], sequences=[("a", "b")])
result = engine.recognize(NeuroSequence("sample", [
    NeuroPatch("patch-1", glyph_a, segmentation_confidence=0.9, attention=1.0),
    NeuroPatch("patch-2", glyph_b, segmentation_confidence=0.8, attention=0.5),
]))
```

Recognition is read-only. To make a new learning version, call `learn_verified(...)` with independently verified labels and explicit user authorization. The original model remains unchanged.

## 6. Current completion matrix

| Phase | Status | Evidence |
|---|---|---|
| N0 Baseline audit and isolation | COMPLETE | Existing A–J preserved; new package isolated |
| N1 Early visual processing | COMPLETE | Gabor filter bank, normalization, pooling, tests |
| N2 Distributed grapheme populations | COMPLETE | Prototype distributions, uncertainty and `<UNK>`, tests |
| N3 Recurrent prediction/error | COMPLETE | Iterative prior/posterior traces and ablation |
| N4 Attention/precision | COMPLETE | Explicit gate and ablation |
| N5 Plasticity | COMPLETE | Gated Oja-style child model; parent immutability tests |
| N6 Metrics, traces, ablations, integrity | COMPLETE | Deterministic tests and bundle round-trip/tamper checks |
| N7 Stage A bridge/demo | COMPLETE | End-to-end demo on supplied Stage A request fixture |
| N8 Independent empirical validation | PENDING | Requires new independent real-world and neurocognitive evidence |

## 7. Scientific context and guardrails

The reading system is distributed and literacy-tuned; it should not be reduced to a single “word area.” Recent studies report differentiation within the ventral occipitotemporal reading system and interactions between visual and language-related cortical networks. Predictive coding is a useful, testable account of how orthographic expectations could interact with visual evidence, but it remains a hypothesis to test against alternative models, not a settled description of every neural computation.

Useful starting literature:

- Gagl et al. (2020), *An orthographic prediction error as the basis for efficient visual word recognition*: https://pmc.ncbi.nlm.nih.gov/articles/PMC7284316/
- 2024 Human Brain Mapping study on posterior/anterior VWFA functional specialization: https://pmc.ncbi.nlm.nih.gov/articles/PMC10941549/
- 2024 study of distributed cortical networks in visual word processing: https://pmc.ncbi.nlm.nih.gov/articles/PMC10831251/
- 2026 review, *The reading brain – Transforming vision into language*: https://pmc.ncbi.nlm.nih.gov/articles/PMC13491551/

These papers motivate hypotheses and evaluation targets. They do not validate this implementation.

## 8. Next development after v1.0

Only after the N8 corpus protocol and measurements are available should the next software branch be selected. Likely candidates are: (a) learned generative feature-map feedback rather than class-only priors, (b) multi-scale receptive fields and retinotopic position, (c) uncertainty propagation from image segmentation hypotheses, (d) writer/font-invariant population learning, (e) multi-timescale plasticity and forgetting control, and (f) behaviorally grounded eye-movement/serial-attention experiments. Each must be treated as a separately testable hypothesis and introduced behind an ablation switch.
