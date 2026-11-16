# ORE VWFA — Six-stage visual word-form implementation

**Package:** \`ore_vwfa\`  
**Operation registry:** \`VWFA-001\`–\`VWFA-024\`  
**Stages:** A–F, with 4 / 4 / 4 / 5 / 4 / 3 operations  
**Evidence tracks:** functional computation and a distinct population-level LIF reference

## Purpose and scientific boundary

This additive ORE subsystem implements the six stages and 24 numbered steps in \`VWFA.txt\`. It does not replace \`ore_brain_networks\`, N1–N9, or the integrated reading packages. It provides a focused, traceable implementation of early visual feature extraction, hierarchical visual populations, posterior-to-anterior VWFA specialization, whole-word and morphological-form processing, cortical-network exchange, and literacy-dependent plasticity.

The biological components are **computational reference models**. Gabor filtering, softmax, graph propagation, prototype distances, delta-rule learning, and LIF dynamics are useful mathematical abstractions; they are not claims that neurons literally execute those exact software operations. Default numerical values are reproducible engineering settings, not physiological constants fitted to a participant.

The package deliberately does not invent OCR output, grapheme labels, word entries, morpheme meanings, phonological targets, or human measurements. A full image can produce visual features and candidate regions, but symbol identification requires compatible labelled \`GraphemePrototype\` records or an upstream \`orthographic_sequence\`. Whole-word candidates require a caller-provided lexicon. Morphological analyses require a caller-provided morpheme inventory. Semantic and grammatical features are read only from those records or the request context.

## Operation registry

| ID | Stage / step | Functional and mathematical implementation | Biological reference |
|---|---|---|---|
| VWFA-001 | A · 1. Acquire visual input | Sample image; Gaussian local luminance and variance; local contrast and contrast normalization | Photoreceptor adaptation and retinal/LGN contrast pathways |
| VWFA-002 | A · 2. Elementary visual features | Even/odd Gabor convolution, quadrature energy, multi-scale orientation channels, divisive normalization | V1 orientation- and spatial-frequency-selective populations |
| VWFA-003 | A · 3. Visual conjunctions | Weighted feature summation, local coherence, nonlinear \`tanh\` activation | Intermediate visual populations integrating edges, curves, corners and intersections |
| VWFA-004 | A · 4. Spatial relations | Connected candidate regions, bounding boxes, centroids, normalized positions and ordered evidence | Retinotopic coding and position-sensitive visual structure |
| VWFA-005 | B · 5. Simple detector populations | Distributed feature vectors, rectification, L2 normalization | Population coding of visual properties |
| VWFA-006 | B · 6. Complex receptive fields | Two deterministic affine/nonlinear hierarchy levels, matrix products and \`tanh\` | Increasing receptive-field complexity through V2/V4 and ventral visual cortex |
| VWFA-007 | B · 7. Position tolerance | Local feature smoothing/pooling with a separate ordinal-position channel | Tolerance to small visual variation while retaining letter order |
| VWFA-008 | B · 8. Candidate orthography | Variance-scaled Mahalanobis distance, radial-basis activation, softmax and unknown mass | Distributed learned grapheme-sensitive populations |
| VWFA-009 | C · 9. Posterior VWFA | Posterior population activation from position-sensitive visual evidence | Posterior ventral occipitotemporal/VWFA-1 visual-detail bias |
| VWFA-010 | C · 10. Sublexical units | Overlapping bi-/tri-/quadrigrams and directed letter pairs weighted by \`exp(-gap/τ)\` | Intermediate orthographic combination sensitivity |
| VWFA-011 | C · 11. Anterior VWFA | Nonlinear integration of posterior and sublexical information | More abstract higher-order orthographic representations anteriorly |
| VWFA-012 | C · 12. Invariance | Cosine similarity across supplied visual views, Unicode/case normalization, explicit order preservation | Experience-dependent tolerance to fonts/forms without indiscriminate mirror invariance |
| VWFA-013 | D · 13. Whole words | Ordered edit similarity plus \`λ log(1+frequency)\` and softmax candidate competition | Familiar whole-word form activation |
| VWFA-014 | D · 14. Morphology | Bounded substring/boundary search, log-frequency score, \`P(M|O) ∝ P(O|M)P(M)\` | Sensitivity to recurring morphemic orthographic form |
| VWFA-015 | D · 15. Parallel activation | Separate whole-word, stem, prefix, suffix and inflection population activities | Co-activation of whole-word and constituent-form representations |
| VWFA-016 | D · 16. Lexical/semantic association | Retrieve supplied feature maps; weighted vector composition and grammatical-feature union | Distributed temporal/frontal language network; not VWFA-only semantics |
| VWFA-017 | D · 17. Stabilization | Recurrent belief update, context fit, softmax competition, confidence/margin and convergence | Interactive visual-language inference and recurrent settling |
| VWFA-018 | E · 18. Bottom-up connectivity | Directed adjacency multiplication, weighted summation, nonlinear transfer and damping | Visual feedforward pathway toward ventral occipitotemporal cortex |
| VWFA-019 | E · 19. Language-network connectivity | Recurrent graph propagation and correlation across iterations | VWFA exchange with IFG, MTG, ATL, STG, angular gyrus and parietal systems |
| VWFA-020 | E · 20. Prediction/feedback | \`e=x−x̂\`, inverse-variance precision, precision-weighted feedback | Top-down language/context influence and bottom-up sensory correction |
| VWFA-021 | E · 21. Attention | Softmax attention, weighted pooling, task gain | Attention-related modulation of visual-language populations |
| VWFA-022 | F · 22. Initial associations | Deterministic orthographic feature hash, prediction error and \`ΔW=ηxᵀδ\` | Experience-dependent association between visual forms and supplied phonological targets |
| VWFA-023 | F · 23. Word selectivity | \`S_word=R_word−R_control\`, softmax/cross-entropy contrastive objective | Increasing word-sensitive population response with literacy |
| VWFA-024 | F · 24. Consolidation | Fast/slow exponential moving averages, frequency accumulation and pattern prototypes | Repeated-experience stabilization and long-term orthographic memory |

The stages interact recurrently; the registry is an inspectable execution order, not a claim that human reading is strictly serial.

## Mathematical reference

### Visual and population processing

For image intensity \`I(x,y)\`, local contrast normalization is approximated by

\[
C=I-G_\sigma*I, \qquad
I_n=\frac{C}{\sqrt{G_\sigma*(I^2)-(G_\sigma*I)^2+\epsilon}}.
\]

For each orientation/frequency pair, even and odd Gabor channels produce

\[
R_e=I*g_e,\quad R_o=I*g_o,\quad E=\sqrt{R_e^2+R_o^2+\epsilon}.
\]

Population gain control divides each channel by pooled local energy. A hierarchical population approximation uses

\[
\mathbf h^{(l)}=\tanh(W^{(l)}\mathbf h^{(l-1)}+\mathbf b^{(l)}).
\]

For supplied prototype \`k\`, the feature evidence is

\[
d_k^2=\sum_i\frac{(x_i-\mu_{ki})^2}{v_{ki}},\qquad
r_k=\exp(-d_k^2/2),\qquad
p_k=\operatorname{softmax}(\log(r_k+\epsilon)).
\]

### Orthographic and morphological processing

Ordered character similarity uses normalized edit distance. N-gram features preserve start offsets. A non-adjacent pair \`(i,j)\` receives a proximity factor \`K(j-i)=exp(-(j-i)/τ)\`. Whole-word candidate \`j\` receives

\[
S_j=\operatorname{sim}(O,T_j)+\lambda\log(1+F_j),\qquad
P(W_j\mid O)=\operatorname{softmax}(S_j/\tau).
\]

Morphology search enumerates bounded concatenations from the supplied inventory and ranks them using form compatibility and an exposure-frequency prior. The implementation does not automatically decide that a substring is meaningful without an inventory record. Whole-word and morphological activities are stored in separate channels and can coexist.

The semantic composition stage aggregates caller-supplied stem/affix feature vectors and grammatical tags. It does not assign meaning to characters or claim that VWFA alone computes semantics. Stabilization iteratively updates candidate beliefs from visual/lexical evidence, context fit, and recurrent activity, then checks the L2 change against a convergence tolerance.

### Cortical graph and predictive feedback

A directed region graph updates population state as

\[
\mathbf x_{t+1}=\tanh(A\mathbf x_t+B\mathbf u).
\]

Prediction residual and precision are

\[
\mathbf e=\mathbf x-\hat{\mathbf x},\qquad
\Pi=(\operatorname{Var}(\mathbf e)+\epsilon)^{-1}.
\]

Attention weights use softmax, and the attended vector is a weighted sum over compatible position features. The reported network correlation is estimated across recurrent update iterations, not interpreted as empirical fMRI/EEG connectivity.

### Literacy and neural dynamics

For an explicitly supplied pair of written form \`x\` and phonological target \`y\`, the functional association learner uses

\[
\hat y=xW,\quad\delta=y-\hat y,\quad
W\leftarrow W+\eta x^T\delta.
\]

Word selectivity is represented by the contrast between labelled word and control examples. Fast and slow prototype memories use

\[
\mu_{t+1}=(1-\eta)\mu_t+\eta x_t.
\]

The separate biological reference advances 16 named populations using a current-based leaky integrate-and-fire approximation:

\[
C_m\frac{dV}{dt}=g_L(E_L-V)+I_{\mathrm{syn}}+I_{\mathrm{external}}.
\]

The code uses forward-Euler integration, threshold/reset, refractory periods, recurrent population drive, and a bounded graph-weight update for authorized learning examples. It reports membrane voltage (mV), current (pA), firing rate (Hz), spikes, and simulated time (ms). These are simulated state variables, not measured participant signals.

## Python API

\`\`\`python
from ore_vwfa import (
    VWFAProcessor, VWFARequest, VWFAConfig,
    LexicalEntry, MorphemeEntry,
)

request = VWFARequest(
    sample_id="reading-001",
    orthographic_sequence=("c", "a", "t", "s"),  # upstream-recognized orthography
    lexicon=(LexicalEntry("cats", frequency=25.0, semantic_features={"ANIMAL": 1.0}),),
    morphology_inventory=(
        MorphemeEntry("cat", role="stem", frequency=15.0, semantic_features={"ANIMAL": 1.0}),
        MorphemeEntry("s", role="inflection", frequency=20.0,
                      semantic_features={"PLURAL": 1.0},
                      grammatical_features={"NUMBER_PLURAL": 1.0}),
    ),
    context_features={"ANIMAL": 0.7},
)

processor = VWFAProcessor(VWFAConfig(), register_annotations=True)
result = processor.process(request)
print(len(result.operation_trace))  # 24
print(result.functional["selected_candidate"])
print(result.biological["final_snapshot"]["populations"]["vwfa_anterior"])
\`\`\`

For image input, pass \`visual_image=...\`. To label image-derived regions, provide \`GraphemePrototype\` vectors whose dimensions match Stage B's pooled descriptors. To compare font/scale views, supply at least two compatible images in \`visual_views\`. The processor does not perform OCR or promise that connected image regions correspond one-to-one with characters.

\`VWFAExample\` records are accepted for literacy-learning experiments. No persistent parameter update occurs unless \`learning_authorized=True\`; examples need a consistent phonological target dimension. Fast/slow memories live in the processor instance and can be cleared with \`reset_learning_state()\`. The current implementation does not save learned weights to a permanent model file.

## SQLite annotation integration

Every module, class, method, helper, and nested function includes a descriptive Python docstring. ORE's existing AST catalogue indexes these docstrings into the shared \`annotations\` table:

\`\`\`python
from ore_vwfa import register_vwfa_annotations

report = register_vwfa_annotations()
print(report["database_path"])
\`\`\`

The default database is \`ore_logs/ore_logger.sqlite3\`. \`ORE_LOGGER_DB\` or the \`database_path\` argument can select another SQLite file. The SQLite database is a local runtime artifact and is not committed to Git; source docstrings are version-controlled, and the logger can rebuild or refresh the catalogue. When \`VWFAProcessor\` is created, annotation registration is enabled by default; isolated tests can use \`register_annotations=False\`.

## Run and test

\`\`\`bash
python -m pip install -e ".[dev]"
python -m ore_vwfa
python -m pytest tests/test_vwfa.py -q
python -m pytest
\`\`\`

## Limitations and validation requirements

- The model does not perform learned OCR, automatic character segmentation, or script-specific decoding. It operates on image-derived candidate regions or caller-supplied orthography.
- Grapheme prototypes, lexical entries, morpheme inventories, semantic attributes, and phonological targets require provenance and caller verification.
- The current concatenative morphology search is a bounded reference algorithm. It is not a universal morphological analyser and does not model all spelling alternations or non-concatenative morphology.
- Graph edges and population labels are simplified hypotheses, not a diffusion MRI connectome or a precise anatomical localization of VWFA subregions.
- LIF outputs are computationally simulated, not recordings. Parameters need empirical fitting before physiological interpretation.
- Stage F learning is opt-in, instance-local, and not yet a complete developmental model of childhood literacy acquisition.
- Software tests establish code contracts and numerical sanity only. Biological validity requires preregistered comparisons with independent human reading, behavioral, EEG/MEG/fMRI, developmental, and lesion datasets, plus uncertainty estimates and baseline comparisons.
