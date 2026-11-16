# ORE Hub-and-Spoke Semantic Concept Integration — 21-stage implementation

**Package:** \`ore_hub_spoke/\`  
**Registry:** \`ORE-HUB-SPOKE-21\`  
**Status:** executable functional reference with an optional current-based LIF population reference  
**Source specification:** Google Drive file \`Master Algorithm - Hub-and-Spoke Semantic Concept Integration.txt\` (retrieved 2026-11-16)

## 1. Purpose and architectural boundary

This package implements the source algorithm as an additive subsystem in ORE-Neurocomputational. It does not replace \`ore_semantic\` (the existing 44-operation concept-access subsystem), \`ore_reading\`, or the N1–N9 packages. It reuses the existing \`ore_semantic.SemanticConcept\` and \`WordFormEpisode\` contracts so existing concept inventories can be passed into the new pipeline.

The new package makes the source document's 21-stage sequence explicit, including the portions that are not just concept recognition. It consumes **pre-extracted, named feature maps**; raw image pixels, audio waveforms, video, and sensor streams must be processed by upstream perception modules before they are supplied as spoke evidence. This keeps the semantic integration layer separate from low-level sensory transduction:

- concurrent modality-specific input and seven semantic spokes;
- associative memory retrieval and spoke-to-hub projections;
- reliability-weighted nonlinear hub integration and recurrent settling;
- feedback from the hub to the spokes;
- modality-invariant and modality-specific representations;
- concept identification, category relations, and semantic control;
- experience acquisition and authorized plasticity;
- cross-modal alignment and incremental knowledge updates;
- separate hub, spoke, and connection lesion simulations.

The functional and biological-reference tracks are separate. Functional feature vectors and concept decisions are not reported as neural measurements. The LIF track reports a generic population simulation with explicit physical units. Neither software tests nor a biologically motivated design establish that the brain performs these exact 21 operations, uses these equations, or follows this exact order.

## 2. The 21-stage execution registry

All stages have stable identifiers \`HS-01\` through \`HS-21\`. Each stage is decomposed into individually traceable sub-operations with stable IDs such as \`HS-07.3\`. The canonical source of names, equations/approximations, responsibilities, and biological reference text is \`ore_hub_spoke/registry.py\`.

| ID | Stage | Executable purpose |
|---|---|---|
| HS-01 | Receive conceptual information | Validate and encode available modality channels; preserve simultaneous evidence and its source |
| HS-02 | Activate modality-specific spokes | Route inputs to visual, auditory, tactile, olfactory/gustatory, spatial/motor, linguistic, and emotional/interoceptive feature populations |
| HS-03 | Extract individual semantic features | Retain graded feature activation, threshold active features, and construct feature-association matrices |
| HS-04 | Retrieve previously acquired semantic information | Retrieve concept/episode patterns, reactivate supported features, blend current evidence and memory, and retain retrieval confidence |
| HS-05 | Transmit spoke information toward the hub | Project each modality vector into a common hub dimensionality using its own connection matrix |
| HS-06 | Estimate and combine modality contributions | Estimate channel reliability, normalize gates, and preserve cross-modal disagreement |
| HS-07 | Integrate features in the amodal semantic hub | Apply nonlinear transformations to bind features and represent shared conceptual structure |
| HS-08 | Stabilize the integrated concept | Run bounded recurrent updates and monitor the norm of the state change |
| HS-09 | Send integrated information back to spokes | Predict modality-specific feature patterns, compute prediction error, and refine working spoke states |
| HS-10 | Develop a modality-independent representation | Estimate common cross-modal structure while preserving modality-specific residuals |
| HS-11 | Identify the concept and retrieve its identity | Compare the hub state with concept templates, rank candidates, and estimate confidence/margin |
| HS-12 | Construct category relationships and generalize | Compute shared/distinctive features, category centroids, taxonomic membership, and explicitly supplied association edges |
| HS-13 | Apply semantic control | Boost task-relevant features and suppress caller-marked distractors without changing stored concept knowledge |
| HS-14 | Generate the required conceptual output | Produce recognition, description, comparison-ready output, or an explicit abstention |
| HS-15 | Acquire information across experiences | Validate sourced episodes and estimate confidence-weighted feature evidence |
| HS-16 | Learn spoke-to-hub mappings | Calculate prediction error and squared loss; propose gradient-style and Hebbian mapping updates |
| HS-17 | Learn modality-invariant representations | Align same-concept route projections while retaining route-specific residuals |
| HS-18 | Continuously update semantic knowledge | Commit bounded updates only after explicit authorization; regularize toward prior weights |
| HS-19 | Simulate degradation of the semantic hub | Attenuate selected hub dimensions and measure changes in candidate specificity |
| HS-20 | Simulate damage to individual semantic spokes | Attenuate each spoke in turn, recompute the hub, and measure selective effects |
| HS-21 | Simulate damage to hub-spoke connections | Attenuate each projection matrix in turn while retaining the endpoint populations |

### Important distinctions kept in the implementation

**Missing is not negative.** A modality absent from the request is not assigned a negative semantic value. It receives no current-input gate unless supported by a retrieved memory cue.

**Observed and inferred information remain separable.** Stage 09 keeps original input evidence, hub-predicted spoke patterns, prediction errors, and feedback-refined working vectors as different values. Feedback does not overwrite the request.

**A hub is not a simple average.** Stage 07 uses a learned nonlinear two-layer transform, and Stage 08 uses bounded recurrent refinement. The reliability-weighted spoke pool is an input to this transformation, not the final concept representation.

**Modality invariance does not delete modality detail.** Stage 10 returns a shared hub state and route-specific residuals. Similarity is measured, not imposed as exact equality.

**Relations are provenance-aware.** Category centroids are computed from the supplied \`prototype_id\`/lexical category inventory. Association edges are emitted only from the optional caller-supplied \`associations\` mapping; the package does not invent factual relations.

**Lesions are counterfactual experiments.** Stages 19–21 do not mutate the processor's concept inventory or active projection weights. They create temporary degraded states and compare their outputs with a baseline.

## 3. Mathematical implementation

The formulas below describe the implemented computational approximations. Parameters are software settings, not measured human-neural constants.

### 3.1 Spoke activation and feature relations

For modality \(m\), the input feature vector \(x_m\) is encoded in a stable feature vocabulary. The input transform and logistic activation are:

\[
s_m = \sigma(W_{\mathrm{input},m}x_m + b_m)
\]

The current implementation initializes the input transform as a scaled identity, then retains its graded activity. A threshold \(\theta\) creates an active-feature readout while keeping the original magnitude:

\[
f_{m,i} = \mathbb{1}[s_{m,i}>\theta]
\]

Within-spoke concept co-activation produces a normalized relation matrix:

\[
R_m = \operatorname{Normalize}(X_m^\top X_m), \quad R_{m,ii}=0
\]

This is a corpus/inventory co-occurrence proxy constructed from the configured concept entries. It is not a measured neural synaptic matrix.

### 3.2 Associative retrieval and current-memory integration

Candidate concepts are retrieved from normalized labels/aliases and supplied episodic word-form traces. Feature memory is combined with current spoke activity:

\[
s_m^{\mathrm{combined}} =
\alpha_m s_m^{\mathrm{current}} +
(1-\alpha_m)s_m^{\mathrm{memory}}
\]

The coefficient \(\alpha_m\) depends on the amount of current evidence. If current evidence is absent, the model can use a supported retrieved pattern; if neither current nor retrieved evidence exists, it does not fabricate an observed feature.

### 3.3 Spoke-to-hub transmission and reliability gates

Each spoke has its own projection:

\[
u_m = W_m s_m
\]

A reliability score \(q_m\) is derived from current feature coverage and retrieved-memory confidence. The normalized modality gate is:

\[
g_m = \frac{\exp(q_m/T)}{\sum_j \exp(q_j/T)}
\]

Inactive channels receive a strongly suppressed score; when no spoke is active, all gates remain zero. Cross-modal disagreement is retained as pairwise distance between route projections.

### 3.4 Nonlinear integration and recurrent settling

The first integration stage forms a weighted drive:

\[
U = \sum_m g_m u_m
\]

The hub is a two-layer nonlinear transform:

\[
z = \tanh(W_1U+b_1), \qquad h_0=\tanh(W_2z+b_2)
\]

It is then iteratively stabilized:

\[
h_{t+1} =
\tanh\left(\gamma W_{hh}h_t +(1-\gamma)h_0\right)
\]

The loop stops when:

\[
\|h_{t+1}-h_t\|_2 < \epsilon
\]

or when the configured maximum number of iterations is reached. The trace reports iteration count, convergence flag, and the full delta history.

### 3.5 Hub feedback, cross-modal alignment, and concept matching

A feedback matrix predicts each spoke from the hub:

\[
\hat{s}_m = \sigma(P_m h), \qquad e_m=s_m-\hat{s}_m
\]

The working spoke state uses a bounded interpolation:

\[
s_m^{\mathrm{refined}} =
(1-\beta)s_m+\beta\hat{s}_m
\]

The observed value and prediction error remain in the result. For route-specific hub states \(h_m\), a shared representation and residuals are calculated:

\[
h_{\mathrm{shared}} =
\frac{\sum_m g_m h_m}{\sum_m g_m}, \qquad
r_m=h_m-h_{\mathrm{shared}}
\]

A concept candidate is scored from multiple visible channels, including hub-template cosine fit, lexical fit, modality fit, contextual fit, and episodic fit. The weighted score is normalized with softmax for candidate posteriors. An explicit confidence and top-candidate margin gate controls selection; alternatives are retained when the system abstains.

### 3.6 Category prototypes and semantic control

For a category \(C\), the current prototype feature value is the arithmetic centroid across supplied member concepts:

\[
P_C(f)=\frac{1}{|C|}\sum_{c\in C} f_c
\]

The task-control stage applies caller-supplied relevance and irrelevance weights. It also recalculates modality gates from the task-relevance signal while retaining the unmodified bottom-up gates for comparison:

\[
a_i' = \operatorname{clip}
\left(a_i+\lambda R_i-\mu I_i,0,1\right)
\]

This is intentionally not a hard-coded claim about the meaning of a task. The caller supplies feature relevance/distractor weights; unspecified features are not assigned arbitrary task priorities.

### 3.7 Experience-dependent learning

A \`LearningExperience\` binds a known concept to one or more feature maps, confidence, reviewer, and source reference. The system first estimates target hub states and computes a prediction error:

\[
e_m = h_{\mathrm{target}} - W_mx_m,\qquad
L_m=\frac{1}{2}\|e_m\|_2^2
\]

The proposed mapping update combines a gradient-style error term with a co-activation/Hebbian term:

\[
\Delta W_m =
\eta(e_mx_m^\top) +
\eta_H(h_{\mathrm{target}}x_m^\top)
\]

For same-concept modality pairs, cross-modal alignment uses route distance:

\[
L_{\mathrm{cross}} =
\sum_{i,j} y_{ij}\|h_i-h_j\|_2^2
\]

The implementation computes an update proposal during stages 16–17. Stage 18 commits only when \`learning_authorized=True\` and experiences are supplied. A prior-weight anchor penalty and bounded clipping limit drift. This is a controlled computational learning rule; it is not evidence that biological learning uses backpropagation.

### 3.8 Hub, spoke, and connection lesions

The hub lesion applies a diagonal degradation mask:

\[
h'=D_hh
\]

The spoke lesion scales one spoke at a time:

\[
s_m'=D_ms_m
\]

The connection lesion scales one projection at a time:

\[
W_m'=D_mW_m,\qquad u_m'=W_m's_m
\]

The output reports baseline and impaired candidate scores/margins, affected dimensions or pathways, and whether the remaining routes compensate. These are software lesion experiments, not diagnostic criteria or simulations of a specific neurological disease.

## 4. Biological-reference track

The optional \`HubSpokeLIFSimulator\` runs populations corresponding to the seven spokes, distributed semantic features, ATL-like hub, semantic control, episodic memory, and output systems.

For neuron \(i\), the forward-Euler update is:

\[
V_i(t+\Delta t)=V_i(t)+
\frac{\Delta t}{\tau_m}
\left[V_{\mathrm{rest}}-V_i(t)+R_mI_i(t)\right]
\]

The code uses \(R_m\) in MΩ and \(I\) in pA; because \(1\,\mathrm{M\Omega}\times1\,\mathrm{pA}=10^{-3}\,\mathrm{mV}\), the current-to-voltage conversion factor is \(0.001\). The reference network adds bounded recurrent excitation and population-level inhibition, applies a threshold/reset, and enforces a refractory interval.

Every stage has a matching biological population snapshot. The biological trace reports:

- membrane voltage in mV;
- synaptic current in pA;
- mean firing rate in Hz;
- simulated time in ms;
- spike counts and activity fractions.

The network is a generic LIF reference, not a fitted or validated reproduction of the ATL, any sensory association cortex, semantic-control circuitry, or the whole human semantic system. Disable it with \`HubAndSpokeConfig(simulate_biology=False)\`; the 21 functional stages and sub-operation trace remain complete.

## 5. Python API

### Concept inventory

The package consumes the existing provenance-tracked \`SemanticConcept\` contract:

\`\`\`python
from ore_semantic.contracts import SemanticConcept
from ore_hub_spoke import (
    HubAndSpokeConfig,
    HubAndSpokeProcessor,
    HubAndSpokeRequest,
)

concepts = [
    SemanticConcept(
        concept_id="concept-giraffe",
        label="giraffe",
        aliases=("giraffe",),
        semantic_features={
            "ANIMAL": 1.0, "MAMMAL": 0.9, "LONG_NECK": 0.9, "SPOTTED": 0.8
        },
        modality_features={
            "visual": {"LONG_NECK": 1.0, "SPOTTED": 0.9},
            "auditory": {"LOW_CALL": 0.6},
            "spatial_motor": {"BROWSE_HIGH_LEAVES": 0.9},
        },
        lexical_category="noun",
        prototype_id="mammal",
        verified_by="reviewer",
        source_reference="dataset://concepts/giraffe",
    ),
]

processor = HubAndSpokeProcessor(
    concepts,
    config=HubAndSpokeConfig(simulate_biology=True),
)
result = processor.process(HubAndSpokeRequest(
    sequence_id="sample-001",
    text="giraffe",
    modality_inputs={"visual": {"LONG_NECK": 0.95, "SPOTTED": 0.8}},
    task="describe",
))
print(result.output)
print(len(result.stage_trace))       # 21
print(len(result.operation_trace))   # all registered sub-operations
print(result.biological["layers"]["atl_hub"]["mean_firing_rate_hz"])
\`\`\`

A sensory-only request is also supported; omit both \`text\` and \`tokens\` and provide at least one \`modality_inputs\` entry.

### Explicit learning

\`\`\`python
from ore_hub_spoke import LearningExperience

experience = LearningExperience(
    concept_id="concept-giraffe",
    modality_features={
        "visual": {"LONG_NECK": 1.0, "SPOTTED": 0.9},
        "auditory": {"LOW_CALL": 0.6},
    },
    semantic_features={"ANIMAL": 1.0, "MAMMAL": 0.9},
    confidence=0.95,
    verified_by="reviewer",
    source_reference="dataset://episodes/giraffe-001",
)
learned_result = processor.process(
    request,
    learning_experiences=[experience],
    learning_authorized=True,
)
print(learned_result.learning["stage_18"]["status"])  # committed
\`\`\`

Without \`learning_authorized=True\`, stages 15–17 compute and report proposals but stage 18 does not mutate mapping weights or experience memory. Recognition alone is read-only.

### Reuse an existing semantic request

\`\`\`python
from ore_semantic.contracts import SemanticRequest
from ore_hub_spoke import request_from_semantic_request

legacy_request = SemanticRequest(
    sequence_id="from-ore-semantic",
    text="giraffe",
    modality_evidence={"visual": {"LONG_NECK": 0.9}},
)
hub_request = request_from_semantic_request(legacy_request)
result = processor.process(hub_request)
\`\`\`

## 6. SQLite annotations and execution traces

All classes and functions in \`ore_hub_spoke\` have docstrings that the existing AST logger stores as symbol annotations. Explicitly index the repository with:

\`\`\`python
from ore_hub_spoke import index_hub_spoke_annotations

report = index_hub_spoke_annotations(
    repository_root=".",
    database_path="ore_logs/ore_logger.sqlite3",
)
print(report["hub_spoke_annotations"])
\`\`\`

The helper uses \`ore_logger.LoggerDatabase\` and \`ore_logger.sync_repository\`; it writes to the existing annotations table rather than introducing another database format.

To store one run and its stage/sub-operation trace:

\`\`\`python
from ore_hub_spoke import record_hub_spoke_result

run_metadata = record_hub_spoke_result(
    result,
    repository_root=".",
    database_path="ore_logs/ore_logger.sqlite3",
)
print(run_metadata["run_id"])
\`\`\`

The trace logger stores operation explanations, the associated mathematical operation, biological reference, stage ID, and execution status. It does not serialize raw user input feature values into the log events.

## 7. Validation

Run focused tests:

\`\`\`bash
python -m pytest tests/test_hub_spoke.py -q
\`\`\`

Run the full repository suite:

\`\`\`bash
python -m pip install -e ".[dev]"
python -m pytest
\`\`\`

The automated tests cover registry integrity, sensory-only input, missing-modality handling, end-to-end execution, feedback/invariance traces, authorized learning, lesion simulations, LIF units, compatibility with the existing semantic request contract, and SQLite annotations/events.

Acceptance evidence must remain separate:

- **Software integrity:** stage/operation ordering, finite numerical state, deterministic schema, authorization, and logger persistence can be tested.
- **Behavioral validity:** requires held-out human-labeled concept retrieval, cross-modal recognition, generalization, and semantic-control experiments.
- **Biological validity:** requires independently collected neural data, task-matched predictions, held-out comparisons, and replication.

Passing the tests establishes software behavior only. It does not establish human-like semantic accuracy or biological validity.
