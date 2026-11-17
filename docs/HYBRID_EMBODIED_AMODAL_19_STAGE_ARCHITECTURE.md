# Hybrid Embodied–Amodal Concept Formation and Processing — 19 stages

**Package:** `ore_hybrid_embodied_amodal/`  
**Registry:** `ORE-HEC-19`  
**Source specification:** Google Drive file `Hybrid Embodied–Amodal Concept Formation and Processing.txt`  
**Implementation:** functional computational pipeline plus a separate optional current-based LIF reference  
**Status:** research-oriented software approximation; not a biologically validated model

## 1. Architectural principle

This package implements the specification's hybrid position. It does not force a choice between fully grounded concepts and entirely amodal symbols.

- **Embodied / modality-specific branch:** visual, auditory, motor, somatosensory, olfactory, gustatory, affective, interoceptive, linguistic, and internally generated evidence remain distributed in modality-specific representations.
- **Shared / amodal branch:** a bilateral ATL-inspired convergence state represents information common across routes and supports cross-modal similarity, category abstraction, and conceptual generalization.
- **Bidirectional interaction:** spoke evidence projects toward the hub; the hub can reconstruct and reactivate modality-specific representations; prediction errors can refine the working conceptual state.
- **Preservation:** shared information does not replace the spokes. Modality-specific residuals are explicitly returned so visual detail, sound, movement, touch, and other embodied properties remain available.

The 19-stage sequence is a traceable computational decomposition. It is not a claim that biological cognition follows a strict serial schedule; real brain processing is highly parallel and recurrent.

## 2. The 19-stage registry

The canonical registry in `ore_hybrid_embodied_amodal/registry.py` uses stage IDs `HEC-01.01`–`HEC-01.19`. Each stage contains ordered substeps with stable IDs such as `HEC-01.07.03`. Each substep carries a functional responsibility, a mathematical expression, and a stage-level biological reference.

| Stage | Name | Main functional work | Biological reference |
|---:|---|---|---|
| 1 | Environmental and internal stimulus acquisition | Accept external and internally generated modality streams; estimate signal statistics and normalize | Retina, visual thalamic pathways/cortex, cochlea/auditory pathways/cortex, somatosensory and motor systems, olfactory/gustatory pathways, amygdala, insula/interoception |
| 2 | Parallel distribution to modality-specific processing networks | Route each available stream to its own spoke; retain recurrent exchange | Cortical/subcortical populations, feedforward and feedback pathways, association pathways, long-range white matter |
| 3 | Modality-specific feature extraction | Extract graded modality features and hierarchical combinations | Selective visual, auditory, motor, somatosensory, affective, linguistic populations; association cortices |
| 4 | Attention-dependent feature selection | Compute goal/context gates and suppress caller-marked distractors | Frontal/parietal control, sensory association cortex, thalamic pathways, neuromodulatory systems |
| 5 | Construction of modality-specific semantic representations | Build rich within-modality semantic representations and feature interactions | Visual/auditory association cortex, premotor/motor, somatosensory association, language-related temporal networks |
| 6 | Activation of previously acquired conceptual knowledge | Compare current activity with supplied concept prototypes and spread supported activation | Distributed associative memory, recurrent modality/association interactions, ATL participation |
| 7 | Temporal alignment and cross-modal binding | Combine temporal proximity, spatial compatibility, and learned association evidence | Sensory and temporal association networks, attention and convergence systems |
| 8 | Feedforward transmission to the ATL semantic hub | Project each modality into a common representational dimension | Bilateral ATL; graded connectivity, with ventrolateral ATL proposed as a strong transmodal contributor |
| 9 | Reliability estimation and weighted modality integration | Estimate uncertainty, compute normalized modality weights, combine projected inputs | Reliability-sensitive influence shaped by signal quality, attention, task, activity, and connectivity |
| 10 | Cross-modal integration within the ATL | Nonlinear convergence and recurrent/leaky stabilization of the shared state | Interconnected ATL populations receiving distributed cortical input |
| 11 | Extraction of modality-independent conceptual information | Estimate common conceptual state, route similarity, invariance error, and unique information | ATL transmodal convergence plus continued spoke contributions |
| 12 | Preservation of embodied and modality-specific information | Reconstruct each modality from the common state, calculate residuals, and combine according to task | Visual/auditory association, premotor, somatosensory, affective/interoceptive systems |
| 13 | Conceptual categorization and semantic generalization | Build category prototypes, score membership, retain between-concept differences | ATL hub plus detailed distinctions supplied by distributed spokes |
| 14 | Context-dependent conceptual abstraction | Encode context, adapt the hub, and weight task-relevant features | ATL, frontal semantic control, temporoparietal, linguistic and episodic memory networks |
| 15 | Top-down activation of embodied representations | Project the adapted hub into modality spokes and gate partial reactivation | Visual/auditory association, premotor/motor, somatosensory, emotional/introspective systems |
| 16 | Predictive reconstruction and completion of missing information | Predict sensory features, calculate residual errors, update the hub, and estimate contextual posterior | Recurrent higher-order conceptual, sensory association, predictive and attentional systems |
| 17 | Semantic decision and response preparation | Score candidate responses, select or abstain, suppress alternatives, prepare response | Inferior frontal semantic control, temporal semantic networks, language production, premotor/motor |
| 18 | Feedback from the result to the conceptual system | Compare target and actual evidence, adjust representation/attention, schedule reprocessing | Recurrent semantic, control, memory, sensory and motor interactions |
| 19 | Learning, synaptic modification, and conceptual development | Propose or authorize updates from verified experiences; stabilize and integrate associations | Possible LTP/LTD, synaptic efficacy and structural plasticity, E/I modulation, neuromodulation, consolidation |

The registry is the programmatic source of truth:

```python
from ore_hybrid_embodied_amodal import STAGE_REGISTRY, ALL_OPERATIONS, registry_as_dicts

assert len(STAGE_REGISTRY) == 19
print(len(ALL_OPERATIONS))
print(registry_as_dicts()[6])  # Stage 7 and its ordered substeps
```

## 3. Mathematical operations

The equations below are computational forms of the source's proposals. They are not claims that the brain explicitly executes Python, matrix operations, or these exact loss functions.

### Stages 1–5: sensory input, routing, feature construction, and selection

1. **Transduction and normalization**
   \[
   X_m(t)=F_m(I_m(t)), \qquad
   X'_m=\frac{X_m-\mu_m}{\sigma_m+\epsilon}
   \]
   The code takes finite named feature maps from an upstream front end, estimates within-channel mean/standard deviation, and retains constant-valued channels without dividing by zero.

2. **Parallel distribution and recurrent communication**
   \[
   Y_j=\phi\left(\sum_i W_{ij}X_i+b_j\right), \qquad
   X_m(t+1)=\phi(W_{mm}X_m(t)+W_{hm}H(t))
   \]
   The software preserves separate spoke states. Higher-level feedback is implemented explicitly in Stages 15–18, rather than fabricating a sensory input in this stage.

3. **Feature extraction and hierarchy**
   \[
   S_m=\phi(W_m^{(1)}X_m+b_m^{(1)}),\quad
   S_m^{(k+1)}=\phi(W_m^{(k+1)}S_m^{(k)}+b_m^{(k+1)})
   \]
   Feature similarity is measured with cosine similarity:
   \[
   \operatorname{sim}(a,b)=\frac{a\cdot b}{\|a\|\|b\|+\epsilon}
   \]

4. **Attention and inhibition**
   \[
   A_m=\sigma(W_qQ+W_cC+b),\quad
   \widetilde S_m=A_m\odot S_m,\quad
   S_m^{selected}=A_m\odot S_m-\lambda I_m
   \]
   The prototype makes the evidence explicit: task and context feature maps create gates, while the caller identifies distractor features. Missing task/context information does not create a hidden task-specific target.

5. **Modality-specific semantic construction**
   \[
   Z_m=\phi(W_m^{(2)}\widetilde S_m+b_m^{(2)}),\qquad
   Z_m=\phi(W_1S_m+W_2(S_m\otimes S_m))
   \]
   The code uses bounded nonlinear transforms and feature conjunctions as engineering approximations.

### Stages 6–10: memory, binding, ATL projection, reliability, integration

6. **Associative retrieval**
   \[
   A_i=\operatorname{sim}(Z_m,P_i),\quad
   A'_i=A_i+\alpha C_i+\beta F_i,\quad
   a^{(t+1)}=\phi(Wa^{(t)}+x)
   \]
   Stored concept prototypes are the explicit memory inventory. A learned-memory extension is kept separate and is updated only in the authorized learning transaction.

7. **Cross-modal binding**
   \[
   B_{ij}=e^{-|t_i-t_j|/\tau},\quad
   B_s=\operatorname{sim}(P_i,P_j),\quad
   B^*_{ij}=\alpha B_{ij}+\beta B_s+\gamma B_{learned}
   \]
   \[
   \Delta W_{ij}=\eta x_ix_j
   \]
   Temporal values, spatial evidence and learned priors are separate request fields. The Hebbian term is proposed/updated only with authorized learning.

8. **Projection into the hub space**
   \[
   U_m=P_mZ_m,\qquad U_m\in\mathbb R^d
   \]
   The common dimension \(d\) is a configurable software setting, not a measured number of ATL neurons.

9. **Reliability-weighted integration**
   \[
   R_m=\frac{1}{\sigma_m^2+\epsilon},\quad
   w_m=\frac{R_m}{\sum_jR_j},\quad \sum_mw_m=1,\quad
   U_{combined}=\sum_mw_mU_m
   \]
   An additional task-relevance gate can modulate reliability in logit space. Only modalities supplied in the request receive a weight.

10. **ATL nonlinear and recurrent convergence**
    \[
    H_{input}=\sum_mw_mU_m
    \]
    \[
    H^{(t+1)}=\phi(W_{in}H_{input}+W_{rec}H^{(t)}+b)
    \]
    \[
    H^{(t+1)}=(1-\lambda)H^{(t)}+\lambda H_{new}
    \]
    The implementation uses a deterministic bounded recurrent matrix, a leaky update, and a convergence-delta trace.

### Stages 11–15: shared information, residuals, category, context, reactivation

11. **Shared conceptual state and invariance diagnostic**
    \[
    H_{common}=\sum_mw_mH_m,\quad
    S_{ij}=\frac{H_i\cdot H_j}{\|H_i\|\|H_j\|+\epsilon}
    \]
    \[
    L_{invariance}=\sum_{i\ne j}\|H_i-H_j\|^2
    \]
    Similarity and invariance error are measured and reported; the system does not force exact identity across routes.

12. **Embodied residual preservation**
    \[
    \widehat Z_m=\phi(D_mH_{common}+b_m),\quad
    R_m=Z_m-\widehat Z_m,\quad
    Z_m^*=\alpha H_{common}+\beta R_m
    \]
    In software the decoded common component and residual are combined in the modality feature space, avoiding an invalid addition of vectors with different dimensions. Task-specific gates can emphasize visual detail for imagery or motor/somatosensory detail for action.

13. **Category prototype and generalization**
    \[
    P_c=\frac{1}{N}\sum_{i=1}^{N}H_i,\quad
    S_c=\operatorname{sim}(H,P_c),\quad
    P(c\mid H)=\frac{e^{\beta S_c}}{\sum_je^{\beta S_j}}
    \]
    \[
    D(H_i,H_j)=\|H_i-H_j\|,\quad
    \widehat c=\arg\max_cP(c\mid H)
    \]
    Category membership is based on supplied concept/category prototypes; the code does not invent a taxonomy.

14. **Contextual abstraction**
    \[
    C=\phi(W_cX_c+b_c),\quad H_c=\phi(W_hH+W'_cC)
    \]
    \[
    A_i=\operatorname{softmax}(\beta S_i+\gamma C_i),\quad
    H_{adapted}=\sum_iA_iF_i
    \]
    Context changes the features used for current processing; it need not change the stored identity of the concept.

15. **Top-down embodied reactivation**
    \[
    \widehat Z_m=\phi(V_mH_{adapted}+b_m),\quad
    \widehat Z_m^*=G_m\odot\widehat Z_m,\quad
    \widehat Z_m^*=\rho_m\widehat Z_m
    \]
    with \(0\leq\rho_m\leq1\). The task gates determine which spokes are reactivated and how strongly.

### Stages 16–19: prediction, response, feedback, learning

16. **Predictive reconstruction**
    \[
    \widehat X_m=F_m(H,C),\quad E_m=X_m-\widehat X_m,\quad
    H_{t+1}=H_t+\eta W_eE_m
    \]
    \[
    P(H\mid X,C)\propto P(X\mid H)P(H\mid C)
    \]
    The implementation reports channel-wise prediction residuals and precision-weights their contribution to a bounded hub update.

17. **Response selection**
    \[
    S_i=w_1R_i+w_2C_i+w_3T_i,\quad
    i^*=\arg\max_iS_i,\quad
    A_i^*=A_i-\lambda A_{competitors}
    \]
    \[
    Y=\phi(W_oH_{adapted}+b_o)
    \]
    Candidate alternatives remain visible. Selection is gated by confidence and the top-candidate margin; otherwise the model abstains.

18. **Output feedback**
    \[
    E_o=Y_{target}-Y_{actual},\quad
    H_{new}=H_{old}+\eta W_oE_o,\quad
    A_m^{new}=A_m^{old}+\eta_A\Delta A_m
    \]
    When target evidence is absent, the system does not invent a target. Low confidence produces explicit requests for additional feature extraction, memory retrieval, context analysis, or cross-modal integration.

19. **Experience-dependent modification**
    \[
    \Delta W_{ij}=\eta x_iy_j,\quad
    \Delta W_{ij}=\eta x_iy_j-\lambda W_{ij}
    \]
    \[
    \Delta W=-\eta\frac{\partial L}{\partial W},\quad
    H_{updated}=\frac{\sum_i\alpha_iH_i}{\sum_i\alpha_i},\quad
    W(t)=W_0e^{-\lambda t}
    \]
    The gradient expression is an artificial-network analogy, not evidence that neurons calculate a global backpropagation gradient. The code previews learning by default and commits only when explicitly authorized with non-empty reviewer/source provenance.

## 4. Python API

### Build a concept and process multimodal evidence

Inputs to this package are **named feature maps**, not raw pixels or raw audio. Raw acquisition/transduction and low-level feature extraction belong to the upstream perception stages.

```python
from ore_hybrid_embodied_amodal import (
    HybridConcept,
    HybridEmbodiedAmodalProcessor,
    HybridRequest,
)

concept = HybridConcept(
    concept_id="guitar",
    label="guitar",
    aliases=("guitar",),
    category="musical instrument",
    features={"instrument": 1.0, "strings": 0.9, "music": 0.8},
    modality_features={
        "visual": {"strings": 0.9, "neck": 0.8},
        "auditory": {"music": 0.9, "sound": 0.8},
        "motor": {"strumming": 0.9},
    },
    verified_by="reviewer",
    source_reference="dataset://concepts/guitar",
)

processor = HybridEmbodiedAmodalProcessor([concept])
request = HybridRequest(
    sequence_id="guitar-001",
    modality_inputs={
        "visual": {"strings": 0.9, "neck": 0.8},
        "auditory": {"music": 0.9, "sound": 0.8},
        "motor": {"strumming": 0.9},
    },
    task="recognition",
    context_features={"instrument": 0.9, "music": 0.8},
    task_features={"instrument": 1.0},
    modality_times={"visual": 0.0, "auditory": 0.05, "motor": 0.08},
    response_mode="recognition",
)
result = processor.process(request)

print(result.selected_concept)
print(len(result.stage_trace))       # 19
print(len(result.operation_trace))   # all registered substeps
print(result.modality_residuals)     # modality-specific unique information
print(result.common_representation)  # shared ATL-inspired state
```

Existing ORE `SemanticConcept` objects can be supplied directly; the processor adapts their `semantic_features`, `modality_features`, aliases, category, frequency and provenance fields without changing the old package contract.

### Explicit experience learning

```python
from ore_hybrid_embodied_amodal import HybridLearningExample

experience = HybridLearningExample(
    concept_id="guitar",
    modality_inputs={
        "visual": {"strings": 1.0, "neck": 0.9},
        "auditory": {"music": 0.95},
    },
    shared_features={"instrument": 1.0, "music": 0.9},
    confidence=0.95,
    verified_by="independent-reviewer",
    source_reference="dataset://verified/guitar-episode-001",
)

learning_request = HybridRequest(
    sequence_id="guitar-learning-001",
    modality_inputs={"visual": {"strings": 1.0}},
    learning_examples=(experience,),
    learning_authorized=True,
)
learned = processor.process(learning_request)
print(learned.learning["status"])  # committed, only with valid authorization/provenance
```

Without `learning_authorized=True`, updates are preview-only. Examples without reviewer and source provenance are blocked. The execution logger does not authorize or persist semantic learning.

### Configure biological reference

```python
from ore_hybrid_embodied_amodal import HybridConfig, HybridEmbodiedAmodalProcessor

config = HybridConfig(
    simulate_biology=True,
    population_size=8,
    steps_per_stage=2,
    dt_ms=1.0,
)
processor = HybridEmbodiedAmodalProcessor([concept], config)
result = processor.process(request)
print(result.biological["layers"]["atl_transmodal_hub"])
```

The LIF reference reports membrane voltage (mV), synaptic current (pA), firing rate (Hz), spike counts, and simulated time (ms). The dimensional conversion \(1\,M\Omega \times 1\,pA=10^{-3}\,mV\) is explicit in code. This is a generic population reference, not a fitted simulation of the named anatomical systems. Disable it with `HybridConfig(simulate_biology=False)`.

## 5. SQLite annotations and operation/run trace

The package integrates with the existing repository-wide ORE logger. It does not introduce a separate database or table format.

```python
from ore_hybrid_embodied_amodal import (
    index_hybrid_annotations,
    record_hybrid_result,
)

# Index every module, class, function and method docstring in this package.
annotation_report = index_hybrid_annotations(repository_root=".")
print(annotation_report["hybrid_symbol_count"])

# Record each mathematical/functional/biological sub-operation and all 19 stages.
run_report = record_hybrid_result(result, repository_root=".")
print(run_report["run_id"])
print(run_report["operation_count"])
```

Default database:

```text
ore_logs/ore_logger.sqlite3
```

The existing AST catalogue stores each class/function/method explanation once in `annotations`; run events link to symbol annotations through foreign keys. The new package supplies docstrings for its classes and callables, plus explicit operation-level explanations with equations and biological references. Raw input feature values and full neural arrays are not written to run-event metadata.

Run the built-in demonstration from the repository root:

```bash
python -m ore_hybrid_embodied_amodal
```

## 6. Tests and validation

Install and test:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_hybrid_embodied_amodal.py -q
python -m pytest
```

The focused tests cover:
- exact 19-stage order and unique operation IDs;
- end-to-end functional execution and distinct shared/embodied outputs;
- normalized reliability weights and temporal binding;
- task attention, inhibition, and contextual output;
- preview-only and provenance-gated learning;
- optional LIF units, numerical validity, and stage trace;
- compatibility with the existing `SemanticConcept` contract;
- SQLite symbol annotations and operation/run events;
- invalid/non-finite input rejection.

Software tests demonstrate code behavior only. Behavioral validity requires held-out human concept-recognition, cross-modal generalization, and semantic-control experiments. Biological validity requires independent, task-matched neural measurements, pre-registered predictions, held-out evaluation, and replication.

## 7. Scientific scope and limits

- This package is a transparent computational approximation of the supplied hybrid theory.
- Neural region assignments are broad biological references, not exclusive one-region/one-function mappings.
- The exact 19-stage serial ordering is for software traceability; neural processing is parallel and recurrent.
- The ATL hub dimensionality and network weights are engineering settings, not measured biological quantities.
- LIF population rates and voltage are simulated outputs, not real recordings.
- The feature map front end is external. This package does not perform raw image recognition, cochlear transduction, speech recognition, or chemical sensing.
- The common representation is not a claim that routes produce identical neural patterns. Cross-modal overlap and route-specific residuals are separately exposed.
- Gradient descent in Stage 19 is explicitly an artificial-network analogy; biological plasticity is more complex.
- No behavior or neural evidence is fabricated. Empirical validation status remains open until independent datasets are supplied.
