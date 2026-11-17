# Language Comprehension Point 14 — Clinical and Real-World Functional Profiling

## Purpose and scientific boundary

Point 14 implements the source's clinical and real-world implications as a **task-specific research model**. It separates perceptual access, lexical and syntactic processing, semantic composition, working memory, discourse coherence, pragmatic inference, social-state inference, world knowledge, and executive control.

It is **not a diagnostic classifier**, a substitute for clinical assessment, or a method for inferring a person's abilities from a diagnosis or brain scan. Clinical-context labels are optional descriptive metadata. They never select a fixed deficit vector or determine the result.

The algorithm preserves heterogeneity across individuals and conditions. It distinguishes comprehension from production, literal meaning from inferred meaning, social from nonsocial inference, current measurements from historical estimates, and measured abilities from missing evidence.

## Install and use

    from ore_language_comprehension import (
        Point14Request,
        Point14ClinicalProfileProcessor,
        register_point14_annotations,
    )

    request = Point14Request(
        participant_id="research-participant-01",
        target_outcome="understanding-an-indirect-request",
        task_set=[
            {
                "task_id": "literal-1",
                "task_type": "literal",
                "meaning_level": "literal",
                "accuracy": 1.0,
                "demands": {"lexical_access": 0.2, "semantic_composition": 0.3},
            },
            {
                "task_id": "inference-1",
                "task_type": "inference",
                "meaning_level": "inference",
                "accuracy": 0.6,
                "inference_domain": "social",
                "speaker_knowledge_explicit": False,
                "cue_weights": {"prosody": 0.6, "context": 0.4},
                "demands": {"pragmatic_inference": 0.8, "working_memory": 0.5},
            },
        ],
        component_measures={
            "lexical_access": [0.9, 0.8],
            "working_memory": [0.6, 0.7],
        },
    )

    result = Point14ClinicalProfileProcessor().process(request)
    print(result.profile["literal_vs_inferred_meaning"])
    print(result.profile["functional_components"]["components"])
    print(result.profile["transparent_report"])

Every public class, function, and method has a docstring and can be indexed in the shared SQLite annotation catalogue:

    annotation_report = register_point14_annotations()

The registration uses the existing ore_logger.annotations table; it does not create a separate Point 14 database.

## The 19-step algorithm

| Step | Functional operation | Mathematical / algorithmic operation | Possible biological reference |
|---:|---|---|---|
| 1 | Define the exact observable outcome | Binary/ordinal/continuous coding; response-time distributions; error categories | Task-set and attentional selection |
| 2 | Map tasks to component demands | Task-by-component matrix; rank and condition-number checks | Distributed interaction among language, memory, control, and social systems |
| 3 | Establish sensory and basic language access | Modality-specific accuracy; matched low-load contrasts; item-level estimates | Auditory/visual pathways and language-selective temporal/frontal populations |
| 4 | Separate literal and inferred meaning | Separate means; literal-minus-inference cost; context interaction | Lexical-semantic representations with discourse and control networks |
| 5 | Test communicative cue integration | Reliability-weighted cue scores; normalized weights; Bayesian-style cue combination | Parallel cue-sensitive populations and recurrent feedback |
| 6 | Separate social and nonsocial inference | Independent social/physical inference scores; explicit speaker-knowledge benefit | Language and theory-of-mind systems can cooperate without being identical |
| 7 | Measure discourse maintenance and coherence | Entity/event graph; edge precision/recall; distractor cost | Recurrent temporal, parietal, frontal, and memory interactions |
| 8 | Test memory and executive load | Accuracy-versus-load slope; high/low load difference; switching cost | Recurrent maintenance, attention, and inhibitory interactions |
| 9 | Optional right-hemisphere discourse profile | Within-person sentence/discourse contrast; separate inference/prosody/nonverbal tasks | Distributed bilateral discourse and cognitive-communication systems |
| 10 | Optional autism-related pragmatic profile | Individual task/cue/context analysis; explicitness effects | Heterogeneous language, sensory, social-cognitive, and executive networks |
| 11 | Optional psychosis-spectrum discourse profile | Separate comprehension/production, semantic selection, coherence, and control measures | Distributed language, semantic, social, default-mode, and control networks |
| 12 | Differentiate neurodegenerative profiles | Component-specific semantic, memory, executive, and pragmatic trajectories | Person- and condition-dependent distributed network change |
| 13 | Optional aphasia-related profile | Matched linguistic/nonlinguistic task contrast; language/concept separation | Language-selective circuitry interacting with conceptual and domain-general systems |
| 14 | Estimate cross-domain abilities and preservation | Component means, sample spread, counts, within-profile ranking | Distributed population coding and interacting networks |
| 15 | Compare competing explanations | Prediction likelihood/error, Brier score, complexity-aware comparison | Prediction and error-sensitive network updating as a broad hypothesis |
| 16 | Test support and compensation | Matched support benefit; context/load/cue contrasts; generalization checks | Attention and recurrent feedback alter effective cue availability |
| 17 | Validate independently | Held-out accuracy, Brier/MSE, calibration, test-retest checks | Neural evidence can constrain models when measured appropriately |
| 18 | Update longitudinally | Component-specific slopes and bounded state updates | Plasticity, state variation, and development/acquired/neurodegenerative change |
| 19 | Report transparently | Evidence summary, uncertainty, alternatives, support effects, further tests | No single biological module is assumed to explain the complete profile |

## Core representations

The ten component dimensions are:

- perceptual_access
- lexical_access
- syntactic_parsing
- semantic_composition
- working_memory
- discourse_coherence
- pragmatic_inference
- social_state_inference
- world_knowledge
- executive_control

For task j, the optional demand vector is

\[
\mathbf d_j =
[d_{jP},d_{jL},d_{jY},d_{jS},d_{jM},d_{jD},d_{jR},d_{jW},d_{jE},d_{jK}].
\]

The demand matrix exposes whether the task battery provides enough independent variation to distinguish these dimensions. A low-rank or ill-conditioned matrix is reported as a limitation; the model does not force a unique explanation from confounded tasks.

For descriptive modelling, one can use a logistic response function:

\[
z_j = \beta_0 + \sum_k \beta_k c_k d_{jk}
      - \sum_q \gamma_q \ell_{jq} + \epsilon_j,
\qquad
P(y_j=1)=\frac{1}{1+e^{-z_j}}.
\]

Here c_k terms are estimated functional measures, d_jk are task demands, and ell_jq are additional resource/interference loads. This is a psychometric model. The software does not claim that neurons execute logistic regression.

## Important computations and interpretation

**Literal versus inferred meaning.** When both conditions exist, the algorithm reports

\[
\Delta_{\mathrm{infer}}=\bar y_{\mathrm{literal}}-\bar y_{\mathrm{inference}}.
\]

The difference describes a task contrast; it does not identify whether the cause is pragmatic inference, memory, cue access, vocabulary, or task familiarity.

**Cue combination.** Cue estimates can include linguistic, prosodic, scene, or speaker evidence. Non-negative estimates are averaged and normalized to sum to one. This normalization is a convenient representation of relative cue weighting, not evidence that the brain explicitly normalizes probability weights.

**Discourse graphs.** Nodes/edges are not required for every task. When expected and produced edges are provided, the model computes precision, recall, missing required relations, and unsupported relations. Semantic similarity alone is not treated as logical consistency or truth.

**Working memory and executive control.** Accuracy can be related to an explicit numeric load variable. The regression slope describes a behavioural load effect, not the identity of a unique neural cause.

**Support conditions.** A support benefit is estimated only from items with the same pair identifier and explicit supported/unsupported labels:

\[
B_{\mathrm{support}}=\bar y_{\mathrm{support}}-\bar y_{\mathrm{baseline}}.
\]

A benefit indicates that performance changed under that condition. It does not show that the underlying mechanism was repaired.

**Longitudinal estimates.** A component trajectory requires at least two distinct measurement times. Current component estimates require current measurements. Previous estimates are retained as historical references but are never silently reused as current evidence.

**Validation.** Only tasks explicitly marked validation, held_out, or test with a supplied prediction probability enter held-out metrics. Training observations are not silently recycled as validation.

## Clinical-context profiles

The optional context families summarize matched, explicitly tagged tasks:

- **right_hemisphere:** sentence/discourse contrasts, inference, prosody, and nonverbal cue measures. The model does not assume an exclusive right-hemisphere pragmatics centre or preserved syntax.
- **autism_related:** task-specific indirect requests, irony, implicatures, cue use, explicitness, and communication preferences. It does not presume one uniform profile or treat group-level imaging findings as an individual biomarker.
- **psychosis_spectrum:** separate semantic/context selection, discourse coherence, social/executive measures, and comprehension versus production. No fixed diagnosis-specific circuit is assumed.
- **neurodegenerative:** separate semantic, episodic-memory, executive, and pragmatic trajectories. No single generic dementia degradation rule is used.
- **aphasia:** linguistic and nonlinguistic versions of matched concepts/tasks are kept separate. Retained nonlinguistic knowledge is possible but is not assumed.

A profile family is available only when relevant task records exist. Supplying a context label does not create a deficit score.

## Request data conventions

Task dictionaries may include:

- task_id, task_type, stage, meaning_level, modality, accuracy or score
- demands: mapping from a component name to a numeric demand
- inference_domain: social, belief, intention, emotion, nonsocial, physical, or causal
- speaker_knowledge_explicit: boolean
- cue_weights or cue_weight_estimates
- discourse_graph: required_edges and produced_edges
- memory_load, switch_cost, distractor_present
- profile_family, response_mode, presentation, communication_preferences
- pair_id, support_present, support_generalization
- split and predicted_probability for independent validation

Component measures accept a scalar, a sequence, or a mapping with values. Longitudinal records use component, time, and score. Missing measurements remain None; they are never converted to zero.

## Biological and statistical safeguards

The code exposes possible biological references such as excitation/inhibition, population coding, recurrent maintenance, cue-sensitive processing, plasticity, and distributed network interaction. These are qualitative links to plausible neural functions. They are not claims that a named clinical profile maps to a single area or that the displayed statistical equations are literal neural computations.

The following operations are analyst-side procedures: matrix rank and conditioning, logistic regression, graph metrics, Brier scores, model comparison, cross-validation, confidence intervals, and longitudinal regression. Neural activity, lesion evidence, stimulation effects, and functional connectivity must be interpreted at their appropriate measurement levels.

The output always declares diagnosis_not_performed=True. It must not be used to assign a clinical label, infer character or intelligence, or replace qualified multidisciplinary assessment.
