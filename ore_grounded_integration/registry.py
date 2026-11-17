"""Stable stage and mathematical-operation registry for the 22-stage source algorithm."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one executable mathematical or logical operation and its biological analogy."""

    operation_id: str
    stage_id: str
    name: str
    mathematical_form: str
    functional_description: str
    biological_reference: str


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Describe one of the 22 sequential semantic stages and its registered operations."""

    stage_id: str
    stage_number: int
    name: str
    objective: str
    biological_reference: str
    operations: tuple[OperationSpec, ...]


_STAGE_ROWS = (
    ("Acquire and synchronize multimodal information",
     "Collect language, visual, auditory, motor and situational evidence and estimate temporal relatedness.",
     "Parallel sensory pathways encode distinct signals; attention and temporal coincidence can support event binding.",
     (("measure_temporal_gap", "Delta_t(i,j)=|t_i-t_j|", "Calculate the absolute arrival-time gap for each input pair."),
      ("weight_temporal_coincidence", "A_ij=exp(-Delta_t(i,j)/tau)", "Assign stronger association to inputs within the configured temporal window."))),
    ("Extract modality-specific representations",
     "Convert each channel into a fixed-dimensional, normalized feature representation.",
     "Early sensory and language systems extract partially specialized features before convergence.",
     (("encode_modality", "Z_m=f_m(X_m)", "Apply the modality-specific encoder supplied by the caller or baseline feature hashing."),
      ("normalize_features", "z_hat=(z-mean(z))/(std(z)+epsilon)", "Place feature values on a comparable scale without merging the original modalities."))),
    ("Construct distributional linguistic representations",
     "Represent words using their local linguistic contexts and learned co-occurrence statistics.",
     "Repeated language exposure strengthens associations among co-activated lexical and conceptual populations.",
     (("collect_context_window", "N(w)=tokens[w-k:w+k]", "Extract neighboring words around each target token."),
      ("construct_distributional_vector", "v(w)=hash(context_counts(w))", "Encode observed co-occurrence counts in a stable vector space."),
      ("measure_lexical_similarity", "sim(a,b)=cos(v_a,v_b)", "Compare word-context vectors using cosine similarity."))),
    ("Determine the current semantic context",
     "Estimate which situation or domain best explains the current linguistic and non-linguistic evidence.",
     "Contextual expectations and task state bias which semantic populations become active.",
     (("score_context_hypotheses", "s(c)=sum_f evidence(f)*weight(f,c)", "Score candidate contexts from explicitly supplied context features."),
      ("normalize_context_posterior", "P(c|x)=softmax(s(c)/T)", "Normalize competing context scores into comparable probabilities."))),
    ("Align cross-modal representations",
     "Associate modality vectors that refer to the same event while distinguishing mismatched examples.",
     "Cross-modal co-activation and associative plasticity can link distributed sensory and linguistic representations.",
     (("measure_cross_modal_similarity", "sim_ij=cos(z_i,z_j)", "Measure compatibility between each pair of available modality vectors."),
      ("calculate_contrastive_objective", "L=max(0, margin-s_pos+s_neg)", "Calculate a margin loss for positive and mismatched cross-modal pairs."))),
    ("Connect concepts to actions and affordances",
     "Represent actions that can be performed with or around a concept and their expected outcomes.",
     "Perception can activate learned action possibilities through interacting sensory and motor systems.",
     (("estimate_action_affordance", "A(o,a)=P(success|o,a)", "Estimate action suitability from motor features and explicitly supplied evidence."),
      ("represent_state_transition", "P(S_(t+1)|S_t,A_t)", "Represent an action as a transition between situation states."))),
    ("Integrate linguistic, sensory and motor information",
     "Construct a shared concept state while preserving the separate modality representations.",
     "Transmodal convergence, including anterior temporal contributions, is modeled as one component of a distributed network.",
     (("project_modalities", "H_i=W_i*Z_i+b_i", "Map each modality into a common representational dimension."),
      ("normalize_modality_weights", "alpha_i=softmax(reliability_i+relevance_i)", "Weight each channel by availability, confidence and task relevance."),
      ("integrate_concept", "C=sum_i(alpha_i*H_i)", "Form the weighted common representation without deleting channel-specific data."))),
    ("Build a semantic relationship network",
     "Connect concepts through co-occurrence, category, functional and event relationships.",
     "Distributed semantic networks contain learned associations; activation can spread through connected populations.",
     (("construct_semantic_graph", "G=(V,E,W)", "Represent concept nodes, typed relationships and connection strengths."),
      ("spread_activation", "a_i(t+1)=f(sum_j(W_ji*a_j(t))-lambda*a_i(t))", "Propagate activation with decay over a bounded number of steps."))),
    ("Select the meaning appropriate to context",
     "Resolve ambiguity by comparing candidate meanings against current context and available evidence.",
     "Semantic-control and context networks bias competing interpretations and suppress less relevant alternatives.",
     (("score_candidate_sense", "s(m)=sum_f context(f)*sense_weight(m,f)", "Score each candidate sense against context features."),
      ("select_posterior_meaning", "P(m|c)=softmax(s(m)/T)", "Retain the ranked alternatives and the leading interpretation."))),
    ("Compose word meanings into sentence meaning",
     "Combine token representations with grammatical roles and event relationships.",
     "Language networks integrate lexical, syntactic and semantic-role information into proposition-level meaning.",
     (("contextualize_tokens", "h_i=f(z_i, z_1...z_n)", "Allow each token representation to incorporate sentence context."),
      ("construct_event_frame", "S=f(tokens, roles, context)", "Bind agent, action, object and contextual relations into one event frame."))),
    ("Generate predictions about upcoming language and events",
     "Estimate likely next words or event outcomes from context and learned transition statistics.",
     "Predictive-processing theories model expectations that can bias incoming processing; exact neural implementation remains debated.",
     (("estimate_next_token", "P(w_(t+1)|w_1...w_t)", "Estimate next-token probabilities from observed transition counts."),
      ("predict_event_state", "S_hat_(t+1)=F(S_t, M)", "Predict a possible next state from the integrated concept and stored associations."))),
    ("Compare predictions with incoming information",
     "Quantify mismatch between expected and observed features or events.",
     "Prediction-error signals are proposed to influence attention, updating and subsequent processing.",
     (("calculate_vector_residual", "e_t=x_t-x_hat_t", "Compute signed residuals and squared reconstruction error."),
      ("calculate_surprise", "I(x)=-log(P(x|context))", "Measure the surprise of an observed next item under the prediction distribution."))),
    ("Regulate attention according to relevance and uncertainty",
     "Allocate processing weight to evidence relevant to the current task and uncertain predictions.",
     "Attention and control systems can modulate the gain of task-relevant sensory and semantic populations.",
     (("score_attention", "r_i=alpha*C_i+beta*U_i+gamma*T_i", "Combine context relevance, uncertainty and task importance."),
      ("allocate_attention", "a_i=softmax(r_i/T)", "Normalize relevance scores and construct an attention-weighted representation."))),
    ("Retrieve semantic and episodic memories",
     "Retrieve prior concept knowledge and specific experiences that resemble the current input.",
     "Hippocampal and neocortical systems interact in episodic retrieval and semantic knowledge access.",
     (("rank_memory_matches", "score_i=cos(Q,m_i)+recency_i+importance_i", "Rank stored experiences by similarity, recency and importance."),
      ("estimate_retrieval_uncertainty", "H(memory)=-sum_i(p_i*log(p_i))", "Quantify ambiguity across retrieved memory candidates."))),
    ("Perform semantic inference and generalization",
     "Derive implied relationships and transfer stable conceptual properties to new examples.",
     "Reasoning recruits semantic knowledge, memory and executive processes to combine previously learned relations.",
     (("compose_graph_paths", "A->B and B->C => A->C", "Infer candidate relationships by following bounded graph paths."),
      ("generalize_from_examples", "G(x)=sum_i(w_i*cos(x,m_i))", "Estimate similarity-based transfer from stored examples."))),
    ("Produce language or behavior from the integrated concept",
     "Route task-relevant semantic evidence into a description, classification, decision or action.",
     "Semantic representations interact with lexical production, response selection and motor-output systems.",
     (("select_response_content", "C_response=Select(C_integrated,Task)", "Select only features relevant to the requested response."),
      ("choose_action", "A*=argmax_A Q(C,A)", "Choose the highest-scoring supported action when action output is requested."))),
    ("Learn from feedback and update associations",
     "Use new co-activation and outcome feedback to update semantic links and transition expectations.",
     "Synaptic plasticity and reward-related prediction errors provide candidate biological analogies; backpropagation is not assumed.",
     (("apply_hebbian_update", "DeltaW=eta*outer(x,y)-lambda*W", "Strengthen supported co-activity associations while applying weight decay."),
      ("calculate_temporal_difference", "delta=r+gamma*V(s_next)-V(s)", "Use outcome feedback to update a simple value estimate."))),
    ("Consolidate learning and maintain knowledge over time",
     "Stabilize repeated knowledge, preserve episodic detail and reduce reliance on one learning episode.",
     "Memory consolidation involves time-dependent interactions between hippocampal and neocortical systems, including sleep-related processes.",
     (("decay_memory_accessibility", "S(t)=S0*exp(-lambda*t)", "Model gradual decline in accessibility between retrieval events."),
      ("strengthen_retrieved_memory", "S_new=S_old+eta*(1-S_old)", "Increase accessibility after successful retrieval."),
      ("measure_generalization", "G(C)=mean_i(sim(C,X_i))", "Evaluate whether a representation transfers across distinct examples."))),
    ("Construct multisensory teaching representations",
     "Coordinate text, image, sound, action and context evidence into an instructional representation.",
     "Meaningful cross-modal teaching can recruit complementary sensory and conceptual systems; extra modalities alone do not guarantee learning.",
     (("assemble_teaching_channels", "T={Z_text,Z_image,Z_sound,Z_action,Z_context}", "Preserve each available learning channel in a single lesson object."),
      ("measure_teaching_coherence", "Coherence=mean_ij(sim(Z_i,Z_j))", "Measure alignment only across available meaningful modality pairs."),
      ("estimate_channel_reliability", "R_i=1/(1+Error_i)", "Give more influence to channels with lower measured error."))),
    ("Construct meaningful usage examples",
     "Represent stable and context-dependent uses of a concept across different situations.",
     "Repeated exposure across contexts helps learners distinguish general semantic structure from situation-specific meaning.",
     (("estimate_shared_features", "C_common=intersection_i Features(C_i)", "Identify features shared across the observed uses."),
      ("isolate_context_features", "C_specific_i=Features(C_i)-C_common", "Keep contextual variations separate from shared meaning."))),
    ("Evaluate learning using retrieval and prediction",
     "Measure recall, identification, action, contextual comprehension and generalization.",
     "Retrieval practice and performance monitoring provide observable evidence about learning progress.",
     (("compute_learning_score", "L=sum_k(w_k*score_k)", "Combine learner performance dimensions using task-specific weights."),
      ("measure_learner_uncertainty", "H=-sum_i(p_i*log(p_i))", "Estimate uncertainty and identify dimensions requiring further instruction."))),
    ("Adapt instruction to the learner's knowledge",
     "Select the next teaching activity to address the learner's weakest measured semantic connection.",
     "Adaptive instruction approximates experience-dependent changes in attention, practice and memory retrieval.",
     (("construct_learner_state", "L_t=[R_t,I_t,A_t,C_t,G_t]", "Represent the learner's measured recall, identification, action, context and transfer scores."),
      ("select_next_activity", "a*=argmax_a ExpectedLearningGain(a,L_t)", "Choose an activity expected to improve the weakest supported knowledge dimension."),
      ("estimate_expected_gain", "Gain(a)=E[score_after]-score_before", "Estimate a relative gain from the current learner state."))),
)

STAGE_REGISTRY: tuple[StageSpec, ...] = tuple(
    StageSpec(
        stage_id=f"GS-{index:02d}",
        stage_number=index,
        name=row[0],
        objective=row[1],
        biological_reference=row[2],
        operations=tuple(
            OperationSpec(
                operation_id=f"GS-OP-{index:02d}-{operation_index:02d}",
                stage_id=f"GS-{index:02d}",
                name=operation[0],
                mathematical_form=operation[1],
                functional_description=operation[2],
                biological_reference=row[2],
            )
            for operation_index, operation in enumerate(row[3], start=1)
        ),
    )
    for index, row in enumerate(_STAGE_ROWS, start=1)
)

OPERATION_REGISTRY: tuple[OperationSpec, ...] = tuple(
    operation for stage in STAGE_REGISTRY for operation in stage.operations
)

if len(STAGE_REGISTRY) != 22 or tuple(stage.stage_id for stage in STAGE_REGISTRY) != tuple(
    f"GS-{index:02d}" for index in range(1, 23)
):
    raise RuntimeError("grounded semantic registry must contain exactly 22 ordered stages")


def registry_as_dicts() -> list[dict[str, Any]]:
    """Return a JSON-friendly copy of all stage, biological and mathematical-operation metadata."""
    return [
        {
            **asdict(stage),
            "operations": [asdict(operation) for operation in stage.operations],
        }
        for stage in STAGE_REGISTRY
    ]


def registry_summary() -> dict[str, int]:
    """Return the registered stage and operation counts for validation and logger reporting."""
    return {
        "stage_count": len(STAGE_REGISTRY),
        "operation_count": len(OPERATION_REGISTRY),
    }
