"""Canonical source-ordered 17-stage registry for Point 01 active engagement."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe a Point 01 operation and its mathematical and biological interpretation."""

    operation_id: str
    source_step: int
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the operation specification as a plain dictionary."""
        return asdict(self)


_ROWS = (
    (0, "Initialize an active learning episode",
     "Set the learner goal, task mode, provisional attention policy, stopping conditions, and an empty episode record.",
     "V_i=w_G*Rel(i,G)+w_N*Novelty(i)+w_U*Uncertainty(i); attention=normalize(V)",
     "Prefrontal goal maintenance interacts with parietal attention and sensory/language processing; the episode record is computational.",
     "distributed_control_supported; scalar priority is a computational abstraction",
     ("material", "goal", "context"), ("episode_state", "attention_policy")),
    (1, "Construct an online representation of incoming material",
     "Encode text or supplied transcripts into a local representation, retain sequence context, and update a recurrent population-state surrogate.",
     "x_t=f_theta(M_t); r_t=sigma(W_x*x_t+W_h*r_(t-1)+b)",
     "Sensory and language-related populations interact recurrently with memory and control systems; this implementation uses a deterministic feature encoder.",
     "modality_specific_encoding_supported; implemented feature vector is a model",
     ("input_segments", "prior_stream_state"), ("online_representation", "population_activity")),
    (2, "Segment the stream and extract propositions",
     "Detect sentence-like boundaries, extract provisional proposition units and relation cues, preserve source text, and mark uncertain segmentation.",
     "Score(S)=lambda_1*Coherence+lambda_2*BoundaryEvidence-lambda_3*Fragmentation",
     "Temporal/contextual grouping and distributed language/conceptual representations are plausible contributors; proposition tuples are software structures.",
     "segmentation_and_composition_supported; explicit scoring is computational",
     ("online_representation",), ("propositions", "local_relations")),
    (3, "Select the central claim and comprehension target",
     "Rank proposition units by goal relevance, local centrality, and uncertainty while retaining supporting details.",
     "Priority(p_i)=a*Rel(p_i,G)+b*Centrality(p_i,P)+c*Uncertainty(p_i)",
     "Frontoparietal attention and working-memory control bias processing toward task-relevant information.",
     "goal_directed_selection_supported; priority equation is computational",
     ("propositions", "goal"), ("target_proposition", "supporting_propositions")),
    (4, "Generate questions and provisional predictions",
     "Generate why/how/what/evidence/counterfactual questions, estimate their information value, and keep unselected questions pending.",
     "H(Z)=-sum_i p_i*log(p_i); q*=argmax_q[IG(q)*Importance(q)/(Cost(q)+epsilon)]",
     "Prediction, uncertainty monitoring, and controlled search involve recurrent temporal, frontal, parietal, and memory-related systems; entropy is a model-level metric.",
     "question_generation_and_active_search_supported; expected information gain is abstract",
     ("target", "propositions", "goal"), ("prioritized_questions", "predictions", "uncertainty")),
    (5, "Allocate attention to question-relevant evidence",
     "Score source segments against the selected question, normalize attention weights, and increase the effective gain of relevant features without eliminating distractors.",
     "s_i=cos(q,x_i); alpha_i=softmax(s_i/tau); r_i'=g_i*r_i",
     "Top-down frontoparietal signals, sensory/association populations, and inhibitory competition can bias processing toward relevant evidence.",
     "selective_attention_supported; similarity and softmax are computational surrogates",
     ("question", "source_segments"), ("attention_weights", "selected_evidence")),
    (6, "Generate candidate answers from evidence",
     "Retrieve relevant evidence, construct competing candidate responses, distinguish direct source statements from learner inferences, and penalize unsupported additions.",
     "Score(z_k)=alpha*cos(z_k,q)+beta*Fit(z_k,E)-gamma*Unsupported(z_k); P_k=softmax(Score)",
     "Distributed temporal/parietal semantic representations interact with frontal control; hippocampal pattern completion may reinstate recent episode details.",
     "semantic_retrieval_and_control_supported; candidate scoring is computational",
     ("question", "evidence", "local_representations"), ("candidate_answers", "support_labels")),
    (7, "Reconstruct meaning in the learner's own words",
     "Use a supplied learner explanation when available; otherwise create an explicitly labeled extractive scaffold and score meaning preservation and proposition coverage.",
     "F=cos(m_source,m_explanation); C=sum_i w_i*I(p_i represented)/sum_i w_i; Q_para=lambda_F*F+lambda_C*C-lambda_D*D",
     "Semantic retrieval, lexical selection, working-memory maintenance, and language production interact; a template scaffold is not proof of learner understanding.",
     "generation_and_reconstruction_supported; automatic paraphrase proxy is limited",
     ("candidate_answer", "source_propositions", "optional_learner_explanation"), ("explanation", "fidelity", "coverage")),
    (8, "Elaborate with useful supported relations",
     "Add explicit causal, temporal, conditional, contrastive, or explanatory links, create a local proposition graph, and mark unsupported inferences.",
     "S(e_ij)=w_1*Compatibility+w_2*Evidence+w_3*Value-w_4*Speculation; Coverage_G=supported_edges/required_edges",
     "Semantic retrieval, controlled inference, and distributed association networks may support elaboration.",
     "elaborative_encoding_supported; graph edge scores are computational",
     ("explanation", "source_propositions"), ("explanation_graph", "supported_relations", "gaps")),
    (9, "Check coherence, completeness, and evidential support",
     "Check target coverage, relation direction, contradictions, unsupported claims, circularity proxies, and missing intermediate links.",
     "Q=aF+bC+cR+dE_v-eK-fU",
     "Conflict/error monitoring compares expected and currently active representations across frontal and memory-related networks.",
     "metacognitive_monitoring_supported; scalar quality is a modeling summary",
     ("explanation", "source", "question"), ("quality_score", "contradictions", "unresolved_gaps")),
    (10, "Repair gaps through targeted reprocessing",
     "Classify gaps, generate focused repair questions, retrieve only source-supported repairs, and keep unresolved issues explicit when evidence is missing.",
     "i*=argmax_i Importance_i*ErrorMagnitude_i*EvidenceAvailability_i; Z_(t+1)=Z_t+DeltaZ",
     "Recurrent activity can bias renewed attention and retrieval after a mismatch; local error vectors are computational abstractions.",
     "error_guided_revision_supported; vector repair is an implementation model",
     ("quality_diagnostics", "source_evidence"), ("revised_explanation", "repair_questions", "unresolved_gaps")),
    (11, "Prepare a teachable explanation",
     "Adapt detail and vocabulary to a listener model, order prerequisites before dependent claims, predict questions, and detect listener-facing gaps.",
     "U_teach=lambda_1*ComprehensionGain-lambda_2*Omission-lambda_3*Complexity; order*=minimize prerequisite violations",
     "Language generation, sequencing, self-monitoring, and audience perspective-taking recruit interacting frontal, temporal, and social-cognitive systems.",
     "self_explanation_and_teaching_supported; audience utility is computational",
     ("explanation", "listener_level", "proposition_graph"), ("teaching_script", "anticipated_questions")),
    (12, "Construct optional imagery or a causal/spatial model",
     "Only when enabled and suitable, create a structured visual/spatial representation, preserve relations, and align it with the verbal explanation.",
     "Alignment=cos(i_t,z_t); s_(t+1)=T(s_t,Action_t,Condition_t)",
     "Imagery can reactivate aspects of perceptual and association systems; hippocampal/cortical binding may support scene relations.",
     "mental_imagery_supported; symbolic scene graph is a proxy, not a rendered image",
     ("target", "explanation_graph", "imagery_policy"), ("optional_scene_graph", "alignment_score")),
    (13, "Bind source content and learner-generated processing",
     "Bind propositions, questions, explanations, predictions, actions, context, and optional imagery into a source-linked episode with multiple retrieval cues.",
     "h=phi(W_m*m+W_z*z+W_q*q+W_i*i+W_a*a); B=sum_k u_k*v_k^T",
     "Hippocampal formation contributes to binding episodic elements while cortical systems represent distributed content; a single episode vector is a software abstraction.",
     "episodic_binding_supported; vector aggregation is computational",
     ("source", "questions", "explanation", "context", "imagery"), ("episode_memory", "retrieval_cues")),
    (14, "Apply bounded activity-dependent plasticity",
     "Use co-activation, a learning gate, and bounded timing-sensitive updates to strengthen or weaken modeled associations without unbounded weights.",
     "Delta w_ij=eta*g*r_i*r_j; STDP(Delta t)=A+*exp(-Delta t/tau+) or -A-*exp(Delta t/tau-)",
     "Synaptic plasticity, hippocampal-cortical interaction, and neuromodulatory gates contribute to learning; the rule is illustrative rather than universal.",
     "activity_dependent_plasticity_supported; Hebbian/STDP approximation is simplified",
     ("episode_embedding", "attention", "learning_gate", "association_state"), ("bounded_association_updates", "trace_strength")),
    (15, "Monitor proposition-level comprehension and confidence",
     "Estimate fidelity, coverage, coherence, unresolved uncertainty, and proposition-level confidence; compute calibration error only when independent correctness labels exist.",
     "Brier=mean((c_i-y_i)^2); C_t=aF+bC+cR-dU",
     "Metacognitive monitoring may involve frontal and memory-related networks; the scalar confidence is a model estimate.",
     "metacognitive_monitoring_supported; calibration requires external correctness labels",
     ("quality", "gaps", "external_accuracy_if_available"), ("comprehension_estimate", "confidence", "calibration")),
    (16, "Close the cycle or return to a focused unresolved gap",
     "Close only when quality and critical-gap criteria are met; otherwise retain an incomplete episode and identify the next targeted question.",
     "Stop=I[Q(Z)>=theta_Q and U_critical<=theta_U]",
     "Goal-control systems can shift attention when a unit is resolved or resources are redirected; encoding need not stop at this boundary.",
     "goal_completion_and_control_supported; thresholds are task-dependent",
     ("comprehension", "critical_gaps", "episode_memory"), ("episode_status", "next_action")),
)


OPERATION_REGISTRY = tuple(
    OperationSpec(
        operation_id=f"P01-{index + 1:02d}",
        source_step=row[0],
        name=row[1],
        purpose=row[2],
        mathematical_form=row[3],
        biological_reference=row[4],
        evidence_level=row[5],
        inputs=row[6],
        outputs=row[7],
    )
    for index, row in enumerate(_ROWS)
)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return the complete 17-stage Point 01 operation registry."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
