"""Canonical source-ordered 16-stage operation registry for Point 14."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe a source-aligned operation and its mathematical and biological interpretation."""

    operation_id: str
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
    ("P14-01", "Define the learning episode and interacting agents",
     "Initialize learner and partner states, shared environment, current goal, and a time-stamped multimodal interaction stream while keeping social cues distinct from linguistic content.",
     "(L[t+1],P_hat[t+1])=F(L[t],P_hat[t],E[t],o[t],g[t])",
     "Frontotemporal language, superior temporal multisensory, social-cognitive, prefrontal, and hippocampal/medial-temporal systems contribute interacting signals.",
     "social_episode_and_agent_context_supported; explicit state-space record is computational",
     ("learner_state", "partner", "environment", "interaction_event"), ("episode_state", "partner_model", "event_stream")),
    ("P14-02", "Detect a socially relevant partner and orient attention",
     "Prioritize face, voice, biological motion, gestures, turn-opening cues, relevance, novelty, clarity, and goal relevance against distractors.",
     "A_i=w_r R_i+w_g G_i+w_c C_i+w_n N_i-w_d D_i; P(i)=softmax(A_i/tau)",
     "Superior temporal and temporoparietal regions process voice, face, biological motion, and gaze; frontoparietal attention systems select relevant cues.",
     "social_orienting_and_attention_supported; priority scores are computational",
     ("social_signals", "learning_goal", "competing_cues"), ("partner_attention", "cue_priorities")),
    ("P14-03", "Establish joint attention to a shared referent or event",
     "Integrate gaze, pointing, speech, gesture, scene context, partner focus, and learner focus; confirm or revise shared focus instead of assuming it.",
     "P(x|cues,C) proportional to P(x|C)*product_k P(c_k|x,C)^rho_k; J=sigmoid(alpha*match+beta*contingency+b)",
     "Posterior superior temporal sulcus, temporoparietal, and frontal attention/control systems contribute to shared attention.",
     "joint_attention_supported; Bayesian cue integration and J score are computational",
     ("partner_cues", "learner_focus", "referent_candidates", "context"), ("referent_posterior", "joint_attention_confidence")),
    ("P14-04", "Infer the partner's communicative intention",
     "Infer candidate goals and partner knowledge from literal language, prosody, gesture, gaze, history, and shared activity, preserving uncertainty.",
     "P(g|u,C) proportional to P(u|g,C)P(g|C); D_KL(B_L||B_P)=sum_s B_L(s)log(B_L(s)/B_P(s))",
     "Medial prefrontal, temporoparietal junction, posterior superior temporal, and precuneus/posterior cingulate systems may contribute to mental-state reasoning.",
     "communicative_intention_and_perspective_inference_supported; inverse-planning and KL estimates are computational",
     ("utterance", "multimodal_context", "partner_model", "interaction_history"), ("intention_posterior", "partner_knowledge_gap", "uncertainty")),
    ("P14-05", "Integrate speech with gesture, prosody, face, and action",
     "Extract modality-specific features, align them in time, weight reliability, resolve conflicts, bind compatible content, and preserve the source modality.",
     "z_joint=sum_m alpha_m*z_m; alpha_m proportional to 1/sigma_m^2; temporal_alignment=likelihood(delta_t)",
     "Auditory/visual association cortex, superior temporal, inferior frontal, parietal multisensory, and motor/premotor systems interact.",
     "multimodal_speech_gesture_integration_supported; precision weights are computational",
     ("speech", "gaze", "gesture", "prosody", "facial_cues", "action"), ("joint_event_representation", "modality_provenance")),
    ("P14-06", "Map words and relational expressions onto shared experience",
     "Infer word-to-referent, property, action, or relation mappings from joint attention, syntax, speaker intention, scene, and prior vocabulary; update from examples.",
     "P(r|w,C,S) proportional to P(w|r,S)P(r|C); a_wr[t+1]=a_wr[t]+eta*q_t*(y_t-yhat_t)",
     "Frontotemporal language network, temporal/temporoparietal semantic systems, and hippocampal episodic binding contribute.",
     "grounded_word_learning_supported; association update is computational",
     ("target_word", "referent_posterior", "speaker_intention", "scene", "prior_vocabulary"), ("word_concept_mapping", "mapping_confidence", "learning_episode")),
    ("P14-07", "Coordinate conversational turns and predict when to listen or speak",
     "Estimate continue, yield, and repair states from syntax, prosody, pauses, gaze, gesture, history, and response readiness.",
     "P(T in {continue,yield,repair}|cues,history); initiate when P(yield|cues)>theta; use turn-end hazard h(tau)",
     "Auditory-motor integration, inferior frontal and premotor, temporal language, and cognitive control networks support turn coordination.",
     "conversational_turn_coordination_supported; state/hazard model is computational",
     ("prosody", "syntax", "gaze", "pause", "gesture", "response_readiness"), ("turn_state_posterior", "response_timing")),
    ("P14-08", "Generate a response adapted to the partner",
     "Select response detail, vocabulary, examples, and scaffolding from estimated listener knowledge while monitoring comprehension.",
     "u*=argmax_u[E(I_shared|u,B_L)-lambda*C(u)-mu*A(u)]",
     "Language production interacts with working memory, executive control, and mentalizing/audience-design systems.",
     "audience_design_and_adaptation_supported; communicative utility is computational",
     ("communicative_goal", "partner_knowledge", "candidate_responses", "listener_feedback"), ("audience_adapted_response", "detail_level")),
    ("P14-09", "Detect mismatch, misunderstanding, and conversational repair",
     "Compare predicted and observed partner responses, classify mismatch, reopen alternatives, clarify, and retain corrective evidence.",
     "delta=o-yhat; E=-log P(o|H); reopen when E exceeds a context-sensitive threshold",
     "Language comprehension/monitoring, prefrontal revision, temporal semantic retrieval, and hippocampal episode systems may contribute.",
     "repair_and_prediction_error_supported; thresholds are computational",
     ("predicted_response", "partner_response", "feedback", "current_interpretation"), ("mismatch_type", "repair_action", "corrected_interpretation")),
    ("P14-10", "Learn from social feedback and contingent reinforcement",
     "Separate informational correction, task outcome, and social-affective reward; weight source reliability and never treat approval as proof of truth.",
     "Q[a] <- Q[a]+eta*(r-Q[a]); delta=r+gamma*V(s')-V(s); factual_update weighted by rho",
     "Ventral striatal/midbrain reward-learning, prefrontal evaluation, hippocampal episode, and cortical semantic systems interact.",
     "social_feedback_and_reinforcement_supported; scalar value updates are computational",
     ("feedback_kind", "source_reliability", "factual_evidence", "observed_outcome"), ("updated_beliefs", "action_values", "learning_confidence")),
    ("P14-11", "Use questions and guided scaffolding",
     "Estimate competence and task gap, provide the minimum useful prompt/hint/example, observe attempts, adjust support, and fade it after success.",
     "s=clip(k*(d-c),s_min,s_max); c[t+1]=c[t]+eta*(p_t-p_hat_t)",
     "Prefrontal/parietal executive systems, language instruction networks, and hippocampal/cortical memory systems contribute; partner support is external.",
     "adaptive_scaffolding_supported; scaffold policy and competence update are computational",
     ("learner_competence", "task_difficulty", "attempt_performance", "prior_support"), ("scaffold_type", "support_level", "independence_estimate")),
    ("P14-12", "Consolidate understanding through explaining and teaching",
     "Prompt self-explanation, retrieve examples and relations, assemble a coherent explanation, detect gaps/contradictions, and test later retrieval.",
     "C=sum_i w_i*1(correctly_explained_i)/sum_i w_i; S[t+1]=S[t]+eta*R_t*K_t",
     "Language production/comprehension, hippocampal retrieval, and prefrontal organization/self-monitoring interact.",
     "self_explanation_and_retrieval_practice_supported; coverage and strength formulas are computational",
     ("learner_explanation", "required_terms", "required_relations", "stored_knowledge"), ("coverage_score", "knowledge_gaps", "revised_concept")),
    ("P14-13", "Integrate another perspective without equating agreement with truth",
     "Compare learner and partner claims, premises, and sources; weight evidence reliability and dependence; preserve disagreement and source provenance.",
     "theta_new=theta_old+eta*rho*(theta_partner-theta_old); P(H|e) proportional to P(e|H)P(H)",
     "Medial prefrontal/temporoparietal mentalizing, language/semantic, prefrontal control, and hippocampal source-binding systems may contribute.",
     "perspective_comparison_and_evidence_evaluation_supported; weighted update is computational",
     ("learner_belief", "partner_claim", "source_reliability", "evidence_quality"), ("revised_belief", "disagreement_record", "source_attribution")),
    ("P14-14", "Use stories and dialogue to construct social and causal event models",
     "Track agents, actions, goals, knowledge, causal and temporal relations, narrator information, character beliefs, and next-event predictions.",
     "G=(V,E_typed); P(e[t+1]|history,schema,agent_beliefs); surprise=-log P(e[t+1]|history)",
     "Language, medial prefrontal/temporoparietal mentalizing, hippocampal, and posterior medial event systems interact.",
     "narrative_event_modeling_supported; graph prediction is computational",
     ("narrative_events", "character_goals", "character_knowledge", "event_history"), ("narrative_graph", "agent_belief_states", "next_event_predictions")),
    ("P14-15", "Coordinate social learning with episodic memory and consolidation",
     "Bind concept content to partner/source, shared referent, scene, modality, feedback, confidence, and time; support later replay and retrieval.",
     "M_i=alpha*A_i+beta*R_i+gamma*N_i+delta*E_i+epsilon*G_i; memory=(content,source,context,time,confidence)",
     "Hippocampal/medial temporal episodic binding, amygdala and neuromodulatory modulation, and cortical concept systems contribute.",
     "social_episodic_binding_and_consolidation_supported; priority score is computational",
     ("interaction_content", "partner_source", "context", "attention", "feedback", "confidence"), ("source_linked_episode", "consolidation_strength")),
    ("P14-16", "Reuse the learned social-conceptual model in future interactions",
     "Retrieve prior interactions and semantic knowledge, adapt partner estimates cautiously, and generalize only stable patterns across different partners.",
     "P(y|x,C,P)=sum_z P(y|x,z,C)P(z|P,C); online updates bounded by evidence and repetition",
     "Distributed language, social-cognitive, executive, and memory systems support adaptation across partners and contexts.",
     "context_sensitive_social_transfer_supported; latent-state mixture and cautious adaptation are computational",
     ("partner_history", "semantic_knowledge", "episodic_memory", "new_interaction_evidence"), ("updated_partner_model", "cross_partner_patterns", "future_predictions")),
)


OPERATION_REGISTRY = tuple(
    OperationSpec(
        operation_id=row[0], name=row[1], purpose=row[2],
        mathematical_form=row[3], biological_reference=row[4],
        evidence_level=row[5], inputs=row[6], outputs=row[7],
    )
    for row in _ROWS
)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return the complete source-ordered Point 14 operation registry."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
