"""Canonical source-ordered 16-stage registry for Point 13 emotion and memory modulation."""
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
    ("P13-01", "Receive an experience and construct an initial event representation",
     "Extract candidate features and bind the event's semantic, sensory, spatial, and temporal relations while retaining uncertainty.",
     "x_t=[x_1,...,x_n]; xbar_t=sum_k w_k*x_(t-k); G_t=(V_t,E_t)",
     "Distributed sensory and language pathways provide feature representations; hippocampal-cortical systems bind event elements.",
     "sensory_feature_and_episodic_binding_supported; vector and graph forms are computational",
     ("sensory_event", "context", "prior_memories"), ("event_features", "event_relations")),
    ("P13-02", "Estimate affective significance",
     "Estimate valence and arousal separately and assess threat, reward relevance, novelty, uncertainty, and current-goal significance.",
     "S=aT+bR+cN+dU; valence in [-1,1], arousal in [0,1]",
     "Amygdala and distributed salience, cortical, hippocampal, hypothalamic, and bodily systems contribute to appraisal.",
     "affective_appraisal_dimensions_supported; weighted significance score is computational",
     ("event_features", "context", "goals", "prior_memories"), ("appraisal", "valence", "arousal", "significance")),
    ("P13-03", "Generate and regulate the arousal response",
     "Update fast arousal and slower stress-load processes separately; avoid collapsing LC-NE and HPA/glucocorticoid responses.",
     "A_fast[t+1]=rho_f*A_fast[t]+(1-rho_f)*q(S); A_slow[t+1]=rho_s*A_slow[t]+(1-rho_s)*threat",
     "LC-norepinephrine and autonomic responses have faster dynamics; HPA/glucocorticoid responses are distinct and slower.",
     "multiple_timescale_arousal_supported; state equations are computational",
     ("appraisal", "body_state"), ("fast_arousal", "slow_stress_load", "NE_gain_proxy", "glucocorticoid_proxy")),
    ("P13-04", "Compute attentional priority",
     "Combine bottom-up salience with top-down goal and contextual relevance, then allocate a finite processing budget.",
     "P_i=alpha*S_i+beta*G_i+gamma*C_i; p_i=softmax(P_i/tau); sum_i a_i<=B",
     "Competitive attention and goal-dependent control involve distributed sensory, parietal, prefrontal, and salience systems.",
     "competitive_and_goal_directed_attention_supported; priority allocation is computational",
     ("event_features", "appraisal", "goal"), ("feature_priorities", "attention_weights")),
    ("P13-05", "Apply neuromodulatory gain to selected representations",
     "Apply state-dependent gain and divisive normalization; permit both facilitation and reduced flexible processing at excessive arousal.",
     "y_i=g(A)*x_i; g(A)=gmax*exp(-(A-A*)^2/(2*sigma^2)); y_i=x_i/(eps+sum_j w_ij*x_j)",
     "LC-NE and basal-forebrain acetylcholine influence neural gain; effects depend on receptors, dose, timing, region, and task.",
     "neuromodulatory_gain_supported; inverted-U and normalization functions are models",
     ("attention_weights", "arousal_state", "event_features"), ("gain_modulated_features", "normalized_activity")),
    ("P13-06", "Encode emotional and goal-relevant features more strongly",
     "Bind attended features and their relations into an episodic trace while preserving unequal encoding of central and peripheral details.",
     "x'=m odot x; h_new=F(x',r,c); E_i=E_0+lambda_1*a_i+lambda_2*s_i+lambda_3*g_i",
     "Hippocampus binds event elements; cortical systems encode modality/semantic information; BLA can modulate consolidation.",
     "episodic_binding_and_emotional_modulation_supported; feature-weight rule is computational",
     ("attended_features", "relations", "context", "attention"), ("episodic_memory_trace", "feature_strengths")),
    ("P13-07", "Integrate stress and arousal effects without assuming more is always better",
     "Estimate task-specific control capacity and attentional trade-offs separately from memory trace strength.",
     "B(A)=B0*f_task(A); detail_i proportional to attention_i*relevance_i",
     "Stress effects vary with intensity, duration, predictability, controllability, timing, and cognitive operation.",
     "non_monotonic_task_dependent_stress_effects_supported; capacity curve is computational",
     ("arousal_state", "task_demands", "memory_trace"), ("control_capacity", "stress_tradeoffs")),
    ("P13-08", "Evaluate reward value and motivational relevance",
     "Estimate expected outcome value and subtract effort, cognitive, time, and opportunity costs.",
     "Q(a)=sum_o P(o|a)V(o); U(a)=Q(a)-kappa*C(a); P(a)=softmax(U(a)/tau)",
     "Midbrain-striatal and midbrain-prefrontal dopamine pathways contribute to reward value and incentive motivation.",
     "reward_value_and_effort_tradeoffs_supported; utility model is computational",
     ("goal", "expected_outcomes", "costs", "resources"), ("expected_values", "motivational_utility")),
    ("P13-09", "Compute prediction errors and update reward expectations",
     "Compare observed and expected outcomes and update value estimates, while retaining the state/action/outcome lineage.",
     "delta_t=r_t+gamma*V(s_(t+1))-V(s_t); V(s_t) <- V(s_t)+alpha*delta_t",
     "Phasic dopamine in some pathways is modeled as reward-prediction error; dopamine signals are heterogeneous.",
     "reinforcement_learning_and_RPE_framework_supported; scalar TD error is a model",
     ("expected_value", "observed_outcome", "next_state"), ("prediction_error", "updated_action_values")),
    ("P13-10", "Use motivation to sustain cognitive control and effort",
     "Maintain the goal, monitor progress and errors, and persist, change strategy, pause, or stop according to expected benefit and cost.",
     "e*=argmax_e[E(R|e)-kappa*C(e)]; G[t+1]=rho*G[t]+eta*I_goal-zeta*D[t]",
     "Prefrontal and cingulate systems support goal maintenance, monitoring, and control allocation; dopamine influences effort valuation.",
     "effort_allocation_and_control_supported; persistence rule is computational",
     ("goal", "progress", "reward_value", "fatigue", "task_demands"), ("effort_decision", "persistence_state")),
    ("P13-11", "Promote deeper, elaborative processing when engaged",
     "Connect new content to prior knowledge, build relevant associations, generate explanations and predictions, and test them.",
     "A_new=A_old+sum_i w_i*sim(x,new_i); L=sum_i q_i*r_i; epsilon=y-yhat",
     "Hippocampal, semantic cortical, and prefrontal systems interact during elaborative encoding and retrieval practice.",
     "elaborative_processing_benefits_supported; link score is computational",
     ("new_content", "prior_knowledge", "motivation", "effort"), ("elaboration_links", "retrieval_cues")),
    ("P13-12", "Coordinate emotional modulation with hippocampal encoding and consolidation",
     "Apply state-dependent, time-limited modulation to consolidation separately from initial encoding, including replay support.",
     "w[t+1]=w[t]+eta*m[t]*delta_w; m(t)=m0*exp(-t/tau); R(t)=R0*exp(-k*t)",
     "BLA modulation of memory consolidation is supported; hippocampal and cortical consolidation also depend on later experience and sleep.",
     "amygdala_modulated_consolidation_supported; decay and retention curves are simplified models",
     ("memory_trace", "arousal", "modulatory_window", "sleep_replay"), ("consolidated_trace", "retention_strength")),
    ("P13-13", "Retrieve memory and assess emotional effects on recollection",
     "Retrieve by cue similarity and learned strength while keeping vividness, confidence, and objective accuracy separate.",
     "sim(c,m)=cos(c,m); P(m|c) proportional to P(c|m)P(m); calibration=1-abs(confidence-accuracy)",
     "Hippocampal pattern completion interacts with cortical representations; affect and context influence reconstructive retrieval.",
     "cue_dependent_retrieval_and_reconstruction_supported; confidence calibration is computational",
     ("retrieval_cues", "memory_bank", "current_context", "mood"), ("recalled_features", "vividness", "confidence", "accuracy_if_known")),
    ("P13-14", "Detect attentional narrowing and memory trade-offs",
     "Measure attention concentration, central-to-peripheral retention, and contextual completeness; do not equate vividness with completeness.",
     "H(A)=-sum_i p_i*log(p_i); T=M_central/(M_peripheral+eps); C=context_retrieved/context_available",
     "Arousal-biased competition predicts prioritization of salient content with possible suppression of competing details.",
     "attentional_tradeoffs_supported; ratios and entropy are model metrics",
     ("attention_weights", "memory_trace", "retrieval"), ("attention_entropy", "central_peripheral_ratio", "context_completeness")),
    ("P13-15", "Reappraise and update the emotional and motivational model",
     "Compare predicted and observed significance, update cue-outcome associations and goals, and retain earlier episodic content.",
     "epsilon_S=S_observed-S_predicted; w[t+1]=w[t]+alpha*epsilon_S*x; P(H|E) proportional to P(E|H)P(H)",
     "Reappraisal and updating involve distributed prefrontal-amygdala-hippocampal interactions.",
     "reappraisal_and_associative_update_supported; scalar update is computational",
     ("appraisal", "outcome", "prediction_error", "context"), ("revised_appraisal", "updated_associations", "revised_goal")),
    ("P13-16", "Produce an updated state-dependent learning outcome",
     "Return a selective memory trace, updated action values, current goal state, and explicit uncertainty about omissions and distortions.",
     "Z=F(X,A,P,U,E,C); K_new=K_old+omega*delta_K; generalization limited by shared evidence",
     "Episodic, semantic, control, and neuromodulatory systems jointly shape later retrieval and learning.",
     "multisystem_learning_outcome_supported; composite state is an engineering abstraction",
     ("event", "attention", "appraisal", "motivation", "memory", "feedback"), ("updated_memory", "updated_values", "goal_state", "uncertainty")),
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
    """Return the source-aligned 16-stage Point 13 registry."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
