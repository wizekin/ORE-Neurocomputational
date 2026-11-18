"""Canonical 17-operation registry for the lifespan-developmental algorithm (Point 16)."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class DevelopmentalOperationSpec:
    """Describe one developmental operation, its model equation, and evidence boundary."""

    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a plain mapping for documentation and runtime operation traces."""
        return asdict(self)


_ROWS = (
    ("P16-01", "Establish broad early auditory sensitivity",
     "Represent acoustic contrasts in a multidimensional feature space and retain broad sensitivity before extensive environment-specific tuning.",
     "D(x_a,x_b)=sqrt((x_a-x_b)^T W (x_a-x_b)); P(diff)=sigmoid(beta*(D-theta))",
     "Cochlea, brainstem auditory pathways, thalamus, and distributed auditory cortex; broad responsiveness is not adult-like phoneme knowledge.",
     "auditory_discrimination_supported; distances and thresholds are computational abstractions",
     ("acoustic_frames", "audio_samples", "initial_sensitivity"), ("acoustic_contrasts", "broad_sensitivity")),
    ("P16-02", "Extract time-varying speech features",
     "Analyze successive acoustic windows, preserve rapid transitions and slower rhythm, and maintain uncertainty where signal features are ambiguous.",
     "X(tau,omega)=STFT(x); z_t=alpha*z_(t-1)+(1-alpha)*phi(x_t)",
     "Ascending auditory pathways and auditory cortical population responses support multi-timescale acoustic analysis.",
     "auditory_feature_extraction_supported; supplied features and digital transform are measurement/model abstractions",
     ("audio_samples", "sample_rate", "acoustic_frames"), ("frame_features", "temporal_envelope", "uncertainty")),
    ("P16-03", "Build provisional speech-sound representations",
     "Compare across speakers and speaking conditions, cluster recurring patterns, and revise category-like prototypes as new examples arrive.",
     "p(x)=sum_k pi_k*p(x|C_k); mu_k'=mu_k+eta*r_tk*(x_t-mu_k)",
     "Superior temporal auditory regions contribute to speech-sound pattern processing; prototype clustering is not a literal neural algorithm.",
     "experience_sensitive_sound_categories_supported; mixture and prototype updates are computational",
     ("frame_features", "speaker_context", "phonetic_feedback"), ("phonetic_prototypes", "category_confidence")),
    ("P16-04", "Learn environmental distributions and recurring regularities",
     "Estimate sound and syllable frequencies, co-occurrences, and conditional transitions, giving repeated patterns more weight than isolated coincidences.",
     "P_hat(a)=(n(a)+alpha)/(N+alpha*K); H(X)=-sum P log P; I(X;Y)=sum P(x,y) log[P(x,y)/(P(x)P(y))]",
     "Auditory and statistical-learning systems accumulate distributional regularities across exposure.",
     "statistical_learning_supported; explicit counts/probabilities are model-level state",
     ("syllables", "acoustic_features", "language_context"), ("frequencies", "transition_probabilities", "uncertainty")),
    ("P16-05", "Detect candidate word boundaries in continuous speech",
     "Combine syllable-transition evidence with stress, rhythm, pauses, phonotactics, familiar forms, and context; keep boundaries probabilistic.",
     "TP(B|A)=P(A,B)/P(A); S_boundary=sum_i w_i*c_i",
     "Temporal speech processing and prosodic/syllabic cue integration support early segmentation.",
     "candidate_word_segmentation_supported; cue weights and threshold are explanatory parameters",
     ("transition_counts", "syllables", "prosody", "context"), ("candidate_words", "boundary_confidence")),
    ("P16-06", "Tune perception to the language environment",
     "Strengthen sensitivity to recurrent native-language distinctions while preserving graded access to unfamiliar contrasts and avoiding all-or-none deletion.",
     "P(C_k|x)=P(x|C_k)P(C_k)/sum_j P(x|C_j)P(C_j); theta'=theta+eta*delta",
     "Experience-dependent auditory and temporal-language network plasticity; perceptual narrowing varies by contrast, language, age, and task.",
     "perceptual_tuning_supported; category strengths are computational model parameters",
     ("native_language", "contrastive_exposure", "phonetic_feedback"), ("native_contrast_strength", "unfamiliar_contrast_retention")),
    ("P16-07", "Integrate social and multimodal evidence during learning",
     "Combine speech, visual referents, gaze, gesture, and contingent communicative interaction while retaining ambiguity when cues disagree.",
     "P(R|A,V,G,C) proportional to P(A|R)P(V|R)P(G|R)P(C|R)P(R)",
     "Superior temporal, visual/multisensory, and social-attention systems provide interacting information sources.",
     "multimodal_word_learning_supported; cue independence is only an approximation",
     ("speech_candidates", "scene_referents", "gaze", "gesture", "contingency"), ("referent_posterior", "cue_confidence")),
    ("P16-08", "Map recurring word forms to candidate meanings",
     "Link recognized word forms to entities, actions, properties, and relations using scene evidence, joint attention, syntax, and prior knowledge.",
     "P(m_i|w,s,c) proportional to P(w|m_i,s,c)P(m_i|c); delta A=eta*q*r*error",
     "Temporal semantic and multisensory systems contribute to grounded word-referent learning.",
     "grounded_word_learning_supported; association update is schematic",
     ("word_forms", "scene", "syntax", "existing_lexicon"), ("candidate_meanings", "meaning_confidence")),
    ("P16-09", "Accelerate word learning from accumulated knowledge",
     "Use known words, categories, syntactic cues, and social support to reduce ambiguity when learning later vocabulary.",
     "eta_t=eta_0+alpha*L_t+beta*C_t+gamma*S_t; H(M|w,c)=-sum P(m|w,c)log P(m|w,c)",
     "Developing temporal semantic and language systems benefit from prior lexical and conceptual structure.",
     "vocabulary_acceleration_supported; efficiency gain and entropy are model variables",
     ("candidate_word", "lexicon", "concepts", "syntactic_hints"), ("lexical_entry", "learning_efficiency")),
    ("P16-10", "Build grounded concepts and hierarchical categories",
     "Identify stable features across instances, discount incidental features, form superordinate/subordinate categories, and link semantic relations.",
     "c_k=[w_1 f_1,...,w_d f_d]; Sim(x,c)=x.c/(||x||||c||); P(child)=P(child|parent)P(parent)",
     "Distributed temporal semantic, sensory, and action-linked systems contribute to concept formation.",
     "grounded_concept_formation_supported; feature vectors/hierarchy are computational",
     ("concept_examples", "sensory_features", "action_features"), ("concept_prototypes", "relations", "category_confidence")),
    ("P16-11", "Stabilize learning through developing memory systems",
     "Bind specific experiences and sources, reactivate useful episodes, and gradually consolidate recurring relations into semantic knowledge.",
     "W_ij(t+1)=W_ij(t)+eta*x_i*x_j; replay-weighted repeated evidence",
     "Hippocampal/medial-temporal episodic systems interact with cortical semantic learning across timescales.",
     "episodic_and_semantic_memory_supported; associative update is a coarse approximation",
     ("word", "referent", "context", "speaker", "experience_history"), ("episode", "semantic_association")),
    ("P16-12", "Develop working-memory maintenance and updating",
     "Select task-relevant items, maintain and update them through recurrent activity, resist distraction, and allow chunking to reduce coordination cost.",
     "h_t=lambda*h_(t-1)+B*u_t-G*d_t; sum_i a_i <= K_t",
     "Distributed frontoparietal and parietal attention systems contribute to verbal, spatial, and event maintenance.",
     "working_memory_development_supported; capacity is flexible and task-dependent",
     ("working_memory_items", "goal", "distractors", "schemas"), ("active_items", "load", "effective_capacity")),
    ("P16-13", "Refine executive control operations",
     "Maintain goals, prioritize relevant representations, inhibit competing responses, update task rules, shift strategy, and learn from errors.",
     "Q_i=w_g*G_i+w_r*R_i-w_d*D_i; P(a_i)=softmax(Q_i/tau); delta w=eta*error*x",
     "Prefrontal and distributed frontoparietal systems contribute to inhibition, updating, flexible shifting, and monitoring.",
     "executive_control_development_supported; action scores/updates are computational",
     ("goal", "actions", "rule", "working_memory", "feedback"), ("selected_action", "control_weights", "switch_state")),
    ("P16-14", "Refine neural circuits through activity-dependent development",
     "Strengthen repeatedly coactive useful pathways, slowly weaken underused links where evidence accumulates, and model myelination-related timing changes.",
     "delta w_ij=eta*g(A_i,A_j,delta_t,M,C); L=L_task+lambda*Omega(W); T_cond approximately L/v",
     "Activity-dependent synaptic plasticity, circuit refinement, and distributed white-matter/myelination development.",
     "developmental_circuit_refinement_supported; regularization is an analogy, not biological pruning",
     ("circuit_activations", "coactivity", "active_pathways", "path_lengths"), ("connectivity", "pruning_history", "conduction_delay")),
    ("P16-15", "Integrate hippocampal memory with prefrontal control",
     "Retrieve relevant prior episodes, identify transferable parts, compare candidate actions with current and long-term goals, and coordinate memory with control gradually.",
     "Q(a|s)=sum_m P(m|s)U(a,m); delta=r+gamma*V(s')-V(s)",
     "Hippocampal and prefrontal systems become more effectively coordinated through development and experience; no single adolescent cutoff is assumed.",
     "memory_guided_control_supported; utility/prediction-error equations are theoretical analogies",
     ("episodic_memory", "working_memory", "goal", "action_candidates", "age_context"), ("retrieved_memories", "integrated_value")),
    ("P16-16", "Support increasingly complex multi-step comprehension and reasoning",
     "Represent subtasks and dependencies, retain completed and pending steps, update plans when constraints change, and suppress premature action.",
     "G=(V,E); S_t=(V_done,V_pending,G_active); load=sum c_i+rho*sum r_ij",
     "Working-memory, prefrontal control, semantic schemas, and hippocampal retrieval interact to support multi-step tasks.",
     "multi_step_reasoning_supported; dependency graph is computational",
     ("instruction", "plan_steps", "dependencies", "schema", "intermediate_results"), ("completed_steps", "pending_steps", "goal_check")),
    ("P16-17", "Run continuous developmental calibration",
     "Track multiple developmental domains separately, adjust task difficulty, pacing, repetition, and support using persistent performance evidence rather than age-only rules.",
     "theta_(t+1)=theta_t+eta_t*e_t; d_(t+1)=d_t+k*(p_t-p*)",
     "Distributed developmental and learning systems adapt to experience; developmental measures are observations, not one global maturity mechanism.",
     "multidimensional_developmental_calibration_supported; target-success controller is instructional analogy",
     ("domain_performance", "task_difficulty", "age_context", "experience_history"), ("domain_parameters", "future_support", "task_difficulty")),
)


OPERATION_REGISTRY = tuple(
    DevelopmentalOperationSpec(
        operation_id=row[0], name=row[1], purpose=row[2], mathematical_form=row[3],
        biological_reference=row[4], evidence_level=row[5], inputs=row[6], outputs=row[7],
    )
    for row in _ROWS
)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return the 17 developmental operations in their exact source order."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
