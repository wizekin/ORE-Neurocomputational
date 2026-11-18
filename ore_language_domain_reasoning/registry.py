"""Canonical source-ordered 15-operation registry for Point 15."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one algorithmic operation and its functional/biological scope."""

    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return an operation specification as a plain mapping."""
        return asdict(self)


_ROWS = (
    ("P15-01", "Establish the task and preserve the linguistic signal",
     "Identify modality and task; retain ordered signal segments, timing/prosodic or orthographic evidence, missing information, and input quality before interpretation.",
     "z_t=f_perceptual(x_{t-r:t}); q_signal in [0,1]",
     "Auditory pathways for speech and visual/orthographic pathways for text provide modality-specific input; attention and task goals influence prioritization.",
     "perceptual_encoding_supported; feature vectors and quality scores are computational abstractions",
     ("sensory_stream", "modality", "task_goal", "cognitive_state"), ("ordered_features", "signal_quality", "uncertainty")),
    ("P15-02", "Convert sensory features into language-relevant representations",
     "Extract candidate phonological or orthographic units, preserve ambiguous alternatives, and combine sensory fit with neighboring linguistic context.",
     "score(u_i)=alpha*Fit_sensory(u_i)+beta*Fit_context(u_i)",
     "Auditory/visual perceptual systems extract sound, timing, letter, and word-form features before higher-level linguistic processing.",
     "perceptual_to_linguistic_mapping_supported; candidate scores are model parameters",
     ("sensory_features", "local_context", "lexical_alternatives"), ("candidate_units", "candidate_posteriors")),
    ("P15-03", "Activate the distributed language-selective network",
     "Route language-relevant features into a recurrent linguistic state and keep linguistic-form confidence separate from real-world interpretation confidence.",
     "L_t=F_L(z_t,L_{t-1},G_lang)",
     "A distributed, commonly left-lateralized frontotemporal language network supports linguistic representations; it is not one isolated language center.",
     "language_selective_network_supported; recurrent state is a computational abstraction",
     ("language_features", "prior_language_state", "language_knowledge"), ("active_language_state", "language_activation")),
    ("P15-04", "Retrieve lexical knowledge",
     "Access word-form candidates and learned linguistic properties, combine sensory evidence with context/frequency, and retain plausible alternatives until evidence is sufficient.",
     "P(w|z,c) proportional to P(z|w)*P(w|c)",
     "Frontotemporal language circuitry supports access to stored lexical information; competing activation is a useful model-level description.",
     "lexical_access_supported; Bayesian-style posterior is normative/computational",
     ("candidate_units", "lexicon", "linguistic_context", "learned_lexical_strengths"), ("lexical_candidates", "lexical_confidence")),
    ("P15-05", "Retrieve learned constructions and structural constraints",
     "Retrieve word-order, agreement, phrase, dependency, and constructional patterns; update partial structure incrementally and preserve unresolved dependencies.",
     "T*=argmax_T P(T|w_1:t,G_lang)",
     "The language-selective network contributes to learned syntactic and constructional constraints; exact parsing mechanisms remain model-dependent.",
     "incremental_structural_processing_supported; ranked parse candidates are computational",
     ("lexical_candidates", "construction_patterns", "linguistic_context"), ("linguistic_structure", "structural_alternatives")),
    ("P15-06", "Compose sentence meaning from linguistic units",
     "Combine lexical and constructional contributions according to structural relations while preserving more than one plausible composition when needed.",
     "m_t=Compose(m_{t-1},ell_t,r_t)",
     "Distributed frontotemporal language-network interactions support compositional linguistic interpretation.",
     "compositional_language_processing_supported; symbolic proposition format is computational",
     ("lexical_candidates", "linguistic_structure"), ("sentence_meaning", "explicit_content")),
    ("P15-07", "Retrieve relevant conceptual knowledge",
     "Cue distributed semantic/conceptual memory using the composed linguistic representation, retrieve task-relevant properties and relations, and keep them distinct from what was explicitly stated.",
     "K_t=Retrieve(ConceptCue(L_t),K_long,Context_t); R(k)=Similarity(k,C)*Reliability(k)",
     "Conceptual knowledge is distributed across modality-linked and associative systems and interacts with language processing.",
     "conceptual_retrieval_supported; relevance metric is a computational approximation",
     ("sentence_meaning", "conceptual_memory", "situation"), ("activated_concepts", "conceptual_confidence")),
    ("P15-08", "Estimate processing demand independently from sentence content",
     "Estimate ambiguity, working-memory maintenance, novelty, dependency load, and deliberate-comparison requirements as a control-demand signal.",
     "D_t=a*A_t+b*W_t+c*N_t+d*Q_t",
     "Multiple-demand/control systems respond to cognitive difficulty across verbal and non-verbal tasks; they do not thereby become the core syntax/lexical engine.",
     "domain_general_control_demand_supported; scalar demand is a model abstraction",
     ("linguistic_ambiguity", "dependency_load", "novelty", "task_goal", "cognitive_state"), ("demand_score", "capacity_estimate")),
    ("P15-09", "Recruit working-memory and cognitive control when needed",
     "Compare demand with available capacity; allocate attention, maintain dependencies, and compare competing interpretations only when the task requires additional control.",
     "C_recruit=sigma(gamma*(D_t-Capacity_t))",
     "Distributed bilateral frontal/parietal multiple-demand and control-related systems support maintenance, selection, and flexible task control.",
     "multiple_demand_control_supported; sigmoid recruitment curve is a computational analogy",
     ("demand_score", "capacity_estimate", "candidate_interpretations"), ("control_allocation", "maintenance_plan")),
    ("P15-10", "Separate linguistic interpretation from deliberate reasoning",
     "Preserve directly encoded content; apply explicit reasoning rules only when the task requires calculation, proof, comparison, or multi-step inference; record conclusions as inferred.",
     "R=ApplyRules(K,L,C); said_or_encoded != inferred",
     "Language-selective processing and domain-general reasoning/control are functionally distinguishable while interacting.",
     "language_and_reasoning_distinction_supported; rule execution is a software approximation",
     ("sentence_meaning", "conceptual_knowledge", "reasoning_rules", "task_goal"), ("said_or_encoded", "inferred", "reasoning_result")),
    ("P15-11", "Recruit social-pragmatic inference conditionally",
     "Estimate speaker knowledge, beliefs, goals, and likely communicative intention only when task or context makes intention inference relevant; retain multiple hypotheses.",
     "P(I|U,S,B) proportional to P(U|I,S,B)*P(I|S,B)",
     "Medial prefrontal, temporoparietal, posterior-midline, and other social-cognition systems may contribute to perspective and intention inference.",
     "social_pragmatic_inference_supported; posterior over intentions is computational",
     ("literal_meaning", "situation", "speaker_model", "task_goal"), ("intention_hypotheses", "pragmatic_interpretation")),
    ("P15-12", "Integrate outputs while preserving source and inference boundaries",
     "Coordinate language, conceptual, control/reasoning, and pragmatic results without collapsing their identities; mark conclusions that depend on background assumptions.",
     "M=Integrate(L,K,C,S,Goal); gated/recurrent interactions",
     "Language, semantic, control, and social-cognition systems exchange information while retaining partially distinct contributions.",
     "network_cooperation_supported; integration gates are computational",
     ("L", "K", "C", "S", "task_goal"), ("integrated_interpretation", "provenance_map")),
    ("P15-13", "Detect prediction mismatch and revise the affected layer",
     "Compare predicted with observed evidence, classify perceptual/lexical/structural/conceptual/pragmatic mismatch, and reopen only affected representations.",
     "E_t=Observed_t-Predicted_t; revise layer j when |E_j|>theta_j",
     "Incremental comprehension can be revised when later words, prosody, context, or feedback conflict with an earlier interpretation.",
     "prediction_error_and_reanalysis_supported; error classes and thresholds are computational",
     ("expectations", "observed_signal", "integrated_interpretation", "new_evidence"), ("mismatch_type", "revision", "updated_uncertainty")),
    ("P15-14", "Produce a response, action, or stable comprehension state",
     "Select whether to retain, answer, follow an instruction, or formulate language; preserve uncertainty and avoid adding details unsupported by the input.",
     "Action*=argmax_a Utility(a|M,Goal); response constrained by evidence",
     "Language production engages language, conceptual, motor, and task-related systems; comprehension may end without an overt response.",
     "task-oriented response selection_supported; utility policy is computational",
     ("integrated_interpretation", "task_goal", "uncertainty"), ("task_result", "response_plan")),
    ("P15-15", "Update learned representations from reliable repeated experience",
     "Retain useful lexical/constructional evidence, reduce erroneous associations after correction, require repeated/reliable evidence for general updates, and make changes available to later tasks.",
     "theta_{t+1}=theta_t+eta*reliability*repetition_weight*UpdateSignal_t",
     "Experience-dependent cortical language learning and hippocampal/medial-temporal episodic support interact across timescales.",
     "experience_dependent_learning_supported; bounded parameter updates are computational",
     ("feedback", "linguistic_representation", "conceptual_interpretation", "prior_learning_state"), ("updated_lexicon", "construction_strengths", "learning_record")),
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
    """Return all Point 15 operations in their original source order."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
