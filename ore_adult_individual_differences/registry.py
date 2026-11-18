"""Canonical source-ordered 15-operation registry for Point 17."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class AdultOperationSpec:
    """Describe an adult individual-differences operation and its biological evidence boundary."""

    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a plain mapping for documentation and operation logs."""
        return asdict(self)


_ROWS = (
    ("P17-01", "Detect the task and establish its processing goal",
     "Represent the current objective as a normalized vector over detail, gist, relations, procedure, and source accuracy; prioritize only information relevant to the requested output.",
     "G=(g_detail,g_gist,g_relation,g_procedure,g_source); Rel(f_i,G)=sum_j g_j*m_ij",
     "Prefrontal and parietal control networks maintain task goals and bias distributed perceptual, language, and memory processing.",
     "goal_sensitive_attention_supported; goal vectors and feature relevance are computational abstractions",
     ("content", "task_instruction", "desired_output"), ("goal_vector", "feature_priorities")),
    ("P17-02", "Estimate current processing conditions and available resources",
     "Estimate task-specific attention and working-memory support from present distraction, fatigue, stress, motivation, and external scaffolding, never as a permanent ability label.",
     "A_t=sigmoid(theta_0+theta_1*focus-theta_2*D-theta_3*fatigue); W_t=W_base*A_t*q_t",
     "Distributed prefrontal/frontoparietal attention and working-memory systems are state-sensitive and affected by interference and task context.",
     "state_and_task_sensitivity_supported; scalar attention and availability are model estimates",
     ("attention_baseline", "distraction", "fatigue", "stress", "motivation", "external_support"), ("attention_estimate", "wm_availability")),
    ("P17-03", "Segment input into meaningful, source-linked units",
     "Parse words, phrases, propositions, event features, and procedural steps in original order; preserve ambiguity and attach source and confidence metadata.",
     "P(boundary_i=1|x)=sigmoid(alpha_0+sum_j alpha_j*cue_ij)",
     "Distributed auditory/visual, language, and temporal integration systems segment continuous material using perceptual, linguistic, and contextual cues.",
     "incremental_segmentation_supported; boundary posteriors are model variables",
     ("ordered_input", "syntax", "prosody", "context"), ("ordered_units", "unit_confidence", "source_links")),
    ("P17-04", "Retrieve relevant prior knowledge and domain expertise",
     "Retrieve definitions, examples, relations, procedures, and schemas relevant to the current domain and goal; score relevance, reliability, recency, and conflict.",
     "Score(K_k|x,G)=w_s*Sim+w_g*GoalFit+w_r*Reliability-w_c*Conflict; P(K_k)=softmax(Score/tau)",
     "Cortical semantic networks, medial-temporal memory, and prefrontal selection interact in knowledge retrieval.",
     "prior_knowledge_and_schema_retrieval_supported; similarity scores are not truth scores",
     ("domain", "units", "goal", "semantic_memory", "expertise"), ("ranked_schemas", "schema_uncertainty")),
    ("P17-05", "Convert familiar details into chunks when schemas justify compression",
     "Compare the representation cost of separate elements with a schema chunk; use a chunk only when fit and confidence are adequate, while preserving exceptions and novel details.",
     "Cost_separate=sum_i c_i; Cost_chunk=c_schema+c_exceptions; use chunk if Cost_chunk<Cost_separate and match>threshold",
     "Expert knowledge can reorganize effective processing units and reduce maintenance demands; the advantage is domain-specific.",
     "expert_chunking_supported; cost optimization is an explanatory analogy",
     ("units", "expertise", "schema_match", "exceptions"), ("chunks", "uncompressed_details", "effective_unit_count")),
    ("P17-06", "Estimate element interactivity and working-memory load",
     "Estimate independent units, dependencies, unresolved references, and concurrent alternatives; compare demand with task-specific maintenance support.",
     "Load=sum_i c_i+lambda*sum_(i<j) a_ij; Pressure=Load/(W+epsilon)",
     "Frontoparietal working-memory and attention systems coordinate task-relevant elements; semantic schemas change the effective units.",
     "element_interactivity_and_load_supported; numeric edge costs are model parameters",
     ("units", "chunks", "dependency_graph", "wm_availability"), ("load_estimate", "load_pressure", "decomposition_need")),
    ("P17-07", "Account for processing speed and temporal coordination",
     "Estimate times for perception, retrieval, integration, and decision separately; identify task bottlenecks and adjust pace or support without equating speed with knowledge or intelligence.",
     "T_total=sum_k T_k; Avail(delta_t)=exp(-delta_t/tau)",
     "Processing speed and temporal coordination contribute to performance across perceptual, retrieval, and integration operations.",
     "processing_speed_effects_supported; exponential availability is a simple time-pressure model",
     ("component_speed", "observed_times", "pace", "time_budget", "load"), ("component_times", "bottlenecks", "time_pressure")),
    ("P17-08", "Weight language cues using proficiency and experience",
     "Use learned cue reliability for the active language and context; preserve competing interpretations and avoid assuming bilingual experience has a single universal effect.",
     "P(h|c,L) proportional to P(h|L)*product_i P(c_i|h,L)^omega_i(L)",
     "Distributed language networks are shaped by language proficiency, exposure, and use; cue weighting differs across language experiences.",
     "experience_sensitive_language_prediction_supported; likelihood weights are computational",
     ("language", "proficiency", "cue_evidence", "candidate_hypotheses"), ("language_posteriors", "cue_weights")),
    ("P17-09", "Incorporate demonstrated cultural and lived-experience associations",
     "Use contextual conventions and learned associations only when supported by individual history and current cues; represent multiple plausible contexts instead of applying group stereotypes.",
     "P(h|x,C)=P(x|h,C)P(h|C)/P(x|C); P(h|x)=sum_c P(h|x,c)P(c|x)",
     "Distributed semantic and social-context systems activate learned conventions and personal associations in context-sensitive ways.",
     "context_and_experience_effects_supported; convention mixtures are computational approximations",
     ("cultural_cues", "individual_context_memory", "situational_context"), ("context_hypotheses", "context_sensitive_posteriors")),
    ("P17-10", "Integrate input with active knowledge and preserve provenance",
     "Bind new information to the best-supported schema while separating observed content, retrieved knowledge, inferences, and unresolved assumptions.",
     "Posterior(h) proportional to Likelihood(x|h)*Prior(h|K,G,C)^beta",
     "Hippocampal/medial-temporal binding, cortical semantic knowledge, and prefrontal control interact during comprehension and new relational learning.",
     "integrative_comprehension_supported; prior tempering is a safeguard heuristic",
     ("observed_units", "schema_candidates", "language_results", "cultural_results"), ("situation_model", "provenance_map")),
    ("P17-11", "Monitor interference and maintain competing interpretations",
     "Track uncertainty and hypothesis competition; defer commitment when alternatives remain close and error cost is high, and reopen hypotheses when later evidence conflicts.",
     "H(H)=-sum_i P(h_i)log P(h_i); Conflict=1-(P(best)-P(second))",
     "Attentional control, working-memory maintenance, and monitoring systems manage interference and competing representations.",
     "conflict_monitoring_supported; entropy thresholds are task-level parameters",
     ("candidate_posteriors", "interference", "error_cost"), ("live_hypotheses", "uncertainty", "commitment_state")),
    ("P17-12", "Adapt processing support to the current person-task match",
     "Select scaffolding dynamically: more definitions, worked examples, and decomposition when relevant knowledge is low; reduce redundant guidance for experts while retaining exception checks.",
     "Support=sigmoid(gamma_0+gamma_1*Load-gamma_2*Expertise)",
     "Prefrontal control and memory systems coordinate goal-sensitive learning; instructional support has an expertise-reversal interaction with task demands.",
     "expertise_reversal_supported; support function is an adaptive heuristic",
     ("expertise", "load", "task_goal", "recent_errors", "learner_preference"), ("support_plan", "pace_adjustment", "externalization_plan")),
    ("P17-13", "Separate gist, episodic detail, and source retrieval",
     "When memory specificity matters, allocate explicit encoding and retrieval cues to who/what/where/when and source; preserve gist separately and avoid universal age penalties.",
     "M=(M_gist,M_detail,M_source); P_retrieve(M_j)=sigmoid(eta_0j+eta_1j*CueQuality+eta_2j*EncodingTime-eta_3j*Interference)",
     "Normal aging shows heterogeneous average changes in speed and some episodic/source details while semantic knowledge and gist can remain relatively preserved.",
     "aging_and_memory_specificity_supported; age is contextual, not a deterministic decline factor",
     ("age_context", "memory_goal", "gist_cues", "detail_cues", "source_cues", "encoding_time"), ("gist_probability", "detail_probability", "source_probability")),
    ("P17-14", "Update knowledge while protecting against false assimilation",
     "Classify new evidence as expected variation, exception, contradiction, or possible input error; update by reliability and repetition while retaining episodic source provenance.",
     "K_(t+1)=K_t+alpha_t*(x_t-K_t); PE_t=Observed_t-Predicted_t",
     "Hippocampal episodic binding and cortical semantic learning support updates; expectations can bias memory and assimilation.",
     "incremental_learning_and_schema_bias_supported; update rates and residuals are model-level",
     ("new_facts", "prior_schema", "reliability", "repetition", "source"), ("updated_knowledge", "exceptions", "contradictions")),
    ("P17-15", "Calibrate confidence and choose the next action",
     "Calibrate confidence using candidate separation, evidence quality, schema reliability, provenance completeness, and inference share; choose to proceed, slow down, retrieve, clarify, or revisit.",
     "Conf*=P(best)*(1-U)*Q; calibration considers schema reliability and provenance completeness",
     "Monitoring and decision systems help select a response, additional evidence, review, or clarification.",
     "confidence_monitoring_supported; final confidence is an explicit design heuristic",
     ("best_posterior", "uncertainty", "evidence_quality", "schema_reliability", "provenance"), ("calibrated_confidence", "next_action", "comprehension_result")),
)


OPERATION_REGISTRY = tuple(
    AdultOperationSpec(
        operation_id=row[0], name=row[1], purpose=row[2], mathematical_form=row[3],
        biological_reference=row[4], evidence_level=row[5], inputs=row[6], outputs=row[7],
    )
    for row in _ROWS
)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return all 15 adult individual-difference operations in source order."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
