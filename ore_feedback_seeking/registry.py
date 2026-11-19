"""Source-aligned operation registry for the Point 07 feedback-seeking algorithm."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one algorithm step, its equations, and its candidate biological systems."""

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
        """Return a JSON-compatible dictionary representation of this operation."""
        return asdict(self)


_ROWS = (
    ("Define the comprehension target", "Decompose the goal into weighted, checkable criteria and score current coverage.", "C=sum_i(w_i*m_i), sum_i(w_i)=1", "Prefrontal and frontoparietal control networks maintain task goals and bias attention; no single target register is implied.", "goal_maintenance_supported; rubric weighting is computational", ("learning_target", "criteria"), ("target_vector", "completeness")),
    ("Generate a candidate interpretation", "Preserve the learner's answer, reasoning chain, competing hypotheses, and uncertainty as an inspectable candidate.", "P(H|E,K) proportional to P(E|H,K)P(H|K)", "Temporal and parietal semantic systems interact with working memory, frontal control, and hippocampal relational retrieval.", "distributed_retrieval_supported; explicit posterior is computational", ("candidate_response", "knowledge", "reasoning_chain"), ("candidate_model", "hypothesis_prior")),
    ("Monitor comprehension online", "Combine confidence, local inconsistencies, and ambiguity to decide whether a focused check is warranted.", "u=a(1-c)+b*d+g*A", "Medial-frontal performance monitoring and prefrontal control contribute to uncertainty/conflict monitoring; signals depend on task type.", "performance_monitoring_supported; scalar uncertainty is computational", ("confidence", "reasoning_chain", "ambiguity"), ("uncertainty", "check_required")),
    ("Estimate confidence separately from correctness", "Represent confidence and hypothesis entropy independently from correctness; compute Brier score only when a correctness label exists.", "Brier=(c-y)^2; H=-sum_i p_i log(p_i)", "Metacognitive confidence draws on evidence and processing fluency across distributed systems; there is no neural Brier calculator.", "metacognition_supported; calibration metrics are computational", ("confidence", "hypothesis_prior", "verified_correctness"), ("confidence_estimate", "entropy", "brier_score")),
    ("Select a suitable feedback route", "Rank sources by expected information gain, domain reliability, relevance, specificity, evidence quality, noise, and access cost.", "s*=argmax_s(EIG_s-lambda*K_s)", "Goals and uncertainty guide orienting and source selection; source reliability is learned from experience, not one global neural counter.", "goal_guided_source_selection_supported; EIG utility is computational", ("feedback_sources", "hypothesis_prior"), ("source_scores", "selected_source")),
    ("Obtain actionable feedback", "Receive evidence that identifies the mismatch and preserves the correct parts of the learner's answer.", "D=sum_i(w_i*d(r_i,z_i))", "External feedback enters through sensory/language pathways and is compared with maintained goals and remembered responses.", "feedback_comparison_supported; weighted semantic discrepancy is computational", ("candidate_response", "feedback_evidence", "criteria"), ("feedback_items", "discrepancy")),
    ("Verify feedback quality and independence", "Evaluate claim relevance, actionability, evidence, source reliability, ambiguity, and dependence before updating beliefs.", "P(H|F) proportional to P(F|H)P(H); V=R+A+E+C-N", "Source evaluation recruits semantic knowledge, memory, and social/contextual reasoning; no exact posterior circuit is established.", "testimony_evaluation_supported; Bayesian aggregation is computational", ("feedback_items", "feedback_sources", "hypothesis_prior"), ("evidence_quality", "posterior", "trusted_feedback")),
    ("Classify and localize the error", "Estimate error-class probabilities and locate the earliest low-consistency reasoning node.", "k*=argmax_k P(k|e,context); i*=min{i:q_i<tau}", "Medial-frontal monitoring can signal a need for adjustment; domain-specific cortical and memory representations help identify the content error.", "error_monitoring_supported; classifier and graph search are computational", ("discrepancy", "reasoning_chain", "error_hints"), ("error_class", "first_divergence")),
    ("Separate knowledge from performance failure", "Estimate durable knowledge separately from fatigue, distraction, load, and time pressure; retain uncertainty when labels are absent.", "P(correct)=sigmoid(theta_K-theta_D-theta_L)", "Working-memory and attentional limits modulate performance while semantic knowledge can remain available; the logistic model is a proxy.", "state-dependent performance_supported; logistic model is computational", ("knowledge_components", "performance_context", "retrieval_correctness"), ("knowledge_strength", "performance_load", "failure_type")),
    ("Apply the smallest justified correction", "Update only named affected components, preserve unrelated knowledge, and keep unsupported corrections provisional.", "theta'=theta-eta*grad(L) or G'=G+DeltaG", "Learning-related plasticity changes distributed representations; this is not literal global gradient descent in the brain.", "plasticity_supported; bounded local update is computational", ("trusted_feedback", "affected_components", "corrected_components"), ("component_updates", "revised_knowledge")),
    ("Integrate correction with evidence and prior knowledge", "Link corrected concepts to supporting concepts, sources, conditions, and related knowledge without erasing provenance.", "w_ij'=w_ij+eta*delta_ij", "Hippocampal relational binding interacts with distributed cortical semantic representations; graph edges are software structures.", "relational_binding_supported; graph-weight update is computational", ("revised_knowledge", "supporting_concepts", "conditions", "provenance"), ("knowledge_graph", "retrieval_cues")),
    ("Explain the correction and expose its boundaries", "Assess accuracy, relational coherence, evidence support, and boundary awareness; use positive, negative, and boundary examples.", "Q=w_A*A+w_R*R+w_E*E+w_B*B", "Explanation recruits semantic retrieval, language, working memory, and control networks rather than one isolated region.", "self_explanation_supported; quality composite is computational", ("corrected_response", "supporting_concepts", "counterexample", "boundary_examples"), ("explanation_model", "explanation_quality")),
    ("Run an independent retrieval check", "Test whether the correction can be reconstructed without seeing the feedback and quantify residual error when labels exist.", "E_post=L(yhat,y*); DeltaE=E_pre-E_post", "Retrieval reactivates memory representations and can support practice-dependent plasticity; scalar improvement is not a neural signal.", "retrieval_practice_supported; loss difference is computational", ("retrieval_response", "corrected_response", "retrieval_correctness"), ("retrieval_success", "post_error", "improvement")),
    ("Recalibrate confidence", "Compare confidence with independently verified outcomes over episodes and report Brier/ECE only with labels.", "ECE=sum_b(n_b/N)*|acc_b-conf_b|", "Metacognitive monitoring can be refined through experience with outcomes; the exact computation remains uncertain.", "metacognitive_calibration_supported; ECE is analytical", ("confidence_history", "retrieval_correctness"), ("calibration", "updated_confidence")),
    ("Detect cognitive overload and chunk feedback", "Estimate feedback complexity from item count, interaction density, and terminology uncertainty; split large corrections.", "L_f=a*N+b*I+g*U; chunks=ceil(L_f/tau)", "Attention and working memory have capacity limits; affective state can alter control and engagement.", "capacity_limits_supported; load score and chunk count are computational", ("feedback_items", "interaction_density", "terminology_uncertainty"), ("feedback_load", "chunks")),
    ("Test transfer to new contexts", "Evaluate application to structurally related but surface-different examples and explicit counterexamples.", "T=(1/M)*sum_j accuracy_j", "Relational abstraction may recruit hippocampal recombination and cortical semantic integration; probe scores are behavioral outputs.", "transfer_supported; mean score is computational", ("transfer_probes", "corrected_relation"), ("transfer_score", "transfer_status")),
    ("Schedule delayed retention and recurrence checks", "Estimate accessibility decay, schedule a future retrieval, and track whether the original misconception recurs.", "A(t)=A0*exp(-lambda*t); R=count_recurrence/count_checks", "Delayed retrieval and sleep-related consolidation involve distributed hippocampal-cortical processes; scheduling is a software function.", "retention_and_consolidation_supported; exponential decay is an approximation", ("delayed_check_days", "retrieval_success", "error_history"), ("accessibility", "review_queue", "recurrence")),
    ("Update future monitoring and strategy", "Update exponentially weighted error-category frequency and strategy outcomes to trigger earlier, better-targeted checks.", "f_k(t)=rho*f_k(t-1)+(1-rho)*I(C_t=k)", "Repeated monitoring and control can shape future attention and strategy selection through distributed learning systems.", "adaptive_control_supported; rolling frequency is computational", ("error_class", "strategy_outcome", "error_history"), ("monitoring_policy", "final_state")),
)

OPERATION_REGISTRY = tuple(
    OperationSpec(
        operation_id=f"P07-{index:02d}", source_step=index, name=row[0], purpose=row[1],
        mathematical_form=row[2], biological_reference=row[3], evidence_level=row[4],
        inputs=row[5], outputs=row[6],
    )
    for index, row in enumerate(_ROWS, start=1)
)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return all eighteen Point 07 operations in source order."""
    return [operation.to_dict() for operation in OPERATION_REGISTRY]
