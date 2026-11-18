"""Canonical 17-step source registry for Point 10 schema-guided inference."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one inference stage, its mathematical abstraction, and candidate neural systems."""

    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a source-aligned registry entry as a plain dictionary."""
        return asdict(self)


_ROWS = (
    ("P10-01", "Construct current-case evidence", "Extract sensory features and relations; attach reliability and observed/contradicted/unknown status without inserting predictions.", "x_i in {present, absent, unknown}; r_i in [0,1]", "Modality-specific sensory association cortices and attention networks contribute distributed feature evidence.", "sensory_feature_processing_supported; sparse record is computational",
     ("sensory_input", "context"), ("evidence_record", "reliability", "provenance")),
    ("P10-02", "Activate candidate schemas", "Retrieve a limited candidate set using partial cues, context, and goals; retain an unknown/other alternative.", "A(S)=b+sum_i r_i*w_i*g(x_i,S)+context_gain; softmax(A/tau)", "Temporal semantic systems interact with medial prefrontal and control networks during prior-guided interpretation.", "schema_guided_retrieval_supported; score is computational",
     ("evidence_record", "context", "goal", "schema_library"), ("candidate_schemas", "candidate_activation")),
    ("P10-03", "Retrieve relational schema structure", "Load roles, slots, relations, constraints, transitions, and episodic links rather than a flat category label.", "S=(V,E,C,T,G,Theta)", "Distributed temporal semantic systems, medial prefrontal networks, and hippocampal retrieval contribute.", "structured_knowledge_supported; graph is computational",
     ("candidate_schemas",), ("candidate_schema_graphs",)),
    ("P10-04", "Align evidence to roles and slots", "Compare values and relations to candidate slot structure while preserving alternative alignments.", "s_i=exp(-d_i^2/(2*sigma_i^2)); M_rel=sum q_uv*compatibility", "Perception interacts with stored knowledge; hippocampal relational binding is relevant for multi-element cases.", "relational_alignment_supported_at_functional_level",
     ("evidence_record", "schema_graphs"), ("slot_alignment", "relation_alignment")),
    ("P10-05", "Weight evidence by diagnosticity and reliability", "Weight trustworthy, discriminating, task-relevant observations and reduce redundant evidence.", "D(f)=log(P(f|S)/P(f|not S)); E=sum reliability*attention*D", "Attention and control networks prioritize goal-relevant diagnostic dimensions.", "attention_and_diagnosticity_supported; exact weights are model choices",
     ("slot_alignment", "goal", "reliability"), ("weighted_evidence", "diagnosticity")),
    ("P10-06", "Estimate candidate evidence fit", "Combine context-sensitive priors with likelihoods and penalties for explicitly contradicted observations; unknowns are neutral.", "P(S|X,C) proportional to P(X|S,C)*P(S|C)", "Bottom-up sensory input and top-down expectations interact across perceptual and memory networks.", "Bayesian_formalization_is_normative_not_literal_neural_arithmetic",
     ("weighted_evidence", "candidate_priors"), ("candidate_posteriors", "uncertainty")),
    ("P10-07", "Predict unobserved feature slots", "Estimate task-relevant feature values conditionally and keep them separate from current observations.", "P(z=v|X_obs,S,C,G)=softmax(theta_v^T*phi)", "Schemas guide expectation, attention, and interpretation through distributed systems.", "schema_based_completion_supported; conditional model is computational",
     ("candidate_posteriors", "schema_slots"), ("inferred_features", "confidence")),
    ("P10-08", "Complete missing relations", "Predict missing links among entities/roles using schema graph constraints and retain inferred provenance.", "G_hat=argmax_G[log P(G|S,C)+lambda*log P(X|G)]", "Hippocampal relational binding works with prefrontal and temporal knowledge systems.", "relational_completion_is_supported_functionally; MAP graph search is computational",
     ("observed_relations", "schema_relations"), ("inferred_relations", "relation_uncertainty")),
    ("P10-09", "Select hierarchical category level", "Choose the most specific category meeting posterior and margin thresholds; otherwise return a broader class or abstain.", "deepest S with P(S|X,C)>=theta and margin>=delta", "Categorization can operate at multiple abstraction levels; specificity depends on available evidence.", "hierarchical_categorization_supported; threshold policy is computational",
     ("candidate_posteriors", "schema_hierarchy"), ("selected_category", "abstention_reason")),
    ("P10-10", "Predict temporal/event progression", "Use event-state, context, and goals to predict next states; keep temporal sequence separate from causation.", "P(s[t+1]|s[t],S,C,G); p[t+k]=p[t]*T^k in a simple Markov model", "Distributed event memory, hippocampal sequence processing, and prefrontal expectations contribute.", "event_prediction_supported; Markov model is simplification",
     ("event_history", "schema_transitions", "context", "goal"), ("next_event_predictions")),
    ("P10-11", "Integrate predictions with new sensory evidence", "Confirm, contradict, or leave predictions unresolved; recompute posterior after important new evidence.", "P(H|e)=P(e|H)P(H)/P(e); surprise=-log P(e|S,C)", "Predictive expectations and sensory evidence interact; mismatch can recruit attention and episodic encoding.", "perception_memory_interaction_supported; scalar error is computational",
     ("inferred_predictions", "new_evidence"), ("revised_interpretation", "prediction_errors")),
    ("P10-12", "Resolve conflict and seek diagnostic information", "Identify discriminating cues, then select safe, affordable information-gathering actions by expected information gain.", "IG(q)=H(S|D)-E_a[H(S|D,a)]; q*=argmax(IG-kappa*cost)", "Prefrontal control and attention help maintain goals and resolve uncertainty; hippocampus can provide episode comparisons.", "active_information_seeking_supported; entropy policy is computational",
     ("candidate_conflict", "query_options", "goal"), ("query_recommendation", "expected_information_gain")),
    ("P10-13", "Produce a source-aware interpretation", "Label each element as current observation, specific episode, schema prediction, or external information.", "M={(item,source,status,confidence,context)}", "Reconstructive memory and source monitoring motivate explicit provenance.", "source_memory_risk_supported; ledger is software safeguard",
     ("observations", "retrieved_episodes", "predictions"), ("source_aware_interpretation", "evidence_ledger")),
    ("P10-14", "Select a goal- and risk-appropriate response", "Estimate expected utility and error costs; prefer conservative responses when confidence is inadequate.", "U(a)=sum_s P(s|D)*R(a,s)-Cost(a)-risk_penalty", "Prefrontal valuation and action systems interact with current goals, uncertainty, and safety.", "utility_based_selection_is_normative_computational_model",
     ("interpretation", "goal", "action_options"), ("selected_action", "expected_utility")),
    ("P10-15", "Compare prediction with outcome", "Score only tested predictions and distinguish contradiction, noisy evidence, and outcomes never measured.", "Brier=(p-y)^2; surprise=-log P(y|S,C); TD error=r+gamma*V(s')-V(s)", "Prediction/outcome mismatch can guide attention, episodic encoding, and learning across networks.", "prediction_error_learning_supported; TD form is analogy",
     ("predictions", "tested_outcomes"), ("prediction_errors", "calibration_metrics")),
    ("P10-16", "Update case and cautiously update schema", "Store the specific episode; update schema probabilities only with repeated, reliable, explicitly authorized feedback.", "Beta posterior: alpha'=alpha+n1; beta'=beta+n0; mean=alpha'/(alpha'+beta')", "Schema-consistent integration and distinct episodic encoding vary with novelty, relevance, and context.", "incremental_update_supported; authorization and thresholds are software policy",
     ("episode", "validated_feedback", "authorization"), ("case_memory", "optional_schema_update")),
    ("P10-17", "Consolidate reusable knowledge with episode detail intact", "Link a case to its schema and neighboring categories while preserving source episodes, exceptions, and versions.", "hierarchical linking + confidence-weighted parameter consolidation", "Hippocampal episodic detail and distributed cortical generalized knowledge are complementary.", "episodic_semantic_complementarity_supported; consolidation policy is computational",
     ("case_memory", "schema_version", "evidence_history"), ("reusable_linked_knowledge", "provenance_history")),
)


OPERATION_REGISTRY = tuple(
    OperationSpec(
        operation_id=row[0], name=row[1], purpose=row[2], mathematical_form=row[3],
        biological_reference=row[4], evidence_level=row[5], inputs=row[6], outputs=row[7],
    )
    for row in _ROWS
)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return the canonical 17-stage Point 10 registry in source order."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
