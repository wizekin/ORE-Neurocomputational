"""Canonical 16-operation registry for Point 04 episodic-to-semantic transformation.

Operation order follows Point_04_Episodic_to_Semantic_Transformation_and_Gist_Extraction.md.
Equations are implementable models or diagnostics, not literal neuronal arithmetic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one source-ordered transformation operation and its biological interpretation."""

    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return this operation definition as a plain mapping for trace and documentation."""
        return asdict(self)


_ROWS = (
    ("ES01", "Reactivate an encoded event or related events",
     "Use a cue or internal trigger to select candidate events and reconstruct partial patterns while retaining uncertainty.",
     "s_i=sim(q,k_i); P(E_i|q)=softmax(beta*s_i).",
     "Hippocampal pattern completion interacts with distributed cortical reinstatement during cued and spontaneous retrieval.",
     "associative_reactivation_supported; similarity and softmax are model operations",
     ("cue", "event_indices", "cortical_patterns"), ("reactivated_candidates", "cue_confidence")),
    ("ES02", "Reconstruct event feature components",
     "Recover content, spatial, temporal, sensory, affective, causal, and relational evidence with feature-level confidence.",
     "x_hat_if=a_if*c_if; preserve unknown separately from an observed zero.",
     "Perceptual, affective, association, and medial-temporal systems contribute distributed information during reconstruction.",
     "distributed_reinstatement_supported; explicit feature maps are a data representation",
     ("reactivated_event", "feature_evidence", "source_confidence"), ("feature_groups", "feature_uncertainty")),
    ("ES03", "Preserve event-specific relational bindings",
     "Bind elements within the same event so similar details from different episodes are not silently merged.",
     "G_i=(V_i,R_i); B_i=|R_supported|/(|R_candidates|+epsilon).",
     "Hippocampal relational binding and index-like functions support event-specific combinations across distributed content.",
     "relational_memory_supported; graph and binding score are computational abstractions",
     ("feature_groups", "relations", "event_identity"), ("event_graph", "binding_quality")),
    ("ES04", "Retrieve and align related experiences",
     "Align corresponding properties across distinct events while retaining unmatched event-specific features.",
     "S(E_i,E_j)=sum_f w_f sim(x_if,x_jf)+lambda sim(R_i,R_j).",
     "Hippocampal-cortical interactions and cortical association networks support retrieval and overlap across experiences.",
     "cross-event_overlap_supported; weighted similarity is a model choice",
     ("event_patterns", "relation_graphs", "stored_episode_set"), ("aligned_episode_set", "similarity_matrix")),
    ("ES05", "Estimate invariant and incidental features",
     "Estimate recurrence, diagnosticity, consistency, source reliability, and feature variability; do not equate frequency with meaning.",
     "P(f|C)=sum_i r_i*x_if/sum_i r_i; IG(f)=H(C)-H(C|f).",
     "Repeated experience and statistical regularities influence cortical knowledge; explicit entropy is a normative analogy.",
     "statistical_regularities_supported; information gain is computational",
     ("aligned_episodes", "feature_observations", "contrastive_examples"), ("invariant_features", "incidental_features", "feature_uncertainty")),
    ("ES06", "Extract recurring relations across episodes",
     "Detect repeated causal, temporal, functional, and role relations while requiring support across independent examples.",
     "P(r|C)=sum_i r_i*I(r in E_i)/sum_i r_i; N_eff=(sum r_i)^2/sum(r_i^2).",
     "Relational and schema knowledge is represented across hippocampal and distributed cortical systems; typed relation counts are an abstraction.",
     "relational_generalization_supported; count and effective-sample metrics are model tools",
     ("event_graphs", "relation_confidence", "episode_diversity"), ("recurring_relations", "relation_support")),
    ("ES07", "Generate a semantic or gist-level representation",
     "Construct a compact abstraction from supported invariant features and recurring relations, not from a blurry copy of one episode.",
     "g_C=sum_i r_i*phi(E_i)/(sum_i r_i+epsilon).",
     "Semantic knowledge is distributed across modality-specific and association cortices and can become more abstract through experience.",
     "gist_and_semantic_abstraction_supported; weighted prototype is computational",
     ("invariant_features", "recurring_relations", "reliability"), ("semantic_gist", "supporting_evidence")),
    ("ES08", "Reduce exact context when it is incidental",
     "Selectively reduce spatial and temporal weights only when context has low diagnostic, retrieval, and meaning value.",
     "w_c=sigmoid(theta_0+theta_1*diagnosticity+theta_2*utility+theta_3*meaning).",
     "Context can become less prominent in semanticized knowledge, but spatial-temporal context remains essential for many episodic and causal tasks.",
     "selective_context_abstraction_supported; logistic weighting is a model rule",
     ("semantic_gist", "context_importance", "diagnosticity"), ("context_weights", "abstracted_gist")),
    ("ES09", "Maintain differentiated episodic and semantic forms",
     "Keep high-resolution event records and generalized knowledge as linked, coexisting representations.",
     "M(C)={H_1,...,H_n,G_C,Links(H_i,G_C)}.",
     "Trace-transformation, multiple-trace, and systems-consolidation accounts differ, but detailed episodic and generalized knowledge should not be collapsed conceptually.",
     "dual_representation_is_theory_informed; dual-store software is a model architecture",
     ("episodic_patterns", "semantic_gist", "event_links"), ("episodic_store", "semantic_store")),
    ("ES10", "Integrate gist into distributed cortical knowledge",
     "Map features and relations into a distributed semantic population and associate them with related cortical patterns.",
     "z=f(W_phi*x-I); z_C'=(1-eta)z_C+eta*z.",
     "Anterior temporal semantic integration interacts with modality-specific spokes and association cortex; one embedding is only a surrogate.",
     "distributed_semantic_networks_supported; population projection is computational",
     ("semantic_gist", "prior_concepts", "feature_relations"), ("distributed_semantic_pattern", "cortical_associations")),
    ("ES11", "Update knowledge while preserving useful distinctions",
     "Integrate supported overlap, preserve exceptions, keep ambiguous features uncertain, and prevent automatic merging of similar concepts.",
     "z_new=(1-rho)z_old+rho*g; conflicting support remains a separate exception or unresolved feature.",
     "Cortical integration coexists with pattern separation and hippocampal event specificity; the balance depends on task and circuit.",
     "integration_and_separation_both_supported; update coefficients are model-specific",
     ("semantic_concept", "new_evidence", "counterevidence", "concept_boundaries"), ("updated_knowledge", "exceptions", "conflicts")),
    ("ES12", "Reweight accessibility of incidental event details over time",
     "Model changing access to a trace separately from trace existence; include decay, interference, and reactivation.",
     "A_f(t)=A_f(0)*exp(-lambda_f*t-kappa*I_f(t)); stored event content is retained.",
     "Forgetting can reflect accessibility, competition, and interference; failure to retrieve is not proof of physical erasure.",
     "accessibility_and_interference_supported; exponential law is a simplified model",
     ("feature_accessibility", "elapsed_time", "interference", "reactivation"), ("accessibility_profile", "retained_trace")),
    ("ES13", "Preserve hippocampal support for specific-episode queries",
     "Classify the retrieval goal and use event-specific indices for exact episodes, semantic concepts for general facts, or both for mixed questions.",
     "R(q)=R_H(H_i) for episodic; R_C(G_C) for semantic; R_combined for mixed.",
     "Detailed event reinstatement often recruits hippocampal networks; generalized semantic retrieval can increasingly rely on distributed cortical knowledge.",
     "query_dependent_memory_support_supported; routing is software logic",
     ("retrieval_goal", "episodic_store", "semantic_store"), ("retrieval_policy", "selected_memory_track")),
    ("ES14", "Reconstruct details with source and uncertainty labels",
     "Distinguish directly reinstated, externally sourced, inferred, generalized, and unresolved details.",
     "C_f=P(f|q,trace,source); preserve provenance with every reconstructed element.",
     "Remembering is reconstructive and can incorporate schemas; source monitoring is necessary to limit false episodic detail.",
     "source_monitoring_and_reconstructive_memory_supported; probability field is computational",
     ("retrieved_features", "event_sources", "semantic_schema", "confidence"), ("provenance_labeled_result", "uncertainty")),
    ("ES15", "Use reactivation and new experience to refine both forms",
     "Update semantic knowledge from reliable new evidence and add new episodic records without silently rewriting earlier events.",
     "g_(t+1)=g_t+eta*c*(e_new-g_t); append immutable episode evidence.",
     "Reactivation, reconsolidation, new learning, and schema refinement can alter memory representations under task-dependent conditions.",
     "memory_update_mechanisms_supported; adaptive-filter equation is illustrative",
     ("new_episode", "retrieval_evidence", "existing_episodic_and_semantic_state"), ("refined_knowledge", "updated_links")),
    ("ES16", "Evaluate transformation quality",
     "Assess generalization, episode discrimination, provenance integrity, and confidence calibration separately.",
     "BS=(1/N)*sum_i(p_i-y_i)^2; measure held-out transfer and lure discrimination.",
     "Behavioral generalization, episodic specificity, source accuracy, and confidence are separable measurable outcomes.",
     "evaluation_metrics_are_normative; metrics do not identify a unique neural mechanism",
     ("episodic_store", "semantic_store", "heldout_examples", "source_labels"), ("quality_report", "refinement_signal")),
)


OPERATION_REGISTRY = tuple(OperationSpec(*row) for row in _ROWS)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return the 16 operation definitions in canonical source order."""
    return [row.to_dict() for row in OPERATION_REGISTRY]
