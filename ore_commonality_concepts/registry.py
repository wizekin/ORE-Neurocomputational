"""Canonical 16-operation registry for Point 06 commonality detection and concept formation.

Operation order follows Point_06_Commonality_Detection_and_Concept_Formation.md. Equations describe
candidate computational operations, not literal arithmetic performed by individual neurons.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one source-ordered operation, its purpose, mathematical form, and biological limits."""

    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the specification in a plain dictionary for annotations and structured logs."""
        return asdict(self)


_ROWS = (
    ("CC01", "Encode each encounter as a distinct experience",
     "Preserve episode identity, context, temporal information, observed properties, relations, and evidence reliability before abstraction.",
     "E_i={X_i,C_i,T_i,A_i,O_i}; retain episode_id separately from feature content.",
     "Sensory and medial-temporal systems, including hippocampal circuits, can bind event elements and contextual relations.",
     "episodic_binding_supported; explicit records are software abstractions",
     ("encounter", "context", "time", "action", "outcome"),
     ("episode_trace", "episode_identity", "provenance")),
    ("CC02", "Extract candidate features across modalities",
     "Represent observed visual, auditory, tactile, motor, and other task-relevant properties; keep unknown features separate from observed absence.",
     "x_ik in [0,1]; m_ik in {0,1}; m_ik=0 means unobserved.",
     "Modality-specific sensory and association pathways extract structured information and integrate complex properties.",
     "modality_specific_feature_processing_supported; vectors and masks are representation choices",
     ("episode_trace", "sensory_attributes", "modality_evidence"),
     ("canonical_feature_map", "observation_mask", "modality_links")),
    ("CC03", "Select and weight features according to attention and goals",
     "Prioritize task-relevant evidence using attention, goal relevance, prior salience, and motivational state.",
     "a_k=.5*A_k+.3*G_k+.2*P_k; alpha_k=softmax(a_k/tau).",
     "Prefrontal and parietal control systems can bias sensory processing and alter feature gain.",
     "attention_and_goal_modulation_supported; softmax is computational",
     ("candidate_features", "attention", "goals", "prior_knowledge"),
     ("feature_gain", "competitive_attention_profile")),
    ("CC04", "Reactivate and retrieve related prior episodes",
     "Use partial feature or relation cues to retrieve distinct memories relevant to the current example.",
     "s_i=sim(X_t,X_i)*reliability_i; R_t=top_k(s_i).",
     "Hippocampal pattern completion and prefrontal retrieval guidance can make separate experiences jointly available.",
     "associative_retrieval_supported; similarity ranking is a model",
     ("current_episode", "episodic_memory", "learning_goal"),
     ("related_episode_ids", "retrieval_scores")),
    ("CC05", "Align comparable features across viewpoint and context variation",
     "Map different surface observations to comparable feature dimensions while retaining meaningful distinctions.",
     "z_i=g(x_i,C_i); d_w(i,j)=sqrt(sum_k alpha_k(z_ik-z_jk)^2).",
     "Perceptual tolerance to viewpoint, scale, lighting, position, and recording variation arises in interacting pathways.",
     "invariance_and_alignment_supported; g and weighted distance are abstractions",
     ("feature_maps", "context", "attention_gains"),
     ("aligned_feature_maps", "nuisance_variation_estimates")),
    ("CC06", "Detect recurring properties across related episodes",
     "Estimate prevalence from observed category examples, weighted by reliability, and exclude unknown values from denominators.",
     "P_hat(f_k|C)=sum_i r_i*m_ik*x_ik / sum_i r_i*m_ik.",
     "Repeated co-activation and distributed plasticity can strengthen common regularities across experiences.",
     "statistical_regularities_supported; explicit frequency estimator is computational",
     ("aligned_target_examples", "observation_masks", "reliability"),
     ("feature_prevalence", "effective_sample_size")),
    ("CC07", "Detect co-occurrence and relations among features",
     "Identify recurring feature conjunctions and typed part, temporal, functional, spatial, or causal relations.",
     "PMI(a,b)=log(P(a,b)/(P(a)P(b))); G_C=(V_C,E_C).",
     "Hippocampal relational binding and cortical co-activation can preserve feature relationships and combinations.",
     "relational_and_coactivation_learning_supported; PMI and graph edges are model forms",
     ("feature_activations", "episode_relations", "episode_context"),
     ("cooccurrence_associations", "typed_relation_graph")),
    ("CC08", "Separate category regularities from background coincidences",
     "Compare target and alternative-category prevalence so frequent but non-diagnostic background details are downweighted.",
     "D_k=log((P(f_k|C)+eps)/(P(f_k|not C)+eps)); IG=H(C)-H(C|f_k).",
     "Contrastive learning, prediction, and feedback can reveal whether a feature distinguishes a category from its surroundings.",
     "contrastive_category_learning_supported; log odds and information gain are computational",
     ("target_prevalence", "comparison_prevalence", "category_uncertainty"),
     ("diagnosticity", "information_gain_estimate")),
    ("CC09", "Estimate regularity stability, variability, and reliability",
     "Measure within-category variability, evidence coverage, effective sample size, and reliability without erasing legitimate exceptions.",
     "sigma_k^2=weighted_var(x_k); r_k=coverage*ESS/(ESS+1)/(1+var/v_floor).",
     "Distributed learning tolerates natural variation while preserving task-relevant distinctions; no single statistic is implied.",
     "variability_and_reliability_relevant; variance aggregation is normative",
     ("feature_observations", "diagnosticity", "reliability_weights"),
     ("feature_variance", "feature_reliability", "uncertainty")),
    ("CC10", "Integrate episodes into a provisional shared structure",
     "Combine reliable, recurring and diagnostic properties and relations into a revisable representation linked to supporting episodes.",
     "mu_t=mu_(t-1)+eta_t*r_t*(z_t-mu_(t-1)); preserve separate episode links.",
     "Hippocampal memory integration, medial-prefrontal interaction, and gradual cortical learning can contribute to regularity extraction.",
     "memory_integration_supported; incremental parameter update is computational",
     ("feature_prevalence", "diagnosticity", "relations", "supporting_episodes"),
     ("provisional_concept", "integrated_evidence")),
    ("CC11", "Compress incidental details while preserving useful distinctions",
     "Select shared feature/relation regularities while retaining episode-level support, variability, exceptions, and discriminative details.",
     "J(theta)=L_prediction(theta)+lambda*C(theta).",
     "Repeated learning can emphasize shared regularities while detailed episodic traces retain some unique context.",
     "selective_abstraction_supported; regularized objective is an analogy",
     ("provisional_concept", "feature_weights", "uncertainty", "exceptions"),
     ("compressed_shared_structure", "preserved_distinctions")),
    ("CC12", "Stabilize common relationships through gradual cortical plasticity",
     "Strengthen recurrent feature-population associations through repeated reliable co-activation and bounded weight normalization.",
     "Delta W_ij=eta*x_i*y_j; model gated plasticity as eta*eligibility*modulatory_state.",
     "Sensory and association cortical populations change gradually with repeated, task-dependent co-activation.",
     "distributed_cortical_plasticity_supported; Hebbian rate rule is a surrogate",
     ("shared_feature_activations", "recurrent_population_state", "reliability"),
     ("cortical_population_pattern", "updated_associative_weights")),
    ("CC13", "Use the provisional concept to predict a new instance",
     "Estimate which properties should occur in an unfamiliar example and assess category membership only when contrastive evidence exists.",
     "P(C|z) proportional to P(z|C)*P(C); absent comparison class yields fit-only output.",
     "Concept-guided expectation can bias perception and attention; predictions remain probabilistic.",
     "predictive_categorization_supported; Bayesian expression is computational",
     ("provisional_concept", "new_episode", "comparison_evidence"),
     ("predicted_features", "membership_probability", "fit_score")),
    ("CC14", "Compute mismatch and revise the common structure",
     "Measure uncertainty-normalized prediction error, distinguish noisy exceptions from repeated counterevidence, and update with labelled feedback when supplied.",
     "delta=z-z_hat; E=delta^T*Sigma^(-1)*delta; eta_t=eta_min+(eta_max-eta_min)*g(E).",
     "Prediction error, feedback, and renewed retrieval can revise category regularities; a single surprise need not split a category.",
     "error-driven_revision_supported; residual and adaptive-rate formulas are model choices",
     ("predictions", "observed_new_features", "known_feedback", "uncertainty"),
     ("mismatch", "revised_concept", "revision_provenance")),
    ("CC15", "Consolidate and index the concept for future retrieval",
     "Connect multiple feature and relation cues to the current concept and supporting episodes without erasing their individual identities.",
     "Index(q)->{concept_id,feature_weights,relation_links,supporting_episode_ids}.",
     "Cortical association pathways can make learned regularities accessible from multiple sensory and contextual cues.",
     "associative_access_supported; explicit lookup table is implementation logic",
     ("shared_features", "relation_edges", "supporting_episodes"),
     ("feature_index", "relation_index", "concept_lookup")),
    ("CC16", "Produce a revisable abstract concept representation",
     "Return shared features, typed relations, weights, uncertainty, exceptions, and evidence provenance in a form that can be updated later.",
     "C={F_shared,R_shared,W,U,M_support}; retain episode and concept identity.",
     "Concept learning is distributed across interacting systems and remains open to new evidence and refinement.",
     "distributed_concept_representation_supported; structured output is a software contract",
     ("integrated_concept", "uncertainty", "supporting_evidence", "revision_history"),
     ("revisable_concept", "provenance", "associative_access")),
)


OPERATION_REGISTRY = tuple(OperationSpec(*row) for row in _ROWS)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return all 16 Point 06 operations in canonical source order."""
    return [operation.to_dict() for operation in OPERATION_REGISTRY]
