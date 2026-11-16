"""Canonical registry for all 21 Hub-and-Spoke stages and their sub-operations.

Every biological description is a functional hypothesis, not proof that the
brain executes the corresponding software operation or equation literally.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one independently traceable algorithmic operation."""

    operation_id: str
    name: str
    mathematical_operation: str
    functional_responsibility: str


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Describe one ordered functional stage and its biological reference."""

    stage_id: str
    name: str
    objective: str
    biological_reference: str
    operations: tuple[OperationSpec, ...]


def _stage(
    number: int,
    name: str,
    objective: str,
    biology: str,
    operations: tuple[tuple[str, str, str], ...],
) -> StageSpec:
    """Build a stage record and assign stable hierarchical operation identifiers."""
    return StageSpec(
        stage_id=f"HS-{number:02d}",
        name=name,
        objective=objective,
        biological_reference=biology,
        operations=tuple(
            OperationSpec(
                operation_id=f"HS-{number:02d}.{index}",
                name=operation_name,
                mathematical_operation=math,
                functional_responsibility=responsibility,
            )
            for index, (operation_name, math, responsibility) in enumerate(operations, 1)
        ),
    )


STAGE_REGISTRY: tuple[StageSpec, ...] = (
    _stage(1, "Receive conceptual information",
        "Receive written/spoken labels, pictures, objects, actions, and other sensory or recalled evidence.",
        "Distributed sensory and language pathways deliver parallel signals with different arrival times.",
        (
            ("encode_input_channels", "typed feature maps / signal encoding", "Encode each supplied modality without collapsing its source identity."),
            ("detect_available_modalities", "binary availability mask A_m", "Mark which spokes contain current evidence; absent channels are not negative evidence."),
            ("retain_simultaneous_signals", "parallel buffering and timestamp alignment", "Keep concurrent modality signals available for later integration."),
        )),
    _stage(2, "Activate modality-specific spokes",
        "Route each signal to its corresponding modality-linked semantic feature network.",
        "Visual, auditory, somatosensory, motor, chemosensory, linguistic, and affective association networks contribute distinct properties.",
        (
            ("route_to_spoke", "modality-keyed projection", "Route each input to the appropriate semantic spoke."),
            ("activate_feature_populations", "s_m = sigmoid(W_input,m x_m + b_m)", "Compute graded feature-population activity."),
            ("preserve_distributed_codes", "population vector s_m", "Retain distributed activity rather than assigning one neuron to one concept."),
        )),
    _stage(3, "Extract individual semantic features",
        "Identify active feature dimensions and preserve learned relations among them.",
        "Distributed association populations encode feature patterns and co-activation rather than isolated symbolic neurons.",
        (
            ("threshold_active_features", "f_i = 1[a_i > theta_i]", "Identify sufficiently active feature dimensions while retaining graded values."),
            ("retain_feature_strengths", "feature vector F_m = [f_1,...,f_k]", "Preserve activation magnitudes for weighted downstream processing."),
            ("construct_feature_associations", "R_m = normalized X_m^T X_m", "Represent learned co-occurrence and relationships between features."),
            ("prepare_hub_features", "typed vector projection input", "Prepare feature patterns for transmodal integration."),
        )),
    _stage(4, "Retrieve previously acquired semantic information",
        "Reactivate stored properties and combine them with current observations while retaining uncertainty.",
        "Cue-driven associative retrieval and pattern completion can reinstate feature information from earlier experiences.",
        (
            ("retrieve_associated_features", "a_new = a_old + R^T a", "Spread activation over learned feature associations."),
            ("reactivate_missing_modalities", "cue -> stored modality feature maps", "Reactivate supported feature patterns not present in current sensory input."),
            ("blend_current_and_memory", "F = alpha F_current + (1-alpha) F_memory", "Combine observation and memory in proportion to current evidence quality."),
            ("retain_retrieval_confidence", "C_i in [0,1]", "Keep confidence and source identity for recalled properties."),
        )),
    _stage(5, "Transmit spoke information toward the hub",
        "Project each spoke activation into the shared hub space.",
        "Distributed spoke-to-ATL pathways carry modality-linked information toward anterior temporal transmodal regions.",
        (
            ("project_spoke_to_hub", "u_m = W_m s_m", "Transform each spoke vector into the hub dimensionality."),
            ("apply_learned_connectivity", "weighted matrix projection", "Scale contributions according to learned spoke-to-hub connections."),
            ("retain_parallel_projections", "U = {u_1,...,u_M}", "Preserve every modality projection independently."),
            ("prepare_bidirectional_routes", "W_m and P_m connection pair", "Retain the reverse route required for hub feedback."),
        )),
    _stage(6, "Estimate and combine modality contributions",
        "Weight channels by availability, relevance, and reliability without deleting disagreement.",
        "Modality precision and network coupling can vary with context, task, and signal quality.",
        (
            ("estimate_modality_reliability", "q_m = f(coverage, confidence, noise)", "Estimate the current evidential reliability of each spoke."),
            ("normalize_modality_weights", "g_m = softmax(q_m / temperature)", "Normalize modality weights to sum to one."),
            ("weight_spoke_signals", "u_tilde_m = g_m u_m", "Scale projections by reliability."),
            ("measure_cross_modal_disagreement", "D_ij = distance(u_i,u_j)", "Retain conflicts as evidence for further processing rather than averaging them away."),
        )),
    _stage(7, "Integrate features in the amodal semantic hub",
        "Transform incomplete modality-specific signals into a nonlinear shared concept state.",
        "Bilateral anterior temporal regions are modeled as a transmodal convergence contribution within a distributed network.",
        (
            ("pool_weighted_spoke_input", "U = sum_m g_m W_m s_m", "Pool the reliability-weighted spoke projections."),
            ("nonlinear_hub_transform", "h = phi(W_h U + b_h)", "Transform pooled evidence into a common hub representation."),
            ("bind_feature_conjunctions", "h = phi_2(W_2 phi_1(W_1 U+b_1)+b_2)", "Represent learned combinations that cannot be captured by independent feature addition."),
            ("abstract_shared_structure", "learned nonlinear feature interaction", "Extract shared conceptual structure across non-identical exemplars."),
            ("retain_distributed_hub_state", "h = [h_1,...,h_n]", "Represent concepts as distributed activation patterns."),
        )),
    _stage(8, "Stabilize the integrated concept",
        "Refine the hub state recurrently and preserve alternatives until evidence is sufficient.",
        "Recurrent interactions among temporal and distributed cortical populations can support stable conceptual states.",
        (
            ("initialize_hub_state", "h_t", "Retain the current hub state before refinement."),
            ("recurrent_hub_update", "h_(t+1)=phi(W_hh h_t + input_t + b)", "Combine the previous hub state with new spoke input."),
            ("iterate_refinement", "bounded fixed-point iteration", "Repeat recurrent updates until convergence or an explicit iteration limit."),
            ("measure_convergence", "Delta_t = norm(h_(t+1)-h_t)", "Measure stabilization using a vector norm."),
            ("retain_competing_hypotheses", "candidate activation vector", "Preserve multiple interpretations when the hub state is ambiguous."),
        )),
    _stage(9, "Send integrated information back to spokes",
        "Use hub feedback to reactivate and refine modality-specific feature states.",
        "Top-down ATL-to-spoke interactions can reinstate sensory, action, and language-related semantic features.",
        (
            ("project_hub_to_spokes", "s_hat_m = P_m h", "Predict a spoke pattern from the integrated hub state."),
            ("reactivate_relevant_features", "bounded nonlinear projection", "Reactivate properties associated with the current concept."),
            ("compute_prediction_error", "e_m = s_m - s_hat_m", "Compare observed and predicted feature states."),
            ("reconcile_observed_and_predicted", "s_m_new = (1-beta)s_m + beta s_hat_m", "Refine the working spoke state without overwriting raw observations."),
            ("continue_bidirectional_updates", "spokes <-> hub message passing", "Allow another integration pass when feedback supplies useful evidence."),
        )),
    _stage(10, "Develop a modality-independent representation",
        "Align the shared concept identity across routes while preserving modality-specific detail.",
        "The hub supports concept access across modalities while remaining connected to modality-specific systems.",
        (
            ("compute_route_specific_hubs", "h_m = H_m(s_m)", "Estimate the hub code induced by each available input route."),
            ("identify_shared_structure", "shared_h = weighted mean(h_m)", "Estimate common structure across input routes."),
            ("preserve_modality_residuals", "specific_m = h_m - shared_h", "Keep details that are unique to a sensory or language channel."),
            ("measure_cross_modal_similarity", "cos(h_i,h_j)", "Quantify route alignment without requiring identical neural patterns."),
        )),
    _stage(11, "Identify the concept and retrieve its identity",
        "Compare the current hub representation with learned concept patterns and rank alternatives.",
        "A transmodal concept representation supports recognition independent of the original input channel.",
        (
            ("access_candidate_concepts", "concept prototype matrix", "Retrieve candidate concept representations from the pinned inventory."),
            ("calculate_concept_similarity", "S(c) = cosine(h, P_c)", "Measure compatibility between the current hub state and each concept."),
            ("rank_concepts", "c_star = argmax_c S(c)", "Rank candidate interpretations by their integrated evidence."),
            ("estimate_identification_confidence", "margin = S_1 - S_2", "Measure separation between the leading and competing concepts."),
            ("request_more_evidence_when_uncertain", "confidence and margin thresholds", "Abstain or retain alternatives when evidence does not support a stable identity."),
        )),
    _stage(12, "Construct category relationships and generalize",
        "Identify shared and distinguishing features and build taxonomic and associative links.",
        "Distributed semantic knowledge supports both feature-based category structure and learned relations between co-experienced concepts.",
        (
            ("compare_concept_representations", "pairwise cosine similarity", "Compare stored concept patterns in a common feature space."),
            ("identify_shared_features", "F_i intersect F_j", "Find feature overlap across related concepts."),
            ("identify_distinctive_features", "F_i minus F_j", "Retain features that distinguish one concept from another."),
            ("estimate_category_prototypes", "P_category = weighted mean(P_c)", "Build category-level feature centroids from exemplars."),
            ("construct_taxonomic_links", "directed category graph", "Represent supplied prototype/category membership hierarchically."),
            ("construct_associative_links", "co-occurrence / conditional probability", "Represent learned associations separately from taxonomic similarity."),
        )),
    _stage(13, "Apply semantic control",
        "Select task-relevant features and suppress competing information without altering stored knowledge.",
        "A distributed control network including inferior frontal and posterior temporal/parietal regions biases retrieval by current goals.",
        (
            ("receive_current_task", "task/goal vector Q", "Represent the current cognitive objective."),
            ("score_feature_relevance", "R_i = relevance(f_i,Q)", "Estimate feature relevance from explicit task constraints."),
            ("boost_relevant_features", "a_i_prime = a_i + lambda R_i", "Increase activation of task-relevant properties."),
            ("suppress_irrelevant_features", "a_i_double_prime = a_i_prime - mu I_i", "Reduce irrelevant or competing properties."),
            ("select_controlled_features", "F_selected = {f_i:a_i>theta}", "Read out the features needed for the current task."),
            ("bias_relevant_spokes", "dynamic modality gating", "Prioritize modality routes useful for the task."),
        )),
    _stage(14, "Generate the required conceptual output",
        "Route the selected concept and feature evidence to a recognition, description, comparison, or action response.",
        "Semantic representations interact with language production, decision, and action systems outside the ATL hub.",
        (
            ("read_selected_hub_state", "h_output = h_selected", "Read the task-relevant semantic representation."),
            ("retrieve_output_features", "controlled feature selection", "Select properties required by the requested task."),
            ("route_to_output_system", "task-conditioned response map", "Map semantic evidence to the requested output family."),
            ("generate_behavioral_response", "classification / response readout", "Produce an inspectable conceptual result."),
        )),
    _stage(15, "Acquire information across experiences",
        "Collect sourced multi-episode examples and estimate stable versus incidental feature evidence.",
        "Experience-dependent plasticity accumulates distributed features across encounters; the software store is an explicit model memory.",
        (
            ("receive_learning_episode", "E_t = {x_m, concept_id, provenance}", "Receive one independently sourced encounter."),
            ("activate_experience_spokes", "s_m(t) from episode features", "Encode observed modality channels for each episode."),
            ("activate_shared_experience_hub", "h_t = phi(sum_m W_m s_m + b)", "Generate an episode-level integrated state."),
            ("accumulate_concept_evidence", "incremental sufficient statistics", "Accumulate evidence by concept and modality."),
            ("estimate_feature_stability", "P(f_i|c) = n(f_i,c)/n(c)", "Estimate how consistently a feature occurs for a concept."),
            ("separate_stable_and_incidental_features", "variance / frequency weighting", "Keep recurring properties distinct from episode-specific details."),
        )),
    _stage(16, "Learn spoke-to-hub mappings",
        "Update modality mappings from reconstruction error and co-activation while bounding drift.",
        "Synaptic plasticity can strengthen connections between co-active populations; backpropagation is only a computational optimization analogue.",
        (
            ("initialize_mapping_connections", "W_m in R^(H x F_m)", "Maintain a separate projection matrix for each spoke."),
            ("compute_hub_targets", "h_target from paired episodes", "Generate target integrated states for observed paired experience."),
            ("predict_cross_modal_features", "y_hat_m = W_m x_m", "Predict hub-level information from each spoke."),
            ("calculate_prediction_error", "e_m = y_m - y_hat_m", "Measure reconstruction mismatch."),
            ("calculate_squared_loss", "L_m = 0.5 * norm(e_m)^2", "Quantify mapping prediction error."),
            ("apply_gradient_and_hebbian_updates", "Delta W = eta * outer(e,x) + eta_h * outer(y,x)", "Update weights with an explicit gradient analogue and a co-activity term."),
            ("repeat_bounded_learning", "W_(t+1) = clip(W_t + Delta W)", "Apply bounded learning across sourced episodes."),
        )),
    _stage(17, "Learn modality-invariant representations",
        "Align routes that refer to the same concept while preserving unique channel information.",
        "Repeated cross-modal experience can shape shared conceptual structure without erasing sensory-specific representations.",
        (
            ("pair_same_concept_routes", "positive pair labels y_ij", "Pair modality routes linked to the same verified concept."),
            ("compute_route_hub_states", "h_m = W_m x_m", "Generate route-specific hub projections."),
            ("minimize_positive_pair_distance", "L_cross = sum y_ij * norm(h_i-h_j)^2", "Encourage same-concept routes to become compatible."),
            ("preserve_modality_details", "residual_m = h_m - shared_h", "Retain channel-specific residual information."),
            ("test_unseen_exemplars", "held-out similarity / retrieval", "Evaluate generalization on examples excluded from the update."),
        )),
    _stage(18, "Continuously update semantic knowledge",
        "Integrate credible new evidence without overwriting useful prior knowledge.",
        "Memory systems balance plasticity with stability; the exact software rule is a testable engineering approximation.",
        (
            ("detect_new_evidence", "new observation versus stored estimate", "Detect a genuinely new feature observation."),
            ("calculate_knowledge_prediction_error", "e_knowledge = x_new - x_hat", "Measure mismatch with the stored estimate."),
            ("evaluate_source_credibility", "confidence-weighted evidence", "Weight updates by supplied confidence and provenance."),
            ("update_appropriate_spoke", "bounded modality-specific parameter update", "Update only the channel supported by the observation."),
            ("update_shared_hub_knowledge", "bounded common-representation update", "Update shared concept knowledge when paired evidence warrants it."),
            ("protect_prior_knowledge", "weight decay / anchor regularization", "Limit drift away from the prior model to reduce forgetting."),
        )),
    _stage(19, "Simulate degradation of the semantic hub",
        "Apply a controlled hub lesion and measure cross-modal loss of conceptual specificity.",
        "Anterior temporal degeneration is associated with multimodal semantic difficulties; a mask is a computational lesion, not a disease simulation.",
        (
            ("choose_hub_units_for_degradation", "diagonal degradation matrix D_h", "Select hub dimensions for counterfactual reduction."),
            ("reduce_hub_activity", "h_prime = D_h h", "Attenuate selected hub units."),
            ("measure_feature_binding_loss", "hub state / feature conjunction residual", "Measure the loss of integrated feature structure."),
            ("measure_semantic_specificity", "correct-vs-competitor score margin", "Estimate reduced discrimination among concepts."),
            ("test_across_input_routes", "cross-modal candidate agreement", "Check whether impaired hub output affects distinct input modalities."),
            ("retain_graded_residual_knowledge", "partial rather than all-or-none mask", "Preserve remaining activity and broader evidence where possible."),
        )),
    _stage(20, "Simulate damage to individual semantic spokes",
        "Ablate one modality-linked pathway at a time and compare its selective effect with hub damage.",
        "A modality-specific lesion should disproportionately affect its associated feature channel while other spokes can still contribute.",
        (
            ("select_spoke_for_ablation", "one-hot modality mask", "Select one spoke without damaging the others."),
            ("attenuate_spoke_activity", "s_m_prime = D_m s_m", "Reduce activity in the selected modality."),
            ("recompute_hub_input", "U_prime = sum_m W_m s_m_prime", "Recalculate integration from remaining activity."),
            ("measure_modality_specific_loss", "per-spoke evidence and margin delta", "Quantify selective loss and possible compensation."),
            ("compare_with_hub_damage", "cross-condition performance profile", "Compare the selective pattern with the broad hub-lesion condition."),
        )),
    _stage(21, "Simulate damage to hub-spoke connections",
        "Reduce coupling while keeping hub and spokes intact, then measure the resulting integration deficit.",
        "White-matter pathways connect transmodal and modality-linked regions; connection masks are counterfactual simulations.",
        (
            ("identify_spoke_hub_weights", "W_m connectivity matrix", "Select the connection matrix for each modality."),
            ("attenuate_connection_weights", "W_m_prime = D_m W_m", "Reduce selected pathway strengths without destroying the endpoint populations."),
            ("recompute_transmitted_signal", "u_m_prime = W_m_prime s_m", "Recalculate the affected spoke projection."),
            ("recompute_hub_representation", "h_prime = phi(sum_m u_m_prime)", "Re-evaluate the integrated concept with reduced coupling."),
            ("test_pathway_compensation", "remaining-pathway contribution", "Measure how much other connections compensate."),
            ("compare_behavioral_profiles", "baseline / hub / spoke / connection conditions", "Produce a side-by-side computational lesion comparison."),
        )),
)

if len(STAGE_REGISTRY) != 21:
    raise RuntimeError("Hub-and-Spoke registry must contain exactly 21 stages")
if tuple(stage.stage_id for stage in STAGE_REGISTRY) != tuple(f"HS-{i:02d}" for i in range(1, 22)):
    raise RuntimeError("Hub-and-Spoke stage identifiers must be contiguous and ordered")
ALL_OPERATIONS = tuple(operation for stage in STAGE_REGISTRY for operation in stage.operations)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return a JSON-friendly copy of the complete 21-stage operation registry."""
    return [
        {
            **asdict(stage),
            "operations": [asdict(operation) for operation in stage.operations],
        }
        for stage in STAGE_REGISTRY
    ]
