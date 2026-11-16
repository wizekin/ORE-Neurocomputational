"""Ordered 19-stage registry for hybrid embodied–amodal concept processing.

The registry is a software trace contract, not a claim that biological cognition
runs serially or implements these equations literally.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one traceable functional substep, its equation, and biological analogue."""

    operation_id: str
    name: str
    mathematical_operation: str
    functional_responsibility: str
    biological_reference: str


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Describe one algorithm stage, its purpose, biological networks, and ordered substeps."""

    stage_id: str
    name: str
    objective: str
    biological_reference: str
    steps: tuple[OperationSpec, ...]


def _stage(
    number: int,
    name: str,
    objective: str,
    biology: str,
    steps: tuple[tuple[str, str, str], ...],
) -> StageSpec:
    """Construct a stage and assign stable hierarchical identifiers to its ordered steps."""
    stage_id = f"HEC-01.{number:02d}"
    operations = tuple(
        OperationSpec(
            operation_id=f"{stage_id}.{index:02d}",
            name=step_name,
            mathematical_operation=equation,
            functional_responsibility=responsibility,
            biological_reference=biology,
        )
        for index, (step_name, equation, responsibility) in enumerate(steps, 1)
    )
    return StageSpec(stage_id, name, objective, biology, operations)


STAGE_REGISTRY: tuple[StageSpec, ...] = (
    _stage(1, "Environmental and internal stimulus acquisition",
        "Acquire external sensory and internal bodily, emotional, mnemonic, imagery, and linguistic signals.",
        "Retina and thalamic visual pathways, cochlea and auditory pathways, somatosensory cortex, motor/premotor systems, olfactory and gustatory pathways, amygdala, insula, and interoceptive networks provide distinct input streams.",
        (
            ("transduce_modality_signals", "X_m(t) = F_m(I_m(t))", "Represent the result of modality-specific biological transduction or caller-provided internal signal encoding."),
            ("estimate_signal_statistics", "mu_m = mean(X_m); sigma_m = std(X_m)", "Estimate local signal statistics independently for each available modality."),
            ("normalize_modality_signals", "X'_m = (X_m - mu_m) / (sigma_m + epsilon)", "Normalize within a channel while protecting against zero variance."),
            ("retain_parallel_input_streams", "X' = {X'_m : m in available modalities}", "Preserve source identity rather than prematurely collapsing all inputs."),
        )),
    _stage(2, "Parallel distribution to modality-specific processing networks",
        "Route each acquired signal to a specialized but interconnected modality spoke.",
        "Cortical and subcortical populations interact through feedforward and feedback projections, association pathways, long-range white-matter connections, and recurrent cortical connections.",
        (
            ("dispatch_to_modality_spokes", "X -> {X_visual, X_auditory, X_motor, X_somatosensory, ...}", "Create a separate working stream for every modality actually present."),
            ("apply_spoke_connectivity", "Y_j = phi(sum_i W_ij X_i + b_j)", "Transform incoming activity with modality-specific connectivity and a nonlinear response."),
            ("maintain_recurrent_spoke_exchange", "X_m(t+1) = phi(W_mm X_m(t) + W_hm H(t))", "Permit higher-order information to modulate modality-specific working activity."),
        )),
    _stage(3, "Modality-specific feature extraction",
        "Extract features relevant to the representational capacity of each modality.",
        "Selective visual, auditory, motor, somatosensory, affective, and linguistic populations respond to distinct properties; association areas build more complex feature combinations.",
        (
            ("extract_modality_features", "S_m = phi(W_m^(1) X_m + b_m^(1))", "Transform each modality stream into a graded feature representation."),
            ("build_hierarchical_features", "S_m^(k+1) = phi(W_m^(k+1) S_m^k + b_m^(k+1))", "Represent successive feature levels, such as edges, contours, parts, and object structure."),
            ("measure_feature_similarity", "sim(S_i,S_j) = (S_i dot S_j) / (||S_i|| ||S_j|| + epsilon)", "Estimate similarity between feature patterns without forcing them to be identical."),
        )),
    _stage(4, "Attention-dependent feature selection",
        "Prioritize task-relevant features and reduce competing or irrelevant activity.",
        "Frontal and parietal control networks, sensory association cortices, thalamic pathways, and neuromodulatory systems can bias sensitivity and activity in selected populations.",
        (
            ("compute_attention_weights", "A_m = sigmoid(W_q Q + W_c C + b)", "Derive feature attention from the current task goal and context."),
            ("gate_relevant_features", "S_tilde_m = A_m elementwise S_m", "Amplify features with high task/context relevance."),
            ("inhibit_competing_features", "S_selected_m = A_m elementwise S_m - lambda I_m", "Subtract caller-identified distractor or competing activation."),
        )),
    _stage(5, "Construction of modality-specific semantic representations",
        "Organize extracted features into meaningful, modality-specific representations.",
        "Visual and auditory association cortices, premotor/motor association networks, somatosensory association cortex, and language-related temporal networks combine features with previously learned patterns.",
        (
            ("transform_selected_features", "Z_m = phi(W_m^(2) S_tilde_m + b_m^(2))", "Build a richer semantic representation within each modality."),
            ("compose_feature_sets", "Z_v = [z_shape; z_strings; z_neck; z_body]", "Keep named properties in a structured modality-specific vector."),
            ("model_nonlinear_feature_interactions", "Z_m = phi(W_1 S_m + W_2 (S_m tensor S_m))", "Represent conjunctions whose meaning depends on combinations of features."),
        )),
    _stage(6, "Activation of previously acquired conceptual knowledge",
        "Retrieve related stored knowledge to interpret current, partial, or ambiguous input.",
        "Distributed associative-memory networks and recurrent interactions between modality-specific areas and higher-order association regions, including ATL semantic representations, support retrieval and spreading activation.",
        (
            ("match_stored_patterns", "A_i = sim(Z_m, P_i)", "Compare current modality evidence with available stored concept patterns."),
            ("adjust_activation_by_context_and_familiarity", "A'_i = A_i + alpha C_i + beta F_i", "Bias retrieval with contextual fit and learned familiarity."),
            ("spread_associative_activation", "a^(t+1) = phi(W a^t + x)", "Propagate activation over stored feature associations while retaining alternatives."),
        )),
    _stage(7, "Temporal alignment and cross-modal binding",
        "Associate signals from different modalities when they refer to the same object or event.",
        "Sensory association networks, temporal association regions, attention systems, and convergence networks contribute; the source does not assume one dedicated synchrony mechanism performs all binding.",
        (
            ("compute_temporal_proximity", "B_ij = exp(-abs(t_i - t_j) / tau)", "Give greater temporal association strength to signals closer in time."),
            ("estimate_spatial_compatibility", "B_s = sim(P_i, P_j)", "Compare caller-provided spatial feature patterns across modalities."),
            ("combine_binding_evidence", "B*_ij = alpha B_ij + beta B_s + gamma B_learned", "Combine temporal, spatial, and previously learned association evidence."),
            ("update_coactivation_associations", "Delta W_ij = eta x_i x_j", "Compute a Hebbian-style association proposal from co-activation."),
        )),
    _stage(8, "Feedforward transmission to the ATL semantic hub",
        "Project each modality-specific semantic representation into a shared convergence space.",
        "Bilateral ATL regions have graded connectivity profiles; the ventrolateral ATL is proposed to contribute strongly to transmodal integration, with surrounding regions showing graded specialization.",
        (
            ("project_modality_representation", "U_m = P_m Z_m", "Map modality-specific representations into a common-dimensional hub input."),
            ("align_representation_dimensions", "U_m in R^d", "Use a common computational dimensionality without claiming it is a measured ATL neuronal dimension."),
        )),
    _stage(9, "Reliability estimation and weighted modality integration",
        "Estimate the usefulness of each available modality and weight it flexibly.",
        "Activity, connectivity, signal quality, attention, task requirements, and learned associations can alter the contribution of each modality.",
        (
            ("estimate_modality_reliability", "R_m = 1 / (sigma_m^2 + epsilon)", "Convert signal uncertainty into a reliability score."),
            ("normalize_modality_weights", "w_m = R_m / sum_j R_j; sum_m w_m = 1", "Normalize the contributions of the available modalities."),
            ("integrate_weighted_inputs", "U_combined = sum_m w_m U_m", "Combine projected evidence with reliability and task relevance weights."),
            ("apply_task_relevance_weights", "w_m = exp(q_m) / sum_j exp(q_j)", "Allow attention and current task demands to change modality influence."),
        )),
    _stage(10, "Cross-modal integration within the ATL",
        "Combine modality information into a coherent representation of the shared concept.",
        "Interconnected ATL populations receive converging cortical input; recurrent interactions support coherent representations. Hub and spokes make complementary contributions.",
        (
            ("form_initial_hub_convergence", "H_input = sum_m w_m U_m", "Create the weighted input drive for transmodal integration."),
            ("integrate_recurrently", "H^(t+1) = phi(W_in H_input + W_rec H^t + b)", "Update a shared hub state using incoming and recurrent signals."),
            ("apply_leaky_temporal_integration", "H^(t+1) = (1-lambda) H^t + lambda H_new", "Balance the newly integrated evidence with the current state."),
            ("combine_nonlinear_crossmodal_features", "H = phi(sum_m W_m U_m + b)", "Allow interactions between features from different modalities."),
        )),
    _stage(11, "Extraction of modality-independent conceptual information",
        "Extract information common across routes while retaining route-specific information.",
        "The ATL provides transmodal convergence; modality-specific spokes retain details not represented by the common conceptual state. The source discusses graded ATL specialization.",
        (
            ("extract_common_representation", "H_common = sum_m w_m H_m", "Estimate the weighted conceptual component shared by modality access routes."),
            ("measure_crossmodal_similarity", "S_ij = (H_i dot H_j) / (||H_i|| ||H_j|| + epsilon)", "Measure whether different access routes activate overlapping semantic states."),
            ("calculate_invariance_objective", "L_invariance = sum_(i != j) ||H_i - H_j||^2", "Quantify route disagreement for same-concept examples; this is an artificial-model objective."),
            ("decompose_shared_and_unique_information", "Z_m = D_m(H_common) + R_m_unique", "Retain the modality-specific remainder rather than deleting embodied detail."),
        )),
    _stage(12, "Preservation of embodied and modality-specific information",
        "Maintain modality-specific contributions alongside abstract conceptual information.",
        "Visual association regions, auditory regions, premotor cortex, somatosensory cortex, and emotional/interoceptive systems continue contributing according to task demands.",
        (
            ("reconstruct_modality_from_common_state", "Zhat_m = phi(D_m H_common + b_m)", "Predict the modality-specific representation from the shared hub state."),
            ("calculate_modality_residual", "R_m = Z_m - Zhat_m", "Retain information not captured by the common-state reconstruction."),
            ("combine_common_and_embodied_information", "Z*_m = alpha H_common + beta R_m", "Weight the common and unique components according to the current task."),
        )),
    _stage(13, "Conceptual categorization and semantic generalization",
        "Identify category membership while distinguishing similar concepts and generalizing across instances.",
        "ATL hub supports generalizable conceptual representations; distributed spokes contribute distinguishing sensory and motor properties through recurrent interaction.",
        (
            ("estimate_category_prototype", "P_c = (1/N) sum_i H_i", "Compute a category prototype from supplied instances."),
            ("score_category_similarity", "S_c = sim(H, P_c)", "Compare the current integrated representation with category prototypes."),
            ("normalize_category_posterior", "P(c|H) = exp(beta S_c) / sum_j exp(beta S_j)", "Convert category scores into a normalized candidate distribution."),
            ("measure_conceptual_distinction", "D(H_i,H_j) = ||H_i - H_j||", "Retain differences between related concepts within a common category."),
            ("select_generalized_category", "c_hat = argmax_c P(c|H)", "Return the best-supported category without removing concept-level alternatives."),
        )),
    _stage(14, "Context-dependent conceptual abstraction",
        "Adapt which conceptual properties are emphasized without necessarily changing concept identity.",
        "ATL semantic representations interact with frontal semantic-control networks, temporoparietal semantic networks, and linguistic/episodic memory systems.",
        (
            ("encode_current_context", "C = phi(W_c X_c + b_c)", "Encode the available contextual evidence."),
            ("transform_concept_with_context", "H_c = phi(W_h H + W_c_prime C)", "Adapt the integrated representation using contextual information."),
            ("weight_features_by_context", "A_i = softmax(beta S_i + gamma C_i)", "Combine conceptual relevance and contextual relevance."),
            ("construct_adapted_representation", "H_adapted = sum_i A_i F_i", "Emphasize features useful for the current situation."),
        )),
    _stage(15, "Top-down activation of embodied representations",
        "Use the context-adjusted ATL state to reactivate relevant sensory, motor, and other modality systems.",
        "Top-down association signals can influence visual and auditory association cortex, premotor and motor cortex, somatosensory cortex, and emotional/introspective systems.",
        (
            ("project_hub_to_each_modality", "Zhat_m = phi(V_m H_adapted + b_m)", "Reconstruct a modality-specific pattern from the integrated representation."),
            ("gate_reactivation_by_task", "Zhat*_m = G_m elementwise Zhat_m", "Reactivate only modalities relevant to the current task."),
            ("scale_partial_simulation", "Zhat*_m = rho_m Zhat_m; 0 <= rho_m <= 1", "Allow reactivation to be partial and task dependent."),
        )),
    _stage(16, "Predictive reconstruction and completion of missing information",
        "Use conceptual knowledge and context to predict and complete incomplete input.",
        "Recurrent processing links higher-order conceptual networks, sensory association networks, and predictive/attentional systems.",
        (
            ("generate_modality_prediction", "Xhat_m = F_m(H, C)", "Predict expected information for each available modality route."),
            ("calculate_prediction_error", "E_m = X_m - Xhat_m", "Measure disagreement between received and reconstructed information."),
            ("update_hub_from_prediction_error", "H_(t+1) = H_t + eta W_e E_m", "Use bounded prediction-error feedback to refine the interpretation."),
            ("compute_bayesian_interpretation", "P(H|X,C) proportional to P(X|H) P(H|C)", "Combine input compatibility with the concept's contextual prior."),
        )),
    _stage(17, "Semantic decision and response preparation",
        "Select and prepare a task-appropriate recognition, description, comparison, language, or action response.",
        "Inferior frontal regions contribute to semantic control; temporal semantic networks support representation; language-production and premotor/motor systems contribute according to response type.",
        (
            ("score_response_candidates", "S_i = w_1 R_i + w_2 C_i + w_3 T_i", "Score candidate responses using semantic relevance, context fit, and task suitability."),
            ("select_best_supported_response", "i* = argmax_i S_i", "Choose the candidate with the strongest combined evidence."),
            ("suppress_competing_responses", "A_i* = A_i - lambda A_competitors", "Reduce the influence of alternatives without erasing the stored knowledge."),
            ("generate_output_representation", "Y = phi(W_o H_adapted + b_o)", "Create a structured output suitable for the requested response mode."),
        )),
    _stage(18, "Feedback from the result to the conceptual system",
        "Evaluate output quality and feed errors back into representation and attentional selection.",
        "Recurrent communication links semantic representations, control networks, memory systems, and relevant sensory/motor regions; errors can change attention and neural activation.",
        (
            ("calculate_output_error", "E_o = Y_target - Y_actual", "Compare expected target evidence with the produced response when target evidence is supplied."),
            ("revise_internal_representation", "H_new = H_old + eta W_o E_o", "Use bounded error feedback to adjust the current hub state."),
            ("adjust_attention", "A_m_new = A_m_old + eta_A Delta A_m", "Increase or decrease the weight of relevant feature evidence."),
            ("request_targeted_reprocessing", "If P(H|X,C) < theta: reprocess", "Request additional feature extraction, memory retrieval, contextual analysis, or cross-modal integration when confidence is low."),
        )),
    _stage(19, "Learning, synaptic modification, and conceptual development",
        "Use verified new experience to refine distributed associations and shared conceptual representations.",
        "Potential mechanisms include LTP, LTD, synaptic efficacy changes, structural plasticity, excitation/inhibition changes, neuromodulation, and memory consolidation; cell types are not specified by the source.",
        (
            ("propose_hebbian_update", "Delta W_ij = eta x_i y_j", "Strengthen proposed links under co-activation in an explicitly authorized learning transaction."),
            ("stabilize_weights", "Delta W_ij = eta x_i y_j - lambda W_ij", "Regularize updates to limit uncontrolled strengthening."),
            ("propose_error_dependent_update", "Delta W = -eta partial L / partial W", "Offer a gradient-style computational analogy, not a claim of biological backpropagation."),
            ("integrate_repeated_experiences", "H_updated = sum_i alpha_i H_i / sum_i alpha_i", "Integrate experience representations using confidence/provenance weights."),
            ("model_gradual_association_decay", "W(t) = W_0 exp(-lambda t)", "Represent possible gradual weakening of unused associations."),
        )),
)


ALL_OPERATIONS: tuple[OperationSpec, ...] = tuple(
    operation for stage in STAGE_REGISTRY for operation in stage.steps
)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return JSON-compatible stage and substep records for documentation and result metadata."""
    return [
        {
            "stage": asdict(stage),
            "steps": [asdict(step) for step in stage.steps],
        }
        for stage in STAGE_REGISTRY
    ]


def get_stage(stage_number: int) -> StageSpec:
    """Return one registered stage by its one-based stage number."""
    if stage_number < 1 or stage_number > len(STAGE_REGISTRY):
        raise IndexError(f"stage number must be between 1 and {len(STAGE_REGISTRY)}")
    return STAGE_REGISTRY[stage_number - 1]
