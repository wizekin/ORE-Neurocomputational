"""Canonical 19-stage registry for amodal semantic processing through the ATL.

The biological references summarize plausible neural systems. The equations
are executable computational analogies, not literal neuronal arithmetic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one independently traceable mathematical operation and its functional purpose."""

    operation_id: str
    name: str
    mathematical_operation: str
    functional_responsibility: str


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Describe one ordered algorithmic stage, its objective, biological reference, and sub-operations."""

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
    """Create a stage record and assign stable hierarchical identifiers to its operations."""
    return StageSpec(
        stage_id=f"ATL-{number:02d}",
        name=name,
        objective=objective,
        biological_reference=biology,
        operations=tuple(
            OperationSpec(
                operation_id=f"ATL-{number:02d}.{index:02d}",
                name=operation_name,
                mathematical_operation=math,
                functional_responsibility=responsibility,
            )
            for index, (operation_name, math, responsibility) in enumerate(operations, 1)
        ),
    )


STAGE_REGISTRY: tuple[StageSpec, ...] = (
    _stage(1, "Receive information through sensory channels",
        "Accept written and spoken words, pictures, environmental sounds, touch, actions, and contextual evidence.",
        "Parallel sensory and language pathways initially preserve channel-specific neural codes and timing.",
        (
            ("encode_input_channels", "X = {X_visual, X_auditory, X_tactile, X_motor, X_linguistic}", "Preserve each incoming channel as a separately identified evidence source."),
            ("detect_available_modalities", "A_m = 1 when modality m contains evidence", "Identify which channels are present without treating absence as negative evidence."),
            ("merge_contextual_cues", "C = encode(context_features)", "Prepare contextual evidence for later biasing of ATL convergence."),
        )),
    _stage(2, "Extract modality-specific features",
        "Transform each raw channel feature map into a graded population representation.",
        "Visual, auditory, somatosensory, motor, and language-associated cortical populations extract different feature classes.",
        (
            ("encode_modality_features", "x_m = RandomFeatureProjection(X_m)", "Project named features into a common computational dimensionality while retaining channel identity."),
            ("apply_modality_nonlinearity", "e_m = tanh(x_m)", "Model nonlinear population response and bounded activation."),
            ("retain_feature_provenance", "E = {(m, e_m)}", "Keep modality labels attached to their feature vectors."),
        )),
    _stage(3, "Activate distributed semantic spokes",
        "Activate feature populations in distributed modality-related association systems.",
        "Distributed cortical spokes contribute visual, auditory, tactile, action, linguistic, affective, and functional semantic properties.",
        (
            ("project_into_spoke_populations", "s_m = sigmoid(W_spoke,m e_m + b_m)", "Transform each modality code into its distributed semantic-spoke activation."),
            ("preserve_population_patterns", "s_m = [s_1,...,s_d]", "Represent each concept as a pattern across populations rather than one symbolic unit."),
            ("retain_sparse_activity", "s_m[i] = 0 when below theta", "Keep weak or absent feature dimensions from dominating downstream integration."),
        )),
    _stage(4, "Normalize and weight incoming feature information",
        "Account for channel strength, coverage, and reliability before combining heterogeneous signals.",
        "Neural gain and effective connectivity vary with signal quality, attention, context, and task demands.",
        (
            ("normalize_spoke_activity", "s_hat_m = tanh((s_m - mean(s_m))/(std(s_m)+epsilon))", "Place channel activity on a comparable bounded scale."),
            ("estimate_channel_reliability", "r_m = 0.5*coverage_m + 0.5*mean(abs(s_hat_m))", "Estimate how much usable evidence each present modality contributes."),
            ("calculate_modality_gates", "g_m = softmax(r_m / temperature)", "Normalize relative channel weights so their sum is one."),
        )),
    _stage(5, "Transmit distributed information toward the ATL",
        "Project modality-specific spoke patterns toward left and right anterior temporal processing populations.",
        "Heterogeneous cortico-temporal pathways transmit different modality-weighted information to bilateral ATL subdivisions.",
        (
            ("project_to_left_atl", "u_left,m = W_left,m s_hat_m", "Transform each spoke signal into the left ATL input space."),
            ("project_to_right_atl", "u_right,m = W_right,m s_hat_m", "Transform each spoke signal into the right ATL input space."),
            ("gate_pathway_strength", "U_side = sum_m g_m u_side,m", "Pool projections while retaining the estimated reliability of each modality."),
        )),
    _stage(6, "Integrate information within the bilateral ATL",
        "Recurrently combine left and right ATL inputs, interhemispheric influence, and contextual information.",
        "Bilateral ATL regions are modeled as interacting transmodal populations with heterogeneous afferent and recurrent connectivity.",
        (
            ("update_left_atl_state", "hL[t+1] = tanh(W_LL hL[t] + W_RL hR[t] + U_L + C)", "Compute the next left ATL activation from recurrent, contralateral, sensory, and context input."),
            ("update_right_atl_state", "hR[t+1] = tanh(W_RR hR[t] + W_LR hL[t] + U_R + C)", "Compute the corresponding right ATL activation."),
            ("iterate_to_stability", "Delta_t = norm(h[t+1] - h[t])", "Repeat integration until the change is small or the iteration limit is reached."),
        )),
    _stage(7, "Establish a common semantic representation",
        "Create a shared concept code across different input routes while retaining modality-specific residuals.",
        "The ATL hub hypothesis predicts overlapping transmodal semantic representations without eliminating modality-specific information.",
        (
            ("construct_route_specific_codes", "z_m = H_m(s_m)", "Estimate the hub state reached through each available input modality."),
            ("pool_shared_structure", "z_shared = sum_m g_m z_m", "Extract the common semantic signal across the available routes."),
            ("retain_modality_residuals", "r_m = z_m - z_shared", "Preserve details that remain specific to an input modality."),
            ("measure_route_alignment", "cos(z_i, z_j)", "Quantify cross-modal representational overlap without requiring identical activity."),
        )),
    _stage(8, "Perform cross-modal generalization",
        "Test whether shared representations support access to the same learned concept from different modalities.",
        "Convergent ATL representations are proposed to support semantic generalization across spoken, written, visual, and experiential input.",
        (
            ("build_concept_prototypes", "mu_c = encode(semantic_features_c, modality_features_c)", "Prepare stored concept representations in the common semantic space."),
            ("score_cross_modal_matches", "score(c) = cosine(z_shared, mu_c)", "Compare the current shared state with each known concept prototype."),
            ("estimate_candidate_probabilities", "P(c|z) = softmax(beta * score(c))", "Convert compatibility scores into a normalized candidate distribution."),
        )),
    _stage(9, "Complete the representation using feedback",
        "Reactivate expected spoke information from the hub and compare predictions with current evidence.",
        "Top-down interactions between transmodal temporal regions and distributed semantic spokes may reinstate associated features.",
        (
            ("decode_hub_to_spokes", "s_hat_m = W_feedback,m z_shared", "Predict modality-specific feature activity from the common semantic representation."),
            ("calculate_spoke_prediction_error", "e_m = s_m - s_hat_m", "Measure the difference between current and hub-predicted features."),
            ("refine_spoke_state", "s_m_new = tanh(s_m + gain*s_hat_m)", "Blend top-down reactivation with the original evidence without overwriting raw input."),
        )),
    _stage(10, "Retrieve the concept for semantic understanding",
        "Rank concept identities using semantic compatibility and available lexical evidence.",
        "A transmodal semantic representation supports concept access beyond the sensory form that initiated processing.",
        (
            ("calculate_lexical_match", "L(c) in [0,1]", "Compare the supplied cue with concept labels and aliases."),
            ("combine_lexical_and_hub_evidence", "S(c) = 0.8*cosine(z,mu_c) + 0.2*L(c)", "Combine semantic compatibility and label evidence without reducing semantics to a dictionary lookup."),
            ("select_or_abstain", "c* = argmax S(c), confidence >= threshold", "Return a leading concept only when the available evidence is sufficiently informative."),
        )),
    _stage(11, "Activate the noun semantic representation",
        "Initialize the noun concept that will be retained and modified during phrase processing.",
        "Word-related semantic activity engages distributed feature spokes and bilateral ATL representations.",
        (
            ("resolve_noun_identity", "n = prototype(noun) or encode(noun)", "Use a matching concept prototype when available, otherwise encode the noun cue."),
            ("activate_noun_features", "n0 = tanh(W_noun n)", "Transform noun evidence into a compact distributed semantic state."),
            ("bind_noun_to_context", "n_context = tanh(n0 + W_context C)", "Bias noun activation with the current semantic context."),
        )),
    _stage(12, "Maintain the noun representation over time",
        "Retain relevant noun information while the adjective and phrase relation are being processed.",
        "Sustained and recurrent activity can preserve task-relevant semantic information across successive processing moments.",
        (
            ("initialize_noun_trace", "n[0] = n_context", "Create the initial noun state for phrase composition."),
            ("apply_leaky_maintenance", "n[t+1] = rho*n[t] + (1-rho)*n_context", "Retain the noun while permitting gradual incorporation of renewed evidence."),
            ("measure_noun_stability", "stability = cosine(n[t], n[t-1])", "Quantify how consistently the maintained noun representation is preserved."),
        )),
    _stage(13, "Process the adjective semantic representation",
        "Encode the adjective independently and allow its features to vary with contextual information.",
        "Adjective and modifier information may have a more time-varying ATL representational format than noun concreteness.",
        (
            ("encode_adjective_cue", "a_raw = encode(adjective_features or adjective_text)", "Construct a distributed feature vector for the adjective."),
            ("apply_contextual_adjective_transform", "a = tanh(W_adj a_raw + W_ctx C)", "Allow the adjective state to depend on the noun phrase context."),
            ("retain_modifier_state", "a[t] = contextual activation trace", "Keep adjective information available for relation binding."),
        )),
    _stage(14, "Establish the adjective–noun relationship",
        "Represent the modifier's effect on the noun rather than merely adding two independent word vectors.",
        "Semantic composition depends on interactions between the modifier and noun concept representations.",
        (
            ("compute_elementwise_interaction", "r1 = a odot n", "Let adjective dimensions interact with corresponding noun dimensions."),
            ("compute_bilinear_interaction", "r2 = a^T B n", "Model learned adjective-by-noun effects that vary with the noun."),
            ("construct_relation_state", "r = tanh(r1 + r2)", "Produce a relation representation for the complete modifier–noun pairing."),
        )),
    _stage(15, "Construct the combined phrase representation",
        "Integrate noun, adjective, and relation information into a new semantic state.",
        "Bilateral ATL activity is proposed to contribute to semantic composition in addition to single-word semantic access.",
        (
            ("preserve_noun_contribution", "W_n n", "Carry noun identity and its semantic features into the phrase state."),
            ("incorporate_modifier_and_relation", "W_a a + W_r r", "Add the adjective and its noun-conditioned relationship."),
            ("nonlinear_phrase_composition", "p = tanh(W_n n + W_a a + W_r r + b)", "Construct a unified phrase vector rather than concatenating independent word meanings."),
        )),
    _stage(16, "Determine adjective effects on denotation",
        "Separate conceptual resemblance to the noun from actual membership in the noun category.",
        "Subsective modifiers preserve noun-category membership, whereas privative modifiers can describe non-members that resemble the noun.",
        (
            ("measure_noun_resemblance", "sim = (1 + cosine(p,n))/2", "Estimate how much the phrase representation resembles the original noun."),
            ("estimate_category_membership", "membership = f(phrase_type, sim)", "Estimate whether phrase referents remain members of the noun category."),
            ("apply_denotational_constraint", "privative => membership << resemblance", "Prevent a similar-looking fake or imitation from being treated as a genuine category member."),
        )),
    _stage(17, "Dynamically refine the phrase representation",
        "Update the phrase state as modifier meaning and its relationship to the noun become clearer.",
        "Semantic composition unfolds over time; modifier-related representational changes can alter the evolving noun phrase state.",
        (
            ("generate_refined_phrase_target", "p_target = F(n,a,r)", "Recompute the phrase target from the current noun, adjective, and relation."),
            ("apply_temporal_update", "p[t+1] = (1-alpha)*p[t] + alpha*p_target", "Blend new compositional evidence with the previous phrase state."),
            ("track_interpretation_uncertainty", "H(p) = -sum_i P_i log(P_i)", "Quantify ambiguity in the phrase/category interpretation."),
        )),
    _stage(18, "Produce a unified semantic representation for the expression",
        "Return the complete concept or phrase representation with its denotational and confidence information.",
        "Shared ATL-related semantic mechanisms are proposed to contribute to single-word access and phrase-level composition.",
        (
            ("finalize_expression_vector", "z_expression = normalize(p or z_shared)", "Return the final integrated semantic vector for the expression."),
            ("derive_phrase_interpretation", "meaning = {noun, modifier, relation}", "Expose the phrase meaning and its constituent semantic roles."),
            ("report_denotation_and_confidence", "output = {resemblance, membership, uncertainty}", "Keep similarity, category membership, and uncertainty separately available."),
        )),
    _stage(19, "Strengthen cross-modal semantic connections through experience",
        "Use repeated paired experiences to propose or apply plasticity to spoke-to-ATL pathways.",
        "Co-activation-dependent synaptic plasticity is a plausible learning principle for linking modality-specific populations with shared semantic states.",
        (
            ("construct_experience_pairs", "(x_m, y_hub) from paired episodes", "Pair each modality-specific activation with the experienced or inferred shared semantic state."),
            ("compute_hebbian_weight_change", "Delta W = eta*outer(y_hub,x_m) - lambda*W", "Strengthen co-active pathways while applying bounded decay/regularization."),
            ("apply_only_authorized_learning", "W_new = W + Delta W when authorized", "Keep proposed learning separate from persistent model mutation unless the caller explicitly authorizes it."),
        )),
)


ALL_OPERATIONS: tuple[OperationSpec, ...] = tuple(
    operation for stage in STAGE_REGISTRY for operation in stage.operations
)


def registry_as_dicts() -> dict[str, list[dict[str, object]]]:
    """Export the ordered stage and operation registry as JSON-compatible dictionaries."""
    return {
        "stages": [
            {
                **{key: value for key, value in asdict(stage).items() if key != "operations"},
                "operations": [asdict(operation) for operation in stage.operations],
            }
            for stage in STAGE_REGISTRY
        ],
        "operations": [asdict(operation) for operation in ALL_OPERATIONS],
    }
