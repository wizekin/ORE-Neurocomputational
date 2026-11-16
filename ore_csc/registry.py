"""Canonical 18-stage registry for the Controlled Semantic Cognition model.

Descriptions are source-traceable computational hypotheses. They do not claim
that the brain literally executes the listed software operations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one independently traceable functional and mathematical operation."""

    operation_id: str
    name: str
    mathematical_operation: str
    functional_responsibility: str


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Describe one CSC stage, objective, biological reference, and operations."""

    stage_id: str
    name: str
    objective: str
    biological_reference: str
    operations: tuple[OperationSpec, ...]


def _stage(number: int, name: str, objective: str, biology: str,
           operations: tuple[tuple[str, str, str], ...]) -> StageSpec:
    """Construct an immutable stage and assign stable hierarchical operation IDs."""
    return StageSpec(
        stage_id="CSC-{0:02d}".format(number),
        name=name,
        objective=objective,
        biological_reference=biology,
        operations=tuple(
            OperationSpec(
                operation_id="CSC-{0:02d}.{1}".format(number, index),
                name=operation_name,
                mathematical_operation=equation,
                functional_responsibility=responsibility,
            )
            for index, (operation_name, equation, responsibility) in enumerate(operations, 1)
        ),
    )


STAGE_REGISTRY: tuple[StageSpec, ...] = (
    _stage(1, "Sensory and linguistic input encoding",
        "Convert available sensory, linguistic, and caller-supplied modality evidence into a normalized input state.",
        "Auditory and visual association pathways, somatosensory systems, and language-related cortex provide distributed input features. Fourier and normalization equations are engineering analogies, not literal neural instructions.",
        (
            ("transform_signal", "X(f) = sum_t x(t) exp(-i 2 pi f t / N)", "Z-score caller-supplied numeric samples and transform them with a real FFT; audio decoding and raw image extraction remain external front ends."),
            ("extract_feature_vector", "x = [x_1, ..., x_n]", "Preserve token, modality, and context features as named graded evidence."),
            ("normalize_feature_scale", "x_hat_i = (x_i - mu_i) / (sigma_i + epsilon)", "Normalize numeric features safely while keeping public activations bounded."),
            ("integrate_temporal_input", "X_t = sum_tau alpha_tau x_(t-tau)", "Carry ordered tokens and the current cue into one request-level input representation."),
        )),
    _stage(2, "Candidate word and concept activation",
        "Activate plausible lexical/conceptual candidates in parallel without prematurely choosing a sense.",
        "Distributed lexical and semantic populations can show graded candidate activation; familiarity and learned frequency can bias initial activity.",
        (
            ("match_lexical_candidates", "s_i = sim(X, W_i)", "Compare normalized input tokens with concept labels and aliases."),
            ("apply_frequency_prior", "A_i(prior) = P(m_i)", "Use inventory frequency as a weak prior, not as a substitute for context."),
            ("accumulate_candidate_evidence", "z_i = log P(X|m_i) + log P(m_i)", "Combine lexical evidence and prior evidence in a stable log-score."),
            ("retain_parallel_candidates", "A = [A_1, A_2, ..., A_n]", "Keep all candidates and their evidence for later controlled selection."),
        )),
    _stage(3, "Distributed semantic memory activation",
        "Activate concept features and spread activity over features that co-occur in the supplied inventory.",
        "Distributed semantic feature populations interact through excitatory association pathways, recurrent activity, and activity-dependent decay.",
        (
            ("spread_semantic_activation", "A_j(t+1) = f(sum_i W_ij A_i(t) + B_j)", "Propagate graded activity through feature co-occurrence associations."),
            ("weight_associative_input", "I_j = sum_i W_ij A_i", "Weight each source feature by its association with the receiving feature."),
            ("decay_unsupported_activity", "A_i(t) = A_i(0) exp(-lambda_i t)", "Apply bounded activity decay to unsupported feature activation."),
            ("update_recurrent_semantic_state", "A_(t+1) = F(W A_t + b)", "Iterate recurrent feature updates while preserving distributed alternatives."),
        )),
    _stage(4, "ATL hub integration and concept formation",
        "Integrate available modality-specific feature evidence into a shared concept state.",
        "The anterior temporal lobes are modeled as a transmodal integration hub interacting bidirectionally with distributed sensory, motor, linguistic, and functional spokes.",
        (
            ("integrate_multimodal_inputs", "H = f(sum_m W_m S_m + b)", "Reliability-weight and combine modality evidence actually supplied."),
            ("measure_hub_candidate_similarity", "sim(H, C_i) = (H dot C_i) / (||H|| ||C_i||)", "Compare the shared hub state with candidate concept features."),
            ("complete_supported_patterns", "H* = argmax_H P(H | S_available)", "Fill only feature values supported by the candidate inventory and available evidence."),
            ("stabilize_recurrent_hub", "H_(t+1) = f(W_HH H_t + W_SH S_t)", "Use damped recurrent updates to stabilize the integrated representation."),
        )),
    _stage(5, "Context construction and contextual constraint extraction",
        "Construct the current context and estimate how well each candidate fits it.",
        "Temporal and language networks maintain recent context and bias semantic representations through recurrent contextual interactions.",
        (
            ("extract_context_features", "C_t = [c_1, ..., c_n]", "Combine explicitly supplied contextual features with transparent token cues."),
            ("accumulate_context", "C_t = alpha C_(t-1) + (1-alpha) X_t", "Maintain a bounded context state using an exponential moving average."),
            ("measure_context_association", "R_i = cosine(m_i, C_t)", "Quantify candidate/context feature overlap."),
            ("estimate_contextual_posterior", "P(m_i|C) = P(C|m_i)P(m_i) / sum_j P(C|m_j)P(m_j)", "Reweight candidate hypotheses by context compatibility."),
        )),
    _stage(6, "Task and intention representation",
        "Represent what the current task requires and maintain that goal during semantic retrieval.",
        "Prefrontal goal-maintenance systems interact with language and semantic networks; task representations alter the priority of particular features.",
        (
            ("encode_task_target", "T = [t_1, ..., t_n]", "Build a target-feature vector from caller-supplied task features."),
            ("match_task_to_features", "G_i = sim(T, F_i)", "Estimate how strongly each semantic feature matches the task target."),
            ("assign_task_priority", "Q_i = omega_i G_i", "Weight feature relevance by caller-supplied priority."),
            ("maintain_task_goal", "T_t = gamma T_(t-1) + (1-gamma) T_new", "Blend previous and current task evidence without dropping the active goal abruptly."),
        )),
    _stage(7, "Estimation of semantic control demand",
        "Estimate whether automatic retrieval is sufficient or stronger control is needed.",
        "Frontal monitoring and semantic-control mechanisms are expected to respond to uncertainty, weak target activation, competition, and unusual task constraints. The combined demand function is a model assumption.",
        (
            ("measure_semantic_entropy", "H(M) = -sum_i P_i log(P_i)", "Measure uncertainty across the active candidate distribution."),
            ("measure_candidate_conflict", "K = 1 - (P_best - P_second)", "Estimate competition from the gap between the two leading candidates."),
            ("estimate_target_weakness", "U = 1 - A_target", "Estimate whether task-relevant feature evidence is weak."),
            ("estimate_task_novelty", "N = 1 - sim(T, T_familiar)", "Estimate mismatch between current task features and the available familiar target profile."),
            ("combine_control_demand", "D = sigmoid(w_h H + w_k K + w_u U + w_n N + b)", "Combine uncertainty, conflict, weakness, and novelty into a bounded control-demand signal."),
        )),
    _stage(8, "Recruitment of the semantic control network",
        "Translate demand into distributed, task-dependent control-network activity.",
        "The reference includes left IFG, posterior middle/inferior temporal regions, dorsomedial PFC, dorsolateral PFC, and optional parietal attention. Network boundaries and parietal contributions vary by task and model.",
        (
            ("generate_control_signal", "Q = W_T T + W_C C + W_D D", "Combine task, context, and demand into a control drive."),
            ("recruit_control_regions", "N_r = f(W_r Q + b_r)", "Compute bounded activity for named control-network populations."),
            ("propagate_network_activity", "dN/dt = -Lambda N + W_N N + Q", "Perform a damped recurrent update among control populations."),
            ("modulate_effective_connectivity", "W_ij* = W_ij + Delta W_ij(D)", "Increase or relax task-specific coupling as control demand changes."),
        )),
    _stage(9, "Generation of targeted semantic retrieval signals",
        "Convert the task goal into feature-specific relevance, gain, and search priorities.",
        "Frontal control signals bias temporal semantic retrieval and relevant modality-specific feature populations rather than replacing their stored content.",
        (
            ("construct_feature_target", "F_T = [f_1, ..., f_n]", "Build a sparse target vector from task and context features."),
            ("calculate_feature_relevance", "R_i = sim(F_i, F_T)", "Measure each feature's fit to the target."),
            ("generate_feature_gain", "G_i = sigmoid(k R_i + b)", "Translate relevance into a bounded excitatory gain."),
            ("prioritize_weak_relevant_features", "E_i = (R_i + epsilon) / (A_i + epsilon)", "Prioritize features that are relevant but currently weakly activated."),
        )),
    _stage(10, "Amplification of task-relevant semantic features",
        "Increase the influence of relevant features without deleting other stored knowledge.",
        "Control-dependent excitatory modulation and recurrent feedback increase the relative accessibility of task-relevant distributed features.",
        (
            ("apply_multiplicative_gain", "A_i' = G_i A_i", "Multiply each feature activation by its control gain."),
            ("apply_additive_excitation", "A_i' = A_i + eta G_i", "Add a bounded excitation term to relevant features."),
            ("amplify_recurrently", "A_i(t+1) = f(A_i(t) + G_i + sum_j W_ji A_j(t))", "Allow relevant features to reinforce associated features through recurrent links."),
            ("modulate_feature_vector", "A' = G elementwise_product A", "Apply the feature-specific gain vector element by element."),
        )),
    _stage(11, "Inhibition of task-irrelevant semantic features",
        "Reduce current influence of irrelevant or competing activity while retaining underlying knowledge.",
        "The biological hypothesis involves interacting excitatory, inhibitory, and gain/sensitivity changes; it is not a claim that every competitor is silenced by one dedicated inhibitory mechanism.",
        (
            ("compute_subtractive_inhibition", "A_i' = A_i - I_i", "Calculate feature-specific inhibitory drive."),
            ("rectify_nonnegative_activity", "A_i' = max(0, A_i - I_i)", "Prevent negative activation values after subtractive inhibition."),
            ("apply_competitive_suppression", "I_i = beta sum_(j != i) A_j", "Estimate lateral competition from co-active features."),
            ("scale_inhibition_by_irrelevance", "I_i = lambda (1 - R_i)", "Increase inhibition when feature relevance to the task is low."),
            ("combine_excitation_and_inhibition", "A_i(t+1) = f(sum_j W_ji A_j + G_i - I_i + B_i)", "Update controlled semantic state from recurrent excitation, task gain, inhibition, and bias."),
        )),
    _stage(12, "Controlled retrieval of weak or non-dominant knowledge",
        "Iteratively retrieve relevant but weakly activated or unconventional semantic knowledge.",
        "IFG/pMTG control is modeled as directed bias on distributed semantic retrieval; stable activity is a computational stopping rule, not a measured neural endpoint.",
        (
            ("spread_targeted_activation", "A_j(t+1) = f(A_j(t) + sum_i G_i W_ij A_i(t))", "Spread activation through target-weighted associative connections."),
            ("prioritize_retrieval_candidates", "P_j = exp(beta R_j A_j) / sum_k exp(beta R_k A_k)", "Normalize relevance-weighted activation into a retrieval priority distribution."),
            ("iterate_controlled_retrieval", "A_(t+1) = F(A_t, G, W, I)", "Repeat controlled retrieval using current gain and inhibition."),
            ("detect_retrieval_convergence", "||A_(t+1) - A_t|| < epsilon", "Stop refinement when feature-state change is small or the iteration budget ends."),
        )),
    _stage(13, "Candidate meaning and feature evaluation",
        "Evaluate each candidate against semantic content, task requirements, feature relevance, and context.",
        "Distributed semantic representations and control-sensitive temporal/frontal interactions jointly evaluate candidate fit; component weights are engineering parameters.",
        (
            ("score_semantic_match", "M_i = sim(m_i, C)", "Compare each candidate with the integrated semantic representation."),
            ("score_task_match", "T_i = sim(m_i, T)", "Compare candidate features with the task target."),
            ("aggregate_feature_relevance", "F_i = sum_j w_j r_ij", "Aggregate activation and relevance of candidate features."),
            ("combine_candidate_evidence", "Z_i = alpha M_i + beta T_i + gamma F_i + delta C_i", "Combine semantic, task, feature, and contextual scores transparently."),
            ("normalize_candidate_probabilities", "P_i = softmax(Z_i / tau)", "Convert candidate scores into a temperature-controlled distribution."),
        )),
    _stage(14, "Competition resolution and semantic selection",
        "Select a meaning or compatible feature set only when confidence and separation meet explicit criteria.",
        "Competitive population dynamics and frontal/temporal selection are represented by posterior comparison and a confidence gate; software winner-take-all is not a literal neural switch.",
        (
            ("compute_winner", "m* = argmax_i Z_i", "Identify the highest-scoring candidate while retaining all alternatives."),
            ("calculate_selection_margin", "Delta Z = Z_best - Z_second", "Measure separation between leading candidate scores."),
            ("apply_selection_gate", "select when P_best >= theta_p and Delta Z >= theta_m", "Abstain when posterior confidence or candidate margin is inadequate."),
            ("report_selection_confidence", "C_selection = P(m* | X, C, T)", "Attach the leading posterior as a model-internal confidence value."),
            ("select_compatible_features", "F* = {f_i : R_i > theta_f}", "Select multiple task-relevant features when the task requires a conjunction."),
        )),
    _stage(15, "Performance monitoring and control adjustment",
        "Monitor task fit and feed errors back into gain, inhibition, and demand for another retrieval pass.",
        "Monitoring and control are attributed to interacting dmPFC/IFG/prefrontal and semantic retrieval circuitry; error signals and update rates are computational abstractions.",
        (
            ("calculate_retrieval_error", "e_t = Z_required - Z_current", "Estimate the gap between required and achieved task fit."),
            ("adjust_feature_gain", "G_(t+1) = clip(G_t + eta e_t, 0, 1)", "Increase or decrease relevant feature gain based on retrieval error."),
            ("adjust_inhibitory_drive", "I_(t+1) = clip(I_t + mu e_interference, 0, 1)", "Adjust competitor suppression in response to interference."),
            ("adapt_control_demand", "D_(t+1) = clip(D_t + rho e_t, 0, 1)", "Raise or relax control demand within a bounded range."),
            ("revise_semantic_state", "A_(t+1) = F(A_t, G_(t+1), I_(t+1))", "Run a bounded feedback loop and retain the iteration history."),
        )),
    _stage(16, "Integration of selected semantic information",
        "Combine selected features and their relations into a task-specific coherent representation.",
        "Transmodal integration and recurrent semantic binding are modeled as weighted feature convergence; feature provenance and consistency remain separately inspectable.",
        (
            ("aggregate_selected_features", "H* = sum_i w_i A_i F_i", "Build a weighted vector from selected feature activations."),
            ("construct_feature_relations", "R(F_i, F_j) = rel(F_i, F_j)", "Represent feature relations using co-occurrence and shared-concept evidence."),
            ("form_integrated_interpretation", "C_final = f(sum_i A_i F_i + sum_(i,j) R(F_i,F_j))", "Construct the selected task-specific semantic representation."),
            ("measure_relation_consistency", "K_c = compatible_relations / evaluated_relations", "Report the proportion of feature relations supported by the inventory."),
        )),
    _stage(17, "Generation of the appropriate response or behaviour",
        "Map the integrated semantic state to a permitted verbal, judgement, reasoning, or action label.",
        "Language and action output networks receive the selected semantic representation and current task state; the package ranks supplied response labels but does not generate unrestricted language.",
        (
            ("score_response_candidates", "O_i = sim(C_final, R_i)", "Compare supplied response options with the integrated representation."),
            ("choose_response", "O* = argmax_i O_i", "Select the best-supported response candidate."),
            ("gate_response_output", "emit only when C_selection >= theta_o", "Suppress output when semantic selection has not passed its confidence gate."),
            ("construct_task_response", "Y = g(C_final, T, language_context)", "Return a compact structured interpretation or a caller-supplied response label."),
        )),
    _stage(18, "Adaptive learning and control updating",
        "Separate reversible task-local control changes from explicitly authorized longer-term learning.",
        "Hebbian/co-activation, reward-modulated plasticity, and error-driven parameter adjustment are broad synaptic-learning analogies with distinct biological timescales; no neural constants are claimed.",
        (
            ("update_hebbian_associations", "Delta W_ij = eta A_i A_j", "Strengthen co-activated feature links only when external feedback authorizes learning."),
            ("apply_error_driven_update", "Delta W = -eta gradient_W L", "Adjust bounded control parameters using explicit feedback error."),
            ("apply_reward_modulation", "Delta W_ij = eta r A_i A_j", "Scale association learning by an explicit external reward signal."),
            ("update_accessibility_priors", "A_i(prior) = g(retrieval_frequency, recency)", "Update instance-local retrieval counts without silently changing the supplied concept inventory."),
            ("adapt_control_parameters", "theta_(t+1) = theta_t - eta gradient_theta L", "Update model parameters only under explicit learning authorization."),
        )),
)


ALL_OPERATIONS: tuple[OperationSpec, ...] = tuple(
    operation for stage in STAGE_REGISTRY for operation in stage.operations
)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return the complete stage and operation registry as JSON-compatible dictionaries."""
    return [
        {
            **asdict(stage),
            "operations": [asdict(operation) for operation in stage.operations],
        }
        for stage in STAGE_REGISTRY
    ]
