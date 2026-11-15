"""Canonical registry for the nine detailed BNBEA stages and their operations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from ore_logger.database import LoggerDatabase


@dataclass(frozen=True, slots=True)
class StageSpec:
    """One major BNBEA stage with its biological reference and mathematical basis."""

    index: int
    code: str
    name: str
    purpose: str
    biological_reference: str
    mathematics: str
    operations: tuple[str, ...]


STAGES: tuple[StageSpec, ...] = (
    StageSpec(1, "NFS", "Neural simulation foundations",
        "Build the shared neuron, synapse, population, and time-step substrate.",
        "Membrane capacitance and leak, excitatory and inhibitory synaptic conductances, spike thresholding, refractory dynamics, and population firing rates.",
        "Conductance-based current sums; C_m dV/dt = -g_L(V-E_L)+I; Euler integration; exponential synaptic decay; spike counts divided by time; noisy state updates.",
        ("define_individual_neuron_models", "model_synaptic_transmission",
         "separate_excitatory_inhibitory_inputs", "integrate_membrane_voltage",
         "generate_spikes", "enforce_refractory_period", "estimate_firing_rate",
         "construct_neural_populations", "model_population_competition",
         "inject_neural_variability", "advance_neural_time_engine")),
    StageSpec(2, "BVP", "Biological visual processing",
        "Transform localized grayscale observations into contrast, ON/OFF, orientation, and spatial feature signals.",
        "Retinal photoreceptor adaptation, retinal ganglion ON/OFF and center-surround responses, LGN relay/gain control, and V1 orientation-selective populations.",
        "Gaussian convolution and local averaging; logarithmic response compression; DoG subtraction; positive/negative rectification; Gabor convolution; quadrature energy; divisive normalization.",
        ("simulate_photoreceptor_input", "perform_light_adaptation",
         "separate_retinal_on_off", "apply_center_surround_receptive_fields",
         "model_retinal_temporal_response", "relay_retinal_channels_to_lgn",
         "apply_lgn_contrast_gain_control", "construct_orientation_selective_populations",
         "compute_simple_cell_responses", "compute_complex_cell_energy",
         "normalize_orientation_populations", "extract_spatial_visual_feature_maps")),
    StageSpec(3, "HVA", "Hierarchical visual architecture",
        "Integrate local features into increasingly complex shape and spatial representations.",
        "Feedforward, lateral, and feedback interactions across visual cortical populations; ventral identification and dorsal spatial-motion pathways.",
        "Weighted matrix sums; receptive-field convolution; nonlinear feature composition; max/mean pooling; transformation-invariant similarity; motion differencing; softmax attention and feedback gain.",
        ("define_visual_hierarchy", "integrate_neighboring_receptive_fields",
         "compose_complex_visual_features", "pool_receptive_field_activity",
         "learn_partial_visual_invariance", "process_ventral_identification_stream",
         "process_dorsal_spatial_stream", "integrate_ventral_and_dorsal_streams",
         "propagate_descending_visual_feedback", "resolve_competing_visual_hypotheses",
         "allocate_visual_attention", "construct_integrated_visual_state")),
    StageSpec(4, "NOR", "Neural orthographic representations",
        "Represent grapheme identity, letter position, ordered combinations, word forms, and phonological links.",
        "Retinotopic visual populations, posterior/intermediate/anterior orthographic populations including VWFA-related processing, and learned links to phonological populations.",
        "Feature vectors and prototype distances (including Mahalanobis distance); softmax normalization; entropy; ordinal position encoding; n-gram counts; edit distance; cross-modal conditional probabilities.",
        ("segment_candidate_grapheme_regions", "construct_grapheme_selective_populations",
         "compute_grapheme_population_similarity", "preserve_grapheme_ambiguity",
         "encode_retinotopic_letter_positions", "transform_to_ordinal_letter_positions",
         "construct_ordered_letter_ngrams", "compare_ordinal_sequences",
         "construct_posterior_orthographic_populations",
         "construct_intermediate_orthographic_populations",
         "construct_anterior_lexical_form_populations", "learn_visual_form_invariance",
         "map_orthography_to_phonology", "assemble_orthographic_output")),
    StageSpec(5, "NCR", "Neural connectivity and recurrence",
        "Connect specialized populations with directed feedforward, feedback, and lateral pathways.",
        "Distributed cortical population nodes, recurrent excitatory/inhibitory loops, activity-dependent gain, and population-level rhythms.",
        "Directed adjacency and weight matrices; matrix-vector products; recurrent state equations; inhibitory subtraction; phase-locking value |mean(exp(i phase))|; stability and gain checks.",
        ("define_network_population_nodes", "construct_directed_connectivity_matrix",
         "initialize_connection_properties", "propagate_feedforward_signals",
         "propagate_feedback_signals", "propagate_lateral_signals",
         "integrate_population_inputs", "simulate_recurrent_network_loops",
         "stabilize_recurrent_activity", "measure_population_phase_coherence",
         "resolve_recurrent_competition", "measure_network_level_activity",
         "modulate_effective_connection_strengths")),
    StageSpec(6, "BLM", "Biological learning and memory",
        "Adapt synaptic efficacy and store, retrieve, replay, and consolidate experience representations.",
        "Hebbian co-activation, timing-dependent plasticity, LTP/LTD-like changes, homeostatic regulation, metaplasticity, and fast/slow memory systems.",
        "Co-activity products; STDP exponential timing windows; positive/negative weight updates; clipping/L2 normalization; moving averages; eligibility traces; cosine retrieval; decaying traces and slow consolidation.",
        ("measure_synaptic_coactivation", "apply_hebbian_plasticity",
         "apply_spike_timing_dependent_plasticity", "model_long_term_potentiation",
         "model_long_term_depression", "constrain_synaptic_weights",
         "monitor_homeostatic_activity", "apply_homeostatic_scaling",
         "adapt_metaplastic_thresholds", "encode_new_experience",
         "store_fast_associative_memory", "retrieve_associated_memory",
         "replay_stored_neural_sequences", "consolidate_fast_memory_to_slow_memory",
         "model_experience_dependent_forgetting", "separate_memory_timescales")),
    StageSpec(7, "PCP", "Predictive and contextual processing",
        "Combine contextual priors and sensory likelihoods, estimate prediction errors, and revise beliefs.",
        "Hierarchical top-down projections, precision-sensitive error signalling, context-conditioned population activity, and recurrent hypothesis competition.",
        "Bayes' rule; posterior normalization; residual subtraction; inverse-variance precision; weighted squared error; negative log probability/surprisal; entropy; cosine similarity; information gain.",
        ("construct_hierarchical_generative_model", "retrieve_contextual_prior",
         "generate_top_down_predictions", "maintain_multilevel_predictions",
         "compute_sensory_prediction_error", "weight_error_by_precision",
         "propagate_prediction_error", "update_latent_beliefs_from_error",
         "compute_bayesian_posterior", "rank_hypotheses_by_negative_log_cost",
         "predict_next_lexical_candidates", "compute_word_surprisal",
         "resolve_lexical_population_competition", "integrate_semantic_world_knowledge",
         "revise_interpretations_from_new_evidence", "quantify_hypothesis_uncertainty",
         "allocate_attention_by_information_gain",
         "update_predictive_associations_from_learning")),
    StageSpec(8, "BDV", "Brain-data validation",
        "Compare model states with independently supplied behavioral and neural measurements.",
        "EEG/MEG sensor projection and inverse estimation, ERP and oscillatory analyses, BOLD/HRF modeling, fMRI GLMs, lesion/perturbation outcomes, and human behavior.",
        "Linear forward/inverse models; ridge least squares; trial averaging; short-time Fourier transform; HRF convolution; GLM; representational dissimilarity; correlation; MAE/RMSE; bootstrap intervals and ablation deltas.",
        ("define_brain_validation_experiment", "synchronize_multimodal_experimental_data",
         "preprocess_experimental_measurements", "project_sources_to_eeg_sensors",
         "project_sources_to_meg_sensors", "estimate_sources_from_sensor_measurements",
         "compute_event_related_potentials", "compute_time_frequency_power",
         "predict_bold_from_neural_activity", "fit_regional_glm_responses",
         "compare_representational_geometry", "simulate_lesion_effects",
         "evaluate_perturbation_predictions", "compare_behavioral_outputs",
         "evaluate_on_heldout_data", "quantify_validation_uncertainty",
         "run_component_ablation_studies", "grade_evidence_of_biological_correspondence",
         "assign_falsifiable_validation_status")),
    StageSpec(9, "INO", "Integrated neurocomputational ORE",
        "Coordinate all functional modules and the biological simulation while preserving provenance and uncertainty.",
        "Distributed, specialized neural populations and their recurrent integration; the functional and biological tracks are compared but not conflated.",
        "Typed state contracts; weighted module messages; recurrence; uncertainty/error-based priority; integrity/fingerprint checks; versioned update acceptance and structured reporting.",
        ("construct_master_ore_controller", "construct_specialized_stage_controllers",
         "define_shared_computational_contracts", "initialize_integrated_neural_network",
         "initialize_sensory_observation", "execute_biological_sensory_processing",
         "execute_visual_hierarchical_processing", "construct_integrated_orthographic_representation",
         "activate_language_network_interfaces", "integrate_global_neural_activity",
         "coordinate_recurrent_module_communication",
         "coordinate_functional_and_biological_tracks",
         "coordinate_memory_and_learning", "coordinate_predictive_processing",
         "propagate_cross_module_prediction_errors", "monitor_global_network_stability",
         "allocate_computational_resources", "maintain_persistent_model_states",
         "determine_processing_completion", "collect_execution_annotations_and_traces",
         "assemble_integrated_neurocomputational_report",
         "compare_integrated_outputs_with_experimental_evidence",
         "govern_controlled_model_updates", "return_complete_neurocomputational_output")),
)


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Traceable small operation linked to one of the nine master stages."""

    operation_id: str
    stage_index: int
    stage_code: str
    stage_name: str
    name: str
    explanation: str
    biological_reference: str
    mathematics: str


def build_operation_registry() -> tuple[OperationSpec, ...]:
    """Expand stage definitions into uniquely numbered operation specifications."""
    rows: list[OperationSpec] = []
    for stage in STAGES:
        for index, name in enumerate(stage.operations, start=1):
            operation_id = f"{stage.code}-{index:03d}"
            explanation = (
                f"{operation_id} {name.replace('_', ' ')}. {stage.purpose} "
                f"Biological reference: {stage.biological_reference} "
                f"Mathematical operations: {stage.mathematics} "
                "This is a computational approximation and requires independent empirical validation."
            )
            rows.append(OperationSpec(
                operation_id, stage.index, stage.code, stage.name, name,
                explanation, stage.biological_reference, stage.mathematics,
            ))
    identifiers = [row.operation_id for row in rows]
    names = [row.name for row in rows]
    if len(set(identifiers)) != len(identifiers) or len(set(names)) != len(names):
        raise RuntimeError("BNBEA operation identifiers and names must be unique")
    return tuple(rows)


OPERATION_SPECS = build_operation_registry()
STAGE_OPERATION_COUNTS = {stage.code: len(stage.operations) for stage in STAGES}


def operation_symbol_id(operation_name: str) -> str:
    """Return the stable SQLite annotation identifier for a master operation."""
    source_path = "ore_bnbea/operation_registry.py"
    raw = f"{source_path}|operation|{operation_name}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def register_operation_annotations(database: LoggerDatabase) -> int:
    """Store biological and mathematical explanations for all detailed operations."""
    for spec in OPERATION_SPECS:
        database.register_annotation(
            symbol_id=operation_symbol_id(spec.name),
            module_name="ore_bnbea.operation_registry",
            qualified_name=spec.name,
            symbol_kind="operation",
            source_path="ore_bnbea/operation_registry.py",
            annotation=spec.explanation,
            annotation_source="operation_catalog",
            source_hash=hashlib.sha256(spec.explanation.encode("utf-8")).hexdigest(),
        )
    return len(OPERATION_SPECS)


def get_stage_spec(stage_index: int) -> StageSpec:
    """Retrieve a stage definition by its one-based stage index."""
    if not 1 <= stage_index <= len(STAGES):
        raise ValueError("stage_index must be between 1 and 9")
    return STAGES[stage_index - 1]


__all__ = [
    "STAGES", "OperationSpec", "OPERATION_SPECS", "STAGE_OPERATION_COUNTS",
    "build_operation_registry", "operation_symbol_id",
    "register_operation_annotations", "get_stage_spec",
]
