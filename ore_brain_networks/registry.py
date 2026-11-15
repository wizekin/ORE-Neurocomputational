"""Stable 12-operation registry for BNBEA, including equations and biological qualifications."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one BNBEA operation's purpose, mathematical model, biological analogue and I/O."""

    operation_id: str
    operation_name: str
    subsystem: str
    objective: str
    mathematical_operations: tuple[str, ...]
    biological_reference: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    evidence_track: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready operation specification."""
        return asdict(self)


_OPERATION_ROWS = (
    ("Gabor-inspired early visual processing", "Early visual encoding",
     "Convert an image into local orientation, spatial-frequency, contrast and edge-energy maps.",
     ("z-score intensity normalization", "Gabor convolution R=I*g", "quadrature energy sqrt(Re^2+Ro^2)", "ReLU thresholding", "divisive normalization r_i/(sigma+sum_j w_ij r_j)"),
     "Retinal/LGN contrast pathways and V1 orientation- and spatial-frequency-selective receptive fields; Gabor kernels are a computational approximation, not literal neural code.",
     ("visual_image", "Gabor configuration"), ("normalized_image", "orientation_energy_maps", "visual_feature_summary"), "functional"),
    ("Neural population-style grapheme prototypes", "Early visual encoding",
     "Segment candidate grapheme regions, extract feature vectors, compare against multiple prototypes, and preserve ordinal letter position.",
     ("connected components and thresholding", "feature vector v=[f1,...,fd]", "diagonal Mahalanobis distance", "RBF tuning exp(-d^2/2)", "population summation", "softmax competition", "EMA prototype update"),
     "Distributed population coding and experience-dependent tuning are broad analogies; prototype vectors are an engineering representation, not a discovered storage format.",
     ("visual_feature_maps", "grapheme_prototypes"), ("candidate_regions", "grapheme_probabilities", "position_preserving_sequence"), "functional"),
    ("Explicit simulated neural dynamics", "Neural population dynamics",
     "Advance membrane voltage, synaptic drive, spikes, refractory states and recurrent population activity over time.",
     ("I_i=I_external+sum_j w_ij s_j", "C_m dV/dt=-g_L(V-E_L)+I", "forward Euler integration", "threshold/reset", "synaptic exponential decay", "population firing rate"),
     "Current-based leaky integrate-and-fire neurons and rate populations are explicit computational reference models; parameters are not fitted human physiological constants.",
     ("population_drives", "recurrent_connectivity", "LIF_parameters"), ("voltage_traces", "spike_trains", "firing_rates", "population_activity_time_series"), "biological_reference"),
    ("Biological hierarchical visual populations", "Biological visual hierarchy",
     "Transform low-level visual features through retinotopic and increasingly complex visual populations with attention and feedback.",
     ("layer-wise affine/convolutional transforms", "nonlinear activation", "spatial pooling", "softmax attention", "recurrent feedback update"),
     "Retina/LGN, V1, V2, V4 and ventral/dorsal visual systems motivate the hierarchy; these simplified layer labels do not imply exclusive one-region/one-function assignments.",
     ("visual_feature_maps", "attention_map", "top_down_context"), ("hierarchical_visual_representations", "attention_weights", "position_sensitive_features"), "functional_and_biological"),
    ("Biological VWFA hierarchy", "Biological visual hierarchy",
     "Build posterior letter features, intermediate n-grams and anterior lexical candidates while preserving sequence order and learned visual invariance.",
     ("ordered sequence matching", "bigram/trigram extraction", "template similarity", "softmax lexical competition", "experience-dependent delta rule"),
     "Posterior-to-anterior left ventral occipitotemporal/VWFA hierarchy motivates graded orthographic complexity; exact subregional roles remain hypotheses to test.",
     ("position_preserving_sequence", "visual_hierarchy_output", "lexicon"), ("posterior_orthography", "sublexical_ngrams", "lexical_candidates"), "functional_and_biological"),
    ("Anatomical IFG/MTG/ATL network", "Multimodal language networks",
     "Propagate orthographic, lexical, contextual and conceptual evidence through a recurrent distributed language network.",
     ("feature hashing", "weighted graph propagation x[t+1]=phi(Ax[t]+Bu[t])", "vector normalization", "time-series correlation"),
     "IFG, MTG and ATL are modeled as interacting populations alongside VWFA/STG/AG; their roles are task- and subregion-dependent rather than one-to-one mappings.",
     ("lexical_candidates", "semantic_features", "context_features"), ("regional_language_activity", "integrated_concept_vector", "functional_connectivity"), "functional_and_biological"),
    ("Auditory dorsal and ventral streams", "Multimodal language networks",
     "Convert acoustic input into frequency/time features, phonological evidence, comprehension-oriented ventral activity and auditory-motor dorsal activity.",
     ("short-time Fourier transform", "power spectrum |X|^2", "cochlear-inspired filterbank", "temporal-envelope statistics", "linear modality projections"),
     "Auditory cortical/temporal ventral comprehension mappings and posterior temporal-parietal-to-IFG dorsal sensorimotor mappings are simplified network hypotheses.",
     ("auditory_waveform", "sample_rate", "context_features"), ("auditory_feature_map", "phonological_candidates", "ventral_semantic_vector", "dorsal_motor_vector"), "functional_and_biological"),
    ("EEG/MEG signal modeling", "Neural measurement models",
     "Project synchronized neural source proxies to sensor space, reconstruct sources under regularization and derive ERP/time-frequency diagnostics.",
     ("y=L s+epsilon", "Tikhonov/minimum-norm inverse", "trial averaging ERP", "STFT/wavelet power", "waveform MSE"),
     "EEG scalp potentials and MEG magnetic fields reflect summed, geometry-dependent neural currents; a generic lead field is only a toy projection unless a subject-specific forward model is supplied.",
     ("source_time_series", "EEG/MEG lead fields", "optional empirical recordings"), ("predicted_sensor_signals", "source_estimates", "ERP", "time_frequency_power", "signal_errors"), "experimental_prediction"),
    ("fMRI comparison", "Neural measurement models",
     "Aggregate regional activity, transform it with a hemodynamic response function, fit a GLM and compare regional representational geometry.",
     ("regional averaging", "HRF convolution BOLD=h*n", "least-squares GLM", "representational dissimilarity matrix", "correlation"),
     "BOLD is an indirect, delayed hemodynamic measure; it must not be compared directly with voltage or spike timing.",
     ("regional_activity", "HRF_parameters", "optional human_BOLD", "design_matrix"), ("predicted_BOLD", "GLM_coefficients", "RDM", "fMRI_discrepancy"), "experimental_prediction"),
    ("Lesion-based validation", "Causal biological validation",
     "Apply fractional virtual region or connection lesions, estimate behavioral changes and optionally regress real patient lesion-symptom data.",
     ("fractional attenuation x'=(1-rho)x", "edge attenuation W'=(1-rho)W", "delta accuracy/RT", "OLS lesion-symptom model", "graph global efficiency"),
     "Lesion observations can constrain necessity claims; a virtual lesion is a model intervention and does not reproduce all consequences of human brain injury.",
     ("healthy_network_activity", "region/edge_lesion_fractions", "optional patient_matrix_and_scores"), ("virtual_lesion_effects", "patient_regression", "network_efficiency"), "causal_model_and_empirical"),
    ("TMS-based validation", "Causal biological validation",
     "Apply time- and target-specific virtual perturbation waveforms, compare active/sham/control effects and score observed-versus-model differences.",
     ("pulse envelope P(t)", "x_TMS(t)=x(t)+P(t) or (1-kappa)x(t)", "active-minus-sham contrasts", "effect discrepancy squared"),
     "This operation simulates intervention effects only. It does not choose stimulation settings or prescribe human TMS; real effects may increase, decrease or redistribute activity.",
     ("baseline_population_traces", "virtual_target", "perturbation_profile", "optional_observed_effects"), ("active_sham_control_traces", "behavioral_effect_estimates", "TMS_model_error"), "causal_model_and_empirical"),
    ("Independent human-reading comparison", "Independent empirical validation",
     "Compare behavioral, electrophysiological and representational predictions on held-out participants/stimuli; test ablations and report uncertainty and limitations.",
     ("accuracy", "MAE/RMSE", "waveform MSE", "RDM correlation", "group/subject-disjoint splits", "bootstrap confidence intervals", "ablation delta loss"),
     "Independent reading datasets and pre-registered outcomes are required for empirical claims; software-only demonstrations cannot establish human or biological validity.",
     ("human_reading_data", "model_predictions", "subject/stimulus/language identifiers", "alternative_model_predictions"), ("behavioral_metrics", "neural_correspondence", "generalization_report", "ablation_report", "uncertainty_intervals"), "independent_empirical"),
)


OPERATION_REGISTRY = tuple(
    OperationSpec(
        operation_id=f"BNBEA-{index:03d}",
        operation_name=row[0],
        subsystem=row[1],
        objective=row[2],
        mathematical_operations=tuple(row[3]),
        biological_reference=row[4],
        inputs=tuple(row[5]),
        outputs=tuple(row[6]),
        evidence_track=row[7],
    )
    for index, row in enumerate(_OPERATION_ROWS, start=1)
)

if len(OPERATION_REGISTRY) != 12:
    raise RuntimeError("BNBEA must contain exactly 12 operations")
if tuple(spec.operation_id for spec in OPERATION_REGISTRY) != tuple(
    f"BNBEA-{index:03d}" for index in range(1, 13)
):
    raise RuntimeError("BNBEA operation identifiers must be ordered BNBEA-001..BNBEA-012")


def registry_as_dicts() -> dict[str, list[dict[str, Any]]]:
    """Return the ordered 12-operation registry for documentation, tests and JSON reports."""
    return {"operations": [spec.to_dict() for spec in OPERATION_REGISTRY]}
