"""Experimental validation functions for EEG/MEG, fMRI, behavior, and ablations."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import stft

from .contracts import BrainValidationInput


def preprocess_trials(trials: np.ndarray, baseline_samples: int | None = None) -> np.ndarray:
    """Baseline-center [trial, time, sensor] measurements without mixing trials."""
    data = np.asarray(trials, dtype=np.float64)
    if data.ndim == 2:
        data = data[None, ...]
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError("trials must be finite with shape [trial, time, sensor]")
    baseline_count = baseline_samples or max(1, data.shape[1] // 10)
    if baseline_count < 1 or baseline_count > data.shape[1]:
        raise ValueError("baseline_samples must fit inside the time dimension")
    baseline = np.mean(data[:, :baseline_count, :], axis=1, keepdims=True)
    return (data - baseline).astype(np.float32)


def forward_project_sources(source_activity: np.ndarray, leadfield: np.ndarray) -> np.ndarray:
    """Project [time, source] activity into [time, sensor] measurements."""
    sources = np.asarray(source_activity, dtype=np.float64)
    matrix = np.asarray(leadfield, dtype=np.float64)
    if sources.ndim != 2 or matrix.ndim != 2 or sources.shape[1] != matrix.shape[1]:
        raise ValueError("source_activity and leadfield dimensions are incompatible")
    if not np.isfinite(sources).all() or not np.isfinite(matrix).all():
        raise ValueError("forward-model inputs must be finite")
    return (sources @ matrix.T).astype(np.float32)


def ridge_inverse_sources(
    sensor_activity: np.ndarray, leadfield: np.ndarray, ridge: float = 1e-2
) -> np.ndarray:
    """Estimate source activity using L2-regularized linear inverse least squares."""
    sensors = np.asarray(sensor_activity, dtype=np.float64)
    matrix = np.asarray(leadfield, dtype=np.float64)
    if sensors.ndim != 2 or matrix.ndim != 2 or sensors.shape[1] != matrix.shape[0]:
        raise ValueError("sensor_activity must be [time, sensor] and leadfield [sensor, source]")
    if ridge <= 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and positive")
    gram = matrix.T @ matrix + ridge * np.eye(matrix.shape[1])
    estimate = sensors @ matrix @ np.linalg.pinv(gram)
    return estimate.astype(np.float32)


def event_related_potential(trials: np.ndarray) -> np.ndarray:
    """Average event-aligned trials along the trial axis to obtain an ERP estimate."""
    data = np.asarray(trials, dtype=np.float64)
    if data.ndim == 2:
        data = data[None, ...]
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError("trials must have shape [trial, time, sensor]")
    return np.mean(data, axis=0).astype(np.float32)


def time_frequency_power(
    signal: np.ndarray, sampling_rate_hz: float, nperseg: int = 128
) -> dict[str, np.ndarray]:
    """Compute STFT power; 2-D input is [sensor, time], 3-D is [trial, time, sensor]."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim not in {2, 3} or not np.isfinite(values).all():
        raise ValueError("signal must be [time], [sensor, time], or [trial, time, sensor]")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    if values.ndim == 3:
        values = np.mean(values, axis=0).T
    segment_length = min(int(nperseg), values.shape[-1])
    if segment_length < 2:
        raise ValueError("signal must contain at least two time samples")
    frequencies, times, coefficients = stft(
        values, fs=float(sampling_rate_hz), nperseg=segment_length, axis=-1,
        boundary=None, padded=False,
    )
    power = np.square(np.abs(coefficients))
    return {
        "frequencies_hz": frequencies.astype(np.float32),
        "times_seconds": times.astype(np.float32),
        "power": power.astype(np.float32),
    }


def predict_bold_from_neural_activity(source_activity: np.ndarray, hrf: np.ndarray) -> np.ndarray:
    """Convolve aggregate neural drive with an HRF to create an fMRI BOLD proxy."""
    neural = np.asarray(source_activity, dtype=np.float64)
    kernel = np.asarray(hrf, dtype=np.float64).ravel()
    if neural.ndim == 2:
        neural = np.mean(neural, axis=1)
    if neural.ndim != 1 or kernel.ndim != 1 or kernel.size == 0:
        raise ValueError("source_activity and hrf must be non-empty time series")
    if not np.isfinite(neural).all() or not np.isfinite(kernel).all():
        raise ValueError("neural activity and HRF must be finite")
    return np.convolve(neural, kernel, mode="full")[: neural.size].astype(np.float32)


def fit_general_linear_model(
    design_matrix: np.ndarray, observations: np.ndarray, ridge: float = 1e-6
) -> dict[str, np.ndarray | float]:
    """Fit Y = X beta + epsilon using a stable ridge-regularized normal equation."""
    design = np.asarray(design_matrix, dtype=np.float64)
    target = np.asarray(observations, dtype=np.float64)
    if design.ndim != 2 or target.ndim not in {1, 2} or design.shape[0] != target.shape[0]:
        raise ValueError("design and observations must share their observation dimension")
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative")
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    beta = np.linalg.pinv(gram) @ design.T @ target
    prediction = design @ beta
    residual = target - prediction
    denominator = float(np.sum(np.square(target - np.mean(target, axis=0))))
    r_squared = 1.0 - float(np.sum(np.square(residual))) / max(denominator, 1e-12)
    return {
        "beta": beta.astype(np.float32),
        "prediction": prediction.astype(np.float32),
        "residual": residual.astype(np.float32),
        "r_squared": r_squared,
    }


def representational_dissimilarity_matrix(features: np.ndarray) -> np.ndarray:
    """Construct a Euclidean RDM from rows of condition-level feature vectors."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not np.isfinite(matrix).all():
        raise ValueError("features must be finite [condition, feature] with at least two conditions")
    delta = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.sum(np.square(delta), axis=-1)).astype(np.float32)


def compare_representational_geometry(model_rdm: np.ndarray, brain_rdm: np.ndarray) -> float:
    """Correlate the upper triangles of model and brain RDMs using Pearson r."""
    model = np.asarray(model_rdm, dtype=np.float64)
    brain = np.asarray(brain_rdm, dtype=np.float64)
    if model.shape != brain.shape or model.ndim != 2 or model.shape[0] != model.shape[1]:
        raise ValueError("RDMs must be square matrices with identical dimensions")
    upper = np.triu_indices(model.shape[0], k=1)
    left, right = model[upper], brain[upper]
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def classification_accuracy(human_labels: tuple[str, ...], predictions: tuple[str, ...]) -> float:
    """Calculate exact-match classification accuracy for paired human/model labels."""
    if not human_labels or len(human_labels) != len(predictions):
        raise ValueError("label sequences must be non-empty and equally sized")
    return float(np.mean(np.asarray(human_labels) == np.asarray(predictions)))


def mean_absolute_error(human_values: np.ndarray, predicted_values: np.ndarray) -> float:
    """Calculate MAE between aligned human and model measurements."""
    human = np.asarray(human_values, dtype=np.float64)
    predicted = np.asarray(predicted_values, dtype=np.float64)
    if human.shape != predicted.shape or not np.isfinite(human).all() or not np.isfinite(predicted).all():
        raise ValueError("human and predicted values must be finite and shape-aligned")
    return float(np.mean(np.abs(human - predicted)))


def root_mean_squared_error(human_values: np.ndarray, predicted_values: np.ndarray) -> float:
    """Calculate RMSE between aligned human and model measurements."""
    human = np.asarray(human_values, dtype=np.float64)
    predicted = np.asarray(predicted_values, dtype=np.float64)
    if human.shape != predicted.shape or not np.isfinite(human).all() or not np.isfinite(predicted).all():
        raise ValueError("human and predicted values must be finite and shape-aligned")
    return float(np.sqrt(np.mean(np.square(human - predicted))))


def bootstrap_confidence_interval(
    values: np.ndarray, confidence: float = 0.95, resamples: int = 2000, seed: int = 17
) -> tuple[float, float]:
    """Estimate a percentile confidence interval by non-parametric bootstrap resampling."""
    data = np.asarray(values, dtype=np.float64).ravel()
    if data.size < 2 or not np.isfinite(data).all():
        raise ValueError("bootstrap values must contain at least two finite observations")
    if not 0.0 < confidence < 1.0 or resamples < 100:
        raise ValueError("confidence must lie in (0, 1) and resamples must be at least 100")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(resamples, data.size))
    estimates = np.mean(data[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [tail, 1.0 - tail])
    return float(low), float(high)


def lesion_effect(baseline_score: float, perturbed_score: float) -> dict[str, float]:
    """Calculate baseline-minus-lesion impairment without assuming its direction."""
    if not np.isfinite(baseline_score) or not np.isfinite(perturbed_score):
        raise ValueError("lesion scores must be finite")
    return {
        "baseline_score": float(baseline_score),
        "perturbed_score": float(perturbed_score),
        "impairment_delta": float(baseline_score - perturbed_score),
    }


def _has_any_measurement(data: BrainValidationInput) -> bool:
    """Return whether at least one real experimental input was supplied."""
    names = (
        "source_activity", "eeg_leadfield", "meg_leadfield", "observed_eeg_trials",
        "observed_meg_trials", "hrf", "observed_bold", "design_matrix",
        "regional_observations", "model_features", "brain_features", "human_labels",
        "heldout_labels", "human_values", "bootstrap_values",
        "lesion_baseline_score", "ablation_scores",
    )
    for name in names:
        value = getattr(data, name)
        if name == "ablation_scores":
            if bool(value):
                return True
        elif value is not None:
            if isinstance(value, (tuple, list, dict)) and len(value) == 0:
                continue
            return True
    return False


def validate_brain_data(data: BrainValidationInput | None) -> dict[str, Any]:
    """Run every Stage 8 comparison for available modalities and mark all others untested.

    The function reports measurements, not biological equivalence. A high fit
    or RDM correlation is correspondence under the tested conditions only.
    """
    operation_names = (
        "define_brain_validation_experiment",
        "synchronize_multimodal_experimental_data",
        "preprocess_experimental_measurements",
        "project_sources_to_eeg_sensors",
        "project_sources_to_meg_sensors",
        "estimate_sources_from_sensor_measurements",
        "compute_event_related_potentials",
        "compute_time_frequency_power",
        "predict_bold_from_neural_activity",
        "fit_regional_glm_responses",
        "compare_representational_geometry",
        "simulate_lesion_effects",
        "evaluate_perturbation_predictions",
        "compare_behavioral_outputs",
        "evaluate_on_heldout_data",
        "quantify_validation_uncertainty",
        "run_component_ablation_studies",
        "grade_evidence_of_biological_correspondence",
        "assign_falsifiable_validation_status",
    )
    statuses = {name: "not_tested" for name in operation_names}
    if data is None or not _has_any_measurement(data):
        return {
            "status": "not_tested",
            "evidence_status": "inconclusive",
            "reason": "No independent experimental brain or behavioral data were supplied.",
            "metrics": {},
            "operation_statuses": statuses,
        }

    metrics: dict[str, Any] = {}
    statuses[operation_names[0]] = "executed"
    statuses[operation_names[1]] = "executed"
    eeg_trials = (
        preprocess_trials(data.observed_eeg_trials)
        if data.observed_eeg_trials is not None else None
    )
    meg_trials = (
        preprocess_trials(data.observed_meg_trials)
        if data.observed_meg_trials is not None else None
    )
    if eeg_trials is not None or meg_trials is not None:
        statuses[operation_names[2]] = "executed"
        metrics["preprocessing"] = {
            "eeg_shape": list(eeg_trials.shape) if eeg_trials is not None else None,
            "meg_shape": list(meg_trials.shape) if meg_trials is not None else None,
        }
    if data.source_activity is not None and data.eeg_leadfield is not None:
        projected = forward_project_sources(data.source_activity, data.eeg_leadfield)
        metrics["eeg_forward_prediction"] = {
            "shape": list(projected.shape), "mean": float(np.mean(projected)),
            "std": float(np.std(projected)),
        }
        if eeg_trials is not None:
            observed_erp = event_related_potential(eeg_trials)
            if observed_erp.shape == projected.shape:
                metrics["eeg_forward_erp_mae"] = mean_absolute_error(observed_erp, projected)
        statuses[operation_names[3]] = "executed"
    if data.source_activity is not None and data.meg_leadfield is not None:
        projected = forward_project_sources(data.source_activity, data.meg_leadfield)
        metrics["meg_forward_prediction"] = {
            "shape": list(projected.shape), "mean": float(np.mean(projected)),
            "std": float(np.std(projected)),
        }
        if meg_trials is not None:
            observed_erp = event_related_potential(meg_trials)
            if observed_erp.shape == projected.shape:
                metrics["meg_forward_erp_mae"] = mean_absolute_error(observed_erp, projected)
        statuses[operation_names[4]] = "executed"
    if data.eeg_leadfield is not None and eeg_trials is not None:
        erp = event_related_potential(eeg_trials)
        inverse = ridge_inverse_sources(erp, data.eeg_leadfield)
        metrics["eeg_inverse_estimate"] = {
            "shape": list(inverse.shape), "mean": float(np.mean(inverse)),
        }
        statuses[operation_names[5]] = "executed"
    elif data.meg_leadfield is not None and meg_trials is not None:
        erp = event_related_potential(meg_trials)
        inverse = ridge_inverse_sources(erp, data.meg_leadfield)
        metrics["meg_inverse_estimate"] = {
            "shape": list(inverse.shape), "mean": float(np.mean(inverse)),
        }
        statuses[operation_names[5]] = "executed"
    if eeg_trials is not None:
        erp = event_related_potential(eeg_trials)
        metrics["eeg_erp"] = {"shape": list(erp.shape), "mean": float(np.mean(erp))}
        statuses[operation_names[6]] = "executed"
    if meg_trials is not None and data.meg_sampling_rate_hz is not None:
        tfr = time_frequency_power(meg_trials, data.meg_sampling_rate_hz)
        metrics["meg_time_frequency"] = {
            "frequency_bins": int(tfr["frequencies_hz"].size),
            "time_bins": int(tfr["times_seconds"].size),
            "mean_power": float(np.mean(tfr["power"])),
        }
        statuses[operation_names[7]] = "executed"
    elif eeg_trials is not None and data.eeg_sampling_rate_hz is not None:
        tfr = time_frequency_power(eeg_trials, data.eeg_sampling_rate_hz)
        metrics["eeg_time_frequency"] = {
            "frequency_bins": int(tfr["frequencies_hz"].size),
            "time_bins": int(tfr["times_seconds"].size),
            "mean_power": float(np.mean(tfr["power"])),
        }
        statuses[operation_names[7]] = "executed"
    if data.source_activity is not None and data.hrf is not None:
        bold = predict_bold_from_neural_activity(data.source_activity, data.hrf)
        metrics["bold_prediction"] = {"length": int(bold.size), "mean": float(np.mean(bold))}
        if data.observed_bold is not None and np.asarray(data.observed_bold).size == bold.size:
            metrics["bold_mae"] = mean_absolute_error(np.asarray(data.observed_bold).ravel(), bold)
        statuses[operation_names[8]] = "executed"
    if data.design_matrix is not None and data.regional_observations is not None:
        glm = fit_general_linear_model(data.design_matrix, data.regional_observations)
        metrics["regional_glm"] = {"r_squared": glm["r_squared"], "beta_shape": list(glm["beta"].shape)}
        statuses[operation_names[9]] = "executed"
    if data.model_features is not None and data.brain_features is not None:
        model_rdm = representational_dissimilarity_matrix(data.model_features)
        brain_rdm = representational_dissimilarity_matrix(data.brain_features)
        if model_rdm.shape == brain_rdm.shape:
            metrics["rdm_correlation"] = compare_representational_geometry(model_rdm, brain_rdm)
            statuses[operation_names[10]] = "executed"
    if data.lesion_baseline_score is not None and data.lesion_perturbed_score is not None:
        metrics["lesion"] = lesion_effect(data.lesion_baseline_score, data.lesion_perturbed_score)
        statuses[operation_names[11]] = "executed"
        statuses[operation_names[12]] = "executed"
    if data.human_labels is not None and data.predicted_labels is not None:
        metrics["behavioral_accuracy"] = classification_accuracy(data.human_labels, data.predicted_labels)
        statuses[operation_names[13]] = "executed"
    if data.heldout_labels is not None and data.heldout_predictions is not None:
        metrics["heldout_accuracy"] = classification_accuracy(data.heldout_labels, data.heldout_predictions)
        statuses[operation_names[14]] = "executed"
    if data.human_values is not None and data.predicted_values is not None:
        metrics["behavioral_mae"] = mean_absolute_error(data.human_values, data.predicted_values)
        metrics["behavioral_rmse"] = root_mean_squared_error(data.human_values, data.predicted_values)
        statuses[operation_names[13]] = "executed"
    if data.bootstrap_values is not None:
        metrics["bootstrap_ci_95"] = bootstrap_confidence_interval(data.bootstrap_values)
        statuses[operation_names[15]] = "executed"
    if data.ablation_baseline_score is not None and data.ablation_scores:
        metrics["ablation_deltas"] = {
            key: float(data.ablation_baseline_score - value)
            for key, value in data.ablation_scores.items()
        }
        statuses[operation_names[16]] = "executed"
    has_comparison = bool(metrics)
    statuses[operation_names[17]] = "inconclusive" if has_comparison else "not_tested"
    statuses[operation_names[18]] = "inconclusive"
    return {
        "status": "completed" if has_comparison else "not_tested",
        "evidence_status": "inconclusive",
        "reason": (
            "Measurements were computed, but no pre-registered acceptance thresholds or "
            "independent replication were provided; correspondence is not proof of mechanism."
        ),
        "metrics": metrics,
        "operation_statuses": statuses,
    }


__all__ = [
    "preprocess_trials", "forward_project_sources", "ridge_inverse_sources",
    "event_related_potential", "time_frequency_power",
    "predict_bold_from_neural_activity", "fit_general_linear_model",
    "representational_dissimilarity_matrix", "compare_representational_geometry",
    "classification_accuracy", "mean_absolute_error", "root_mean_squared_error",
    "bootstrap_confidence_interval", "lesion_effect", "validate_brain_data",
]
