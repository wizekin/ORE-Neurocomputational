"""Operation 8: EEG/MEG forward projection, source reconstruction and temporal signal diagnostics."""
from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy import signal

from .contracts import BNBEAConfig


def make_toy_lead_field(sensor_count: int, source_count: int, seed: int = 17) -> np.ndarray:
    """Create a deterministic normalized projection for demonstrations without a head model.

    This matrix is intentionally labelled synthetic. It is not a substitute for an EEG/MEG
    forward solution based on source geometry, orientations, head tissues and sensor positions.
    """
    if sensor_count < 1 or source_count < 1:
        raise ValueError("sensor_count and source_count must be positive")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 1.0, size=(sensor_count, source_count))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def forward_model(
    source_activity: np.ndarray,
    lead_field: np.ndarray,
    noise_sd: float = 0.0,
    seed: int = 17,
) -> np.ndarray:
    """Project [time, source] activity into [time, sensor] observations using y(t)=L*s(t)+noise."""
    sources = np.asarray(source_activity, dtype=np.float64)
    lead = np.asarray(lead_field, dtype=np.float64)
    if sources.ndim != 2 or lead.ndim != 2 or lead.shape[1] != sources.shape[1]:
        raise ValueError("lead_field must be [sensor, source] and match source_activity columns")
    if noise_sd < 0.0:
        raise ValueError("noise_sd must be non-negative")
    predicted = sources @ lead.T
    if noise_sd:
        predicted = predicted + np.random.default_rng(seed).normal(0.0, noise_sd, predicted.shape)
    return predicted


def inverse_model(
    sensor_activity: np.ndarray, lead_field: np.ndarray, regularization: float = 0.05
) -> np.ndarray:
    """Estimate neural sources with a Tikhonov/minimum-norm solution to the ill-posed inverse problem."""
    sensors = np.asarray(sensor_activity, dtype=np.float64)
    lead = np.asarray(lead_field, dtype=np.float64)
    if sensors.ndim != 2 or lead.ndim != 2 or sensors.shape[1] != lead.shape[0]:
        raise ValueError("sensor_activity must be [time, sensor] and match lead_field rows")
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    system = lead.T @ lead + float(regularization) * np.eye(lead.shape[1])
    right = lead.T @ sensors.T
    estimate = np.linalg.solve(system + 1e-10 * np.eye(system.shape[0]), right)
    return estimate.T


def event_related_average(epochs: np.ndarray) -> np.ndarray:
    """Average trial-aligned [trial, time, sensor] epochs into an event-related response."""
    values = np.asarray(epochs, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError("epochs must have [trial, time, sensor] dimensions")
    return np.mean(values, axis=0)


def time_frequency_power(values: np.ndarray, sample_rate_hz: float) -> dict[str, np.ndarray]:
    """Calculate short-time Fourier power for the first sensor/source channel as a compact diagnostic."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        series = array[:, 0]
    elif array.ndim == 1:
        series = array
    else:
        raise ValueError("time-frequency input must be one- or two-dimensional")
    if series.size < 4:
        return {"frequencies_hz": np.zeros(0), "times_seconds": np.zeros(0), "power": np.zeros((0, 0))}
    frequencies, times, transformed = signal.stft(
        series, fs=sample_rate_hz, nperseg=min(32, series.size),
        noverlap=min(16, max(0, min(32, series.size) - 1)),
        boundary=None, padded=False,
    )
    return {"frequencies_hz": frequencies, "times_seconds": times, "power": np.abs(transformed) ** 2}


class EEGMEGModel:
    """Produce separate EEG and MEG sensor predictions and optional empirical discrepancy metrics."""

    def __init__(self, config: BNBEAConfig | None = None) -> None:
        """Store noise, inverse regularization and deterministic synthetic-model settings."""
        self.config = config or BNBEAConfig()

    def process(
        self,
        source_activity: np.ndarray,
        source_names: tuple[str, ...],
        experiment_data: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Build forward/inverse EEG and MEG predictions and compare only shape-compatible recordings."""
        data = dict(experiment_data or {})
        sources = np.asarray(source_activity, dtype=np.float64)
        if sources.ndim != 2 or sources.shape[1] != len(source_names):
            raise ValueError("source_activity columns must match source_names")
        if sources.shape[0] == 0:
            return {"status": "no_neural_time_series", "source_names": list(source_names)}
        source_count = sources.shape[1]
        eeg_lead = np.asarray(data.get("eeg_lead_field", make_toy_lead_field(32, source_count, self.config.random_seed)))
        meg_lead = np.asarray(data.get("meg_lead_field", make_toy_lead_field(24, source_count, self.config.random_seed + 1)))
        eeg = forward_model(sources, eeg_lead, self.config.eeg_noise_sd, self.config.random_seed)
        meg = forward_model(sources, meg_lead, self.config.meg_noise_sd, self.config.random_seed + 1)
        eeg_sources = inverse_model(eeg, eeg_lead, self.config.inverse_regularization)
        meg_sources = inverse_model(meg, meg_lead, self.config.inverse_regularization)
        empirical_errors: dict[str, float | None] = {}
        for key, predicted in (("human_eeg", eeg), ("human_meg", meg)):
            observed = data.get(key)
            if observed is None:
                empirical_errors[key] = None
                continue
            observed_array = np.asarray(observed, dtype=np.float64)
            if observed_array.shape != predicted.shape:
                empirical_errors[key] = None
                continue
            empirical_errors[key] = float(np.mean((observed_array - predicted) ** 2))
        return {
            "status": "predicted_synthetic_forward_model" if "eeg_lead_field" not in data or "meg_lead_field" not in data else "predicted_user_supplied_forward_model",
            "eeg_sensor_prediction": eeg,
            "meg_sensor_prediction": meg,
            "eeg_source_estimate": eeg_sources,
            "meg_source_estimate": meg_sources,
            "eeg_lead_field": eeg_lead,
            "meg_lead_field": meg_lead,
            "eeg_time_frequency": time_frequency_power(eeg, 1000.0 / self.config.dt_ms),
            "meg_time_frequency": time_frequency_power(meg, 1000.0 / self.config.dt_ms),
            "empirical_waveform_mse": empirical_errors,
            "source_names": list(source_names),
            "lead_field_provenance": {
                "eeg": "user_supplied" if "eeg_lead_field" in data else "synthetic_toy_projection_not_head_model",
                "meg": "user_supplied" if "meg_lead_field" in data else "synthetic_toy_projection_not_head_model",
            },
            "limitations": (
                "No source localization claim is supported by synthetic lead fields. Real comparisons "
                "require valid sensor geometry, source spaces, conductivity/head models and preprocessing."
            ),
        }
