"""Operation 9: regional activity, hemodynamic convolution and fMRI representational comparison."""
from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy import signal
from scipy.stats import gamma


def canonical_hrf(sample_interval_seconds: float, duration_seconds: float = 32.0) -> np.ndarray:
    """Construct a normalized double-gamma hemodynamic response function approximation."""
    if sample_interval_seconds <= 0.0 or duration_seconds <= 0.0:
        raise ValueError("HRF sample interval and duration must be positive")
    times = np.arange(0.0, duration_seconds, sample_interval_seconds)
    response = gamma.pdf(times, 6.0) - gamma.pdf(times, 16.0) / 6.0
    response /= max(float(np.sum(np.abs(response))), 1e-12)
    return response


def build_rdm(patterns: np.ndarray) -> np.ndarray:
    """Calculate a Euclidean representational dissimilarity matrix for row-wise condition patterns."""
    matrix = np.asarray(patterns, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("patterns must have [condition, feature] dimensions")
    differences = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.sum(differences**2, axis=-1))


def rdm_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    """Correlate upper-triangle entries of two representational dissimilarity matrices."""
    first, second = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("RDMs must be square and have identical shapes")
    indices = np.triu_indices(first.shape[0], k=1)
    x, y = first[indices], second[indices]
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def fit_glm(design_matrix: np.ndarray, observations: np.ndarray) -> dict[str, np.ndarray | float]:
    """Fit an ordinary least-squares GLM and return coefficients, fitted values and residual error."""
    design = np.asarray(design_matrix, dtype=np.float64)
    observed = np.asarray(observations, dtype=np.float64)
    if design.ndim != 2 or observed.ndim not in (1, 2) or design.shape[0] != observed.shape[0]:
        raise ValueError("design and observations must share their first dimension")
    coefficients, _, _, _ = np.linalg.lstsq(design, observed, rcond=None)
    fitted = design @ coefficients
    return {
        "coefficients": coefficients,
        "fitted": fitted,
        "residuals": observed - fitted,
        "residual_mse": float(np.mean((observed - fitted) ** 2)),
    }


class FMRIMeasurementModel:
    """Convert simulated regional activity into a delayed BOLD proxy and optional human-data comparisons."""

    def __init__(self, sample_interval_seconds: float = 0.01, duration_seconds: float = 32.0) -> None:
        """Store the neural sample interval and canonical HRF duration for the simulation."""
        self.sample_interval_seconds = float(sample_interval_seconds)
        self.duration_seconds = float(duration_seconds)
        if self.sample_interval_seconds <= 0 or self.duration_seconds <= 0:
            raise ValueError("sample interval and HRF duration must be positive")

    def process(
        self,
        regional_activity: Mapping[str, np.ndarray],
        experiment_data: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Aggregate regions, convolve the HRF, fit a supplied GLM and compare user-provided RDMs."""
        data = dict(experiment_data or {})
        if not regional_activity:
            return {"status": "no_regional_activity"}
        region_names = list(regional_activity)
        series = [np.asarray(regional_activity[name], dtype=np.float64).reshape(-1) for name in region_names]
        lengths = {item.size for item in series}
        if len(lengths) != 1:
            raise ValueError("all regional activity traces must have the same length")
        matrix = np.column_stack(series)
        hrf = canonical_hrf(self.sample_interval_seconds, self.duration_seconds)
        bold = np.column_stack([
            signal.fftconvolve(matrix[:, index], hrf, mode="full")[:matrix.shape[0]]
            for index in range(matrix.shape[1])
        ])
        design = data.get("fmri_design_matrix")
        glm_result = fit_glm(np.asarray(design), bold) if design is not None else None
        observed_bold = data.get("human_bold")
        bold_mse = None
        if observed_bold is not None:
            observed = np.asarray(observed_bold, dtype=np.float64)
            if observed.shape == bold.shape:
                bold_mse = float(np.mean((observed - bold) ** 2))
        model_patterns = data.get("model_rdm_patterns")
        human_patterns = data.get("human_rdm_patterns")
        model_rdm = build_rdm(np.asarray(model_patterns)) if model_patterns is not None else build_rdm(matrix.T)
        human_rdm = build_rdm(np.asarray(human_patterns)) if human_patterns is not None else None
        similarity = rdm_correlation(model_rdm, human_rdm) if human_rdm is not None else None
        return {
            "status": "predicted_bold_proxy",
            "regional_activity": matrix,
            "region_names": region_names,
            "predicted_bold": bold,
            "hrf": hrf,
            "glm": glm_result,
            "human_bold_mse": bold_mse,
            "model_rdm": model_rdm,
            "human_rdm": human_rdm,
            "rdm_similarity": similarity,
            "limitations": (
                "BOLD is an indirect hemodynamic response. The generated signal is a mathematical proxy, "
                "not scanner-equivalent data or an estimate of an individual's vascular response."
            ),
        }
