"""Operation 10: virtual lesion interventions and optional lesion-symptom statistical validation."""
from __future__ import annotations

from typing import Mapping

import numpy as np


def apply_virtual_lesion(
    regional_activity: Mapping[str, np.ndarray],
    region_lesions: Mapping[str, float],
    edge_weights: Mapping[str, float] | None = None,
    edge_lesions: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Attenuate selected regional activity and modeled edges by fractional damage in [0, 1]."""
    lesioned: dict[str, np.ndarray] = {}
    normalized_lesions: dict[str, float] = {}
    for region, values in regional_activity.items():
        fraction = float(region_lesions.get(region, 0.0))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"lesion fraction for {region} must lie in [0, 1]")
        normalized_lesions[region] = fraction
        lesioned[region] = np.asarray(values, dtype=np.float64) * (1.0 - fraction)
    adjusted_edges: dict[str, float] = {}
    for edge, weight in (edge_weights or {}).items():
        fraction = float((edge_lesions or {}).get(edge, 0.0))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"edge lesion fraction for {edge} must lie in [0, 1]")
        adjusted_edges[edge] = float(weight) * (1.0 - fraction)
    baseline_score = float(np.mean([np.mean(np.abs(value)) for value in regional_activity.values()])) if regional_activity else 0.0
    lesioned_score = float(np.mean([np.mean(np.abs(value)) for value in lesioned.values()])) if lesioned else 0.0
    return {
        "baseline_activity": {name: np.asarray(value) for name, value in regional_activity.items()},
        "lesioned_activity": lesioned,
        "region_lesion_fractions": normalized_lesions,
        "adjusted_edge_weights": adjusted_edges,
        "baseline_activity_proxy": baseline_score,
        "lesioned_activity_proxy": lesioned_score,
        "activity_proxy_change": baseline_score - lesioned_score,
        "intervention_type": "virtual_fractional_attenuation",
    }


def lesion_symptom_regression(
    lesion_matrix: np.ndarray,
    behavioral_scores: np.ndarray,
    covariates: np.ndarray | None = None,
) -> dict[str, object]:
    """Fit an OLS lesion-symptom model with optional covariates using a stable least-squares solver."""
    lesions = np.asarray(lesion_matrix, dtype=np.float64)
    scores = np.asarray(behavioral_scores, dtype=np.float64).reshape(-1, 1)
    if lesions.ndim != 2 or lesions.shape[0] != scores.shape[0]:
        raise ValueError("lesion_matrix rows must match behavioral_scores")
    columns = [np.ones((lesions.shape[0], 1)), lesions]
    if covariates is not None:
        control = np.asarray(covariates, dtype=np.float64)
        if control.ndim == 1:
            control = control[:, None]
        if control.ndim != 2 or control.shape[0] != lesions.shape[0]:
            raise ValueError("covariates rows must match lesion_matrix")
        columns.append(control)
    design = np.column_stack(columns)
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, scores, rcond=None)
    predicted = design @ coefficients
    residual_sum = float(np.sum((scores - predicted) ** 2))
    total_sum = float(np.sum((scores - np.mean(scores)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 1e-12 else None
    return {
        "coefficients": coefficients[:, 0],
        "predicted_scores": predicted[:, 0],
        "residual_mse": float(np.mean((scores - predicted) ** 2)),
        "r_squared": r_squared,
        "design_rank": int(rank),
        "singular_values": singular_values,
    }


def graph_global_efficiency(edge_weights: np.ndarray) -> float:
    """Calculate weighted graph global efficiency from positive connection strengths."""
    weights = np.asarray(edge_weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("edge_weights must be square")
    count = weights.shape[0]
    if count < 2:
        return 0.0
    distances = np.full(weights.shape, np.inf, dtype=np.float64)
    np.fill_diagonal(distances, 0.0)
    positive = weights > 0.0
    distances[positive] = 1.0 / weights[positive]
    for pivot in range(count):
        distances = np.minimum(distances, distances[:, pivot, None] + distances[None, pivot, :])
    reciprocal = np.divide(1.0, distances, out=np.zeros_like(distances), where=np.isfinite(distances) & (distances > 0))
    return float(np.sum(reciprocal) / (count * (count - 1)))


class LesionValidationModel:
    """Compare healthy and virtual-lesion model behavior and optionally analyze supplied patient data."""

    def process(
        self,
        regional_activity: Mapping[str, np.ndarray],
        experiment_data: Mapping[str, object],
        edge_weights: Mapping[str, float] | None = None,
    ) -> dict[str, object]:
        """Apply requested model lesions, compare observed patient profiles and estimate lesion associations."""
        region_lesions = experiment_data.get("region_lesions", {})
        edge_lesions = experiment_data.get("edge_lesions", {})
        patient_matrix = experiment_data.get("patient_lesion_matrix")
        patient_scores = experiment_data.get("patient_behavioral_scores")
        if not region_lesions and not edge_lesions and (patient_matrix is None or patient_scores is None):
            return {"status": "awaiting_lesion_intervention_or_patient_data"}
        result = apply_virtual_lesion(regional_activity, region_lesions, edge_weights, edge_lesions)
        regression = None
        if patient_matrix is not None and patient_scores is not None:
            regression = lesion_symptom_regression(
                np.asarray(patient_matrix), np.asarray(patient_scores),
                np.asarray(experiment_data["patient_covariates"]) if "patient_covariates" in experiment_data else None,
            )
        return {
            "status": "virtual_lesion_simulated",
            **result,
            "patient_lesion_symptom_regression": regression,
            "interpretation": (
                "Virtual lesions test consequences inside this model. Necessity claims require independent "
                "patient evidence and controls for lesion extent, disconnection and covariates."
            ),
        }
