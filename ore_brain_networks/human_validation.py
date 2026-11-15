"""Operation 12: independent human-reading comparison, held-out evaluation and ablation diagnostics."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def mean_absolute_error(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate mean absolute error for shape-matched human observations and model predictions."""
    left, right = np.asarray(observed, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("observed and predicted arrays must be non-empty and have equal shapes")
    return float(np.mean(np.abs(left - right)))


def root_mean_squared_error(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate root mean squared error for shape-matched continuous behavioral measures."""
    left, right = np.asarray(observed, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("observed and predicted arrays must be non-empty and have equal shapes")
    return float(np.sqrt(np.mean((left - right) ** 2)))


def split_subjects(
    subject_ids: Sequence[str], train_fraction: float = 0.6, validation_fraction: float = 0.2, seed: int = 17
) -> dict[str, list[str]]:
    """Create deterministic subject-disjoint train, validation and test groups."""
    unique = sorted({str(item) for item in subject_ids})
    if len(unique) < 3:
        raise ValueError("at least three distinct subjects are required for train/validation/test splits")
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must be positive and sum to less than one")
    shuffled = np.asarray(unique, dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    train_end = max(1, min(len(shuffled) - 2, int(round(len(shuffled) * train_fraction))))
    validation_count = max(
        1, min(len(shuffled) - train_end - 1, int(round(len(shuffled) * validation_fraction)))
    )
    validation_end = train_end + validation_count
    groups = {
        "train": shuffled[:train_end].tolist(),
        "validation": shuffled[train_end:validation_end].tolist(),
        "test": shuffled[validation_end:].tolist(),
    }
    if set(groups["train"]) & set(groups["test"]):
        raise RuntimeError("subject leakage detected between training and test groups")
    return groups


def bootstrap_interval(
    values: np.ndarray, confidence: float = 0.95, samples: int = 1000, seed: int = 17
) -> tuple[float, float] | None:
    """Estimate a percentile confidence interval for the mean using a deterministic bootstrap."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 2:
        return None
    if not 0.0 < confidence < 1.0 or samples < 100:
        raise ValueError("confidence must lie in (0,1) and bootstrap samples must be >= 100")
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, array.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def _rdm_vector(matrix: np.ndarray) -> np.ndarray:
    """Extract the upper-triangle vector from a square representational dissimilarity matrix."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("RDM must be square")
    return array[np.triu_indices(array.shape[0], k=1)]


class HumanReadingValidation:
    """Evaluate only explicitly supplied independent human data and return missing-data requirements."""

    def process(self, human_data: Mapping[str, object] | None) -> dict[str, object]:
        """Compare behavioral/neural observations, subject splits, ablations and uncertainty intervals."""
        data = dict(human_data or {})
        if not data:
            return {
                "status": "awaiting_independent_human_data",
                "required_data": [
                    "human/model reading accuracy and reaction times",
                    "human/model EEG or MEG waveforms with matching channels and time axes",
                    "human/model representational patterns or RDMs",
                    "subject, stimulus and language identifiers for held-out generalization",
                ],
                "scientific_claim": "No empirical human or biological validity is asserted without these data.",
            }
        metrics: dict[str, object] = {}
        if "human_accuracy" in data and "model_accuracy" in data:
            human = np.asarray(data["human_accuracy"], dtype=np.float64)
            model = np.asarray(data["model_accuracy"], dtype=np.float64)
            metrics["accuracy_mae"] = mean_absolute_error(human, model)
            metrics["human_accuracy_mean"] = float(np.mean(human))
            metrics["model_accuracy_mean"] = float(np.mean(model))
        if "human_reading_time_ms" in data and "model_reading_time_ms" in data:
            human_rt = np.asarray(data["human_reading_time_ms"], dtype=np.float64)
            model_rt = np.asarray(data["model_reading_time_ms"], dtype=np.float64)
            metrics["reading_time_mae_ms"] = mean_absolute_error(human_rt, model_rt)
            metrics["reading_time_rmse_ms"] = root_mean_squared_error(human_rt, model_rt)
            metrics["reading_time_error_ci95"] = bootstrap_interval(np.abs(human_rt - model_rt))
        if "human_eeg" in data and "model_eeg" in data:
            human_eeg = np.asarray(data["human_eeg"], dtype=np.float64)
            model_eeg = np.asarray(data["model_eeg"], dtype=np.float64)
            metrics["eeg_waveform_mse"] = float(np.mean((human_eeg - model_eeg) ** 2)) if human_eeg.shape == model_eeg.shape else None
        if "human_meg" in data and "model_meg" in data:
            human_meg = np.asarray(data["human_meg"], dtype=np.float64)
            model_meg = np.asarray(data["model_meg"], dtype=np.float64)
            metrics["meg_waveform_mse"] = float(np.mean((human_meg - model_meg) ** 2)) if human_meg.shape == model_meg.shape else None
        if "human_rdm" in data and "model_rdm" in data:
            human_vector = _rdm_vector(np.asarray(data["human_rdm"]))
            model_vector = _rdm_vector(np.asarray(data["model_rdm"]))
            if human_vector.shape == model_vector.shape and human_vector.size > 1 and np.std(human_vector) > 1e-12 and np.std(model_vector) > 1e-12:
                metrics["rdm_similarity"] = float(np.corrcoef(human_vector, model_vector)[0, 1])
            else:
                metrics["rdm_similarity"] = None
        subject_splits = None
        if "subject_ids" in data:
            subject_splits = split_subjects(
                data["subject_ids"],
                float(data.get("train_fraction", 0.6)),
                float(data.get("validation_fraction", 0.2)),
                int(data.get("random_seed", 17)),
            )
        ablations = {}
        for model_name, error_value in dict(data.get("alternative_model_errors", {})).items():
            if "complete_model_error" in data:
                ablations[str(model_name)] = float(error_value) - float(data["complete_model_error"])
        return {
            "status": "evaluated_supplied_independent_data" if metrics else "partial_data_metrics_unavailable",
            "metrics": metrics,
            "subject_disjoint_splits": subject_splits,
            "ablation_error_delta_vs_complete": ablations,
            "held_out_stimulus_count": len(set(data.get("test_stimulus_ids", ()))),
            "held_out_language_count": len(set(data.get("test_languages", ()))),
            "uncertainty_note": "Bootstrap intervals quantify sampling variability only; they do not correct dataset bias.",
            "limitations": (
                "Comparisons are meaningful only when stimuli, preprocessing, measurement units, subjects "
                "and time/channel axes are aligned and the test set was not used for parameter fitting."
            ),
        }
