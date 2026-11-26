"""Predictive metrics for promotion gates (specification Section 11)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ("false_alarms_per_unit_time", "horizon_errors", "interval_coverage",
           "mean_absolute_error", "precision_recall_f1", "root_mean_squared_error")


def mean_absolute_error(observed: Sequence[float], predicted: Sequence[float]) -> float:
    """Compute MAE with aligned time and units."""
    if len(observed) != len(predicted) or not observed:
        raise ValueError("series must align and be non-empty")
    return float(np.mean(np.abs(np.asarray(observed) - np.asarray(predicted))))


def root_mean_squared_error(observed: Sequence[float], predicted: Sequence[float]) -> float:
    """Compute RMSE with aligned time and units."""
    if len(observed) != len(predicted) or not observed:
        raise ValueError("series must align and be non-empty")
    return float(np.sqrt(np.mean((np.asarray(observed) - np.asarray(predicted)) ** 2)))


def precision_recall_f1(true_labels: Sequence[bool], predicted_labels: Sequence[bool]
                        ) -> dict[str, float]:
    """Compute event detection precision, recall, and F1 on labelled data."""
    if len(true_labels) != len(predicted_labels) or not true_labels:
        raise ValueError("label series must align and be non-empty")
    tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t and p)
    fp = sum(1 for t, p in zip(true_labels, predicted_labels) if not t and p)
    fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positives": float(tp), "false_positives": float(fp),
            "false_negatives": float(fn)}


def false_alarms_per_unit_time(predicted_labels: Sequence[bool],
                               true_labels: Sequence[bool], duration: float) -> float:
    """Compute false alarms per unit time on labelled data."""
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    false_alarms = sum(1 for t, p in zip(true_labels, predicted_labels) if p and not t)
    return float(false_alarms) / float(duration)


def interval_coverage(observed: Sequence[float],
                      intervals: Sequence[tuple[float, float]]) -> float:
    """Compute the empirical coverage of prediction intervals."""
    if len(observed) != len(intervals) or not observed:
        raise ValueError("series must align and be non-empty")
    covered = sum(1 for value, (low, high) in zip(observed, intervals) if low <= value <= high)
    return float(covered) / float(len(observed))


def horizon_errors(observed_by_horizon: dict[int, Sequence[float]],
                   predicted_by_horizon: dict[int, Sequence[float]]) -> dict[int, float]:
    """Compute horizon-specific forecast error for each declared horizon."""
    output: dict[int, float] = {}
    for horizon, observed in observed_by_horizon.items():
        predicted = predicted_by_horizon.get(horizon)
        if predicted is None or len(predicted) != len(observed):
            raise ValueError(f"horizon {horizon} series must align")
        output[int(horizon)] = mean_absolute_error(observed, predicted)
    return output
