"""Task quality and calibration monitoring (Point 14, Step 48).

Task-specific accuracy, false-positive/negative rates, calibration error,
abstention, disagreement, latency and resource use are tracked per task class.
Confidence scores from different models are not comparable unless calibrated -- the
monitor therefore stores each model's calibration context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..config import LearningConfig
from ..functional import brier_score, expected_calibration_error, reliability_bins

__all__ = ("CalibrationReport", "TaskQualityMonitor", "quality_metrics")


def quality_metrics(predictions: Sequence[float], outcomes: Sequence[float],
                    *, threshold: float = 0.5) -> dict[str, Any]:
    """Accuracy, false-positive/negative rates, and abstention for binary tasks."""
    predicted = np.asarray(predictions, dtype=float)
    actual = np.asarray(outcomes, dtype=float)
    if predicted.shape != actual.shape or predicted.size == 0:
        raise ValueError("predictions and outcomes must align and be non-empty")
    labels = (predicted >= threshold).astype(int)
    truth = (actual >= threshold).astype(int)
    tp = int(np.sum((labels == 1) & (truth == 1)))
    fp = int(np.sum((labels == 1) & (truth == 0)))
    fn = int(np.sum((labels == 0) & (truth == 1)))
    tn = int(np.sum((labels == 0) & (truth == 0)))
    accuracy = (tp + tn) / max(labels.size, 1)
    return {"accuracy": round(float(accuracy), 6),
            "false_positive_rate": round(fp / max(fp + tn, 1), 6),
            "false_negative_rate": round(fn / max(fn + tp, 1), 6),
            "n": int(labels.size)}


@dataclass(slots=True)
class CalibrationReport:
    """Step 48 -- calibration measurements for one model on one task class."""

    model_id: str
    brier: float
    ece: float
    reliability: dict[str, list[float]]
    sample_size: int
    comparable_across_models: bool
    note: str = "confidence is comparable only when calibration context matches"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"model_id": self.model_id, "brier": round(self.brier, 6),
                "ece": round(self.ece, 6), "reliability": self.reliability,
                "sample_size": self.sample_size,
                "comparable_across_models": self.comparable_across_models,
                "note": self.note}


class TaskQualityMonitor:
    """Step 48 -- per-task-class quality and calibration tracking."""

    def __init__(self, config: LearningConfig | None = None) -> None:
        """Configure alert thresholds."""
        self.config = config or LearningConfig()
        self.reports: dict[str, list[dict[str, Any]]] = {}

    def calibration_report(self, model_id: str, probabilities: Sequence[float],
                           outcomes: Sequence[float], *,
                           task_class: str = "default",
                           n_bins: int = 10) -> CalibrationReport:
        """Compute Brier, ECE, and the reliability diagram for one model."""
        probability = np.asarray(probabilities, dtype=float)
        outcome = np.asarray(outcomes, dtype=float)
        if probability.shape != outcome.shape or probability.size == 0:
            raise ValueError("probabilities and outcomes must align and be non-empty")
        brier = brier_score(probability, outcome)
        ece = expected_calibration_error(probability, outcome, n_bins=n_bins)
        proportions, accuracies, counts = reliability_bins(probability, outcome,
                                                           n_bins=n_bins)
        report = CalibrationReport(
            model_id=model_id, brier=brier, ece=ece,
            reliability={"bin_confidence": [round(float(v), 6) for v in proportions],
                         "bin_accuracy": [round(float(v), 6) for v in accuracies],
                         "bin_counts": [int(v) for v in counts]},
            sample_size=int(probability.size),
            comparable_across_models=False)
        self.reports.setdefault(task_class, []).append(report.to_dict())
        return report

    def ece_alert(self, report: CalibrationReport) -> bool:
        """True when calibration error crosses the configured alert threshold."""
        return report.ece > self.config.calibration_ece_alert

    def record_execution(self, task_class: str, *, latency_ms: float,
                         memory_mb: float = 0.0, abstained: bool = False,
                         disagreement: float | None = None) -> dict[str, Any]:
        """Record one execution's operational metrics for the task class."""
        entry = {"latency_ms": round(float(latency_ms), 3),
                 "memory_mb": round(float(memory_mb), 3),
                 "abstained": bool(abstained),
                 "disagreement": (round(float(disagreement), 6)
                                  if disagreement is not None else None)}
        self.reports.setdefault(task_class, []).append(entry)
        return entry

    def summarize(self, task_class: str) -> dict[str, Any]:
        """Summarize tracked metrics for one task class."""
        entries = self.reports.get(task_class, [])
        latencies = [entry["latency_ms"] for entry in entries
                     if "latency_ms" in entry]
        summary: dict[str, Any] = {"entries": len(entries)}
        if latencies:
            ordered = sorted(latencies)
            summary["p50_latency_ms"] = round(ordered[len(ordered) // 2], 3)
            summary["p95_latency_ms"] = round(
                ordered[min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)], 3)
        eces = [entry["ece"] for entry in entries if "ece" in entry]
        if eces:
            summary["latest_ece"] = round(float(eces[-1]), 6)
            summary["calibration_alert"] = bool(
                float(eces[-1]) > self.config.calibration_ece_alert)
        return summary
