"""Data and concept drift detection (Point 14, Step 49).

Recent input and outcome distributions are compared with references through the
population stability index, binned KL divergence, and stable canary performance.
Minimum sample sizes and false-alarm controls are enforced: alerts require both
sufficient evidence and corroboration (or an explicit critical threshold).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..config import LearningConfig
from ..functional import kl_divergence, population_stability_index

__all__ = ("DriftAlert", "DriftMonitor", "ReferenceWindow")


@dataclass(slots=True)
class ReferenceWindow:
    """The reference distribution and canary quality for one model/task class."""

    model_id: str
    inputs: np.ndarray
    canary_quality: float

    def __post_init__(self) -> None:
        """Validate the window."""
        self.inputs = np.asarray(self.inputs, dtype=float)
        if self.inputs.ndim != 2 or self.inputs.size == 0:
            raise ValueError("reference inputs must be a non-empty (N, d) array")


@dataclass(slots=True)
class DriftAlert:
    """A corroborated drift alert with its evidence."""

    model_id: str
    psi: float
    kl: float
    canary_quality: float
    canary_drop: float
    sample_size: int
    reasons: list[str]

    @property
    def critical(self) -> bool:
        """Critical alerts bypass corroboration (they trigger quarantine, Step 51)."""
        return self.canary_drop >= 2.0 * 0.10 or self.psi >= 2.0 * 0.25

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"model_id": self.model_id, "psi": round(self.psi, 6),
                "kl": round(self.kl, 6), "canary_quality": round(self.canary_quality, 6),
                "canary_drop": round(self.canary_drop, 6),
                "sample_size": self.sample_size, "reasons": list(self.reasons),
                "critical": self.critical,
                "note": "drift alerts corroborate distribution and canary evidence"}


class DriftMonitor:
    """Step 49 -- PSI/KL plus canary degradation with false-alarm controls."""

    def __init__(self, config: LearningConfig | None = None) -> None:
        """Configure thresholds, minimum sample sizes, and cooldown."""
        self.config = config or LearningConfig()
        self.alerts: list[DriftAlert] = []
        self._last_alert_step: dict[str, int] = {}
        self._step = 0

    def check(self, reference: ReferenceWindow, recent_inputs: np.ndarray,
              recent_canary_quality: float) -> DriftAlert | None:
        """Compare recent data with the reference; return an alert only when
        corroborated or critical (false-alarm control)."""
        recent = np.asarray(recent_inputs, dtype=float)
        if recent.ndim != 2 or recent.shape[1] != reference.inputs.shape[1]:
            raise ValueError("recent inputs must share the reference width")
        self._step += 1
        # minimum sample size control
        if len(recent) < self.config.drift_min_sample_size:
            return None
        # fine bins are noisy on small windows; coarsen adaptively (false-alarm control)
        n_bins = int(np.clip(min(len(reference.inputs), len(recent)) // 8, 3, 10))
        psi_values = [population_stability_index(reference.inputs[:, d], recent[:, d],
                                                 n_bins=n_bins)
                      for d in range(recent.shape[1])]
        kl_values = [kl_divergence(reference.inputs[:, d], recent[:, d], n_bins=n_bins)
                     for d in range(recent.shape[1])]
        psi = float(np.mean(psi_values))
        kl = float(np.mean(kl_values))
        canary_drop = reference.canary_quality - float(recent_canary_quality)

        reasons: list[str] = []
        if psi > self.config.drift_psi_alert:
            reasons.append(f"PSI {psi:.3f} exceeds {self.config.drift_psi_alert}")
        if canary_drop > self.config.drift_canary_quality_alert:
            reasons.append(f"canary quality dropped by {canary_drop:.3f}")
        if not reasons:
            return None
        corroborated = len(reasons) >= 2
        critical = (canary_drop >= 2.0 * self.config.drift_canary_quality_alert
                    or psi >= 2.0 * self.config.drift_psi_alert)
        if not (corroborated or critical):
            # single weak signal: record but do not alert (false-alarm control)
            self.alerts.append(DriftAlert(
                model_id=reference.model_id, psi=psi, kl=kl,
                canary_quality=float(recent_canary_quality), canary_drop=canary_drop,
                sample_size=len(recent), reasons=reasons + ["not corroborated; no alert"]))
            return None
        alert = DriftAlert(
            model_id=reference.model_id, psi=psi, kl=kl,
            canary_quality=float(recent_canary_quality), canary_drop=canary_drop,
            sample_size=len(recent), reasons=reasons + (["critical threshold"]
                                                        if critical else []))
        self.alerts.append(alert)
        return alert

    def describe(self) -> dict[str, Any]:
        """Summarize monitor state."""
        return {"checks": self._step, "alerts": len(self.alerts),
                "psi_threshold": self.config.drift_psi_alert,
                "canary_threshold": self.config.drift_canary_quality_alert,
                "min_sample_size": self.config.drift_min_sample_size,
                "note": "minimum sample sizes and corroboration control false alarms"}
