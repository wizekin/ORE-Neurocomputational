"""Cognitive world model cooperation with simulated/observed separation (Step 49).

The world model estimates P(outcome | state, action) from *observed*
trajectories only. Every prediction is tagged as simulated
(EvidenceKind.PREDICTED in ORE terms) and can never silently mix into the
trusted empirical dataset: ``TrajectoryStore.record_step(origin="SIMULATED")``
rows are excluded from all default training loaders, and the store refuses to
mark simulated rewards as trusted training evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import OREState, stable_hash


@dataclass(frozen=True, slots=True)
class PredictedOutcome:
    """A tagged, simulated prediction of one candidate action's consequence."""

    action_id: str
    predicted_success_probability: float
    predicted_quality_delta: float
    predicted_evidence_delta: float
    predicted_cost: float
    uncertainty: float
    sample_count: int
    origin: str = "SIMULATED"
    model_version: str = "empirical-world-model-1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "predicted_success_probability": self.predicted_success_probability,
            "predicted_quality_delta": self.predicted_quality_delta,
            "predicted_evidence_delta": self.predicted_evidence_delta,
            "predicted_cost": self.predicted_cost,
            "uncertainty": self.uncertainty,
            "sample_count": self.sample_count,
            "origin": self.origin,
            "model_version": self.model_version,
        }


class EmpiricalWorldModel:
    """Learns per-action outcome statistics from validated observed transitions.

    The model is deliberately simple — per (action_id, discretized context)
    statistics of success, quality delta, evidence delta, and cost — with
    uncertainty shrinking as 1/sqrt(n). Predictions are used for cautious
    counterfactual analysis and option evaluation, never as experience.
    """

    def __init__(self, *, context_bins: int = 4) -> None:
        self.context_bins = int(context_bins)
        self._stats: dict[tuple[str, int], dict[str, float]] = {}
        self._global: dict[str, dict[str, float]] = {}
        self.fitted = False

    def _context_bin(self, state: OREState) -> int:
        difficulty = state.features.get("task.difficulty", 0.5)
        remaining = state.features.get("resource.remaining_module_calls", 0.5)
        score = 0.6 * float(difficulty) + 0.4 * float(remaining)
        return int(np.clip(score * self.context_bins, 0, self.context_bins - 1))

    def fit(
        self,
        transitions: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Fit from observed transitions: action, context bin, outcome stats."""
        for item in transitions:
            if item.get("origin", "OBSERVED") == "SIMULATED":
                continue  # never train the world model on its own predictions
            action_id = item["action_id"]
            key = (action_id, int(item.get("context_bin", 0)))
            bucket = self._stats.setdefault(
                key,
                {
                    "count": 0.0,
                    "success": 0.0,
                    "quality_delta": 0.0,
                    "evidence_delta": 0.0,
                    "cost": 0.0,
                },
            )
            global_bucket = self._global.setdefault(
                action_id,
                {
                    "count": 0.0,
                    "success": 0.0,
                    "quality_delta": 0.0,
                    "evidence_delta": 0.0,
                    "cost": 0.0,
                },
            )
            for field_name in ("count", "success", "quality_delta", "evidence_delta", "cost"):
                value = float(item.get(field_name, 0.0)) if field_name != "count" else 1.0
                bucket[field_name] += value
                global_bucket[field_name] += value
        self.fitted = True
        return {"buckets": len(self._stats), "actions": len(self._global)}

    def predict(self, state: OREState, action_id: str) -> PredictedOutcome:
        """Predict the outcome of one candidate action, tagged SIMULATED."""
        context_bin = self._context_bin(state)
        bucket = self._stats.get((action_id, context_bin)) or self._global.get(
            action_id
        ) or {"count": 0.0, "success": 0.0, "quality_delta": 0.0, "evidence_delta": 0.0, "cost": 0.0}
        count = max(1.0, bucket["count"])
        return PredictedOutcome(
            action_id=action_id,
            predicted_success_probability=bucket["success"] / count,
            predicted_quality_delta=bucket["quality_delta"] / count,
            predicted_evidence_delta=bucket["evidence_delta"] / count,
            predicted_cost=bucket["cost"] / count,
            uncertainty=1.0 / float(np.sqrt(count)),
            sample_count=int(bucket["count"]),
        )

    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": "empirical-world-model-1.0.0",
                "context_bins": self.context_bins,
                "buckets": {f"{a}:{b}": sorted(v.items()) for (a, b), v in self._stats.items()},
            }
        )

    def summary(self) -> dict[str, Any]:
        return {
            "fitted": self.fitted,
            "context_bins": self.context_bins,
            "bucket_count": len(self._stats),
            "action_count": len(self._global),
            "fingerprint": self.fingerprint(),
        }
