"""Result contract for the seven-stage predictive orthography processor."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np

def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value

@dataclass(slots=True)
class PredictiveOrthographyResult:
    """Full outputs from the N3→N4 functional track and parallel LIF reference."""
    sequence_id: str
    n3_result: Any
    n4_result: Any
    left_neighbor_predictions: np.ndarray
    right_neighbor_predictions: np.ndarray
    left_neighbor_available: np.ndarray
    right_neighbor_available: np.ndarray
    visual_context_divergence: np.ndarray
    attended_feature_residuals: np.ndarray
    stage_outputs: dict[str, Any] = field(default_factory=dict)
    stage_trace: list[dict[str, Any]] = field(default_factory=list)
    operation_trace: list[dict[str, Any]] = field(default_factory=list)
    biological: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def posterior_beliefs(self) -> np.ndarray:
        return self.n4_result.posterior_beliefs

    @property
    def candidates(self) -> list[dict[str, Any]]:
        return self.n4_result.candidates

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ore.predictive_orthography.seven_stage.v1",
            "sequence_id": self.sequence_id,
            "n3_result": self.n3_result.to_dict(),
            "n4_result": self.n4_result.to_dict(),
            "left_neighbor_predictions": self.left_neighbor_predictions.tolist(),
            "right_neighbor_predictions": self.right_neighbor_predictions.tolist(),
            "left_neighbor_available": self.left_neighbor_available.astype(bool).tolist(),
            "right_neighbor_available": self.right_neighbor_available.astype(bool).tolist(),
            "visual_context_divergence": self.visual_context_divergence.tolist(),
            "attended_feature_residuals": self.attended_feature_residuals.tolist(),
            "stage_outputs": _jsonable(self.stage_outputs),
            "stage_trace": _jsonable(self.stage_trace),
            "operation_trace": _jsonable(self.operation_trace),
            "biological": _jsonable(self.biological),
            "diagnostics": _jsonable(self.diagnostics),
            "metadata": _jsonable(self.metadata),
        }
