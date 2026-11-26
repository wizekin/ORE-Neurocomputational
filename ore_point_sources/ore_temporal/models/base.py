"""The common temporal-model interface shared by every candidate family."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from ..contracts import ModelFamily, ModelIdentity

__all__ = ("ModelFamilyError", "ModelRunResult", "ModelStepOutput", "TemporalModel")


class ModelFamilyError(ValueError):
    """Raised when a model family is unavailable or misconfigured."""


@dataclass(frozen=True, slots=True)
class ModelStepOutput:
    """The result of one model step: prediction, state summary, and diagnostics."""

    prediction: np.ndarray
    state_summary: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelRunResult:
    """The aggregate result of running one model over a temporal window."""

    family: ModelFamily
    predictions: list[np.ndarray] = field(default_factory=list)
    final_state: np.ndarray | None = None
    solver_diagnostics: list[Mapping[str, Any]] = field(default_factory=list)
    total_spikes: int = 0
    compute_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    steps: int = 0

    def prediction_array(self) -> np.ndarray:
        """Return predictions as a (n_steps, n_channels) array."""
        if not self.predictions:
            return np.zeros((0, 0), dtype=float)
        return np.vstack([np.asarray(item, dtype=float).reshape(1, -1)
                          for item in self.predictions])


@runtime_checkable
class TemporalModel(Protocol):
    """The backend-neutral interface every temporal model family implements."""

    def identity(self) -> ModelIdentity: ...

    def reset(self) -> None: ...

    def state_array(self) -> np.ndarray: ...

    def load_state(self, state: np.ndarray) -> None: ...

    def step(self, values: np.ndarray, dt: float) -> ModelStepOutput: ...


def run_model_over_window(model: TemporalModel, values: np.ndarray,
                          deltas: np.ndarray, *, progress: Callable[[], None] | None = None,
                          ) -> ModelRunResult:
    """Run a model chronologically over aligned values and elapsed times."""
    import time as _time

    if values.ndim != 2:
        raise ValueError("values must be a (n_samples, n_channels) array")
    if deltas.shape[0] != values.shape[0]:
        raise ValueError("deltas must carry one elapsed time per sample (first is unused)")
    result = ModelRunResult(family=model.identity().family)
    started = _time.perf_counter()
    for index in range(values.shape[0]):
        dt = float(deltas[index])
        output = model.step(values[index], dt)
        result.predictions.append(np.asarray(output.prediction, dtype=float))
        result.final_state = np.asarray(output.state_summary, dtype=float).copy()
        result.solver_diagnostics.append(dict(output.diagnostics))
        if "spikes" in output.diagnostics:
            result.total_spikes += int(output.diagnostics["spikes"])
        if "warning" in output.diagnostics:
            result.warnings.append(str(output.diagnostics["warning"]))
        result.steps += 1
        if progress is not None:
            progress()
    result.compute_ms = round((_time.perf_counter() - started) * 1000.0, 3)
    return result
