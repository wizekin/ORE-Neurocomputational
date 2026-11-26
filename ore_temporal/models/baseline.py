"""Deterministic baselines (specification Step 3 -- the control condition).

Every model-benefit claim is measured against these baselines: moving average,
exponential smoothing, a threshold detector, and a one-dimensional-per-channel
Kalman filter.  They are also the documented fallback when admission or health
checks fail.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..contracts import ModelFamily, ModelIdentity
from .base import ModelStepOutput

__all__ = ("BaselineKind", "ExponentialSmoothingBaseline", "MovingAverageBaseline",
           "SimpleKalmanBaseline", "ThresholdDetectorBaseline", "build_baseline")


def _default_identity(variant: str) -> ModelIdentity:
    """Return the shared identity of deterministic baselines."""
    return ModelIdentity(
        family=ModelFamily.BASELINE,
        model_version=f"{variant}-1.0.0",
        config_hash="deterministic-no-parameters",
        backend="cpu_reference",
        parameters_owner="ore_temporal.deterministic_baseline",
    )


class MovingAverageBaseline:
    """Trailing moving-average forecaster with a documented window."""

    def __init__(self, n_channels: int, window: int = 8, seed: int = 0) -> None:
        """Configure the window; the seed is accepted for interface uniformity."""
        if window <= 0:
            raise ValueError("window must be positive")
        self.n_channels = int(n_channels)
        self.window = int(window)
        self._history: deque[np.ndarray] = deque(maxlen=self.window)
        self._identity = _default_identity("moving_average")

    def identity(self) -> ModelIdentity:
        """Return the baseline identity."""
        return self._identity

    def reset(self) -> None:
        """Clear the trailing history."""
        self._history.clear()

    def state_array(self) -> np.ndarray:
        """Return the flat trailing history as state."""
        if not self._history:
            return np.zeros(0, dtype=float)
        return np.concatenate([np.asarray(item, dtype=float).reshape(-1)
                                for item in self._history])

    def load_state(self, state: np.ndarray) -> None:
        """Restore a trailing history produced by another instance."""
        array = np.asarray(state, dtype=float)
        if array.size == 0 or array.size % self.n_channels != 0:
            raise ValueError("state does not align with the channel count")
        rows = array.reshape(-1, self.n_channels)
        self._history.clear()
        for row in rows:
            self._history.append(row.copy())

    def step(self, values: np.ndarray, dt: float) -> ModelStepOutput:
        """Predict the next value as the trailing mean of the observed window."""
        current = np.asarray(values, dtype=float).reshape(-1)
        self._history.append(current.copy())
        prediction = (np.mean(np.vstack(self._history), axis=0)
                      if self._history else current.copy())
        return ModelStepOutput(prediction=prediction,
                               state_summary=self.state_array(),
                               diagnostics={"kind": "moving_average"})

    def predict(self, values: np.ndarray, dt: float) -> np.ndarray:
        """Return the moving-average prediction for the next step."""
        return self.step(values, dt).prediction


class ExponentialSmoothingBaseline:
    """Constant-level exponential smoothing s_t = alpha * x_t + (1 - alpha) * s_{t-1}."""

    def __init__(self, n_channels: int, alpha: float = 0.5, seed: int = 0) -> None:
        """Configure the smoothing level; alpha must be in (0, 1]."""
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.n_channels = int(n_channels)
        self.alpha = float(alpha)
        self._level: np.ndarray | None = None
        self._identity = _default_identity("exponential_smoothing")

    def identity(self) -> ModelIdentity:
        """Return the baseline identity."""
        return self._identity

    def reset(self) -> None:
        """Reset the smoothing level to cold start."""
        self._level = None

    def state_array(self) -> np.ndarray:
        """Return the current smoothing level as state."""
        return (self._level.copy() if self._level is not None
                else np.zeros(self.n_channels, dtype=float))

    def load_state(self, state: np.ndarray) -> None:
        """Restore a smoothing level."""
        array = np.asarray(state, dtype=float).reshape(-1)
        if array.size != self.n_channels:
            raise ValueError("state does not align with the channel count")
        self._level = array.copy()

    def step(self, values: np.ndarray, dt: float) -> ModelStepOutput:
        """Update the level and predict it as the next value."""
        current = np.asarray(values, dtype=float).reshape(-1)
        if self._level is None:
            self._level = current.copy()
        else:
            self._level = self.alpha * current + (1.0 - self.alpha) * self._level
        return ModelStepOutput(prediction=self._level.copy(),
                               state_summary=self.state_array(),
                               diagnostics={"kind": "exponential_smoothing"})

    def predict(self, values: np.ndarray, dt: float) -> np.ndarray:
        """Return the smoothed prediction for the next step."""
        return self.step(values, dt).prediction


@dataclass(frozen=True, slots=True)
class _KalmanTuning:
    """Process and observation noise variances for the constant-level Kalman baseline."""

    process_variance: float = 1e-3
    observation_variance: float = 1.0


class SimpleKalmanBaseline:
    """A scalar-per-channel constant-level Kalman filter baseline."""

    def __init__(self, n_channels: int, process_variance: float = 1e-3,
                 observation_variance: float = 1.0, seed: int = 0) -> None:
        """Configure the noise variances."""
        if process_variance <= 0.0 or observation_variance <= 0.0:
            raise ValueError("noise variances must be positive")
        self.n_channels = int(n_channels)
        self.tuning = _KalmanTuning(process_variance, observation_variance)
        self._level = np.zeros(self.n_channels, dtype=float)
        self._variance = np.ones(self.n_channels, dtype=float)
        self._initialized = False
        self._identity = _default_identity("kalman")

    def identity(self) -> ModelIdentity:
        """Return the baseline identity."""
        return self._identity

    def reset(self) -> None:
        """Reset the filter to its prior."""
        self._level = np.zeros(self.n_channels, dtype=float)
        self._variance = np.ones(self.n_channels, dtype=float)
        self._initialized = False

    def state_array(self) -> np.ndarray:
        """Return the concatenated level and variance state."""
        return np.concatenate([self._level, self._variance])

    def load_state(self, state: np.ndarray) -> None:
        """Restore level and variance state."""
        array = np.asarray(state, dtype=float).reshape(-1)
        if array.size != 2 * self.n_channels:
            raise ValueError("state does not align with the Kalman state layout")
        self._level = array[: self.n_channels].copy()
        self._variance = array[self.n_channels:].copy()
        self._initialized = True

    def step(self, values: np.ndarray, dt: float) -> ModelStepOutput:
        """Run the predict/update cycle and return the filtered level."""
        current = np.asarray(values, dtype=float).reshape(-1)
        if not self._initialized:
            self._level = current.copy()
            self._variance = np.ones(self.n_channels, dtype=float)
            self._initialized = True
        else:
            self._variance = self._variance + self.tuning.process_variance
            gain = self._variance / (self._variance + self.tuning.observation_variance)
            self._level = self._level + gain * (current - self._level)
            self._variance = (1.0 - gain) * self._variance
        return ModelStepOutput(prediction=self._level.copy(),
                               state_summary=self.state_array(),
                               diagnostics={"kind": "kalman"})

    def predict(self, values: np.ndarray, dt: float) -> np.ndarray:
        """Return the Kalman-filtered prediction for the next step."""
        return self.step(values, dt).prediction


class ThresholdDetectorBaseline:
    """A deterministic threshold detector used as the event-detection control condition."""

    def __init__(self, n_channels: int, threshold: float = 3.0, seed: int = 0) -> None:
        """Configure the absolute threshold applied per channel."""
        self.n_channels = int(n_channels)
        self.threshold = float(threshold)
        self._level = np.zeros(self.n_channels, dtype=float)
        self._identity = _default_identity("threshold_detector")

    def identity(self) -> ModelIdentity:
        """Return the baseline identity."""
        return self._identity

    def reset(self) -> None:
        """Reset the reference level."""
        self._level = np.zeros(self.n_channels, dtype=float)

    def state_array(self) -> np.ndarray:
        """Return the reference level as state."""
        return self._level.copy()

    def load_state(self, state: np.ndarray) -> None:
        """Restore the reference level."""
        array = np.asarray(state, dtype=float).reshape(-1)
        if array.size != self.n_channels:
            raise ValueError("state does not align with the channel count")
        self._level = array.copy()

    def step(self, values: np.ndarray, dt: float) -> ModelStepOutput:
        """Record the current level and predict its persistence."""
        current = np.asarray(values, dtype=float).reshape(-1)
        self._level = current.copy()
        crossings = int(np.count_nonzero(np.abs(current) > self.threshold))
        return ModelStepOutput(prediction=current.copy(),
                               state_summary=self.state_array(),
                               diagnostics={"kind": "threshold_detector",
                                             "crossings": crossings})

    def predict(self, values: np.ndarray, dt: float) -> np.ndarray:
        """Return the persistence prediction for the next step."""
        return self.step(values, dt).prediction

    def crossings(self, values: np.ndarray) -> np.ndarray:
        """Return a boolean matrix of threshold crossings."""
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return np.abs(array) > self.threshold


class BaselineKind:
    """Registry of deterministic baseline constructors."""

    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    KALMAN = "kalman"
    THRESHOLD = "threshold"


def build_baseline(kind: str, n_channels: int, seed: int = 0) -> Any:
    """Construct a deterministic baseline by kind."""
    if kind == BaselineKind.MOVING_AVERAGE:
        return MovingAverageBaseline(n_channels, window=8, seed=seed)
    if kind == BaselineKind.EXPONENTIAL_SMOOTHING:
        return ExponentialSmoothingBaseline(n_channels, alpha=0.5, seed=seed)
    if kind == BaselineKind.KALMAN:
        return SimpleKalmanBaseline(n_channels, seed=seed)
    if kind == BaselineKind.THRESHOLD:
        return ThresholdDetectorBaseline(n_channels, seed=seed)
    raise ValueError(f"unknown baseline kind {kind!r}")
