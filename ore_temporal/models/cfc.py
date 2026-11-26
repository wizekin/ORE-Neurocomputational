"""Closed-form Continuous-time (CfC) candidate (specification Step 19).

The CfC family passes the actual elapsed time dt into a closed-form exponential update
x(t+dt) = x_inf + (x - x_inf) * exp(-dt / tau), which is exact for a constant target over
the interval.  Compute time and numerical deviation versus a numerical reference are
recorded per step; one variant's guarantees are never generalized to every implementation.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from ..contracts import ModelFamily, ModelIdentity
from ..functional import euler_step, exponential_relaxation
from .base import ModelStepOutput

__all__ = ("ClosedFormContinuousNetwork",)


class ClosedFormContinuousNetwork:
    """A CfC-style network computing an input-dependent target with an exact relaxation."""

    def __init__(self, n_channels: int, n_state: int = 16, tau: float = 1.0,
                 reference_substeps: int = 64, seed: int = 0) -> None:
        """Construct seeded parameters and the numerical-reference configuration."""
        if n_channels <= 0 or n_state <= 0:
            raise ValueError("n_channels and n_state must be positive")
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        if reference_substeps <= 0:
            raise ValueError("reference_substeps must be positive")
        self.n_channels = int(n_channels)
        self.n_state = int(n_state)
        self.tau = float(tau)
        self.reference_substeps = int(reference_substeps)
        rng = np.random.default_rng(seed)
        self.weights_state = rng.normal(scale=1.0 / math.sqrt(n_state), size=(n_state, n_state))
        self.weights_input = rng.normal(scale=1.0 / math.sqrt(n_channels),
                                        size=(n_state, n_channels))
        self.bias = rng.normal(scale=0.1, size=n_state)
        self.readout = rng.normal(scale=1.0 / math.sqrt(n_state), size=(n_channels, n_state))
        self.readout_bias = rng.normal(scale=0.1, size=n_channels)
        self._state = np.zeros(n_state, dtype=float)
        self._seed = int(seed)
        self._steps = 0
        self._last_elapsed_dt: float | None = None

    def identity(self) -> ModelIdentity:
        """Return the model identity with a deterministic parameter fingerprint."""
        fingerprint = hashlib.sha256(json.dumps({
            "family": "cfc",
            "n_channels": self.n_channels,
            "n_state": self.n_state,
            "tau": self.tau,
            "seed": self._seed,
            "w_state": self.weights_state.tolist(),
            "w_input": self.weights_input.tolist(),
            "w_readout": self.readout.tolist(),
        }, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return ModelIdentity(family=ModelFamily.CFC, model_version="cfc-1.0.0",
                             config_hash=fingerprint, backend="cpu_reference")

    def reset(self) -> None:
        """Cold-start the hidden state."""
        self._state = np.zeros(self.n_state, dtype=float)
        self._steps = 0
        self._last_elapsed_dt = None

    def state_array(self) -> np.ndarray:
        """Return the hidden state."""
        return self._state.copy()

    def load_state(self, state: np.ndarray) -> None:
        """Restore a hidden state produced by a compatible instance."""
        array = np.asarray(state, dtype=float).reshape(-1)
        if array.size != self.n_state:
            raise ValueError("state does not align with the hidden dimension")
        self._state = array.copy()

    def target(self, control: np.ndarray) -> np.ndarray:
        """Compute the input-dependent relaxation target x_inf = tanh(W_x x + W_u u + b)."""
        return np.tanh(self.weights_state @ self._state
                       + self.weights_input @ control + self.bias)

    def closed_form_update(self, state: np.ndarray, target: np.ndarray, dt: float
                           ) -> np.ndarray:
        """Apply the exact relaxation update (Section 3.3)."""
        return exponential_relaxation(state, target, dt, self.tau)

    def numerical_reference(self, state: np.ndarray, target: np.ndarray, dt: float
                            ) -> np.ndarray:
        """Integrate the same relaxation ODE with small explicit Euler substeps."""
        reference = np.asarray(state, dtype=float).copy()
        if dt <= 0.0:
            return reference
        step = dt / self.reference_substeps
        for _ in range(self.reference_substeps):
            derivative = (target - reference) / self.tau
            reference = euler_step(reference, step, lambda _x: derivative)
        return reference

    def step(self, values: np.ndarray, dt: float) -> ModelStepOutput:
        """Update with the actual elapsed dt and record the numerical deviation."""
        control = np.asarray(values, dtype=float).reshape(-1)
        if control.size != self.n_channels:
            raise ValueError("values must match the channel count")
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        target = self.target(control)
        new_state = self.closed_form_update(self._state, target, dt)
        reference = self.numerical_reference(self._state, target, dt)
        deviation = float(np.max(np.abs(new_state - reference))) if new_state.size else 0.0
        self._state = new_state
        self._steps += 1
        self._last_elapsed_dt = float(dt)
        prediction = self.readout @ self._state + self.readout_bias
        return ModelStepOutput(
            prediction=prediction,
            state_summary=self._state.copy(),
            diagnostics={
                "family": "cfc",
                "elapsed_dt": float(dt),
                "numerical_reference_deviation": deviation,
                "compute_mode": "closed_form",
                "steps": self._steps,
            })

    @property
    def last_elapsed_dt(self) -> float | None:
        """Return the most recent elapsed time actually consumed by the update."""
        return self._last_elapsed_dt

    def describe(self) -> dict[str, Any]:
        """Describe the model configuration for execution records."""
        return {
            "family": "cfc",
            "n_state": self.n_state,
            "tau": self.tau,
            "reference_substeps": self.reference_substeps,
            "seed": self._seed,
        }
