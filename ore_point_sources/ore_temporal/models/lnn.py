"""Liquid Neural Network candidate (specification Step 18).

Continuously evolving hidden state dx/dt = f_theta(x, u, t) integrated between
observations with a bounded solver.  Used when a continuously evolving hidden state
is meaningful.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from ..contracts import ModelFamily, ModelIdentity
from ..solvers import SolverFailure, integrate_interval
from .base import ModelStepOutput

__all__ = ("LiquidNeuralNetwork",)


class LiquidNeuralNetwork:
    """A continuous-time latent-state model with tanh liquid dynamics and a linear readout."""

    def __init__(self, n_channels: int, n_state: int = 16, tau: float = 1.0,
                 solver_method: str = "euler", max_step: float = 0.1,
                 max_evaluations: int = 10_000, state_norm_bound: float = 1e6,
                 seed: int = 0) -> None:
        """Construct seeded dynamics; identical seeds yield identical behaviour."""
        if n_channels <= 0 or n_state <= 0:
            raise ValueError("n_channels and n_state must be positive")
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        self.n_channels = int(n_channels)
        self.n_state = int(n_state)
        self.tau = float(tau)
        self.solver_method = solver_method
        self.max_step = float(max_step)
        self.max_evaluations = int(max_evaluations)
        self.state_norm_bound = float(state_norm_bound)
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
        self.last_diagnostics: Mapping[str, Any] = {}

    def identity(self) -> ModelIdentity:
        """Return the model identity with a deterministic parameter fingerprint."""
        import hashlib
        import json

        fingerprint = hashlib.sha256(json.dumps({
            "family": "lnn",
            "n_channels": self.n_channels,
            "n_state": self.n_state,
            "tau": self.tau,
            "seed": self._seed,
            "w_state": self.weights_state.tolist(),
            "w_input": self.weights_input.tolist(),
            "w_readout": self.readout.tolist(),
        }, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return ModelIdentity(family=ModelFamily.LNN, model_version="lnn-1.0.0",
                             config_hash=fingerprint, backend="cpu_reference")

    def reset(self) -> None:
        """Cold-start the latent state."""
        self._state = np.zeros(self.n_state, dtype=float)
        self._steps = 0

    def state_array(self) -> np.ndarray:
        """Return the latent state."""
        return self._state.copy()

    def load_state(self, state: np.ndarray) -> None:
        """Restore a latent state produced by a compatible instance."""
        array = np.asarray(state, dtype=float).reshape(-1)
        if array.size != self.n_state:
            raise ValueError("state does not align with the latent dimension")
        self._state = array.copy()

    def vector_field(self, state: np.ndarray, control: np.ndarray, t: float) -> np.ndarray:
        """Evaluate f_theta(x, u, t) = (tanh(W_x x + W_u u + b) - x) / tau."""
        del t  # the liquid dynamics are time-invariant; t is kept in the signature
        pre = (self.weights_state @ state + self.weights_input @ control + self.bias)
        return (np.tanh(pre) - state) / self.tau

    def step(self, values: np.ndarray, dt: float) -> ModelStepOutput:
        """Integrate the latent state over the actual elapsed time, then read out."""
        control = np.asarray(values, dtype=float).reshape(-1)
        if control.size != self.n_channels:
            raise ValueError("values must match the channel count")
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        try:
            new_state, diagnostics = integrate_interval(
                lambda state, t: self.vector_field(state, control, t),
                self._state, 0.0, dt, method=self.solver_method, max_step=self.max_step,
                max_evaluations=self.max_evaluations,
                state_norm_bound=self.state_norm_bound)
        except SolverFailure as failure:
            self.last_diagnostics = {"failure": str(failure), "steps": self._steps}
            raise
        self._state = new_state
        prediction = self.readout @ self._state + self.readout_bias
        self._steps += 1
        output_diagnostics = {"family": "lnn", "solver": diagnostics.to_dict(),
                              "state_norm": float(np.linalg.norm(self._state)),
                              "steps": self._steps}
        self.last_diagnostics = output_diagnostics
        return ModelStepOutput(prediction=prediction,
                               state_summary=self._state.copy(),
                               diagnostics=output_diagnostics)
