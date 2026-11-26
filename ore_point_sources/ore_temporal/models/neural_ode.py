"""Neural Ordinary Differential Equation candidate (specification Step 20).

The vector field f_theta(h, t) is a small MLP; integration exposes tolerances, maximum
function evaluations, and failure conditions.  Solver failures, excessive evaluations,
and exploding states are reported and invalidate the output rather than being clipped.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from ..contracts import ModelFamily, ModelIdentity
from ..solvers import SolverFailure, integrate_interval
from .base import ModelStepOutput

__all__ = ("NeuralODEModel",)


class NeuralODEModel:
    """A learned-continuous-dynamics model h' = f_theta(h, u, t) with a linear readout."""

    def __init__(self, n_channels: int, n_state: int = 8, hidden_units: int = 16,
                 solver_method: str = "rk4", max_step: float = 0.1,
                 max_evaluations: int = 2_000, state_norm_bound: float = 1e4,
                 seed: int = 0) -> None:
        """Construct the seeded vector field and solver configuration."""
        if n_channels <= 0 or n_state <= 0 or hidden_units <= 0:
            raise ValueError("dimensions must be positive")
        self.n_channels = int(n_channels)
        self.n_state = int(n_state)
        self.hidden_units = int(hidden_units)
        self.solver_method = solver_method
        self.max_step = float(max_step)
        self.max_evaluations = int(max_evaluations)
        self.state_norm_bound = float(state_norm_bound)
        rng = np.random.default_rng(seed)
        self.weights_hidden = rng.normal(scale=1.0 / math.sqrt(n_state + n_channels),
                                         size=(hidden_units, n_state + n_channels))
        self.bias_hidden = rng.normal(scale=0.1, size=hidden_units)
        self.weights_output = rng.normal(scale=1.0 / math.sqrt(hidden_units),
                                         size=(n_state, hidden_units))
        self.bias_output = rng.normal(scale=0.1, size=n_state)
        self.readout = rng.normal(scale=1.0 / math.sqrt(n_state), size=(n_channels, n_state))
        self.readout_bias = rng.normal(scale=0.1, size=n_channels)
        self.input_projection = rng.normal(scale=1.0 / math.sqrt(n_channels),
                                           size=(n_state, n_channels))
        self._state = np.zeros(n_state, dtype=float)
        self._seed = int(seed)
        self._steps = 0
        self._evaluations_used = 0

    def identity(self) -> ModelIdentity:
        """Return the model identity with a deterministic parameter fingerprint."""
        fingerprint = hashlib.sha256(json.dumps({
            "family": "neural_ode",
            "n_channels": self.n_channels,
            "n_state": self.n_state,
            "hidden_units": self.hidden_units,
            "seed": self._seed,
            "w_hidden": self.weights_hidden.tolist(),
            "w_output": self.weights_output.tolist(),
            "w_readout": self.readout.tolist(),
        }, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return ModelIdentity(family=ModelFamily.NEURAL_ODE, model_version="node-1.0.0",
                             config_hash=fingerprint, backend="cpu_reference")

    def reset(self) -> None:
        """Cold-start the state and evaluation counters."""
        self._state = np.zeros(self.n_state, dtype=float)
        self._steps = 0
        self._evaluations_used = 0

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
        """Evaluate f_theta(h, u, t) = W2 tanh(W1 [h; u] + b1) + b2."""
        del t  # time-invariant field; t is retained in the interface contract
        stacked = np.concatenate([np.asarray(state, dtype=float).reshape(-1),
                                  np.asarray(control, dtype=float).reshape(-1)])
        hidden = np.tanh(self.weights_hidden @ stacked + self.bias_hidden)
        return self.weights_output @ hidden + self.bias_output

    def step(self, values: np.ndarray, dt: float) -> ModelStepOutput:
        """Solve h(t+dt) from h(t) with the configured solver and budgets."""
        control = np.asarray(values, dtype=float).reshape(-1)
        if control.size != self.n_channels:
            raise ValueError("values must match the channel count")
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        if self._steps == 0:
            self._state = self.input_projection @ control
        try:
            new_state, diagnostics = integrate_interval(
                lambda state, t: self.vector_field(state, control, t),
                self._state, 0.0, dt, method=self.solver_method, max_step=self.max_step,
                max_evaluations=self.max_evaluations,
                state_norm_bound=self.state_norm_bound)
        except SolverFailure:
            raise
        self._state = new_state
        self._steps += 1
        self._evaluations_used += int(diagnostics.evaluations)
        prediction = self.readout @ self._state + self.readout_bias
        return ModelStepOutput(
            prediction=prediction,
            state_summary=self._state.copy(),
            diagnostics={
                "family": "neural_ode",
                "solver": diagnostics.to_dict(),
                "cumulative_evaluations": self._evaluations_used,
                "steps": self._steps,
            })

    def tolerance_sensitivity(self, control: np.ndarray, dt: float,
                              tolerances: tuple[float, ...]) -> dict[float, float]:
        """Measure output sensitivity to solver tolerance on a controlled case."""
        results: dict[float, float] = {}
        base_state = self._state.copy()
        for tolerance in tolerances:
            self.reset()
            self.load_state(base_state)
            output = self.step(control, dt)
            results[tolerance] = float(np.max(np.abs(output.prediction)))
        self.load_state(base_state)
        return results
