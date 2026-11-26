"""Selective state-space model candidate (specification Step 21).

Processes ordered observations through recurrent state updates h_t = A_t h_{t-1} + B_t x_t
with input-dependent (selective) decay and input gates, compact temporal memory, and
bounded state growth.  Memory is retained only for the permitted stream or session.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from ..contracts import ModelFamily, ModelIdentity
from .base import ModelStepOutput

__all__ = ("SelectiveStateSpaceModel",)


class SelectiveStateSpaceModel:
    """A compact selective SSM with input-dependent decay and bounded memory."""

    def __init__(self, n_channels: int, n_state: int = 8, base_decay: float = 0.9,
                 state_norm_bound: float = 1e3, max_sequence_length: int = 10_000,
                 seed: int = 0) -> None:
        """Construct the seeded selective gates and documented bounds."""
        if n_channels <= 0 or n_state <= 0:
            raise ValueError("n_channels and n_state must be positive")
        if not 0.0 < base_decay < 1.0:
            raise ValueError("base_decay must be in (0, 1)")
        self.n_channels = int(n_channels)
        self.n_state = int(n_state)
        self.base_decay = float(base_decay)
        self.state_norm_bound = float(state_norm_bound)
        self.max_sequence_length = int(max_sequence_length)
        rng = np.random.default_rng(seed)
        self.input_weights = rng.normal(scale=1.0 / math.sqrt(n_channels),
                                        size=(n_state, n_channels))
        self.gate_weights = rng.normal(scale=1.0 / math.sqrt(n_channels),
                                       size=(n_state, n_channels))
        self.select_weights = rng.normal(scale=1.0 / math.sqrt(n_channels),
                                         size=(n_state, n_channels))
        self.readout = rng.normal(scale=1.0 / math.sqrt(n_state), size=(n_channels, n_state))
        self.skip = rng.normal(scale=0.5, size=(n_channels, n_channels))
        self._state = np.zeros(n_state, dtype=float)
        self._seed = int(seed)
        self._steps = 0
        self._truncated_steps = 0

    def identity(self) -> ModelIdentity:
        """Return the model identity with a deterministic parameter fingerprint."""
        fingerprint = hashlib.sha256(json.dumps({
            "family": "ssm",
            "n_channels": self.n_channels,
            "n_state": self.n_state,
            "base_decay": self.base_decay,
            "seed": self._seed,
            "w_input": self.input_weights.tolist(),
            "w_readout": self.readout.tolist(),
        }, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return ModelIdentity(family=ModelFamily.SSM, model_version="ssm-1.0.0",
                             config_hash=fingerprint, backend="cpu_reference")

    def reset(self) -> None:
        """Cold-start the recurrent memory."""
        self._state = np.zeros(self.n_state, dtype=float)
        self._steps = 0
        self._truncated_steps = 0

    def state_array(self) -> np.ndarray:
        """Return the recurrent memory."""
        return self._state.copy()

    def load_state(self, state: np.ndarray) -> None:
        """Restore a recurrent memory produced by a compatible instance."""
        array = np.asarray(state, dtype=float).reshape(-1)
        if array.size != self.n_state:
            raise ValueError("state does not align with the memory dimension")
        self._state = array.copy()

    def selective_decay(self, control: np.ndarray) -> np.ndarray:
        """Compute the input-dependent decay A_t (Section 3.4, selective parameters)."""
        selectivity = 1.0 / (1.0 + np.exp(-(self.select_weights @ control)))
        return np.power(self.base_decay, selectivity)

    def input_gate(self, control: np.ndarray) -> np.ndarray:
        """Compute the input-dependent write gate B_t."""
        return 1.0 / (1.0 + np.exp(-(self.gate_weights @ control)))

    def step(self, values: np.ndarray, dt: float) -> ModelStepOutput:
        """Apply one selective state update and bound the memory norm."""
        control = np.asarray(values, dtype=float).reshape(-1)
        if control.size != self.n_channels:
            raise ValueError("values must match the channel count")
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        decay = self.selective_decay(control)
        gate = self.input_gate(control)
        injected = self.input_weights @ control
        # Scale decay with elapsed time so irregular sampling modulates memory.
        time_scaled_decay = np.power(decay, min(dt, 1.0)) if dt > 0.0 else np.ones_like(decay)
        self._state = time_scaled_decay * self._state + gate * injected
        norm = float(np.linalg.norm(self._state))
        truncated = False
        if norm > self.state_norm_bound:
            self._state = self._state * (self.state_norm_bound / norm)
            truncated = True
            self._truncated_steps += 1
        self._steps += 1
        prediction = self.readout @ self._state + self.skip @ control
        diagnostics: dict[str, Any] = {
            "family": "ssm",
            "steps": self._steps,
            "state_norm": float(np.linalg.norm(self._state)),
            "memory_truncated": truncated,
        }
        if self._steps >= self.max_sequence_length:
            diagnostics["warning"] = (
                f"sequence length {self._steps} reached the documented maximum "
                f"{self.max_sequence_length}; memory bounded")
        return ModelStepOutput(prediction=prediction,
                               state_summary=self._state.copy(),
                               diagnostics=diagnostics)

    @property
    def truncated_steps(self) -> int:
        """Return the number of steps whose memory norm was bounded."""
        return self._truncated_steps

    def memory_bytes(self) -> int:
        """Return the byte size of the retained recurrent memory."""
        return int(self._state.nbytes)
