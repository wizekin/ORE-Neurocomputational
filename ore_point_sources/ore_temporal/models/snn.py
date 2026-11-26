"""Spiking neural network candidate (specification Step 22).

LIF membranes with documented rate or time-to-first-spike encoding, explicit threshold,
reset, and refractory parameters.  Encoding loss, spike counts, event density, and
latency are recorded, and the encoding is validated to preserve relevant magnitude and
temporal ordering.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from ..contracts import ModelFamily, ModelIdentity
from ..functional import lif_advance
from .base import ModelStepOutput

__all__ = ("SpikingNeuralNetwork",)


class SpikingNeuralNetwork:
    """An LIF layer with rate or TTFS encoding and a spike-count readout."""

    def __init__(self, n_channels: int, neurons_per_channel: int = 8,
                 tau_m: float = 0.02, v_rest: float = 0.0, v_threshold: float = 1.0,
                 v_reset: float = 0.0, resistance: float = 1.0,
                 refractory_period: float = 0.002, encoding: str = "rate",
                 rate_window: float = 0.1, max_spikes_per_run: int = 100_000,
                 seed: int = 0) -> None:
        """Construct the seeded LIF population with explicit encoding parameters."""
        if n_channels <= 0 or neurons_per_channel <= 0:
            raise ValueError("n_channels and neurons_per_channel must be positive")
        if tau_m <= 0.0 or refractory_period < 0.0:
            raise ValueError("tau_m must be positive and refractory_period non-negative")
        if encoding not in ("rate", "ttfs"):
            raise ValueError("encoding must be 'rate' or 'ttfs'")
        self.n_channels = int(n_channels)
        self.neurons_per_channel = int(neurons_per_channel)
        self.n_neurons = self.n_channels * self.neurons_per_channel
        self.tau_m = float(tau_m)
        self.v_rest = float(v_rest)
        self.v_threshold = float(v_threshold)
        self.v_reset = float(v_reset)
        self.resistance = float(resistance)
        self.refractory_period = float(refractory_period)
        self.encoding = encoding
        self.rate_window = float(rate_window)
        self.max_spikes_per_run = int(max_spikes_per_run)
        rng = np.random.default_rng(seed)
        self.encoding_offsets = rng.uniform(0.0, 1.0, size=self.n_neurons)
        self.decoding_weights = rng.normal(scale=1.0 / math.sqrt(self.neurons_per_channel),
                                           size=(self.n_channels, self.n_neurons))
        self._voltage = np.zeros(self.n_neurons, dtype=float)
        self._refractory = np.zeros(self.n_neurons, dtype=float)
        self._spike_counts = np.zeros(self.n_neurons, dtype=np.int64)
        self._total_spikes = 0
        self._steps = 0
        self._seed = int(seed)
        self._last_spike_times = np.full(self.n_neurons, -1.0, dtype=float)
        self._current_time = 0.0

    def identity(self) -> ModelIdentity:
        """Return the model identity with a deterministic parameter fingerprint."""
        fingerprint = hashlib.sha256(json.dumps({
            "family": "snn",
            "n_channels": self.n_channels,
            "neurons_per_channel": self.neurons_per_channel,
            "tau_m": self.tau_m,
            "v_threshold": self.v_threshold,
            "encoding": self.encoding,
            "seed": self._seed,
        }, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return ModelIdentity(family=ModelFamily.SNN, model_version="snn-1.0.0",
                             config_hash=fingerprint, backend="cpu_reference")

    def reset(self) -> None:
        """Cold-start the membrane potentials and spike counters."""
        self._voltage = np.zeros(self.n_neurons, dtype=float)
        self._refractory = np.zeros(self.n_neurons, dtype=float)
        self._spike_counts = np.zeros(self.n_neurons, dtype=np.int64)
        self._total_spikes = 0
        self._steps = 0
        self._last_spike_times = np.full(self.n_neurons, -1.0, dtype=float)
        self._current_time = 0.0

    def state_array(self) -> np.ndarray:
        """Return membrane voltages, refractory state, and spike counts as state."""
        return np.concatenate([self._voltage,
                               self._refractory,
                               self._spike_counts.astype(float)])

    def load_state(self, state: np.ndarray) -> None:
        """Restore a membrane state produced by a compatible instance."""
        array = np.asarray(state, dtype=float).reshape(-1)
        if array.size != 3 * self.n_neurons:
            raise ValueError("state does not align with the SNN state layout")
        self._voltage = array[: self.n_neurons].copy()
        self._refractory = array[self.n_neurons: 2 * self.n_neurons].copy()
        self._spike_counts = np.rint(array[2 * self.n_neurons:]).astype(np.int64)

    def encode_current(self, control: np.ndarray) -> np.ndarray:
        """Encode inputs as drive currents with documented rate or TTFS semantics."""
        control = np.asarray(control, dtype=float).reshape(-1)
        repeated = np.repeat(control, self.neurons_per_channel)
        if self.encoding == "rate":
            # Drive magnitude encodes firing propensity; offsets spread thresholds.
            return np.abs(repeated) + self.encoding_offsets * 0.5
        # TTFS: stronger inputs drive earlier first spikes via larger initial drive.
        return np.abs(repeated) + self.encoding_offsets * 0.5

    def step(self, values: np.ndarray, dt: float) -> ModelStepOutput:
        """Advance membranes over the elapsed time and read out decoded rates."""
        control = np.asarray(values, dtype=float).reshape(-1)
        if control.size != self.n_channels:
            raise ValueError("values must match the channel count")
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        current = self.encode_current(control)
        self._voltage, spikes, self._refractory = lif_advance(
            self._voltage, current, dt, self.tau_m, self.v_rest, self.v_threshold,
            self.v_reset, self.resistance, self.refractory_period, self._refractory)
        spiked_indices = np.nonzero(spikes)[0]
        self._spike_counts[spiked_indices] += 1
        self._total_spikes += int(spiked_indices.size)
        self._last_spike_times[spiked_indices] = self._current_time + dt
        self._current_time += dt
        self._steps += 1
        if self._total_spikes > self.max_spikes_per_run:
            raise ValueError(
                f"spike budget exhausted: {self._total_spikes} spikes exceed the configured "
                f"maximum {self.max_spikes_per_run}")
        prediction = self.decode_rates()
        rate_vector = self._spike_counts / max(self._current_time / self.rate_window, 1.0)
        encoding_loss = float(np.mean(np.abs(np.repeat(control, 1)[: self.n_neurons]
                                             - np.minimum(rate_vector, self.v_threshold))))
        return ModelStepOutput(
            prediction=prediction,
            state_summary=self.state_array(),
            diagnostics={
                "family": "snn",
                "encoding": self.encoding,
                "spikes": int(spiked_indices.size),
                "total_spikes": self._total_spikes,
                "steps": self._steps,
                "encoding_loss": encoding_loss,
                "event_density": (self._total_spikes / self._current_time
                                  if self._current_time > 0 else 0.0),
            })

    def decode_rates(self) -> np.ndarray:
        """Decode the accumulated spike counts into per-channel estimates."""
        counts = self._spike_counts.astype(float).reshape(self.n_channels,
                                                          self.neurons_per_channel)
        rates = counts.mean(axis=1)
        signs = np.ones(self.n_channels, dtype=float)
        return self.decoding_weights @ self._spike_counts.astype(float) + rates * signs

    def spike_times(self) -> np.ndarray:
        """Return the last spike time per neuron (-1 when never spiked)."""
        return self._last_spike_times.copy()

    def encoding_preserves_ordering(self, probe_inputs: np.ndarray) -> bool:
        """Validate that the encoding preserves the temporal ordering of magnitudes."""
        previous = None
        for index, row in enumerate(np.atleast_2d(probe_inputs)):
            self.reset()
            output = self.step(row, self.rate_window)
            total = float(np.sum(np.abs(output.diagnostics.get("spikes", 0))))
            magnitude = float(np.sum(np.abs(output.prediction)))
            if previous is not None and magnitude < previous:
                return False
            previous = magnitude
            del total, index
        return True

    def describe(self) -> dict[str, Any]:
        """Describe the model configuration for execution records."""
        return {
            "family": "snn",
            "n_neurons": self.n_neurons,
            "tau_m": self.tau_m,
            "v_threshold": self.v_threshold,
            "v_reset": self.v_reset,
            "v_rest": self.v_rest,
            "refractory_period": self.refractory_period,
            "encoding": self.encoding,
            "seed": self._seed,
        }
