"""Deterministic LIF reference populations for top-down language comprehension.

The populations are coarse computational analogies, not an anatomical map or a validated
simulation of human sentence comprehension.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


POPULATIONS = (
    "sentence_context",
    "syntactic_prediction",
    "semantic_prediction",
    "world_knowledge",
    "lexical_input",
    "prediction_error",
    "conflict_monitor",
    "working_memory",
    "learning",
)

_CONNECTIONS = {
    "sentence_context": {"working_memory": 0.15, "syntactic_prediction": 0.10, "semantic_prediction": 0.12},
    "syntactic_prediction": {"sentence_context": 0.08, "working_memory": 0.12, "conflict_monitor": 0.08},
    "semantic_prediction": {"sentence_context": 0.08, "world_knowledge": 0.12, "prediction_error": 0.08},
    "world_knowledge": {"semantic_prediction": 0.10, "sentence_context": 0.08, "conflict_monitor": 0.05},
    "lexical_input": {"syntactic_prediction": 0.08, "semantic_prediction": 0.12, "prediction_error": 0.10},
    "prediction_error": {"working_memory": 0.08, "conflict_monitor": 0.15, "semantic_prediction": 0.05},
    "conflict_monitor": {"syntactic_prediction": 0.10, "working_memory": 0.14, "learning": 0.05},
    "working_memory": {"sentence_context": 0.12, "syntactic_prediction": 0.08, "semantic_prediction": 0.08},
    "learning": {"sentence_context": 0.08, "semantic_prediction": 0.06},
}


@dataclass(frozen=True, slots=True)
class LIFConfig:
    """Define voltage, current, time, and population parameters for the LIF reference model."""
    population_size: int = 12
    steps_per_operation: int = 2
    dt_ms: float = 1.0
    membrane_tau_ms: float = 20.0
    refractory_ms: float = 2.0
    rest_mv: float = -70.0
    reset_mv: float = -68.0
    threshold_mv: float = -50.0
    resistance_mohm: float = 100.0
    current_gain_pa: float = 300.0
    recurrent_gain_pa: float = 18.0
    inhibition_gain_pa: float = 20.0

    def __post_init__(self) -> None:
        """Ensure the reference neuron parameters are numerically safe."""
        if self.population_size < 2 or self.steps_per_operation < 1:
            raise ValueError("population_size must be >= 2 and steps_per_operation positive")
        for name in ("dt_ms", "membrane_tau_ms", "resistance_mohm", "current_gain_pa"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("refractory_ms", "recurrent_gain_pa", "inhibition_gain_pa"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.rest_mv <= self.reset_mv < self.threshold_mv:
            raise ValueError("require rest_mv <= reset_mv < threshold_mv")


class ComprehensionLIFSimulator:
    """Simulate coupled leaky-integrate-and-fire populations driven by each language operation."""

    def __init__(self, config: LIFConfig | None = None) -> None:
        """Allocate deterministic heterogeneous neuron populations and initialize resting state."""
        self.config = config or LIFConfig()
        count = self.config.population_size
        self.voltage_mv = {
            name: np.full(count, self.config.rest_mv, dtype=np.float64) for name in POPULATIONS
        }
        self.refractory_ms = {
            name: np.zeros(count, dtype=np.float64) for name in POPULATIONS
        }
        self.spike_counts = {
            name: np.zeros(count, dtype=np.int64) for name in POPULATIONS
        }
        self.activity = {name: 0.0 for name in POPULATIONS}
        self.last_current_pa = {
            name: np.zeros(count, dtype=np.float64) for name in POPULATIONS
        }
        self.thresholds_mv = {
            name: self.config.threshold_mv + np.linspace(-0.7, 0.7, count)
            for name in POPULATIONS
        }
        self.time_ms = 0.0
        self.operation_count = 0

    def _advance(self, drives: Mapping[str, float]) -> None:
        """Advance every population by one Euler step with recurrent excitation and global inhibition."""
        cfg = self.config
        global_activity = float(np.mean(tuple(self.activity.values())))
        next_activity: dict[str, float] = {}
        for name in POPULATIONS:
            afferent = sum(weight * self.activity[source] for source, weight in _CONNECTIONS[name].items())
            drive = float(np.clip(drives.get(name, 0.0), 0.0, 1.0))
            current_scalar = (
                cfg.current_gain_pa * drive
                + cfg.recurrent_gain_pa * (afferent + self.activity[name])
                - cfg.inhibition_gain_pa * global_activity
            )
            heterogeneity = np.linspace(0.90, 1.10, cfg.population_size)
            current = np.maximum(0.0, current_scalar * heterogeneity)
            self.last_current_pa[name] = current
            voltage = self.voltage_mv[name]
            refractory = self.refractory_ms[name]
            active = refractory <= 0.0
            voltage[~active] = cfg.reset_mv
            refractory[~active] = np.maximum(0.0, refractory[~active] - cfg.dt_ms)
            delta = (cfg.dt_ms / cfg.membrane_tau_ms) * (
                cfg.rest_mv - voltage + cfg.resistance_mohm * current * 0.001
            )
            voltage[active] += delta[active]
            spiked = active & (voltage >= self.thresholds_mv[name])
            if np.any(spiked):
                self.spike_counts[name][spiked] += 1
                voltage[spiked] = cfg.reset_mv
                refractory[spiked] = cfg.refractory_ms
            decay = math.exp(-cfg.dt_ms / max(cfg.membrane_tau_ms, cfg.dt_ms))
            instantaneous_rate = float(np.count_nonzero(spiked)) / cfg.population_size
            next_activity[name] = decay * self.activity[name] + (1.0 - decay) * instantaneous_rate
        self.activity.update(next_activity)
        self.time_ms += cfg.dt_ms

    def step(self, operation_id: str, drives: Mapping[str, float]) -> dict[str, object]:
        """Run the configured LIF cycles for one operation and return its population activity snapshot."""
        unknown = set(drives) - set(POPULATIONS)
        if unknown:
            raise KeyError(f"unknown comprehension populations: {sorted(unknown)}")
        if any(not math.isfinite(float(value)) for value in drives.values()):
            raise ValueError("neural drives must be finite")
        if any(not 0.0 <= float(value) <= 1.0 for value in drives.values()):
            raise ValueError("neural drives must be in [0, 1]")
        for _ in range(self.config.steps_per_operation):
            self._advance(drives)
        self.operation_count += 1
        return {
            "operation_id": operation_id,
            "simulated_time_ms": float(self.time_ms),
            "active_populations": [name for name, value in drives.items() if value > 0.0],
            "mean_network_activity": float(np.mean(tuple(self.activity.values()))),
            "population_activity": {name: float(value) for name, value in self.activity.items()},
            "total_spikes": int(sum(int(np.sum(values)) for values in self.spike_counts.values())),
        }

    def snapshot(self) -> dict[str, object]:
        """Summarize firing rates, voltage, current, and activity for every reference population."""
        elapsed_seconds = max(self.time_ms / 1000.0, 1e-9)
        layers: dict[str, object] = {}
        for name in POPULATIONS:
            layers[name] = {
                "mean_firing_rate_hz": float(np.mean(self.spike_counts[name]) / elapsed_seconds),
                "spike_count": int(np.sum(self.spike_counts[name])),
                "mean_voltage_mv": float(np.mean(self.voltage_mv[name])),
                "mean_current_pa": float(np.mean(self.last_current_pa[name])),
                "mean_activity": float(self.activity[name]),
            }
        finite = all(
            np.isfinite(values).all()
            for values in (self.voltage_mv[name] for name in POPULATIONS)
        )
        return {
            "enabled": True,
            "model": "coupled_current_based_LIF_language_reference",
            "simulated_time_ms": float(self.time_ms),
            "operation_count": int(self.operation_count),
            "layers": layers,
            "numerically_valid": bool(finite),
            "units": {"voltage": "mV", "current": "pA", "time": "ms", "firing_rate": "Hz"},
            "scientific_scope": (
                "Deterministic heterogeneous LIF populations are a coarse computational analogy. "
                "They are not anatomically fitted, calibrated to EEG, or a biological validation of language processing."
            ),
        }


def population_drives(
    operation_number: int,
    activation: float,
    *,
    prediction_error: float = 0.0,
    conflict: float = 0.0,
    working_memory_load: float = 0.0,
) -> dict[str, float]:
    """Map one functional operation's activation and error signals onto coarse neural populations."""
    if not 1 <= int(operation_number) <= 38:
        raise ValueError("operation_number must be between 1 and 38")
    clamp = lambda value: float(np.clip(float(value), 0.0, 1.0))
    if operation_number <= 3:
        target = "sentence_context"
    elif operation_number <= 8:
        target = "syntactic_prediction"
    elif operation_number <= 12:
        target = "semantic_prediction"
    elif operation_number <= 16:
        target = "world_knowledge"
    elif operation_number <= 20:
        target = "lexical_input"
    elif operation_number <= 24:
        target = "prediction_error"
    elif operation_number <= 28:
        target = "conflict_monitor"
    elif operation_number <= 37:
        target = "working_memory"
    else:
        target = "learning"
    return {
        target: clamp(activation),
        "prediction_error": clamp(prediction_error),
        "conflict_monitor": clamp(conflict),
        "working_memory": clamp(working_memory_load),
    }
