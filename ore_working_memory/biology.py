"""Coupled current-based LIF reference for working-memory and control populations.

This is a configurable computational analogy, not an anatomical model, a fitted human network,
or evidence that the brain literally executes the software operation registry.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .contracts import WorkingMemoryConfig


POPULATIONS = (
    "phonological_loop", "lexical_maintenance", "syntactic_workspace", "semantic_workspace",
    "parietal_attention", "prefrontal_control", "conflict_monitor", "task_set_network",
)

_STAGE_POPULATION = {
    "S1": "phonological_loop",
    "S2": "syntactic_workspace",
    "S3": "prefrontal_control",
    "S4": "semantic_workspace",
    "S5": "task_set_network",
}

_CONNECTIONS = {
    "phonological_loop": {
        "lexical_maintenance": 0.10, "parietal_attention": 0.06, "prefrontal_control": 0.05,
    },
    "lexical_maintenance": {
        "phonological_loop": 0.08, "semantic_workspace": 0.12, "syntactic_workspace": 0.08,
    },
    "syntactic_workspace": {
        "lexical_maintenance": 0.10, "prefrontal_control": 0.13, "semantic_workspace": 0.08,
    },
    "semantic_workspace": {
        "syntactic_workspace": 0.10, "prefrontal_control": 0.12, "conflict_monitor": 0.08,
    },
    "parietal_attention": {
        "phonological_loop": 0.08, "syntactic_workspace": 0.10, "semantic_workspace": 0.08,
    },
    "prefrontal_control": {
        "parietal_attention": 0.12, "conflict_monitor": 0.12, "task_set_network": 0.12,
        "syntactic_workspace": 0.08,
    },
    "conflict_monitor": {
        "prefrontal_control": 0.14, "semantic_workspace": 0.08, "task_set_network": 0.06,
    },
    "task_set_network": {
        "prefrontal_control": 0.12, "parietal_attention": 0.10, "phonological_loop": 0.05,
    },
}


class WorkingMemoryLIFSimulator:
    """Advance coupled neural-reference populations once for each registered operation.

    The voltage update is a forward-Euler current-based leaky integrate-and-fire approximation:
    dV = dt/tau_m * (V_rest - V + 1e-3 * R_m * I[pA]). The populations are functional labels,
    not claims of one-to-one anatomical localization.
    """

    def __init__(self, config: WorkingMemoryConfig) -> None:
        """Allocate voltage, refractory, spike, activity, and current state for each population."""
        self.config = config
        size = config.population_size
        self.voltage_mv = {
            name: np.full(size, config.rest_mv, dtype=np.float64) for name in POPULATIONS
        }
        self.refractory_ms = {
            name: np.zeros(size, dtype=np.float64) for name in POPULATIONS
        }
        self.spike_counts = {
            name: np.zeros(size, dtype=np.int64) for name in POPULATIONS
        }
        self.activity = {name: 0.0 for name in POPULATIONS}
        self.last_current_pa = {
            name: np.zeros(size, dtype=np.float64) for name in POPULATIONS
        }
        self.time_ms = 0.0
        self.operation_count = 0

    def _advance(self, drives: Mapping[str, float]) -> None:
        """Integrate one population-network step using recurrent excitation and global inhibition."""
        cfg = self.config
        global_activity = float(np.mean(tuple(self.activity.values())))
        next_activity: dict[str, float] = {}
        for name in POPULATIONS:
            afferent = sum(
                weight * self.activity[source]
                for source, weight in _CONNECTIONS[name].items()
            )
            external = float(np.clip(drives.get(name, 0.0), 0.0, 1.0))
            current_scalar = (
                cfg.current_gain_pa * external
                + cfg.recurrent_gain_pa * (afferent + self.activity[name])
                - cfg.inhibition_gain_pa * global_activity
            )
            current = np.full(cfg.population_size, current_scalar, dtype=np.float64)
            self.last_current_pa[name] = current
            voltage = self.voltage_mv[name]
            refractory = self.refractory_ms[name]
            active = refractory <= 0.0
            voltage[~active] = cfg.reset_mv
            refractory[~active] = np.maximum(0.0, refractory[~active] - cfg.dt_ms)
            voltage[active] += (
                (cfg.dt_ms / cfg.membrane_tau_ms)
                * (
                    cfg.rest_mv - voltage[active]
                    + 0.001 * cfg.membrane_resistance_mohm * current[active]
                )
            )
            spiked = active & (voltage >= cfg.threshold_mv)
            if np.any(spiked):
                self.spike_counts[name][spiked] += 1
                voltage[spiked] = cfg.reset_mv
                refractory[spiked] = cfg.refractory_ms
            decay = math.exp(-cfg.dt_ms / max(cfg.membrane_tau_ms, cfg.dt_ms))
            instant_rate = float(np.count_nonzero(spiked)) / cfg.population_size
            next_activity[name] = decay * self.activity[name] + (1.0 - decay) * instant_rate
        self.activity.update(next_activity)
        self.time_ms += cfg.dt_ms

    def step(self, operation_id: str, stage_id: str, signal: float) -> dict[str, object]:
        """Advance the stage-linked population drive and return one operation-level neural trace."""
        if stage_id not in _STAGE_POPULATION:
            raise KeyError(f"unknown working-memory stage: {stage_id}")
        if not math.isfinite(float(signal)):
            raise ValueError("neural drive signal must be finite")
        target = _STAGE_POPULATION[stage_id]
        bounded_signal = float(np.clip(signal, 0.0, 1.0))
        for _ in range(self.config.steps_per_operation):
            self._advance({target: bounded_signal})
        self.operation_count += 1
        return {
            "operation_id": operation_id, "stage_id": stage_id,
            "active_population": target, "drive": bounded_signal,
            "simulated_time_ms": float(self.time_ms),
            "network_activity": float(np.mean(tuple(self.activity.values()))),
        }

    def snapshot(self) -> dict[str, object]:
        """Return rates, spike counts, mean voltages, currents, and explicit model limitations."""
        elapsed_s = max(self.time_ms / 1000.0, 1e-9)
        layers: dict[str, object] = {}
        for name in POPULATIONS:
            layers[name] = {
                "mean_firing_rate_hz": float(np.mean(self.spike_counts[name]) / elapsed_s),
                "spike_count": int(np.sum(self.spike_counts[name])),
                "mean_voltage_mv": float(np.mean(self.voltage_mv[name])),
                "mean_current_pa": float(np.mean(self.last_current_pa[name])),
                "mean_activity": float(self.activity[name]),
            }
        return {
            "model": "current_based_LIF_working_memory_reference",
            "simulated_time_ms": float(self.time_ms),
            "operation_count": self.operation_count,
            "layers": layers,
            "units": {
                "voltage": "mV", "current": "pA", "time": "ms",
                "firing_rate": "Hz", "spike_count": "count",
            },
            "scope": (
                "Deterministic population-level software analogy. It is not an anatomically fitted "
                "network, not a cellular model of language, and not physiological validation."
            ),
        }
