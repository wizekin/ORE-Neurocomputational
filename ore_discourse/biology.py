"""Current-based LIF population reference for the eight discourse stages.

The simulator is a configurable computational analogy. It is not an anatomical
model, a fitted model of human discourse, or evidence of literal brain equations.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .contracts import DiscourseConfig


POPULATIONS = (
    "discourse_memory",
    "proposition_binding",
    "speaker_model",
    "pragmatic_control",
    "incongruity_monitor",
    "situation_model",
    "temporal_order",
    "narrative_coherence",
    "prediction_error",
)

_STAGE_POPULATION = {
    "S1": "discourse_memory",
    "S2": "proposition_binding",
    "S3": "speaker_model",
    "S4": "pragmatic_control",
    "S5": "incongruity_monitor",
    "S6": "situation_model",
    "S7": "temporal_order",
    "S8": "narrative_coherence",
}

_CONNECTIONS = {
    "discourse_memory": {"proposition_binding": 0.10, "speaker_model": 0.06, "narrative_coherence": 0.08},
    "proposition_binding": {"discourse_memory": 0.08, "situation_model": 0.14, "narrative_coherence": 0.12},
    "speaker_model": {"pragmatic_control": 0.16, "discourse_memory": 0.08, "narrative_coherence": 0.06},
    "pragmatic_control": {"speaker_model": 0.10, "incongruity_monitor": 0.12, "situation_model": 0.08},
    "incongruity_monitor": {"pragmatic_control": 0.12, "proposition_binding": 0.06, "prediction_error": 0.16},
    "situation_model": {"temporal_order": 0.14, "narrative_coherence": 0.14, "discourse_memory": 0.08},
    "temporal_order": {"situation_model": 0.10, "narrative_coherence": 0.16, "prediction_error": 0.08},
    "narrative_coherence": {"discourse_memory": 0.08, "proposition_binding": 0.10, "prediction_error": 0.10},
    "prediction_error": {"pragmatic_control": 0.10, "narrative_coherence": 0.10},
}


class DiscourseLIFSimulator:
    """Advance coupled population activity once for each registered discourse operation.

    The membrane equation is a forward-Euler approximation:
    dV = dt/tau_m * (V_rest - V + 1e-3 * R_m * I[pA]).
    A threshold crossing increments a spike count, resets voltage and starts
    a refractory interval. Units are reported as mV, pA, ms and Hz.
    """

    def __init__(self, config: DiscourseConfig) -> None:
        """Allocate neural-reference state for all discourse-related populations."""
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
        """Integrate one coupled population time step with bounded recurrent currents."""
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
        """Advance neural reference populations for one functional operation."""
        if stage_id not in _STAGE_POPULATION:
            raise KeyError(f"unknown discourse stage: {stage_id}")
        if not math.isfinite(float(signal)):
            raise ValueError("neural signal must be finite")
        target = _STAGE_POPULATION[stage_id]
        drives = {target: float(np.clip(signal, 0.0, 1.0))}
        for _ in range(self.config.steps_per_operation):
            self._advance(drives)
        self.operation_count += 1
        return {
            "operation_id": operation_id,
            "stage_id": stage_id,
            "simulated_time_ms": float(self.time_ms),
            "active_population": target,
            "signal": float(np.clip(signal, 0.0, 1.0)),
            "network_activity": float(np.mean(tuple(self.activity.values()))),
        }

    def snapshot(self) -> dict[str, object]:
        """Return final population rates, voltages, currents, activity and model scope."""
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
            "model": "current_based_LIF_discourse_reference",
            "simulated_time_ms": float(self.time_ms),
            "operation_count": self.operation_count,
            "layers": layers,
            "scope": (
                "Deterministic coupled population analogy; not an anatomically fitted "
                "network, not a cellular simulation of language, and not physiological validation."
            ),
            "units": {
                "voltage": "mV", "current": "pA", "time": "ms",
                "firing_rate": "Hz", "spike_count": "count",
            },
        }
