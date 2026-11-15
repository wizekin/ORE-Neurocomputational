"""Deterministic LIF reference populations for syntax processing.

The named populations are coarse computational analogies. The model is not
anatomically complete, fitted, or evidence of a literal cortical algorithm.
"""
from __future__ import annotations

from typing import Mapping
import math
import numpy as np

from .contracts import SyntaxConfig

POPULATIONS = (
    "orthographic_input",
    "lexical_pos",
    "temporal_syntax",
    "frontal_control",
    "phrase_binding",
    "dependency_relations",
    "thematic_roles",
    "working_memory",
    "prediction_error",
)

_CONNECTIONS = {
    "orthographic_input": {"lexical_pos": 0.16, "temporal_syntax": 0.04},
    "lexical_pos": {"temporal_syntax": 0.16, "frontal_control": 0.06},
    "temporal_syntax": {"phrase_binding": 0.14, "dependency_relations": 0.18, "frontal_control": 0.08},
    "frontal_control": {"temporal_syntax": 0.10, "dependency_relations": 0.14, "working_memory": 0.15},
    "phrase_binding": {"temporal_syntax": 0.08, "dependency_relations": 0.10, "working_memory": 0.10},
    "dependency_relations": {"thematic_roles": 0.16, "working_memory": 0.14, "prediction_error": 0.06},
    "thematic_roles": {"frontal_control": 0.08, "dependency_relations": 0.06},
    "working_memory": {"frontal_control": 0.10, "dependency_relations": 0.10},
    "prediction_error": {"frontal_control": 0.12, "temporal_syntax": 0.04},
}


class SyntaxLIFSimulator:
    """Small coupled LIF network with explicit voltage/current units and diagnostics."""

    def __init__(self, config: SyntaxConfig) -> None:
        self.config = config
        n = config.population_size
        self.voltage_mv = {name: np.full(n, config.rest_mv, dtype=np.float64) for name in POPULATIONS}
        self.refractory_ms = {name: np.zeros(n, dtype=np.float64) for name in POPULATIONS}
        self.spike_counts = {name: np.zeros(n, dtype=np.int64) for name in POPULATIONS}
        self.activity = {name: 0.0 for name in POPULATIONS}
        self.last_current_pa = {name: np.zeros(n, dtype=np.float64) for name in POPULATIONS}
        self.time_ms = 0.0
        self.operation_count = 0

    def _advance(self, drives: Mapping[str, float]) -> None:
        cfg = self.config
        global_activity = float(np.mean(tuple(self.activity.values())))
        next_activity: dict[str, float] = {}
        for name in POPULATIONS:
            afferent = sum(weight * self.activity[source] for source, weight in _CONNECTIONS[name].items())
            current_scalar = (
                cfg.current_gain_pa * float(np.clip(drives.get(name, 0.0), 0.0, 1.0))
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
            dv = (cfg.dt_ms / cfg.membrane_tau_ms) * (
                cfg.rest_mv - voltage + cfg.membrane_resistance_mohm * current * 0.001
            )
            voltage[active] += dv[active]
            spiked = active & (voltage >= cfg.threshold_mv)
            if np.any(spiked):
                self.spike_counts[name][spiked] += 1
                voltage[spiked] = cfg.reset_mv
                refractory[spiked] = cfg.refractory_ms
            elapsed_decay = math.exp(-cfg.dt_ms / max(cfg.membrane_tau_ms, cfg.dt_ms))
            instant_rate = float(np.count_nonzero(spiked)) / cfg.population_size
            next_activity[name] = elapsed_decay * self.activity[name] + (1.0 - elapsed_decay) * instant_rate
        self.activity.update(next_activity)
        self.time_ms += cfg.dt_ms

    def step(
        self, operation_id: str, drives: Mapping[str, float], *, cycles: int | None = None
    ) -> dict[str, object]:
        unknown = set(drives) - set(POPULATIONS)
        if unknown:
            raise KeyError(f"unknown syntax populations: {sorted(unknown)}")
        if any(not math.isfinite(float(value)) for value in drives.values()):
            raise ValueError("neural drives must be finite")
        for _ in range(cycles or self.config.steps_per_operation):
            self._advance(drives)
        self.operation_count += 1
        return {
            "operation_id": operation_id,
            "simulated_time_ms": float(self.time_ms),
            "active_populations": [name for name, value in drives.items() if value > 0.0],
            "mean_network_activity": float(np.mean(tuple(self.activity.values()))),
        }

    def snapshot(self) -> dict[str, object]:
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
            "model": "current_based_LIF_syntax_reference",
            "simulated_time_ms": float(self.time_ms),
            "operation_count": self.operation_count,
            "layers": layers,
            "scope": (
                "Coarse deterministic computational populations; not a fitted anatomical model, "
                "not a cellular simulation of language, and not physiological validation."
            ),
            "units": {"voltage": "mV", "current": "pA", "time": "ms", "firing_rate": "Hz"},
        }
