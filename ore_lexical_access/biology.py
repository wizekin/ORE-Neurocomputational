"""Seven-population current-based LIF reference for lexical access.

This is a configurable computational reference, not an anatomically fitted or
empirically validated simulation of human reading or language processing.
"""
from __future__ import annotations

from typing import Mapping
import math
import numpy as np

from .contracts import LexicalAccessConfig

POPULATIONS = (
    "orthographic",
    "lexical_identity",
    "semantic",
    "grammatical_category",
    "morphological",
    "selection_control",
    "recurrent_candidate",
)


class LexicalAccessLIFSimulator:
    """Advance coupled rate/neuron populations while preserving stage diagnostics."""

    _CONNECTIONS = {
        "orthographic": {"lexical_identity": 0.20, "morphological": 0.10},
        "lexical_identity": {
            "orthographic": 0.07, "semantic": 0.16,
            "grammatical_category": 0.12, "recurrent_candidate": 0.18,
        },
        "semantic": {"lexical_identity": 0.12, "selection_control": 0.12, "recurrent_candidate": 0.08},
        "grammatical_category": {"lexical_identity": 0.08, "selection_control": 0.14},
        "morphological": {"lexical_identity": 0.15, "semantic": 0.12, "selection_control": 0.08},
        "selection_control": {"lexical_identity": 0.12, "recurrent_candidate": 0.18},
        "recurrent_candidate": {"lexical_identity": 0.12, "semantic": 0.06, "grammatical_category": 0.06},
    }

    def __init__(self, config: LexicalAccessConfig) -> None:
        self.config = config
        size = config.population_size
        self.voltage_mv = {name: np.full(size, config.rest_mv, dtype=float) for name in POPULATIONS}
        self.refractory_ms = {name: np.zeros(size, dtype=float) for name in POPULATIONS}
        self.adaptation = {name: np.zeros(size, dtype=float) for name in POPULATIONS}
        self.spike_counts = {name: np.zeros(size, dtype=np.int64) for name in POPULATIONS}
        self.activity = {name: 0.0 for name in POPULATIONS}
        self.last_current_pa = {name: np.zeros(size, dtype=float) for name in POPULATIONS}
        self.time_ms = 0.0
        self.operation_trace: list[dict[str, object]] = []

    def _advance(self, drives: Mapping[str, float]) -> None:
        cfg = self.config
        global_activity = float(np.mean(list(self.activity.values())))
        new_voltage: dict[str, np.ndarray] = {}
        new_refractory: dict[str, np.ndarray] = {}
        new_adaptation: dict[str, np.ndarray] = {}
        new_activity: dict[str, float] = {}
        for name in POPULATIONS:
            drive = float(np.clip(drives.get(name, 0.0), 0.0, 1.0))
            afferent = sum(weight * self.activity[source] for source, weight in self._CONNECTIONS[name].items())
            scalar_current = (
                cfg.current_gain_pa * drive
                + cfg.recurrent_gain_pa * (afferent + self.activity[name])
                - cfg.inhibition_gain_pa * global_activity
            )
            current = np.full(cfg.population_size, scalar_current, dtype=float)
            current -= cfg.adaptation_current_gain_pa * self.adaptation[name]
            self.last_current_pa[name] = current.copy()

            voltage = self.voltage_mv[name].copy()
            refractory = self.refractory_ms[name].copy()
            adaptation = self.adaptation[name].copy()
            active = refractory <= 0.0
            voltage[~active] = cfg.reset_mv
            refractory[~active] = np.maximum(0.0, refractory[~active] - cfg.dt_ms)
            dv = (cfg.dt_ms / cfg.membrane_tau_ms) * (
                cfg.rest_mv - voltage + cfg.membrane_resistance_mohm * current * 0.001
            )
            voltage[active] += dv[active]
            spiked = active & (voltage >= cfg.threshold_mv)
            self.spike_counts[name][spiked] += 1
            voltage[spiked] = cfg.reset_mv
            refractory[spiked] = cfg.refractory_ms
            adaptation *= math.exp(-cfg.dt_ms / max(cfg.membrane_tau_ms * 4.0, cfg.dt_ms))
            adaptation[spiked] += 1.0
            tau_rate = max(cfg.membrane_tau_ms, cfg.dt_ms)
            decay = math.exp(-cfg.dt_ms / tau_rate)
            spike_fraction = float(np.count_nonzero(spiked)) / cfg.population_size
            new_activity[name] = decay * self.activity[name] + (1.0 - decay) * spike_fraction
            new_voltage[name] = voltage
            new_refractory[name] = refractory
            new_adaptation[name] = adaptation

        self.voltage_mv.update(new_voltage)
        self.refractory_ms.update(new_refractory)
        self.adaptation.update(new_adaptation)
        self.activity.update(new_activity)
        self.time_ms += cfg.dt_ms

    def step(self, operation_id: str, drives: Mapping[str, float], *, cycles: int | None = None) -> dict[str, object]:
        unknown = set(drives) - set(POPULATIONS)
        if unknown:
            raise KeyError(f"unknown neural populations: {sorted(unknown)}")
        if any(not math.isfinite(float(value)) for value in drives.values()):
            raise ValueError("neural drives must be finite")
        for _ in range(cycles or self.config.steps_per_character):
            self._advance(drives)
        record = {
            "operation_id": operation_id,
            "status": "simulated",
            "simulated_time_ms": float(self.time_ms),
            "population_activity": dict(self.activity),
        }
        self.operation_trace.append(record)
        return record

    def snapshot(self) -> dict[str, object]:
        elapsed_s = max(self.time_ms / 1000.0, 1e-9)
        layers: dict[str, dict[str, object]] = {}
        for name in POPULATIONS:
            layers[name] = {
                "mean_voltage_mv": float(np.mean(self.voltage_mv[name])),
                "final_voltage_mv": self.voltage_mv[name].tolist(),
                "mean_current_pa": float(np.mean(self.last_current_pa[name])),
                "spike_count": int(np.sum(self.spike_counts[name])),
                "mean_firing_rate_hz": float(
                    np.sum(self.spike_counts[name]) / (self.config.population_size * elapsed_s)
                ),
                "activity_fraction": float(self.activity[name]),
                "mean_adaptation": float(np.mean(self.adaptation[name])),
            }
        finite = all(
            np.isfinite(array).all()
            for group in (self.voltage_mv, self.last_current_pa, self.adaptation)
            for array in group.values()
        )
        return {
            "enabled": True,
            "model": "generic_recurrent_current_based_LIF_reference",
            "layers": layers,
            "layer_order": list(POPULATIONS),
            "simulated_time_ms": float(self.time_ms),
            "numerically_valid": bool(finite),
            "operation_trace": list(self.operation_trace),
            "units": {
                "membrane_voltage": "mV", "synaptic_current": "pA",
                "firing_rate": "Hz", "simulated_time": "ms", "spike_count": "count",
            },
            "scope": (
                "Seven coupled generic LIF reference populations with recurrent excitation, "
                "global inhibition, refractory reset and adaptation. Not anatomically fitted, "
                "not calibrated to human recordings, and not evidence of biological validity."
            ),
        }
