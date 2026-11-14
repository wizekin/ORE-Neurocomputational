"""Current-based leaky integrate-and-fire reference populations.

This is a transparent, rate-level computational analogue. It is not a fitted
model of ATL, IFG, hippocampus, or a complete biological reading circuit.
"""
from __future__ import annotations

from typing import Mapping
import math
import numpy as np

from .contracts import SemanticConfig


POPULATIONS = (
    "lexical",
    "episodic_wordform",
    "visual_spoke",
    "auditory_spoke",
    "action_spoke",
    "somatosensory_spoke",
    "chemosensory_interoceptive_spoke",
    "affective_spoke",
    "spatiotemporal_spoke",
    "distributed_features",
    "atl_hub",
    "ifg_control",
    "abstract_prototype",
)

POPULATION_LABELS = {
    "lexical": "lexical access population",
    "episodic_wordform": "episodic word-form population",
    "visual_spoke": "visual semantic spoke population",
    "auditory_spoke": "auditory/phonological semantic spoke population",
    "action_spoke": "action/motor semantic spoke population",
    "somatosensory_spoke": "somatosensory semantic spoke population",
    "chemosensory_interoceptive_spoke": "olfactory/gustatory/interoceptive spoke population",
    "affective_spoke": "affective semantic spoke population",
    "spatiotemporal_spoke": "spatial/temporal semantic spoke population",
    "distributed_features": "distributed semantic feature population",
    "atl_hub": "anterior temporal transmodal hub population",
    "ifg_control": "inferior frontal semantic-control population",
    "abstract_prototype": "abstract lexical prototype population",
}


class SemanticLIFSimulator:
    """Small deterministic LIF population network with explicit physical units."""

    def __init__(self, config: SemanticConfig, seed: int = 17) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        n = config.population_size
        self.voltage_mv = {
            name: np.full(n, config.rest_mv, dtype=np.float64) for name in POPULATIONS
        }
        self.refractory_ms = {
            name: np.zeros(n, dtype=np.float64) for name in POPULATIONS
        }
        self.spike_counts = {
            name: np.zeros(n, dtype=np.int64) for name in POPULATIONS
        }
        self.activity = {name: 0.0 for name in POPULATIONS}
        self.last_current_pa = {name: np.zeros(n, dtype=np.float64) for name in POPULATIONS}
        self.time_ms = 0.0
        self.operation_count = 0

    def step(self, drives: Mapping[str, float], operation_id: str) -> dict[str, object]:
        """Advance the network for the configured substeps and return a snapshot."""
        for name in drives:
            if name not in self.voltage_mv:
                raise KeyError(f"unknown semantic population: {name}")
        clean = {
            name: float(np.clip(float(drives.get(name, 0.0)), 0.0, 1.0))
            for name in POPULATIONS
        }
        for _ in range(self.config.steps_per_operation):
            global_activity = float(np.mean(tuple(self.activity.values())))
            next_activity: dict[str, float] = {}
            for name in POPULATIONS:
                v = self.voltage_mv[name]
                ref = self.refractory_ms[name]
                local_drive = clean[name]
                recurrent = self.config.recurrent_gain_pa * self.activity[name]
                inhibitory = self.config.inhibition_gain_pa * global_activity
                current = (
                    self.config.current_gain_pa * local_drive
                    + recurrent
                    - inhibitory
                )
                self.last_current_pa[name] = np.full_like(v, current)
                active = ref <= 0.0
                dv = (self.config.dt_ms / self.config.membrane_tau_ms) * (
                    self.config.rest_mv
                    - v
                    + self.config.membrane_resistance_mohm * current * 0.001
                )
                v[active] += dv[active]
                ref[:] = np.maximum(0.0, ref - self.config.dt_ms)
                spiked = active & (v >= self.config.threshold_mv)
                if np.any(spiked):
                    self.spike_counts[name][spiked] += 1
                    v[spiked] = self.config.reset_mv
                    ref[spiked] = self.config.refractory_ms
                v[~active] = self.config.reset_mv
                instant_fraction = float(np.count_nonzero(spiked)) / max(len(v), 1)
                decay = math.exp(-self.config.dt_ms / 20.0)
                next_activity[name] = decay * self.activity[name] + (1.0 - decay) * instant_fraction
            self.activity.update(next_activity)
            self.time_ms += self.config.dt_ms

        self.operation_count += 1
        snapshot = self.snapshot()
        active_names = [
            name for name, values in clean.items()
            if values > 0.0
        ]
        return {
            "operation_id": operation_id,
            "simulated_time_ms": self.time_ms,
            "active_populations": active_names,
            "mean_network_activity": float(np.mean(tuple(self.activity.values()))),
            "population_metrics": {
                name: {
                    "mean_firing_rate_hz": snapshot["layers"][name]["mean_firing_rate_hz"],
                    "mean_voltage_mv": snapshot["layers"][name]["mean_voltage_mv"],
                    "mean_current_pa": snapshot["layers"][name]["mean_current_pa"],
                    "spike_count": snapshot["layers"][name]["spike_count"],
                }
                for name in active_names
            },
        }

    def snapshot(self) -> dict[str, object]:
        elapsed_seconds = max(self.time_ms / 1000.0, 1e-9)
        layers: dict[str, dict[str, float | int]] = {}
        for name in POPULATIONS:
            layers[name] = {
                "mean_voltage_mv": float(np.mean(self.voltage_mv[name])),
                "mean_current_pa": float(np.mean(self.last_current_pa[name])),
                "spike_count": int(np.sum(self.spike_counts[name])),
                "mean_firing_rate_hz": float(
                    np.sum(self.spike_counts[name]) / (len(self.spike_counts[name]) * elapsed_seconds)
                ),
                "activity_fraction": float(self.activity[name]),
            }
        return {
            "layers": layers,
            "simulated_time_ms": float(self.time_ms),
            "operation_count": int(self.operation_count),
            "units": {
                "membrane_voltage": "mV",
                "synaptic_current": "pA",
                "firing_rate": "Hz",
                "simulated_time": "ms",
                "spike_count": "count",
            },
            "model_scope": (
                "Generic current-based LIF population reference; not a fitted or "
                "validated reproduction of human ATL, IFG, episodic memory, or semantic cognition."
            ),
            "numerically_valid": bool(
                all(np.all(np.isfinite(values)) for values in self.voltage_mv.values())
                and all(np.all(np.isfinite(values)) for values in self.last_current_pa.values())
            ),
        }
