"""Optional current-based LIF reference network for the hybrid embodied–amodal pipeline.

The named populations are computational analogues, not a fitted reproduction
of human cortex or proof of a particular anatomical implementation.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .contracts import HybridConfig


POPULATION_NAMES = (
    "visual_spoke",
    "auditory_spoke",
    "motor_spoke",
    "somatosensory_spoke",
    "olfactory_gustatory_spoke",
    "affective_interoceptive_spoke",
    "linguistic_spoke",
    "temporal_binding",
    "atl_transmodal_hub",
    "semantic_control",
    "prediction_error",
    "response_system",
    "plasticity",
)

MODALITY_POPULATIONS = {
    "visual": "visual_spoke",
    "auditory": "auditory_spoke",
    "motor": "motor_spoke",
    "somatosensory": "somatosensory_spoke",
    "olfactory": "olfactory_gustatory_spoke",
    "gustatory": "olfactory_gustatory_spoke",
    "emotional": "affective_interoceptive_spoke",
    "interoceptive": "affective_interoceptive_spoke",
    "linguistic": "linguistic_spoke",
    "internal": "affective_interoceptive_spoke",
}


class HybridLIFSimulator:
    """Advance named modality, ATL, control, prediction, response, and plasticity populations."""

    def __init__(self, config: HybridConfig) -> None:
        """Allocate membrane state, refractory counters, spike counts, and population activity."""
        self.config = config
        size = config.population_size
        self.voltage_mv = {
            name: np.full(size, config.rest_mv, dtype=np.float64)
            for name in POPULATION_NAMES
        }
        self.refractory_ms = {
            name: np.zeros(size, dtype=np.float64) for name in POPULATION_NAMES
        }
        self.spike_counts = {
            name: np.zeros(size, dtype=np.int64) for name in POPULATION_NAMES
        }
        self.last_current_pa = {
            name: np.zeros(size, dtype=np.float64) for name in POPULATION_NAMES
        }
        self.activity = {name: 0.0 for name in POPULATION_NAMES}
        self.time_ms = 0.0
        self.trace: list[dict[str, Any]] = []

    def step(self, stage_id: str, drives: Mapping[str, float]) -> dict[str, Any]:
        """Advance the full reference network for one stage and return physical-unit summaries."""
        unknown = set(drives) - set(POPULATION_NAMES)
        if unknown:
            raise KeyError(f"unknown hybrid LIF populations: {sorted(unknown)}")
        clean = {
            name: float(np.clip(float(drives.get(name, 0.0)), 0.0, 1.0))
            for name in POPULATION_NAMES
        }
        for _ in range(self.config.steps_per_stage):
            global_activity = float(np.mean(tuple(self.activity.values())))
            next_activity: dict[str, float] = {}
            for name in POPULATION_NAMES:
                voltage = self.voltage_mv[name]
                refractory = self.refractory_ms[name]
                recurrent = self.config.recurrent_current_pa * self.activity[name]
                inhibitory = self.config.inhibitory_current_pa * global_activity
                current_pa = self.config.current_gain_pa * clean[name] + recurrent - inhibitory
                self.last_current_pa[name][:] = current_pa
                active = refractory <= 0.0
                # MOhm * pA = 10^-3 mV; retain this dimensional conversion explicitly.
                current_drive_mv = self.config.membrane_resistance_mohm * current_pa * 0.001
                derivative_mv = (
                    self.config.rest_mv - voltage + current_drive_mv
                ) / self.config.membrane_tau_ms
                voltage[active] += self.config.dt_ms * derivative_mv[active]
                refractory[:] = np.maximum(0.0, refractory - self.config.dt_ms)
                spiked = active & (voltage >= self.config.threshold_mv)
                if np.any(spiked):
                    self.spike_counts[name][spiked] += 1
                    voltage[spiked] = self.config.reset_mv
                    refractory[spiked] = self.config.refractory_ms
                voltage[~active] = self.config.reset_mv
                spike_fraction = float(np.count_nonzero(spiked)) / len(voltage)
                decay = math.exp(-self.config.dt_ms / 20.0)
                next_activity[name] = (
                    decay * self.activity[name] + (1.0 - decay) * spike_fraction
                )
            self.activity.update(next_activity)
            self.time_ms += self.config.dt_ms
        snapshot = self.snapshot()
        active_names = [name for name, drive in clean.items() if drive > 0.0]
        event = {
            "stage_id": stage_id,
            "simulated_time_ms": float(self.time_ms),
            "active_populations": active_names,
            "mean_network_activity": float(np.mean(tuple(self.activity.values()))),
            "population_metrics": {
                name: snapshot["layers"][name] for name in active_names
            },
        }
        self.trace.append(event)
        return event

    def snapshot(self) -> dict[str, Any]:
        """Return voltage, current, firing-rate, spike, and scientific-scope diagnostics."""
        elapsed_seconds = max(self.time_ms / 1000.0, 1e-9)
        layers: dict[str, dict[str, float | int]] = {}
        for name in POPULATION_NAMES:
            count = int(np.sum(self.spike_counts[name]))
            size = int(self.spike_counts[name].size)
            layers[name] = {
                "mean_voltage_mv": float(np.mean(self.voltage_mv[name])),
                "mean_current_pa": float(np.mean(self.last_current_pa[name])),
                "spike_count": count,
                "mean_firing_rate_hz": float(count / (size * elapsed_seconds)),
                "activity_fraction": float(self.activity[name]),
            }
        finite = all(
            np.all(np.isfinite(values))
            for values in (*self.voltage_mv.values(), *self.last_current_pa.values())
        )
        return {
            "enabled": True,
            "layers": layers,
            "operation_trace": list(self.trace),
            "simulated_time_ms": float(self.time_ms),
            "stage_count": len(self.trace),
            "units": {
                "membrane_voltage": "mV",
                "synaptic_current": "pA",
                "firing_rate": "Hz",
                "simulated_time": "ms",
                "spike_count": "count",
            },
            "numerically_valid": bool(finite),
            "model_scope": (
                "Generic current-based leaky integrate-and-fire reference. It is not fitted "
                "to human ATL, sensory association, control, or modality-specific recordings."
            ),
        }


def stage_drives(
    stage_number: int,
    modality_strengths: Mapping[str, float],
    hub_strength: float,
    control_strength: float,
    prediction_error: float = 0.0,
    response_strength: float = 0.0,
    plasticity_strength: float = 0.0,
) -> dict[str, float]:
    """Map functional signals to broad LIF reference populations for one registered stage."""
    drives = {name: 0.0 for name in POPULATION_NAMES}
    for modality, strength in modality_strengths.items():
        population = MODALITY_POPULATIONS.get(modality)
        if population:
            drives[population] = max(drives[population], float(np.clip(strength, 0.0, 1.0)))
    if stage_number in (7, 8, 9, 10, 11, 12):
        drives["temporal_binding"] = min(
            1.0, float(np.mean(tuple(modality_strengths.values()))) if modality_strengths else 0.0
        )
    if stage_number in (8, 9, 10, 11, 12, 13, 14, 15, 16, 18):
        drives["atl_transmodal_hub"] = float(np.clip(hub_strength, 0.0, 1.0))
    if stage_number in (4, 9, 14, 15, 17, 18):
        drives["semantic_control"] = float(np.clip(control_strength, 0.0, 1.0))
    if stage_number in (16, 18):
        drives["prediction_error"] = float(np.clip(prediction_error, 0.0, 1.0))
    if stage_number == 17:
        drives["response_system"] = float(np.clip(response_strength, 0.0, 1.0))
    if stage_number == 19:
        drives["plasticity"] = float(np.clip(plasticity_strength, 0.0, 1.0))
    return drives
