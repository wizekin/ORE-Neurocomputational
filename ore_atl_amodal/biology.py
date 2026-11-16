"""A generic bilateral ATL leaky-integrate-and-fire reference simulation.

This module provides an explicitly separate biological-dynamics track. It is
not fitted to human neural recordings and does not assert that the brain
literally performs the software equations used by the functional pipeline.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .contracts import AmodalATLConfig, MODALITIES


POPULATION_NAMES = (
    "visual_spoke",
    "auditory_spoke",
    "tactile_spoke",
    "spatial_motor_spoke",
    "linguistic_spoke",
    "affective_spoke",
    "functional_spoke",
    "distributed_spokes",
    "atl_left",
    "atl_right",
    "atl_hub",
    "semantic_feedback",
    "phrase_composition",
    "semantic_memory",
)

MODALITY_POPULATIONS = {
    "visual": "visual_spoke",
    "auditory": "auditory_spoke",
    "tactile": "tactile_spoke",
    "spatial_motor": "spatial_motor_spoke",
    "linguistic": "linguistic_spoke",
    "affective": "affective_spoke",
    "functional": "functional_spoke",
}


class ATLBiologySimulator:
    """Simulate coarse population activity with current-based leaky-integrate-and-fire dynamics."""

    def __init__(self, config: AmodalATLConfig) -> None:
        """Allocate membrane, refractory, activity, current, and spike state for each modeled population."""
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
        self.activity = {name: 0.0 for name in POPULATION_NAMES}
        self.last_current_pa = {
            name: np.zeros(size, dtype=np.float64) for name in POPULATION_NAMES
        }
        self.time_ms = 0.0
        self.stage_count = 0
        self.trace: list[dict[str, Any]] = []

    def step(self, stage_id: str, drives: Mapping[str, float]) -> dict[str, Any]:
        """Advance each named population and record a physical-unit snapshot for one algorithmic stage."""
        unknown = set(drives) - set(POPULATION_NAMES)
        if unknown:
            raise KeyError(f"Unknown biological populations: {sorted(unknown)}")
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
                current_pa = (
                    self.config.input_current_pa * clean[name]
                    + recurrent - inhibitory
                )
                self.last_current_pa[name][:] = current_pa
                active = refractory <= 0.0
                # Ohm's law in these units: MOhm * pA * 0.001 = mV.
                voltage_drive_mv = (
                    self.config.membrane_resistance_mohm * current_pa * 0.001
                )
                derivative_mv = (
                    self.config.rest_mv - voltage + voltage_drive_mv
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

        self.stage_count += 1
        snapshot = self.snapshot()
        active_populations = [name for name, drive in clean.items() if drive > 0.0]
        event = {
            "stage_id": stage_id,
            "simulated_time_ms": float(self.time_ms),
            "active_populations": active_populations,
            "mean_network_activity": float(np.mean(tuple(self.activity.values()))),
            "population_metrics": {
                name: snapshot["layers"][name] for name in active_populations
            },
        }
        self.trace.append(event)
        return event

    def snapshot(self) -> dict[str, Any]:
        """Return population voltage, current, spike-rate, validity, and model-scope diagnostics."""
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
            "stage_count": int(self.stage_count),
            "units": {
                "membrane_voltage": "mV",
                "synaptic_current": "pA",
                "firing_rate": "Hz",
                "simulated_time": "ms",
                "spike_count": "count",
            },
            "numerically_valid": bool(finite),
            "model_scope": (
                "Generic current-based LIF population reference. It is not a fitted or "
                "validated reproduction of human ATL subdivisions, sensory spokes, or "
                "semantic cognition. Numerical parameters are illustrative simulation settings."
            ),
        }


def biological_drives(
    stage_number: int,
    modality_activations: Mapping[str, float],
    hub_activity: float = 0.0,
    feedback_activity: float = 0.0,
    phrase_activity: float = 0.0,
    memory_activity: float = 0.0,
) -> dict[str, float]:
    """Map interpretable functional signals onto coarse modality, bilateral ATL, feedback, and composition populations."""
    drives = {name: 0.0 for name in POPULATION_NAMES}
    for modality, activation in modality_activations.items():
        population = MODALITY_POPULATIONS.get(modality)
        if population is not None:
            drives[population] = float(np.clip(activation, 0.0, 1.0))
    if 1 <= stage_number <= 5:
        drives["distributed_spokes"] = float(
            np.clip(np.mean(tuple(modality_activations.values())) if modality_activations else 0.0, 0.0, 1.0)
        )
    if 5 <= stage_number <= 10:
        drives["atl_left"] = float(np.clip(hub_activity * 0.98, 0.0, 1.0))
        drives["atl_right"] = float(np.clip(hub_activity, 0.0, 1.0))
        drives["atl_hub"] = float(np.clip(hub_activity, 0.0, 1.0))
    if stage_number == 9:
        drives["semantic_feedback"] = float(np.clip(feedback_activity, 0.0, 1.0))
    if 10 <= stage_number <= 12:
        drives["semantic_memory"] = float(np.clip(memory_activity, 0.0, 1.0))
        drives["atl_hub"] = max(drives["atl_hub"], float(np.clip(hub_activity, 0.0, 1.0)))
    if 11 <= stage_number <= 18:
        drives["phrase_composition"] = float(np.clip(phrase_activity, 0.0, 1.0))
        drives["atl_left"] = max(drives["atl_left"], float(np.clip(hub_activity * 0.98, 0.0, 1.0)))
        drives["atl_right"] = max(drives["atl_right"], float(np.clip(hub_activity, 0.0, 1.0)))
    if stage_number == 19:
        drives["semantic_memory"] = max(
            drives["semantic_memory"], float(np.clip(memory_activity, 0.0, 1.0))
        )
        drives["atl_hub"] = max(drives["atl_hub"], float(np.clip(hub_activity, 0.0, 1.0)))
    return drives
