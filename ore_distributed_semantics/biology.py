"""Current-based LIF reference network for the 36-stage semantic model.

The functional track and this biological-reference track are deliberately
separate. LIF values are synthetic model state, not measured human neural data.
"""
from __future__ import annotations

from typing import Any, Mapping
import math

import numpy as np

from .contracts import DistributedSemanticConfig


POPULATION_NAMES = (
    "visual_spoke", "auditory_spoke", "tactile_spoke",
    "olfactory_gustatory_spoke", "spatial_motor_spoke", "linguistic_spoke",
    "atl_left", "atl_right", "semantic_control_ifg", "angular_gyrus",
    "posterior_temporal", "hippocampal_memory", "default_mode_network",
    "inhibitory_interneurons",
)

STAGE_POPULATIONS = {
    1: ("visual_spoke", "auditory_spoke", "tactile_spoke", "linguistic_spoke"),
    2: ("linguistic_spoke", "posterior_temporal"),
    3: ("posterior_temporal", "atl_left", "atl_right"),
    4: ("visual_spoke", "auditory_spoke", "spatial_motor_spoke", "atl_left", "atl_right"),
    5: ("visual_spoke", "auditory_spoke", "atl_left", "atl_right"),
    6: ("atl_left", "atl_right"),
    7: ("atl_left", "atl_right", "angular_gyrus"),
    8: ("atl_left", "atl_right", "angular_gyrus"),
    9: ("linguistic_spoke", "posterior_temporal", "semantic_control_ifg"),
    10: ("atl_left", "atl_right", "linguistic_spoke"),
    11: ("posterior_temporal", "semantic_control_ifg", "angular_gyrus"),
    12: ("posterior_temporal", "atl_left", "semantic_control_ifg"),
    13: ("atl_left", "atl_right", "hippocampal_memory", "default_mode_network"),
    14: ("atl_left", "atl_right", "semantic_control_ifg"),
    15: ("atl_left", "atl_right", "semantic_control_ifg"),
    16: ("semantic_control_ifg", "inhibitory_interneurons"),
    17: ("angular_gyrus", "posterior_temporal", "default_mode_network"),
    18: ("posterior_temporal", "angular_gyrus"),
    19: ("posterior_temporal", "semantic_control_ifg"),
    20: ("posterior_temporal", "atl_left", "atl_right"),
    21: ("semantic_control_ifg", "posterior_temporal"),
    22: ("hippocampal_memory", "atl_left", "atl_right"),
    23: ("hippocampal_memory", "default_mode_network", "angular_gyrus"),
    24: ("default_mode_network", "hippocampal_memory", "angular_gyrus"),
    25: ("semantic_control_ifg", "inhibitory_interneurons"),
    26: ("hippocampal_memory", "atl_left", "atl_right"),
    27: ("hippocampal_memory", "atl_left", "atl_right"),
    28: ("atl_left", "atl_right", "semantic_control_ifg"),
    29: ("linguistic_spoke", "semantic_control_ifg", "atl_left"),
    30: ("visual_spoke", "auditory_spoke", "spatial_motor_spoke", "atl_left", "atl_right"),
    31: ("atl_left", "atl_right"),
    32: ("atl_left", "atl_right"),
    33: ("atl_left", "atl_right"),
    34: ("semantic_control_ifg", "inhibitory_interneurons"),
    35: ("visual_spoke", "auditory_spoke", "spatial_motor_spoke", "atl_left", "atl_right"),
    36: ("posterior_temporal", "angular_gyrus", "semantic_control_ifg"),
}


def biological_drives(stage_number: int, output: Mapping[str, Any]) -> dict[str, float]:
    """Convert a stage's functional activity summary into bounded population input drives."""
    active = STAGE_POPULATIONS.get(stage_number, ())
    strength = float(output.get("activity", output.get("confidence", 0.45)) or 0.0)
    strength = min(1.0, max(0.0, strength))
    drives = {name: 0.0 for name in POPULATION_NAMES}
    for name in active:
        drives[name] = strength
    if output.get("conflict", False):
        drives["semantic_control_ifg"] = max(drives["semantic_control_ifg"], 0.75)
        drives["inhibitory_interneurons"] = max(drives["inhibitory_interneurons"], 0.65)
    return drives


class SemanticLIFSimulator:
    """Advance named semantic populations with current-based leaky integrate-and-fire dynamics."""

    def __init__(self, config: DistributedSemanticConfig) -> None:
        """Allocate membrane, refractory, recurrent, spike, and time state reproducibly."""
        self.config = config
        count = config.population_size
        self.voltage_mv = {
            name: np.full(count, config.rest_mv, dtype=np.float64)
            for name in POPULATION_NAMES
        }
        self.refractory_ms = {
            name: np.zeros(count, dtype=np.float64) for name in POPULATION_NAMES
        }
        self.spike_counts = {
            name: np.zeros(count, dtype=np.int64) for name in POPULATION_NAMES
        }
        self.activity = {name: 0.0 for name in POPULATION_NAMES}
        self.last_current_pa = {
            name: np.zeros(count, dtype=np.float64) for name in POPULATION_NAMES
        }
        self.time_ms = 0.0
        self._rng = np.random.default_rng(config.random_seed)
        self._weights = self._rng.normal(0.0, 0.025, (len(POPULATION_NAMES), len(POPULATION_NAMES)))
        np.fill_diagonal(self._weights, 0.12)
        self.trace: list[dict[str, Any]] = []

    def step(self, stage_id: str, drives: Mapping[str, float]) -> dict[str, Any]:
        """Integrate every named population for one stage and return a physical-unit snapshot."""
        unknown = set(drives) - set(POPULATION_NAMES)
        if unknown:
            raise KeyError(f"Unknown semantic populations: {sorted(unknown)}")
        names = POPULATION_NAMES
        drive_vector = np.asarray([
            float(np.clip(drives.get(name, 0.0), 0.0, 1.0)) for name in names
        ], dtype=np.float64)
        for _ in range(self.config.steps_per_stage):
            previous_activity = np.asarray([self.activity[name] for name in names], dtype=np.float64)
            recurrent = self._weights @ previous_activity
            global_inhibition = float(np.mean(previous_activity))
            next_activity: dict[str, float] = {}
            for index, name in enumerate(names):
                voltage = self.voltage_mv[name]
                refractory = self.refractory_ms[name]
                current_pa = (
                    self.config.current_gain_pa * drive_vector[index]
                    + self.config.recurrent_gain_pa * recurrent[index]
                    - self.config.inhibitory_gain_pa * global_inhibition
                )
                self.last_current_pa[name][:] = current_pa
                active = refractory <= 0.0
                # R(MOhm) * I(pA) * 0.001 converts current to millivolt drive.
                voltage_drive_mv = self.config.membrane_resistance_mohm * current_pa * 0.001
                derivative = (self.config.rest_mv - voltage + voltage_drive_mv) / self.config.membrane_tau_ms
                voltage[active] += self.config.dt_ms * derivative[active]
                refractory[:] = np.maximum(0.0, refractory - self.config.dt_ms)
                spiked = active & (voltage >= self.config.threshold_mv)
                if np.any(spiked):
                    self.spike_counts[name][spiked] += 1
                    voltage[spiked] = self.config.reset_mv
                    refractory[spiked] = self.config.refractory_ms
                voltage[~active] = self.config.reset_mv
                rate_fraction = float(np.count_nonzero(spiked)) / max(1, len(voltage))
                decay = math.exp(-self.config.dt_ms / 20.0)
                next_activity[name] = decay * self.activity[name] + (1.0 - decay) * rate_fraction
            self.activity.update(next_activity)
            self.time_ms += self.config.dt_ms

        snapshot = self.snapshot()
        event = {
            "stage_id": stage_id,
            "simulated_time_ms": round(self.time_ms, 4),
            "active_populations": [name for name in names if drive_vector[names.index(name)] > 0],
            "populations": snapshot,
            "scientific_status": "synthetic_lif_reference_not_measured_neural_data",
        }
        self.trace.append(event)
        return event

    def snapshot(self) -> dict[str, Any]:
        """Summarize voltage, input current, firing rate, and spike counts by population."""
        result: dict[str, Any] = {}
        for name in POPULATION_NAMES:
            result[name] = {
                "mean_voltage_mv": float(np.mean(self.voltage_mv[name])),
                "mean_current_pa": float(np.mean(self.last_current_pa[name])),
                "firing_rate_hz": float(self.activity[name] * 50.0),
                "spike_count": int(np.sum(self.spike_counts[name])),
            }
        return result
