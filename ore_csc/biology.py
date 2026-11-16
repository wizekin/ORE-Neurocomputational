"""Current-based LIF population reference for the CSC control network.

This is a transparent computational hypothesis, not a fitted or validated
simulation of human semantic cognition or a specific anatomical circuit.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .contracts import CSCConfig

POPULATION_NAMES = (
    "sensory_input",
    "lexical_candidates",
    "distributed_semantics",
    "atl_hub",
    "context_task",
    "left_ifg",
    "posterior_mtg",
    "posterior_itg",
    "dorsomedial_pfc",
    "dorsolateral_pfc",
    "parietal_attention",
    "response_system",
    "plasticity",
)


class CSCLIFSimulator:
    """Advance named CSC reference populations using current-based leaky integrate-and-fire dynamics."""

    def __init__(self, config: CSCConfig) -> None:
        """Allocate membrane voltage, refractory counters, spike counts, and recurrent activity."""
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
        self.trace: list[dict[str, Any]] = []

    def step(self, stage_id: str, drives: Mapping[str, float]) -> dict[str, Any]:
        """Advance all populations for one algorithm stage and return a physical-unit snapshot."""
        unknown = set(drives) - set(POPULATION_NAMES)
        if unknown:
            raise KeyError("unknown CSC neural populations: {0}".format(sorted(unknown)))
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
                # MOhm * pA = 1e-3 mV; the conversion keeps reported units explicit.
                voltage_drive_mv = self.config.membrane_resistance_mohm * current_pa * 0.001
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
                rate_fraction = float(np.count_nonzero(spiked)) / len(voltage)
                decay = math.exp(-self.config.dt_ms / 20.0)
                next_activity[name] = (
                    decay * self.activity[name] + (1.0 - decay) * rate_fraction
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
        """Return voltage, current, firing-rate, spike, and model-scope diagnostics."""
        elapsed_seconds = max(self.time_ms / 1000.0, 1e-9)
        layers: dict[str, dict[str, float | int]] = {}
        for name in POPULATION_NAMES:
            count = int(np.sum(self.spike_counts[name]))
            population_size = int(self.spike_counts[name].size)
            layers[name] = {
                "mean_voltage_mv": float(np.mean(self.voltage_mv[name])),
                "mean_current_pa": float(np.mean(self.last_current_pa[name])),
                "spike_count": count,
                "mean_firing_rate_hz": float(count / (population_size * elapsed_seconds)),
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
                "Generic current-based LIF population reference. It is not fitted to human "
                "ATL, IFG, pMTG, pITG, prefrontal, or parietal recordings and is not a "
                "validated reproduction of human controlled semantic cognition."
            ),
        }


def stage_drives(stage_number: int, demand: float, lexical: float,
                 semantic: float, hub: float, context: float,
                 control: float, selected: float) -> dict[str, float]:
    """Map functional CSC stage signals onto biological-reference population input drives."""
    drives = {name: 0.0 for name in POPULATION_NAMES}
    if stage_number == 1:
        drives["sensory_input"] = 0.9
    if stage_number == 2:
        drives["lexical_candidates"] = lexical
    if stage_number in (3, 10, 11, 12, 13, 14, 16):
        drives["distributed_semantics"] = semantic
    if stage_number in (4, 16):
        drives["atl_hub"] = hub
    if stage_number in (5, 6):
        drives["context_task"] = context
    if stage_number in (7, 8, 9, 10, 11, 12, 13, 14, 15):
        drives["left_ifg"] = control
        drives["posterior_mtg"] = control * 0.9
        drives["posterior_itg"] = control * 0.65
        drives["dorsomedial_pfc"] = demand * 0.8
        drives["dorsolateral_pfc"] = control * 0.75
        # Parietal involvement is conditional, not mandatory for every semantic task.
        drives["parietal_attention"] = control * (0.5 if stage_number in (8, 9, 15) else 0.0)
    if stage_number == 17:
        drives["response_system"] = selected
    if stage_number == 18:
        drives["plasticity"] = demand
    return drives
