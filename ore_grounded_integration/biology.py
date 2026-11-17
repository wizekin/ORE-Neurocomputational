"""A small neural-dynamics reference model for the grounded semantic stages.

The simulator uses an explicit current-based leaky integrate-and-fire (LIF) equation for named
functional populations. It is a configurable computational analogy, not an anatomically fitted,
physiologically validated or empirically identified model of a human brain.
"""
from __future__ import annotations

from typing import Any, Mapping
import math

import numpy as np

from .contracts import GroundedSemanticConfig

POPULATION_NAMES = (
    "visual_spoke",
    "auditory_spoke",
    "linguistic_spoke",
    "motor_spoke",
    "context_spoke",
    "distributed_feature_network",
    "atl_transmodal_hub",
    "semantic_control_network",
    "predictive_network",
    "hippocampal_memory_network",
    "plasticity_network",
    "language_output_network",
    "instruction_network",
)

LIF_STEPS_PER_STAGE = 40

_STAGE_POPULATIONS = {
    "GS-01": ("visual_spoke", "auditory_spoke", "linguistic_spoke", "context_spoke"),
    "GS-02": ("visual_spoke", "auditory_spoke", "linguistic_spoke", "motor_spoke", "context_spoke"),
    "GS-03": ("linguistic_spoke", "distributed_feature_network"),
    "GS-04": ("context_spoke", "semantic_control_network"),
    "GS-05": ("atl_transmodal_hub", "distributed_feature_network"),
    "GS-06": ("motor_spoke", "distributed_feature_network"),
    "GS-07": ("atl_transmodal_hub", "distributed_feature_network"),
    "GS-08": ("atl_transmodal_hub", "hippocampal_memory_network"),
    "GS-09": ("semantic_control_network", "atl_transmodal_hub"),
    "GS-10": ("linguistic_spoke", "atl_transmodal_hub"),
    "GS-11": ("predictive_network", "linguistic_spoke"),
    "GS-12": ("predictive_network", "semantic_control_network"),
    "GS-13": ("semantic_control_network",),
    "GS-14": ("hippocampal_memory_network", "atl_transmodal_hub"),
    "GS-15": ("atl_transmodal_hub", "distributed_feature_network"),
    "GS-16": ("language_output_network", "motor_spoke"),
    "GS-17": ("plasticity_network", "atl_transmodal_hub"),
    "GS-18": ("hippocampal_memory_network", "plasticity_network"),
    "GS-19": ("instruction_network", "visual_spoke", "auditory_spoke", "linguistic_spoke"),
    "GS-20": ("instruction_network", "language_output_network"),
    "GS-21": ("instruction_network", "semantic_control_network"),
    "GS-22": ("instruction_network", "plasticity_network"),
}


class GroundedNeuralReference:
    """Advance distributed modality, hub, control, memory, prediction and output populations.

    Membrane voltage is reported in mV, current in pA, simulated time in ms and activity as a
    population firing fraction. The population names denote candidate functional roles, not
    one-to-one anatomical localization.
    """

    def __init__(self, config: GroundedSemanticConfig) -> None:
        """Allocate deterministic membrane voltage, refractory and spike state for each population."""
        self.config = config
        size = config.population_size
        self.voltage_mv = {
            name: np.full(size, config.rest_mv, dtype=np.float64) for name in POPULATION_NAMES
        }
        self.refractory_ms = {
            name: np.zeros(size, dtype=np.float64) for name in POPULATION_NAMES
        }
        self.spikes = {
            name: np.zeros(size, dtype=np.int64) for name in POPULATION_NAMES
        }
        self.activity = {name: 0.0 for name in POPULATION_NAMES}
        self.current_pa = {
            name: np.zeros(size, dtype=np.float64) for name in POPULATION_NAMES
        }
        self.elapsed_ms = 0.0
        self.trace: list[dict[str, Any]] = []

    def advance(self, stage_id: str, drive: float | Mapping[str, float]) -> dict[str, Any]:
        """Advance the LIF network for one stage and return its physical-unit snapshot.

        Each stage is integrated for 40 Euler time steps. The membrane equation is
        V(t+dt)=V(t)+(dt/tau)*(Vrest-V+0.001*Rm*I[pA]). Stage drives excite the relevant
        functional populations; recurrent activity and global inhibition provide a minimal
        network interaction term.
        """
        if stage_id not in _STAGE_POPULATIONS:
            raise KeyError(f"unknown grounded-semantic stage: {stage_id}")
        if isinstance(drive, Mapping):
            drives = {name: float(drive.get(name, 0.0)) for name in POPULATION_NAMES}
        else:
            value = float(drive)
            drives = {name: value if name in _STAGE_POPULATIONS[stage_id] else 0.0
                      for name in POPULATION_NAMES}
        if any(not math.isfinite(value) for value in drives.values()):
            raise ValueError("biological drives must be finite")
        drives = {name: float(np.clip(value, 0.0, 1.0)) for name, value in drives.items()}
        cfg = self.config
        for _ in range(LIF_STEPS_PER_STAGE):
            global_activity = float(np.mean(tuple(self.activity.values())))
            next_activity: dict[str, float] = {}
            for name in POPULATION_NAMES:
                voltage = self.voltage_mv[name]
                refractory = self.refractory_ms[name]
                recurrent = cfg.recurrent_gain_pa * self.activity[name]
                network_inhibition = cfg.inhibitory_gain_pa * global_activity
                current = cfg.current_gain_pa * drives[name] + recurrent - network_inhibition
                self.current_pa[name][:] = current
                active = refractory <= 0.0
                voltage_drive = 0.001 * cfg.membrane_resistance_mohm * current
                derivative = (cfg.rest_mv - voltage + voltage_drive) / cfg.membrane_tau_ms
                voltage[active] += cfg.dt_ms * derivative[active]
                refractory[:] = np.maximum(0.0, refractory - cfg.dt_ms)
                fired = active & (voltage >= cfg.threshold_mv)
                self.spikes[name][fired] += 1
                voltage[fired] = cfg.reset_mv
                refractory[fired] = cfg.refractory_ms
                voltage[~active] = cfg.reset_mv
                instantaneous = float(np.count_nonzero(fired)) / float(len(voltage))
                decay = math.exp(-cfg.dt_ms / 20.0)
                next_activity[name] = (
                    decay * self.activity[name] + (1.0 - decay) * instantaneous
                )
            self.activity.update(next_activity)
            self.elapsed_ms += cfg.dt_ms
        snapshot = {
            "stage_id": stage_id,
            "simulated_time_ms": float(self.elapsed_ms),
            "integration_steps": LIF_STEPS_PER_STAGE,
            "target_populations": list(_STAGE_POPULATIONS[stage_id]),
            "active_populations": [name for name, value in drives.items() if value > 0.0],
            "population_activity": dict(self.activity),
            "mean_membrane_voltage_mv": {
                name: float(np.mean(values)) for name, values in self.voltage_mv.items()
            },
            "mean_synaptic_current_pa": {
                name: float(np.mean(values)) for name, values in self.current_pa.items()
            },
            "spike_counts": {name: int(np.sum(values)) for name, values in self.spikes.items()},
            "scientific_scope": "abstract LIF reference; not human neural measurements",
        }
        self.trace.append(snapshot)
        return snapshot

    def summary(self) -> dict[str, Any]:
        """Return current activity, cumulative spikes and explicit scope limits for the simulator."""
        return {
            "model": "grounded_semantic_current_based_lif_reference",
            "equation": "V += (dt_ms/tau_ms) * (Vrest_mV - V_mV + 0.001 * Rm_MOhm * I_pA)",
            "integration_steps_per_stage": LIF_STEPS_PER_STAGE,
            "units": {
                "membrane_voltage": "mV",
                "synaptic_current": "pA",
                "simulated_time": "ms",
                "spike_count": "count",
                "activity": "fraction",
            },
            "activity": dict(self.activity),
            "spike_counts": {name: int(np.sum(values)) for name, values in self.spikes.items()},
            "simulated_time_ms": float(self.elapsed_ms),
            "scientific_scope": (
                "Configurable computational analogy only; not an anatomically fitted or "
                "physiologically validated model and not a direct measurement of a brain."
            ),
        }
