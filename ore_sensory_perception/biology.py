"""Configurable LIF reference network for the visual and auditory sensory-perception stages.

The populations and projections below are explicit computational hypotheses. They reproduce neither
full cochlear/cortical anatomy nor participant-specific physiology, and their activity is not measured
brain data. They provide a separate, inspectable dynamics track alongside the functional algorithm.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import SensoryConfig


POPULATIONS = (
    "retina_lgn", "v1_visual", "visual_text_grouping", "visual_attention",
    "ventral_visual_letters", "visual_language_feedback", "cochlea_tonotopy",
    "auditory_nerve", "auditory_brainstem", "auditory_scene_analysis",
    "auditory_attention", "auditory_cortex_acoustic", "stg_phonetic",
    "phoneme_population", "language_prediction", "multisensory_integration",
)

_PROJECTIONS = {
    "retina_lgn": ("v1_visual",),
    "v1_visual": ("visual_text_grouping", "ventral_visual_letters"),
    "visual_text_grouping": ("visual_attention",),
    "visual_attention": ("ventral_visual_letters",),
    "ventral_visual_letters": ("visual_language_feedback", "multisensory_integration"),
    "visual_language_feedback": ("visual_attention", "phoneme_population"),
    "cochlea_tonotopy": ("auditory_nerve",),
    "auditory_nerve": ("auditory_brainstem",),
    "auditory_brainstem": ("auditory_scene_analysis", "auditory_cortex_acoustic"),
    "auditory_scene_analysis": ("auditory_attention",),
    "auditory_attention": ("auditory_cortex_acoustic",),
    "auditory_cortex_acoustic": ("stg_phonetic",),
    "stg_phonetic": ("phoneme_population",),
    "phoneme_population": ("language_prediction", "multisensory_integration"),
    "language_prediction": ("stg_phonetic", "auditory_attention"),
    "multisensory_integration": ("visual_language_feedback",),
}

_OPERATION_TARGETS = (
    "retina_lgn", "v1_visual", "visual_text_grouping", "visual_text_grouping", "visual_text_grouping",
    "visual_attention", "ventral_visual_letters", "ventral_visual_letters", "ventral_visual_letters",
    "ventral_visual_letters", "visual_language_feedback", "visual_language_feedback",
    "cochlea_tonotopy", "cochlea_tonotopy", "auditory_nerve", "auditory_brainstem",
    "auditory_scene_analysis", "auditory_attention", "auditory_cortex_acoustic", "auditory_cortex_acoustic",
    "auditory_cortex_acoustic", "auditory_cortex_acoustic", "auditory_cortex_acoustic", "auditory_cortex_acoustic",
    "stg_phonetic", "stg_phonetic", "stg_phonetic", "stg_phonetic", "stg_phonetic", "stg_phonetic",
    "stg_phonetic", "phoneme_population", "phoneme_population", "phoneme_population", "phoneme_population",
    "language_prediction", "stg_phonetic", "phoneme_population", "multisensory_integration",
)


class LIFPopulation:
    """Simulate one homogeneous current-based leaky-integrate-and-fire population.

    Membrane voltage is represented in millivolts, input current in picoamperes, membrane resistance
    in megaohms and time in milliseconds. The factor 0.001 converts pA·MΩ to mV.
    """

    def __init__(self, name: str, size: int, config: SensoryConfig, rng: np.random.Generator) -> None:
        """Initialize membrane voltages, refractory counters and spike counters for a named population."""
        self.name = name
        self.config = config
        self.voltage_mv = rng.normal(config.rest_mv, 0.5, size).astype(np.float64)
        self.refractory_remaining_ms = np.zeros(size, dtype=np.float64)
        self.spike_counts = np.zeros(size, dtype=np.int64)
        self.last_current_pa = np.zeros(size, dtype=np.float64)
        self.last_rate_hz = 0.0
        self.elapsed_ms = 0.0

    def reset(self) -> None:
        """Return the population to resting voltage and clear its refractory and spike history."""
        self.voltage_mv.fill(self.config.rest_mv)
        self.refractory_remaining_ms.fill(0.0)
        self.spike_counts.fill(0)
        self.last_current_pa.fill(0.0)
        self.last_rate_hz = 0.0
        self.elapsed_ms = 0.0

    def advance(self, current_pa: float, steps: int = 1) -> dict[str, Any]:
        """Integrate current-based LIF membrane dynamics and return voltage, current and firing-rate summaries.

        The update is Euler discretization of tau*dV/dt=(Vrest-V)+R*I. Refractory handling and
        reset thresholds are simplified and configurable; they are not fitted to a human participant.
        """
        cfg = self.config
        drive = max(0.0, float(current_pa))
        self.last_current_pa.fill(drive)
        spikes_total = 0
        for _ in range(max(1, int(steps))):
            active = self.refractory_remaining_ms <= 0.0
            effective_current_mv = 0.001 * cfg.membrane_resistance_mohm * drive
            delta = (cfg.dt_ms / cfg.membrane_tau_ms) * (
                cfg.rest_mv - self.voltage_mv + effective_current_mv
            )
            self.voltage_mv[active] += delta[active]
            self.refractory_remaining_ms[~active] -= cfg.dt_ms
            spiked = active & (self.voltage_mv >= cfg.threshold_mv)
            count = int(np.count_nonzero(spiked))
            spikes_total += count
            self.spike_counts[spiked] += 1
            self.voltage_mv[spiked] = cfg.reset_mv
            self.refractory_remaining_ms[spiked] = cfg.dt_ms * 2.0
            self.elapsed_ms += cfg.dt_ms
        self.last_rate_hz = float(spikes_total / max(self.voltage_mv.size * steps * cfg.dt_ms / 1000.0, 1e-12))
        return {
            "population": self.name,
            "mean_voltage_mv": float(np.mean(self.voltage_mv)),
            "min_voltage_mv": float(np.min(self.voltage_mv)),
            "max_voltage_mv": float(np.max(self.voltage_mv)),
            "mean_current_pa": drive,
            "spikes_this_update": spikes_total,
            "mean_firing_rate_hz": self.last_rate_hz,
            "elapsed_ms": self.elapsed_ms,
        }


class SensoryBiologicalSimulator:
    """Advance a recurrent 16-population LIF reference network once for every sensory operation.

    Visual and auditory pathways begin in separate modeled populations and interact through top-down
    language prediction and a multisensory readout. Connectivity and all parameters remain explicit
    model assumptions; output must never be presented as an empirical biological measurement.
    """

    def __init__(self, config: SensoryConfig) -> None:
        """Create coupled LIF populations using deterministic initialization and a directed projection graph."""
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.populations = {
            name: LIFPopulation(name, config.lif_population_size, config, self.rng)
            for name in POPULATIONS
        }
        self.outgoing = dict(_PROJECTIONS)
        self.time_ms = 0.0
        self.operation_count = 0

    def reset(self) -> None:
        """Reset every population and the shared simulation clock while preserving connectivity and settings."""
        for population in self.populations.values():
            population.reset()
        self.time_ms = 0.0
        self.operation_count = 0

    def advance_operation(self, operation_index: int, operation_id: str, drive: float) -> dict[str, Any]:
        """Advance all connected populations for one operation and return a separate neural-state trace.

        ``drive`` is a bounded functional activity proxy obtained from that operation's report. It is
        injected only into the target population; recurrent projections carry prior population rates
        forward, while a small shared inhibitory term limits runaway activity.
        """
        if not 1 <= int(operation_index) <= len(_OPERATION_TARGETS):
            raise ValueError("operation_index must identify one of the 39 registered sensory operations")
        target = _OPERATION_TARGETS[int(operation_index) - 1]
        activity = float(np.clip(drive, 0.0, 1.0))
        prior_rates = {name: unit.last_rate_hz for name, unit in self.populations.items()}
        rates = np.asarray(list(prior_rates.values()), dtype=np.float64)
        inhibitory = float(np.mean(rates)) if rates.size else 0.0
        reports: dict[str, dict[str, Any]] = {}
        for name in POPULATIONS:
            incoming = [source for source, targets in self.outgoing.items() if name in targets]
            recurrent = float(np.mean([prior_rates[source] for source in incoming])) if incoming else 0.0
            direct = self.config.input_current_pa * activity if name == target else 0.0
            synaptic = self.config.recurrent_current_pa * np.tanh(recurrent / 80.0)
            inhibitory_pa = 0.02 * self.config.recurrent_current_pa * np.tanh(inhibitory / 80.0)
            current = max(0.0, direct + synaptic - inhibitory_pa)
            reports[name] = self.populations[name].advance(current, self.config.lif_steps_per_operation)
        self.time_ms += self.config.dt_ms * self.config.lif_steps_per_operation
        self.operation_count += 1
        target_report = reports[target]
        return {
            "operation_id": operation_id,
            "operation_index": int(operation_index),
            "target_population": target,
            "time_ms": self.time_ms,
            "target_state": target_report,
            "network_mean_rate_hz": float(np.mean([item["mean_firing_rate_hz"] for item in reports.values()])),
            "network_mean_voltage_mv": float(np.mean([item["mean_voltage_mv"] for item in reports.values()])),
            "population_states": reports,
            "biological_status": "simulated_reference_not_measured_brain_data",
        }
