"""Nine-population current-based LIF reference for sentence-semantic processing.

This deterministic simulator is an explicit computational analogy. It is not
an anatomically fitted model or evidence that brains execute the software steps.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from .contracts import SemanticCompositionConfig


POPULATIONS = {
    "concept_retrieval": "STAGE-S1",
    "semantic_binding": "STAGE-S2",
    "predicate_argument": "STAGE-S3",
    "thematic_role": "STAGE-S4",
    "proposition": "STAGE-S5",
    "negation_modality": "STAGE-S6",
    "quantifier_scope": "STAGE-S7",
    "causal_inference": "STAGE-S8",
    "sentence_integration": "STAGE-S9",
}


class SemanticCompositionLIFSimulator:
    """Advance nine abstract processing populations with a current-based LIF equation.

    The simulator reports membrane voltage in mV, current in pA, firing rate in
    Hz, spike counts, and simulated time in ms. Population names are functional
    hypotheses and should not be read as exact anatomical localization.
    """

    def __init__(self, config: SemanticCompositionConfig) -> None:
        """Initialize each population's voltage, refractory state, and spike counter."""
        self.config = config
        self.names = tuple(POPULATIONS)
        self.voltage = {
            name: np.full(config.population_size, config.rest_mv, dtype=np.float64)
            for name in self.names
        }
        self.refractory = {
            name: np.zeros(config.population_size, dtype=np.float64)
            for name in self.names
        }
        self.spikes = {name: np.zeros(config.population_size, dtype=np.int64) for name in self.names}
        self.currents = {name: np.zeros(config.population_size, dtype=np.float64) for name in self.names}
        self.elapsed_ms = 0.0
        self._previous_activity = {name: 0.0 for name in self.names}
        self._last_rates = {name: 0.0 for name in self.names}

    def advance(self, stage_id: str, drive: float) -> dict[str, Any]:
        """Advance the population assigned to a stage and return its measurable state.

        The Euler update is V(t+dt)=V(t)+(dt/tau)*(Vrest-V+10^-3*Rm*I[pA]).
        External drive is combined with bounded recurrent excitation and
        population-level inhibition. Threshold crossing produces a reset spike.
        """
        population = next((name for name, value in POPULATIONS.items() if value == stage_id), None)
        if population is None:
            raise KeyError(f"unknown semantic-composition stage: {stage_id}")
        x = float(np.clip(drive, 0.0, 1.0))
        cfg = self.config
        for _ in range(cfg.steps_per_operation):
            previous = self._previous_activity[population]
            recurrent = cfg.recurrent_gain_pa * previous
            inhibition = cfg.inhibition_gain_pa * previous
            current = np.full(
                cfg.population_size,
                cfg.current_gain_pa * x + recurrent - inhibition,
                dtype=np.float64,
            )
            self.currents[population] = current
            refractory_mask = self.refractory[population] > 0.0
            self.refractory[population][refractory_mask] -= cfg.dt_ms
            active_mask = ~refractory_mask
            voltage = self.voltage[population]
            delta = (cfg.dt_ms / cfg.membrane_tau_ms) * (
                cfg.rest_mv - voltage + 1e-3 * cfg.membrane_resistance_mohm * current
            )
            voltage[active_mask] += delta[active_mask]
            fired = active_mask & (voltage >= cfg.threshold_mv)
            self.spikes[population][fired] += 1
            voltage[fired] = cfg.reset_mv
            self.refractory[population][fired] = cfg.refractory_ms
            self.elapsed_ms += cfg.dt_ms
        duration_s = max(cfg.steps_per_operation * cfg.dt_ms / 1000.0, 1e-12)
        rate = float(np.sum(self.spikes[population]) / (cfg.population_size * duration_s))
        self._last_rates[population] = rate
        activity = float(np.mean(self.spikes[population] > 0))
        self._previous_activity[population] = activity
        return self._population_summary(population)

    def _population_summary(self, population: str) -> dict[str, Any]:
        """Summarize one population without exposing mutable NumPy state arrays."""
        return {
            "mean_membrane_voltage_mv": float(np.mean(self.voltage[population])),
            "mean_synaptic_current_pa": float(np.mean(self.currents[population])),
            "mean_firing_rate_hz": float(self._last_rates[population]),
            "spike_count": int(np.sum(self.spikes[population])),
            "neuron_count": int(self.config.population_size),
            "simulated_time_ms": float(self.elapsed_ms),
        }

    def summary(self) -> dict[str, Any]:
        """Return all nine population summaries and the simulator's scientific scope note."""
        return {
            "model": "current_based_leaky_integrate_and_fire_reference",
            "equation": "V += (dt_ms/tau_ms) * (Vrest_mV - V_mV + 1e-3 * Rm_MOhm * I_pA)",
            "units": {
                "membrane_voltage": "mV",
                "synaptic_current": "pA",
                "firing_rate": "Hz",
                "time": "ms",
                "spike_count": "count",
            },
            "layers": {name: self._population_summary(name) for name in self.names},
            "simulated_time_ms": float(self.elapsed_ms),
            "scientific_scope": (
                "Configurable computational reference only; not an anatomically fitted, "
                "physiologically validated, or empirically identified model of human language."
            ),
        }
