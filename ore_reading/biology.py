"""Biologically inspired leaky integrate-and-fire reading populations.

This is a computational population model, not a fitted or validated model of
human reading circuitry. It exposes membrane voltage (mV), current (pA), spike
counts, firing rates (Hz), and simulation time (ms) as separate quantities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import math

import numpy as np


@dataclass(frozen=True, slots=True)
class LIFConfig:
    dt_ms: float = 1.0
    steps: int = 30
    tau_m_ms: float = 20.0
    refractory_ms: float = 2.0
    v_rest_mv: float = -70.0
    v_threshold_mv: float = -55.0
    v_reset_mv: float = -70.0
    input_gain_pa: float = 38.0
    recurrent_gain_pa: float = 2.5
    inhibition_gain_pa: float = 3.0

    def __post_init__(self) -> None:
        positive = (self.dt_ms, self.tau_m_ms, self.steps, self.input_gain_pa)
        if any(not math.isfinite(float(x)) or float(x) <= 0 for x in positive):
            raise ValueError("LIF dt, tau, steps and input gain must be positive")
        if self.v_threshold_mv <= self.v_reset_mv:
            raise ValueError("v_threshold_mv must exceed v_reset_mv")
        if self.refractory_ms < 0 or self.recurrent_gain_pa < 0 or self.inhibition_gain_pa < 0:
            raise ValueError("refractory and recurrent/inhibitory gains must be non-negative")


class LIFPopulation:
    """Deterministic current-based LIF population with recurrent inhibition."""

    def __init__(self, size: int, config: LIFConfig | None = None, seed: int = 0) -> None:
        if size < 1:
            raise ValueError("population size must be positive")
        self.size = int(size)
        self.config = config or LIFConfig()
        self.seed = int(seed)

    def simulate(self, drive: np.ndarray, *, feedback: np.ndarray | None = None) -> dict[str, object]:
        cfg = self.config
        external = np.asarray(drive, dtype=np.float64).reshape(-1)
        if external.size != self.size:
            raise ValueError("drive length must match population size")
        if not np.isfinite(external).all():
            raise ValueError("drive must be finite")
        external = np.clip(external, 0.0, 1.0)
        feedback_vector = np.zeros_like(external) if feedback is None else np.asarray(feedback, dtype=np.float64).reshape(-1)
        if feedback_vector.size not in (1, self.size) or not np.isfinite(feedback_vector).all():
            raise ValueError("feedback must be scalar-like or match the population size")
        if feedback_vector.size == 1:
            feedback_vector = np.full(self.size, float(feedback_vector[0]))
        feedback_vector = np.clip(feedback_vector, 0.0, 1.0)

        voltage = np.full(self.size, cfg.v_rest_mv, dtype=np.float64)
        refractory = np.zeros(self.size, dtype=np.float64)
        filtered_rate = np.zeros(self.size, dtype=np.float64)
        spike_counts = np.zeros(self.size, dtype=np.int64)
        voltage_sum = np.zeros(self.size, dtype=np.float64)
        current_sum = np.zeros(self.size, dtype=np.float64)
        rate_decay = math.exp(-cfg.dt_ms / max(cfg.tau_m_ms, cfg.dt_ms))
        steps_refractory = int(math.ceil(cfg.refractory_ms / cfg.dt_ms))
        spike_events: list[tuple[int, int]] = []

        for step in range(cfg.steps):
            recurrent_drive = filtered_rate / 1000.0
            population_inhibition = float(np.mean(filtered_rate)) / 1000.0
            current = (
                cfg.input_gain_pa * external
                + cfg.recurrent_gain_pa * recurrent_drive
                + cfg.recurrent_gain_pa * feedback_vector
                - cfg.inhibition_gain_pa * population_inhibition
            )
            active = refractory <= 0.0
            voltage[~active] = cfg.v_reset_mv
            refractory[~active] -= cfg.dt_ms
            dv = (cfg.dt_ms / cfg.tau_m_ms) * (
                cfg.v_rest_mv - voltage + current
            )
            voltage[active] += dv[active]
            spiked = active & (voltage >= cfg.v_threshold_mv)
            for neuron_index in np.flatnonzero(spiked):
                spike_events.append((step, int(neuron_index)))
            spike_counts[spiked] += 1
            voltage[spiked] = cfg.v_reset_mv
            refractory[spiked] = steps_refractory * cfg.dt_ms
            instantaneous_hz = spiked.astype(np.float64) * (1000.0 / cfg.dt_ms)
            filtered_rate = rate_decay * filtered_rate + (1.0 - rate_decay) * instantaneous_hz
            voltage_sum += voltage
            current_sum += current

        duration_ms = cfg.steps * cfg.dt_ms
        mean_rate = spike_counts.astype(np.float64) * (1000.0 / duration_ms)
        return {
            "population_size": self.size,
            "spike_counts": spike_counts.astype(int).tolist(),
            "spike_events": [{"step": step, "neuron": neuron} for step, neuron in spike_events],
            "mean_firing_rate_hz": float(np.mean(mean_rate)),
            "firing_rates_hz": mean_rate.tolist(),
            "mean_membrane_potential_mv": float(np.mean(voltage_sum / cfg.steps)),
            "final_membrane_potential_mv": voltage.tolist(),
            "mean_synaptic_current_pa": float(np.mean(current_sum / cfg.steps)),
            "duration_ms": float(duration_ms),
            "units": {"voltage": "mV", "current": "pA", "firing_rate": "Hz", "time": "ms"},
        }


class BiologicalReadingSimulator:
    """Run named reading-related LIF populations from functional activation."""

    LAYER_ORDER = (
        "orthographic", "lexical", "phonological", "auditory_wordform", "semantic", "control"
    )

    def __init__(self, config: LIFConfig | None = None) -> None:
        self.config = config or LIFConfig()

    @staticmethod
    def _prepare(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size == 0:
            vector = np.zeros(1, dtype=np.float64)
        if not np.isfinite(vector).all():
            raise ValueError("neural activation vectors must be finite")
        if np.max(vector) > 0:
            vector = vector / np.max(vector)
        return np.clip(vector, 0.0, 1.0)

    def run(self, drives: Mapping[str, np.ndarray | list[float] | tuple[float, ...]]) -> dict[str, object]:
        layers: dict[str, dict[str, object]] = {}
        previous_feedback: np.ndarray | None = None
        for index, name in enumerate(self.LAYER_ORDER):
            vector = self._prepare(drives.get(name, np.zeros(1)))
            feedback = None
            if previous_feedback is not None:
                feedback = np.array([float(np.mean(previous_feedback)) / 1000.0], dtype=np.float64)
            population = LIFPopulation(vector.size, self.config, seed=index)
            state = population.simulate(vector, feedback=feedback)
            layers[name] = state
            previous_feedback = np.asarray(state["firing_rates_hz"], dtype=np.float64)
        return {
            "model": "deterministic_current_based_LIF_population_reference",
            "layers": layers,
            "layer_order": list(self.LAYER_ORDER),
            "simulated_time_ms_per_layer": float(self.config.steps * self.config.dt_ms),
            "scope": (
                "Biologically inspired population simulation only; not a fitted, "
                "anatomically complete or empirically validated model of human reading."
            ),
        }
