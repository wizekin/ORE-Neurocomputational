"""Current-based leaky integrate-and-fire reference for distributed populations.

This module simulates population-level neuron dynamics as a biological reference
track. It is separate from the functional semantic feature states and is not
fitted to human recordings or presented as a cellularly accurate brain model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import POPULATIONS


@dataclass(frozen=True)
class DistributedLIFConfig:
    """Configure simplified current-based LIF neurons with explicit physical units."""

    neuron_count: int = 32
    dt_ms: float = 0.5
    steps_per_stage: int = 40
    membrane_tau_ms: float = 20.0
    membrane_resistance_mohm: float = 100.0
    rest_mv: float = -65.0
    threshold_mv: float = -50.0
    reset_mv: float = -65.0
    refractory_ms: float = 2.0
    input_gain_pa: float = 320.0
    recurrent_current_pa: float = 35.0
    cross_population_current_pa: float = 80.0
    inhibitory_current_pa: float = 24.0
    synaptic_noise_pa: float = 0.25

    def validate(self) -> None:
        """Reject invalid neuron counts, time steps, membrane parameters, or current gains."""
        if self.neuron_count < 1 or self.steps_per_stage < 1:
            raise ValueError("neuron_count and steps_per_stage must be positive.")
        if self.dt_ms <= 0.0 or self.membrane_tau_ms <= 0.0:
            raise ValueError("LIF time step and membrane time constant must be positive.")
        if self.membrane_resistance_mohm <= 0.0:
            raise ValueError("Membrane resistance must be positive.")
        if self.threshold_mv <= self.reset_mv:
            raise ValueError("LIF threshold_mv must exceed reset_mv.")
        numeric = (
            self.dt_ms,
            self.membrane_tau_ms,
            self.membrane_resistance_mohm,
            self.rest_mv,
            self.threshold_mv,
            self.reset_mv,
            self.refractory_ms,
            self.input_gain_pa,
            self.recurrent_current_pa,
            self.cross_population_current_pa,
            self.inhibitory_current_pa,
            self.synaptic_noise_pa,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Every LIF configuration value must be finite.")


class DistributedLIFSimulator:
    """Simulate independent leaky-integrate-and-fire neuron groups for each population."""

    def __init__(
        self,
        config: DistributedLIFConfig | None = None,
        *,
        populations: Sequence[str] = POPULATIONS,
        random_seed: int = 17,
    ) -> None:
        """Allocate membrane voltage, refractory state, spikes, and reproducible noise."""
        self.config = config or DistributedLIFConfig()
        self.config.validate()
        self.populations = tuple(populations)
        if not self.populations or len(set(self.populations)) != len(self.populations):
            raise ValueError("Population names must be non-empty and unique.")
        self._index = {name: index for index, name in enumerate(self.populations)}
        self._rng = np.random.default_rng(random_seed)
        size = self.config.neuron_count
        self.voltage_mv = {
            name: self.config.rest_mv + self._rng.normal(0.0, 0.2, size)
            for name in self.populations
        }
        self.refractory_ms = {
            name: np.zeros(size, dtype=np.float64) for name in self.populations
        }
        self.spike_counts = {
            name: np.zeros(size, dtype=np.int64) for name in self.populations
        }
        self.activity_fraction = {name: 0.0 for name in self.populations}
        self.last_current_pa = {
            name: np.zeros(size, dtype=np.float64) for name in self.populations
        }
        self.time_ms = 0.0
        self.stage_count = 0
        self.operation_trace: list[dict[str, Any]] = []

    def step(
        self,
        stage_id: str,
        population_drives: Mapping[str, float],
        connectivity: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Advance all neuron groups for one semantic stage using Euler membrane integration."""
        unknown = set(population_drives) - set(self.populations)
        if unknown:
            raise KeyError(f"Unknown biological populations: {sorted(unknown)}")
        drives = np.asarray([
            float(np.clip(population_drives.get(name, 0.0), 0.0, 1.0))
            for name in self.populations
        ], dtype=np.float64)
        if not np.all(np.isfinite(drives)):
            raise ValueError("Biological population drives must be finite.")
        if connectivity is None:
            connections = np.zeros((len(self.populations), len(self.populations)), dtype=np.float64)
        else:
            connections = np.asarray(connectivity, dtype=np.float64)
            expected = (len(self.populations), len(self.populations))
            if connections.shape != expected or not np.all(np.isfinite(connections)):
                raise ValueError(f"connectivity must be a finite matrix with shape {expected}.")
        dt = self.config.dt_ms
        tau = self.config.membrane_tau_ms
        decay = math.exp(-dt / 20.0)
        for _ in range(self.config.steps_per_stage):
            previous_activity = np.asarray([
                self.activity_fraction[name] for name in self.populations
            ], dtype=np.float64)
            cross_population_drive = connections @ previous_activity
            next_activity: dict[str, float] = {}
            for index, name in enumerate(self.populations):
                voltage = self.voltage_mv[name]
                refractory = self.refractory_ms[name]
                recurrent = self.config.recurrent_current_pa * previous_activity[index]
                cross = self.config.cross_population_current_pa * cross_population_drive[index]
                inhibitory = self.config.inhibitory_current_pa * previous_activity[index]
                noise = self._rng.normal(
                    0.0, self.config.synaptic_noise_pa, size=self.config.neuron_count
                )
                current_pa = np.maximum(
                    0.0,
                    self.config.input_gain_pa * drives[index]
                    + recurrent + cross - inhibitory + noise,
                )
                self.last_current_pa[name][:] = current_pa
                active = refractory <= 0.0
                voltage_drive_mv = (
                    self.config.membrane_resistance_mohm * current_pa * 0.001
                )
                derivative_mv = (self.config.rest_mv - voltage + voltage_drive_mv) / tau
                voltage[active] += dt * derivative_mv[active]
                refractory[:] = np.maximum(0.0, refractory - dt)
                spiked = active & (voltage >= self.config.threshold_mv)
                if np.any(spiked):
                    self.spike_counts[name][spiked] += 1
                    voltage[spiked] = self.config.reset_mv
                    refractory[spiked] = self.config.refractory_ms
                voltage[~active] = self.config.reset_mv
                instantaneous_fraction = float(np.count_nonzero(spiked)) / self.config.neuron_count
                next_activity[name] = (
                    decay * self.activity_fraction[name]
                    + (1.0 - decay) * instantaneous_fraction
                )
            self.activity_fraction.update(next_activity)
            self.time_ms += dt
        self.stage_count += 1
        snapshot = self.snapshot()
        event = {
            "stage_id": stage_id,
            "simulated_time_ms": float(self.time_ms),
            "population_activity": {
                name: snapshot["populations"][name]
                for name in self.populations
            },
            "mean_network_activity_fraction": float(np.mean(tuple(self.activity_fraction.values()))),
            "units": snapshot["units"],
        }
        self.operation_trace.append(event)
        return event

    def snapshot(self) -> dict[str, Any]:
        """Return membrane, current, spike, and firing-rate metrics for every population."""
        elapsed_seconds = max(self.time_ms / 1000.0, 1e-9)
        populations: dict[str, dict[str, float | int]] = {}
        for name in self.populations:
            spikes = int(np.sum(self.spike_counts[name]))
            neuron_count = int(self.spike_counts[name].size)
            populations[name] = {
                "mean_membrane_voltage_mv": float(np.mean(self.voltage_mv[name])),
                "mean_synaptic_current_pa": float(np.mean(self.last_current_pa[name])),
                "spike_count": spikes,
                "mean_firing_rate_hz": float(spikes / (neuron_count * elapsed_seconds)),
                "activity_fraction": float(self.activity_fraction[name]),
            }
        finite = all(
            np.all(np.isfinite(values))
            for values in (*self.voltage_mv.values(), *self.last_current_pa.values())
        )
        return {
            "enabled": True,
            "model": "current_based_leaky_integrate_and_fire",
            "populations": populations,
            "stage_count": int(self.stage_count),
            "simulated_time_ms": float(self.time_ms),
            "units": {
                "membrane_voltage": "mV",
                "synaptic_current": "pA",
                "firing_rate": "Hz",
                "simulated_time": "ms",
                "spike_count": "count",
            },
            "configuration": asdict(self.config),
            "numerically_valid": bool(finite),
            "scope": (
                "Simplified LIF population reference, not fitted to human neural recordings "
                "and not a disease or cellularly detailed simulation."
            ),
        }
