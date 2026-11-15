"""Biologically motivated neuron, synapse, population, recurrence, and memory models."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .config import BNBEAConfig


@dataclass(slots=True)
class LIFNeuron:
    """Single leaky integrate-and-fire neuron with explicit SI-unit state.

    The membrane follows C_m dV/dt = g_L(E_L - V) + I_exc + I_inh + I_ext.
    A threshold crossing emits a spike, resets the voltage, and starts a
    refractory interval. This is a point-neuron approximation, not a
    compartmental model of a biological neuron.
    """

    voltage_v: float = -0.070
    refractory_remaining_s: float = 0.0
    capacitance_f: float = 200e-12
    leak_conductance_s: float = 10e-9
    resting_potential_v: float = -0.070
    threshold_v: float = -0.050
    reset_potential_v: float = -0.065
    refractory_period_s: float = 0.002

    def advance(
        self,
        dt_seconds: float,
        external_current_a: float = 0.0,
        excitatory_current_a: float = 0.0,
        inhibitory_current_a: float = 0.0,
    ) -> bool:
        """Advance one Euler time step and return whether a spike was emitted."""
        if not np.isfinite(dt_seconds) or dt_seconds <= 0:
            raise ValueError("dt_seconds must be finite and positive")
        if self.refractory_remaining_s > 0:
            self.refractory_remaining_s = max(0.0, self.refractory_remaining_s - dt_seconds)
            self.voltage_v = self.reset_potential_v
            return False
        total_current = external_current_a + excitatory_current_a + inhibitory_current_a
        derivative = (
            self.leak_conductance_s * (self.resting_potential_v - self.voltage_v)
            + total_current
        ) / self.capacitance_f
        self.voltage_v += dt_seconds * derivative
        if self.voltage_v >= self.threshold_v:
            self.voltage_v = self.reset_potential_v
            self.refractory_remaining_s = self.refractory_period_s
            return True
        return False


@dataclass(slots=True)
class NeuralPopulation:
    """Vectorized population of LIF neurons with rate and spike-history state."""

    name: str
    voltage_v: np.ndarray
    refractory_remaining_s: np.ndarray
    last_spikes: np.ndarray
    spike_history: list[np.ndarray] = field(default_factory=list)
    firing_rate_hz: float = 0.0
    mean_activity: float = 0.0

    @classmethod
    def initialize(
        cls, name: str, neuron_count: int, config: BNBEAConfig, rng: np.random.Generator
    ) -> "NeuralPopulation":
        """Create a population with resting voltages and zero refractory/spike state."""
        if neuron_count < 1:
            raise ValueError("neuron_count must be positive")
        voltage = config.resting_potential_v + rng.normal(
            0.0, config.initial_voltage_jitter_v, neuron_count
        )
        return cls(
            name=name,
            voltage_v=voltage.astype(np.float64),
            refractory_remaining_s=np.zeros(neuron_count, dtype=np.float64),
            last_spikes=np.zeros(neuron_count, dtype=np.float64),
        )

    @property
    def size(self) -> int:
        """Return the number of neurons in this population."""
        return int(self.voltage_v.size)

    def advance(
        self,
        current_a: np.ndarray,
        dt_seconds: float,
        config: BNBEAConfig,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Integrate population voltages, emit spikes, and update firing-rate state."""
        current = np.asarray(current_a, dtype=np.float64)
        if current.shape != (self.size,) or not np.isfinite(current).all():
            raise ValueError(f"current_a for {self.name} must be a finite vector of length {self.size}")
        refractory_mask = self.refractory_remaining_s > 0
        self.refractory_remaining_s[refractory_mask] = np.maximum(
            0.0, self.refractory_remaining_s[refractory_mask] - dt_seconds
        )
        active = self.refractory_remaining_s <= 0
        noise = (
            rng.normal(0.0, config.noise_std_a, self.size)
            if config.noise_std_a
            else 0.0
        )
        membrane_current = (
            config.leak_conductance_s * (config.resting_potential_v - self.voltage_v)
            + current + noise
        )
        self.voltage_v[active] += (
            dt_seconds * membrane_current[active] / config.membrane_capacitance_f
        )
        self.voltage_v[~active] = config.reset_potential_v
        spikes = (self.voltage_v >= config.firing_threshold_v) & active
        self.voltage_v[spikes] = config.reset_potential_v
        self.refractory_remaining_s[spikes] = config.refractory_seconds
        self.last_spikes = spikes.astype(np.float64)
        self.spike_history.append(self.last_spikes.copy())
        instantaneous_rate = float(np.sum(spikes) / (self.size * dt_seconds))
        self.firing_rate_hz = 0.85 * self.firing_rate_hz + 0.15 * instantaneous_rate
        self.mean_activity = 0.90 * self.mean_activity + 0.10 * float(np.mean(self.last_spikes))
        return self.last_spikes.copy()

    def reset(self, config: BNBEAConfig) -> None:
        """Return the population to its resting state and clear its spike history."""
        self.voltage_v.fill(config.resting_potential_v)
        self.refractory_remaining_s.fill(0.0)
        self.last_spikes.fill(0.0)
        self.spike_history.clear()
        self.firing_rate_hz = 0.0
        self.mean_activity = 0.0


@dataclass(slots=True)
class SynapticProjection:
    """Directed projection with conductance, eligibility traces, and a delay queue."""

    source: str
    target: str
    weights_s: np.ndarray
    synapse_type: str = "excitatory"
    tau_seconds: float = 0.006
    delay_steps: int = 1
    conductance_s: np.ndarray | None = None
    pre_eligibility: np.ndarray | None = None
    post_eligibility: np.ndarray | None = None
    pending_spikes: deque[np.ndarray] = field(default_factory=deque)

    def __post_init__(self) -> None:
        """Initialize conductance and plasticity traces to match the weight matrix."""
        if self.synapse_type not in {"excitatory", "inhibitory"}:
            raise ValueError("synapse_type must be excitatory or inhibitory")
        if self.delay_steps < 0 or self.tau_seconds <= 0:
            raise ValueError("delay_steps must be non-negative and tau_seconds positive")
        self.weights_s = np.asarray(self.weights_s, dtype=np.float64)
        if self.weights_s.ndim != 2 or not np.isfinite(self.weights_s).all():
            raise ValueError("weights_s must be a finite two-dimensional matrix")
        self.conductance_s = np.zeros_like(self.weights_s)
        self.pre_eligibility = np.zeros(self.weights_s.shape[1], dtype=np.float64)
        self.post_eligibility = np.zeros(self.weights_s.shape[0], dtype=np.float64)
        self.pending_spikes = deque(
            (np.zeros(self.weights_s.shape[1], dtype=np.float64)
             for _ in range(self.delay_steps)),
            maxlen=max(1, self.delay_steps + 1),
        )

    def delayed_input(self, spikes: np.ndarray) -> np.ndarray:
        """Return presynaptic spikes after the configured discrete delay."""
        current = np.asarray(spikes, dtype=np.float64)
        if current.shape != (self.weights_s.shape[1],):
            raise ValueError("presynaptic spike vector does not match projection width")
        if self.delay_steps == 0:
            return current.copy()
        if len(self.pending_spikes) < self.delay_steps:
            self.pending_spikes.append(current.copy())
            return np.zeros_like(current)
        delayed = self.pending_spikes.popleft()
        self.pending_spikes.append(current.copy())
        return delayed


class NeuralSimulationEngine:
    """Run distributed LIF populations over a recurrent excitatory/inhibitory graph.

    The default graph is a deliberately small reference circuit. Its node names
    refer to computational roles, not a claim of one-to-one anatomical mapping.
    """

    POPULATION_NAMES = (
        "sensory_retina_lgn_v1", "visual_hierarchy", "orthographic_vwfa",
        "language_association", "associative_memory", "predictive_context",
        "executive_control", "validation_observer", "ore_integrator",
    )

    def __init__(
        self,
        populations: Mapping[str, NeuralPopulation],
        projections: list[SynapticProjection],
        config: BNBEAConfig,
        seed: int | None = None,
    ) -> None:
        """Attach validated populations and projections to a simulation engine."""
        if not populations:
            raise ValueError("at least one neural population is required")
        self.populations = dict(populations)
        self.projections = list(projections)
        self.config = config
        self.rng = np.random.default_rng(config.seed if seed is None else seed)
        self.time_seconds = 0.0
        self.step_count = 0
        self.last_history: dict[str, np.ndarray] = {}
        for projection in self.projections:
            if projection.source not in self.populations or projection.target not in self.populations:
                raise ValueError("projection endpoints must be registered populations")
            source = self.populations[projection.source]
            target = self.populations[projection.target]
            if projection.weights_s.shape != (target.size, source.size):
                raise ValueError("projection weight shape must be [target_neuron, source_neuron]")

    @classmethod
    def build_default_network(cls, config: BNBEAConfig | None = None) -> "NeuralSimulationEngine":
        """Construct the nine-node recurrent reference circuit with seeded weights."""
        cfg = config or BNBEAConfig()
        rng = np.random.default_rng(cfg.seed)
        populations = {
            name: NeuralPopulation.initialize(name, cfg.neurons_per_population, cfg, rng)
            for name in cls.POPULATION_NAMES
        }
        edges = (
            ("sensory_retina_lgn_v1", "visual_hierarchy", "excitatory"),
            ("visual_hierarchy", "orthographic_vwfa", "excitatory"),
            ("orthographic_vwfa", "language_association", "excitatory"),
            ("language_association", "associative_memory", "excitatory"),
            ("associative_memory", "predictive_context", "excitatory"),
            ("predictive_context", "orthographic_vwfa", "excitatory"),
            ("language_association", "executive_control", "excitatory"),
            ("executive_control", "orthographic_vwfa", "inhibitory"),
            ("language_association", "validation_observer", "excitatory"),
            ("validation_observer", "ore_integrator", "excitatory"),
            ("executive_control", "ore_integrator", "excitatory"),
            ("ore_integrator", "predictive_context", "excitatory"),
        )
        projections: list[SynapticProjection] = []
        for source_name, target_name, synapse_type in edges:
            source = populations[source_name]
            target = populations[target_name]
            low, high = (0.30e-9, 0.90e-9) if synapse_type == "excitatory" else (0.10e-9, 0.45e-9)
            weights = rng.uniform(low, high, size=(target.size, source.size))
            projections.append(SynapticProjection(
                source_name, target_name, weights, synapse_type,
                tau_seconds=cfg.synaptic_tau_seconds, delay_steps=1,
            ))
        return cls(populations, projections, cfg)

    def advance(
        self,
        external_inputs: Mapping[str, np.ndarray | float],
        steps: int | None = None,
        dt_seconds: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Advance all populations synchronously and return [time, neuron] spike rasters."""
        count = self.config.simulation_steps if steps is None else int(steps)
        dt = self.config.dt_seconds if dt_seconds is None else float(dt_seconds)
        if count < 1 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("steps and dt_seconds must be positive")
        currents = {
            name: np.zeros(population.size, dtype=np.float64)
            for name, population in self.populations.items()
        }
        for name, value in external_inputs.items():
            if name not in self.populations:
                raise KeyError(f"unknown population: {name}")
            array = np.asarray(value, dtype=np.float64)
            if array.ndim == 0:
                array = np.full(self.populations[name].size, float(array))
            if array.shape != (self.populations[name].size,) or not np.isfinite(array).all():
                raise ValueError(f"external input for {name} must be a finite population vector")
            currents[name] += array
        history = {
            name: np.zeros((count, population.size), dtype=np.float32)
            for name, population in self.populations.items()
        }
        for tick in range(count):
            step_currents = {name: value.copy() for name, value in currents.items()}
            for projection in self.projections:
                source = self.populations[projection.source]
                target = self.populations[projection.target]
                delayed = projection.delayed_input(source.last_spikes)
                decay = np.exp(-dt / projection.tau_seconds)
                projection.conductance_s *= decay
                projection.conductance_s += projection.weights_s * delayed[None, :]
                np.clip(
                    projection.conductance_s, 0.0,
                    self.config.max_synaptic_conductance_s,
                    out=projection.conductance_s,
                )
                reversal = (
                    self.config.excitatory_reversal_v
                    if projection.synapse_type == "excitatory"
                    else self.config.inhibitory_reversal_v
                )
                driving_force = reversal - target.voltage_v[:, None]
                current = np.sum(projection.conductance_s * driving_force, axis=1)
                step_currents[projection.target] += current
            for name, population in self.populations.items():
                spikes = population.advance(step_currents[name], dt, self.config, self.rng)
                history[name][tick] = spikes
            self.time_seconds += dt
            self.step_count += 1
        self.last_history = history
        return {name: values.copy() for name, values in history.items()}

    def apply_learning(self) -> dict[str, float]:
        """Apply Hebbian and STDP-like updates, LTP/LTD, clipping, and homeostasis."""
        hebbian_total = 0.0
        potentiation_total = 0.0
        depression_total = 0.0
        for projection in self.projections:
            pre = self.populations[projection.source].last_spikes
            post = self.populations[projection.target].last_spikes
            pre_activity = self.populations[projection.source].mean_activity
            post_activity = self.populations[projection.target].mean_activity
            hebbian = self.config.hebbian_learning_rate * np.outer(
                np.full(post.size, post_activity), np.full(pre.size, pre_activity)
            ) * 1e-9
            potentiation = self.config.stdp_a_plus * np.outer(
                post, projection.pre_eligibility
            ) * 1e-9
            depression = self.config.stdp_a_minus * np.outer(
                projection.post_eligibility, pre
            ) * 1e-9
            projection.pre_eligibility *= np.exp(
                -self.config.dt_seconds / self.config.stdp_tau_pre_seconds
            )
            projection.post_eligibility *= np.exp(
                -self.config.dt_seconds / self.config.stdp_tau_post_seconds
            )
            projection.pre_eligibility += pre
            projection.post_eligibility += post
            delta_ltp = hebbian + potentiation
            delta_ltd = depression
            projection.weights_s += delta_ltp - delta_ltd
            projection.weights_s[:] = np.clip(
                projection.weights_s, 0.0, self.config.weight_max_s
            )
            hebbian_total += float(np.sum(np.abs(hebbian)))
            potentiation_total += float(np.sum(delta_ltp))
            depression_total += float(np.sum(delta_ltd))
        homeostatic_error = 0.0
        for projection in self.projections:
            target = self.populations[projection.target]
            scale = np.exp(np.clip(
                self.config.homeostatic_gain
                * (self.config.homeostatic_target_hz - target.firing_rate_hz),
                -0.1, 0.1,
            ))
            before = projection.weights_s.copy()
            projection.weights_s *= scale
            np.clip(projection.weights_s, 0.0, self.config.weight_max_s, out=projection.weights_s)
            homeostatic_error += float(np.mean(np.abs(projection.weights_s - before)))
        return {
            "hebbian_abs_delta_s": hebbian_total,
            "ltp_delta_s": potentiation_total,
            "ltd_delta_s": depression_total,
            "homeostatic_abs_adjustment_s": homeostatic_error,
        }

    def lesion(self, population_name: str, strength: float) -> None:
        """Apply a reversible gain reduction to the specified model population."""
        if population_name not in self.populations:
            raise KeyError(population_name)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("lesion strength must lie in [0, 1]")
        population = self.populations[population_name]
        population.last_spikes *= 1.0 - strength
        population.mean_activity *= 1.0 - strength
        population.firing_rate_hz *= 1.0 - strength

    def state_summary(self) -> dict[str, object]:
        """Summarize membrane, rate, spike, and synaptic state without raw arrays."""
        population_rows = {}
        for name, population in self.populations.items():
            population_rows[name] = {
                "neuron_count": population.size,
                "mean_voltage_v": float(np.mean(population.voltage_v)),
                "mean_firing_rate_hz": float(population.firing_rate_hz),
                "mean_activity": float(population.mean_activity),
                "spike_count": int(sum(np.sum(item) for item in population.spike_history)),
            }
        all_weights = np.concatenate([
            projection.weights_s.ravel() for projection in self.projections
        ]) if self.projections else np.asarray([], dtype=np.float64)
        return {
            "time_seconds": float(self.time_seconds),
            "steps": int(self.step_count),
            "population_count": len(self.populations),
            "projection_count": len(self.projections),
            "populations": population_rows,
            "synaptic_weight_mean_s": float(np.mean(all_weights)) if all_weights.size else 0.0,
            "synaptic_weight_max_s": float(np.max(all_weights)) if all_weights.size else 0.0,
            "model_scope": "reduced LIF/conductance network; not a cellular brain replica",
        }


class AssociativeMemory:
    """Two-timescale associative store with fast traces and slow consolidated traces."""

    def __init__(self, fast_decay: float = 0.92, consolidation_rate: float = 0.10) -> None:
        """Initialize empty fast and slow associative stores."""
        if not 0.0 <= fast_decay <= 1.0 or not 0.0 <= consolidation_rate <= 1.0:
            raise ValueError("memory rates must lie in [0, 1]")
        self.fast_decay = fast_decay
        self.consolidation_rate = consolidation_rate
        self.fast: dict[str, np.ndarray] = {}
        self.slow: dict[str, np.ndarray] = {}
        self.replay_log: list[str] = []

    def encode(self, key: str, pattern: np.ndarray) -> None:
        """Bind a named experience to a normalized fast associative pattern."""
        vector = np.asarray(pattern, dtype=np.float64)
        if not key or vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError("memory key and finite non-empty pattern are required")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("cannot store a zero-norm associative pattern")
        self.fast[key] = vector / norm

    def retrieve(self, cue: np.ndarray, top_k: int = 3) -> list[dict[str, float | str]]:
        """Retrieve ranked fast/slow patterns by cosine similarity to a partial cue."""
        vector = np.asarray(cue, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if vector.ndim != 1 or norm <= 1e-12 or not np.isfinite(vector).all():
            return []
        candidates: list[dict[str, float | str]] = []
        for store_name, store in (("fast", self.fast), ("slow", self.slow)):
            for key, pattern in store.items():
                if pattern.shape != vector.shape:
                    continue
                score = float(np.dot(pattern, vector / norm))
                candidates.append({"key": key, "store": store_name, "similarity": score})
        return sorted(candidates, key=lambda row: float(row["similarity"]), reverse=True)[:top_k]

    def replay(self, key: str) -> np.ndarray | None:
        """Reactivate a stored pattern and return a detached replay vector."""
        pattern = self.fast.get(key, self.slow.get(key))
        if pattern is None:
            return None
        self.replay_log.append(key)
        return pattern.copy()

    def consolidate(self, key: str) -> bool:
        """Blend a fast trace into the slower store using the consolidation rate."""
        if key not in self.fast:
            return False
        if key in self.slow:
            combined = (
                (1.0 - self.consolidation_rate) * self.slow[key]
                + self.consolidation_rate * self.fast[key]
            )
        else:
            combined = self.fast[key].copy()
        norm = float(np.linalg.norm(combined))
        self.slow[key] = combined / max(norm, 1e-12)
        return True

    def decay_fast_memory(self) -> None:
        """Apply gradual accessibility decay to fast traces, retaining stored keys."""
        for key, value in list(self.fast.items()):
            self.fast[key] = value * self.fast_decay

    def summary(self) -> dict[str, object]:
        """Return counts and keys for the two memory timescales."""
        return {
            "fast_trace_count": len(self.fast),
            "slow_trace_count": len(self.slow),
            "replay_count": len(self.replay_log),
            "fast_keys": sorted(self.fast),
            "slow_keys": sorted(self.slow),
        }


def project_drive(values: np.ndarray, population_size: int, gain_a: float) -> np.ndarray:
    """Map a normalized functional vector onto a population's external-current inputs."""
    vector = np.asarray(values, dtype=np.float64).ravel()
    if not vector.size or not np.isfinite(vector).all():
        vector = np.zeros(1, dtype=np.float64)
    repeated = np.resize(np.clip(vector, 0.0, 1.0), population_size)
    return repeated * gain_a


def phase_coherence(phases_radians: np.ndarray) -> float:
    """Compute the phase-locking value |mean(exp(i theta))| across oscillators."""
    phases = np.asarray(phases_radians, dtype=np.float64)
    if phases.ndim != 1 or phases.size == 0 or not np.isfinite(phases).all():
        raise ValueError("phases_radians must be a finite non-empty vector")
    return float(np.abs(np.mean(np.exp(1j * phases))))


__all__ = [
    "LIFNeuron", "NeuralPopulation", "SynapticProjection",
    "NeuralSimulationEngine", "AssociativeMemory", "project_drive",
    "phase_coherence",
]
