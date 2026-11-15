"""Validated configuration for the Master Biological Neurocomputational ORE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class BNBEAConfig:
    """Numerical controls for the functional/biological two-track architecture.

    Values are simulation defaults, not measurements of a particular human
    neuron. SI units are used for membrane and conductance quantities.
    """

    dt_seconds: float = 0.001
    simulation_steps: int = 24
    neurons_per_population: int = 12
    membrane_capacitance_f: float = 200e-12
    leak_conductance_s: float = 10e-9
    resting_potential_v: float = -0.070
    firing_threshold_v: float = -0.050
    reset_potential_v: float = -0.065
    excitatory_reversal_v: float = 0.0
    inhibitory_reversal_v: float = -0.080
    refractory_seconds: float = 0.002
    synaptic_tau_seconds: float = 0.006
    external_current_gain_a: float = 2.5e-10
    noise_std_a: float = 0.0
    initial_voltage_jitter_v: float = 0.001
    max_synaptic_conductance_s: float = 8e-9
    hebbian_learning_rate: float = 0.002
    stdp_a_plus: float = 0.0015
    stdp_a_minus: float = 0.0012
    stdp_tau_pre_seconds: float = 0.020
    stdp_tau_post_seconds: float = 0.020
    homeostatic_target_hz: float = 5.0
    homeostatic_gain: float = 0.002
    weight_max_s: float = 8e-9
    seed: int = 17
    epsilon: float = 1e-8
    strict_numerics: bool = True

    def __post_init__(self) -> None:
        """Reject invalid time, electrical, learning, and numerical parameters."""
        positive = (
            "dt_seconds", "simulation_steps", "neurons_per_population",
            "membrane_capacitance_f", "leak_conductance_s",
            "refractory_seconds", "synaptic_tau_seconds",
            "external_current_gain_a", "max_synaptic_conductance_s",
            "stdp_tau_pre_seconds", "stdp_tau_post_seconds", "weight_max_s",
            "epsilon",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        integer_fields = ("simulation_steps", "neurons_per_population", "seed")
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        finite_fields = (
            "resting_potential_v", "firing_threshold_v", "reset_potential_v",
            "excitatory_reversal_v", "inhibitory_reversal_v", "noise_std_a",
            "initial_voltage_jitter_v", "hebbian_learning_rate", "stdp_a_plus",
            "stdp_a_minus", "homeostatic_target_hz", "homeostatic_gain",
        )
        if any(not isfinite(float(getattr(self, name))) for name in finite_fields):
            raise ValueError("all electrical and learning parameters must be finite")
        if self.firing_threshold_v <= self.reset_potential_v:
            raise ValueError("firing_threshold_v must exceed reset_potential_v")
        if self.noise_std_a < 0 or self.initial_voltage_jitter_v < 0:
            raise ValueError("noise and initial-voltage jitter must be non-negative")
        if self.hebbian_learning_rate < 0 or self.stdp_a_plus < 0 or self.stdp_a_minus < 0:
            raise ValueError("plasticity rates must be non-negative")
        if self.homeostatic_gain < 0 or self.homeostatic_target_hz < 0:
            raise ValueError("homeostatic parameters must be non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not isinstance(self.strict_numerics, bool):
            raise ValueError("strict_numerics must be a boolean")

    @property
    def membrane_time_constant_seconds(self) -> float:
        """Return tau_m = C_m / g_L, the passive membrane time constant."""
        return self.membrane_capacitance_f / self.leak_conductance_s

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible snapshot of numerical controls."""
        return asdict(self)
