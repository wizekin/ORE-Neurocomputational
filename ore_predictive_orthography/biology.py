"""Current-based LIF reference network for seven predictive-orthography stages.

This biological reference is not an anatomically fitted model. Its populations are
computational stand-ins for interacting sensory, association, attentional,
predictive and error-sensitive neural pools.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Mapping
import numpy as np

BIOLOGICAL_STAGE_NAMES: tuple[str, ...] = (
    "bottom_up_character_evidence",
    "orthographic_transition_memory",
    "neighbor_prediction",
    "attention_and_sensory_gain",
    "prior_precision_gain",
    "posterior_settling",
    "prediction_error_diagnostics",
)

@dataclass(frozen=True, slots=True)
class LIFConfig:
    """Units: ms, mV, pA and Hz; values are configurable hypotheses, not measured constants."""
    neuron_count: int = 24
    time_step_ms: float = 1.0
    simulation_steps: int = 120
    membrane_time_constant_ms: float = 20.0
    refractory_period_ms: float = 2.0
    resting_potential_mV: float = -70.0
    reset_potential_mV: float = -68.0
    threshold_potential_mV: float = -52.0
    membrane_resistance_GOhm: float = 0.10
    baseline_current_pA: float = 8.0
    sensory_drive_gain_pA: float = 220.0
    interpopulation_gain_pA: float = 26.0
    recurrent_gain_pA: float = 10.0
    global_inhibition_gain_pA: float = 14.0
    adaptation_increment_pA: float = 8.0
    adaptation_time_constant_ms: float = 120.0
    rate_time_constant_ms: float = 18.0
    maximum_rate_hz: float = 100.0

    def __post_init__(self) -> None:
        if self.neuron_count < 2 or self.simulation_steps < 1:
            raise ValueError("neuron_count must be >=2 and simulation_steps must be positive")
        for name in ("time_step_ms", "membrane_time_constant_ms", "membrane_resistance_GOhm",
                     "adaptation_time_constant_ms", "rate_time_constant_ms", "maximum_rate_hz"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("refractory_period_ms", "baseline_current_pA", "sensory_drive_gain_pA",
                     "interpopulation_gain_pA", "recurrent_gain_pA", "global_inhibition_gain_pA",
                     "adaptation_increment_pA"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not (self.resting_potential_mV < self.threshold_potential_mV
                and self.reset_potential_mV < self.threshold_potential_mV):
            raise ValueError("threshold_potential_mV must exceed resting and reset potentials")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

def _project_signal(values: object, neuron_count: int) -> np.ndarray:
    """Map a finite stage signal to a bounded population drive."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return np.zeros(neuron_count, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("biological stage signals must contain only finite values")
    magnitude = np.abs(array)
    scale = float(np.max(magnitude))
    normalized = np.zeros_like(magnitude) if scale <= 1e-12 else np.clip(magnitude / scale, 0.0, 1.0)
    return np.interp(np.linspace(0.0, 1.0, neuron_count),
                     np.linspace(0.0, 1.0, normalized.size), normalized)

def _connectivity() -> np.ndarray:
    """Rows are source pools; columns are target pools."""
    weights = np.zeros((7, 7), dtype=np.float64)
    for source in range(6):
        weights[source, source + 1] = 0.55
    weights[5, 2] = 0.24
    weights[6, 3] = 0.28
    weights[6, 0] = 0.12
    weights[5, 1] = 0.10
    return weights

def simulate_lif_reference(
    stage_signals: Mapping[str, object], config: LIFConfig | None = None
) -> dict[str, object]:
    """Simulate seven interacting populations without writing back to functional beliefs."""
    cfg = config or LIFConfig()
    missing = [name for name in BIOLOGICAL_STAGE_NAMES if name not in stage_signals]
    if missing:
        raise ValueError(f"stage_signals is missing biological populations: {missing}")
    count = cfg.neuron_count
    population_count = len(BIOLOGICAL_STAGE_NAMES)
    drives = np.stack([_project_signal(stage_signals[name], count) for name in BIOLOGICAL_STAGE_NAMES])
    connectivity = _connectivity()
    voltage = np.full((population_count, count), cfg.resting_potential_mV, dtype=np.float64)
    refractory = np.zeros((population_count, count), dtype=np.float64)
    adaptation = np.zeros((population_count, count), dtype=np.float64)
    rate_state = np.zeros((population_count, count), dtype=np.float64)
    spike_counts = np.zeros((population_count, count), dtype=np.int64)
    final_current = np.zeros((population_count, count), dtype=np.float64)
    trace: list[dict[str, object]] = []
    dt = cfg.time_step_ms
    membrane_decay = dt / cfg.membrane_time_constant_ms
    adaptation_decay = float(np.exp(-dt / cfg.adaptation_time_constant_ms))
    rate_decay = float(np.exp(-dt / cfg.rate_time_constant_ms))

    for step in range(cfg.simulation_steps):
        activity = np.clip(rate_state / cfg.maximum_rate_hz, 0.0, 1.0)
        mean_activity = activity.mean(axis=1)
        incoming = connectivity.T @ mean_activity
        external = cfg.baseline_current_pA + cfg.sensory_drive_gain_pA * drives
        feedforward = cfg.interpopulation_gain_pA * incoming[:, None]
        recurrent = cfg.recurrent_gain_pA * activity
        inhibition = cfg.global_inhibition_gain_pA * mean_activity[:, None]
        current = external + feedforward + recurrent - inhibition - adaptation
        final_current = current
        refractory = np.maximum(0.0, refractory - dt)
        available = refractory <= 0.0
        voltage[available] += membrane_decay * (
            cfg.resting_potential_mV - voltage[available]
            + cfg.membrane_resistance_GOhm * current[available]
        )
        voltage[~available] = cfg.reset_potential_mV
        spikes = available & (voltage >= cfg.threshold_potential_mV)
        spike_counts += spikes.astype(np.int64)
        voltage[spikes] = cfg.reset_potential_mV
        refractory[spikes] = cfg.refractory_period_ms
        adaptation = adaptation * adaptation_decay + spikes.astype(np.float64) * cfg.adaptation_increment_pA
        rate_state = rate_state * rate_decay + spikes.astype(np.float64) * (1.0 - rate_decay) * cfg.maximum_rate_hz

        if (step + 1) % max(1, cfg.simulation_steps // 20) == 0 or step == cfg.simulation_steps - 1:
            trace.append({
                "step": step + 1,
                "time_ms": float((step + 1) * dt),
                "mean_population_rate_hz": [float(x) for x in rate_state.mean(axis=1)],
                "population_spike_counts": [int(x) for x in spike_counts.sum(axis=1)],
            })

    duration_seconds = cfg.simulation_steps * dt / 1000.0
    populations: dict[str, object] = {}
    for index, name in enumerate(BIOLOGICAL_STAGE_NAMES):
        rates = spike_counts[index].astype(np.float64) / duration_seconds
        populations[name] = {
            "neuron_count": count,
            "spike_counts": spike_counts[index].astype(int).tolist(),
            "firing_rates_hz": rates.astype(float).tolist(),
            "mean_firing_rate_hz": float(rates.mean()),
            "mean_membrane_potential_mV": float(voltage[index].mean()),
            "mean_synaptic_current_pA": float(final_current[index].mean()),
            "mean_adaptation_current_pA": float(adaptation[index].mean()),
        }
    finite = all(np.isfinite(x).all() for x in (voltage, adaptation, rate_state, final_current, drives))
    return {
        "enabled": True,
        "model": "seven-pool current-based leaky-integrate-and-fire reference",
        "stage_names": list(BIOLOGICAL_STAGE_NAMES),
        "populations": populations,
        "interpopulation_connectivity": connectivity.tolist(),
        "population_trace": trace,
        "simulated_time_ms": float(cfg.simulation_steps * dt),
        "time_step_ms": float(dt),
        "units": {"voltage": "mV", "current": "pA", "rate": "Hz", "time": "ms"},
        "config": cfg.to_dict(),
        "numerically_valid": bool(finite),
        "scientific_status": "generic LIF reference, not anatomically mapped, fitted or validated against human reading data",
    }
