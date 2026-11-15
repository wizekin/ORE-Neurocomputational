"""Operation 3 biological-reference track: recurrent current-based leaky integrate-and-fire populations."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .contracts import BNBEAConfig


POPULATIONS = (
    "retina_lgn", "v1", "v2", "v4", "vwfa_posterior", "vwfa_anterior",
    "mtg", "ifg", "atl", "stg_ventral", "stg_dorsal", "parietal_dorsal",
)

_CONNECTIONS = {
    "retina_lgn": {"v1": 0.18},
    "v1": {"v2": 0.18, "retina_lgn": 0.04},
    "v2": {"v1": 0.05, "v4": 0.16},
    "v4": {"v2": 0.05, "vwfa_posterior": 0.16, "parietal_dorsal": 0.08},
    "vwfa_posterior": {"v4": 0.05, "vwfa_anterior": 0.18, "mtg": 0.10},
    "vwfa_anterior": {"vwfa_posterior": 0.05, "mtg": 0.13},
    "mtg": {"vwfa_anterior": 0.06, "ifg": 0.12, "atl": 0.12, "stg_ventral": 0.05},
    "ifg": {"mtg": 0.08, "atl": 0.06, "parietal_dorsal": 0.05},
    "atl": {"mtg": 0.08, "ifg": 0.04},
    "stg_ventral": {"mtg": 0.12, "atl": 0.07},
    "stg_dorsal": {"ifg": 0.10, "parietal_dorsal": 0.10},
    "parietal_dorsal": {"ifg": 0.07, "stg_dorsal": 0.08, "v4": 0.04},
}


class NeuralDynamicsSimulator:
    """Simulate coupled LIF populations and retain a time series for later measurement models.

    The model is deterministic, current-based and population-level. Labels are functional network
    proxies and must not be read as a fully specified anatomical circuit or a fitted human brain.
    """

    def __init__(self, config: BNBEAConfig | None = None) -> None:
        """Allocate voltage, refractory, spike, synaptic, activity and recurrent state arrays."""
        self.config = config or BNBEAConfig()
        size = self.config.lif_population_size
        self.voltage_mv = {name: np.full(size, self.config.rest_mv) for name in POPULATIONS}
        self.refractory_ms = {name: np.zeros(size) for name in POPULATIONS}
        self.spike_counts = {name: np.zeros(size, dtype=np.int64) for name in POPULATIONS}
        self.synaptic_activity = {name: np.zeros(size) for name in POPULATIONS}
        self.last_current_pa = {name: np.zeros(size) for name in POPULATIONS}
        self.activity = {name: 0.0 for name in POPULATIONS}
        self.time_ms = 0.0
        self.history: list[dict[str, float]] = []
        self.synaptic_history: list[dict[str, float]] = []
        self.spike_history: list[dict[str, int]] = []

    def _advance_one_step(self, drives: Mapping[str, float]) -> None:
        """Integrate one Euler time step with recurrent excitation, global inhibition and spikes."""
        cfg = self.config
        global_activity = float(np.mean(tuple(self.activity.values())))
        updated_activity: dict[str, float] = {}
        for name in POPULATIONS:
            recurrent = sum(weight * self.activity[source] for source, weight in _CONNECTIONS[name].items())
            external = float(np.clip(drives.get(name, 0.0), 0.0, 1.0))
            current_scalar = (
                cfg.current_gain_pa * external
                + cfg.recurrent_gain_pa * (recurrent + self.activity[name])
                - cfg.inhibition_gain_pa * global_activity
            )
            current = np.full(cfg.lif_population_size, current_scalar)
            self.last_current_pa[name] = current
            voltage = self.voltage_mv[name]
            refractory = self.refractory_ms[name]
            active = refractory <= 0.0
            voltage[~active] = cfg.reset_mv
            refractory[~active] = np.maximum(0.0, refractory[~active] - cfg.dt_ms)
            voltage[active] += (cfg.dt_ms / cfg.membrane_tau_ms) * (
                cfg.rest_mv - voltage[active]
                + 0.001 * cfg.membrane_resistance_mohm * current[active]
            )
            spiked = active & (voltage >= cfg.threshold_mv)
            if np.any(spiked):
                self.spike_counts[name][spiked] += 1
                voltage[spiked] = cfg.reset_mv
                refractory[spiked] = cfg.refractory_ms
            instant_rate = float(np.count_nonzero(spiked)) / cfg.lif_population_size
            decay = math.exp(-cfg.dt_ms / max(cfg.membrane_tau_ms, cfg.dt_ms))
            updated_activity[name] = decay * self.activity[name] + (1.0 - decay) * instant_rate
            self.synaptic_activity[name] = (
                decay * self.synaptic_activity[name] + (1.0 - decay) * current
            )
        self.activity.update(updated_activity)
        self.time_ms += cfg.dt_ms
        self.history.append(dict(self.activity))
        self.synaptic_history.append({name: float(np.mean(self.synaptic_activity[name])) for name in POPULATIONS})
        self.spike_history.append({name: int(np.sum(values)) for name, values in self.spike_counts.items()})

    def advance(self, drives: Mapping[str, float], steps: int = 1) -> dict[str, object]:
        """Advance all neural populations for a fixed number of steps and return a population snapshot."""
        if steps < 1:
            raise ValueError("steps must be positive")
        unknown = set(drives) - set(POPULATIONS)
        if unknown:
            raise KeyError(f"unknown neural populations: {sorted(unknown)}")
        for value in drives.values():
            if not math.isfinite(float(value)):
                raise ValueError("neural population drives must be finite")
        for _ in range(steps):
            self._advance_one_step(drives)
        return self.snapshot()

    def run_schedule(self, schedule: Sequence[Mapping[str, float]]) -> np.ndarray:
        """Run a sequence of population drives and return [time, population] activity history."""
        for drive in schedule:
            self.advance(drive, steps=1)
        if not self.history:
            return np.zeros((0, len(POPULATIONS)), dtype=np.float64)
        return np.asarray([[row[name] for name in POPULATIONS] for row in self.history])

    def snapshot(self) -> dict[str, object]:
        """Summarize rates, voltages, synaptic currents, spike counts and simulation time."""
        elapsed_seconds = max(self.time_ms / 1000.0, 1e-9)
        populations: dict[str, dict[str, float | int]] = {}
        for name in POPULATIONS:
            populations[name] = {
                "mean_firing_rate_hz": float(np.sum(self.spike_counts[name]) / self.config.lif_population_size / elapsed_seconds),
                "spike_count": int(np.sum(self.spike_counts[name])),
                "mean_voltage_mv": float(np.mean(self.voltage_mv[name])),
                "mean_current_pa": float(np.mean(self.last_current_pa[name])),
                "mean_synaptic_current_pa": float(np.mean(self.synaptic_activity[name])),
                "mean_activity": float(self.activity[name]),
            }
        return {
            "model": "BNBEA_current_based_LIF_population_reference",
            "time_ms": float(self.time_ms),
            "populations": populations,
            "population_order": list(POPULATIONS),
            "units": {"voltage": "mV", "current": "pA", "time": "ms", "rate": "Hz"},
            "scope": (
                "Configurable mathematical reference model; not a fitted cellular simulation, "
                "not a complete anatomical connectome and not biological validation."
            ),
        }

    def source_time_series(self) -> np.ndarray:
        """Return a time-by-source matrix of mean synaptic-current proxies for EEG/MEG forward models.

        The current values are population-level model variables, not reconstructed cortical dipole
        moments. A valid sensor prediction still requires an appropriate source/lead-field model.
        """
        if not self.synaptic_history:
            return np.zeros((0, len(POPULATIONS)), dtype=np.float64)
        return np.asarray([[row[name] for name in POPULATIONS] for row in self.synaptic_history], dtype=np.float64)

    def activity_time_series(self) -> np.ndarray:
        """Return a time-by-population activity matrix for regional summaries and hemodynamic proxies."""
        if not self.history:
            return np.zeros((0, len(POPULATIONS)), dtype=np.float64)
        return np.asarray([[row[name] for name in POPULATIONS] for row in self.history], dtype=np.float64)


def normalized_drive(value: float) -> float:
    """Map a finite scalar feature strength to the bounded population-drive interval [0, 1]."""
    if not math.isfinite(float(value)):
        raise ValueError("drive must be finite")
    return float(np.clip(value, 0.0, 1.0))
