"""Explicit LIF population reference for the six-stage visual-word-form pathway."""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .contracts import VWFAConfig


POPULATIONS = (
    "retina_lgn", "v1", "v2", "v4", "votc_posterior", "vwfa_posterior",
    "vwfa_intermediate", "vwfa_anterior", "ifg", "mtg", "atl", "stg_ventral",
    "angular_gyrus", "parietal_attention", "phonological_association", "orthographic_memory",
)

# Target -> presynaptic source -> effective signed connection. Values are dimensionless
# population-coupling hypotheses, not tract-strength measurements.
_BASE_CONNECTIONS = {
    "retina_lgn": {"v1": 0.20},
    "v1": {"retina_lgn": 0.06, "v2": 0.18},
    "v2": {"v1": 0.04, "v4": 0.18},
    "v4": {"v2": 0.04, "votc_posterior": 0.17, "parietal_attention": 0.08},
    "votc_posterior": {"v4": 0.04, "vwfa_posterior": 0.18},
    "vwfa_posterior": {"votc_posterior": 0.05, "vwfa_intermediate": 0.18, "mtg": 0.08},
    "vwfa_intermediate": {"vwfa_posterior": 0.08, "vwfa_anterior": 0.17, "ifg": 0.04},
    "vwfa_anterior": {"vwfa_intermediate": 0.06, "mtg": 0.13, "atl": 0.08, "ifg": 0.05},
    "ifg": {"vwfa_anterior": 0.06, "mtg": 0.09, "parietal_attention": 0.05},
    "mtg": {"vwfa_anterior": 0.08, "ifg": 0.07, "atl": 0.12, "stg_ventral": 0.05},
    "atl": {"mtg": 0.08, "angular_gyrus": 0.06},
    "stg_ventral": {"mtg": 0.08, "phonological_association": 0.13},
    "angular_gyrus": {"atl": 0.06, "ifg": 0.05, "parietal_attention": 0.05},
    "parietal_attention": {"v4": 0.04, "ifg": 0.08, "angular_gyrus": 0.05},
    "phonological_association": {"stg_ventral": 0.08, "vwfa_anterior": 0.07},
    "orthographic_memory": {"vwfa_anterior": 0.10, "vwfa_intermediate": 0.06},
}

_OPERATION_DRIVES = {
    1: ("retina_lgn",), 2: ("v1",), 3: ("v1", "v2"), 4: ("v2", "v4"),
    5: ("v1", "v2"), 6: ("v2", "v4"), 7: ("v4", "parietal_attention"),
    8: ("votc_posterior", "vwfa_posterior"), 9: ("vwfa_posterior",),
    10: ("vwfa_intermediate",), 11: ("vwfa_anterior",), 12: ("vwfa_anterior", "vwfa_intermediate"),
    13: ("vwfa_anterior",), 14: ("vwfa_intermediate", "mtg"),
    15: ("vwfa_intermediate", "vwfa_anterior"), 16: ("mtg", "atl", "ifg"),
    17: ("vwfa_anterior", "ifg", "mtg"), 18: ("v4", "votc_posterior", "vwfa_posterior"),
    19: ("vwfa_anterior", "ifg", "mtg", "atl", "angular_gyrus"),
    20: ("ifg", "mtg", "vwfa_anterior"), 21: ("parietal_attention", "vwfa_posterior"),
    22: ("vwfa_posterior", "stg_ventral", "phonological_association"),
    23: ("vwfa_intermediate", "vwfa_anterior"),
    24: ("orthographic_memory", "vwfa_anterior", "vwfa_intermediate"),
}


class VWFABiologicalSimulator:
    """Advance connected current-based LIF populations while keeping neural state separate from scores.

    This is a small deterministic population-level reference circuit. Regional labels are approximate
    functional groupings; parameters are configurable software defaults, not fitted human constants.
    """

    def __init__(self, config: VWFAConfig | None = None) -> None:
        """Allocate membrane, refractory, spike, activity, and connectivity state for each population."""
        self.config = config or VWFAConfig()
        size = self.config.lif_population_size
        self.voltage_mv = {name: np.full(size, self.config.rest_mv, dtype=np.float64) for name in POPULATIONS}
        self.refractory_ms = {name: np.zeros(size, dtype=np.float64) for name in POPULATIONS}
        self.spike_counts = {name: np.zeros(size, dtype=np.int64) for name in POPULATIONS}
        self.activity = {name: 0.0 for name in POPULATIONS}
        self.synaptic_current_pa = {name: np.zeros(size, dtype=np.float64) for name in POPULATIONS}
        self.connectivity = np.zeros((len(POPULATIONS), len(POPULATIONS)), dtype=np.float64)
        self.population_index = {name: index for index, name in enumerate(POPULATIONS)}
        for target, sources in _BASE_CONNECTIONS.items():
            for source, weight in sources.items():
                self.connectivity[self.population_index[target], self.population_index[source]] = weight
        self.time_ms = 0.0
        self.history: list[dict[str, float]] = []
        self.last_step: dict[str, object] = {}

    def reset_activity(self) -> None:
        """Reset per-sample membrane and activity state while preserving learned connectivity weights."""
        size = self.config.lif_population_size
        self.voltage_mv = {name: np.full(size, self.config.rest_mv, dtype=np.float64) for name in POPULATIONS}
        self.refractory_ms = {name: np.zeros(size, dtype=np.float64) for name in POPULATIONS}
        self.spike_counts = {name: np.zeros(size, dtype=np.int64) for name in POPULATIONS}
        self.activity = {name: 0.0 for name in POPULATIONS}
        self.synaptic_current_pa = {name: np.zeros(size, dtype=np.float64) for name in POPULATIONS}
        self.time_ms = 0.0
        self.history = []
        self.last_step = {}

    def _advance_one_step(self, external_drives: Mapping[str, float]) -> None:
        """Integrate one Euler step of leak, recurrent synaptic current, inhibition, threshold, and reset."""
        cfg = self.config
        previous = np.asarray([self.activity[name] for name in POPULATIONS], dtype=np.float64)
        graph_drive = self.connectivity @ previous
        global_activity = float(np.mean(previous))
        next_activity: dict[str, float] = {}
        for index, name in enumerate(POPULATIONS):
            external = float(np.clip(external_drives.get(name, 0.0), 0.0, 1.0))
            recurrent_pa = cfg.input_current_pa * cfg.recurrent_gain * graph_drive[index]
            external_pa = cfg.input_current_pa * external
            inhibition_pa = cfg.input_current_pa * cfg.inhibition_gain * global_activity
            current_scalar = external_pa + recurrent_pa - inhibition_pa
            current = np.full(cfg.lif_population_size, current_scalar, dtype=np.float64)
            self.synaptic_current_pa[name] = current
            voltage = self.voltage_mv[name]
            refractory = self.refractory_ms[name]
            active = refractory <= 0.0
            voltage[~active] = cfg.reset_mv
            refractory[~active] = np.maximum(0.0, refractory[~active] - cfg.dt_ms)
            # C dV/dt = (E_L - V)/R + I; tau=R*C, with R*I converted to mV.
            voltage[active] += (cfg.dt_ms / cfg.membrane_tau_ms) * (
                cfg.rest_mv - voltage[active]
                + 0.001 * cfg.membrane_resistance_mohm * current[active]
            )
            spiked = active & (voltage >= cfg.threshold_mv)
            if np.any(spiked):
                self.spike_counts[name][spiked] += 1
                voltage[spiked] = cfg.reset_mv
                refractory[spiked] = cfg.refractory_ms
            instantaneous_rate = float(np.count_nonzero(spiked)) / cfg.lif_population_size
            decay = math.exp(-cfg.dt_ms / max(cfg.membrane_tau_ms, cfg.dt_ms))
            next_activity[name] = decay * self.activity[name] + (1.0 - decay) * instantaneous_rate
        self.activity.update(next_activity)
        self.time_ms += cfg.dt_ms
        self.history.append(dict(self.activity))

    def advance_operation(self, operation_index: int, strength: float, steps: int | None = None) -> dict[str, object]:
        """Drive the populations associated with one VWFA operation and return a compact state snapshot."""
        if operation_index not in _OPERATION_DRIVES:
            raise ValueError(f"unknown VWFA operation index: {operation_index}")
        if not math.isfinite(float(strength)):
            raise ValueError("biological drive strength must be finite")
        count = self.config.lif_steps_per_operation if steps is None else int(steps)
        if count < 1:
            raise ValueError("steps must be positive")
        targets = _OPERATION_DRIVES[operation_index]
        drive = float(np.clip(strength, 0.0, 1.0))
        inputs = {name: drive for name in targets}
        for _ in range(count):
            self._advance_one_step(inputs)
        snapshot = self.snapshot()
        self.last_step = {
            "operation_index": operation_index,
            "driven_populations": list(targets),
            "normalized_drive": drive,
            "time_ms": self.time_ms,
            "population_activity": snapshot["population_activity"],
        }
        return dict(self.last_step)

    def apply_hebbian_update(self, pre_activity: Mapping[str, float], post_activity: Mapping[str, float]) -> dict[str, object]:
        """Apply a bounded, explicitly authorized co-activity update to selected graph projections."""
        updates: dict[str, float] = {}
        rate = self.config.learning_rate
        for target, post_value in post_activity.items():
            if target not in self.population_index:
                raise KeyError(f"unknown post population: {target}")
            for source, pre_value in pre_activity.items():
                if source not in self.population_index:
                    raise KeyError(f"unknown pre population: {source}")
                i = self.population_index[target]
                j = self.population_index[source]
                delta = rate * float(np.clip(pre_value, 0.0, 1.0)) * float(np.clip(post_value, 0.0, 1.0))
                before = float(self.connectivity[i, j])
                self.connectivity[i, j] = float(np.clip(before + delta, 0.0, self.config.max_synaptic_weight))
                updates[f"{target}<-{source}"] = float(self.connectivity[i, j] - before)
        return {"updates": updates, "weight_min": float(np.min(self.connectivity)),
                "weight_max": float(np.max(self.connectivity)), "learning_rate": rate}

    def consolidate_weights(self, baseline: np.ndarray | None = None, rate: float | None = None) -> dict[str, float]:
        """Move current recurrent weights a small step toward a reference matrix with bounded magnitudes."""
        reference = np.asarray(baseline, dtype=np.float64) if baseline is not None else np.asarray(
            [[_BASE_CONNECTIONS.get(target, {}).get(source, 0.0) for source in POPULATIONS] for target in POPULATIONS],
            dtype=np.float64,
        )
        if reference.shape != self.connectivity.shape or not np.all(np.isfinite(reference)):
            raise ValueError("baseline connectivity must be finite and match the circuit matrix")
        eta = self.config.consolidation_rate if rate is None else float(rate)
        if not 0.0 <= eta <= 1.0:
            raise ValueError("consolidation rate must lie in [0, 1]")
        self.connectivity = np.clip((1.0 - eta) * self.connectivity + eta * reference,
                                    0.0, self.config.max_synaptic_weight)
        return {"rate": eta, "weight_mean": float(np.mean(self.connectivity)),
                "weight_max": float(np.max(self.connectivity))}

    def snapshot(self) -> dict[str, object]:
        """Summarize population voltage, synaptic current, spikes, rates, coupling, and model scope."""
        elapsed_seconds = max(self.time_ms / 1000.0, 1e-9)
        populations = {}
        for name in POPULATIONS:
            populations[name] = {
                "mean_voltage_mv": float(np.mean(self.voltage_mv[name])),
                "mean_current_pa": float(np.mean(self.synaptic_current_pa[name])),
                "spike_count": int(np.sum(self.spike_counts[name])),
                "mean_firing_rate_hz": float(np.sum(self.spike_counts[name]) /
                                              self.config.lif_population_size / elapsed_seconds),
                "mean_activity": float(self.activity[name]),
            }
        return {
            "model": "ore_vwfa_current_based_LIF_population_reference",
            "time_ms": float(self.time_ms),
            "population_order": list(POPULATIONS),
            "population_activity": {name: float(self.activity[name]) for name in POPULATIONS},
            "populations": populations,
            "connectivity": self.connectivity.tolist(),
            "units": {"voltage": "mV", "current": "pA", "time": "ms", "firing_rate": "Hz"},
            "scope": "Population-level computational reference; not anatomically fitted or empirically validated.",
        }

    def activity_series(self) -> np.ndarray:
        """Return simulated time-by-population activity for diagnostics and connectivity analysis."""
        if not self.history:
            return np.zeros((0, len(POPULATIONS)), dtype=np.float64)
        return np.asarray([[row[name] for name in POPULATIONS] for row in self.history], dtype=np.float64)
