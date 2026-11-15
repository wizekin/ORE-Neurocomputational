"""Population-level leaky-integrate-and-fire reference for visual orthographic processing."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .contracts import VisualOrthographyConfig


BIOLOGICAL_POPULATIONS = (
    "retina_lgn",
    "v1",
    "v2_v4",
    "vwfa_posterior",
    "letter_identity",
    "letter_position",
    "letter_conjunction",
    "open_bigram",
    "orthography",
    "lexical",
)


class BiologicalReferenceSimulator:
    """Advance connected population activity using a configurable current-based LIF approximation.

    Population names indicate broad functional analogies, not one-to-one anatomical assignments.
    The default weights and physiological parameters are illustrative software settings, not
    subject-specific estimates or experimentally fitted human-neuron values.
    """

    def __init__(self, config: VisualOrthographyConfig | None = None) -> None:
        """Initialize membrane, refractory, rate and recurrent-connection state for each population."""
        self.config = config or VisualOrthographyConfig()
        self._index = {name: index for index, name in enumerate(BIOLOGICAL_POPULATIONS)}
        count = len(BIOLOGICAL_POPULATIONS)
        neurons = self.config.neurons_per_population
        self._voltage = np.full((count, neurons), self.config.rest_mv, dtype=np.float64)
        self._refractory = np.zeros((count, neurons), dtype=np.float64)
        self._rates = np.zeros(count, dtype=np.float64)
        self._spike_counts = np.zeros(count, dtype=np.int64)
        self._last_current = np.zeros(count, dtype=np.float64)
        self._time_ms = 0.0
        self._history: list[np.ndarray] = []
        self._weights = self._build_recurrent_weights()

    def _build_recurrent_weights(self) -> np.ndarray:
        """Create a sparse excitatory projection matrix between broad visual and orthographic pools."""
        weights = np.zeros(
            (len(BIOLOGICAL_POPULATIONS), len(BIOLOGICAL_POPULATIONS)),
            dtype=np.float64,
        )
        edges = (
            ("v1", "retina_lgn", 0.45),
            ("v2_v4", "v1", 0.42),
            ("vwfa_posterior", "v2_v4", 0.35),
            ("letter_identity", "vwfa_posterior", 0.40),
            ("letter_position", "v1", 0.18),
            ("letter_conjunction", "letter_identity", 0.32),
            ("letter_conjunction", "letter_position", 0.22),
            ("open_bigram", "letter_identity", 0.25),
            ("orthography", "letter_conjunction", 0.30),
            ("orthography", "open_bigram", 0.35),
            ("lexical", "orthography", 0.30),
            ("orthography", "lexical", 0.08),
            ("letter_identity", "letter_conjunction", 0.06),
        )
        for target, source, weight in edges:
            weights[self._index[target], self._index[source]] = weight
        return weights

    def advance(self, drives: Mapping[str, float], steps: int = 1) -> dict[str, object]:
        """Advance every population together through Euler-integrated LIF membrane updates."""
        if steps < 1:
            raise ValueError("steps must be positive")
        unknown = set(drives) - set(BIOLOGICAL_POPULATIONS)
        if unknown:
            raise KeyError(f"unknown biological populations: {sorted(unknown)}")
        drive_vector = np.zeros(len(BIOLOGICAL_POPULATIONS), dtype=np.float64)
        for name, strength in drives.items():
            value = float(strength)
            if not np.isfinite(value):
                raise ValueError(f"drive for {name} must be finite")
            drive_vector[self._index[name]] = np.clip(value, 0.0, 1.0)

        config = self.config
        for _ in range(int(steps)):
            normalized_rates = np.clip(self._rates / 100.0, 0.0, 1.0)
            recurrent = self._weights @ normalized_rates
            total_activity = float(np.sum(normalized_rates))
            lateral_competition = np.maximum(total_activity - normalized_rates, 0.0)
            current = (
                config.input_gain_pa * drive_vector
                + config.recurrent_gain_pa * recurrent
                - config.inhibition_gain_pa * lateral_competition / max(len(normalized_rates) - 1, 1)
            )
            self._last_current = current
            dt = config.dt_ms
            fraction = dt / config.membrane_tau_ms
            membrane_drive_mv = config.membrane_resistance_mohm * current * 0.001
            self._voltage += fraction * (
                config.rest_mv - self._voltage + membrane_drive_mv[:, None]
            )
            self._refractory = np.maximum(self._refractory - dt, 0.0)
            eligible = self._refractory <= 0.0
            spikes = eligible & (self._voltage >= config.threshold_mv)
            self._voltage[spikes] = config.reset_mv
            self._refractory[spikes] = config.refractory_ms
            population_spikes = np.sum(spikes, axis=1).astype(np.int64)
            self._spike_counts += population_spikes
            instantaneous_rate = population_spikes / self.config.neurons_per_population
            instantaneous_rate *= 1000.0 / dt
            self._rates = 0.85 * self._rates + 0.15 * instantaneous_rate
            self._rates = np.clip(self._rates, 0.0, 100.0)
            self._time_ms += dt
            self._history.append(self._rates.copy())
        return self.snapshot()

    def snapshot(self) -> dict[str, object]:
        """Return population voltage, current, firing-rate and spike summaries with explicit units."""
        populations = {}
        for index, name in enumerate(BIOLOGICAL_POPULATIONS):
            populations[name] = {
                "mean_voltage_mv": float(np.mean(self._voltage[index])),
                "mean_current_pa": float(self._last_current[index]),
                "firing_rate_hz": float(self._rates[index]),
                "spike_count": int(self._spike_counts[index]),
            }
        history = (
            np.stack(self._history, axis=0)
            if self._history else np.zeros((0, len(BIOLOGICAL_POPULATIONS)), dtype=np.float64)
        )
        return {
            "time_ms": float(self._time_ms),
            "populations": populations,
            "population_names": list(BIOLOGICAL_POPULATIONS),
            "population_rate_history_hz": history,
            "scope": (
                "Configurable population-level current-based LIF reference. It is a computational "
                "hypothesis, not an anatomically fitted or empirically validated human-brain simulation."
            ),
        }
