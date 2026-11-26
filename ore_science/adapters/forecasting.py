"""Probabilistic ensemble forecasting interfaces (Steps 30-31, Section 5.6).

Ensemble forecasts retain member identity and lead times. Statistics follow Section 5.6:
mean mu_hat = (1/M) sum_m x^(m), sample variance sigma_hat^2 = (1/(M-1)) sum_m (x^(m)-mu_hat)^2,
and event probabilities are estimated by the fraction of members satisfying the event
condition. Calibration evidence (Brier score, reliability bins, coverage, sharpness) is
computed against verifying observations, and a forecast that is accurate on average but
poorly calibrated is reported as such.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "EnsembleForecast",
    "StochasticForecastSimulator",
    "brier_score",
    "calibration_report",
]


@dataclass(frozen=True, slots=True)
class EnsembleForecast:
    """An ensemble forecast with member identity, lead times, and metadata."""

    members: np.ndarray  # (M, T) member trajectories
    lead_times: np.ndarray  # (T,)
    quantity_name: str
    unit_symbol: str
    model_id: str
    model_version: str
    initialization: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate member and lead-time shapes."""
        if self.members.ndim != 2:
            raise ValueError("members must form an (M, T) array")
        if self.lead_times.shape[0] != self.members.shape[1]:
            raise ValueError("lead times must match the trajectory length")
        if self.members.shape[0] < 2:
            raise ValueError("ensembles require at least two members")

    # -- Section 5.6 statistics ------------------------------------------------
    def mean(self) -> np.ndarray:
        """Ensemble mean mu_hat = (1/M) sum_m x^(m)."""
        return self.members.mean(axis=0)

    def variance(self) -> np.ndarray:
        """Sample variance sigma_hat^2 with the (M-1) denominator."""
        return self.members.var(axis=0, ddof=1)

    def std(self) -> np.ndarray:
        """Ensemble standard deviation (spread)."""
        return np.sqrt(self.variance())

    def quantile(self, q: float) -> np.ndarray:
        """Return the q-quantile trajectory across members."""
        if not 0.0 <= q <= 1.0:
            raise ValueError("quantile must lie in [0, 1]")
        return np.quantile(self.members, q, axis=0)

    def event_probability(self, condition: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Estimate event probabilities as member fractions satisfying the condition.

        ``condition`` maps a (M, T) array to a boolean (M, T) array.
        """
        satisfied = np.asarray(condition(self.members), dtype=bool)
        return satisfied.mean(axis=0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the forecast metadata without inlining member trajectories."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "member_count": int(self.members.shape[0]),
            "lead_times": [float(value) for value in self.lead_times],
            "quantity_name": self.quantity_name,
            "unit_symbol": self.unit_symbol,
            "initialization": self.initialization,
            "metadata": dict(self.metadata),
        }


class StochasticForecastSimulator:
    """Synthetic AR(1) ensemble generator for tests and demonstrations.

    Generates members x_m(t+1) = phi x_m(t) + sigma eta_t with a shared deterministic
    drift. The simulator is a test instrument, not a weather model; the adapter records
    this so no claim of physical forecasting skill is implied.
    """

    model_id = "forecast.ensemble_ar1"
    model_version = "1.0.0"

    def __init__(self, *, phi: float = 0.97, sigma: float = 0.35, drift: float = 0.01,
                 seed: int = 2026) -> None:
        """Configure the autoregressive process and random stream."""
        if not -1.0 < phi < 1.0:
            raise ValueError("AR(1) coefficient must lie strictly inside (-1, 1)")
        if sigma < 0.0:
            raise ValueError("noise scale must be non-negative")
        self.phi = phi
        self.sigma = sigma
        self.drift = drift
        self.seed = seed

    def simulate(
        self,
        *,
        initial_state: float,
        members: int,
        lead_times: Sequence[float],
        observation_noise: float = 0.0,
    ) -> tuple[EnsembleForecast, np.ndarray]:
        """Return the ensemble forecast and the verifying (noisy) truth trajectory.

        The truth trajectory uses a separate deterministic random stream so verifying
        observations are not a member of the ensemble.
        """
        if members < 2:
            raise ValueError("simulation requires at least two members")
        steps = len(lead_times)
        generator = np.random.default_rng(self.seed)
        members_array = np.empty((members, steps), dtype=float)
        state = initial_state
        for member_index in range(members):
            value = state
            for step_index in range(steps):
                value = self.phi * value + self.drift + self.sigma * generator.normal()
                members_array[member_index, step_index] = value
        truth_generator = np.random.default_rng(self.seed + 991)
        truth = np.empty(steps, dtype=float)
        value = state
        for step_index in range(steps):
            value = self.phi * value + self.drift + self.sigma * truth_generator.normal()
            truth[step_index] = value
        observations = truth + observation_noise * np.random.default_rng(self.seed + 7).normal(
            size=steps
        )
        forecast = EnsembleForecast(
            members=members_array,
            lead_times=np.asarray(lead_times, dtype=float),
            quantity_name="wind speed",
            unit_symbol="m/s",
            model_id=self.model_id,
            model_version=self.model_version,
            initialization=f"initial state {initial_state:g}",
            metadata={
                "process": "AR(1) with drift",
                "phi": self.phi,
                "sigma": self.sigma,
                "drift": self.drift,
                "seed": self.seed,
                "disclosure": "synthetic test instrument, not a weather model",
            },
        )
        return forecast, observations


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Compute the Brier score for probabilistic event predictions."""
    probabilities = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if probabilities.shape != outcomes.shape:
        raise ValueError("probabilities and outcomes must have equal shapes")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    if np.any((outcomes < 0.0) | (outcomes > 1.0)):
        raise ValueError("outcomes must be binary indicators")
    return float(np.mean((probabilities - outcomes) ** 2))


def calibration_report(
    probabilities: np.ndarray, outcomes: np.ndarray, *, bins: int = 5
) -> dict[str, Any]:
    """Reliability, Brier score, resolution, and sharpness for event forecasts."""
    probabilities = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    reliability: list[dict[str, Any]] = []
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        in_bin = (probabilities >= low) & (probabilities < high) if index < bins - 1 else (
            (probabilities >= low) & (probabilities <= high)
        )
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        mean_probability = float(np.mean(probabilities[in_bin]))
        observed_frequency = float(np.mean(outcomes[in_bin]))
        reliability.append(
            {
                "bin": f"[{low:.2f}, {high:.2f})",
                "count": count,
                "mean_forecast_probability": mean_probability,
                "observed_frequency": observed_frequency,
                "abs_gap": abs(mean_probability - observed_frequency),
            }
        )
    base_rate = float(np.mean(outcomes))
    return {
        "brier_score": brier_score(probabilities, outcomes),
        "reliability_bins": reliability,
        "climatological_base_rate": base_rate,
        "mean_probability": float(np.mean(probabilities)),
        "sharpness_std": float(np.std(probabilities)),
        "max_abs_reliability_gap": max(
            (entry["abs_gap"] for entry in reliability), default=0.0
        ),
    }
