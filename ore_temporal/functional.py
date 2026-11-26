"""Pure mathematical operations for the Point 13 temporal layer.

Every function here implements one governed quantity from the specification's
mathematical-operations table: time deltas, Euler integration, exponential relaxation,
state-space updates, leaky integrate-and-fire dynamics, sampling irregularity, event
density and sparsity, ensemble spread, residuals, CUSUM updates, and robust z-scores.
Functions are pure so tests can verify them against references directly.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np

__all__ = (
    "coefficient_of_variation",
    "compute_delta_times",
    "cusum_update",
    "ensemble_mean_variance",
    "event_density",
    "event_sparsity",
    "exponential_relaxation",
    "gap_quantiles",
    "lagged_correlation",
    "lif_advance",
    "lif_derivative",
    "robust_zscore",
    "residual",
    "rolling_mean",
    "rk4_step",
    "euler_step",
    "ssm_step",
)


def compute_delta_times(timestamps: Sequence[float]) -> tuple[tuple[float, ...], bool]:
    """Step 11 -- compute elapsed times between consecutive observations.

    Negative intervals are rejected; zero intervals are reported as simultaneous events.
    Returns the elapsed times and whether any zero interval occurred.
    """
    if len(timestamps) == 0:
        return ((), False)
    deltas: list[float] = []
    simultaneous = False
    previous = float(timestamps[0])
    if not math.isfinite(previous):
        raise ValueError("timestamps must be finite")
    for raw in timestamps[1:]:
        current = float(raw)
        if not math.isfinite(current):
            raise ValueError("timestamps must be finite")
        delta = current - previous
        if delta < 0.0:
            raise ValueError(f"negative elapsed time {delta} at timestamp {current}; "
                             "never silently rewrite historical ordering")
        if delta == 0.0:
            simultaneous = True
        deltas.append(delta)
        previous = current
    return (tuple(deltas), simultaneous)


def euler_step(x: np.ndarray, dt: float,
               derivative: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Section 3.2 -- explicit Euler update x_{k+1} = x_k + dt * f(x_k)."""
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    return np.asarray(x, dtype=float) + dt * np.asarray(derivative(np.asarray(x, dtype=float)),
                                                        dtype=float)


def rk4_step(x: np.ndarray, dt: float,
             derivative: Callable[[np.ndarray, float], np.ndarray], t: float = 0.0) -> np.ndarray:
    """Section 3.2 -- fourth-order Runge-Kutta update evaluating intermediate points."""
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    x = np.asarray(x, dtype=float)
    k1 = np.asarray(derivative(x, t), dtype=float)
    k2 = np.asarray(derivative(x + 0.5 * dt * k1, t + 0.5 * dt), dtype=float)
    k3 = np.asarray(derivative(x + 0.5 * dt * k2, t + 0.5 * dt), dtype=float)
    k4 = np.asarray(derivative(x + dt * k3, t + dt), dtype=float)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def exponential_relaxation(x: np.ndarray, x_target: np.ndarray, dt: float,
                          tau: float) -> np.ndarray:
    """Section 3.3 -- closed-form update x(t+dt) = x_inf + (x - x_inf) * exp(-dt / tau)."""
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    x = np.asarray(x, dtype=float)
    target = np.asarray(x_target, dtype=float)
    return target + (x - target) * math.exp(-dt / tau)


def ssm_step(h: np.ndarray, x: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Section 3.4 -- state-space memory update h_t = A h_{t-1} + B x_t."""
    h = np.asarray(h, dtype=float)
    x = np.asarray(x, dtype=float)
    if h.ndim != 1 or x.ndim != 1:
        raise ValueError("state and input must be one-dimensional vectors")
    return np.asarray(A, dtype=float) @ h + np.asarray(B, dtype=float) @ x


def lif_derivative(v: np.ndarray | float, current: np.ndarray | float, tau_m: float,
                   v_rest: float, resistance: float) -> np.ndarray:
    """Section 3.5 -- leaky integrate-and-fire membrane derivative tau_m dV/dt."""
    if tau_m <= 0.0:
        raise ValueError("tau_m must be positive")
    v = np.asarray(v, dtype=float)
    current = np.asarray(current, dtype=float)
    return (-(v - v_rest) + resistance * current) / tau_m


def lif_advance(v: np.ndarray, current: np.ndarray, dt: float, tau_m: float, v_rest: float,
                v_threshold: float, v_reset: float, resistance: float,
                refractory_period: float = 0.0,
                refractory_remaining: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance LIF membranes over dt and apply threshold, reset, and refractory rules.

    Returns (new_voltage, spikes, refractory_remaining).  Sub-stepping keeps the explicit
    Euler integration stable for dt larger than tau_m; threshold crossings emit spikes and
    reset the membrane, and refractory membranes hold at reset until the period elapses.
    """
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    if tau_m <= 0.0:
        raise ValueError("tau_m must be positive")
    if refractory_period < 0.0:
        raise ValueError("refractory_period must be non-negative")
    v = np.asarray(v, dtype=float).copy()
    current = np.asarray(current, dtype=float)
    n_neurons = v.size
    spikes = np.zeros(n_neurons, dtype=bool)
    refractory = (np.zeros(n_neurons, dtype=float) if refractory_remaining is None
                  else np.asarray(refractory_remaining, dtype=float).copy())
    if refractory.size != n_neurons:
        raise ValueError("refractory_remaining must match the membrane vector size")
    substeps = max(1, int(math.ceil(dt / max(tau_m * 0.25, 1e-9))))
    step = dt / substeps
    for _ in range(substeps):
        active = refractory <= 0.0
        dv = np.zeros(n_neurons, dtype=float)
        if np.any(active):
            dv[active] = lif_derivative(v[active], current[active], tau_m, v_rest, resistance)
        v = v + step * dv
        spiked = (v >= v_threshold) & (refractory <= 0.0)
        if np.any(spiked):
            spikes |= spiked
            v[spiked] = v_reset
            refractory[spiked] = refractory[spiked] + refractory_period
        refractory = np.maximum(refractory - step, 0.0)
    return (v, spikes, refractory)


def event_density(n_events: int, duration: float) -> float:
    """Section 3.6 -- event density rho = N_events / T over a consistent interval."""
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if n_events < 0:
        raise ValueError("n_events must be non-negative")
    return float(n_events) / float(duration)


def event_sparsity(n_events: int, n_baseline: int) -> float:
    """Section 3.6 -- event sparsity s = 1 - N_events / N_baseline."""
    if n_baseline <= 0:
        raise ValueError("n_baseline must be positive")
    if n_events < 0:
        raise ValueError("n_events must be non-negative")
    return 1.0 - float(n_events) / float(n_baseline)


def coefficient_of_variation(samples: Iterable[float]) -> float:
    """Section 3.6 -- CV = sigma / mu with explicit zero-mean handling."""
    values = [float(item) for item in samples]
    if not values:
        raise ValueError("samples must not be empty")
    mean = sum(values) / len(values)
    if mean == 0.0:
        return 0.0 if all(item == 0.0 for item in values) else math.inf
    variance = sum((item - mean) ** 2 for item in values) / max(len(values) - 1, 1)
    return math.sqrt(variance) / abs(mean)


def ensemble_mean_variance(predictions: Sequence[float]) -> tuple[float, float]:
    """Section 3.6 -- ensemble mean and sample variance across M predictions."""
    values = [float(item) for item in predictions]
    count = len(values)
    if count < 1:
        raise ValueError("ensemble requires at least one prediction")
    mean = sum(values) / count
    if count == 1:
        return (mean, 0.0)
    variance = sum((item - mean) ** 2 for item in values) / (count - 1)
    return (mean, variance)


def residual(observed: float, predicted: float) -> float:
    """Section 8 -- residual r_t = y_t - y_hat_t with aligned time and units."""
    return float(observed) - float(predicted)


def rolling_mean(values: Sequence[float], window: int) -> list[float]:
    """Compute the trailing moving average used by deterministic baselines."""
    if window <= 0:
        raise ValueError("window must be positive")
    output: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += float(value)
        if index >= window:
            running -= float(values[index - window])
        effective = min(index + 1, window)
        output.append(running / effective)
    return output


def cusum_update(statistic: float, innovation: float, drift: float) -> float:
    """Step 27 -- bounded one-sided CUSUM statistic max(0, stat + innovation - drift)."""
    return max(0.0, float(statistic) + float(innovation) - float(drift))


def robust_zscore(value: float, median: float, mad: float) -> float:
    """Step 28 -- robust z-score using median absolute deviation."""
    if mad <= 0.0:
        return 0.0 if value == median else math.copysign(math.inf, value - median)
    return 0.6745 * (float(value) - float(median)) / float(mad)


def gap_quantiles(deltas: Sequence[float], quantiles: Sequence[float]) -> dict[float, float]:
    """Step 14 -- gap quantiles of the elapsed-time distribution."""
    for q in quantiles:
        if not 0.0 <= q <= 1.0:
            raise ValueError("quantiles must be in [0, 1]")
    if not deltas:
        return {q: 0.0 for q in quantiles}
    ordered = sorted(float(item) for item in deltas)
    result: dict[float, float] = {}
    for q in quantiles:
        position = q * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            result[q] = ordered[lower]
        else:
            weight = position - lower
            result[q] = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return result


def lagged_correlation(series_a: Sequence[float], series_b: Sequence[float],
                       max_lag: int) -> dict[int, float]:
    """Step 30 -- lagged correlation; correlation is not causation."""
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    if len(series_a) != len(series_b) or not series_a:
        raise ValueError("series must be non-empty and of equal length")
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    a_centered = a - a.mean() if a.size > 1 else a
    b_centered = b - b.mean() if b.size > 1 else b
    denom = math.sqrt(float(a_centered @ a_centered) * float(b_centered @ b_centered))
    output: dict[int, float] = {}
    for lag in range(0, max_lag + 1):
        if lag >= a.size:
            output[lag] = 0.0
            continue
        left = a_centered[: a.size - lag]
        right = b_centered[lag:]
        local = math.sqrt(float(left @ left) * float(right @ right))
        output[lag] = float(left @ right) / local if local > 0.0 and denom > 0.0 else 0.0
    return output
