"""Signal, selection, timing, similarity, and evaluation operations for sleep replay.

These mathematical helpers make a computational hypothesis inspectable. They are not literal
arithmetic attributed to individual neurons.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from .contracts import SleepConsolidationConfig


def sigmoid(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Apply a numerically stable logistic activation to a population vector."""
    array = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(array, -60.0, 60.0)))


def normalize_vector(values: Sequence[float] | np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Return an L2-normalized finite vector, preserving exact zero vectors."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError("vector must contain only finite values")
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > epsilon else np.zeros_like(vector)


def fit_vector(values: Sequence[float], width: int) -> np.ndarray:
    """Pad or truncate a finite feature vector to the configured model dimension."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(vector)) or width < 1:
        raise ValueError("feature vector and target width must be valid")
    result = np.zeros(width, dtype=np.float64)
    result[:min(width, vector.size)] = vector[:width]
    return result


def cosine_similarity(left: Sequence[float], right: Sequence[float], epsilon: float = 1e-12) -> float:
    """Compute cosine similarity after padding unequal vectors to a common feature dimension."""
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("similarity inputs must be finite")
    width = max(a.size, b.size)
    a, b = fit_vector(a, width), fit_vector(b, width)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= epsilon:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    """Convert finite replay-priority scores into a stable normalized choice distribution."""
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("softmax scores must be a non-empty finite vector")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive")
    exponentials = np.exp(np.clip((scores - np.max(scores)) / temperature, -60.0, 0.0))
    return exponentials / max(float(np.sum(exponentials)), 1e-12)


def classify_offline_state(
    state: str, confidence: float, threshold: float, allow_awake_rest: bool
) -> dict[str, object]:
    """Gate replay eligibility from a supplied brain-state label and classifier confidence."""
    normalized = state.lower()
    accepted = normalized == "nrem" or (normalized == "awake_rest" and allow_awake_rest)
    eligible = bool(accepted and confidence >= threshold)
    return {"state": normalized, "confidence": float(confidence), "eligible": eligible,
            "reason": "eligible_offline_state" if eligible else "state_or_confidence_below_gate"}


def estimate_oscillation_phase(
    signal: Sequence[float], sample_rate_hz: float, band_hz: tuple[float, float] = (0.3, 1.25)
) -> dict[str, object]:
    """Estimate slow-oscillation phase using band-pass filtering and the analytic signal.

    Filtering and Hilbert phase are analyst-side signal-processing tools, not neural calculations.
    """
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    low, high = band_hz
    if values.size < 20 or not np.all(np.isfinite(values)):
        raise ValueError("phase estimation requires at least 20 finite signal samples")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 2.0 * high:
        raise ValueError("sample_rate_hz must exceed twice the high cutoff")
    if not 0.0 < low < high:
        raise ValueError("band_hz must satisfy 0 < low < high")
    sos = butter(3, (low, high), btype="bandpass", fs=sample_rate_hz, output="sos")
    filtered = sosfiltfilt(sos, values)
    analytic = hilbert(filtered)
    phase = np.angle(analytic)
    return {"phase_rad": float(phase[-1]), "mean_amplitude": float(np.mean(np.abs(analytic))),
            "filtered_signal": filtered}


def detect_spindle(
    envelope: float, duration_ms: float, config: SleepConsolidationConfig
) -> dict[str, object]:
    """Detect a spindle-like burst from supplied sigma-envelope and duration summaries."""
    detected = bool(envelope >= config.spindle_envelope_threshold
                    and duration_ms >= config.min_spindle_duration_ms)
    strength = float(np.clip(envelope / max(config.spindle_envelope_threshold * 2.0, 1e-9),
                             0.0, 1.0)) if detected else 0.0
    return {"detected": detected, "envelope": float(envelope), "duration_ms": float(duration_ms),
            "strength": strength, "criterion": "envelope_and_minimum_duration"}


def detect_sharp_wave_ripple(
    ripple_band_z: float, sharp_wave_z: float, duration_ms: float,
    config: SleepConsolidationConfig,
) -> dict[str, object]:
    """Detect an SWR candidate from ripple-band and sharp-wave evidence separately.

    A detected SWR is not itself proof that a particular episodic trace was replayed.
    """
    score = 0.6 * float(ripple_band_z) + 0.4 * float(sharp_wave_z)
    duration_valid = config.min_ripple_duration_ms <= duration_ms <= config.max_ripple_duration_ms
    detected = bool(score >= config.swr_score_threshold and duration_valid)
    return {"detected": detected, "score": score, "ripple_band_z": float(ripple_band_z),
            "sharp_wave_z": float(sharp_wave_z), "duration_ms": float(duration_ms),
            "duration_valid": duration_valid}


def replay_priority(
    recency: float, salience: float, value: float, goal_relevance: float,
    interference_risk: float,
) -> float:
    """Compute a transparent model-only replay-priority score from supplied trace attributes."""
    return float(0.25 * recency + 0.20 * salience + 0.20 * value
                 + 0.25 * goal_relevance - 0.20 * interference_risk)


def phase_locking_value(phases_rad: Sequence[float]) -> float | None:
    """Calculate phase-locking value across observed ripple times, or None without events."""
    phases = np.asarray(phases_rad, dtype=np.float64).reshape(-1)
    if phases.size == 0:
        return None
    if not np.all(np.isfinite(phases)):
        raise ValueError("phase values must be finite")
    return float(np.abs(np.mean(np.exp(1j * phases))))


def rhythm_alignment(
    slow_phase: float | None,
    spindle_detected: bool,
    spindle_strength: float,
    spindle_phase: float | None,
    preferred_phase: float,
    ripple_detected: bool,
    prior_ripple_phases: Sequence[float],
) -> dict[str, object]:
    """Combine phase preference, spindle presence, and ripple evidence into a graded coupling score."""
    if not ripple_detected or slow_phase is None:
        return {"score": 0.0, "phase_alignment": 0.0, "spindle_alignment": 0.0,
                "phase_locking_value": phase_locking_value(prior_ripple_phases),
                "status": "insufficient_rhythm_or_ripple_evidence"}
    phase_alignment = 0.5 * (1.0 + math.cos(slow_phase - preferred_phase))
    spindle_alignment = 0.0
    if spindle_detected:
        spindle_alignment = spindle_strength
        if spindle_phase is not None:
            spindle_alignment *= 0.5 * (1.0 + math.cos(spindle_phase - slow_phase))
    score = float(np.clip(phase_alignment * spindle_alignment, 0.0, 1.0))
    return {"score": score, "phase_alignment": float(phase_alignment),
            "spindle_alignment": float(spindle_alignment),
            "phase_locking_value": phase_locking_value(prior_ripple_phases),
            "status": "rhythms_coordinated" if score > 0.0 else "weak_or_absent_spindle_coupling"}


def stdp_timing_kernel(interval_ms: float, tau_ms: float) -> float:
    """Return a signed exponential pre/post timing factor under post-minus-pre convention."""
    if tau_ms <= 0.0 or not math.isfinite(interval_ms):
        raise ValueError("tau_ms must be positive and interval_ms finite")
    if interval_ms > 0.0:
        return float(math.exp(-interval_ms / tau_ms))
    if interval_ms < 0.0:
        return float(-math.exp(interval_ms / tau_ms))
    return 1.0


def compressed_replay_times(
    sequence_length: int, gamma: float, source_interval_ms: float = 100.0
) -> np.ndarray:
    """Generate compressed model timestamps while preserving sequence order."""
    if sequence_length < 1 or not 0.0 < gamma <= 1.0 or source_interval_ms < 0.0:
        raise ValueError("sequence length, compression ratio, and source interval are invalid")
    source = np.arange(sequence_length, dtype=np.float64) * source_interval_ms
    return gamma * source


def reinstatement_similarity(activity: Sequence[float], target: Sequence[float]) -> dict[str, float]:
    """Score whole-pattern and coarse feature-group reinstatement against a stored cortical code."""
    current = np.asarray(activity, dtype=np.float64).reshape(-1)
    expected = np.asarray(target, dtype=np.float64).reshape(-1)
    score = cosine_similarity(current, expected)
    width = min(current.size, expected.size)
    groups = np.array_split(np.arange(width), min(4, max(1, width)))
    partial = [cosine_similarity(current[index], expected[index]) for index in groups if index.size]
    return {"overall_similarity": score,
            "mean_group_similarity": float(np.mean(partial)) if partial else score,
            "active_group_count": int(sum(float(np.linalg.norm(current[index])) > 1e-12
                                          for index in groups if index.size))}


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Return Pearson correlation for at least two non-constant paired observations."""
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return None
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("correlation inputs must be finite")
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])
