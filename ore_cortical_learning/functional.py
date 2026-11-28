"""Functional mathematical operations for gradual cortical learning from repeated encounters.

The routines provide explicit, testable abstractions for distributed activity, similarity,
co-activity, reliability, prediction, plasticity gating, interference control, and evaluation.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def sigmoid(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Apply a numerically stable logistic population-response function."""
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("sigmoid input must be finite")
    return 1.0 / (1.0 + np.exp(-np.clip(array, -60.0, 60.0)))


def normalize_vector(values: Sequence[float] | np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Return an L2-normalized copy of a finite vector while preserving the zero vector."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError("vector must contain only finite values")
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, epsilon) if norm > epsilon else np.zeros_like(vector)


def compose_encounter_vector(
    sensory: Sequence[float], action: Sequence[float], context: Sequence[float],
    goals: Sequence[float], episode_trace: Sequence[float], dimensions: Sequence[int],
) -> np.ndarray:
    """Concatenate fixed-width input channels, zero-filling only omitted optional channels."""
    channels = (sensory, action, context, goals, episode_trace)
    if len(dimensions) != len(channels):
        raise ValueError("dimensions must define all five encounter channels")
    normalized: list[np.ndarray] = []
    for index, (values, width) in enumerate(zip(channels, dimensions)):
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if width == 0 and vector.size == 0:
            normalized.append(vector)
            continue
        if vector.size == 0 and index > 0:
            normalized.append(np.zeros(width, dtype=np.float64))
            continue
        if vector.size != width:
            raise ValueError(f"encounter channel {index} expected {width} values, received {vector.size}")
        if not np.all(np.isfinite(vector)):
            raise ValueError("encounter channels must contain only finite values")
        normalized.append(vector)
    return np.concatenate(normalized)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity for two equal-length activity patterns."""
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError("similarity patterns must have equal dimensions")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("similarity patterns must be finite")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def compare_with_history(
    current: Sequence[float], history: Sequence[Mapping[str, Any]], context_id: str,
    similarity_threshold: float,
) -> dict[str, Any]:
    """Rank retained encounter patterns and report whether an overlapping pattern is present."""
    scores = []
    for row in history:
        similarity = cosine_similarity(current, row["activity"])
        scores.append({"encounter_id": row["encounter_id"], "context_id": row["context_id"],
                       "similarity": similarity, "context_match": row["context_id"] == context_id})
    scores.sort(key=lambda item: item["similarity"], reverse=True)
    best = scores[0] if scores else None
    return {
        "comparison_count": len(scores),
        "nearest_encounter": best,
        "similarity": float(best["similarity"]) if best else None,
        "overlap_detected": bool(best and best["similarity"] >= similarity_threshold),
        "context_match": bool(best and best["context_match"]),
        "top_matches": scores[:5],
    }


def ema_update(previous: np.ndarray, current: np.ndarray, decay: float) -> np.ndarray:
    """Update a running mean or statistic using an exponentially weighted incremental average."""
    old = np.asarray(previous, dtype=np.float64)
    new = np.asarray(current, dtype=np.float64)
    if old.shape != new.shape:
        raise ValueError("EMA states must have equal shapes")
    if not 0.0 <= decay <= 1.0 or not np.all(np.isfinite(old)) or not np.all(np.isfinite(new)):
        raise ValueError("EMA inputs must be finite and decay must lie in [0, 1]")
    return (1.0 - decay) * old + decay * new


def coactivity_statistics(activity: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return the activity outer product and centered covariance contribution for one encounter."""
    vector = np.asarray(activity, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError("activity must be finite")
    coactivity = np.outer(vector, vector)
    centered = vector - float(np.mean(vector))
    covariance = np.outer(centered, centered)
    return coactivity, covariance


def relationship_observations(activity: Sequence[float], threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Count conditional co-activation successes and failures for currently active source units."""
    vector = np.asarray(activity, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(vector)) or not 0.0 <= threshold <= 1.0:
        raise ValueError("activity must be finite and threshold must lie in [0, 1]")
    active = vector >= threshold
    successes = np.outer(active, active).astype(np.float64)
    failures = np.logical_and(active[:, None], np.logical_not(active[None, :])).astype(np.float64)
    return successes, failures


def conditional_reliability(
    success_counts: np.ndarray, failure_counts: np.ndarray,
    prior_alpha: float = 1.0, prior_beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate smoothed conditional relationship probabilities and effective observation counts."""
    success = np.asarray(success_counts, dtype=np.float64)
    failure = np.asarray(failure_counts, dtype=np.float64)
    if success.shape != failure.shape or not np.all(np.isfinite(success)) or not np.all(np.isfinite(failure)):
        raise ValueError("relationship count matrices must be finite and have equal shapes")
    if np.any(success < 0.0) or np.any(failure < 0.0) or min(prior_alpha, prior_beta) <= 0.0:
        raise ValueError("relationship counts must be non-negative and priors positive")
    alpha = prior_alpha + success
    beta = prior_beta + failure
    return alpha / (alpha + beta), success + failure


def estimate_reliability_for_activity(
    reliability: np.ndarray, activity: Sequence[float], threshold: float = 0.5,
) -> dict[str, Any]:
    """Summarize conditional reliability for links whose source populations are active now."""
    probabilities = np.asarray(reliability, dtype=np.float64)
    vector = np.asarray(activity, dtype=np.float64).reshape(-1)
    if probabilities.shape != (vector.size, vector.size):
        raise ValueError("reliability matrix must be square and match activity size")
    source_indices = np.flatnonzero(vector >= threshold)
    if source_indices.size:
        selected = probabilities[source_indices, :]
    else:
        selected = probabilities
    values = selected.reshape(-1)
    mean = float(np.mean(values)) if values.size else 0.5
    entropy = -(values * np.log(np.clip(values, 1e-9, 1.0))
                + (1.0 - values) * np.log(np.clip(1.0 - values, 1e-9, 1.0)))
    return {"mean_reliability": mean,
            "mean_binary_entropy": float(np.mean(entropy) / math.log(2.0)) if values.size else 1.0,
            "active_source_units": [int(index) for index in source_indices],
            "relationship_count": int(values.size),
            "status": "estimated_from_accumulated_encounters"}


def compute_prediction_error(
    observed: Sequence[float] | None, predicted: Sequence[float],
) -> dict[str, Any]:
    """Return an explicit not-observed status or a target-minus-prediction error vector."""
    prediction = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if observed is None:
        return {"status": "not_observed", "error": None, "mse": None}
    target = np.asarray(observed, dtype=np.float64).reshape(-1)
    if (
        target.size == 0 or target.size != prediction.size
        or not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction))
    ):
        raise ValueError(
            "observed and predicted activity must be non-empty, finite, and match in dimensions"
        )
    error = target - prediction
    return {"status": "observed", "error": error, "mse": float(np.mean(np.square(error)))}


def calculate_modulatory_gate(state: Mapping[str, float], floor: float = 0.1) -> dict[str, Any]:
    """Combine supplied attention, novelty, reward, and arousal into a bounded learning gate."""
    keys = ("attention", "novelty", "reward", "arousal")
    values = {key: float(state.get(key, 0.5)) for key in keys}
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError("modulatory values must be finite and in [0, 1]")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("gate floor must lie in [0, 1]")
    raw = sum(values.values()) / len(values)
    gate = float(floor + (1.0 - floor) * raw)
    return {"gate": gate, "components": values,
            "interpretation": "bounded software summary of several modulatory influences"}


def timing_eligibility(interval_ms: float, tau_ms: float) -> float:
    """Return a signed STDP-inspired timing factor with exponential decay by absolute interval."""
    if not math.isfinite(interval_ms) or not math.isfinite(tau_ms) or tau_ms <= 0.0:
        raise ValueError("timing interval must be finite and tau must be positive")
    if interval_ms == 0.0:
        return 1.0
    magnitude = math.exp(-abs(interval_ms) / tau_ms)
    return magnitude if interval_ms > 0.0 else -magnitude


def bounded_update(delta: np.ndarray, max_step: float) -> tuple[np.ndarray, dict[str, float]]:
    """Clip local weight increments and report their pre/post clipping magnitudes."""
    update = np.asarray(delta, dtype=np.float64)
    if not np.all(np.isfinite(update)) or not math.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("update must be finite and max_step positive")
    bounded = np.clip(update, -max_step, max_step)
    return bounded, {"raw_max_abs_step": float(np.max(np.abs(update))) if update.size else 0.0,
                     "bounded_max_abs_step": float(np.max(np.abs(bounded))) if bounded.size else 0.0,
                     "clipped_fraction": float(np.mean(np.abs(update) > max_step)) if update.size else 0.0}


def mean_squared_error(observed: Sequence[float], predicted: Sequence[float]) -> float:
    """Calculate mean squared error for an observed and predicted activity pattern."""
    target = np.asarray(observed, dtype=np.float64).reshape(-1)
    prediction = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if target.size != prediction.size or target.size == 0:
        raise ValueError("evaluation vectors must be non-empty and have equal dimensions")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction)):
        raise ValueError("evaluation vectors must be finite")
    return float(np.mean(np.square(target - prediction)))


def beta_uncertainty(alpha: np.ndarray, beta: np.ndarray) -> dict[str, Any]:
    """Summarize Beta-style relationship uncertainty without implying neural probability storage."""
    a = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    if a.shape != b.shape or np.any(a <= 0.0) or np.any(b <= 0.0):
        raise ValueError("Beta parameters must have equal shapes and be positive")
    mean = a / (a + b)
    entropy = -(mean * np.log(np.clip(mean, 1e-9, 1.0))
                + (1.0 - mean) * np.log(np.clip(1.0 - mean, 1e-9, 1.0)))
    return {"mean_probability": float(np.mean(mean)),
            "mean_binary_entropy": float(np.mean(entropy) / math.log(2.0)),
            "mean_evidence_count": float(np.mean(a + b - 2.0)),
            "representation": "Beta posterior used as a statistical uncertainty proxy"}
