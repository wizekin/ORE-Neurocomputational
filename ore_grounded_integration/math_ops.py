"""Numerical operations used by grounded semantic integration.

Each function is a small, inspectable mathematical primitive. Neural descriptions are functional
analogies: the software uses vectors, probability normalization and explicit update equations,
not a claim that biological neurons execute these exact algorithms.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def stable_softmax(scores: Mapping[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Convert finite named candidate scores into a stable normalized probability distribution."""
    if not scores:
        return {}
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    names = list(scores)
    values = np.asarray([float(scores[name]) for name in names], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("softmax scores must be finite")
    shifted = (values - np.max(values)) / float(temperature)
    exp_values = np.exp(np.clip(shifted, -700.0, 0.0))
    probabilities = exp_values / max(float(np.sum(exp_values)), np.finfo(float).tiny)
    return {name: float(value) for name, value in zip(names, probabilities)}


def unit_normalize(values: Sequence[float] | np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Normalize a finite numeric vector to unit length while preserving a zero vector safely."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError("vector values must be finite")
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, epsilon)


def zscore_normalize(values: Sequence[float] | np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Center and scale a finite feature vector using its own mean and standard deviation."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError("vector values must be finite")
    return (vector - float(np.mean(vector))) / (float(np.std(vector)) + epsilon)


def cosine_similarity(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> float:
    """Compute cosine similarity for equal-dimensional vectors with a defined zero-vector result."""
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"vector shapes differ: {a.shape} != {b.shape}")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def hashed_text_vector(text: str, dimension: int = 48) -> np.ndarray:
    """Encode tokens into a deterministic signed feature-hash vector without external NLP weights."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    values = np.zeros(dimension, dtype=np.float64)
    tokens = re.findall(r"[\w'-]+", str(text).casefold())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        index = int.from_bytes(digest[:8], "big") % dimension
        sign = 1.0 if digest[8] & 1 else -1.0
        values[index] += sign
    return unit_normalize(values)


def feature_vector(value: Any, dimension: int = 48) -> np.ndarray:
    """Convert a named feature map, text string or numeric array into a fixed-size feature-hash vector.

    Numeric arrays are reduced into evenly sized bins. Named features are hashed to stable
    dimensions. This is a transparent baseline encoder, not a substitute for vision or audio
    models trained on raw pixels or waveforms.
    """
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if value is None:
        return np.zeros(dimension, dtype=np.float64)
    if isinstance(value, str):
        return hashed_text_vector(value, dimension)
    output = np.zeros(dimension, dtype=np.float64)
    if isinstance(value, Mapping):
        for name, raw in value.items():
            digest = hashlib.blake2b(str(name).casefold().encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            try:
                numeric = float(raw)
            except (TypeError, ValueError):
                numeric_array = np.asarray(raw, dtype=np.float64).reshape(-1)
                numeric = float(np.mean(numeric_array)) if numeric_array.size else 0.0
            if not math.isfinite(numeric):
                raise ValueError(f"feature {name!r} must be finite")
            output[index] += sign * numeric
        return unit_normalize(output)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("numeric modality input must contain only finite values")
    if array.size == 0:
        return output
    bins = np.array_split(array, dimension)
    output = np.asarray([float(np.mean(part)) if part.size else 0.0 for part in bins])
    return unit_normalize(output)


def weighted_vector_sum(vectors: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    """Calculate a reliability-weighted sum of equal-sized modality vectors."""
    if not vectors:
        return np.zeros(0, dtype=np.float64)
    dimension = next(iter(vectors.values())).shape
    total = np.zeros(dimension, dtype=np.float64)
    weight_total = 0.0
    for name, vector in vectors.items():
        array = np.asarray(vector, dtype=np.float64).reshape(-1)
        if array.shape != dimension:
            raise ValueError("all modality vectors must have the same dimension")
        weight = max(0.0, float(weights.get(name, 0.0)))
        total += weight * array
        weight_total += weight
    if weight_total <= 1e-12:
        return np.mean(np.stack(list(vectors.values())), axis=0)
    return total / weight_total


def temporal_association(delta_seconds: float, tau_seconds: float) -> float:
    """Return exponential temporal co-occurrence weight exp(-absolute time gap / tau)."""
    delta = abs(float(delta_seconds))
    tau = float(tau_seconds)
    if not math.isfinite(delta) or not math.isfinite(tau) or tau <= 0:
        raise ValueError("time gap must be finite and tau must be finite and positive")
    return float(math.exp(-delta / tau))


def contrastive_alignment_loss(
    positive_similarity: float,
    negative_similarities: Sequence[float],
    margin: float = 0.25,
) -> float:
    """Compute a margin-ranking loss that rewards same-event and penalizes mismatched modality pairs."""
    positive = float(positive_similarity)
    negatives = [float(value) for value in negative_similarities]
    if not all(math.isfinite(value) for value in [positive, *negatives]):
        raise ValueError("similarities must be finite")
    hardest_negative = max(negatives, default=-1.0)
    return float(max(0.0, float(margin) - positive + hardest_negative))


def prediction_residual(observed: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    """Measure a vector prediction residual and return its signed and Euclidean error components."""
    actual = np.asarray(observed, dtype=np.float64).reshape(-1)
    estimate = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual.shape != estimate.shape:
        raise ValueError("observed and predicted vectors must have equal dimensions")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(estimate)):
        raise ValueError("prediction vectors must be finite")
    residual = actual - estimate
    return {
        "residual": residual,
        "squared_error": float(0.5 * np.dot(residual, residual)),
        "absolute_error": float(np.mean(np.abs(residual))) if residual.size else 0.0,
        "euclidean_error": float(np.linalg.norm(residual)),
    }


def hebbian_update(weights: np.ndarray, pre_activity: np.ndarray, post_activity: np.ndarray,
                   learning_rate: float, decay: float = 0.0) -> np.ndarray:
    """Apply a bounded co-activity update DeltaW = eta * outer(post, pre) - decay * W."""
    matrix = np.asarray(weights, dtype=np.float64)
    pre = np.asarray(pre_activity, dtype=np.float64).reshape(-1)
    post = np.asarray(post_activity, dtype=np.float64).reshape(-1)
    if matrix.shape != (post.size, pre.size):
        raise ValueError("weight matrix shape must equal (post units, pre units)")
    if not all(np.all(np.isfinite(item)) for item in (matrix, pre, post)):
        raise ValueError("Hebbian inputs must be finite")
    updated = matrix + float(learning_rate) * np.outer(post, pre) - float(decay) * matrix
    return np.clip(updated, -1.0, 1.0)


def temporal_difference_error(reward: float, next_value: float, current_value: float,
                              discount: float = 0.9) -> float:
    """Calculate TD error delta = reward + discount * next value - current value."""
    values = [float(reward), float(next_value), float(current_value), float(discount)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("temporal-difference inputs must be finite")
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must lie in [0, 1]")
    return float(reward + discount * next_value - current_value)


def categorical_entropy(probabilities: Mapping[str, float], epsilon: float = 1e-12) -> float:
    """Compute Shannon entropy in nats from a normalized or normalizable non-negative distribution."""
    if not probabilities:
        return 0.0
    values = np.asarray([float(value) for value in probabilities.values()], dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(np.sum(values))
    if total <= epsilon:
        return 0.0
    normalized = values / total
    return float(-np.sum(normalized * np.log(np.maximum(normalized, epsilon))))


def reliability_from_error(error: float) -> float:
    """Map non-negative prediction error to a bounded reliability score 1/(1+error)."""
    value = float(error)
    if not math.isfinite(value) or value < 0:
        raise ValueError("error must be finite and non-negative")
    return float(1.0 / (1.0 + value))
