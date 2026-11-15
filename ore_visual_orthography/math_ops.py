"""Numerical operations for flexible but order-preserving orthographic representations."""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np


def softmax(values: np.ndarray, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """Convert activation scores to a stable probability distribution along one axis."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.zeros_like(array)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    scaled = array / float(temperature)
    shifted = scaled - np.max(scaled, axis=axis, keepdims=True)
    exponentials = np.exp(np.clip(shifted, -80.0, 0.0))
    return exponentials / np.maximum(np.sum(exponentials, axis=axis, keepdims=True), 1e-12)


def normalize_vector(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Scale a numeric feature vector to unit Euclidean norm when it contains evidence."""
    array = np.asarray(values, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(array))
    return array / max(norm, float(epsilon))


def normalized_positions(centers: Sequence[float]) -> np.ndarray:
    """Map observed horizontal centers to relative positions in the closed interval [0, 1]."""
    values = np.asarray(centers, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("letter centers must be finite")
    if values.size == 1:
        return np.asarray([0.5], dtype=np.float64)
    minimum = float(np.min(values))
    span = float(np.max(values) - minimum)
    if span <= 1e-12:
        return np.linspace(0.0, 1.0, values.size, dtype=np.float64)
    return (values - minimum) / span


def gaussian_position_distributions(
    centers: Sequence[float], uncertainty: float, grid_size: int = 32
) -> np.ndarray:
    """Represent each letter's approximate location with a normalized Gaussian population."""
    if uncertainty <= 0.0 or grid_size < 2:
        raise ValueError("uncertainty must be positive and grid_size must be >= 2")
    positions = np.linspace(0.0, 1.0, int(grid_size), dtype=np.float64)
    means = np.asarray(centers, dtype=np.float64)
    if means.size == 0:
        return np.zeros((0, grid_size), dtype=np.float64)
    logits = -0.5 * ((positions[None, :] - means[:, None]) / float(uncertainty)) ** 2
    probabilities = np.exp(np.clip(logits, -80.0, 0.0))
    return probabilities / np.maximum(np.sum(probabilities, axis=1, keepdims=True), 1e-12)


def matrix_from_evidence(
    evidence: Sequence[Mapping[str, float]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Vectorize all per-position letter hypotheses into one shared alphabet activation matrix."""
    alphabet = tuple(sorted({str(label) for row in evidence for label in row}))
    if not evidence or not alphabet:
        return np.zeros((len(evidence), 0), dtype=np.float64), alphabet
    lookup = {label: index for index, label in enumerate(alphabet)}
    matrix = np.zeros((len(evidence), len(alphabet)), dtype=np.float64)
    for row_index, row in enumerate(evidence):
        for label, strength in row.items():
            matrix[row_index, lookup[str(label)]] = max(0.0, float(strength))
        total = float(np.sum(matrix[row_index]))
        if total > 0.0:
            matrix[row_index] /= total
    return matrix, alphabet


def position_specific_conjunctions(
    evidence: Sequence[Mapping[str, float]],
    positions: Sequence[float],
    distance_sigma: float = 0.12,
) -> dict[str, object]:
    """Generate ordered letter conjunctions weighted by learned relative-spacing compatibility."""
    if distance_sigma <= 0.0:
        raise ValueError("distance_sigma must be positive")
    count = min(len(evidence), len(positions))
    pair_rows: list[dict[str, object]] = []
    aggregate: dict[str, float] = {}
    denominator = max(count - 1, 1)
    for left in range(count):
        for right in range(left + 1, count):
            observed_gap = float(positions[right] - positions[left])
            expected_gap = (right - left) / denominator
            tuning = float(np.exp(-0.5 * ((observed_gap - expected_gap) / distance_sigma) ** 2))
            for first, first_probability in evidence[left].items():
                for second, second_probability in evidence[right].items():
                    activation = float(first_probability) * float(second_probability) * tuning
                    pair = f"{first}{second}"
                    aggregate[pair] = aggregate.get(pair, 0.0) + activation
                    pair_rows.append({
                        "pair": pair,
                        "start_index": left,
                        "end_index": right,
                        "start_position": float(positions[left]),
                        "end_position": float(positions[right]),
                        "relative_gap": observed_gap,
                        "spacing_tuning": tuning,
                        "activation": activation,
                    })
    return {"pairs": pair_rows, "pair_activations": aggregate}


def open_bigram_activations(
    evidence: Sequence[Mapping[str, float]], distance_decay: float = 0.35
) -> dict[str, object]:
    """Build all forward ordered letter pairs while softly discounting distant positions."""
    if distance_decay < 0.0:
        raise ValueError("distance_decay must be non-negative")
    pair_rows: list[dict[str, object]] = []
    aggregate: dict[str, float] = {}
    for left in range(len(evidence)):
        for right in range(left + 1, len(evidence)):
            distance = right - left
            weight = float(np.exp(-distance_decay * max(0, distance - 1)))
            for first, first_probability in evidence[left].items():
                for second, second_probability in evidence[right].items():
                    activation = float(first_probability) * float(second_probability) * weight
                    pair = f"{first}{second}"
                    aggregate[pair] = aggregate.get(pair, 0.0) + activation
                    pair_rows.append({
                        "pair": pair,
                        "start_index": left,
                        "end_index": right,
                        "distance": distance,
                        "distance_weight": weight,
                        "activation": activation,
                    })
    return {"pairs": pair_rows, "pair_activations": aggregate}


def jaccard_similarity(left: Sequence[str] | set[str], right: Sequence[str] | set[str]) -> float:
    """Calculate set-overlap similarity as intersection size divided by union size."""
    first, second = set(left), set(right)
    union = first | second
    return 1.0 if not union else len(first & second) / len(union)


def multiset_similarity(left: str, right: str) -> float:
    """Compare letter identity counts while deliberately ignoring their spatial positions."""
    first, second = Counter(left), Counter(right)
    union = sum((first | second).values())
    intersection = sum((first & second).values())
    return 1.0 if union == 0 else float(intersection / union)


def normalized_edit_similarity(left: str, right: str) -> float:
    """Compute normalized Levenshtein similarity while preserving character order."""
    if not left and not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for row, char_left in enumerate(left, start=1):
        current = [row]
        for column, char_right in enumerate(right, start=1):
            substitution = 0 if char_left == char_right else 1
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + substitution,
            ))
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right), 1)


def recurrent_letter_update(
    probabilities: np.ndarray,
    iterations: int = 8,
    damping: float = 0.35,
    lateral_inhibition: float = 0.15,
) -> tuple[np.ndarray, list[float]]:
    """Settle letter populations with neighbor feedback, lateral competition and damped updates."""
    current = np.asarray(probabilities, dtype=np.float64).copy()
    if current.ndim != 2:
        raise ValueError("probabilities must have shape [positions, letter_candidates]")
    if current.size == 0:
        return current, []
    if iterations < 1 or not 0.0 <= damping <= 1.0 or not 0.0 <= lateral_inhibition <= 1.0:
        raise ValueError("invalid recurrent-update parameters")
    row_sums = np.sum(current, axis=1, keepdims=True)
    current = current / np.maximum(row_sums, 1e-12)
    deltas: list[float] = []
    base_logits = np.log(np.maximum(current, 1e-9))
    for _ in range(iterations):
        neighbor_message = np.roll(current, 1, axis=0) + np.roll(current, -1, axis=0)
        competitor_activity = np.sum(current, axis=1, keepdims=True) - current
        logits = (
            base_logits
            + damping * neighbor_message
            - lateral_inhibition * competitor_activity
        )
        target = softmax(logits, temperature=1.0, axis=1)
        updated = (1.0 - damping) * current + damping * target
        updated /= np.maximum(np.sum(updated, axis=1, keepdims=True), 1e-12)
        deltas.append(float(np.linalg.norm(updated - current)))
        current = updated
    return current, deltas


def compare_orthographic_strings(left: str, right: str) -> dict[str, float]:
    """Compare two spellings through letter identity, ordered edit distance and open-bigram overlap."""
    first = "".join(character for character in left.casefold() if not character.isspace())
    second = "".join(character for character in right.casefold() if not character.isspace())
    first_bigrams = {first[i] + first[j] for i in range(len(first)) for j in range(i + 1, len(first))}
    second_bigrams = {second[i] + second[j] for i in range(len(second)) for j in range(i + 1, len(second))}
    bigram_score = jaccard_similarity(first_bigrams, second_bigrams)
    identity_score = multiset_similarity(first, second)
    position_score = normalized_edit_similarity(first, second)
    combined = 0.25 * identity_score + 0.35 * position_score + 0.40 * bigram_score
    return {
        "identity_multiset_similarity": float(identity_score),
        "ordered_edit_similarity": float(position_score),
        "bigram_jaccard_similarity": float(bigram_score),
        "combined_similarity": float(combined),
    }


def integrate_representation_channels(
    channels: Mapping[str, np.ndarray], weights: Mapping[str, float]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Normalize separate evidence channels and concatenate them using explicit contribution weights."""
    normalized: dict[str, np.ndarray] = {}
    pieces: list[np.ndarray] = []
    for name, values in channels.items():
        vector = normalize_vector(np.asarray(values, dtype=np.float64))
        normalized[name] = vector
        pieces.append(float(weights.get(name, 0.0)) * vector)
    if not pieces:
        return np.zeros(0, dtype=np.float64), normalized
    fused = np.concatenate(pieces)
    return normalize_vector(fused), normalized
