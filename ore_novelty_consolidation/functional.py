"""Functional mathematics used by the Point 05 consolidation pipeline.

The equations are explicit, inspectable model operations. They are not asserted to be literal
computations performed by neurons or synapses.
"""
from __future__ import annotations

import math
from typing import Sequence


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Bound a finite model value to a closed interval."""
    if not math.isfinite(float(value)):
        raise ValueError("value must be finite")
    if low > high:
        raise ValueError("low must not exceed high")
    return min(high, max(low, float(value)))


def sigmoid(value: float) -> float:
    """Compute a numerically stable logistic nonlinearity."""
    clipped = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-clipped))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity, returning zero when either vector has no magnitude."""
    if len(left) != len(right) or not left:
        raise ValueError("vectors must be non-empty and have equal length")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) ** 2 for a in left))
    right_norm = math.sqrt(sum(float(b) ** 2 for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return clamp(dot / (left_norm * right_norm), -1.0, 1.0)


def normalized_prediction_error(
    observed: Sequence[float], predicted: Sequence[float]
) -> dict[str, float]:
    """Compute vector error, root-mean-square deviation, and a bounded mismatch estimate."""
    if len(observed) != len(predicted) or not observed:
        raise ValueError("observed and predicted vectors must have equal non-zero length")
    squared = [(float(x) - float(y)) ** 2 for x, y in zip(observed, predicted)]
    rms = math.sqrt(sum(squared) / len(squared))
    mismatch = 1.0 - math.exp(-rms)
    return {
        "rms_prediction_error": rms,
        "normalized_mismatch": clamp(mismatch),
        "mean_squared_error": sum(squared) / len(squared),
    }


def information_surprise(probability: float) -> float:
    """Compute self-information from a bounded predictive probability."""
    probability = clamp(probability)
    return -math.log(max(probability, 1e-12))


def classify_novelty(
    features: Sequence[float],
    prior_patterns: Sequence[Sequence[float]],
    predicted_features: Sequence[float] | None,
    novelty_threshold: float,
    common_similarity_threshold: float,
) -> dict[str, float | str | bool | None]:
    """Estimate familiarity, mismatch, and cautious novelty class from prior representations."""
    if not features:
        raise ValueError("features must not be empty")
    similarities = [cosine_similarity(features, pattern) for pattern in prior_patterns]
    familiarity = (max(((item + 1.0) / 2.0 for item in similarities))
                   if similarities else None)
    prediction = None
    if predicted_features is not None:
        prediction = normalized_prediction_error(features, predicted_features)
    if familiarity is None and prediction is None:
        return {
            "novelty_type": "undetermined",
            "novelty_score": 0.5,
            "familiarity": None,
            "mismatch": None,
            "surprise": None,
            "classification_confident": False,
            "reason": "no_prior_pattern_or_prediction",
        }
    if familiarity is None:
        mismatch = float(prediction["normalized_mismatch"])
        novelty = mismatch
        novelty_type = "undetermined" if novelty < novelty_threshold else "unclassified_novelty"
        confidence = False
        reason = "prediction_only_no_familiarity_reference"
    else:
        novelty = clamp(1.0 - familiarity)
        mismatch = (float(prediction["normalized_mismatch"])
                    if prediction is not None else novelty)
        if novelty < novelty_threshold:
            novelty_type = "familiar"
        elif familiarity >= common_similarity_threshold:
            novelty_type = "common"
        else:
            novelty_type = "distinct"
        confidence = True
        reason = "similarity_to_prior_knowledge_and_prediction"
    predictive_probability = clamp(math.exp(-mismatch))
    return {
        "novelty_type": novelty_type,
        "novelty_score": clamp(novelty),
        "familiarity": familiarity,
        "mismatch": clamp(mismatch),
        "surprise": information_surprise(predictive_probability),
        "classification_confident": confidence,
        "reason": reason,
    }


def temporal_eligibility(elapsed_ms: float, tau_ms: float) -> float:
    """Estimate residual temporal eligibility using an exponential time-window model."""
    if elapsed_ms < 0.0 or tau_ms <= 0.0:
        raise ValueError("elapsed_ms must be non-negative and tau_ms positive")
    return math.exp(-float(elapsed_ms) / float(tau_ms))


def decay_state(value: float, elapsed_ms: float, tau_ms: float) -> float:
    """Apply exponential decay to a transient biochemical or eligibility state."""
    return clamp(value) * temporal_eligibility(elapsed_ms, tau_ms)


def hill_occupancy(concentration: float, affinity: float, coefficient: float = 1.0) -> float:
    """Estimate nonlinear receptor occupancy with a Hill-type binding approximation."""
    if concentration < 0.0 or affinity <= 0.0 or coefficient <= 0.0:
        raise ValueError("concentration must be non-negative and affinity/coefficient positive")
    numerator = float(concentration) ** float(coefficient)
    denominator = float(affinity) ** float(coefficient) + numerator
    return clamp(numerator / denominator) if denominator else 0.0


def tagged_resource_capture(
    tag_strength: float, local_resource: float, compatibility: float
) -> float:
    """Compute tag/resource/compatibility coincidence for one learning-active synapse."""
    return clamp(tag_strength) * clamp(local_resource) * clamp(compatibility)


def stabilization_probability(capture: float, threshold: float, slope: float) -> float:
    """Convert accumulated synaptic capture into a graded stabilization probability."""
    if capture <= 0.0:
        return 0.0
    return sigmoid(slope * (float(capture) - float(threshold)))


def aggregate_feature_support(
    values: Sequence[float], indices: Sequence[int], width: int
) -> tuple[float, ...]:
    """Pool synapse persistence estimates into feature-aligned ensemble support."""
    if len(values) != len(indices) or width < 1:
        raise ValueError("values and indices must align and width must be positive")
    sums = [0.0] * width
    counts = [0.0] * width
    for value, index in zip(values, indices):
        if index < 0 or index >= width:
            raise ValueError("feature index is outside the requested width")
        sums[index] += clamp(value)
        counts[index] += 1.0
    return tuple(sums[i] / counts[i] if counts[i] else 0.0 for i in range(width))


def estimate_interference(
    target_pattern: Sequence[float],
    competitors: Sequence[tuple[Sequence[float], float, float]],
    sensitivity: float = 1.0,
) -> dict[str, object]:
    """Estimate similarity-weighted retrieval competition without erasing either trace."""
    burden = 0.0
    rows: list[dict[str, float | str]] = []
    for index, (pattern, strength, conflict) in enumerate(competitors):
        overlap = max(0.0, cosine_similarity(target_pattern, pattern))
        contribution = overlap * clamp(strength) * clamp(conflict)
        burden += contribution
        rows.append({
            "competitor_index": str(index),
            "overlap": overlap,
            "strength": clamp(strength),
            "conflict": clamp(conflict),
            "burden": contribution,
        })
    risk = clamp(1.0 - math.exp(-max(0.0, sensitivity) * burden))
    return {"burden": burden, "risk": risk, "competitors": rows}


def softmax_probabilities(scores: Sequence[float], temperature: float) -> tuple[float, ...]:
    """Normalize candidate retrieval scores with a temperature-controlled softmax."""
    if not scores or temperature <= 0.0:
        raise ValueError("scores must be non-empty and temperature positive")
    scaled = [float(score) / float(temperature) for score in scores]
    maximum = max(scaled)
    values = [math.exp(max(-60.0, min(60.0, item - maximum))) for item in scaled]
    total = sum(values)
    return tuple(value / total for value in values)


def retrieval_score(
    persistence: float,
    cue_match: float,
    context_match: float,
    interference_risk: float,
    persistence_weight: float,
    match_weight: float,
    context_weight: float,
    interference_weight: float,
) -> float:
    """Combine latent trace persistence, cue/context compatibility, and retrieval competition."""
    return (
        persistence_weight * clamp(persistence)
        + match_weight * clamp(cue_match)
        + context_weight * clamp(context_match)
        - interference_weight * clamp(interference_risk)
    )
