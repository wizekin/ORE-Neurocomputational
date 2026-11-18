"""Functional feature, attention, temporal, novelty, graph, and trace-quality operations.

The numeric expressions are transparent computational approximations inspired by the source
algorithm. They are not claims that biological neurons explicitly calculate the named metrics.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import EventInput, EncodingConfig


def normalize_vector(values: Sequence[float] | np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Return an L2-normalized finite vector, preserving an all-zero input as zeros."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError("vector values must be finite")
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, epsilon) if norm > epsilon else np.zeros_like(vector)


def sigmoid(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Apply a numerically stable logistic nonlinearity to each model activation."""
    vector = np.asarray(values, dtype=np.float64)
    clipped = np.clip(vector, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def softmax(values: Sequence[float] | np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert finite competition scores into a normalized probability-like weight vector."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("softmax requires a non-empty finite score vector")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    scaled = (vector - np.max(vector)) / temperature
    exponentials = np.exp(np.clip(scaled, -60.0, 0.0))
    return exponentials / max(float(np.sum(exponentials)), 1e-12)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute bounded cosine similarity after zero-padding two unequal feature vectors."""
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("similarity vectors must contain only finite values")
    size = max(a.size, b.size)
    if size == 0:
        return 1.0
    a = np.pad(a, (0, size - a.size))
    b = np.pad(b, (0, size - b.size))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def extract_parallel_features(sensory_signals: Mapping[str, Sequence[float]]) -> dict[str, np.ndarray]:
    """Normalize each observed sensory channel without collapsing its modality-specific identity."""
    if not sensory_signals:
        raise ValueError("at least one sensory channel is required")
    return {name: normalize_vector(vector) for name, vector in sorted(sensory_signals.items())}


def _padded_weighted_pool(vectors: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Pool unequal-length vectors after zero-padding while preserving weighted contribution."""
    if not vectors:
        return np.zeros(1, dtype=np.float64)
    width = max(vector.size for vector in vectors)
    total = np.zeros(width, dtype=np.float64)
    weight_sum = 0.0
    for vector, weight in zip(vectors, weights):
        total[:vector.size] += float(weight) * vector
        weight_sum += float(weight)
    return total / max(weight_sum, 1e-12)


def apply_attention_gain(
    features: Mapping[str, np.ndarray],
    goals: Mapping[str, float],
    salience: Mapping[str, float],
    uncertainty: Mapping[str, float],
    temperature: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Allocate relative channel gain from goal relevance, salience, and uncertainty."""
    names = list(features)
    if not names:
        raise ValueError("attention requires at least one feature channel")
    scores = np.asarray([
        float(goals.get(name, 0.0)) + float(salience.get(name, 0.0))
        + (1.0 - float(uncertainty.get(name, 0.5)))
        for name in names
    ], dtype=np.float64)
    weights_array = softmax(scores, temperature)
    weights = {name: float(weight) for name, weight in zip(names, weights_array)}
    gain = {name: float(np.clip(weight * len(names), 0.0, 2.0)) for name, weight in weights.items()}
    attended = {name: vector * gain[name] for name, vector in features.items()}
    return weights, attended


def build_item_representations(
    event: EventInput, attended_features: Mapping[str, np.ndarray], attention_weights: Mapping[str, float]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Construct per-item feature conjunctions and a pooled perirhinal-like item code."""
    component_codes: dict[str, np.ndarray] = {}
    for index, component in enumerate(event.components):
        raw = component.get("features")
        if raw is None:
            continue
        item_id = str(component.get("id", f"component-{index + 1}"))
        component_codes[item_id] = normalize_vector(raw)
    if component_codes:
        item_vectors = list(component_codes.values())
        item_weights = [1.0] * len(item_vectors)
    else:
        item_vectors = [attended_features[name] for name in attended_features]
        item_weights = [float(attention_weights[name]) for name in attended_features]
    pooled = normalize_vector(_padded_weighted_pool(item_vectors, item_weights))
    return pooled, component_codes


def build_context_representation(event: EventInput, attended_features: Mapping[str, np.ndarray]) -> np.ndarray:
    """Construct a scene/context code from explicit context and scene-like observed channels."""
    context_parts = [normalize_vector(event.context)] if event.context else []
    for name, vector in attended_features.items():
        if any(token in name.lower() for token in ("scene", "spatial", "environment", "context")):
            context_parts.append(normalize_vector(vector))
    if not context_parts:
        context_parts = [normalize_vector(vector) for vector in attended_features.values()]
    return normalize_vector(np.concatenate(context_parts))


def build_temporal_representation(
    timestamp_ms: float, component_count: int, frequency_count: int
) -> np.ndarray:
    """Encode elapsed model time with a multiscale sinusoidal basis plus component positions."""
    if timestamp_ms < 0.0 or component_count < 0 or frequency_count < 1:
        raise ValueError("temporal arguments are outside their valid ranges")
    seconds = float(timestamp_ms) / 1000.0
    frequencies = np.logspace(-1.0, 1.0, num=frequency_count, dtype=np.float64)
    phases = 2.0 * np.pi * frequencies * seconds
    time_code = np.column_stack((np.sin(phases), np.cos(phases))).reshape(-1)
    if component_count:
        positions = np.linspace(0.0, 1.0, component_count, dtype=np.float64)
        return normalize_vector(np.concatenate((time_code, positions)))
    return normalize_vector(time_code)


def converge_entorhinal_streams(
    item_code: Sequence[float], context_code: Sequence[float], temporal_code: Sequence[float]
) -> np.ndarray:
    """Concatenate normalized item, context, and temporal streams as an EC input abstraction."""
    parts = [normalize_vector(item_code), normalize_vector(context_code), normalize_vector(temporal_code)]
    return normalize_vector(np.concatenate(parts))


def estimate_novelty(
    current_code: Sequence[float], prior_codes: Sequence[Sequence[float]]
) -> dict[str, Any]:
    """Estimate distinctiveness against stored codes and report the closest prior episode."""
    if not prior_codes:
        return {"score": 1.0, "nearest_similarity": None, "nearest_episode_id": None,
                "comparison_count": 0, "status": "no_prior_episode_available"}
    similarities = [cosine_similarity(current_code, prior) for prior in prior_codes]
    best_index = int(np.argmax(similarities))
    nearest_similarity = float(similarities[best_index])
    return {"score": float(np.clip(1.0 - nearest_similarity, 0.0, 1.0)),
            "nearest_similarity": nearest_similarity, "nearest_index": best_index,
            "comparison_count": len(prior_codes), "status": "compared_with_prior_episodes"}


def detect_event_boundary(
    current_code: Sequence[float], previous_code: Sequence[float] | None,
    current_context: Sequence[float], previous_context: Sequence[float] | None,
    goal_relevance: float, threshold: float,
) -> dict[str, Any]:
    """Score feature/context change and mark a boundary while retaining the prior context."""
    if previous_code is None:
        return {"boundary": True, "score": 1.0, "feature_change": None,
                "context_change": None, "goal_relevance": float(goal_relevance),
                "reason": "first_observed_event"}
    feature_change = 0.5 * (1.0 - cosine_similarity(current_code, previous_code))
    context_change = 0.0 if previous_context is None else 0.5 * (
        1.0 - cosine_similarity(current_context, previous_context)
    )
    score = float(np.clip(0.65 * feature_change + 0.25 * context_change
                          + 0.10 * float(goal_relevance), 0.0, 1.0))
    boundary = score >= threshold
    return {"boundary": boundary, "score": score, "feature_change": float(feature_change),
            "context_change": float(context_change), "goal_relevance": float(goal_relevance),
            "reason": "change_score_exceeded_threshold" if boundary else "event_continuity_candidate"}


def build_relational_event_index(
    event: EventInput, ca1_code: Sequence[float], item_codes: Mapping[str, np.ndarray],
    context_code: Sequence[float], temporal_code: Sequence[float],
) -> dict[str, Any]:
    """Bind supplied item/context/time nodes and observed relations to one hippocampal event index."""
    nodes: list[dict[str, Any]] = [{"id": "hippocampal-index", "kind": "event_index",
                                   "label": event.event_id, "code": normalize_vector(ca1_code).tolist()}]
    edges: list[dict[str, Any]] = []
    for item_id, code in item_codes.items():
        nodes.append({"id": item_id, "kind": "item", "code": code.tolist()})
        edges.append({"source": "hippocampal-index", "type": "indexes", "target": item_id})
    nodes.extend([
        {"id": "event-context", "kind": "context", "code": normalize_vector(context_code).tolist()},
        {"id": "event-time", "kind": "temporal", "code": normalize_vector(temporal_code).tolist()},
    ])
    edges.extend([
        {"source": "hippocampal-index", "type": "occurred_in", "target": "event-context"},
        {"source": "hippocampal-index", "type": "occurred_at_time", "target": "event-time"},
    ])
    for relation in event.relations:
        if relation.get("source") is not None and relation.get("target") is not None:
            edges.append({"source": str(relation["source"]), "type": str(relation.get("type", "related_to")),
                          "target": str(relation["target"]), "evidence": "caller_supplied"})
    ordered_ids = [str(item.get("id", f"component-{index + 1}")) for index, item in enumerate(event.components)]
    for left, right in zip(ordered_ids, ordered_ids[1:]):
        edges.append({"source": left, "type": "preceded", "target": right,
                      "evidence": "caller_supplied_component_order"})
    return {"event_id": event.event_id, "nodes": nodes, "edges": edges,
            "hippocampal_index": normalize_vector(ca1_code).tolist(),
            "storage_interpretation": "computational relational index; not literal symbolic neuronal storage"}


def recurrent_cortical_update(
    previous_state: Sequence[float] | None, ca1_code: Sequence[float], feedback_gain: float = 0.35
) -> np.ndarray:
    """Mix a prior distributed cortical state with CA1 output as a recurrent network surrogate."""
    current = normalize_vector(ca1_code)
    if previous_state is None:
        return current
    previous = normalize_vector(previous_state)
    width = max(previous.size, current.size)
    previous = np.pad(previous, (0, width - previous.size))
    current = np.pad(current, (0, width - current.size))
    gain = float(np.clip(feedback_gain, 0.0, 1.0))
    return normalize_vector((1.0 - gain) * previous + gain * current)


def assess_initial_trace_quality(
    novelty: float, event_index: Mapping[str, Any], plasticity_magnitude: float,
    ca1_code: Sequence[float],
) -> dict[str, Any]:
    """Estimate initial trace distinctiveness, relational binding, plasticity, and activity stability."""
    node_count = len(event_index.get("nodes", []))
    edge_count = len(event_index.get("edges", []))
    binding = float(np.clip(edge_count / max(node_count * 1.5, 1.0), 0.0, 1.0))
    plasticity = float(np.clip(plasticity_magnitude * 20.0, 0.0, 1.0))
    stability = float(np.clip(np.mean(np.abs(np.asarray(ca1_code, dtype=np.float64))), 0.0, 1.0))
    weights = {"distinctiveness": 0.35, "binding": 0.30, "plasticity": 0.20, "stability": 0.15}
    score = (weights["distinctiveness"] * float(np.clip(novelty, 0.0, 1.0))
             + weights["binding"] * binding + weights["plasticity"] * plasticity
             + weights["stability"] * stability)
    return {"score": float(np.clip(score, 0.0, 1.0)), "components": {
                "distinctiveness": float(novelty), "binding": binding,
                "plasticity": plasticity, "stability": stability},
            "weights": weights, "persistence_claim": "not_inferred",
            "status": "initial_trace_estimated_not_permanent_storage"}


def stdp_weight_delta(
    pre_spike_times_ms: Sequence[float], post_spike_times_ms: Sequence[float],
    potentiation_amplitude: float = 0.01, depression_amplitude: float = 0.012,
    potentiation_tau_ms: float = 20.0, depression_tau_ms: float = 20.0,
) -> float:
    """Compute a pair-based STDP teaching-rule delta from supplied pre/post spike times."""
    if potentiation_tau_ms <= 0.0 or depression_tau_ms <= 0.0:
        raise ValueError("STDP time constants must be positive")
    delta = 0.0
    for pre_time in pre_spike_times_ms:
        for post_time in post_spike_times_ms:
            difference = float(post_time) - float(pre_time)
            if difference > 0.0:
                delta += potentiation_amplitude * math.exp(-difference / potentiation_tau_ms)
            elif difference < 0.0:
                delta -= depression_amplitude * math.exp(difference / depression_tau_ms)
    return float(delta)
