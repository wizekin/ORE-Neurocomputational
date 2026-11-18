"""Mathematical operations for episode alignment, semantic abstraction, and provenance.

These functions provide transparent computational analogies for recurring features, diagnosticity,
relation extraction, uncertainty, and temporal accessibility. None asserts that neurons literally
calculate cosine similarity, entropy, softmax, or exponential decay.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import EpisodeInput, FEATURE_GROUPS, MemoryTransformationConfig, RelationEvidence


GROUP_WEIGHTS = {
    "content": 1.00,
    "causal": 0.95,
    "sensory": 0.75,
    "affective": 0.70,
    "spatial": 0.55,
    "temporal": 0.50,
}


def sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    """Apply a numerically stable logistic transformation to scalar or array input."""
    if isinstance(value, np.ndarray):
        return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, float(value)))))


def entropy(probabilities: Sequence[float], epsilon: float = 1e-12) -> float:
    """Compute Shannon entropy in nats after normalizing a non-negative probability vector."""
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("entropy requires a non-empty finite non-negative vector")
    total = float(values.sum())
    if total <= epsilon:
        return 0.0
    probabilities_array = values / total
    positive = probabilities_array[probabilities_array > epsilon]
    return float(-np.sum(positive * np.log(positive)))


def normalise_nested_features(
    groups: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Return canonical feature groups, preserving missing values as unknown and zeroes as observed."""
    result = {group: {} for group in FEATURE_GROUPS}
    for group, features in groups.items():
        if group not in result:
            continue
        result[group] = {str(key): float(value) for key, value in features.items()}
    return result


def flatten_feature_groups(
    groups: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Flatten nested feature observations into stable group-qualified keys."""
    return {f"{group}:{feature}": float(value)
            for group, values in groups.items() for feature, value in values.items()}


def relation_signature(relation: RelationEvidence) -> tuple[str, str, str]:
    """Return the directed, typed identity of one relation without discarding its meaning."""
    return (relation.subject, relation.predicate, relation.object)


def relation_similarity(
    left: Sequence[RelationEvidence], right: Sequence[RelationEvidence]
) -> float:
    """Compute confidence-weighted Jaccard overlap for typed directed relations."""
    left_map = {relation_signature(item): item.confidence for item in left}
    right_map = {relation_signature(item): item.confidence for item in right}
    keys = set(left_map) | set(right_map)
    if not keys:
        return 1.0
    numerator = sum(min(left_map.get(key, 0.0), right_map.get(key, 0.0)) for key in keys)
    denominator = sum(max(left_map.get(key, 0.0), right_map.get(key, 0.0)) for key in keys)
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def weighted_feature_similarity(
    left: Mapping[str, Mapping[str, float]],
    right: Mapping[str, Mapping[str, float]],
) -> float:
    """Compute group-weighted cosine similarity with shared-key alignment and no missing-as-zero penalty."""
    group_scores: list[tuple[float, float]] = []
    for group in FEATURE_GROUPS:
        a = left.get(group, {})
        b = right.get(group, {})
        shared = sorted(set(a) | set(b))
        if not shared:
            continue
        av = np.asarray([a.get(key, 0.0) for key in shared], dtype=np.float64)
        bv = np.asarray([b.get(key, 0.0) for key in shared], dtype=np.float64)
        denominator = float(np.linalg.norm(av) * np.linalg.norm(bv))
        if denominator <= 1e-12:
            score = 1.0 if np.allclose(av, bv) else 0.0
        else:
            score = float(np.clip(np.dot(av, bv) / denominator, 0.0, 1.0))
        group_scores.append((GROUP_WEIGHTS[group], score))
    if not group_scores:
        return 0.0
    return float(sum(weight * score for weight, score in group_scores)
                 / sum(weight for weight, _ in group_scores))


def episode_similarity(left: EpisodeInput, right: EpisodeInput) -> dict[str, float]:
    """Compare two events using feature and relation overlap while exposing each component."""
    feature = weighted_feature_similarity(left.feature_groups(), right.feature_groups())
    relation = relation_similarity(left.relations, right.relations)
    score = 0.82 * feature + 0.18 * relation if left.relations or right.relations else feature
    return {"overall": float(np.clip(score, 0.0, 1.0)),
            "feature": feature, "relation": relation}


def cue_similarity(
    cue_features: Mapping[str, Mapping[str, float]],
    cue_relations: Sequence[RelationEvidence],
    episode: EpisodeInput,
) -> dict[str, float]:
    """Compare a partial query cue with an episode without treating unspecified fields as negative."""
    feature = weighted_feature_similarity(cue_features, episode.feature_groups()) if cue_features else 0.0
    relation = relation_similarity(cue_relations, episode.relations) if cue_relations else 0.0
    if cue_features and cue_relations:
        overall = 0.82 * feature + 0.18 * relation
    else:
        overall = feature if cue_features else relation
    return {"overall": overall, "feature": feature, "relation": relation}


def softmax(values: Sequence[float], temperature: float) -> np.ndarray:
    """Normalize finite competition scores into a stable probability-like vector."""
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if scores.size == 0 or not np.all(np.isfinite(scores)) or temperature <= 0.0:
        raise ValueError("softmax requires non-empty finite scores and positive temperature")
    exponents = np.exp(np.clip((scores - np.max(scores)) / temperature, -60.0, 0.0))
    return exponents / max(float(np.sum(exponents)), 1e-12)


def effective_sample_size(weights: Sequence[float]) -> float:
    """Estimate effective independent support from non-negative reliability weights."""
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(values < 0.0) or not np.all(np.isfinite(values)):
        return 0.0
    denominator = float(np.sum(values ** 2))
    return float(np.sum(values) ** 2 / denominator) if denominator > 1e-12 else 0.0


def binary_presence(value: float, threshold: float = 0.5) -> int:
    """Map an observed normalized feature value to binary presence for information-gain analysis."""
    return int(float(value) >= threshold)


def information_gain(
    values: Sequence[int], labels: Sequence[str], threshold: float = 0.5
) -> float:
    """Estimate label information gain from presence/absence of explicitly observed feature values."""
    if len(values) != len(labels) or not labels:
        return 0.0
    label_values = sorted(set(labels))
    counts = [labels.count(label) for label in label_values]
    base = entropy(counts)
    partitions: dict[int, list[str]] = {0: [], 1: []}
    for value, label in zip(values, labels):
        partitions[int(float(value) >= threshold)].append(label)
    total = len(labels)
    conditional = 0.0
    for partition in partitions.values():
        if partition:
            weight = len(partition) / total
            pcounts = [partition.count(label) for label in label_values]
            conditional += weight * entropy(pcounts)
    if base <= 1e-12:
        return 0.0
    return float(np.clip((base - conditional) / base, 0.0, 1.0))


def estimate_feature_statistics(
    episodes: Sequence[EpisodeInput],
    labels: Sequence[str],
    config: MemoryTransformationConfig,
    diagnostic_episodes: Sequence[EpisodeInput] = (),
    diagnostic_labels: Sequence[str] = (),
) -> dict[str, dict[str, Any]]:
    """Estimate target prevalence/variance plus contrastive information gain and evidence reliability."""
    if labels and len(labels) != len(episodes):
        raise ValueError("labels must align with target episodes")
    if diagnostic_labels and len(diagnostic_labels) != len(diagnostic_episodes):
        raise ValueError("diagnostic_labels must align with diagnostic episodes")
    keys = sorted({f"{group}:{feature}" for episode in episodes
                   for group, features in episode.feature_groups().items() for feature in features})
    output: dict[str, dict[str, Any]] = {}
    for key in keys:
        group, feature = key.split(":", 1)
        observations: list[float] = []
        weights: list[float] = []
        context_values: list[float] = []
        for episode in episodes:
            group_values = getattr(episode, f"{group}_features")
            if feature not in group_values:
                continue
            value = float(group_values[feature])
            confidence = episode.feature_confidence.get(group, {}).get(feature, 1.0)
            weight = episode.overall_confidence * episode.source_confidence * confidence
            weight *= min(episode.repetition_count, 5) ** 0.5
            observations.append(value)
            weights.append(weight)
            if group in {"spatial", "temporal"}:
                context_values.append(episode.context_importance.get(group, {}).get(
                    feature, config.context_default_importance))
        total_weight = float(sum(weights))
        prevalence = (float(np.dot(observations, weights) / total_weight)
                      if total_weight > 1e-12 else 0.0)
        variance = (float(np.average((np.asarray(observations) - prevalence) ** 2,
                                     weights=weights)) if total_weight > 1e-12 else 0.0)
        mean_confidence = (float(np.average(weights)) if weights else 0.0)
        diagnostic_values: list[int] = []
        diagnostic_class_labels: list[str] = []
        for item, label in list(zip(episodes, labels)) + list(zip(diagnostic_episodes, diagnostic_labels)):
            value_map = getattr(item, f"{group}_features")
            if feature in value_map:
                diagnostic_values.append(binary_presence(value_map[feature]))
                diagnostic_class_labels.append(label)
        diagnosticity = (information_gain(diagnostic_values, diagnostic_class_labels)
                         if diagnostic_class_labels else 0.0)
        if group in {"spatial", "temporal"}:
            context_importance = float(np.average(context_values, weights=weights)) if weights else (
                config.context_default_importance)
        else:
            context_importance = 1.0
        support = float(len(observations))
        effective_support = effective_sample_size(weights)
        output[key] = {
            "prevalence": prevalence,
            "variance": variance,
            "diagnosticity": diagnosticity,
            "mean_reliability": float(np.clip(mean_confidence, 0.0, 1.0)),
            "observed_episode_count": int(support),
            "effective_support": effective_support,
            "context_importance": context_importance,
            "is_invariant": bool(prevalence >= config.invariant_prevalence_threshold
                                 and variance <= 0.18
                                 and effective_support >= config.minimum_effective_support),
            "is_conflicting": bool(config.conflict_lower_threshold < prevalence
                                   < config.conflict_upper_threshold and variance > 0.02),
        }
    return output


def estimate_relation_statistics(
    episodes: Sequence[EpisodeInput],
) -> dict[str, dict[str, Any]]:
    """Estimate relation recurrence and reliability while preserving event-specific supporting IDs."""
    support: dict[tuple[str, str, str], list[tuple[str, float, str]]] = {}
    for episode in episodes:
        for relation in episode.relations:
            key = relation_signature(relation)
            weight = episode.overall_confidence * episode.source_confidence * relation.confidence
            support.setdefault(key, []).append((episode.episode_id, weight, relation.source_type))
    result: dict[str, dict[str, Any]] = {}
    for (subject, predicate, obj), items in sorted(support.items()):
        key = f"{subject}|{predicate}|{obj}"
        weights = [value for _, value, _ in items]
        result[key] = {
            "subject": subject, "predicate": predicate, "object": obj,
            "prevalence": len(items) / max(len(episodes), 1),
            "mean_reliability": float(np.mean(weights)) if weights else 0.0,
            "effective_support": effective_sample_size(weights),
            "supporting_episode_ids": [episode_id for episode_id, _, _ in items],
            "source_types": sorted({source for _, _, source in items}),
        }
    return result


def semantic_context_weight(
    diagnosticity: float,
    context_importance: float,
    prevalence: float,
    config: MemoryTransformationConfig,
) -> float:
    """Estimate context retention from diagnostic value, future utility, and recurring support."""
    logit = (config.context_logit_bias
             + config.context_prevalence_gain * float(prevalence)
             + config.context_utility_gain * float(context_importance)
             + config.context_meaning_gain * float(diagnosticity))
    return float(np.clip(sigmoid(logit), 0.0, 1.0))


def accessibility_decay(
    accessibility: float,
    elapsed_hours: float,
    decay_rate_per_hour: float,
    interference: float,
    interference_rate_per_hour: float,
) -> float:
    """Update retrieval accessibility with exponential decay and interference without deleting content."""
    if min(elapsed_hours, decay_rate_per_hour, interference, interference_rate_per_hour) < 0.0:
        raise ValueError("time and decay/interference values must be non-negative")
    exponent = -(decay_rate_per_hour * elapsed_hours
                 + interference_rate_per_hour * interference * elapsed_hours)
    return float(np.clip(accessibility * math.exp(max(-60.0, exponent)), 0.0, 1.0))


def source_provenance(
    supporting_episodes: Sequence[EpisodeInput],
    feature_key: str | None = None,
) -> dict[str, Any]:
    """Summarize origins and evidence IDs without converting inference into direct experience."""
    source_ids = sorted({episode.source_id for episode in supporting_episodes if episode.source_id})
    types = sorted({episode.source_type for episode in supporting_episodes})
    episode_ids = [episode.episode_id for episode in supporting_episodes]
    if len(episode_ids) > 1:
        status = "generalized_across_multiple_episodes"
    elif types == ["direct_experience"]:
        status = "single_direct_experience"
    elif "inference" in types:
        status = "inferred_or_mixed_evidence"
    else:
        status = "single_non_direct_source"
    return {"status": status, "source_types": types, "source_ids": source_ids,
            "supporting_episode_ids": episode_ids, "feature_key": feature_key}


def brier_score(probabilities: Sequence[float], outcomes: Sequence[float]) -> float | None:
    """Compute Brier score only when externally supplied binary outcomes are available."""
    if len(probabilities) != len(outcomes) or not probabilities:
        return None
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if (not np.all(np.isfinite(p)) or not np.all(np.isfinite(y))
            or np.any((p < 0.0) | (p > 1.0)) or np.any((y < 0.0) | (y > 1.0))):
        raise ValueError("Brier score inputs must be finite values in [0, 1]")
    return float(np.mean((p - y) ** 2))


def leave_one_out_gist_similarity(
    episodes: Sequence[EpisodeInput],
) -> dict[str, Any]:
    """Evaluate whether each episode resembles a prototype built without that same episode."""
    if len(episodes) < 2:
        return {"status": "insufficient_examples", "mean_similarity": None, "scores": {}}
    scores: dict[str, float] = {}
    for index, episode in enumerate(episodes):
        others = [item for j, item in enumerate(episodes) if j != index]
        group_vectors: dict[str, dict[str, float]] = {group: {} for group in FEATURE_GROUPS}
        keys = {group: sorted({key for other in others
                               for key in getattr(other, f"{group}_features")})
                for group in FEATURE_GROUPS}
        for group in FEATURE_GROUPS:
            for feature in keys[group]:
                observed = [(getattr(other, f"{group}_features")[feature],
                             other.overall_confidence * other.source_confidence)
                            for other in others if feature in getattr(other, f"{group}_features")]
                if observed:
                    weights = [weight for _, weight in observed]
                    group_vectors[group][feature] = float(np.average(
                        [value for value, _ in observed], weights=weights))
        scores[episode.episode_id] = weighted_feature_similarity(
            episode.feature_groups(), group_vectors)
    return {"status": "leave_one_episode_out", "mean_similarity": float(np.mean(list(scores.values()))),
            "scores": scores}
