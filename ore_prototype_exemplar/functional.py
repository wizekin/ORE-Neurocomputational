"""Mathematical operations for prototype/exemplar comparison and hybrid categorization.

Equations are computational hypotheses. Their implementation does not imply that any single
brain region explicitly computes vector means, exponentials, covariance matrices, or probabilities.
"""
from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import CategoryState, EpisodeInput, ExemplarTrace, FeatureValue, PrototypeExemplarConfig


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Bound a finite scalar to a closed interval."""
    number = float(value)
    if not math.isfinite(number) or low > high:
        raise ValueError("value and bounds must be finite and ordered")
    return min(high, max(low, number))


def sigmoid(value: float) -> float:
    """Apply a numerically stable logistic transform."""
    clipped = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-clipped))


def stable_softmax(values: Sequence[float], temperature: float = 1.0) -> tuple[float, ...]:
    """Convert competitive activations into a stable probability distribution."""
    if not values or temperature <= 0.0:
        raise ValueError("values must be non-empty and temperature positive")
    scaled = [float(value) / temperature for value in values]
    maximum = max(scaled)
    exps = [math.exp(max(-60.0, min(0.0, item - maximum))) for item in scaled]
    total = sum(exps)
    return tuple(item / total for item in exps)


def canonical_features(
    episode: EpisodeInput, aliases: Mapping[str, str]
) -> dict[str, FeatureValue]:
    """Map observed feature aliases to their canonical names without fabricating missing values."""
    result: dict[str, FeatureValue] = {}
    for name, value in episode.features.items():
        result[aliases.get(name, name)] = value
    return result


def canonical_relations(
    episode: EpisodeInput, aliases: Mapping[str, str]
) -> tuple[tuple[str, str, str], ...]:
    """Normalize feature aliases inside typed relations while preserving relation identity."""
    return tuple(dict.fromkeys(
        (aliases.get(subject, subject), predicate, aliases.get(object_, object_))
        for subject, predicate, object_ in episode.relations
    ))


def _is_numeric(value: FeatureValue) -> bool:
    """Return whether a feature should use continuous rather than categorical comparison."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _category_scale(state: CategoryState, feature: str) -> float:
    """Estimate a nonzero feature scale from prototype variance with a robust floor."""
    weight = state.prototype.numeric_weight.get(feature, 0.0)
    if weight <= 0.0:
        return 1.0
    variance = state.prototype.numeric_m2.get(feature, 0.0) / max(weight, 1e-12)
    return max(0.15, math.sqrt(max(0.0, variance)))


def feature_distance(
    first: FeatureValue, second: FeatureValue, scale: float = 1.0
) -> float | None:
    """Compute a normalized mismatch for continuous or categorical observed feature pairs."""
    if first is None or second is None:
        return None
    if _is_numeric(first) and _is_numeric(second):
        return clamp(abs(float(first) - float(second)) / max(float(scale), 1e-9), 0.0, 4.0)
    return 0.0 if first == second else 1.0


def weighted_distance(
    features: Mapping[str, FeatureValue],
    reference: Mapping[str, FeatureValue],
    weights: Mapping[str, float],
    scales: Mapping[str, float] | None = None,
    exponent: float = 2.0,
    relations: Sequence[tuple[str, str, str]] = (),
    reference_relations: Sequence[tuple[str, str, str]] = (),
) -> float:
    """Compute feature-weighted Minkowski distance plus a small typed-relation mismatch term."""
    if exponent not in (1.0, 2.0):
        raise ValueError("exponent must be 1 or 2")
    terms: list[tuple[float, float]] = []
    for feature in sorted(set(features) & set(reference)):
        mismatch = feature_distance(
            features[feature], reference[feature], (scales or {}).get(feature, 1.0)
        )
        weight = max(0.0, float(weights.get(feature, 1.0)))
        if mismatch is not None and weight > 0.0:
            terms.append((weight, mismatch))
    if not terms:
        feature_part = 1.0
    else:
        denominator = sum(weight for weight, _ in terms)
        powered = sum(weight * mismatch ** exponent for weight, mismatch in terms) / denominator
        feature_part = powered ** (1.0 / exponent)
    rel_a, rel_b = set(relations), set(reference_relations)
    if rel_a or rel_b:
        relation_mismatch = 1.0 - len(rel_a & rel_b) / max(1, len(rel_a | rel_b))
        return 0.9 * feature_part + 0.1 * relation_mismatch
    return feature_part


def prototype_features(state: CategoryState) -> dict[str, FeatureValue]:
    """Materialize the current central tendency and categorical mode for similarity comparison."""
    result: dict[str, FeatureValue] = dict(state.prototype.numeric_mean)
    for feature, counts in state.prototype.categorical_counts.items():
        if counts:
            token = max(sorted(counts), key=lambda key: counts[key])
            kind, separator, raw = token.partition(":")
            if kind == "bool" and separator:
                result[feature] = raw == "True"
            elif kind == "str" and separator:
                result[feature] = raw
            else:
                result[feature] = token
    return result


def prototype_reliability(state: CategoryState) -> float:
    """Estimate summary reliability from effective sample size and within-category stability."""
    prototype = state.prototype
    if prototype.update_count == 0:
        return 0.0
    reliability_rows = []
    for feature, weight in prototype.numeric_weight.items():
        if weight <= 0.0:
            continue
        variance = prototype.numeric_m2.get(feature, 0.0) / weight
        ess = weight * weight / max(prototype.numeric_weight_sq.get(feature, 1e-12), 1e-12)
        reliability_rows.append((ess / (ess + 3.0)) / (1.0 + variance))
    for feature, counts in prototype.categorical_counts.items():
        total = sum(counts.values())
        if total > 0.0:
            probabilities = [count / total for count in counts.values()]
            concentration = max(probabilities)
            reliability_rows.append(concentration * total / (total + 3.0))
    if not reliability_rows:
        return 0.0
    evidence = min(1.0, prototype.update_count / 4.0)
    return clamp((sum(reliability_rows) / len(reliability_rows)) * evidence)


def memory_accessibility(
    exemplar: ExemplarTrace, now: float, decay_rate: float
) -> float:
    """Estimate exemplar accessibility with exponential decay and rehearsal/retrieval increments."""
    age = max(0.0, float(now) - exemplar.created_at)
    strength = exemplar.base_strength * math.exp(-decay_rate * age)
    for event_time, gain in exemplar.rehearsal_events:
        elapsed = max(0.0, float(now) - event_time)
        strength += gain * math.exp(-decay_rate * elapsed)
    retrieval_boost = 1.0 + 0.04 * min(exemplar.retrieval_count, 10)
    return clamp(strength * retrieval_boost, 0.0, 1.5)


def prototype_similarity(
    episode: EpisodeInput,
    state: CategoryState,
    weights: Mapping[str, float],
    config: PrototypeExemplarConfig,
) -> dict[str, Any]:
    """Compare a new item with a category prototype using weighted distance and an exponential kernel."""
    features = canonical_features(episode, config.feature_aliases)
    reference = prototype_features(state)
    scales = {name: _category_scale(state, name) for name in reference}
    distance = weighted_distance(
        features, reference, weights, scales, config.distance_exponent,
        canonical_relations(episode, config.feature_aliases),
        tuple(tuple(row.split(" :: ")) for row in state.prototype.relation_counts
              if len(row.split(" :: ")) == 3),
    )
    similarity = math.exp(-config.similarity_sensitivity * distance)
    return {"similarity": clamp(similarity), "distance": distance,
            "reliability": prototype_reliability(state),
            "compared_features": sorted(set(features) & set(reference))}


def retrieve_exemplars(
    episode: EpisodeInput,
    state: CategoryState,
    weights: Mapping[str, float],
    config: PrototypeExemplarConfig,
    now: float,
) -> list[dict[str, Any]]:
    """Rank a category's distinct exemplar traces by cue similarity and memory accessibility."""
    features = canonical_features(episode, config.feature_aliases)
    relations = canonical_relations(episode, config.feature_aliases)
    ranked: list[dict[str, Any]] = []
    for trace in state.exemplars.values():
        similarity_distance = weighted_distance(
            features, trace.features, weights, None, config.distance_exponent,
            relations, trace.relations,
        )
        similarity = math.exp(-config.similarity_sensitivity * similarity_distance)
        accessibility = memory_accessibility(trace, now, config.memory_decay_rate)
        relevance = similarity * accessibility * max(0.05, trace.reliability)
        ranked.append({"episode_id": trace.episode_id, "distance": similarity_distance,
                       "similarity": similarity, "accessibility": accessibility,
                       "relevance": relevance, "is_exception": trace.is_exception})
    ranked.sort(key=lambda row: (-row["relevance"], row["episode_id"]))
    return ranked[:config.retrieval_limit]


def exemplar_similarity(retrieved: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate exemplar support while exposing the strongest individual match and its identity."""
    if not retrieved:
        return {"similarity": 0.0, "best_similarity": 0.0,
                "retrieved_ids": [], "support_mass": 0.0, "reliability": 0.0}
    total_access = sum(float(row["accessibility"]) for row in retrieved)
    weighted = sum(float(row["similarity"]) * float(row["relevance"]) for row in retrieved)
    relevance_mass = sum(float(row["relevance"]) for row in retrieved)
    aggregate = weighted / max(relevance_mass, 1e-12)
    best = max(retrieved, key=lambda row: (float(row["relevance"]), row["episode_id"]))
    combined = 0.78 * aggregate + 0.22 * float(best["similarity"])
    reliability = (min(1.0, len(retrieved) / 4.0)
                   * min(1.0, relevance_mass / max(total_access, 1e-12)))
    return {"similarity": clamp(combined), "best_similarity": float(best["similarity"]),
            "best_episode_id": best["episode_id"],
            "retrieved_ids": [str(row["episode_id"]) for row in retrieved],
            "support_mass": relevance_mass, "reliability": clamp(reliability)}


def blend_weight(
    prototype_confidence: float,
    exemplar_confidence: float,
    typicality: float,
    exception_score: float,
    task_context: str,
) -> float:
    """Choose an item- and task-sensitive prototype contribution through a logistic controller."""
    task_bias = 0.30 if "typical" in task_context.lower() or "rapid" in task_context.lower() else 0.0
    logit = (-0.9 + 2.3 * prototype_confidence - 1.4 * exemplar_confidence
             + 1.25 * typicality - 1.45 * exception_score + task_bias)
    return clamp(sigmoid(logit))


def category_posterior(
    hybrid_scores: Mapping[str, float],
    prior_counts: Mapping[str, float],
    temperature: float,
    beta_prior: float,
) -> dict[str, float]:
    """Combine category similarity evidence and category-frequency priors with a Luce softmax."""
    if not hybrid_scores:
        return {}
    categories = sorted(hybrid_scores)
    total_prior = sum(max(beta_prior, float(prior_counts.get(name, beta_prior))) for name in categories)
    logits = [
        math.log(max(float(hybrid_scores[name]), 1e-12))
        + math.log(max(beta_prior, float(prior_counts.get(name, beta_prior))) / total_prior)
        for name in categories
    ]
    probabilities = stable_softmax(logits, temperature)
    return {name: probability for name, probability in zip(categories, probabilities)}


def mahalanobis_outlier_score(
    episode: EpisodeInput, state: CategoryState, aliases: Mapping[str, str]
) -> float:
    """Estimate covariance-aware numeric atypicality with regularization for small samples."""
    features = canonical_features(episode, aliases)
    names = sorted(name for name, value in features.items() if _is_numeric(value)
                   and name in state.prototype.numeric_mean)
    if not names:
        return 0.0
    vector = np.asarray([float(features[name]) for name in names], dtype=float)
    mean = np.asarray([state.prototype.numeric_mean[name] for name in names], dtype=float)
    samples = []
    for row in state.numeric_observations:
        if all(name in row for name in names):
            samples.append([row[name] for name in names])
    if len(samples) >= max(3, len(names) + 1):
        covariance = np.cov(np.asarray(samples, dtype=float), rowvar=False)
        covariance = np.atleast_2d(covariance)
    else:
        variances = [max(0.05, state.prototype.numeric_m2.get(name, 0.0)
                         / max(state.prototype.numeric_weight.get(name, 1.0), 1e-12))
                     for name in names]
        covariance = np.diag(variances)
    covariance = covariance + np.eye(len(names)) * 1e-3
    delta = vector - mean
    try:
        score = float(np.sqrt(max(0.0, delta.T @ np.linalg.solve(covariance, delta))))
    except np.linalg.LinAlgError:
        score = float(np.linalg.norm(delta))
    return score if math.isfinite(score) else 0.0


def estimate_feature_diagnosticity(
    states: Mapping[str, CategoryState], target_category: str
) -> dict[str, float | None]:
    """Estimate category-specific numeric mean separation or categorical log-odds diagnosticity."""
    target = states.get(target_category)
    if target is None:
        return {}
    all_features = set(target.prototype.numeric_mean) | set(target.prototype.categorical_counts)
    result: dict[str, float | None] = {}
    competitors = [state for name, state in states.items() if name != target_category]
    for feature in sorted(all_features):
        if feature in target.prototype.numeric_mean:
            target_mean = target.prototype.numeric_mean[feature]
            target_var = target.prototype.numeric_m2.get(feature, 0.0) / max(
                target.prototype.numeric_weight.get(feature, 1.0), 1e-12)
            comp_means = [state.prototype.numeric_mean[feature]
                          for state in competitors if feature in state.prototype.numeric_mean]
            if comp_means:
                comp_mean = sum(comp_means) / len(comp_means)
                scale = math.sqrt(max(0.05, target_var))
                result[feature] = clamp(abs(target_mean - comp_mean) / (abs(target_mean - comp_mean) + scale))
            else:
                result[feature] = None
        else:
            target_counts = target.prototype.categorical_counts.get(feature, {})
            total_target = sum(target_counts.values())
            mode_target = max(target_counts, key=target_counts.get) if target_counts else None
            comp_total = 0.0
            comp_match = 0.0
            for state in competitors:
                counts = state.prototype.categorical_counts.get(feature, {})
                comp_total += sum(counts.values())
                comp_match += counts.get(str(mode_target), 0.0)
            if total_target > 0 and comp_total > 0:
                p_target = (target_counts.get(str(mode_target), 0.0) + 1.0) / (total_target + 2.0)
                p_other = (comp_match + 1.0) / (comp_total + 2.0)
                result[feature] = clamp(abs(math.log(p_target / (1.0 - p_target))
                                             - math.log(p_other / (1.0 - p_other))) / 8.0)
            else:
                result[feature] = None
    return result


def update_feature_attention(
    current: Mapping[str, float],
    episode: EpisodeInput,
    expected_category: str,
    states: Mapping[str, CategoryState],
    learning_rate: float,
    aliases: Mapping[str, str],
) -> dict[str, float]:
    """Adjust attended feature gains toward mismatches revealed by explicitly labelled feedback."""
    target = states.get(expected_category)
    if target is None:
        return dict(current)
    observed = canonical_features(episode, aliases)
    prototype = prototype_features(target)
    updated = dict(current)
    for feature, value in observed.items():
        if value is None:
            continue
        mismatch = feature_distance(value, prototype.get(feature), _category_scale(target, feature))
        if mismatch is not None:
            old = float(updated.get(feature, 0.5))
            updated[feature] = clamp(old + learning_rate * (float(mismatch) - 0.25))
    return updated


def model_validation_metrics(
    labels: Sequence[str], predictions: Sequence[Mapping[str, Any]], model_key: str
) -> dict[str, float | int | None]:
    """Calculate held-out accuracy, multiclass negative log likelihood, and Brier score."""
    if not labels or len(labels) != len(predictions):
        return {"n": 0, "accuracy": None, "negative_log_likelihood": None, "brier_score": None}
    correct = 0
    nll = 0.0
    brier = 0.0
    category_set = sorted({label for row in predictions
                           for label in row.get(f"{model_key}_probabilities", {})} | set(labels))
    for label, row in zip(labels, predictions):
        probabilities = row.get(f"{model_key}_probabilities", {})
        if probabilities:
            prediction = max(probabilities, key=probabilities.get)
            correct += int(prediction == label)
            nll -= math.log(max(1e-12, float(probabilities.get(label, 0.0))))
            brier += sum((float(probabilities.get(category, 0.0)) - float(category == label)) ** 2
                         for category in category_set)
    return {"n": len(labels), "accuracy": correct / len(labels),
            "negative_log_likelihood": nll / len(labels), "brier_score": brier / len(labels)}
