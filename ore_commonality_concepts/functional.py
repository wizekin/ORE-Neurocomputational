"""Mathematical operations for feature commonality, contrast, reliability, and prediction.

Every equation here is an explicit computational model or diagnostic. It does not imply that
the brain literally calculates a named statistic; neural implementations are distributed and
depend on population activity, plasticity, attention, and task conditions.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from .contracts import ConceptCandidate, EpisodeInput


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Bound a finite scalar to a closed interval."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("value must be finite")
    if low > high:
        raise ValueError("low must not exceed high")
    return min(high, max(low, number))


def sigmoid(value: float) -> float:
    """Compute a numerically stable logistic transform."""
    clipped = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-clipped))


def logit(probability: float, epsilon: float = 1e-9) -> float:
    """Convert a probability to log odds with a finite boundary safeguard."""
    p = min(1.0 - epsilon, max(epsilon, float(probability)))
    return math.log(p / (1.0 - p))


def stable_softmax(values: Sequence[float], temperature: float = 1.0) -> tuple[float, ...]:
    """Normalize competitive activations into a stable probability distribution."""
    if not values or temperature <= 0.0:
        raise ValueError("values must be non-empty and temperature positive")
    scaled = [float(value) / temperature for value in values]
    maximum = max(scaled)
    exponentials = [math.exp(max(-60.0, min(0.0, value - maximum))) for value in scaled]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def canonical_features(
    episode: EpisodeInput, aliases: Mapping[str, str]
) -> dict[str, float | None]:
    """Map aliases to canonical feature names and combine duplicate observed values."""
    grouped: dict[str, list[float]] = {}
    unknown: set[str] = set()
    for feature, value in episode.features.items():
        canonical = aliases.get(feature, feature)
        if value is None:
            unknown.add(canonical)
        else:
            grouped.setdefault(canonical, []).append(float(value))
    output: dict[str, float | None] = {}
    for feature in set(grouped) | unknown:
        observations = grouped.get(feature, [])
        output[feature] = sum(observations) / len(observations) if observations else None
    return output


def canonical_relations(
    episode: EpisodeInput, aliases: Mapping[str, str]
) -> tuple[tuple[str, str, str], ...]:
    """Normalize the node names of typed relations using the same feature-alias map."""
    return tuple(dict.fromkeys(
        (aliases.get(subject, subject), predicate, aliases.get(object_, object_))
        for subject, predicate, object_ in episode.relations
    ))


def estimate_feature_statistics(
    episodes: Sequence[EpisodeInput],
    aliases: Mapping[str, str],
    total_reference_weight: float | None = None,
) -> dict[str, dict[str, float | int]]:
    """Estimate weighted prevalence, variance, observation count, coverage, and effective sample size.

    Missing values are excluded from each feature's denominator. Explicit zero values remain
    observed evidence. Reliability is used as the observation weight; sample variance uses the
    reliability-weighted unbiased denominator when it is defined.
    """
    observations: dict[str, list[tuple[float, float]]] = {}
    total_weight = 0.0
    for episode in episodes:
        weight = episode.reliability
        total_weight += weight
        for feature, value in canonical_features(episode, aliases).items():
            if value is not None:
                observations.setdefault(feature, []).append((value, weight))
    reference = total_reference_weight if total_reference_weight is not None else total_weight
    result: dict[str, dict[str, float | int]] = {}
    for feature, rows in observations.items():
        sum_weight = sum(weight for _, weight in rows)
        sum_weight_sq = sum(weight * weight for _, weight in rows)
        if sum_weight <= 0.0:
            continue
        mean = sum(value * weight for value, weight in rows) / sum_weight
        denominator = sum_weight - sum_weight_sq / sum_weight
        variance = (
            sum(weight * (value - mean) ** 2 for value, weight in rows) / denominator
            if denominator > 1e-12 else 0.0
        )
        effective_n = sum_weight * sum_weight / max(sum_weight_sq, 1e-12)
        result[feature] = {
            "prevalence": clamp(mean),
            "variance": max(0.0, variance),
            "observation_count": len(rows),
            "observed_weight": sum_weight,
            "effective_sample_size": effective_n,
            "coverage": clamp(sum_weight / max(reference, 1e-12)),
        }
    return result


def competitive_attention_weights(
    feature_names: Sequence[str],
    attention: Mapping[str, float],
    goals: Mapping[str, float],
    prior: Mapping[str, float],
    temperature: float = 1.0,
) -> dict[str, float]:
    """Combine top-down goal and prior salience with attention using competitive normalization.

    The normalized softmax is rescaled by its maximum so each feature gain remains in (0, 1].
    """
    names = sorted(set(feature_names))
    if not names:
        return {}
    raw = [
        0.50 * float(attention.get(name, 0.5))
        + 0.30 * float(goals.get(name, 0.5))
        + 0.20 * float(prior.get(name, 0.5))
        for name in names
    ]
    distribution = stable_softmax(raw, temperature)
    maximum = max(distribution)
    return {name: clamp(value / maximum) for name, value in zip(names, distribution)}


def estimate_contrastive_diagnosticity(
    target_episodes: Sequence[EpisodeInput],
    comparison_episodes: Sequence[EpisodeInput],
    aliases: Mapping[str, str],
    beta_prior: float = 1.0,
) -> dict[str, dict[str, float | None]]:
    """Estimate feature log-odds differences between target and comparison examples.

    A Beta-like smoothing prior avoids infinite log odds. If no contrastive examples observe a
    feature, diagnosticity is None rather than being fabricated from missing evidence.
    """
    target_stats = estimate_feature_statistics(target_episodes, aliases)
    comparison_stats = estimate_feature_statistics(comparison_episodes, aliases)
    all_features = sorted(set(target_stats) | set(comparison_stats))
    result: dict[str, dict[str, float | None]] = {}
    for feature in all_features:
        target_row = target_stats.get(feature)
        compare_row = comparison_stats.get(feature)
        p_target: float | None = None
        p_compare: float | None = None
        difference: float | None = None
        if target_row is not None:
            total = float(target_row["observed_weight"])
            successes = float(target_row["prevalence"]) * total
            p_target = (successes + beta_prior) / (total + 2.0 * beta_prior)
        if compare_row is not None:
            total = float(compare_row["observed_weight"])
            successes = float(compare_row["prevalence"]) * total
            p_compare = (successes + beta_prior) / (total + 2.0 * beta_prior)
        if p_target is not None and p_compare is not None:
            difference = logit(p_target) - logit(p_compare)
        result[feature] = {
            "target_prevalence_smoothed": p_target,
            "comparison_prevalence_smoothed": p_compare,
            "log_odds_difference": difference,
        }
    return result


def estimate_pairwise_associations(
    episodes: Sequence[EpisodeInput],
    aliases: Mapping[str, str],
    beta_prior: float = 1.0,
) -> dict[str, float]:
    """Estimate smoothed pointwise mutual information for co-observed feature pairs.

    A feature pair contributes only in episodes where both features were observed. This prevents
    a missing observation from being interpreted as co-absence.
    """
    feature_names = sorted({
        feature
        for episode in episodes
        for feature, value in canonical_features(episode, aliases).items()
        if value is not None
    })
    result: dict[str, float] = {}
    for i, first in enumerate(feature_names):
        for second in feature_names[i + 1:]:
            total = first_count = second_count = joint_count = 0.0
            for episode in episodes:
                values = canonical_features(episode, aliases)
                a, b = values.get(first), values.get(second)
                if a is None or b is None:
                    continue
                weight = episode.reliability
                total += weight
                present_a, present_b = a >= 0.5, b >= 0.5
                first_count += weight * float(present_a)
                second_count += weight * float(present_b)
                joint_count += weight * float(present_a and present_b)
            if total <= 0.0:
                continue
            denominator = total + 4.0 * beta_prior
            p_a = (first_count + 2.0 * beta_prior) / denominator
            p_b = (second_count + 2.0 * beta_prior) / denominator
            p_ab = (joint_count + beta_prior) / denominator
            pmi = math.log(max(p_ab, 1e-12) / max(p_a * p_b, 1e-12))
            result[f"{first}::{second}"] = pmi
    return result


def estimate_relation_strengths(
    episodes: Sequence[EpisodeInput],
    aliases: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    """Accumulate typed relation support while retaining the example IDs that support each edge."""
    support: dict[tuple[str, str, str], float] = {}
    episode_ids: dict[tuple[str, str, str], list[str]] = {}
    total_weight = sum(episode.reliability for episode in episodes)
    for episode in episodes:
        for relation in set(canonical_relations(episode, aliases)):
            support[relation] = support.get(relation, 0.0) + episode.reliability
            episode_ids.setdefault(relation, []).append(episode.episode_id)
    result = {}
    for relation, value in support.items():
        key = " :: ".join(relation)
        result[key] = {
            "support": clamp(value / max(total_weight, 1e-12)),
            "supporting_episode_ids": sorted(set(episode_ids[relation])),
            "relation": list(relation),
        }
    return result


def compute_feature_reliability(
    statistics: Mapping[str, Mapping[str, float | int]],
    variance_floor: float,
) -> dict[str, float]:
    """Combine observation coverage, effective sample size, and consistency into reliability."""
    result = {}
    for feature, row in statistics.items():
        coverage = clamp(float(row["coverage"]))
        effective_n = float(row["effective_sample_size"])
        variance = float(row["variance"])
        sample_support = effective_n / (effective_n + 1.0)
        consistency = 1.0 / (1.0 + variance / variance_floor)
        result[feature] = clamp(coverage * sample_support * consistency)
    return result


def compute_feature_weights(
    target_statistics: Mapping[str, Mapping[str, float | int]],
    diagnosticity: Mapping[str, Mapping[str, float | None]],
    attention_weights: Mapping[str, float],
    feature_reliability: Mapping[str, float],
) -> dict[str, float]:
    """Weight shared evidence by prevalence, task salience, reliability, and contrastive value.

    When contrastive evidence is unavailable, a neutral diagnostic multiplier is used and the
    missing diagnosticity remains explicitly marked in the candidate. It is not treated as a
    measured positive distinction.
    """
    weights: dict[str, float] = {}
    for feature, row in target_statistics.items():
        prevalence = clamp(float(row["prevalence"]))
        salience = clamp(float(attention_weights.get(feature, 0.5)))
        reliability = clamp(float(feature_reliability.get(feature, 0.0)))
        log_odds = diagnosticity.get(feature, {}).get("log_odds_difference")
        diagnostic_multiplier = 0.5 if log_odds is None else sigmoid(float(log_odds))
        commonality = clamp((prevalence - 0.10) / 0.90)
        weights[feature] = clamp(
            commonality * salience * reliability * diagnostic_multiplier
        )
    return weights


def build_concept_candidate(
    concept_id: str,
    target_episodes: Sequence[EpisodeInput],
    comparison_episodes: Sequence[EpisodeInput],
    aliases: Mapping[str, str],
    attention: Mapping[str, float],
    goals: Mapping[str, float],
    prior: Mapping[str, float],
    beta_prior: float,
    variance_floor: float,
    minimum_shared_weight: float,
    previous: ConceptCandidate | None = None,
) -> ConceptCandidate:
    """Construct an integrated, contrastive, uncertainty-aware shared-feature concept candidate."""
    target_stats = estimate_feature_statistics(target_episodes, aliases)
    comparison_stats = estimate_feature_statistics(comparison_episodes, aliases)
    feature_names = sorted(set(target_stats) | set(comparison_stats))
    attention_weights = competitive_attention_weights(
        feature_names, attention, goals, prior
    )
    diagnostic = estimate_contrastive_diagnosticity(
        target_episodes, comparison_episodes, aliases, beta_prior
    )
    reliability = compute_feature_reliability(target_stats, variance_floor)
    weights = compute_feature_weights(target_stats, diagnostic, attention_weights, reliability)
    associations = estimate_pairwise_associations(target_episodes, aliases, beta_prior)
    relation_rows = estimate_relation_strengths(target_episodes, aliases)
    shared_features = sorted(
        feature for feature, weight in weights.items()
        if weight >= minimum_shared_weight
    )
    variance = {feature: float(row["variance"]) for feature, row in target_stats.items()}
    prevalence = {feature: float(row["prevalence"]) for feature, row in target_stats.items()}
    counts = {feature: int(row["observation_count"]) for feature, row in target_stats.items()}
    effective = {feature: float(row["effective_sample_size"]) for feature, row in target_stats.items()}
    comparison_prevalence: dict[str, float | None] = {}
    diagnosticity_values: dict[str, float | None] = {}
    for feature in feature_names:
        comparison_prevalence[feature] = (
            float(comparison_stats[feature]["prevalence"])
            if feature in comparison_stats else None
        )
        diagnosticity_values[feature] = diagnostic.get(feature, {}).get("log_odds_difference")
    relation_strength = {
        feature: float(row["support"]) for feature, row in relation_rows.items()
    }
    relation_edges = [
        {
            "key": key,
            "relation": row["relation"],
            "strength": row["support"],
            "supporting_episode_ids": row["supporting_episode_ids"],
        }
        for key, row in sorted(relation_rows.items())
    ]
    uncertainty = {
        feature: clamp(
            1.0 - reliability.get(feature, 0.0)
            + min(0.35, variance.get(feature, 0.0))
        )
        for feature in target_stats
    }
    exceptions = sorted(
        feature for feature, value in variance.items()
        if value >= variance_floor * 3.0
    )
    candidate = ConceptCandidate(
        concept_id=concept_id,
        feature_prevalence=prevalence,
        feature_variance=variance,
        feature_observation_count=counts,
        feature_effective_sample_size=effective,
        comparison_prevalence=comparison_prevalence,
        diagnosticity=diagnosticity_values,
        attention_weights=attention_weights,
        feature_reliability=reliability,
        feature_weights=weights,
        cooccurrence_pmi=associations,
        relation_strength=relation_strength,
        shared_features=shared_features,
        relation_edges=relation_edges,
        uncertainty=uncertainty,
        exceptions=exceptions,
        supporting_episode_ids=sorted(episode.episode_id for episode in target_episodes),
        comparison_episode_ids=sorted(episode.episode_id for episode in comparison_episodes),
        update_count=(previous.update_count + 1 if previous else 1),
        provenance=[
            *[
                {"source": "target_episode", "episode_id": episode.episode_id,
                 "reliability": episode.reliability}
                for episode in target_episodes
            ],
            *[
                {"source": "comparison_episode", "episode_id": episode.episode_id,
                 "reliability": episode.reliability}
                for episode in comparison_episodes
            ],
        ],
    )
    if previous is not None:
        known_ids = {
            str(row.get("episode_id"))
            for row in previous.provenance
            if row.get("episode_id") is not None
        }
        new_provenance = [
            row for row in candidate.provenance
            if str(row.get("episode_id")) not in known_ids
        ]
        candidate.provenance = previous.provenance + new_provenance
        candidate.update_count = previous.update_count + int(bool(new_provenance))
    return candidate


def retrieve_related_episodes(
    current: EpisodeInput,
    memory: Sequence[EpisodeInput],
    aliases: Mapping[str, str],
    limit: int = 20,
) -> list[dict[str, object]]:
    """Retrieve related episodes by observed feature overlap, similarity, and typed relations."""
    current_features = {
        key: value for key, value in canonical_features(current, aliases).items()
        if value is not None
    }
    current_relations = set(canonical_relations(current, aliases))
    scored = []
    for past in memory:
        if past.episode_id == current.episode_id:
            continue
        past_features = {
            key: value for key, value in canonical_features(past, aliases).items()
            if value is not None
        }
        shared = set(current_features) & set(past_features)
        union = set(current_features) | set(past_features)
        if not union:
            continue
        similarity = (
            sum(1.0 - abs(current_features[key] - past_features[key]) for key in shared)
            / len(shared) if shared else 0.0
        )
        coverage = len(shared) / len(union)
        past_relations = set(canonical_relations(past, aliases))
        relation_overlap = (
            len(current_relations & past_relations) / len(current_relations | past_relations)
            if current_relations or past_relations else 0.0
        )
        score = (
            0.70 * similarity + 0.20 * coverage + 0.10 * relation_overlap
        ) * past.reliability
        if score > 0.0:
            scored.append((score, past.episode_id, sorted(shared), similarity, coverage, relation_overlap))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [
        {
            "episode_id": episode_id,
            "score": score,
            "shared_features": shared,
            "feature_similarity": similarity,
            "feature_coverage": coverage,
            "relation_overlap": relation_overlap,
        }
        for score, episode_id, shared, similarity, coverage, relation_overlap in scored[:limit]
    ]


def compress_shared_structure(
    candidate: ConceptCandidate, minimum_weight: float
) -> dict[str, object]:
    """Select common, informative feature and relation regularities without deleting exceptions."""
    shared = [
        feature for feature in candidate.shared_features
        if candidate.feature_weights.get(feature, 0.0) >= minimum_weight
    ]
    retained_relations = [
        edge for edge in candidate.relation_edges
        if float(edge["strength"]) >= minimum_weight
    ]
    return {
        "shared_features": shared,
        "shared_relations": retained_relations,
        "exceptions": list(candidate.exceptions),
        "discarded_as_incidental_or_weak": sorted(
            set(candidate.feature_prevalence) - set(shared)
        ),
        "supporting_episode_count": len(candidate.supporting_episode_ids),
        "compression_ratio": (
            len(shared) / max(1, len(candidate.feature_prevalence))
        ),
        "episode_details_preserved": True,
    }


def predict_episode(
    episode: EpisodeInput,
    candidate: ConceptCandidate,
    aliases: Mapping[str, str],
    prediction_prior: float,
    threshold: float,
) -> dict[str, object]:
    """Estimate category membership from feature likelihoods and return an evidence-aware score.

    When no contrastive examples exist, the function returns a fit score and leaves posterior
    membership undefined; it does not pretend that positive examples alone establish a boundary.
    """
    values = canonical_features(episode, aliases)
    observed = {name: value for name, value in values.items() if value is not None}
    comparison_available = any(
        value is not None for value in candidate.comparison_prevalence.values()
    )
    feature_rows = []
    weighted_log_likelihood_ratio = 0.0
    fit_error = 0.0
    fit_weight = 0.0
    for feature, value in observed.items():
        if feature not in candidate.feature_prevalence:
            continue
        mean = clamp(candidate.feature_prevalence[feature])
        variance = max(1e-6, candidate.feature_variance.get(feature, 0.0))
        weight = clamp(
            candidate.attention_weights.get(feature, 0.5)
            * max(0.05, candidate.feature_reliability.get(feature, 0.0))
        )
        residual = float(value) - mean
        standardized_error = residual * residual / (variance + 0.025)
        fit_error += weight * standardized_error
        fit_weight += weight
        if comparison_available:
            background = candidate.comparison_prevalence.get(feature)
            if background is None:
                continue
            p_target = min(0.99, max(0.01, mean))
            p_background = min(0.99, max(0.01, float(background)))
            log_ratio = (
                float(value) * math.log(p_target / p_background)
                + (1.0 - float(value)) * math.log(
                    (1.0 - p_target) / (1.0 - p_background)
                )
            )
            weighted_log_likelihood_ratio += weight * log_ratio
        feature_rows.append({
            "feature": feature,
            "observed_value": value,
            "expected_prevalence": mean,
            "residual": residual,
            "standardized_squared_error": standardized_error,
            "weight": weight,
        })
    fit_score = math.exp(-0.5 * fit_error / max(fit_weight, 1e-12)) if fit_weight else 0.0
    if comparison_available:
        posterior_log_odds = logit(prediction_prior) + weighted_log_likelihood_ratio
        posterior = sigmoid(posterior_log_odds)
        predicted = posterior >= threshold
        status = "contrastive_posterior"
    else:
        posterior = None
        predicted = None
        status = "fit_only_no_contrastive_examples"
    return {
        "episode_id": episode.episode_id,
        "status": status,
        "observed_feature_count": len(observed),
        "used_feature_count": len(feature_rows),
        "feature_evidence": feature_rows,
        "fit_score": fit_score,
        "weighted_log_likelihood_ratio": weighted_log_likelihood_ratio,
        "posterior_category_probability": posterior,
        "predicted_category": predicted,
        "expected_category": episode.category_label,
    }


def estimate_prediction_mismatch(
    prediction: Mapping[str, object], mismatch_threshold: float
) -> dict[str, object]:
    """Summarize normalized feature residuals and labelled prediction outcomes."""
    feature_evidence = prediction.get("feature_evidence", [])
    rows = feature_evidence if isinstance(feature_evidence, list) else []
    weight_sum = sum(float(row["weight"]) for row in rows)
    standardized = (
        sum(float(row["weight"]) * float(row["standardized_squared_error"]) for row in rows)
        / max(weight_sum, 1e-12)
    )
    mismatch = clamp(1.0 - math.exp(-0.5 * standardized))
    posterior = prediction.get("posterior_category_probability")
    expected = prediction.get("expected_category")
    correctness = (
        (bool(posterior >= 0.5) == bool(expected))
        if posterior is not None and expected is not None else None
    )
    return {
        "episode_id": prediction.get("episode_id"),
        "normalized_prediction_error": standardized,
        "mismatch_score": mismatch,
        "mismatch_detected": mismatch >= mismatch_threshold,
        "labelled_prediction_correct": correctness,
        "feedback_available": expected is not None,
    }


def build_associative_index(candidate: ConceptCandidate) -> dict[str, object]:
    """Build multi-cue feature and relation indices to the current revisable concept."""
    feature_index = {
        feature: {candidate.concept_id: weight}
        for feature, weight in candidate.feature_weights.items()
        if weight > 0.0
    }
    relation_index = {
        str(edge["key"]): {
            candidate.concept_id: float(edge["strength"])
        }
        for edge in candidate.relation_edges
        if float(edge["strength"]) > 0.0
    }
    return {
        "feature_to_concept": feature_index,
        "relation_to_concept": relation_index,
        "concept_to_supporting_episodes": {
            candidate.concept_id: list(candidate.supporting_episode_ids)
        },
        "lookup_policy": "cue overlap and feature/relation association accumulation",
    }
