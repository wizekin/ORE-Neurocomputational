"""Statistical operations for schema-guided matching, prediction, and active information gathering."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ore_schema_formation.functional import context_signature

from .contracts import (
    ActionDecision, ActionOption, CandidateScore, CaseEvidence, InferenceConfig,
    QueryOption, QueryRecommendation, SchemaProfile, value_key,
)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Bound a finite value to a closed interval."""
    number = float(value)
    if not math.isfinite(number) or low > high:
        raise ValueError("value must be finite and low must not exceed high")
    return min(high, max(low, number))


def entropy(probabilities: Sequence[float], floor: float = 1e-12) -> float:
    """Compute Shannon entropy in nats after safely normalizing non-negative values."""
    values = [max(0.0, float(value)) for value in probabilities]
    total = sum(values)
    if total <= 0.0:
        return 0.0
    return -sum((value / total) * math.log(max(value / total, floor))
                for value in values if value > 0.0)


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Convert finite candidate scores to a stable probability distribution."""
    if not scores:
        return []
    scale = float(temperature)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("temperature must be finite and positive")
    values = [float(score) for score in scores]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("scores must be finite")
    maximum = max(values)
    weights = [math.exp(max(-80.0, min(80.0, (value - maximum) / scale)))
               for value in values]
    total = sum(weights)
    return [value / total for value in weights]


def _log_probability(value: float, floor: float) -> float:
    """Return a stable natural logarithm of a probability-like value."""
    return math.log(max(floor, min(1.0, float(value))))


def _distribution_for_feature(profile: SchemaProfile, feature: str) -> Mapping[str, float]:
    """Resolve feature likelihoods from explicit profile data or Point 09 role type slots."""
    name = feature.casefold()
    explicit = profile.feature_probabilities.get(name)
    if explicit:
        return explicit
    if name.startswith("role:"):
        slot_name = name.split(":", 1)[1]
        slot = profile.frame.slots.get(slot_name)
        if slot:
            return {value_key(key): float(value)
                    for key, value in slot.allowed_type_probabilities.items()}
    return {}


def _feature_background(
    feature: str, profiles: Sequence[SchemaProfile], floor: float
) -> dict[str, float]:
    """Estimate a support-weighted background feature distribution across known schemas."""
    counts: dict[str, float] = {}
    total = 0.0
    for profile in profiles:
        prior = profile.prior_probability
        weight = float(prior) if prior is not None else max(1, len(profile.frame.supporting_episode_ids))
        distribution = _distribution_for_feature(profile, feature)
        for value, probability in distribution.items():
            counts[value] = counts.get(value, 0.0) + weight * probability
            total += weight * probability
    if total <= 0.0:
        return {}
    return {key: max(floor, value / total) for key, value in counts.items()}


def feature_log_likelihood_ratio(
    case: CaseEvidence,
    profile: SchemaProfile,
    all_profiles: Sequence[SchemaProfile],
    config: InferenceConfig,
) -> tuple[float, tuple[str, ...], tuple[str, ...], float]:
    """Score reliable present and contradicted features against a schema and library background."""
    score = 0.0
    weight_total = 0.0
    matched: list[str] = []
    conflicting: list[str] = []
    for item in case.usable_evidence:
        distribution = _distribution_for_feature(profile, item.feature)
        background = _feature_background(item.feature, all_profiles, config.probability_floor)
        if not distribution or not background:
            continue
        observed_value = value_key(item.value)
        p_schema = distribution.get(observed_value, config.probability_floor)
        p_background = background.get(observed_value, config.probability_floor)
        diagnosticity = profile.feature_diagnosticity.get(item.feature, 1.0)
        weight = item.reliability * diagnosticity
        if item.status == "contradicted":
            weight *= config.contradiction_penalty
        contribution = weight * math.log(
            max(config.probability_floor, p_schema) /
            max(config.probability_floor, p_background)
        )
        score += contribution
        weight_total += weight
        if contribution >= 0.0:
            matched.append(item.evidence_id or item.feature)
        else:
            conflicting.append(item.evidence_id or item.feature)
    normalized = score / max(1.0, weight_total)
    return normalized, tuple(sorted(matched)), tuple(sorted(conflicting)), weight_total


def relation_compatibility(
    case: CaseEvidence, profile: SchemaProfile, floor: float
) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    """Score directed relations using Point 09 edge confidence and keep unknown edges neutral."""
    frame_edges = {
        (edge.subject_slot.casefold(), edge.predicate.casefold(), edge.object_slot.casefold(),
         edge.evidential_status): edge
        for edge in profile.frame.relations
    }
    score = 0.0
    matched: list[str] = []
    conflicting: list[str] = []
    for item in case.relations:
        if item.status not in {"observed", "contradicted"}:
            continue
        candidates = [
            edge for key, edge in frame_edges.items()
            if key[0] == item.subject and key[1] == item.predicate and key[2] == item.object
        ]
        confidence = max((float(edge.confidence) for edge in candidates), default=floor)
        if item.status == "contradicted":
            confidence = 1.0 - confidence
        contribution = item.reliability * math.log(
            max(floor, confidence) / max(floor, 1.0 - confidence)
        )
        score += contribution
        if contribution >= 0.0:
            matched.append(item.evidence_id or item.predicate)
        else:
            conflicting.append(item.evidence_id or item.predicate)
    return score, tuple(sorted(matched)), tuple(sorted(conflicting))


def context_compatibility(case: CaseEvidence, profile: SchemaProfile) -> float:
    """Compare context cues with the Point 09 schema context profile."""
    schema_context = set(profile.frame.context_profile)
    observed = {str(key) + "=" + str(value) for key, value in case.context.items()}
    if not observed or not schema_context:
        return 0.0
    intersection = len(observed & schema_context)
    union = len(observed | schema_context)
    raw = (2.0 * intersection - (len(observed - schema_context) * 0.5)) / max(2, 2 * union)
    return max(-1.0, min(1.0, raw))


def goal_compatibility(case: CaseEvidence, profile: SchemaProfile) -> float:
    """Score current goals by the Point 09 schema's learned goal distribution."""
    if not case.goals or not profile.frame.goals:
        return 0.0
    values = [float(profile.frame.goals.get(goal, 0.0)) for goal in case.goals]
    supported = sum(values) / len(values)
    unmatched = sum(goal not in profile.frame.goals for goal in case.goals) / max(1, len(case.goals))
    return max(-0.5, min(1.0, supported - 0.5 * unmatched))


def event_transition_compatibility(
    case: CaseEvidence, profile: SchemaProfile, context: Mapping[str, Any], floor: float
) -> float:
    """Score observed event transitions against schema transition likelihoods."""
    if len(case.event_history) < 2 or not profile.frame.transitions:
        return 0.0
    key = context_signature(context)
    score = 0.0
    count = 0
    event_count = max(1, len(profile.frame.event_types))
    for left, right in zip(case.event_history, case.event_history[1:]):
        matching = [
            row for row in profile.frame.transitions
            if row.get("from_event") == left and row.get("to_event") == right
        ]
        contextual = [row for row in matching if row.get("context_key") == key]
        rows = contextual or [row for row in matching if row.get("context_key") == "*"]
        probability = float(rows[0]["probability"]) if rows else floor
        score += math.log(max(floor, probability) * event_count)
        count += 1
    return score / max(1, count)


def rank_schema_profiles(
    case: CaseEvidence, profiles: Sequence[SchemaProfile], config: InferenceConfig
) -> tuple[CandidateScore, ...]:
    """Rank schema frames and include an explicit unknown/other alternative."""
    if not profiles:
        return ()
    preliminary: list[dict[str, Any]] = []
    for profile in profiles:
        feature, matched, conflicting, evidence_weight = feature_log_likelihood_ratio(
            case, profile, profiles, config
        )
        relation, relation_matched, relation_conflicting = relation_compatibility(
            case, profile, config.probability_floor
        )
        context = context_compatibility(case, profile)
        goal = goal_compatibility(case, profile)
        event = event_transition_compatibility(
            case, profile, case.context, config.probability_floor
        )
        prior = profile.prior_probability
        if prior is None:
            prior = max(1, len(profile.frame.supporting_episode_ids))
        raw_prior = math.log(max(config.probability_floor, float(prior)))
        score = (
            raw_prior + config.feature_weight * feature
            + config.relation_weight * relation / max(1, len(case.relations))
            + config.context_weight * context + config.goal_weight * goal
            + config.event_weight * event
        )
        preliminary.append({
            "profile": profile, "score": score, "feature": feature,
            "relation": relation / max(1, len(case.relations)),
            "context": context, "goal": goal, "event": event,
            "matched": tuple(sorted(set(matched) | set(relation_matched))),
            "conflicting": tuple(sorted(set(conflicting) | set(relation_conflicting))),
            "evidence_weight": evidence_weight,
        })
    unknown_score = config.unknown_log_prior
    scores = [row["score"] for row in preliminary] + [unknown_score]
    probabilities = softmax(scores, config.softmax_temperature)
    output: list[CandidateScore] = []
    for row, posterior in zip(preliminary, probabilities[:-1]):
        profile = row["profile"]
        output.append(CandidateScore(
            schema_id=profile.schema_id, category=profile.frame.name,
            posterior=posterior, log_score=row["score"],
            feature_score=row["feature"], relation_score=row["relation"],
            context_score=row["context"], goal_score=row["goal"], event_score=row["event"],
            matched_evidence_ids=row["matched"], conflicting_evidence_ids=row["conflicting"],
            parent_schema_ids=tuple(profile.frame.parent_schema_ids),
        ))
    output.append(CandidateScore(
        schema_id="unknown", category="unknown/other",
        posterior=probabilities[-1], log_score=unknown_score,
        feature_score=0.0, relation_score=0.0, context_score=0.0,
        goal_score=0.0, event_score=0.0,
    ))
    output.sort(key=lambda item: (-item.posterior, item.schema_id))
    selected = output[:config.max_candidates]
    unknown = next((item for item in output if item.schema_id == "unknown"), None)
    if unknown is not None and all(item.schema_id != "unknown" for item in selected):
        selected = selected[:-1] + [unknown]
        selected.sort(key=lambda item: (-item.posterior, item.schema_id))
    return tuple(selected)


def select_hierarchical_category(
    ranked: Sequence[CandidateScore],
    profiles: Sequence[SchemaProfile],
    config: InferenceConfig,
) -> tuple[str | None, str, float, str]:
    """Select the deepest sufficiently supported category or abstain to unknown/other."""
    if not ranked:
        return None, "unknown/other", 0.0, "no_schema_candidates"
    best = ranked[0]
    second_probability = ranked[1].posterior if len(ranked) > 1 else 0.0
    margin = best.posterior - second_probability
    if best.schema_id != "unknown" and best.posterior >= config.posterior_threshold and margin >= config.candidate_margin:
        return best.schema_id, best.category, best.posterior, "specific_category_selected"
    if best.schema_id != "unknown":
        profile_map = {item.schema_id: item for item in profiles}
        queue = list(best.parent_schema_ids)
        visited: set[str] = set()
        candidate_map = {item.schema_id: item for item in ranked}
        while queue:
            parent_id = queue.pop(0)
            if parent_id in visited:
                continue
            visited.add(parent_id)
            parent_score = candidate_map.get(parent_id)
            parent_profile = profile_map.get(parent_id)
            if parent_score and parent_profile and parent_score.posterior >= config.unknown_confidence_threshold:
                return parent_id, parent_score.category, parent_score.posterior, "broader_parent_selected"
            if parent_profile:
                queue.extend(parent_profile.frame.parent_schema_ids)
    return None, "uncertain/unknown", best.posterior, "abstained_low_confidence_or_margin"


def predict_missing_properties(
    case: CaseEvidence,
    profile: SchemaProfile,
    posterior: float,
    config: InferenceConfig,
) -> tuple[dict[str, Any], ...]:
    """Generate task-relevant unseen feature and missing-role predictions with source labels."""
    output: list[dict[str, Any]] = []
    for feature, distribution in profile.feature_probabilities.items():
        observed = case.features.get(feature)
        if observed and observed.status in {"observed", "contradicted"}:
            continue
        if not distribution:
            continue
        value, probability = max(distribution.items(), key=lambda item: (item[1], item[0]))
        confidence = clamp(posterior * probability)
        if confidence < config.unknown_confidence_threshold:
            continue
        output.append({
            "kind": "feature", "name": feature, "value": value,
            "confidence": confidence, "status": "inferred", "provenance": "schema_prediction",
            "source_schema_id": profile.schema_id,
            "supporting_evidence_ids": [item.evidence_id for item in case.usable_evidence if item.evidence_id],
            "observed": False,
        })
    for name, slot in profile.frame.slots.items():
        feature_name = "role:" + name
        observed = case.features.get(feature_name)
        if observed and observed.status in {"observed", "contradicted"}:
            continue
        if not slot.allowed_type_probabilities:
            continue
        entity_type, probability = max(
            slot.allowed_type_probabilities.items(), key=lambda item: (item[1], item[0])
        )
        confidence = clamp(posterior * slot.coverage * probability)
        if confidence < config.unknown_confidence_threshold:
            continue
        output.append({
            "kind": "missing_role", "name": name, "value": entity_type,
            "confidence": confidence, "status": "inferred", "provenance": "schema_prediction",
            "source_schema_id": profile.schema_id, "supporting_evidence_ids": [],
            "observed": False,
        })
    output.sort(key=lambda item: (-item["confidence"], item["kind"], item["name"]))
    return tuple(output[:config.max_predictions])


def predict_missing_relations(
    case: CaseEvidence, profile: SchemaProfile, posterior: float, config: InferenceConfig
) -> tuple[dict[str, Any], ...]:
    """Predict schema-supported missing directed relations without asserting observation."""
    observed = {
        (item.subject, item.predicate, item.object): item
        for item in case.relations
    }
    output: list[dict[str, Any]] = []
    for edge in profile.frame.relations:
        key = (edge.subject_slot.casefold(), edge.predicate.casefold(), edge.object_slot.casefold())
        evidence = observed.get(key)
        if evidence and evidence.status in {"observed", "contradicted"}:
            continue
        confidence = clamp(posterior * float(edge.confidence))
        if confidence < config.unknown_confidence_threshold:
            continue
        output.append({
            "kind": "relation", "name": edge.predicate,
            "value": {"subject": edge.subject_slot, "predicate": edge.predicate,
                      "object": edge.object_slot},
            "confidence": confidence, "status": "inferred",
            "provenance": "schema_prediction", "source_schema_id": profile.schema_id,
            "evidential_status": edge.evidential_status, "observed": False,
        })
    output.sort(key=lambda item: (-item["confidence"], item["name"]))
    return tuple(output[:config.max_predictions])


def predict_next_events(
    case: CaseEvidence, profile: SchemaProfile, posterior: float, config: InferenceConfig
) -> tuple[dict[str, Any], ...]:
    """Predict next events from the current event using context-specific transitions first."""
    if not case.event_history:
        return ()
    current = case.event_history[-1]
    context_key = context_signature(case.context)
    matching = [
        row for row in profile.frame.transitions if row.get("from_event") == current
    ]
    contextual = [row for row in matching if row.get("context_key") == context_key]
    rows = contextual or [row for row in matching if row.get("context_key") == "*"]
    output = [{
        "kind": "next_event", "name": str(row["to_event"]), "value": row["to_event"],
        "confidence": clamp(posterior * float(row["probability"])),
        "transition_probability": float(row["probability"]), "status": "inferred",
        "provenance": "schema_prediction", "source_schema_id": profile.schema_id,
        "observed": False,
    } for row in rows]
    output.sort(key=lambda item: (-item["confidence"], item["name"]))
    return tuple(output[:config.max_predictions])


def expected_information_gain(
    query: QueryOption,
    ranked: Sequence[CandidateScore],
    profiles: Sequence[SchemaProfile],
    config: InferenceConfig,
) -> QueryRecommendation | None:
    """Estimate entropy reduction for a query from schema-conditioned answer distributions."""
    if not query.available:
        return None
    profile_map = {item.schema_id: item for item in profiles}
    usable = [candidate for candidate in ranked if candidate.schema_id in profile_map]
    if len(usable) < 2:
        return None
    distributions = []
    for candidate in usable:
        dist = _distribution_for_feature(profile_map[candidate.schema_id], query.feature)
        if not dist:
            return None
        distributions.append(dist)
    answer_values = sorted({value for dist in distributions for value in dist})
    if not answer_values:
        return None
    priors = [candidate.posterior for candidate in usable]
    prior_total = sum(priors)
    priors = [value / prior_total for value in priors]
    prior_entropy = entropy(priors, config.probability_floor)
    predicted_answers: dict[str, float] = {}
    expected_posterior_entropy = 0.0
    for answer in answer_values:
        likelihoods = [dist.get(answer, config.probability_floor) for dist in distributions]
        p_answer = sum(prior * likelihood for prior, likelihood in zip(priors, likelihoods))
        predicted_answers[answer] = p_answer
        if p_answer <= 0.0:
            continue
        posterior = [prior * likelihood / p_answer for prior, likelihood in zip(priors, likelihoods)]
        expected_posterior_entropy += p_answer * entropy(posterior, config.probability_floor)
    information_gain = max(0.0, prior_entropy - expected_posterior_entropy)
    net = information_gain - config.query_cost_weight * query.cost - config.query_risk_weight * query.risk
    return QueryRecommendation(
        feature=query.feature, action_label=query.action_label,
        expected_information_gain=information_gain, cost=query.cost, risk=query.risk,
        net_value=net, predicted_answers=predicted_answers,
    )


def select_information_query(
    case: CaseEvidence, ranked: Sequence[CandidateScore],
    profiles: Sequence[SchemaProfile], config: InferenceConfig,
) -> QueryRecommendation | None:
    """Choose the available query with the highest non-negative information value."""
    options = [
        expected_information_gain(query, ranked, profiles, config)
        for query in case.query_options
    ]
    viable = [item for item in options if item is not None]
    if not viable:
        return None
    best = max(viable, key=lambda item: (item.net_value, item.expected_information_gain, item.feature))
    return best if best.net_value > 0.0 else None


def select_action(
    case: CaseEvidence, ranked: Sequence[CandidateScore], config: InferenceConfig
) -> ActionDecision | None:
    """Select a supplied action by posterior expected utility minus its cost and risk."""
    if not case.action_options:
        return None
    output: list[ActionDecision] = []
    for action in case.action_options:
        expected = 0.0
        for candidate in ranked:
            expected += candidate.posterior * float(
                action.utility_by_schema.get(candidate.schema_id,
                                              action.utility_by_schema.get("unknown", 0.0))
            )
        utility = expected - action.cost - config.action_risk_weight * action.risk
        output.append(ActionDecision(
            action_id=action.action_id, expected_utility=utility,
            expected_risk=action.risk, selected=False, description=action.description,
        ))
    best_index = max(range(len(output)), key=lambda index: (output[index].expected_utility, output[index].action_id))
    return ActionDecision(
        action_id=output[best_index].action_id,
        expected_utility=output[best_index].expected_utility,
        expected_risk=output[best_index].expected_risk,
        selected=True, description=output[best_index].description,
    )


def merge_case_evidence(
    previous: CaseEvidence, additional: Sequence[EvidenceItem],
    relations: Sequence[RelationEvidence] = (), event_history: Sequence[str] = (),
) -> CaseEvidence:
    """Create a revised case snapshot and preserve superseded observations in evidence history."""
    features = dict(previous.features)
    history = list(previous.evidence_history)
    for item in additional:
        if item.feature in features:
            existing = features[item.feature]
            if (existing.status in {"observed", "contradicted"}
                    and item.status in {"observed", "contradicted"}
                    and value_key(existing.value) != value_key(item.value)):
                history.append(existing)
            if item.status in {"observed", "contradicted"} or existing.status in {"unknown", "not_observed"}:
                features[item.feature] = item
        else:
            features[item.feature] = item
    return CaseEvidence(
        case_id=previous.case_id, features=features,
        relations=tuple(previous.relations) + tuple(relations),
        event_history=tuple(previous.event_history) + tuple(event_history),
        context=previous.context, goals=previous.goals, query_options=previous.query_options,
        action_options=previous.action_options, source_reference=previous.source_reference,
        evidence_history=tuple(history),
    )
