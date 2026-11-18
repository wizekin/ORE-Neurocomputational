"""Mathematical and functional operations for Point 20 reconstructive memory."""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from .contracts import (
    DetailCandidate, EpisodicFragment, FeedbackRecord, Point20Config, Point20Request,
    Point20State, SchemaProfile, SourceRecord,
)


_WORDS = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a finite value to a closed interval."""
    number = float(value)
    if not math.isfinite(number) or low > high:
        raise ValueError("value must be finite and low must not exceed high")
    return min(high, max(low, number))


def sigmoid(value: float) -> float:
    """Compute a numerically stable logistic activation."""
    number = float(value)
    if number >= 0.0:
        z = math.exp(-min(number, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(number, -700.0))
    return z / (1.0 + z)


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Convert finite scores into a stable temperature-scaled probability distribution."""
    if not scores:
        return []
    scale = float(temperature)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("temperature must be finite and positive")
    values = [float(score) for score in scores]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("scores must be finite")
    maximum = max(values)
    weights = [math.exp(max(-700.0, min(0.0, (value - maximum) / scale))) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def value_key(value: Any) -> str:
    """Create a stable categorical key without conflating booleans, nulls, and text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("feature values must be finite")
        return format(value, ".12g")
    return str(value).strip().casefold()


def feature_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Measure exact categorical overlap or cosine similarity for numeric feature mappings."""
    shared = set(left) & set(right)
    if not shared:
        return 0.0
    categorical = any(
        not isinstance(left[key], (int, float, bool))
        or not isinstance(right[key], (int, float, bool))
        for key in shared
    )
    if categorical:
        return sum(value_key(left[key]) == value_key(right[key]) for key in shared) / len(shared)
    a = [float(left[key]) for key in shared]
    b = [float(right[key]) for key in shared]
    if any(not math.isfinite(x) for x in a + b):
        raise ValueError("numeric feature values must be finite")
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return clamp(sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b), -1.0, 1.0)


def _token_overlap(left: str, right: str) -> float:
    """Estimate text-cue overlap with token-set Jaccard similarity."""
    a = set(_WORDS.findall(left.casefold()))
    b = set(_WORDS.findall(right.casefold()))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def define_retrieval_target(request: Point20Request) -> dict[str, Any]:
    """Represent the retrieval question, target features, precision, and task context."""
    cue_features = dict(request.cue) if isinstance(request.cue, Mapping) else {}
    targets = list(request.target_features)
    if not targets:
        targets = sorted({item.feature_type for item in request.episodic_fragments})
    precision_weights = {
        "gist": {"gist": 0.75, "detail": 0.25, "source": 0.35},
        "balanced": {"gist": 0.45, "detail": 0.55, "source": 0.70},
        "exact": {"gist": 0.15, "detail": 0.85, "source": 1.0},
    }[request.precision]
    return {
        "event_id": request.event_id,
        "cue": dict(request.cue) if isinstance(request.cue, Mapping) else request.cue,
        "cue_features": cue_features,
        "target_features": targets,
        "task_type": request.task_type,
        "precision": request.precision,
        "precision_weights": precision_weights,
        "target_episode_id": request.target_episode_id,
        "context": dict(request.context),
        "what_happened_vs_what_is_typical": "reconstruct_this_event",
    }


def reactivate_episodic_fragments(
    request: Point20Request,
    target: Mapping[str, Any],
    config: Point20Config,
) -> dict[str, Any]:
    """Rank event-specific fragments from cue overlap while preserving episode identity."""
    cue = target["cue"]
    cue_features = target["cue_features"]
    context_cues = request.context.get("retrieval_context_features", {})
    if not isinstance(context_cues, Mapping):
        context_cues = {}
    ranked: list[dict[str, Any]] = []
    for fragment in request.episodic_fragments:
        if cue_features:
            cue_fit = feature_similarity(cue_features, fragment.cue_features)
        elif isinstance(cue, str):
            cue_fit = _token_overlap(cue, str(fragment.content))
        else:
            cue_fit = 0.0
        context_fit = feature_similarity(context_cues, fragment.source_features) if context_cues else 0.0
        activation = (
            0.55 * fragment.retrieval_strength
            + 0.30 * fragment.episodic_context
            + 0.10 * cue_fit
            + 0.05 * context_fit
        )
        if cue_features or (isinstance(cue, str) and cue.strip()):
            activation *= 0.35 + 0.65 * max(cue_fit, context_fit)
        ranked.append({
            "detail_id": fragment.detail_id,
            "feature_type": fragment.feature_type,
            "content": fragment.content,
            "episode_id": fragment.episode_id,
            "source_type": fragment.source_type,
            "source_id": fragment.source_id,
            "cue_fit": cue_fit,
            "context_fit": context_fit,
            "activation": clamp(activation),
            "status": fragment.status,
            "retrieval_strength": fragment.retrieval_strength,
            "episodic_context": fragment.episodic_context,
            "attention_encoding": fragment.attention_encoding,
            "centrality": fragment.centrality,
            "confidence": clamp(0.55 * fragment.retrieval_strength + 0.45 * fragment.episodic_context),
        })
    probabilities = softmax([row["activation"] for row in ranked], config.retrieval_temperature)
    for row, probability in zip(ranked, probabilities):
        row["normalized_activation"] = probability
        row["reactivated"] = row["activation"] >= 0.18
    ranked.sort(key=lambda item: (-item["activation"], item["detail_id"]))
    return {
        "fragments": ranked,
        "reactivated_ids": [row["detail_id"] for row in ranked if row["reactivated"]],
        "episode_ids": sorted({row["episode_id"] for row in ranked if row["reactivated"] and row["episode_id"]}),
        "normalization_sum": sum(probabilities),
        "not_retrieved_is_not_absent": True,
    }


def build_feature_ledger(
    request: Point20Request,
    retrieval: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a per-feature evidence ledger that distinguishes unknown, contradicted, and unqueried."""
    target_features = list(request.target_features)
    if not target_features:
        target_features = sorted({item.feature_type for item in request.episodic_fragments})
    reactivated_ids = set(retrieval.get("reactivated_ids", ()))
    rows: dict[str, dict[str, Any]] = {}
    for feature in target_features:
        fragments = [item for item in request.episodic_fragments if item.feature_type == feature]
        active = [item for item in fragments if item.detail_id in reactivated_ids]
        direct = [item for item in active if item.status == "direct_evidence"]
        retrieved = [item for item in active if item.status == "episodically_retrieved"]
        weak = [item for item in active if item.status in {"weakly_retrieved", "source_uncertain"}]
        contradicted = [item for item in active if item.status == "contradicted"]
        not_retrieved = [item for item in active if item.status == "not_retrieved"]
        supporting = direct + retrieved + weak
        if contradicted and not supporting:
            status = "contradicted"
        elif direct:
            status = "direct_evidence"
        elif retrieved:
            status = "episodically_retrieved"
        elif weak:
            status = "weakly_retrieved"
        elif not_retrieved:
            status = "not_retrieved"
        else:
            status = "not_retrieved"
        rows[feature] = {
            "feature_type": feature,
            "status": status,
            "values": [
                {
                    "detail_id": item.detail_id,
                    "content": item.content,
                    "status": item.status,
                    "retrieval_strength": item.retrieval_strength,
                    "episodic_context": item.episodic_context,
                    "source_type": item.source_type,
                    "source_id": item.source_id,
                    "episode_id": item.episode_id,
                }
                for item in active
            ],
            "evidence_strength": max(
                (item.retrieval_strength * item.attention_encoding for item in supporting),
                default=0.0,
            ),
            "likely_encoded": max((item.attention_encoding for item in fragments), default=0.0),
            "centrality": max((item.centrality for item in fragments), default=0.0),
            "unavailable_not_absent": status in {"not_retrieved", "weakly_retrieved"},
        }
    return {
        "features": rows,
        "retrieved_feature_count": sum(row["status"] in {"direct_evidence", "episodically_retrieved"} for row in rows.values()),
        "missing_feature_count": sum(row["status"] in {"not_retrieved", "weakly_retrieved"} for row in rows.values()),
        "contradicted_feature_count": sum(row["status"] == "contradicted" for row in rows.values()),
        "status_definitions": {
            "not_retrieved": "No sufficiently strong event-specific fragment was reactivated; this does not establish absence.",
            "contradicted": "A supplied, source-labelled observation conflicts with a candidate value.",
            "not_queried": "A feature outside the current target set was not assessed.",
        },
    }


def _schema_evidence_fit(
    schema: SchemaProfile,
    ledger: Mapping[str, Any],
) -> float:
    """Score how well a schema predicts available evidence without treating unknowns as mismatches."""
    scores: list[tuple[float, float]] = []
    for feature, row in ledger["features"].items():
        if row["status"] not in {"direct_evidence", "episodically_retrieved", "weakly_retrieved"}:
            continue
        distribution = schema.feature_probabilities.get(feature, {})
        if not distribution:
            continue
        for item in row["values"]:
            if item["status"] not in {"direct_evidence", "episodically_retrieved", "weakly_retrieved"}:
                continue
            value_probability = distribution.get(value_key(item["content"]), 0.0)
            reliability = clamp(item["retrieval_strength"])
            score = value_probability if value_probability > 0.0 else 0.01
            scores.append((score, reliability))
    if not scores:
        return 0.5
    log_fit = sum(weight * math.log(max(1e-9, score)) for score, weight in scores)
    weight_total = sum(weight for _, weight in scores)
    return clamp(math.exp(log_fit / max(1e-9, weight_total)))


def activate_candidate_schemas(
    request: Point20Request,
    ledger: Mapping[str, Any],
    config: Point20Config,
) -> dict[str, Any]:
    """Activate multiple candidate schemas from evidence fit, context fit, and goal relevance."""
    context = request.context
    goal_values = context.get("schema_goal_relevance", {})
    if not isinstance(goal_values, Mapping):
        goal_values = {}
    ranked: list[dict[str, Any]] = []
    for schema in request.schemas:
        evidence_fit = _schema_evidence_fit(schema, ledger)
        context_fit = feature_similarity(schema.context_profile, {
            str(key): value for key, value in context.items()
            if isinstance(value, (int, float, bool))
        }) if schema.context_profile else 0.0
        goal_fit = clamp(float(goal_values.get(schema.schema_id, schema.goal_relevance)))
        log_prior = math.log(max(config.epsilon, schema.prior_probability))
        score = 0.58 * evidence_fit + 0.27 * context_fit + 0.15 * goal_fit + 0.10 * log_prior
        ranked.append({
            "schema_id": schema.schema_id,
            "name": schema.name,
            "evidence_fit": evidence_fit,
            "context_fit": context_fit,
            "goal_relevance": goal_fit,
            "prior_probability": schema.prior_probability,
            "score": score,
            "cultural_scope": schema.cultural_scope,
        })
    probabilities = softmax([item["score"] for item in ranked], config.schema_temperature)
    for row, probability in zip(ranked, probabilities):
        row["activation_probability"] = probability
    ranked.sort(key=lambda item: (-item["activation_probability"], item["schema_id"]))
    return {
        "ranked_schemas": ranked,
        "active_schema_ids": [item["schema_id"] for item in ranked[:config.max_active_schemas]],
        "normalization_sum": sum(probabilities),
        "schema_is_hypothesis_not_event_record": True,
    }


def generate_inferred_completions(
    request: Point20Request,
    ledger: Mapping[str, Any],
    schema_activation: Mapping[str, Any],
    config: Point20Config,
) -> dict[str, Any]:
    """Generate ranked alternatives only for missing or weak features and mark each as inferred."""
    schema_by_id = {item.schema_id: item for item in request.schemas}
    activation_by_id = {row["schema_id"]: row["activation_probability"]
                        for row in schema_activation["ranked_schemas"]}
    candidates: list[DetailCandidate] = []
    omitted: list[dict[str, Any]] = []
    for feature, feature_row in ledger["features"].items():
        if feature_row["status"] not in {"not_retrieved", "weakly_retrieved"}:
            continue
        proposals: list[tuple[float, SchemaProfile, str, float]] = []
        for schema_id in schema_activation["active_schema_ids"]:
            schema = schema_by_id[schema_id]
            distribution = schema.feature_probabilities.get(feature, {})
            for value, prior in distribution.items():
                score = activation_by_id.get(schema_id, 0.0) * prior
                proposals.append((score, schema, value, prior))
        proposals.sort(key=lambda row: (-row[0], row[1].schema_id, row[2]))
        for score, schema, value, prior in proposals[:config.max_completions_per_feature]:
            candidate_id = f"{schema.schema_id}:{feature}:{value}"
            candidates.append(DetailCandidate(
                candidate_id=candidate_id,
                feature_type=feature,
                content=value,
                status="inferred",
                schema_id=schema.schema_id,
                schema_support=activation_by_id.get(schema.schema_id, 0.0),
                schema_prior=clamp(prior * max(activation_by_id.get(schema.schema_id, 0.0), config.epsilon)),
                inference_proportion=1.0,
                provenance=[f"schema:{schema.schema_id}", "generated_as_inference"],
            ))
        if not proposals:
            omitted.append({"feature_type": feature, "reason": "no compatible schema proposal"})
    return {
        "candidates": candidates,
        "candidate_count": len(candidates),
        "omitted_features": omitted,
        "all_candidates_labelled_inferred": all(item.status == "inferred" for item in candidates),
    }


def evaluate_candidates_bayesian(
    candidates: Sequence[DetailCandidate],
    fragments: Sequence[EpisodicFragment],
    config: Point20Config,
) -> dict[str, Any]:
    """Combine schema priors with fragment likelihoods while retaining each evidence contribution."""
    evaluated: list[DetailCandidate] = []
    raw_scores: dict[str, float] = {}
    for candidate in candidates:
        related = [item for item in fragments if item.feature_type == candidate.feature_type
                   and item.status in {"direct_evidence", "episodically_retrieved", "weakly_retrieved", "contradicted"}]
        log_likelihood = 0.0
        support_total = 0.0
        support_weight = 0.0
        contradiction_total = 0.0
        contradiction_weight = 0.0
        context_total = 0.0
        for fragment in related:
            reliability = clamp(fragment.retrieval_strength * fragment.attention_encoding)
            match = value_key(fragment.content) == value_key(candidate.content)
            if fragment.status == "contradicted":
                match = not match
            if match:
                support_total += reliability * fragment.episodic_context
                support_weight += reliability
                log_likelihood += math.log(max(config.epsilon, reliability))
                context_total += fragment.episodic_context * reliability
                candidate.evidence_ids.append(fragment.detail_id)
            else:
                if fragment.status != "weakly_retrieved":
                    contradiction_total += reliability
                    contradiction_weight += reliability
                    log_likelihood += math.log(max(config.epsilon, 1.0 - reliability))
        candidate.evidence_strength = clamp(support_total / max(config.epsilon, support_weight))
        candidate.episodic_context = clamp(context_total / max(config.epsilon, support_weight))
        candidate.contradiction = clamp(contradiction_total / max(config.epsilon, contradiction_weight))
        candidate.evidence_likelihood = clamp(math.exp(max(-60.0, log_likelihood)))
        candidate.source_specificity = candidate.episodic_context
        log_prior = math.log(max(config.epsilon, candidate.schema_prior))
        raw_score = log_prior + log_likelihood - config.contradiction_penalty * candidate.contradiction
        raw_scores[candidate.candidate_id] = raw_score
        evaluated.append(candidate)
    grouped: dict[str, list[DetailCandidate]] = {}
    for candidate in evaluated:
        grouped.setdefault(candidate.feature_type, []).append(candidate)
    for feature, group in grouped.items():
        probabilities = softmax([raw_scores[item.candidate_id] for item in group], config.posterior_temperature)
        for candidate, probability in zip(group, probabilities):
            candidate.posterior = probability
    return {
        "candidates": evaluated,
        "candidate_count": len(evaluated),
        "feature_groups": {
            feature: {
                "candidate_ids": [item.candidate_id for item in group],
                "posterior_sum": sum(item.posterior for item in group),
            }
            for feature, group in grouped.items()
        },
        "bayesian_caution": "Posterior ranking is conditional on supplied priors and likelihood assumptions; it is not direct evidence of event occurrence.",
    }


def check_spatiotemporal_causal_constraints(
    candidates: Sequence[DetailCandidate],
    request: Point20Request,
) -> dict[str, Any]:
    """Score candidate values against explicit temporal, spatial, participant, and causal constraints."""
    constraints = request.context.get("constraints", ())
    if isinstance(constraints, Mapping):
        constraints = [constraints]
    if not isinstance(constraints, Sequence) or isinstance(constraints, (str, bytes)):
        raise TypeError("context.constraints must be a sequence of constraint mappings")
    updated: list[DetailCandidate] = []
    for candidate in candidates:
        results: list[dict[str, Any]] = []
        penalty = 0.0
        total_weight = 0.0
        for index, raw in enumerate(constraints):
            if not isinstance(raw, Mapping):
                continue
            feature = str(raw.get("feature_type", raw.get("feature", ""))).casefold()
            if feature and feature != candidate.feature_type:
                continue
            weight = clamp(float(raw.get("weight", 1.0)))
            allowed = raw.get("allowed_values")
            forbidden = raw.get("forbidden_values", ())
            minimum = raw.get("minimum")
            maximum = raw.get("maximum")
            satisfied = True
            reason = "no applicable constraint"
            if allowed is not None:
                allowed_values = {value_key(value) for value in allowed}
                satisfied = value_key(candidate.content) in allowed_values
                reason = "allowed_value_match" if satisfied else "not_in_allowed_values"
            if value_key(candidate.content) in {value_key(value) for value in forbidden}:
                satisfied = False
                reason = "explicitly_forbidden_value"
            if minimum is not None or maximum is not None:
                try:
                    number = float(candidate.content)
                    satisfied = satisfied and (minimum is None or number >= float(minimum))
                    satisfied = satisfied and (maximum is None or number <= float(maximum))
                    reason = "numeric_range_satisfied" if satisfied else "numeric_range_violated"
                except (TypeError, ValueError):
                    satisfied = False
                    reason = "non_numeric_value_for_numeric_constraint"
            if allowed is None and not forbidden and minimum is None and maximum is None:
                continue
            total_weight += weight
            if not satisfied:
                penalty += weight
            results.append({
                "constraint_id": str(raw.get("constraint_id", f"constraint-{index}")),
                "constraint_type": str(raw.get("constraint_type", "feature_value")),
                "satisfied": satisfied,
                "weight": weight,
                "hard": bool(raw.get("hard", False)),
                "reason": reason,
            })
        candidate.constraint_results = results
        candidate.contextual_fit = clamp(1.0 - penalty / max(config_epsilon(), total_weight))
        candidate.contradiction = clamp(
            candidate.contradiction + sum(
                item["weight"] for item in results if not item["satisfied"] and item["hard"]
            )
        )
        updated.append(candidate)
    return {
        "candidates": updated,
        "constraint_count": len(constraints),
        "violations": [
            {"candidate_id": candidate.candidate_id, **item}
            for candidate in updated for item in candidate.constraint_results if not item["satisfied"]
        ],
        "coherence_is_not_truth": True,
    }


def config_epsilon() -> float:
    """Return a small positive denominator for normalized constraint aggregation."""
    return 1e-9


def monitor_candidate_sources(
    candidates: Sequence[DetailCandidate],
    fragments: Sequence[EpisodicFragment],
    sources: Sequence[SourceRecord],
    config: Point20Config,
) -> dict[str, Any]:
    """Estimate source probabilities from source cues and explicit lineage rather than familiarity alone."""
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        scores = {name: 0.01 for name in (
            "original_perception", "episodic_retrieval", "schema_inference",
            "imagination", "post_event_information", "later_testimony", "unknown",
        )}
        scores["schema_inference"] += 0.75
        evidence_ids = set(candidate.evidence_ids)
        matching_fragments = [
            item for item in fragments
            if item.feature_type == candidate.feature_type
            and value_key(item.content) == value_key(candidate.content)
        ]
        for fragment in matching_fragments:
            source_type = fragment.source_type
            scores[source_type] = scores.get(source_type, 0.0) + (
                fragment.reliability if hasattr(fragment, "reliability") else
                fragment.retrieval_strength * fragment.episodic_context
            )
            if fragment.detail_id in evidence_ids:
                scores[source_type] += 0.25
        for source in sources:
            detail_match = source.detail_id in {candidate.candidate_id, candidate.feature_type, ""}
            content_match = value_key(source.content) == value_key(candidate.content) if source.content is not None else False
            if detail_match and content_match:
                scores[source.source_type] = scores.get(source.source_type, 0.0) + source.reliability
        probabilities = softmax(
            [math.log(max(config.epsilon, scores[name])) for name in scores],
            config.source_temperature,
        )
        ranked = sorted(zip(scores, probabilities), key=lambda pair: pair[1], reverse=True)
        source_probabilities = {name: probability for name, probability in zip(scores, probabilities)}
        assigned, top_probability = ranked[0]
        candidate.source_probabilities = source_probabilities
        candidate.assigned_source = assigned if top_probability >= config.source_assignment_threshold else "unknown"
        candidate.source_uncertainty = clamp(1.0 - top_probability)
        candidate.source_specificity = max(
            candidate.source_specificity,
            source_probabilities.get("original_perception", 0.0)
            + source_probabilities.get("episodic_retrieval", 0.0),
        )
        if candidate.assigned_source == "unknown":
            candidate.status = "source_uncertain"
        elif candidate.assigned_source in {"original_perception", "episodic_retrieval"}:
            if candidate.evidence_strength >= config.retrieval_promotion_threshold and candidate.episodic_context >= config.retrieval_promotion_threshold and candidate.contradiction < 0.25:
                candidate.status = "episodically_retrieved"
                candidate.inference_proportion = 0.0
            else:
                candidate.status = "source_uncertain"
        elif candidate.assigned_source in {"post_event_information", "later_testimony"}:
            candidate.status = "suggested_after_event"
        results.append({
            "candidate_id": candidate.candidate_id,
            "source_probabilities": source_probabilities,
            "assigned_source": candidate.assigned_source,
            "source_uncertainty": candidate.source_uncertainty,
            "source_specificity": candidate.source_specificity,
        })
    return {"candidates": list(candidates), "source_assignments": results}


def assess_familiarity_and_recollection(
    candidates: Sequence[DetailCandidate],
    fragments: Sequence[EpisodicFragment],
    sources: Sequence[SourceRecord],
    precision: str,
) -> dict[str, Any]:
    """Separate repetition-based familiarity from event-specific contextual recollection."""
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        matches = [
            item for item in fragments
            if item.feature_type == candidate.feature_type
            and value_key(item.content) == value_key(candidate.content)
        ]
        familiar_sources = [
            item for item in sources
            if item.content is not None and value_key(item.content) == value_key(candidate.content)
        ]
        repetition = min(1.0, max(0, len(familiar_sources) - 1) / 4.0)
        fragment_familiarity = max((item.familiarity for item in matches), default=0.0)
        familiarity = clamp(0.55 * fragment_familiarity + 0.45 * repetition)
        recollection = clamp(
            sum(item.recollection * item.episodic_context * item.retrieval_strength for item in matches)
            / max(1, len(matches))
        )
        uncertainty = candidate.source_uncertainty
        familiarity_weight = 0.20 if precision == "exact" else 0.35
        recollection_weight = 0.65 if precision == "exact" else 0.50
        decision_signal = (
            familiarity_weight * familiarity
            + recollection_weight * recollection
            - 0.35 * uncertainty
        )
        candidate.familiarity = familiarity
        candidate.recollection = recollection
        output.append({
            "candidate_id": candidate.candidate_id,
            "familiarity": familiarity,
            "source_specific_recollection": recollection,
            "source_uncertainty": uncertainty,
            "decision_signal": decision_signal,
            "familiarity_not_proof": True,
        })
    return {"candidates": list(candidates), "dimensions": output}


def search_competing_episodes(
    candidates: Sequence[DetailCandidate],
    request: Point20Request,
    fragments: Sequence[EpisodicFragment],
) -> dict[str, Any]:
    """Compare candidate details with similar episodes to identify source blending and overlap."""
    comparisons: list[dict[str, Any]] = []
    for candidate in candidates:
        matches: list[str] = []
        target_support = any(
            item.feature_type == candidate.feature_type
            and value_key(item.content) == value_key(candidate.content)
            and (not request.target_episode_id or item.episode_id == request.target_episode_id)
            for item in fragments
        )
        for episode in request.competing_episodes:
            episode_id = str(episode.get("episode_id", ""))
            raw_fragments = episode.get("fragments", ())
            if isinstance(raw_fragments, Mapping):
                raw_fragments = [raw_fragments]
            for raw in raw_fragments:
                if not isinstance(raw, Mapping):
                    continue
                feature = str(raw.get("feature_type", raw.get("feature", ""))).casefold()
                value = raw.get("content", raw.get("value"))
                if feature == candidate.feature_type and value_key(value) == value_key(candidate.content):
                    matches.append(episode_id or "unspecified_competing_episode")
                    break
        candidate.competing_episode_ids = sorted(set(matches))
        only_competing_source = bool(matches) and not target_support
        if only_competing_source:
            candidate.contradiction = clamp(candidate.contradiction + 0.15)
            candidate.source_uncertainty = clamp(candidate.source_uncertainty + 0.10)
        comparisons.append({
            "candidate_id": candidate.candidate_id,
            "target_episode_support": target_support,
            "matching_competing_episode_ids": candidate.competing_episode_ids,
            "possible_episode_blend": only_competing_source,
            "absence_of_target_match_is_not_a_contradiction": True,
        })
    return {"candidates": list(candidates), "comparisons": comparisons}


def calibrate_detail_confidence(
    candidates: Sequence[DetailCandidate],
    config: Point20Config,
) -> dict[str, Any]:
    """Estimate each detail's confidence independently of how coherent the total story sounds."""
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        inferred = 1.0 if candidate.status == "inferred" or candidate.inference_proportion >= 0.99 else candidate.inference_proportion
        logit = (
            config.calibration_bias
            + config.evidence_weight * candidate.evidence_strength
            + config.recollection_weight * candidate.recollection
            - config.inference_penalty * inferred
            - config.contradiction_penalty * candidate.contradiction
            - config.source_uncertainty_penalty * candidate.source_uncertainty
        )
        confidence = sigmoid(logit)
        candidate.confidence = confidence
        results.append({
            "candidate_id": candidate.candidate_id,
            "confidence": confidence,
            "evidence_strength": candidate.evidence_strength,
            "source_specific_recollection": candidate.recollection,
            "inferred_proportion": inferred,
            "contradiction": candidate.contradiction,
            "source_uncertainty": candidate.source_uncertainty,
            "posterior": candidate.posterior,
            "coherence_used_as_confidence": False,
        })
    return {
        "candidates": list(candidates),
        "per_detail": results,
        "calibration_note": "Illustrative logistic coefficients require empirical calibration; subjective confidence is not accuracy.",
    }


def choose_response_policy(
    request: Point20Request,
    ledger: Mapping[str, Any],
    candidates: Sequence[DetailCandidate],
    config: Point20Config,
) -> dict[str, Any]:
    """Select report, qualify, clarify, or omit using task-specific expected-loss costs."""
    high_stakes = str(request.context.get("stakes", "normal")).casefold() in {"high", "critical"}
    precision = request.precision
    false_claim_cost = 5.0 if high_stakes or precision == "exact" else 2.0
    omission_cost = 0.8 if precision == "gist" else 1.4
    clarification_cost = 0.25 if high_stakes or precision == "exact" else 0.45
    grouped: dict[str, list[DetailCandidate]] = {}
    for item in candidates:
        grouped.setdefault(item.feature_type, []).append(item)
    decisions: list[dict[str, Any]] = []
    feature_importance = request.context.get("feature_importance", {})
    if not isinstance(feature_importance, Mapping):
        feature_importance = {}
    for feature, feature_row in ledger["features"].items():
        importance = clamp(float(feature_importance.get(feature, feature_row["centrality"] or 0.5)))
        options: list[dict[str, Any]] = []
        for candidate in grouped.get(feature, []):
            probability = clamp(candidate.confidence)
            report_loss = (1.0 - probability) * false_claim_cost
            qualify_loss = 0.28 * (1.0 - probability) * false_claim_cost + 0.22
            clarify_loss = clarification_cost + 0.10 * (1.0 - probability)
            omit_loss = omission_cost * importance
            losses = {
                "report": report_loss,
                "qualify": qualify_loss,
                "clarify": clarify_loss,
                "omit": omit_loss,
            }
            action = min(losses, key=losses.get)
            if candidate.status in {"inferred", "source_uncertain"} and action == "report":
                action = "qualify"
            options.append({
                "candidate_id": candidate.candidate_id,
                "content": candidate.content,
                "action": action,
                "expected_losses": losses,
                "confidence": probability,
                "status": candidate.status,
            })
        if options:
            selected = min(
                options,
                key=lambda item: (
                    item["expected_losses"][item["action"]],
                    item["candidate_id"],
                ),
            )
            decisions.append({
                "feature_type": feature,
                "selected": selected,
                "alternatives": options,
                "reason": "minimum expected response loss subject to provenance/status guardrails",
            })
        elif feature_row["status"] in {"not_retrieved", "weakly_retrieved"}:
            action = "clarify" if high_stakes or precision == "exact" or importance >= 0.7 else "omit"
            decisions.append({
                "feature_type": feature,
                "selected": {"candidate_id": None, "content": None, "action": action,
                             "expected_losses": {"clarify": clarification_cost, "omit": omission_cost * importance},
                             "confidence": 0.0, "status": "unknown"},
                "alternatives": [],
                "reason": "no supported or schema-generated candidate; preserve unknown status",
            })
    return {
        "decisions": decisions,
        "reported_features": [item["feature_type"] for item in decisions if item["selected"]["action"] == "report"],
        "qualified_features": [item["feature_type"] for item in decisions if item["selected"]["action"] == "qualify"],
        "clarification_features": [item["feature_type"] for item in decisions if item["selected"]["action"] == "clarify"],
        "omitted_features": [item["feature_type"] for item in decisions if item["selected"]["action"] == "omit"],
        "expected_loss_policy": {
            "false_claim_cost": false_claim_cost,
            "omission_cost": omission_cost,
            "clarification_cost": clarification_cost,
            "high_stakes": high_stakes,
            "precision": precision,
        },
    }


def _has_cycle(nodes: Mapping[str, Any], edges: Sequence[tuple[str, str]]) -> bool:
    """Detect cycles in a provenance graph with a depth-first traversal."""
    adjacency: dict[str, list[str]] = {key: [] for key in nodes}
    for left, right in edges:
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, [])
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        """Return true when a recursive edge reaches a node already on the active path."""
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for child in adjacency.get(node, ()):
            if visit(child):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in adjacency if node not in visited)


def build_provenance_graph(
    request: Point20Request,
    candidates: Sequence[DetailCandidate],
) -> dict[str, Any]:
    """Build a source/detail/schema provenance graph and preserve the timeline of later information."""
    nodes: dict[str, dict[str, Any]] = {
        f"event:{request.event_id}": {"node_type": "original_event", "event_id": request.event_id}
    }
    edges: list[tuple[str, str]] = []
    source_records = list(request.source_history)
    source_by_id = {item.source_id: item for item in source_records}
    for fragment in request.episodic_fragments:
        source_id = fragment.source_id or f"fragment-source:{fragment.detail_id}"
        if source_id not in source_by_id:
            source_by_id[source_id] = SourceRecord(
                source_id=source_id,
                source_type=fragment.source_type,
                detail_id=fragment.detail_id,
                content=fragment.content,
                reliability=fragment.retrieval_strength,
                timestamp=fragment.timestamp,
                parent_source_ids=fragment.provenance_ids,
                source_features=fragment.source_features,
                operation="episodic_fragment",
            )
    source_records = list(source_by_id.values())
    for source in source_records:
        nodes[f"source:{source.source_id}"] = {
            "node_type": "source",
            **source.to_dict(),
        }
        parents = source.parent_source_ids or (f"event:{request.event_id}",)
        for parent in parents:
            parent_key = parent if parent.startswith(("source:", "event:", "detail:")) else f"source:{parent}"
            if parent_key not in nodes:
                nodes[parent_key] = {"node_type": "unresolved_parent_reference", "source_id": parent}
            edges.append((parent_key, f"source:{source.source_id}"))
        if source.detail_id:
            detail_key = f"detail:{source.detail_id}"
            nodes.setdefault(detail_key, {"node_type": "detail", "detail_id": source.detail_id})
            edges.append((f"source:{source.source_id}", detail_key))
    for candidate in candidates:
        candidate_key = f"candidate:{candidate.candidate_id}"
        nodes[candidate_key] = {
            "node_type": "candidate_detail",
            "candidate_id": candidate.candidate_id,
            "feature_type": candidate.feature_type,
            "content": candidate.content,
            "status": candidate.status,
            "assigned_source": candidate.assigned_source,
            "provenance": list(candidate.provenance),
        }
        if candidate.schema_id:
            schema_key = f"schema:{candidate.schema_id}"
            nodes.setdefault(schema_key, {"node_type": "schema", "schema_id": candidate.schema_id})
            edges.append((schema_key, candidate_key))
        for source_id in candidate.evidence_ids:
            source_key = f"detail:{source_id}"
            nodes.setdefault(source_key, {"node_type": "detail", "detail_id": source_id})
            edges.append((source_key, candidate_key))
    feedback_timeline: list[dict[str, Any]] = []
    for feedback in request.feedback:
        source_key = f"source:{feedback.source_id}"
        nodes.setdefault(source_key, {
            "node_type": "feedback_source",
            "source_id": feedback.source_id,
            "source_type": feedback.source_type,
            "reliability": feedback.reliability,
            "timestamp": feedback.timestamp,
            "independence_group": feedback.independence_group,
        })
        feedback_key = f"feedback:{feedback.feedback_id}"
        nodes[feedback_key] = {
            "node_type": "feedback",
            **feedback.to_dict(),
        }
        edges.append((source_key, feedback_key))
        feedback_timeline.append({
            "feedback_id": feedback.feedback_id,
            "feature_type": feedback.feature_type,
            "source_id": feedback.source_id,
            "timestamp": feedback.timestamp,
            "verified": feedback.verified,
        })
    cycle = _has_cycle(nodes, edges)
    if cycle:
        raise ValueError("source provenance contains a cycle; lineage must be a directed acyclic graph")
    return {
        "nodes": nodes,
        "edges": [{"from": left, "to": right} for left, right in edges],
        "source_timeline": [
            {"source_id": item.source_id, "source_type": item.source_type,
             "detail_id": item.detail_id, "timestamp": item.timestamp,
             "independence_group": item.independence_group, "operation": item.operation}
            for item in sorted(source_records, key=lambda row: (row.timestamp, row.source_id))
        ],
        "feedback_timeline": sorted(feedback_timeline, key=lambda row: (row["timestamp"], row["feedback_id"])),
        "cycle_detected": cycle,
        "earliest_source_preserved": True,
        "repeated_retelling_is_not_original_perception": True,
    }


def monitor_retelling_and_post_event_information(
    request: Point20Request,
    state: Point20State,
    candidates: Sequence[DetailCandidate],
) -> dict[str, Any]:
    """Track repeated reports and later suggestions without overwriting the earliest source labels."""
    graph = build_provenance_graph(request, candidates)
    sources = graph["source_timeline"]
    after_event = [
        item for item in sources
        if item["source_type"] in {"post_event_information", "later_testimony", "imagination", "schema_inference"}
    ]
    repeated: dict[str, int] = {}
    for item in sources:
        if item["detail_id"]:
            repeated[item["detail_id"]] = repeated.get(item["detail_id"], 0) + 1
    drift = [
        {"detail_id": detail_id, "source_count": count,
         "retelling_or_post_event_exposure": count > 1}
        for detail_id, count in repeated.items()
    ]
    state.source_versions.extend(
        {"source_id": item["source_id"], "source_type": item["source_type"],
         "detail_id": item["detail_id"], "timestamp": item["timestamp"]}
        for item in sources
    )
    state.source_versions = state.source_versions[-500:]
    return {
        "provenance_graph": graph,
        "post_event_or_reconstruction_sources": after_event,
        "repeated_detail_exposure": drift,
        "source_relabeling_performed": False,
        "reconsolidation_assumption": "No claim that every retrieval triggers reconsolidation.",
    }


def update_reconstruction_from_feedback(
    candidates: Sequence[DetailCandidate],
    feedback: Sequence[FeedbackRecord],
    state: Point20State,
    config: Point20Config,
) -> dict[str, Any]:
    """Update candidate odds with reliable independent feedback while preserving prior versions."""
    updates: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    by_id = {item.candidate_id: item for item in candidates}
    for record in sorted(feedback, key=lambda item: (item.timestamp, item.feedback_id)):
        group = record.independence_group or record.source_id
        if group in state.feedback_groups_seen:
            ignored.append({"feedback_id": record.feedback_id, "reason": "duplicate_independence_group"})
            continue
        if record.reliability < config.min_feedback_reliability:
            ignored.append({"feedback_id": record.feedback_id, "reason": "source_reliability_below_threshold"})
            continue
        matches = []
        if record.candidate_id and record.candidate_id in by_id:
            matches = [by_id[record.candidate_id]]
        else:
            matches = [
                item for item in candidates
                if item.feature_type == record.feature_type
                and value_key(item.content) == value_key(record.value)
            ]
        if not matches:
            ignored.append({"feedback_id": record.feedback_id, "reason": "no_matching_candidate"})
            continue
        state.feedback_groups_seen.append(group)
        for candidate in matches:
            prior = clamp(candidate.posterior if candidate.posterior > 0.0 else candidate.confidence)
            prior_odds = prior / max(config.epsilon, 1.0 - prior)
            likelihood_if_true = record.reliability if record.supports else 1.0 - record.reliability
            likelihood_if_false = 1.0 - record.reliability if record.supports else record.reliability
            likelihood_ratio = likelihood_if_true / max(config.epsilon, likelihood_if_false)
            posterior_odds = prior_odds * likelihood_ratio
            updated = clamp(posterior_odds / (1.0 + posterior_odds))
            before = candidate.posterior
            candidate.posterior = updated
            candidate.feedback_history.append({
                "feedback_id": record.feedback_id,
                "source_id": record.source_id,
                "reliability": record.reliability,
                "supports": record.supports,
                "prior_probability": prior,
                "likelihood_ratio": likelihood_ratio,
                "posterior_probability": updated,
            })
            state.reconstruction_versions.append({
                "event_id": "current",
                "candidate_id": candidate.candidate_id,
                "feedback_id": record.feedback_id,
                "posterior_before": before,
                "posterior_after": updated,
                "source_id": record.source_id,
            })
            updates.append({
                "candidate_id": candidate.candidate_id,
                "feedback_id": record.feedback_id,
                "prior_odds": prior_odds,
                "likelihood_ratio": likelihood_ratio,
                "posterior_odds": posterior_odds,
                "posterior_probability": updated,
            })
    grouped: dict[str, list[DetailCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.feature_type, []).append(candidate)
    for group in grouped.values():
        total = sum(item.posterior for item in group)
        if total > 0.0:
            for item in group:
                item.posterior /= total
    return {
        "updates": updates,
        "ignored_feedback": ignored,
        "independent_feedback_groups_seen": list(state.feedback_groups_seen),
        "previous_versions_preserved": True,
        "schema_updates_performed": False,
    }


def update_general_schema_knowledge(
    request: Point20Request,
    state: Point20State,
    config: Point20Config,
) -> dict[str, Any]:
    """Update general schema feature counts only from verified, reliable, independent feedback."""
    updates: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    used_groups: set[str] = set()
    for record in sorted(request.feedback, key=lambda item: (item.timestamp, item.feedback_id)):
        group = record.independence_group or record.source_id
        if not record.use_for_schema_update:
            continue
        if group in state.schema_feedback_groups_seen or group in used_groups:
            ignored.append({"feedback_id": record.feedback_id, "reason": "duplicate_or_dependent_feedback"})
            continue
        if not record.verified or record.reliability < config.min_feedback_reliability:
            ignored.append({"feedback_id": record.feedback_id, "reason": "not_verified_or_unreliable"})
            continue
        if not record.schema_id:
            ignored.append({"feedback_id": record.feedback_id, "reason": "schema_id_missing"})
            continue
        key = "|".join((record.schema_id, record.feature_type, value_key(record.value)))
        counts = state.schema_beta_counts.setdefault(key, {"alpha": 1.0, "beta": 1.0})
        before = dict(counts)
        increment = config.schema_learning_rate * record.reliability
        if record.supports:
            counts["alpha"] += increment
        else:
            counts["beta"] += increment
        used_groups.add(group)
        state.schema_feedback_groups_seen.append(group)
        updates.append({
            "schema_id": record.schema_id,
            "feature_type": record.feature_type,
            "value": record.value,
            "counts_before": before,
            "counts_after": dict(counts),
            "posterior_mean": counts["alpha"] / (counts["alpha"] + counts["beta"]),
            "feedback_id": record.feedback_id,
            "independence_group": group,
        })
    return {
        "updates": updates,
        "ignored_feedback": ignored,
        "schema_parameter_count": len(state.schema_beta_counts),
        "event_record_overwritten": False,
        "learning_source": "verified_feedback_only",
    }


def finalize_reconstruction(
    request: Point20Request,
    ledger: Mapping[str, Any],
    candidates: Sequence[DetailCandidate],
    policy: Mapping[str, Any],
    provenance_graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble evidence classes, inferred alternatives, uncertainty, confidence, and source lineage."""
    supported: list[dict[str, Any]] = []
    for fragment in request.episodic_fragments:
        if fragment.status in {"direct_evidence", "episodically_retrieved"}:
            supported.append({
                "detail_id": fragment.detail_id,
                "feature_type": fragment.feature_type,
                "content": fragment.content,
                "status": fragment.status,
                "confidence": clamp(fragment.retrieval_strength * fragment.episodic_context),
                "source_type": fragment.source_type,
                "source_id": fragment.source_id,
                "episode_id": fragment.episode_id,
            })
    inferred = [item.to_dict() for item in candidates if item.status == "inferred"]
    source_uncertain = [item.to_dict() for item in candidates
                        if item.status == "source_uncertain" or item.assigned_source == "unknown"]
    contradicted = [
        item.to_dict() for item in candidates
        if item.status == "contradicted" or item.contradiction >= 0.75
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        grouped.setdefault(item.feature_type, []).append({
            "candidate_id": item.candidate_id,
            "content": item.content,
            "posterior": item.posterior,
            "confidence": item.confidence,
            "status": item.status,
            "assigned_source": item.assigned_source,
            "contradiction": item.contradiction,
        })
    unresolved = [
        {"feature_type": feature, "status": row["status"], "likely_encoded": row["likely_encoded"],
         "centrality": row["centrality"], "unavailable_not_absent": row["unavailable_not_absent"]}
        for feature, row in ledger["features"].items()
        if row["status"] in {"not_retrieved", "weakly_retrieved", "contradicted"}
    ]
    selected_decisions = [dict(item) for item in policy["decisions"]]
    gist = request.context.get("gist")
    if gist is None:
        gist = " ".join(str(item["content"]) for item in supported)
    return {
        "event_id": request.event_id,
        "gist": gist,
        "supported_details": supported,
        "inferred_details": inferred,
        "source_uncertain_details": source_uncertain,
        "contradicted_details": contradicted,
        "unresolved_features": unresolved,
        "unresolved_alternatives": grouped,
        "response_decisions": selected_decisions,
        "confidence_by_detail": {
            item.candidate_id: {"confidence": item.confidence, "posterior": item.posterior,
                                "status": item.status, "source": item.assigned_source}
            for item in candidates
        },
        "provenance_graph": provenance_graph,
        "coherent_does_not_mean_historically_accurate": True,
        "not_retrieved_does_not_mean_absent": True,
    }


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int | bool]) -> float:
    """Compute mean squared probability error for externally labelled calibration examples."""
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have equal lengths")
    if not probabilities:
        return 0.0
    errors = []
    for probability, outcome in zip(probabilities, outcomes):
        p = clamp(probability)
        if outcome not in (0, 1, False, True):
            raise ValueError("outcomes must be binary")
        errors.append((p - int(outcome)) ** 2)
    return sum(errors) / len(errors)
