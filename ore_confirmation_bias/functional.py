"""Mathematical and functional operations for Point 21 confirmation-bias modelling."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contracts import (
    EvidenceItem, EvidenceSource, Point21Config, Point21Request, Point21State,
)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Return a finite scalar clipped to the supplied interval."""
    number = float(value)
    if not math.isfinite(number):
        return lower
    return max(lower, min(upper, number))


def softmax(values: Mapping[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Compute a numerically stable softmax distribution."""
    if not values:
        return {}
    tau = max(float(temperature), 1e-8)
    peak = max(float(value) for value in values.values())
    exp_values = {key: math.exp(max(-700.0, (float(value) - peak) / tau))
                  for key, value in values.items()}
    total = sum(exp_values.values())
    return {key: value / total for key, value in exp_values.items()} if total else {
        key: 1.0 / len(values) for key in values
    }


def entropy(distribution: Mapping[str, float]) -> float:
    """Calculate Shannon entropy in nats after normalizing non-negative weights."""
    total = sum(max(0.0, float(value)) for value in distribution.values())
    if total <= 0.0:
        return 0.0
    return -sum(
        (max(0.0, float(value)) / total) * math.log(max(1e-12, max(0.0, float(value)) / total))
        for value in distribution.values() if value > 0.0
    )


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    """Normalize positive weights and use a uniform fallback for a degenerate vector."""
    clipped = {key: max(0.0, float(value)) for key, value in values.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        return {key: 1.0 / len(clipped) for key in clipped} if clipped else {}
    return {key: value / total for key, value in clipped.items()}


def _hypothesis_ids(request: Point21Request) -> list[str]:
    """Return the request's competing hypothesis IDs in their declared order."""
    return [item.hypothesis_id for item in request.hypotheses]


def _focus_id(request: Point21Request, beliefs: Mapping[str, float]) -> str:
    """Return the explicit focal hypothesis or the currently most probable hypothesis."""
    return request.focus_hypothesis_id or max(beliefs, key=beliefs.get)


def _likelihood_vector(item: EvidenceItem, hypothesis_ids: Sequence[str]) -> dict[str, float]:
    """Return likelihoods for all hypotheses, treating an omitted likelihood as neutral."""
    return {key: max(1e-9, float(item.likelihoods.get(key, 1.0))) for key in hypothesis_ids}


def _posterior_for_item(
    prior: Mapping[str, float],
    item: EvidenceItem,
    hypothesis_ids: Sequence[str],
) -> dict[str, float]:
    """Combine one item's likelihood vector with a prior for a normalized interpretation."""
    log_scores = {
        key: math.log(max(1e-12, float(prior.get(key, 1e-12))))
             + math.log(_likelihood_vector(item, hypothesis_ids)[key])
        for key in hypothesis_ids
    }
    return softmax(log_scores)


def _item_congruence(
    item: EvidenceItem,
    focus_id: str,
    hypothesis_ids: Sequence[str],
) -> float:
    """Estimate the observation's relative compatibility with the focal hypothesis."""
    likelihoods = _likelihood_vector(item, hypothesis_ids)
    return _normalize(likelihoods).get(focus_id, 0.5)


def _source_congruence(
    source: EvidenceSource,
    evidence: Sequence[EvidenceItem],
    focus_id: str,
    hypothesis_ids: Sequence[str],
) -> float:
    """Estimate source congruence from its linked observations, without altering source quality."""
    linked = [item for item in evidence if item.source_id == source.source_id]
    if not linked:
        return 0.5
    return sum(_item_congruence(item, focus_id, hypothesis_ids) for item in linked) / len(linked)


def initialize_belief_state(
    request: Point21Request,
    state: Point21State,
) -> dict[str, Any]:
    """Retrieve the current claim belief if present, otherwise initialize from declared priors."""
    prior = _normalize({
        item.hypothesis_id: item.prior_probability for item in request.hypotheses
    })
    carried = state.beliefs_by_claim.get(request.claim_id)
    if carried:
        prior = _normalize({key: carried.get(key, prior.get(key, 0.0)) for key in prior})
        origin = "previous_model_belief"
    else:
        origin = "request_prior"
    return {
        "claim_id": request.claim_id,
        "beliefs": prior,
        "log_beliefs": {key: math.log(max(1e-12, value)) for key, value in prior.items()},
        "origin": origin,
        "focus_hypothesis_id": _focus_id(request, prior),
        "uncertainty_nats": entropy(prior),
    }


def build_expectation_template(
    request: Point21Request,
    beliefs: Mapping[str, float],
) -> dict[str, Any]:
    """Build parallel predictions for each hypothesis and identify discriminative features."""
    focus_id = _focus_id(request, beliefs)
    feature_names = sorted({
        feature for item in request.hypotheses for feature in item.predicted_features
    })
    rows: list[dict[str, Any]] = []
    for feature in feature_names:
        predictions = {
            item.hypothesis_id: float(item.predicted_features.get(feature, 0.5))
            for item in request.hypotheses
        }
        alternative_ids = [key for key in predictions if key != focus_id]
        alternative_mean = (
            sum(predictions[key] for key in alternative_ids) / len(alternative_ids)
            if alternative_ids else 0.5
        )
        rows.append({
            "feature": feature,
            "predictions": predictions,
            "focal_prediction": predictions.get(focus_id, 0.5),
            "alternative_mean": alternative_mean,
            "discriminability": predictions.get(focus_id, 0.5) - alternative_mean,
        })
    return {
        "focus_hypothesis_id": focus_id,
        "expectations": rows,
        "prediction_labels": {item.hypothesis_id: item.statement for item in request.hypotheses},
        "separation_safeguard": "Predicted features are stored separately from observed evidence.",
    }


def select_information_sources(
    request: Point21Request,
    beliefs: Mapping[str, float],
    config: Point21Config,
) -> dict[str, Any]:
    """Rank and select sources using diagnosticity, reliability, access, trust, and challenge cost."""
    ids = _hypothesis_ids(request)
    focus_id = _focus_id(request, beliefs)
    congruence_gain = 0.0 if config.accuracy_first else config.selection_congruence_bias
    challenge_gain = 0.0 if config.accuracy_first else config.selection_challenge_cost
    rows: list[dict[str, Any]] = []
    utilities: dict[str, float] = {}
    for source in request.sources:
        congruence = _source_congruence(source, request.evidence, focus_id, ids)
        linked = [item for item in request.evidence if item.source_id == source.source_id]
        diagnostic = max(
            source.diagnostic_value,
            max((item.uncertainty_reduction for item in linked), default=0.0),
        )
        # Congruence can influence source selection in the biased mode, but never changes
        # the independent reliability estimate.
        utility = (
            0.45 * diagnostic * source.reliability
            + 0.20 * source.accessibility
            + 0.10 * source.social_trust
            + congruence_gain * (2.0 * congruence - 1.0)
            - challenge_gain * source.challenge_cost
        )
        utilities[source.source_id] = utility
        rows.append({
            "source_id": source.source_id,
            "source_type": source.source_type,
            "utility": utility,
            "diagnostic_value": diagnostic,
            "reliability": source.reliability,
            "accessibility": source.accessibility,
            "social_trust": source.social_trust,
            "challenge_cost": source.challenge_cost,
            "belief_congruence": congruence,
            "independence_group": source.independence_group or source.source_id,
        })
    probabilities = softmax(utilities, config.selection_temperature)
    for row in rows:
        row["selection_probability"] = probabilities.get(row["source_id"], 0.0)
    rows.sort(key=lambda row: row["selection_probability"], reverse=True)
    selected = [row["source_id"] for row in rows[:config.max_sources]]
    return {
        "ranked_sources": rows,
        "selected_source_ids": selected,
        "not_yet_sampled_source_ids": [row["source_id"] for row in rows if row["source_id"] not in selected],
        "selection_congruence_bias_enabled": not config.accuracy_first,
        "focus_hypothesis_id": focus_id,
    }


def allocate_attention(
    request: Point21Request,
    selected_source_ids: Sequence[str],
    beliefs: Mapping[str, float],
    config: Point21Config,
) -> dict[str, Any]:
    """Allocate relative attention using task relevance, salience, novelty, uncertainty, and congruence."""
    ids = _hypothesis_ids(request)
    focus_id = _focus_id(request, beliefs)
    selected = set(selected_source_ids)
    candidates = [item for item in request.evidence if item.source_id in selected]
    gains: dict[str, float] = {}
    details: dict[str, dict[str, Any]] = {}
    congruence_gain = 0.0 if config.accuracy_first else config.attention_congruence_bias
    for item in candidates:
        congruence = _item_congruence(item, focus_id, ids)
        gain = (
            0.10 + 0.28 * item.relevance + 0.22 * item.salience
            + 0.22 * item.uncertainty_reduction + 0.12 * item.novelty
            + congruence_gain * (2.0 * congruence - 1.0)
        )
        gains[item.evidence_id] = gain
        details[item.evidence_id] = {
            "evidence_id": item.evidence_id,
            "source_id": item.source_id,
            "belief_congruence": congruence,
            "raw_gain": gain,
            "relevance": item.relevance,
            "salience": item.salience,
            "novelty": item.novelty,
            "uncertainty_reduction": item.uncertainty_reduction,
        }
    allocation = softmax(gains, 0.55)
    rows = []
    for item in candidates:
        row = details[item.evidence_id]
        row["attention_probability"] = allocation.get(item.evidence_id, 0.0)
        # This relative processing-strength estimate is separate from whether the item is true.
        row["attention_strength"] = clamp(
            0.30 + 0.70 * (row["attention_probability"] * max(1, len(candidates)))
        )
        rows.append(row)
    return {
        "attended_evidence": rows,
        "unselected_evidence_ids": [
            item.evidence_id for item in request.evidence if item.source_id not in selected
        ],
        "attention_congruence_bias_enabled": not config.accuracy_first,
    }


def encode_evidence_with_provenance(
    request: Point21Request,
    attention: Mapping[str, Any],
    state: Point21State,
    beliefs: Mapping[str, float],
) -> dict[str, Any]:
    """Store original evidence content with source, context, direction, and encoding strength."""
    ids = _hypothesis_ids(request)
    attention_by_id = {
        row["evidence_id"]: row for row in attention.get("attended_evidence", [])
    }
    encoded: list[dict[str, Any]] = []
    source_lookup = {source.source_id: source for source in request.sources}
    for item in request.evidence:
        source = source_lookup[item.source_id]
        attention_row = attention_by_id.get(item.evidence_id)
        likelihoods = _likelihood_vector(item, ids)
        direction = max(likelihoods, key=likelihoods.get)
        congruence = _item_congruence(
            item, _focus_id(request, beliefs), ids,
        )
        strength = (
            0.0 if attention_row is None else
            clamp(0.25 + 0.75 * attention_row["attention_strength"])
        )
        trace_id = f"{request.claim_id}:{item.evidence_id}"
        trace = {
            "trace_id": trace_id,
            "claim_id": request.claim_id,
            "evidence_id": item.evidence_id,
            "content": item.content,
            "feature": item.feature,
            "source_id": source.source_id,
            "source_type": source.source_type,
            "independence_group": source.independence_group or source.source_id,
            "context": dict(item.context),
            "timestamp": item.timestamp,
            "direct_observation": item.direct_observation,
            "likelihoods": likelihoods,
            "evidential_direction": direction,
            "belief_congruence_at_encoding": congruence,
            "attention_strength": strength,
            "measurement_quality": item.measurement_quality,
            "status": "encoded" if attention_row is not None else "not_selected_for_deep_processing",
        }
        if attention_row is not None:
            state.memory_traces[trace_id] = dict(trace)
        encoded.append(trace)
    return {
        "encoded_count": sum(row["status"] == "encoded" for row in encoded),
        "not_deeply_processed_count": sum(row["status"] != "encoded" for row in encoded),
        "traces": encoded,
        "provenance_preserved": True,
    }


def interpret_ambiguous_evidence(
    request: Point21Request,
    encoded: Mapping[str, Any],
    beliefs: Mapping[str, float],
    config: Point21Config,
) -> dict[str, Any]:
    """Rank each observation's hypothesis interpretations and retain an alternative reading."""
    ids = _hypothesis_ids(request)
    rho = config.interpretation_prior_bias
    items = {item.evidence_id: item for item in request.evidence}
    rows: list[dict[str, Any]] = []
    for trace in encoded.get("traces", []):
        item = items[trace["evidence_id"]]
        likelihoods = _likelihood_vector(item, ids)
        scores = {
            key: rho * math.log(max(1e-12, beliefs.get(key, 1e-12)))
                 + math.log(likelihoods[key])
            for key in ids
        }
        interpretation = softmax(scores, config.interpretation_temperature)
        ranked = sorted(interpretation.items(), key=lambda pair: pair[1], reverse=True)
        likelihood_only = _normalize(likelihoods)
        rows.append({
            "evidence_id": item.evidence_id,
            "interpretation_distribution": interpretation,
            "likelihood_only_distribution": likelihood_only,
            "ranked_interpretations": [
                {"hypothesis_id": key, "probability": value} for key, value in ranked
            ],
            "alternative_interpretation": ranked[1][0] if len(ranked) > 1 else None,
            "prior_influence_parameter": rho,
            "prior_and_observation_kept_separate": True,
        })
    return {"interpretations": rows}


def assess_evidence_quality(
    request: Point21Request,
    attention: Mapping[str, Any],
    selected: Mapping[str, Any],
    config: Point21Config,
) -> dict[str, Any]:
    """Calculate reliability-adjusted evidence weights with a correction for shared provenance."""
    source_lookup = {source.source_id: source for source in request.sources}
    attention_lookup = {row["evidence_id"]: row for row in attention.get("attended_evidence", [])}
    selected_ids = set(selected.get("selected_source_ids", []))
    group_counts: dict[str, int] = {}
    for item in request.evidence:
        if item.source_id in selected_ids:
            source = source_lookup[item.source_id]
            group = source.independence_group or source.source_id
            group_counts[group] = group_counts.get(group, 0) + 1
    rows: list[dict[str, Any]] = []
    for item in request.evidence:
        source = source_lookup[item.source_id]
        attention_row = attention_lookup.get(item.evidence_id)
        group = source.independence_group or source.source_id
        dependence = 1.0 / math.sqrt(max(1, group_counts.get(group, 1)))
        attended = attention_row is not None
        attention_strength = attention_row["attention_strength"] if attended else 0.0
        diagnosticity = source.diagnostic_value
        log_values = [math.log(value) for value in _likelihood_vector(item, _hypothesis_ids(request)).values()]
        discrimination = clamp(
            max(log_values) - min(log_values), 0.0, 4.0
        ) / 4.0
        diagnosticity = max(diagnosticity, discrimination)
        weight = (
            source.reliability * diagnosticity * item.measurement_quality
            * attention_strength * dependence * config.evidence_weight_scale
        )
        rows.append({
            "evidence_id": item.evidence_id,
            "source_id": source.source_id,
            "source_reliability": source.reliability,
            "diagnosticity": diagnosticity,
            "measurement_quality": item.measurement_quality,
            "attention_strength": attention_strength,
            "independence_group": group,
            "group_item_count": group_counts.get(group, 0),
            "dependence_correction": dependence,
            "weight": clamp(weight, 0.0, 2.0),
            "selected_for_evaluation": attended,
        })
    return {
        "weighted_evidence": rows,
        "group_counts": group_counts,
        "reliability_independent_of_agreement": True,
        "shared_sources_downweighted": True,
    }


def compare_evidence_symmetrically(
    request: Point21Request,
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    """Express each observation as centered log likelihood evidence for every hypothesis."""
    ids = _hypothesis_ids(request)
    quality_lookup = {row["evidence_id"]: row for row in quality.get("weighted_evidence", [])}
    rows: list[dict[str, Any]] = []
    for item in request.evidence:
        likelihoods = _likelihood_vector(item, ids)
        logs = {key: math.log(value) for key, value in likelihoods.items()}
        average = sum(logs.values()) / len(logs)
        centered = {key: value - average for key, value in logs.items()}
        rows.append({
            "evidence_id": item.evidence_id,
            "log_likelihoods": logs,
            "centered_log_evidence": centered,
            "quality_weight": quality_lookup[item.evidence_id]["weight"],
            "evidence_direction": max(logs, key=logs.get),
            "symmetric_scale": "The same likelihood transformation is applied to all hypotheses.",
        })
    return {"comparisons": rows}


def readout_evidence_for_judgment(
    request: Point21Request,
    starting_beliefs: Mapping[str, float],
    comparison: Mapping[str, Any],
    config: Point21Config,
) -> dict[str, Any]:
    """Accumulate quality-weighted evidence while optionally exposing congruence-biased readout."""
    ids = _hypothesis_ids(request)
    focus_id = _focus_id(request, starting_beliefs)
    item_lookup = {item.evidence_id: item for item in request.evidence}
    log_scores = {key: math.log(max(1e-12, starting_beliefs.get(key, 1e-12))) for key in ids}
    rows: list[dict[str, Any]] = []
    for row in comparison.get("comparisons", []):
        item = item_lookup[row["evidence_id"]]
        congruence = _item_congruence(item, focus_id, ids)
        gate = 1.0 if config.accuracy_first else clamp(
            1.0 + config.readout_congruence_bias * (2.0 * congruence - 1.0),
            0.10, 2.0,
        )
        effective_weight = row["quality_weight"] * gate
        contributions = {
            key: effective_weight * value
            for key, value in row["centered_log_evidence"].items()
        }
        for key, value in contributions.items():
            log_scores[key] += value
        rows.append({
            "evidence_id": item.evidence_id,
            "belief_congruence": congruence,
            "readout_gate": gate,
            "quality_weight": row["quality_weight"],
            "effective_weight": effective_weight,
            "log_evidence_contribution": contributions,
        })
    posterior = softmax(log_scores)
    return {
        "log_scores": log_scores,
        "evidence_posterior": posterior,
        "item_contributions": rows,
        "readout_congruence_bias_enabled": not config.accuracy_first,
    }


def update_belief_confidence(
    request: Point21Request,
    prior: Mapping[str, float],
    posterior: Mapping[str, float],
    state: Point21State,
    config: Point21Config,
) -> dict[str, Any]:
    """Apply a bounded update rate and preserve the complete before/target/after distributions."""
    rate = config.update_rate
    updated = _normalize({
        key: (1.0 - rate) * prior.get(key, 0.0) + rate * posterior.get(key, 0.0)
        for key in posterior
    })
    focus_id = _focus_id(request, prior)
    new_focus = max(updated, key=updated.get)
    prior_focus = prior.get(focus_id, 0.0)
    posterior_focus = posterior.get(focus_id, 0.0)
    perseverance = (
        prior_focus >= 0.70 and posterior_focus < prior_focus - 0.15
        and updated.get(focus_id, 0.0) > posterior_focus + 0.05
    )
    state.beliefs_by_claim[request.claim_id] = dict(updated)
    return {
        "prior_beliefs": dict(prior),
        "evidence_supported_target": dict(posterior),
        "updated_beliefs": updated,
        "update_rate": rate,
        "confidence": max(updated.values(), default=0.0),
        "uncertainty_nats": entropy(updated),
        "previous_focal_hypothesis": focus_id,
        "new_most_probable_hypothesis": new_focus,
        "belief_changed": new_focus != focus_id,
        "possible_belief_perseverance": perseverance,
        "confidence_is_not_equivalent_to_accuracy": True,
    }


def retrieve_belief_related_memories(
    request: Point21Request,
    state: Point21State,
    beliefs: Mapping[str, float],
    config: Point21Config,
) -> dict[str, Any]:
    """Retrieve encoded observations using cues while leaving original memory content untouched."""
    ids = _hypothesis_ids(request)
    focus_id = _focus_id(request, beliefs)
    cue_features = set(request.context.get("memory_cues", []))
    memories = [
        trace for trace in state.memory_traces.values()
        if trace.get("claim_id") == request.claim_id
    ]
    rows: list[dict[str, Any]] = []
    for trace in memories:
        likelihoods = _normalize({
            key: float(trace.get("likelihoods", {}).get(key, 1.0)) for key in ids
        })
        congruence = likelihoods.get(focus_id, 0.5)
        cue_match = 1.0 if not cue_features or trace.get("feature") in cue_features else 0.65
        base_accessibility = float(trace.get("attention_strength", 0.0)) * cue_match
        gate = 1.0 if config.accuracy_first else clamp(
            1.0 + config.memory_congruence_bias * (2.0 * congruence - 1.0),
            0.10, 2.0,
        )
        rows.append({
            "trace_id": trace["trace_id"],
            "evidence_id": trace["evidence_id"],
            "content": trace["content"],
            "source_id": trace["source_id"],
            "source_type": trace["source_type"],
            "base_accessibility": base_accessibility,
            "current_belief_congruence": congruence,
            "retrieval_gate": gate,
            "retrieval_score": base_accessibility * gate,
            "original_content_preserved": True,
        })
    rows.sort(key=lambda row: row["retrieval_score"], reverse=True)
    return {
        "retrieved_memories": rows,
        "memory_count": len(rows),
        "memory_congruence_bias_enabled": not config.accuracy_first,
        "source_provenance_preserved": all("source_id" in row for row in rows),
    }


def detect_confirmation_feedback(
    selection: Mapping[str, Any],
    attention: Mapping[str, Any],
    interpretations: Mapping[str, Any],
    readout: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> dict[str, Any]:
    """Identify which stages show a possible reinforcing loop without collapsing them into one score."""
    selected_rows = [
        row for row in selection.get("ranked_sources", [])
        if row["source_id"] in set(selection.get("selected_source_ids", []))
    ]
    unselected_rows = [
        row for row in selection.get("ranked_sources", [])
        if row["source_id"] in set(selection.get("not_yet_sampled_source_ids", []))
    ]
    selected_congruence = (
        sum(row["belief_congruence"] for row in selected_rows) / len(selected_rows)
        if selected_rows else 0.5
    )
    unselected_congruence = (
        sum(row["belief_congruence"] for row in unselected_rows) / len(unselected_rows)
        if unselected_rows else 0.5
    )
    attended = attention.get("attended_evidence", [])
    high_congruent = [row for row in attended if row["belief_congruence"] >= 0.5]
    low_congruent = [row for row in attended if row["belief_congruence"] < 0.5]
    mean_high_attention = (
        sum(row["attention_strength"] for row in high_congruent) / len(high_congruent)
        if high_congruent else 0.0
    )
    mean_low_attention = (
        sum(row["attention_strength"] for row in low_congruent) / len(low_congruent)
        if low_congruent else 0.0
    )
    readout_rows = readout.get("item_contributions", [])
    mean_readout_gate = (
        sum(row["readout_gate"] for row in readout_rows) / len(readout_rows)
        if readout_rows else 1.0
    )
    interpretation_rows = interpretations.get("interpretations", [])
    prior_shift = (
        sum(
            max(row["interpretation_distribution"].values())
            - max(row["likelihood_only_distribution"].values())
            for row in interpretation_rows
        ) / len(interpretation_rows)
        if interpretation_rows else 0.0
    )
    memory_rows = memory.get("retrieved_memories", [])
    memory_gates = [row["retrieval_gate"] for row in memory_rows]
    stage_metrics = {
        "source_selection_congruence_gap": selected_congruence - unselected_congruence,
        "attentional_congruence_gap": mean_high_attention - mean_low_attention,
        "interpretation_prior_shift": prior_shift,
        "mean_readout_gate": mean_readout_gate,
        "mean_memory_gate": (
            sum(memory_gates) / len(memory_gates) if memory_gates else 1.0
        ),
    }
    return {
        "stage_metrics": stage_metrics,
        "reinforcing_stage_count": sum(
            value > 0.05 for key, value in stage_metrics.items()
            if key != "mean_readout_gate" and key != "mean_memory_gate"
        ) + sum(value > 1.05 for key, value in stage_metrics.items()
                if key in {"mean_readout_gate", "mean_memory_gate"}),
        "interpretation": "Stage-specific signals can reinforce one another; none alone proves irrationality.",
    }


def _source_information_gain(
    source: EvidenceSource,
    evidence: Sequence[EvidenceItem],
    beliefs: Mapping[str, float],
    hypothesis_ids: Sequence[str],
) -> float:
    """Estimate entropy reduction from the source's available candidate observations."""
    prior_entropy = entropy(beliefs)
    linked = [item for item in evidence if item.source_id == source.source_id]
    if not linked:
        return 0.0
    gains = []
    for item in linked:
        likelihoods = _likelihood_vector(item, hypothesis_ids)
        posterior = _normalize({
            key: beliefs.get(key, 0.0) * likelihoods[key] for key in hypothesis_ids
        })
        gains.append(max(0.0, prior_entropy - entropy(posterior)))
    return sum(gains) / len(gains)


def apply_corrective_control(
    request: Point21Request,
    beliefs: Mapping[str, float],
    config: Point21Config,
) -> dict[str, Any]:
    """Pre-register predictions, produce counter-hypothesis tests, and rank diagnostic next sources."""
    ids = _hypothesis_ids(request)
    focus_id = _focus_id(request, beliefs)
    source_rows: list[dict[str, Any]] = []
    for source in request.sources:
        gain = _source_information_gain(source, request.evidence, beliefs, ids)
        cost = 0.1 + source.challenge_cost + (1.0 - source.accessibility) * 0.25
        score = gain * source.reliability * source.diagnostic_value * source.accessibility / cost
        linked = [item for item in request.evidence if item.source_id == source.source_id]
        focus_likelihood = sum(
            math.log(_likelihood_vector(item, ids)[focus_id]) for item in linked
        ) / len(linked) if linked else 0.0
        alternative_likelihood = max((
            sum(math.log(_likelihood_vector(item, ids)[key]) for item in linked) / len(linked)
            for key in ids if key != focus_id
        ), default=0.0)
        source_rows.append({
            "source_id": source.source_id,
            "estimated_information_gain_nats": gain,
            "reliability": source.reliability,
            "estimated_cost": cost,
            "corrective_score": score,
            "challenges_focus_hypothesis": alternative_likelihood > focus_likelihood,
        })
    source_rows.sort(key=lambda row: row["corrective_score"], reverse=True)
    focus = next(item for item in request.hypotheses if item.hypothesis_id == focus_id)
    alternatives = [item for item in request.hypotheses if item.hypothesis_id != focus_id]
    return {
        "pre_registered_focal_predictions": dict(focus.predicted_features),
        "alternative_hypotheses": [
            {
                "hypothesis_id": item.hypothesis_id,
                "statement": item.statement,
                "predicted_features": dict(item.predicted_features),
                "prior_probability": beliefs.get(item.hypothesis_id, 0.0),
            }
            for item in alternatives
        ],
        "ranked_discriminating_sources": source_rows,
        "recommended_next_source_ids": [
            row["source_id"] for row in source_rows[:config.max_sources]
        ],
        "symmetry_checks": [
            "Use the same reliability and measurement standards for supportive and contradictory sources.",
            "State in advance which observation would count against the focal hypothesis.",
            "Check whether apparently separate sources share one underlying origin.",
            "Keep direct observations separate from interpretations and confidence judgments.",
        ],
        "expected_information_gain_is_approximate": True,
    }


def validate_point21_stages(
    request: Point21Request,
    selection: Mapping[str, Any],
    attention: Mapping[str, Any],
    interpretations: Mapping[str, Any],
    quality: Mapping[str, Any],
    readout: Mapping[str, Any],
    update: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> dict[str, Any]:
    """Report separate operational metrics and failure signatures for the modeled stages."""
    selected = set(selection.get("selected_source_ids", []))
    source_rows = selection.get("ranked_sources", [])
    selected_congruence = [
        row["belief_congruence"] for row in source_rows if row["source_id"] in selected
    ]
    all_congruence = [row["belief_congruence"] for row in source_rows]
    attention_rows = attention.get("attended_evidence", [])
    congruent_attention = [
        row["attention_strength"] for row in attention_rows
        if row["belief_congruence"] >= 0.5
    ]
    disconfirming_attention = [
        row["attention_strength"] for row in attention_rows
        if row["belief_congruence"] < 0.5
    ]
    readout_rows = readout.get("item_contributions", [])
    confirm_gates = [row["readout_gate"] for row in readout_rows if row["belief_congruence"] >= 0.5]
    disconfirm_gates = [row["readout_gate"] for row in readout_rows if row["belief_congruence"] < 0.5]
    memory_rows = memory.get("retrieved_memories", [])
    metrics = {
        "source_selection_bias": (
            sum(selected_congruence) / len(selected_congruence)
            - sum(all_congruence) / len(all_congruence)
            if selected_congruence and all_congruence else 0.0
        ),
        "attention_bias": (
            sum(congruent_attention) / len(congruent_attention)
            - sum(disconfirming_attention) / len(disconfirming_attention)
            if congruent_attention and disconfirming_attention else 0.0
        ),
        "interpretation_count": len(interpretations.get("interpretations", [])),
        "readout_bias": (
            sum(confirm_gates) / len(confirm_gates)
            - sum(disconfirm_gates) / len(disconfirm_gates)
            if confirm_gates and disconfirm_gates else 0.0
        ),
        "confidence": float(update.get("confidence", 0.0)),
        "memory_count": len(memory_rows),
        "source_quality_count": len(quality.get("weighted_evidence", [])),
    }
    failures = []
    if metrics["source_selection_bias"] > 0.10:
        failures.append("selection_failure: selected sources are more belief-congruent than the available pool")
    if metrics["attention_bias"] > 0.10:
        failures.append("attention_failure: congruent observations received more attention")
    if metrics["readout_bias"] > 0.10:
        failures.append("readout_failure: congruent observations received a larger decision gate")
    if update.get("possible_belief_perseverance"):
        failures.append("update_failure: confidence may be resisting substantial contradictory evidence")
    if any(not row.get("source_id") for row in memory_rows):
        failures.append("source_memory_failure: a retrieved trace lacks source attribution")
    return {
        "metrics": metrics,
        "failure_signatures": failures,
        "validation_scope": "Separate process metrics; not a personality score, diagnosis, or universal bias measure.",
        "interpretation": (
            "A zero metric can mean no observed contrast in this input, not proof that the stage is unbiased."
        ),
    }


def build_evidence_audit(
    request: Point21Request,
    selection: Mapping[str, Any],
    attention: Mapping[str, Any],
    quality: Mapping[str, Any],
    comparison: Mapping[str, Any],
    readout: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Join the original observations to the stages that selected, attended, weighted, and read them."""
    selected_sources = set(selection.get("selected_source_ids", []))
    attention_lookup = {
        row["evidence_id"]: row for row in attention.get("attended_evidence", [])
    }
    quality_lookup = {
        row["evidence_id"]: row for row in quality.get("weighted_evidence", [])
    }
    comparison_lookup = {
        row["evidence_id"]: row for row in comparison.get("comparisons", [])
    }
    readout_lookup = {
        row["evidence_id"]: row for row in readout.get("item_contributions", [])
    }
    audit = []
    for item in request.evidence:
        source = next(source for source in request.sources if source.source_id == item.source_id)
        audit.append({
            "evidence_id": item.evidence_id,
            "content": item.content,
            "feature": item.feature,
            "source_id": item.source_id,
            "source_type": source.source_type,
            "direct_observation": item.direct_observation,
            "selected_source": item.source_id in selected_sources,
            "attention": attention_lookup.get(item.evidence_id),
            "quality": quality_lookup.get(item.evidence_id),
            "symmetric_comparison": comparison_lookup.get(item.evidence_id),
            "readout": readout_lookup.get(item.evidence_id),
            "source_provenance_preserved": True,
        })
    return audit
