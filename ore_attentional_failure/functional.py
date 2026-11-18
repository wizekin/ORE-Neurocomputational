"""Mathematical and functional operations for Point 18 attentional-failure handling."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contracts import (
    InformationRequirement, AttentionalConfig, AttentionalExperience, AttentionalState,
)


def clip01(value: float) -> float:
    """Clamp a finite numerical value to the unit interval."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("model values must be finite")
    return max(0.0, min(1.0, number))


def sigmoid(value: float) -> float:
    """Compute a numerically stable logistic activation."""
    number = float(value)
    if number >= 0:
        z = math.exp(-min(number, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(number, -700.0))
    return z / (1.0 + z)


def softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Normalize scores into a stable graded-selection distribution."""
    if not values:
        return []
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    peak = max(float(x) for x in values)
    exponents = [math.exp(max(-700.0, (float(x) - peak) / temperature)) for x in values]
    total = sum(exponents)
    return [x / total for x in exponents]


def cosine_similarity(left: Mapping[str, float], right: Mapping[str, float], epsilon: float = 1e-8) -> float:
    """Compute cosine similarity for sparse feature maps, returning zero for empty vectors."""
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    dot = sum(float(left.get(k, 0.0)) * float(right.get(k, 0.0)) for k in keys)
    left_norm = math.sqrt(sum(float(left.get(k, 0.0)) ** 2 for k in keys))
    right_norm = math.sqrt(sum(float(right.get(k, 0.0)) ** 2 for k in keys))
    return clip01((dot / (left_norm * right_norm + epsilon) + 1.0) / 2.0)


def establish_task_requirements(event: AttentionalExperience) -> dict[str, Any]:
    """Normalize task requirements and expose their weighted goal representation."""
    requirements = [x.to_dict() for x in event.required_information]
    total = sum(float(x["importance"]) for x in requirements)
    normalized = {
        x["key"]: (float(x["importance"]) / total if total > 0 else 0.0)
        for x in requirements
    }
    return {
        "task_goal": event.task_goal,
        "requirements": requirements,
        "importance_weights": normalized,
        "requirement_count": len(requirements),
        "goal_vector": dict(event.goal_vector),
        "goal_is_explicit": True,
    }


def estimate_attentional_state(
    event: AttentionalExperience,
    state: AttentionalState,
    config: AttentionalConfig,
) -> dict[str, Any]:
    """Combine indirect performance and state indicators without inferring a diagnosis."""
    metrics = event.metrics
    alertness = clip01(metrics.get("alertness", state.alertness))
    fatigue = clip01(metrics.get("fatigue", state.fatigue))
    distraction = clip01(metrics.get("distraction_load", state.distraction))
    response_slowing = clip01(metrics.get("response_slowing", 0.0))
    omission_rate = clip01(metrics.get("omission_rate", 0.0))
    error_rate = clip01(metrics.get("error_rate", 0.0))
    variability = clip01(metrics.get("response_time_variability", 0.0))
    disengagement = clip01(metrics.get("disengagement_signal", 0.0))
    lapse_score = (
        0.25 * response_slowing + 0.25 * omission_rate + 0.15 * error_rate
        + 0.15 * fatigue + 0.10 * distraction + 0.10 * variability
    )
    lapse_probability = sigmoid(5.0 * (lapse_score - 0.45))
    availability = clip01(
        alertness * (1.0 - config.fatigue_effect * fatigue)
        * (1.0 - config.distraction_effect * distraction)
    )
    return {
        "alertness": alertness,
        "fatigue": fatigue,
        "distraction": distraction,
        "attentional_availability": availability,
        "lapse_probability": lapse_probability,
        "indicators": {
            "response_slowing": response_slowing,
            "omission_rate": omission_rate,
            "error_rate": error_rate,
            "response_time_variability": variability,
            "disengagement_signal": disengagement,
        },
        "indicator_count": 5,
        "estimates_are_indirect_and_task_sensitive": True,
        "medical_or_fixed_trait_diagnosis_inferred": False,
    }


def assess_competing_inputs(event: AttentionalExperience) -> dict[str, Any]:
    """Classify task-relevant, irrelevant, internal, external, and safety-significant competitors."""
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(event.competing_inputs):
        item = dict(raw)
        kind = str(item.get("kind", "external")).lower()
        relevance = clip01(item.get("goal_relevance", 0.0))
        urgency = clip01(item.get("urgency", 0.0))
        safety = bool(item.get("safety_critical", False))
        salience = clip01(item.get("salience", 0.5))
        rows.append({
            **item,
            "key": str(item.get("key", f"competitor-{index}")),
            "kind": kind if kind in {"external", "internal"} else "external",
            "goal_relevance": relevance,
            "urgency": urgency,
            "salience": salience,
            "safety_critical": safety,
            "classification": (
                "urgent_safety_signal" if safety or urgency >= 0.95
                else "task_relevant" if relevance >= 0.5
                else "potential_distraction"
            ),
            "irrelevance": 1.0 - relevance,
        })
    distraction_load = clip01(sum(x["salience"] * x["irrelevance"] for x in rows))
    return {
        "competitors": rows,
        "external_count": sum(x["kind"] == "external" for x in rows),
        "internal_count": sum(x["kind"] == "internal" for x in rows),
        "distraction_load": distraction_load,
        "safety_signals_preserved": True,
    }


def compute_priority_map(
    event: AttentionalExperience,
    requirements: Mapping[str, Any],
    state: AttentionalState,
    config: AttentionalConfig,
) -> dict[str, Any]:
    """Rank input items using relevance, urgency, salience, novelty, history, and cost."""
    weights = config
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(event.incoming_items):
        item = dict(raw)
        key = str(item.get("key", f"input-{index}"))
        relevance = clip01(item.get(
            "goal_relevance",
            requirements["importance_weights"].get(key, 0.0),
        ))
        urgency = clip01(item.get("urgency", 0.0))
        salience = clip01(item.get("salience", 0.5))
        novelty = clip01(item.get("novelty", 0.0))
        history = clip01(state.selection_history.get(key, item.get("selection_history", 0.0)))
        cost = clip01(item.get("processing_cost", 0.0))
        safety = bool(item.get("safety_critical", False)) or event.safety_critical and urgency >= 0.8
        score = (
            weights.relevance_weight * relevance + weights.urgency_weight * urgency
            + weights.salience_weight * salience + weights.novelty_weight * novelty
            + weights.history_weight * history - weights.cost_weight * cost
        )
        if safety:
            score += 2.0
        rows.append({
            **item, "key": key, "goal_relevance": relevance, "urgency": urgency,
            "salience": salience, "novelty": novelty, "selection_history": history,
            "processing_cost": cost, "safety_critical": safety, "priority_score": score,
        })
    rows.sort(key=lambda x: (x["safety_critical"], x["priority_score"]), reverse=True)
    probabilities = softmax([x["priority_score"] for x in rows], temperature=0.35)
    for row, probability in zip(rows, probabilities):
        row["selection_probability"] = probability
    return {
        "ranked_items": rows,
        "count": len(rows),
        "formula": "wR*relevance + wU*urgency + wS*salience + wN*novelty + wH*history - wC*cost",
        "safety_override_applied": any(x["safety_critical"] for x in rows),
    }


def bias_sensory_representation(
    priority: Mapping[str, Any],
    minimum_gain: float = 0.5,
    maximum_gain: float = 1.5,
) -> dict[str, Any]:
    """Apply relevance-dependent feature gain while retaining low-priority source traces."""
    output: list[dict[str, Any]] = []
    for item in priority.get("ranked_items", ()):
        attention = clip01(item["selection_probability"] * len(priority.get("ranked_items", ())))
        gain = minimum_gain + (maximum_gain - minimum_gain) * attention
        features = item.get("features", {})
        if isinstance(features, Mapping):
            biased = {str(k): float(v) * gain for k, v in features.items()}
        else:
            biased = {"content": item.get("content"), "gain": gain}
        output.append({
            "key": item["key"], "content": item.get("content"),
            "gain": gain, "biased_features": biased,
            "priority": item["priority_score"],
            "raw_input_preserved": True,
            "source_id": item.get("source_id", "incoming"),
        })
    return {
        "items": output,
        "low_priority_traces_preserved": True,
        "gain_range": [minimum_gain, maximum_gain],
    }


def gate_working_memory(
    priority: Mapping[str, Any],
    state_estimate: Mapping[str, Any],
    config: AttentionalConfig,
) -> dict[str, Any]:
    """Select a utility-maximizing greedy subset under a task-sensitive cost budget."""
    budget = config.working_memory_budget * (
        0.5 + 0.5 * clip01(state_estimate["attentional_availability"])
    )
    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    spent = 0.0
    for raw in priority.get("ranked_items", ()):
        item = dict(raw)
        cost = max(0.05, float(item.get("processing_cost", 0.2)))
        if item["safety_critical"] or spent + cost <= budget:
            selected.append(item)
            spent += cost
        else:
            deferred.append({
                "key": item["key"], "source_id": item.get("source_id"),
                "content_available_for_review": "content" in item,
                "defer_reason": "working_memory_budget",
            })
    return {
        "selected_items": selected,
        "deferred_items": deferred,
        "budget": budget,
        "estimated_cost": spent,
        "budget_utilization": clip01(spent / max(config.epsilon, budget)),
        "safety_items_bypass_budget": True,
        "deferred_source_pointers_retained": True,
    }


def update_sustained_attention(
    state_estimate: Mapping[str, Any],
    state: AttentionalState,
    event: AttentionalExperience,
    config: AttentionalConfig,
) -> dict[str, Any]:
    """Update a leaky attentional state and estimate stability over a moving window."""
    engagement = clip01(event.metrics.get("engagement", state_estimate["attentional_availability"]))
    fatigue = state_estimate["fatigue"]
    distraction = state_estimate["distraction"]
    new_attention = clip01(
        config.attention_decay * state.stability
        + (1.0 - config.attention_decay) * engagement
        - 0.20 * fatigue - 0.25 * distraction
    )
    history = (state.attention_history + [new_attention])[-config.attention_window:]
    mean = sum(history) / len(history)
    variance = sum((x - mean) ** 2 for x in history) / len(history)
    return {
        "attention_level": new_attention,
        "history": history,
        "window_mean": mean,
        "window_variance": variance,
        "stability": clip01(mean * (1.0 - min(1.0, variance * 4.0))),
        "sustained_decline": mean < 0.35 or variance > 0.08,
        "pacing_adjustment_recommended": mean < 0.45 or variance > 0.08,
    }


def detect_attention_lapse(
    event: AttentionalExperience,
    state_estimate: Mapping[str, Any],
    sustained: Mapping[str, Any],
    state: AttentionalState,
    config: AttentionalConfig,
) -> dict[str, Any]:
    """Use converging indicators and temporal hysteresis to mark a possible lapse interval."""
    indicators = state_estimate["indicators"]
    score = (
        0.30 * indicators["response_slowing"]
        + 0.25 * indicators["omission_rate"]
        + 0.20 * indicators["disengagement_signal"]
        + 0.15 * state_estimate["distraction"]
        + 0.10 * state_estimate["fatigue"]
    )
    safety_trigger = event.safety_critical and (
        indicators["omission_rate"] > 0.0 or indicators["error_rate"] > 0.0
    )
    recent_scores = [
        float(row.get("score", 0.0)) for row in state.lapse_history[-config.lapse_hysteresis_samples + 1:]
    ]
    consecutive = len(recent_scores) >= config.lapse_hysteresis_samples - 1 and all(
        value >= config.lapse_threshold for value in [*recent_scores, score]
    )
    detected = (score >= config.lapse_threshold and consecutive) or safety_trigger
    confidence = clip01(0.65 * score + 0.35 * state_estimate["lapse_probability"])
    return {
        "detected": detected,
        "score": score,
        "confidence": confidence,
        "start_time": event.metrics.get("lapse_start_time", event.metrics.get("time")),
        "end_time": event.metrics.get("lapse_end_time"),
        "affected_interval": event.metrics.get("affected_interval"),
        "safety_override": safety_trigger,
        "hysteresis_confirmed": consecutive,
        "mark_interval_uncertain": detected,
        "state_variability_considered": sustained["window_variance"],
    }


def reconstruct_missing_information(
    event: AttentionalExperience,
    gate: Mapping[str, Any],
    lapse: Mapping[str, Any],
    config: AttentionalConfig,
) -> dict[str, Any]:
    """Compare required content with encoded keys and recover only from supplied source records."""
    selected = {str(x["key"]): x for x in gate["selected_items"]}
    deferred = {str(x["key"]): x for x in gate["deferred_items"]}
    sources = {str(x.get("key")): dict(x) for x in event.source_items if x.get("key") is not None}
    missing: list[dict[str, Any]] = []
    weak: list[dict[str, Any]] = []
    recovered_candidates: list[dict[str, Any]] = []
    for requirement in event.required_information:
        key = requirement.key
        item = selected.get(key)
        if item:
            confidence = clip01(
                float(item.get("signal_quality", 1.0))
                * float(item.get("integration_support", 1.0))
                * float(event.metrics.get("encoding_factor", 1.0))
            )
            record = {
                "key": key, "content": item.get("content"),
                "confidence": confidence, "source_id": item.get("source_id", "incoming"),
                "provenance": "observed_input",
            }
            if confidence < config.requirement_threshold:
                weak.append(record)
                source = sources.get(key)
                if source is not None:
                    recovered_candidates.append({
                        "key": key, "content": source.get("content"),
                        "confidence": clip01(source.get("confidence", source.get("signal_quality", 0.8))),
                        "source_id": source.get("source_id", "verified_source"),
                        "provenance": "recoverable_source_record",
                        "recovery_evidence": "source_record",
                        "replaces_weak_encoding": True,
                    })
            else:
                recovered_candidates.append(record)
            continue
        source = sources.get(key)
        entry = {
            "key": key, "missingness": 1.0,
            "reason": "deferred_by_gate" if key in deferred else (
                "possible_attentional_lapse" if lapse["detected"] else "not_encountered_or_not_encoded"
            ),
            "source_record_available": source is not None,
        }
        missing.append(entry)
        if source is not None:
            recovered_candidates.append({
                "key": key, "content": source.get("content"),
                "confidence": clip01(source.get("confidence", source.get("signal_quality", 0.8))),
                "source_id": source.get("source_id", "verified_source"),
                "provenance": "recoverable_source_record",
                "recovery_evidence": "source_record",
            })
    return {
        "missing": missing,
        "weakly_encoded": weak,
        "recoverable_from_source": recovered_candidates,
        "unrecoverable_keys": [x["key"] for x in missing if not x["source_record_available"]],
        "schema_guessing_used_as_recovery": False,
        "lapse_detected": bool(lapse["detected"]),
    }


def audit_completeness(
    requirements: Sequence[InformationRequirement],
    representation: Mapping[str, Any],
    config: AttentionalConfig,
) -> dict[str, Any]:
    """Compute weighted coverage, conservative confidence, and dependency-aware missing gaps."""
    records = representation.get("items", {})
    rows: list[dict[str, Any]] = []
    for requirement in requirements:
        item = records.get(requirement.key)
        confidence = clip01(float(item.get("confidence", 0.0))) if item else 0.0
        dependencies_missing = [
            key for key in requirement.depends_on
            if key not in records or float(records[key].get("confidence", 0.0)) < config.requirement_threshold
        ]
        rows.append({
            "key": requirement.key, "importance": requirement.importance,
            "confidence": confidence, "critical": requirement.critical,
            "missing_dependencies": dependencies_missing,
            "adequately_encoded": confidence >= config.requirement_threshold and not dependencies_missing,
        })
    total = sum(x.importance for x in requirements)
    coverage = (
        sum(x["importance"] * x["confidence"] for x in rows) / total
        if total > 0 else 1.0
    )
    if total > 0:
        combined = math.exp(sum(
            (x["importance"] / total) * math.log(max(config.epsilon, x["confidence"]))
            for x in rows
        ))
    else:
        combined = 1.0
    gaps = [x for x in rows if not x["adequately_encoded"]]
    critical_gaps = [x["key"] for x in gaps if x["critical"]]
    return {
        "coverage": clip01(coverage),
        "combined_confidence": clip01(combined),
        "requirements": rows,
        "gaps": [x["key"] for x in gaps],
        "critical_gaps": critical_gaps,
        "residual_gap_score": sum(x["importance"] * (1.0 - x["confidence"]) for x in rows),
        "complete": not gaps,
        "formula": "exp(sum(normalized_importance_i * log(max(epsilon, confidence_i))))",
        "dependency_checks_applied": True,
    }


def choose_recovery_action(
    event: AttentionalExperience,
    state_estimate: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Choose the highest expected recovery benefit minus cost among feasible actions."""
    if not audit["gaps"]:
        return {"action": "continue", "expected_benefit": 0.0, "expected_cost": 0.0, "reason": "no_gaps"}
    availability = event.available_recovery
    fatigue = state_estimate["fatigue"]
    candidates: list[tuple[str, float, float, str]] = []
    if availability.get("source", bool(event.source_items)):
        candidates.append(("review_source", 0.90, 0.20, "source_material_available"))
    if availability.get("repeat", False):
        candidates.append(("repeat", 0.82, 0.25, "live_input_can_be_repeated"))
    if availability.get("clarify", False):
        candidates.append(("clarify", 0.85, 0.35, "clarification_is_available"))
    candidates.append(("reorient", 0.45, 0.10, "restore_task_goal"))
    if fatigue >= 0.70 or state_estimate["attentional_availability"] < 0.25:
        candidates.append(("pause", 0.75, 0.15, "sustained_fatigue_or_low_availability"))
        candidates.append(("reduce_demand", 0.62, 0.10, "reduce_current_processing_load"))
    candidates.append(("abstain", 0.0, 0.0, "no_reliable_recovery_evidence"))
    ranked = [
        {"action": name, "expected_benefit": benefit, "expected_cost": cost,
         "net_value": benefit - 0.5 * cost, "reason": reason}
        for name, benefit, cost, reason in candidates
    ]
    ranked.sort(key=lambda x: x["net_value"], reverse=True)
    selected = ranked[0]
    return {**selected, "alternatives": ranked, "gaps": list(audit["gaps"])}


def reorient_to_task(
    event: AttentionalExperience,
    state: AttentionalState,
    priority: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore the current goal and last verified stream position after a lapse."""
    next_gap = next(iter(audit["gaps"]), None)
    gap_item = next((x for x in priority["ranked_items"] if x["key"] == next_gap), None)
    goal_vector = dict(event.goal_vector)
    active_vector = event.metrics.get("active_goal_vector", {})
    alignment = 0.0
    if isinstance(active_vector, Mapping) and goal_vector:
        alignment = sum(float(goal_vector.get(k, 0.0)) * float(active_vector.get(k, 0.0)) for k in goal_vector)
        norm_g = math.sqrt(sum(float(v) ** 2 for v in goal_vector.values()))
        norm_a = math.sqrt(sum(float(active_vector.get(k, 0.0)) ** 2 for k in goal_vector))
        alignment = clip01((alignment / (norm_g * norm_a + 1e-8) + 1.0) / 2.0)
    return {
        "task_goal": event.task_goal,
        "last_verified_position": state.last_verified_position,
        "next_requirement": next_gap,
        "next_priority_key": gap_item["key"] if gap_item else None,
        "goal_alignment": alignment,
        "goal_reinstatement_required": alignment < 0.5 or bool(audit["gaps"]),
        "verified_context_summary": [
            x["key"] for x in audit["requirements"] if x["adequately_encoded"]
        ],
    }


def reencode_recovered_information(
    event: AttentionalExperience,
    representation: Mapping[str, Any],
    missing: Mapping[str, Any],
    action: Mapping[str, Any],
) -> dict[str, Any]:
    """Integrate verified recovered evidence while keeping original and recovered provenance distinct."""
    items = {str(k): dict(v) for k, v in representation.get("items", {}).items()}
    recovered: list[dict[str, Any]] = []
    if action["action"] in {"review_source", "repeat", "clarify", "reorient"}:
        candidates = missing["recoverable_from_source"]
    else:
        candidates = []
    for candidate in candidates:
        key = candidate["key"]
        confidence = clip01(candidate["confidence"])
        old = items.get(key)
        if old and old.get("content") != candidate.get("content"):
            conflict = True
            if candidate.get("provenance") == "recoverable_source_record":
                # A verified source is stronger than a weak first-pass encoding.
                updated_confidence = max(float(old.get("confidence", 0.0)), confidence)
            else:
                updated_confidence = max(float(old.get("confidence", 0.0)), confidence) * 0.5
        elif old:
            updated_confidence = 1.0 - (1.0 - float(old.get("confidence", 0.0))) * (1.0 - confidence)
            conflict = False
        else:
            updated_confidence = confidence
            conflict = False
        row = {
            "key": key, "content": candidate.get("content"),
            "confidence": clip01(updated_confidence),
            "source_id": candidate.get("source_id"),
            "provenance": candidate.get("provenance", "recovered_source"),
            "conflict_with_prior": conflict,
            "prior_record": old,
        }
        items[key] = row
        recovered.append(row)
    return {
        "items": items,
        "recovered_items": recovered,
        "direct_and_recovered_provenance_separate": True,
        "unsupported_content_generated": False,
    }


def verify_repair(
    requirements: Sequence[InformationRequirement],
    representation: Mapping[str, Any],
    config: AttentionalConfig,
) -> dict[str, Any]:
    """Repeat the coverage and dependency audit after any recovery attempt."""
    audit = audit_completeness(requirements, representation, config)
    high_priority_missing = [
        row["key"] for row in audit["requirements"]
        if row["importance"] >= 0.75 and not row["adequately_encoded"]
    ]
    return {
        **audit,
        "verified": audit["complete"] and not high_priority_missing,
        "high_priority_missing": high_priority_missing,
        "recovered_evidence_kept_separate": True,
    }


def adapt_future_attention(
    event: AttentionalExperience,
    state: AttentionalState,
    lapse: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Update context-specific selection history cautiously after repeated feedback."""
    key = event.context_key or "default"
    policy = dict(state.context_policy.get(key, {}))
    observations = int(policy.get("observations", 0.0)) + 1
    previous_lapse_rate = float(policy.get("lapse_rate", 0.0))
    observed_lapse = 1.0 if lapse["detected"] else 0.0
    rate = previous_lapse_rate + (observed_lapse - previous_lapse_rate) / observations
    policy.update({"observations": float(observations), "lapse_rate": clip01(rate)})
    feedback_success = bool(event.feedback.get("recovery_success", verification["verified"]))
    policy["recovery_success_rate"] = (
        float(policy.get("recovery_success_rate", 0.0))
        + (float(feedback_success) - float(policy.get("recovery_success_rate", 0.0))) / observations
    )
    state.context_policy[key] = policy
    selected_keys = set(str(x) for x in event.feedback.get("selected_keys", ()))
    for item in event.incoming_items:
        item_key = str(item.get("key", ""))
        if item_key:
            prior = state.selection_history.get(item_key, 0.0)
            selected = 1.0 if item_key in selected_keys or item.get("selected", False) else 0.0
            state.selection_history[item_key] = clip01(0.9 * prior + 0.1 * selected)
    return {
        "context_key": key,
        "context_observations": observations,
        "context_lapse_rate": policy["lapse_rate"],
        "recovery_success_rate": policy["recovery_success_rate"],
        "policy_update_is_bounded": True,
        "single_lapse_is_not_a_stable_trait": True,
    }


def preserve_uncertainty(
    missing: Mapping[str, Any],
    verification: Mapping[str, Any],
    prior_uncertainty: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep unrecovered content explicitly unknown and retain uncertainty over alternatives."""
    prior = dict(prior_uncertainty or {})
    # The post-recovery verification is authoritative: recovered keys must no longer
    # remain unresolved merely because they were missing before the recovery cycle.
    unresolved = set(verification["gaps"])
    unresolved_map = {key: "unknown_or_insufficient_evidence" for key in sorted(unresolved)}
    unresolved_map.update(prior)
    return {
        "unresolved": unresolved_map,
        "unresolved_count": len(unresolved_map),
        "missingness_by_key": {x["key"]: x["missingness"] for x in missing["missing"]},
        "alternatives": {},
        "entropy_nats": None,
        "guessing_prohibited": True,
        "direct_recovered_and_inferred_evidence_separated": True,
    }


def decide_downstream_release(
    verification: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
    event: AttentionalExperience,
    config: AttentionalConfig,
) -> dict[str, Any]:
    """Gate downstream processing on confidence, weighted coverage, and critical requirements."""
    critical_safe = not verification["critical_gaps"]
    complete_enough = (
        verification["combined_confidence"] >= config.release_confidence_threshold
        and verification["coverage"] >= config.release_coverage_threshold
        and critical_safe
    )
    release = bool(complete_enough and verification["verified"])
    partial = bool(not release and event.allow_partial_release and not event.safety_critical)
    return {
        "release_downstream": release,
        "release_partial": partial,
        "status": "complete" if release else "partial_flagged" if partial else "blocked",
        "coverage": verification["coverage"],
        "combined_confidence": verification["combined_confidence"],
        "critical_gaps": list(verification["critical_gaps"]),
        "missing_markers": dict(uncertainty["unresolved"]),
        "reason": "requirements_verified" if release else "essential_information_missing_or_uncertain",
        "downstream_must_preserve_missing_markers": True,
    }
