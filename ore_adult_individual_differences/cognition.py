"""Cognitive-demand, temporal, interference, support, aging-memory, and confidence operations."""
from __future__ import annotations

import math
from typing import Any, Mapping

from .contracts import (
    AdultComprehensionExperience, AdultDifferencesConfig, AdultLearnerState,
)
from .functional import clamp, entropy, sigmoid, softmax


def estimate_interactivity_and_load(
    units: Mapping[str, Any],
    chunking: Mapping[str, Any],
    event: AdultComprehensionExperience,
    resources: Mapping[str, Any],
    config: AdultDifferencesConfig,
) -> dict[str, Any]:
    """Estimate active-unit and dependency costs after justified schema chunking."""
    active_units = list(chunking.get("effective_unit_count", 0) for _ in range(1))
    count = int(active_units[0]) if active_units else int(units.get("unit_count", 0))
    dependency_records: list[dict[str, Any]] = []
    edge_cost = 0.0
    for left, right in event.dependency_pairs:
        if left >= int(units.get("unit_count", 0)) or right >= int(units.get("unit_count", 0)):
            continue
        weight = clamp(
            float(event.metadata.get("dependency_weights", {}).get(f"{left}:{right}", 1.0))
        )
        edge_cost += weight
        dependency_records.append({"left": left, "right": right, "weight": weight})
    unresolved_count = int(event.task_context.get("unresolved_references", 0))
    alternatives_count = int(event.task_context.get("simultaneous_alternatives", 0))
    proposition_count = int(event.task_context.get("proposition_count", count))
    unresolved_count = max(0, unresolved_count)
    alternatives_count = max(0, alternatives_count)
    proposition_count = max(0, proposition_count)
    base_cost = float(max(0, count))
    dependency_cost = config.load_dependency_weight * edge_cost
    unresolved_cost = 0.75 * unresolved_count + 0.6 * max(0, alternatives_count - 1)
    coordination_cost = 0.25 * max(0, proposition_count - 1)
    load = base_cost + dependency_cost + unresolved_cost + coordination_cost
    wm = max(0.10, float(resources.get("working_memory_availability", 0.5)))
    pressure = load / wm
    return {
        "effective_unit_count": count,
        "original_unit_count": int(units.get("unit_count", 0)),
        "dependency_count": len(dependency_records),
        "dependency_records": dependency_records,
        "base_unit_cost": base_cost, "dependency_cost": dependency_cost,
        "unresolved_reference_cost": unresolved_cost,
        "proposition_coordination_cost": coordination_cost,
        "load_estimate": load, "working_memory_availability": wm,
        "load_pressure": pressure,
        "decomposition_recommended": pressure >= 2.5,
        "load_formula": "sum(unit_costs)+lambda*sum(dependency_weights)+unresolved_and_alternative_costs",
    }


def estimate_processing_schedule(
    event: AdultComprehensionExperience,
    learner: AdultLearnerState,
    resources: Mapping[str, Any],
    load: Mapping[str, Any],
    config: AdultDifferencesConfig,
) -> dict[str, Any]:
    """Estimate component durations and temporal availability rather than one global speed score."""
    unit_count = max(1, int(load.get("effective_unit_count", 1)))
    components = ("perception", "retrieval", "integration", "decision")
    default_base = {"perception": 70.0, "retrieval": 100.0, "integration": 120.0, "decision": 80.0}
    times: dict[str, float] = {}
    speeds: dict[str, float] = {}
    for component in components:
        component_speed = learner.component_speed.get(component, learner.processing_speed)
        current_speed = clamp(
            component_speed * (0.70 + 0.30 * float(resources["processing_speed_estimate"]))
        )
        speeds[component] = current_speed
        observed = event.processing_time_observations_ms.get(component)
        if observed is not None:
            duration = float(observed)
            source = "observed_task_time"
        else:
            multiplier = 1.0 + 0.16 * (unit_count - 1)
            if component == "retrieval":
                multiplier += 0.20 * float(load.get("load_pressure", 0.0))
            if component == "integration":
                multiplier += 0.18 * float(load.get("dependency_count", 0))
            duration = default_base[component] * multiplier / max(0.10, current_speed)
            source = "task_model_estimate"
        times[component] = max(0.0, duration)
    total_time = sum(times.values())
    if event.time_budget_ms is None:
        allowed_time = total_time * max(0.55, 1.25 - event.pace_pressure)
        budget_source = "adaptive_from_current_task_estimate"
    else:
        allowed_time = event.time_budget_ms
        budget_source = "provided_time_budget"
    temporal_availability = {
        component: math.exp(-duration / config.speed_decay_tau_ms)
        for component, duration in times.items()
    }
    temporal_pressure = max(0.0, total_time - allowed_time) / max(1.0, allowed_time)
    bottleneck = max(times, key=times.get) if times else ""
    return {
        "component_times_ms": times, "component_speed_estimates": speeds,
        "component_temporal_availability": temporal_availability,
        "total_processing_time_ms": total_time,
        "available_time_ms": allowed_time, "time_budget_source": budget_source,
        "temporal_pressure": temporal_pressure,
        "pace_pressure": event.pace_pressure,
        "bottleneck_component": bottleneck,
        "recommended_input_rate": (
            "slow_or_pause" if temporal_pressure > 0.15 or event.pace_pressure > 0.65
            else "maintain_current_pace"
        ),
        "processing_speed_is_not_intelligence_or_knowledge": True,
        "availability_equation": "exp(-component_duration_ms/tau_ms)",
    }


def monitor_interference_and_hypotheses(
    integrated: Mapping[str, Any],
    event: AdultComprehensionExperience,
    resources: Mapping[str, Any],
    load: Mapping[str, Any],
    config: AdultDifferencesConfig,
) -> dict[str, Any]:
    """Maintain bounded interpretation alternatives when uncertainty and error cost warrant it."""
    candidates = list(integrated.get("situation_model", {}).get("candidate_interpretations", ()))
    if not candidates:
        candidates = [{
            "hypothesis_id": "input-preserved",
            "interpretation": "input retained without a semantic commitment",
            "confidence": 1.0,
            "provenance": "observed_input",
        }]
    probabilities = [float(item.get("confidence", 0.0)) for item in candidates]
    total = sum(max(0.0, value) for value in probabilities)
    if total <= 0.0:
        probabilities = [1.0 / len(candidates)] * len(candidates)
    else:
        probabilities = [max(0.0, value) / total for value in probabilities]
    ranked = sorted(
        [
            {**dict(item), "probability": probability}
            for item, probability in zip(candidates, probabilities)
        ],
        key=lambda row: row["probability"], reverse=True,
    )
    normalized_entropy = (
        entropy([row["probability"] for row in ranked]) / math.log(len(ranked))
        if len(ranked) > 1 else 0.0
    )
    best = ranked[0]["probability"] if ranked else 0.0
    second = ranked[1]["probability"] if len(ranked) > 1 else 0.0
    conflict = clamp(1.0 - (best - second))
    interference = clamp(
        0.42 * event.distraction + 0.18 * event.fatigue
        + 0.15 * event.stress
        + 0.15 * float(event.task_context.get("competing_goals", 0.0))
        + 0.10 * (1.0 - float(resources.get("attention_control", 0.5)))
    )
    error_cost = clamp(float(event.task_context.get("wrong_commitment_cost", 0.5)))
    defer = (
        normalized_entropy >= config.ambiguity_entropy_threshold
        or (best - second) <= config.ambiguity_margin_threshold
    ) and error_cost >= 0.35
    live_limit = max(1, config.max_live_hypotheses)
    live = ranked[:live_limit] if defer else ranked[:1]
    return {
        "ranked_hypotheses": ranked[:live_limit],
        "live_hypotheses": live,
        "normalized_entropy": clamp(normalized_entropy),
        "conflict": conflict, "interference": interference,
        "error_cost": error_cost,
        "defer_commitment": defer,
        "commitment_state": "bounded_alternatives" if defer else "provisional_best",
        "later_evidence_can_reopen_discarded_hypotheses": True,
    }


def adapt_processing_support(
    event: AdultComprehensionExperience,
    learner: AdultLearnerState,
    resources: Mapping[str, Any],
    load: Mapping[str, Any],
    retrieved: Mapping[str, Any],
    interference: Mapping[str, Any],
    config: AdultDifferencesConfig,
) -> dict[str, Any]:
    """Adapt scaffolding to domain expertise, load, uncertainty, and recent learner feedback."""
    expertise = learner.expertise_by_domain.get(event.domain, 0.0)
    relevant_knowledge = clamp(float(
        retrieved.get("selected_schema", {}).get("score", 0.0)
        if retrieved.get("selected_schema") else 0.0
    ))
    pressure = float(load.get("load_pressure", 0.0))
    recent = learner.performance_history.get(event.domain, ())
    recent_error = 1.0 - (sum(recent[-5:]) / len(recent[-5:])) if recent else 0.25
    support_drive = (
        1.4 * min(3.0, pressure)
        + 0.65 * float(interference.get("normalized_entropy", 0.0))
        + 0.50 * recent_error
        - 2.0 * expertise
        - 0.55 * relevant_knowledge
    )
    support_level = sigmoid(support_drive)
    if support_level >= 0.68:
        strategy = "explicit_scaffolding"
        actions = ["provide_definitions", "show_worked_example", "decompose_into_smaller_steps",
                   "externalize_intermediate_information"]
    elif expertise >= 0.70 and pressure < 1.8:
        strategy = "expert_adaptive"
        actions = ["reduce_redundant_explanation", "emphasize_exceptions",
                   "compare_competing_models", "check_schema_transfer"]
    else:
        strategy = "balanced_guidance"
        actions = ["state_key_relations", "offer_optional_example", "retain_source_notes"]
    if event.external_support > 0.5:
        actions.append("use_available_external_support")
    return {
        "strategy": strategy, "support_level": support_level,
        "domain_expertise": expertise, "relevant_schema_confidence": relevant_knowledge,
        "recent_domain_error": recent_error, "actions": actions,
        "pace_adjustment": "slow_down" if pressure > 2.5 else "maintain_or_advance",
        "expertise_is_domain_specific": True,
        "expertise_reversal_considered": True,
    }


def retrieve_gist_detail_source_memory(
    event: AdultComprehensionExperience,
) -> dict[str, Any]:
    """Estimate gist, episode-detail, and source retrieval separately from direct memory cues."""
    evidence = event.memory_evidence
    interference = clamp(float(evidence.get("interference", event.distraction)))
    encoding_ms = max(0.0, float(evidence.get("encoding_time_ms", 500.0)))
    encoding_support = clamp(encoding_ms / (encoding_ms + 900.0))
    gist_cue = clamp(float(evidence.get("gist_cue_quality", event.evidence_quality)))
    detail_cue = clamp(float(evidence.get("detail_cue_quality", event.evidence_quality * 0.75)))
    source_cue = clamp(float(evidence.get("source_cue_quality", event.provenance_completeness * 0.65)))
    distinctive = clamp(float(evidence.get("distinctiveness", 0.5)))
    gist_probability = sigmoid(-0.25 + 1.7 * gist_cue + 0.55 * encoding_support - 0.65 * interference)
    detail_probability = sigmoid(
        -0.55 + 1.45 * detail_cue + 0.75 * encoding_support
        + 0.40 * distinctive - 0.90 * interference
        + (0.20 if event.memory_request == "detail" else 0.0)
    )
    source_probability = sigmoid(
        -0.70 + 1.50 * source_cue + 0.70 * encoding_support
        + 0.30 * distinctive - 0.90 * interference
        + (0.25 if event.memory_request == "source" else 0.0)
    )
    goal = event.task_goal
    weighted_value = (
        goal.get("gist", 0.0) * gist_probability
        + goal.get("detail", 0.0) * detail_probability
        + goal.get("source", 0.0) * source_probability
    )
    return {
        "gist_probability": gist_probability,
        "episodic_detail_probability": detail_probability,
        "source_context_probability": source_probability,
        "encoding_time_support": encoding_support,
        "interference": interference, "distinctiveness": distinctive,
        "task_weighted_memory_value": weighted_value,
        "age_context_years": event.age_years,
        "age_is_not_a_deterministic_memory_penalty": True,
        "semantic_knowledge_and_gist_are_not_equated_with_detail_retrieval": True,
    }


def calibrate_decision_and_confidence(
    event: AdultComprehensionExperience,
    integrated: Mapping[str, Any],
    interference: Mapping[str, Any],
    retrieved: Mapping[str, Any],
    load: Mapping[str, Any],
    support: Mapping[str, Any],
    aging_memory: Mapping[str, Any],
    config: AdultDifferencesConfig,
    learner: AdultLearnerState,
) -> dict[str, Any]:
    """Calibrate confidence and choose whether to proceed, slow down, retrieve, or clarify."""
    live = list(interference.get("live_hypotheses", ()))
    best = live[0] if live else {
        "hypothesis_id": "input-retained", "interpretation": "input retained",
        "probability": 0.5, "provenance": "observed_input",
    }
    probability = clamp(float(best.get("probability", 0.0)))
    uncertainty = clamp(float(interference.get("normalized_entropy", 0.0)))
    selected_schema = retrieved.get("selected_schema")
    schema_reliability = (
        float(selected_schema.get("reliability", 0.5)) if selected_schema else 0.45
    )
    inferred_count = len(integrated.get("inferred", ()))
    observed_count = len(integrated.get("observed", ()))
    inference_share = inferred_count / max(1, inferred_count + observed_count)
    provenance = event.provenance_completeness
    evidence = event.evidence_quality
    confidence = clamp(
        probability
        * (1.0 - config.confidence_uncertainty_penalty * uncertainty)
        * (0.55 + 0.45 * evidence)
        * (0.60 + 0.40 * schema_reliability)
        * provenance
        * (1.0 - config.confidence_inference_penalty * inference_share)
    )
    if float(load.get("load_pressure", 0.0)) >= 2.8 or float(support.get("support_level", 0.0)) >= 0.78:
        next_action = "slow_down_or_decompose"
    elif bool(interference.get("defer_commitment")):
        next_action = "seek_disambiguating_context"
    elif provenance < 0.55:
        next_action = "retrieve_source_or_check_provenance"
    elif not selected_schema:
        next_action = "retrieve_definitions_or_examples"
    elif confidence < 0.42:
        next_action = "review_evidence_and_revisit_interpretation"
    else:
        next_action = "proceed_and_check_inferences"
    feedback = event.performance_feedback
    observed_accuracy = feedback.get("accuracy")
    if observed_accuracy is not None:
        accuracy = clamp(float(observed_accuracy))
        history = learner.performance_history.setdefault(event.domain, [])
        history.append(accuracy)
        history[:] = history[-30:]
    learner.task_history.append({
        "experience_id": event.experience_id, "domain": event.domain,
        "confidence": confidence, "uncertainty": uncertainty,
        "next_action": next_action, "load_pressure": load.get("load_pressure", 0.0),
        "support_strategy": support.get("strategy"),
    })
    learner.task_history = learner.task_history[-200:]
    return {
        "selected_interpretation": best.get("interpretation", best.get("label")),
        "selected_hypothesis_id": best.get("hypothesis_id"),
        "live_alternatives": live,
        "best_candidate_probability": probability,
        "normalized_uncertainty": uncertainty,
        "evidence_quality": evidence, "schema_reliability": schema_reliability,
        "provenance_completeness": provenance, "inference_share": inference_share,
        "calibrated_confidence": confidence,
        "gist_confidence": aging_memory.get("gist_probability", 0.0),
        "detail_confidence": aging_memory.get("episodic_detail_probability", 0.0),
        "source_confidence": aging_memory.get("source_context_probability", 0.0),
        "next_action": next_action,
        "confidence_is_evidence_sensitive_not_fluency_sensitive": True,
        "no_individual_ability_or_age_diagnosis": True,
    }
