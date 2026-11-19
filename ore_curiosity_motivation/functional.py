"""Twenty-four functional operations for curiosity, motivation, attention, and learning."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contracts import LearningRequest, LearningState, MotivationConfig


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Bound a scalar to an inclusive range."""
    return min(upper, max(lower, float(value)))


def sigmoid(value: float) -> float:
    """Compute a numerically stable logistic activation."""
    if value >= 0.0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


def softmax(scores: Sequence[float], temperature: float = 1.0) -> tuple[float, ...]:
    """Convert finite action or attention scores into a stable normalized distribution."""
    if not scores:
        return ()
    tau = max(float(temperature), 1e-9)
    scaled = [float(score) / tau for score in scores]
    maximum = max(scaled)
    values = [math.exp(max(-700.0, score - maximum)) for score in scaled]
    total = sum(values)
    return tuple(value / total for value in values)


def shannon_entropy(probabilities: Mapping[str, float]) -> float:
    """Calculate Shannon entropy in nats after normalizing non-negative hypothesis weights."""
    cleaned = {key: max(0.0, float(value)) for key, value in probabilities.items()}
    total = sum(cleaned.values())
    if total <= 0.0:
        return 0.0
    return -sum((value / total) * math.log(value / total)
                for value in cleaned.values() if value > 0.0)


def _unit_values(request: LearningRequest) -> list[dict[str, Any]]:
    """Create mutable per-unit model records while retaining original caller input."""
    return [
        {"unit_id": unit.unit_id, "content": unit.content, "relevance": unit.relevance,
         "novelty": unit.novelty, "uncertainty": unit.uncertainty,
         "utility": unit.utility, "intrinsic_interest": unit.intrinsic_interest,
         "transfer_value": unit.transfer_value, "personal_relevance": unit.personal_relevance,
         "conceptual_distance": unit.conceptual_distance,
         "missing_prerequisites": unit.missing_prerequisites,
         "available_support": unit.available_support, "target": unit.target,
         "temporal_context_link": unit.temporal_context_link,
         "semantic_context_link": unit.semantic_context_link,
         "elaboration_hint": unit.elaboration_hint, "reliability": unit.reliability,
         "future_utility": unit.future_utility,
         "expected_features": list(unit.expected_features),
         "expected_value": unit.expected_value}
        for unit in request.units
    ]


def _by_unit(request: LearningRequest) -> dict[str, Any]:
    """Index immutable input units by their stable identifiers."""
    return {unit.unit_id: unit for unit in request.units}


def _action_map(request: LearningRequest) -> dict[str, Any]:
    """Index candidate actions for deterministic lookup during outcome processing."""
    return {action.action_id: action for action in request.actions}


def _average(values: Sequence[float], default: float = 0.0) -> float:
    """Return the arithmetic mean or an explicit default for an empty sequence."""
    return sum(values) / len(values) if values else default


def step_01_parse_learning_units(request: LearningRequest, outputs: dict[str, Any],
                                 state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Represent supplied material as discrete goal-linked units without inventing segmentation."""
    units = _unit_values(request)
    return {"candidate_units": units, "unit_count": len(units),
            "goal": request.goal, "segmentation_source": "caller_supplied_units",
            "unknown_fields_preserved": True}


def step_02_detect_novelty_and_information_gaps(request: LearningRequest, outputs: dict[str, Any],
                                               state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Combine novelty, uncertainty, prior expectations, and relevance into prioritized information gaps."""
    gaps = []
    for unit in request.units:
        prior = state.knowledge.get(unit.unit_id, {})
        prior_confidence = float(prior.get("confidence", 0.0)) if isinstance(prior, Mapping) else 0.0
        mismatch = clamp(abs(unit.novelty - (1.0 - prior_confidence)))
        gap_entropy = max(0.0, unit.uncertainty) * 0.7 + mismatch * 0.3
        learnability_hint = clamp(1.0 - 0.45 * unit.conceptual_distance
                                  - 0.35 * unit.missing_prerequisites
                                  + 0.35 * unit.available_support)
        gap_score = gap_entropy * unit.relevance * learnability_hint
        gaps.append({"unit_id": unit.unit_id, "novelty": unit.novelty,
                     "prediction_mismatch": mismatch, "gap_entropy_proxy": gap_entropy,
                     "learnability_hint": learnability_hint, "information_gap": clamp(gap_score),
                     "useful_gap_not_noise": gap_score > 0.05})
    gaps.sort(key=lambda item: item["information_gap"], reverse=True)
    return {"information_gaps": gaps,
            "mean_information_gap": _average([item["information_gap"] for item in gaps])}


def step_03_estimate_information_value(request: LearningRequest, outputs: dict[str, Any],
                                      state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Estimate practical, intrinsic, transfer, and personally relevant value for each unit."""
    values = []
    for unit in request.units:
        score = (0.30 * unit.utility + 0.30 * unit.intrinsic_interest
                 + 0.20 * unit.transfer_value + 0.20 * unit.personal_relevance)
        if unit.expected_value is not None:
            score = 0.75 * score + 0.25 * unit.expected_value
        uncertainty = clamp(0.25 * unit.uncertainty + 0.25 * (1.0 - unit.reliability)
                            + 0.50 * (1.0 - unit.available_support))
        values.append({"unit_id": unit.unit_id, "expected_value": clamp(score),
                       "value_uncertainty": uncertainty,
                       "components": {"utility": unit.utility,
                                      "intrinsic_interest": unit.intrinsic_interest,
                                      "transfer_value": unit.transfer_value,
                                      "personal_relevance": unit.personal_relevance}})
    return {"expected_values": values, "mean_value": _average([item["expected_value"] for item in values])}


def step_04_appraise_learnability_and_safety(request: LearningRequest, outputs: dict[str, Any],
                                            state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Estimate conceptual feasibility and reduce challenge when stress, fatigue, or prerequisite burden is excessive."""
    results = []
    for unit in request.units:
        score = (1.0 - 0.45 * unit.conceptual_distance
                 - 0.35 * unit.missing_prerequisites + 0.35 * unit.available_support
                 - 0.20 * request.fatigue)
        learnability = sigmoid(score)
        cognitive_cost = clamp(0.35 * unit.conceptual_distance
                               + 0.35 * unit.missing_prerequisites
                               + 0.20 * request.fatigue + 0.10 * request.distraction)
        threat = clamp(0.65 * request.stress + 0.20 * request.frustration
                       + 0.15 * request.emotional_arousal * (1.0 - request.perceived_control))
        appraisal = (next(item["expected_value"] for item in outputs["step_03_estimate_information_value"]["expected_values"]
                          if item["unit_id"] == unit.unit_id) * learnability
                     - 0.45 * cognitive_cost - 0.55 * threat)
        if request.stress >= config.high_stress_threshold or request.fatigue >= config.high_fatigue_threshold:
            decision = "pause_or_regulate"
        elif learnability < 0.25:
            decision = "build_prerequisites"
        elif appraisal > 0.0:
            decision = "explore_now"
        else:
            decision = "simplify_or_defer"
        results.append({"unit_id": unit.unit_id, "learnability": learnability,
                        "cognitive_cost": cognitive_cost, "threat": threat,
                        "appraisal_value": appraisal, "decision": decision})
    return {"learnability": results,
            "exploration_feasibility": _average([item["learnability"] for item in results])}


def step_05_estimate_curiosity(request: LearningRequest, outputs: dict[str, Any],
                               state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Estimate state curiosity from information gain, expected value, learnability, threat, and fatigue."""
    gaps = {item["unit_id"]: item["information_gap"]
            for item in outputs["step_02_detect_novelty_and_information_gaps"]["information_gaps"]}
    values = {item["unit_id"]: item["expected_value"]
              for item in outputs["step_03_estimate_information_value"]["expected_values"]}
    learnability = {item["unit_id"]: item["learnability"]
                    for item in outputs["step_04_appraise_learnability_and_safety"]["learnability"]}
    estimates = []
    for unit in request.units:
        logit = (config.curiosity_bias + config.curiosity_gap_weight * gaps[unit.unit_id]
                 + config.curiosity_value_weight * values[unit.unit_id]
                 + config.curiosity_learnability_weight * learnability[unit.unit_id]
                 - config.curiosity_threat_weight * request.stress
                 - config.curiosity_fatigue_weight * request.fatigue)
        estimates.append({"unit_id": unit.unit_id, "curiosity": sigmoid(logit),
                          "information_gap": gaps[unit.unit_id],
                          "value": values[unit.unit_id],
                          "learnability": learnability[unit.unit_id]})
    mean_curiosity = _average([item["curiosity"] for item in estimates])
    state.curiosity = mean_curiosity
    return {"curiosity_estimates": estimates, "curiosity": mean_curiosity,
            "curiosity_is_state_specific": True}


def step_06_separate_motivation_sources(request: LearningRequest, outputs: dict[str, Any],
                                        state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Keep intrinsic, extrinsic, mastery, personal-value, and avoidance channels distinct."""
    curiosity = outputs["step_05_estimate_curiosity"]["curiosity"]
    intrinsic = clamp(0.55 * request.intrinsic_motivation + 0.45 * curiosity)
    extrinsic = request.extrinsic_motivation
    personal = request.personal_value
    mastery = clamp(_average([unit.transfer_value for unit in request.units]))
    avoidance = clamp(0.65 * request.stress + 0.35 * request.frustration)
    total = clamp(0.32 * intrinsic + 0.22 * extrinsic + 0.24 * personal
                  + 0.14 * mastery - 0.08 * avoidance)
    state.intrinsic_value = intrinsic
    state.extrinsic_value = extrinsic
    state.interest = clamp(_average([unit.intrinsic_interest for unit in request.units]))
    profile = {"intrinsic": intrinsic, "extrinsic": extrinsic, "personal": personal,
               "mastery": mastery, "avoidance": avoidance}
    return {"motivation_profile": profile, "total_motivation": total,
            "value_channels_remain_separate": True}


def step_07_estimate_effort_and_opportunity_cost(request: LearningRequest, outputs: dict[str, Any],
                                                state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Score candidate actions after accounting for expected success, benefit, effort, time, and stress costs."""
    motivation = outputs["step_06_separate_motivation_sources"]["total_motivation"]
    scores = []
    for action in request.actions:
        expected_gain = action.expected_learning_gain * (0.65 + 0.35 * motivation)
        net = (expected_gain + 0.8 * action.expected_information_gain
               + 0.3 * action.expected_interest_gain - 0.40 * action.effort_cost
               - 0.30 * action.time_cost / max(request.time_available, 0.05)
               - 0.50 * action.stress_cost)
        feasible = (action.available and action.time_cost <= request.time_available
                    and request.attention_available > 0.0)
        scores.append({"action_id": action.action_id, "unit_id": action.unit_id,
                       "action_type": action.action_type, "expected_gain": expected_gain,
                       "effort_cost": action.effort_cost, "time_cost": action.time_cost,
                       "stress_cost": action.stress_cost, "net_utility": net,
                       "feasible": feasible})
    scores.sort(key=lambda item: item["net_utility"], reverse=True)
    return {"action_utilities": scores,
            "effort_budget": clamp(request.attention_available * (1.0-request.fatigue)),
            "best_feasible_action": next((item["action_id"] for item in scores if item["feasible"]), None)}


def step_08_set_goal_and_progress(request: LearningRequest, outputs: dict[str, Any],
                                  state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Define measurable short-term progress, challenge fit, and an adaptive difficulty target."""
    planned = max(request.planned_milestones, 1e-9)
    progress = clamp(request.observed_milestones / planned)
    difficulty = _average([unit.conceptual_distance for unit in request.units])
    learner_level = _average([1.0-unit.missing_prerequisites for unit in request.units])
    challenge_fit = clamp(1.0 - abs(difficulty - learner_level))
    observed_accuracy = request.target_success_rate if request.observed_error_rate is None else 1.0-request.observed_error_rate
    adjusted = clamp(difficulty + config.challenge_rate * (request.target_success_rate-observed_accuracy))
    state.goal = request.goal
    return {"goal": request.goal, "progress": progress,
            "observed_milestones": request.observed_milestones,
            "planned_milestones": request.planned_milestones,
            "challenge_fit": challenge_fit, "challenge_level": adjusted,
            "progress_is_observed": True}


def step_09_generate_questions_and_actions(request: LearningRequest, outputs: dict[str, Any],
                                           state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Turn the highest-value information gaps into focused questions and discriminative actions."""
    actions_by_unit: dict[str, list[Any]] = {}
    for action in request.actions:
        actions_by_unit.setdefault(action.unit_id, []).append(action)
    gaps = outputs["step_02_detect_novelty_and_information_gaps"]["information_gaps"]
    questions = []
    for item in gaps:
        unit = next(unit for unit in request.units if unit.unit_id == item["unit_id"])
        candidates = actions_by_unit.get(unit.unit_id, [])
        best = max(candidates, key=lambda a: a.expected_information_gain-a.effort_cost*0.3, default=None)
        question = best.question if best and best.question else f"What evidence would resolve: {unit.content}?"
        questions.append({"unit_id": unit.unit_id, "question": question,
                          "information_gap": item["information_gap"],
                          "recommended_action_id": best.action_id if best else None,
                          "question_specificity_proxy": clamp(1.0-unit.uncertainty*0.5)})
    return {"questions": questions, "action_plan": [item["recommended_action_id"]
                                                       for item in questions if item["recommended_action_id"]]}


def step_10_allocate_attention(request: LearningRequest, outputs: dict[str, Any],
                              state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Allocate a finite attention budget over relevance, novelty, information gap, reliability, and distraction."""
    gap_map = {item["unit_id"]: item["information_gap"]
               for item in outputs["step_02_detect_novelty_and_information_gaps"]["information_gaps"]}
    curiosity = outputs["step_05_estimate_curiosity"]["curiosity"]
    scores = []
    for unit in request.units:
        score = (0.32*unit.relevance + 0.18*unit.novelty
                 + 0.25*gap_map[unit.unit_id] + 0.15*unit.reliability
                 + 0.10*curiosity - 0.25*request.distraction)
        scores.append(score)
    weights = softmax(scores, config.softmax_temperature)
    allocations = [{"unit_id": unit.unit_id, "priority": scores[index],
                    "attention_weight": weights[index],
                    "attention_amount": weights[index] * request.attention_available}
                   for index, unit in enumerate(request.units)]
    return {"attention_allocations": allocations, "attention_weights_sum": sum(weights),
            "available_attention": request.attention_available}


def step_11_explore_and_sample_evidence(request: LearningRequest, outputs: dict[str, Any],
                                        state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Record supplied action evidence and outcomes as samples; do not perform hidden external searches."""
    action_lookup = _action_map(request)
    evidence = []
    for action in request.actions:
        for item in action.evidence:
            evidence.append({"action_id": action.action_id, "unit_id": action.unit_id,
                             "source_reliability": action.source_reliability,
                             "evidence": dict(item),
                             "weight": action.source_reliability * (1.0-0.0)})
    outcomes = [{"action_id": item.action_id, "completed": item.completed,
                 "observed_learning": item.observed_learning,
                 "observed_interest_gain": item.observed_interest_gain,
                 "evidence_quality": item.evidence_quality, "notes": item.notes,
                 "action_type": action_lookup[item.action_id].action_type}
                for item in request.outcomes]
    sampled_action_ids = [item["action_id"] for item in outcomes if item["completed"]]
    return {"sampled_evidence": evidence, "observed_outcomes": outcomes,
            "sampled_action_ids": sampled_action_ids,
            "unresolved_gaps": [item["unit_id"] for item in
                                outputs["step_02_detect_novelty_and_information_gaps"]["information_gaps"]
                                if item["information_gap"] > 0.25],
            "exploration_mode": "caller_supplied_actions_and_evidence"}


def step_12_update_expectations(request: LearningRequest, outputs: dict[str, Any],
                                state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Compute observed-minus-expected learning errors and update action values only from labelled outcomes."""
    actions = _action_map(request)
    errors = []
    updates = {}
    for outcome in request.outcomes:
        action = actions[outcome.action_id]
        previous = state.action_values.get(action.action_id, action.expected_learning_gain)
        if outcome.observed_learning is None:
            errors.append({"action_id": action.action_id, "prediction_error": None,
                           "observed_label_available": False})
            continue
        prediction_error = outcome.observed_learning - previous
        revised = clamp(previous + config.value_learning_rate * prediction_error)
        errors.append({"action_id": action.action_id, "prediction_error": prediction_error,
                       "expected_learning": previous, "observed_learning": outcome.observed_learning,
                       "observed_label_available": True})
        updates[action.action_id] = {"previous_value": previous, "updated_value": revised}
    return {"prediction_errors": errors, "updated_action_values": updates,
            "reward_prediction_error_available": any(item["prediction_error"] is not None for item in errors),
            "semantic_mismatch_is_not_reward_prediction_error": True}


def step_13_encode_target_information(request: LearningRequest, outputs: dict[str, Any],
                                     state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Strengthen target-memory candidates using attention, elaboration, relevance, evidence, and engagement."""
    attention = {item["unit_id"]: item["attention_weight"]
                 for item in outputs["step_10_allocate_attention"]["attention_allocations"]}
    curiosity = outputs["step_05_estimate_curiosity"]["curiosity"]
    motivation = outputs["step_06_separate_motivation_sources"]["total_motivation"]
    engagement = clamp(0.5 * curiosity + 0.5 * motivation)
    memory = []
    for unit in request.units:
        if not unit.target:
            continue
        quality = clamp(0.30*attention[unit.unit_id] + 0.20*unit.elaboration_hint
                        + 0.20*unit.relevance + 0.15*unit.reliability + 0.15*engagement)
        previous = next((item["trace_strength"] for item in state.memory_candidates
                         if item.get("unit_id") == unit.unit_id), 0.0)
        updated = clamp(previous + config.learning_rate * attention[unit.unit_id]
                        * unit.elaboration_hint * unit.relevance * unit.reliability
                        * engagement * (1.0-previous))
        candidate = {"unit_id": unit.unit_id, "content": unit.content,
                     "trace_strength": updated, "encoding_quality": quality,
                     "attention_weight": attention[unit.unit_id],
                     "evidence_reliability": unit.reliability,
                     "memory_kind": "target", "requires_delayed_test": True}
        memory.append(candidate)
    for candidate in memory:
        state.memory_candidates.append(candidate)
    return {"target_memory_candidates": memory,
            "mean_target_trace_strength": _average([item["trace_strength"] for item in memory])}


def step_14_encode_incidental_information(request: LearningRequest, outputs: dict[str, Any],
                                          state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Encode non-target units conservatively when temporal or semantic context links support incidental learning."""
    curiosity = outputs["step_05_estimate_curiosity"]["curiosity"]
    emotional = request.emotional_relevance * (0.5 + 0.5 * abs(request.emotional_valence))
    attention = {item["unit_id"]: item["attention_weight"]
                 for item in outputs["step_10_allocate_attention"]["attention_allocations"]}
    incidental = []
    for unit in request.units:
        if unit.target:
            continue
        moderate_arousal = max(0.0, 1.0-abs(request.emotional_arousal-0.45)/0.75)
        context = unit.temporal_context_link * unit.semantic_context_link
        probability = clamp(context * moderate_arousal * request.attention_available
                            * (0.55*curiosity+0.25*emotional+0.20*attention[unit.unit_id]))
        previous = next((item["trace_strength"] for item in state.incidental_memory_candidates
                         if item.get("unit_id") == unit.unit_id), 0.0)
        strength = clamp(previous + config.incidental_rate * probability * (1.0-previous))
        incidental.append({"unit_id": unit.unit_id, "content": unit.content,
                           "trace_strength": strength, "incidental_priority": probability,
                           "memory_kind": "incidental", "context_link": context})
    state.incidental_memory_candidates.extend(incidental)
    return {"incidental_memory_candidates": incidental,
            "target_and_incidental_kept_separate": True}


def step_15_appraise_emotion(request: LearningRequest, outputs: dict[str, Any],
                             state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Represent affect dimensions separately and estimate task-specific arousal modulation with an inverted-U curve."""
    arousal_factor = math.exp(-((request.emotional_arousal-config.arousal_optimum)**2)
                              /(2.0*max(config.arousal_width, 1e-6)**2))
    salience = clamp(0.20*abs(request.emotional_valence) + 0.30*arousal_factor
                     + 0.25*request.emotional_relevance + 0.25*request.perceived_control)
    threat = clamp(0.50*request.stress + 0.30*request.frustration
                   + 0.20*request.emotional_arousal*(1.0-request.perceived_control))
    if (request.stress >= config.high_stress_threshold
            or request.fatigue >= config.high_fatigue_threshold
            or threat >= config.high_stress_threshold):
        regulation = "reduce_pressure_or_pause"
    elif request.emotional_arousal > 0.8:
        regulation = "downshift_arousal_and_chunk_task"
    else:
        regulation = "continue_with_current_support"
    state.emotional_salience = salience
    return {"affect_vector": {"valence": request.emotional_valence,
                              "arousal": request.emotional_arousal,
                              "perceived_control": request.perceived_control,
                              "relevance": request.emotional_relevance},
            "arousal_performance_factor": arousal_factor,
            "emotional_salience": salience, "threat_estimate": threat,
            "arousal_regulation": regulation}


def step_16_integrate_motivation_and_emotion(request: LearningRequest, outputs: dict[str, Any],
                                            state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Integrate curiosity, interest, value, attention, threat, and fatigue without collapsing their meanings."""
    curiosity = outputs["step_05_estimate_curiosity"]["curiosity"]
    interest = state.interest
    value = outputs["step_03_estimate_information_value"]["mean_value"]
    goal_fit = outputs["step_08_set_goal_and_progress"]["challenge_fit"]
    threat = outputs["step_15_appraise_emotion"]["threat_estimate"]
    logit = (1.2*curiosity + 0.8*interest + 0.8*value + 0.5*goal_fit
             - 1.1*threat - 0.8*request.fatigue)
    engagement = sigmoid(logit)
    threat_gate = sigmoid(6.0*(threat-0.55))
    explore = engagement * (1.0-threat_gate)
    return {"engagement": engagement, "exploration_gate": explore,
            "signals": {"curiosity": curiosity, "interest": interest, "value": value,
                        "goal_fit": goal_fit, "threat": threat, "fatigue": request.fatigue},
            "signals_kept_separate": True}


def step_17_detect_disengagement(request: LearningRequest, outputs: dict[str, Any],
                                 state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Estimate disengagement from boredom, fatigue, error, switching, and lack of progress."""
    progress = outputs["step_08_set_goal_and_progress"]["progress"]
    error_rate = request.observed_error_rate if request.observed_error_rate is not None else 0.0
    score = sigmoid(-1.0 + 1.1*request.boredom + 1.1*request.fatigue
                    + 0.9*error_rate + 0.7*request.task_switch_rate - 1.0*progress)
    if request.fatigue >= config.high_fatigue_threshold:
        cause = "fatigue"
    elif request.stress >= config.high_stress_threshold or request.frustration > 0.75:
        cause = "stress_or_frustration"
    elif error_rate > 0.6:
        cause = "task_difficulty_or_unclear_explanation"
    elif request.boredom > 0.6:
        cause = "low_challenge_or_low_relevance"
    elif request.distraction > 0.6:
        cause = "attention_distraction"
    else:
        cause = "no_dominant_disengagement_signal"
    return {"disengagement_score": score, "disengagement_detected": score >= config.disengagement_threshold,
            "probable_cause": cause, "observed_error_rate": request.observed_error_rate,
            "behavioral_indicators_available": request.observed_error_rate is not None or request.task_switch_rate > 0.0}


def step_18_adapt_challenge_and_explanation(request: LearningRequest, outputs: dict[str, Any],
                                           state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Recommend prerequisite support, deeper challenge, relevance links, clarification, or rest from diagnosed state."""
    disengagement = outputs["step_17_detect_disengagement"]
    challenge = outputs["step_08_set_goal_and_progress"]["challenge_level"]
    accuracy = 1.0-request.observed_error_rate if request.observed_error_rate is not None else request.target_success_rate
    overload = max(request.fatigue, request.stress, request.frustration)
    adjusted = clamp(challenge + config.challenge_rate*(request.target_success_rate-accuracy)
                     - 0.25*overload)
    if request.fatigue >= config.high_fatigue_threshold or request.stress >= config.high_stress_threshold:
        strategy = "pause_or_shorten_session"
    elif any(unit.missing_prerequisites > 0.65 for unit in request.units):
        strategy = "build_prerequisites_and_smaller_questions"
    elif disengagement["probable_cause"] == "low_challenge_or_low_relevance":
        strategy = "deepen_question_or_add_transfer_example"
    elif disengagement["probable_cause"] == "task_difficulty_or_unclear_explanation":
        strategy = "simplify_explanation_and_add_worked_example"
    elif disengagement["probable_cause"] == "attention_distraction":
        strategy = "reduce_distractions_and_narrow_attention"
    else:
        strategy = "continue_with_current_challenge"
    return {"adaptation_strategy": strategy, "adjusted_challenge": adjusted,
            "overload_estimate": overload, "change_reason": disengagement["probable_cause"]}


def step_19_reinforce_progress_and_competence(request: LearningRequest, outputs: dict[str, Any],
                                              state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Update competence and provide specific progress information only from observed milestones or outcomes."""
    labelled = [item.observed_learning for item in request.outcomes if item.observed_learning is not None]
    observed = _average(labelled, default=outputs["step_08_set_goal_and_progress"]["progress"])
    expected = request.expected_learning_baseline
    prediction_error = observed-expected if labelled else None
    previous = state.competence
    if labelled:
        state.competence = clamp(previous + config.competence_rate*prediction_error)
    feedback = {"what_improved": "Observed learning outcome recorded." if labelled else "No independently scored outcome supplied.",
                "remaining_gap": clamp(1.0-observed),
                "next_step": outputs["step_18_adapt_challenge_and_explanation"]["adaptation_strategy"]}
    return {"competence_before": previous, "competence_after": state.competence,
            "observed_learning": observed if labelled else None,
            "prediction_error": prediction_error, "progress_feedback": feedback,
            "competence_update_is_evidence_based": bool(labelled)}


def step_20_preserve_autonomy_and_agency(request: LearningRequest, outputs: dict[str, Any],
                                         state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Prefer meaningful learner-selected paths when feasible while preserving the current learning goal."""
    action_scores = outputs["step_07_estimate_effort_and_opportunity_cost"]["action_utilities"]
    feasible = [item for item in action_scores if item["feasible"]]
    chosen = next((item for item in feasible
                   if next(action for action in request.actions if action.action_id == item["action_id"]).learner_choice), None)
    chosen = chosen or (feasible[0] if feasible else None)
    if chosen:
        action = next(item for item in request.actions if item.action_id == chosen["action_id"])
        perceived_control = 1.0 if action.learner_choice else request.perceived_control
        goal_fit = next((unit.relevance for unit in request.units if unit.unit_id == action.unit_id), 0.0)
        agency = clamp(perceived_control*goal_fit)
    else:
        action, agency = None, 0.0
    return {"selected_path": action.action_id if action else None,
            "selected_action_type": action.action_type if action else "no_feasible_action",
            "learner_choice_honored": bool(action and action.learner_choice),
            "agency_value": agency, "goal_preserved": True}


def step_21_prioritize_memory_retention(request: LearningRequest, outputs: dict[str, Any],
                                        state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Rank target and incidental candidates by relevance, elaboration, affect, novelty, and future utility."""
    scores = []
    units = _by_unit(request)
    for candidate in state.memory_candidates + state.incidental_memory_candidates:
        unit = units[candidate["unit_id"]]
        score = (0.25*unit.relevance + 0.20*unit.elaboration_hint
                 + 0.15*state.emotional_salience + 0.15*unit.novelty
                 + 0.25*unit.future_utility)
        scores.append({"unit_id": unit.unit_id, "memory_kind": candidate["memory_kind"],
                       "retention_priority": clamp(score),
                       "trace_strength": candidate["trace_strength"],
                       "estimated_accessibility": candidate["trace_strength"]*(0.5+0.5*unit.reliability)})
    scores.sort(key=lambda item: item["retention_priority"], reverse=True)
    return {"retention_priorities": scores, "ranked_unit_ids": [item["unit_id"] for item in scores]}


def step_22_support_consolidation(request: LearningRequest, outputs: dict[str, Any],
                                  state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Model gradual offline consolidation potential with sleep/context support as an explicit proxy."""
    elapsed = max(request.time_since_learning_days, 0.0)
    support = clamp(0.55*request.sleep_quality + 0.25*(1.0-request.stress)
                    + 0.20*state.emotional_salience)
    records = []
    all_candidates = state.memory_candidates + state.incidental_memory_candidates
    for candidate in all_candidates:
        initial = candidate["trace_strength"]
        potential = clamp(initial + 0.35*support*(1.0-initial))
        consolidated = clamp(initial + (potential-initial)*(1.0-math.exp(-config.consolidation_rate*elapsed)))
        candidate["consolidation_strength"] = consolidated
        records.append({"unit_id": candidate["unit_id"], "memory_kind": candidate["memory_kind"],
                        "initial_strength": initial, "consolidation_strength": consolidated,
                        "consolidation_support": support})
    return {"consolidated_traces": records,
            "mean_consolidation_strength": _average([item["consolidation_strength"] for item in records]),
            "sleep_quality_is_context_not_direct_neural_measure": True}


def step_23_test_durable_understanding(request: LearningRequest, outputs: dict[str, Any],
                                      state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Score delayed factual, relational, transfer, and calibration dimensions only when measurements are supplied."""
    results = []
    for check in request.retrieval_checks:
        observed = [value for value in (check.factual_accuracy, check.relational_accuracy,
                                        check.transfer_accuracy) if value is not None]
        learning_score = None
        if observed:
            learning_score = _average(observed)
        calibration = (1.0-abs(check.confidence-_average(observed))
                       if check.confidence is not None and observed else None)
        results.append({"check_id": check.check_id, "target_unit_id": check.target_unit_id,
                        "factual_accuracy": check.factual_accuracy,
                        "relational_accuracy": check.relational_accuracy,
                        "transfer_accuracy": check.transfer_accuracy,
                        "confidence": check.confidence, "calibration": calibration,
                        "durable_learning_score": learning_score,
                        "delayed_days": check.delayed_days,
                        "measured_dimensions": len(observed)})
    measured = [item["durable_learning_score"] for item in results
                if item["durable_learning_score"] is not None]
    return {"retrieval_checks": results,
            "durable_learning": _average(measured) if measured else None,
            "all_checks_unmeasured": not bool(measured),
            "learning_not_inferred_from_interest": True}


def step_24_update_future_motivation(request: LearningRequest, outputs: dict[str, Any],
                                    state: LearningState, config: MotivationConfig) -> dict[str, Any]:
    """Update future topic/action values from observed learning and effort, and retain unresolved questions."""
    outcomes = {item.action_id: item for item in request.outcomes}
    actions = _action_map(request)
    updates = {}
    for action_id, outcome in outcomes.items():
        action = actions[action_id]
        if outcome.observed_learning is None:
            continue
        previous = state.action_values.get(action_id, action.expected_learning_gain)
        error = outcome.observed_learning-previous
        revised = clamp(previous+config.value_learning_rate*error)
        state.action_values[action_id] = revised
        updates[action_id] = {"prediction_error": error, "updated_value": revised}
        metrics = state.strategy_success.setdefault(action.action_type, {"attempts": 0.0, "successes": 0.0})
        metrics["attempts"] += 1.0
        metrics["successes"] += float(outcome.observed_learning >= request.target_success_rate)
    for unit in request.units:
        current = state.topic_values.get(unit.unit_id, unit.utility)
        unit_action_ids = {action.action_id for action in request.actions
                           if action.unit_id == unit.unit_id}
        unit_observations = [item.observed_learning for item in request.outcomes
                             if item.action_id in unit_action_ids and item.observed_learning is not None]
        if unit_observations:
            observed = _average(unit_observations)
            state.topic_values[unit.unit_id] = clamp(
                current+config.value_learning_rate*(observed-current))
    unresolved = outputs["step_02_detect_novelty_and_information_gaps"]["information_gaps"]
    state.unresolved_questions = [
        f"What evidence resolves {item['unit_id']}?" for item in unresolved
        if item["information_gap"] > 0.25
    ]
    state.history.append({"request_id": request.request_id,
                          "curiosity": outputs["step_05_estimate_curiosity"]["curiosity"],
                          "engagement": outputs["step_16_integrate_motivation_and_emotion"]["engagement"],
                          "durable_learning": outputs["step_23_test_durable_understanding"]["durable_learning"]})
    state.episode_count += 1
    policy_scores = []
    policy_actions = []
    for action in request.actions:
        value = state.action_values.get(action.action_id, action.expected_learning_gain)
        cost = 0.4*action.effort_cost+0.3*action.time_cost+0.3*action.stress_cost
        policy_scores.append(value+action.expected_information_gain-cost)
        policy_actions.append(action.action_id)
    probabilities = softmax(policy_scores, config.softmax_temperature)
    policy = [{"action_id": action_id, "probability": probabilities[index]}
              for index, action_id in enumerate(policy_actions)]
    policy.sort(key=lambda item: item["probability"], reverse=True)
    return {"action_value_updates": updates, "future_policy": policy,
            "unresolved_questions": list(state.unresolved_questions),
            "strategy_success": state.strategy_success,
            "episode_count": state.episode_count,
            "future_policy_uses_observed_outcomes_only": True}


FUNCTIONAL_STEPS = (
    step_01_parse_learning_units, step_02_detect_novelty_and_information_gaps,
    step_03_estimate_information_value, step_04_appraise_learnability_and_safety,
    step_05_estimate_curiosity, step_06_separate_motivation_sources,
    step_07_estimate_effort_and_opportunity_cost, step_08_set_goal_and_progress,
    step_09_generate_questions_and_actions, step_10_allocate_attention,
    step_11_explore_and_sample_evidence, step_12_update_expectations,
    step_13_encode_target_information, step_14_encode_incidental_information,
    step_15_appraise_emotion, step_16_integrate_motivation_and_emotion,
    step_17_detect_disengagement, step_18_adapt_challenge_and_explanation,
    step_19_reinforce_progress_and_competence, step_20_preserve_autonomy_and_agency,
    step_21_prioritize_memory_retention, step_22_support_consolidation,
    step_23_test_durable_understanding, step_24_update_future_motivation,
)
