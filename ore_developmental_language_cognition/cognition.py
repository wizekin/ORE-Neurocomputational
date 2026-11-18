"""Working memory, executive control, circuit refinement, and developmental calibration."""
from __future__ import annotations

import math
from typing import Any, Mapping

from .contracts import DevelopmentalConfig, DevelopmentalExperience, DevelopmentalLearnerState
from .functional import clamp, feature_overlap, softmax


def maintain_working_memory(
    event: DevelopmentalExperience,
    state: DevelopmentalLearnerState,
    config: DevelopmentalConfig,
) -> dict[str, Any]:
    """Maintain goal-relevant items with decay, interference, variable capacity, and chunking."""
    old = list(state.working_memory_items)
    for row in old:
        row["activation"] = clamp(float(row.get("activation", 0.0)) * (1.0 - config.working_memory_decay))
    goal_tokens = set(str(event.current_goal).lower().split())
    new_items = [str(item) for item in event.working_memory_items]
    distractors = [str(item) for item in event.distractors]
    schema_chunks = list(event.metadata.get("schema_chunks", ()))
    chunk_map: dict[str, str] = {}
    for chunk in schema_chunks:
        chunk_name = str(chunk.get("name", "chunk"))
        for item in chunk.get("items", ()):
            chunk_map[str(item)] = chunk_name
    for item in new_items:
        match = next((row for row in old if row.get("item") == item), None)
        previous_activation = float(match.get("activation", 0.0)) if match else 0.0
        goal_fit = feature_overlap(item, goal_tokens)
        old[:] = [row for row in old if row.get("item") != item]
        old.append({
            "item": item,
            "activation": clamp(0.45 + 0.35 * goal_fit + 0.20 * previous_activation),
            "goal_relevance": goal_fit,
            "chunk": chunk_map.get(item, ""),
        })
    for distractor in distractors:
        for row in old:
            if row.get("item") != distractor:
                row["activation"] = clamp(float(row.get("activation", 0.0)) * 0.92)
    groups: dict[str, list[str]] = {}
    for row in old:
        if row.get("chunk"):
            groups.setdefault(str(row["chunk"]), []).append(str(row["item"]))
    chunked_items = {item for members in groups.values() for item in members}
    units = len(groups) + sum(1 for row in old if row["item"] not in chunked_items)
    effective_load = float(units) + 0.15 * sum(
        1 for row in old if row.get("item") in distractors
    )
    capacity = max(0.5, state.working_memory_capacity)
    ranked = sorted(old, key=lambda row: (
        float(row.get("activation", 0.0)) + 0.35 * float(row.get("goal_relevance", 0.0))
    ), reverse=True)
    budget = max(1, int(math.ceil(capacity)))
    retained_keys: set[str] = set()
    used_units = 0
    for row in ranked:
        chunk = str(row.get("chunk", ""))
        key = "chunk:" + chunk if chunk else "item:" + str(row["item"])
        if key in retained_keys:
            continue
        if used_units < budget:
            retained_keys.add(key)
            used_units += 1
    retained = [
        row for row in ranked if (
            ("chunk:" + str(row.get("chunk")) if row.get("chunk") else "item:" + str(row["item"]))
            in retained_keys
        )
    ]
    state.working_memory_items = retained
    return {
        "active_items": list(retained), "effective_capacity": capacity,
        "load": effective_load, "chunk_count": len(groups),
        "chunked_items": {key: list(value) for key, value in groups.items()},
        "over_capacity": effective_load > capacity,
        "distraction_items": distractors,
        "capacity_is_flexible_and_chunk_sensitive": True,
    }


def update_executive_control(
    event: DevelopmentalExperience,
    state: DevelopmentalLearnerState,
    working_memory: Mapping[str, Any],
    config: DevelopmentalConfig,
) -> dict[str, Any]:
    """Score goal-relevant actions, inhibit conflict, shift rules, and learn from feedback."""
    weights = state.executive_weights
    goal = event.current_goal.lower()
    candidates: list[dict[str, Any]] = []
    for index, action in enumerate(event.action_candidates):
        identifier = str(action.get("action_id", action.get("name", f"action-{index + 1}")))
        goal_fit = clamp(float(action.get("goal_match", 0.5)))
        relevance = clamp(float(action.get("relevance", 0.5)))
        distraction = clamp(float(action.get("distraction", 0.0)))
        prepotency = clamp(float(action.get("prepotency", 0.5)))
        conflict = clamp(float(action.get("conflict", 0.0)))
        score = (
            weights.get("goal", 0.55) * goal_fit
            + weights.get("relevance", 0.35) * relevance
            - weights.get("distraction", 0.25) * distraction
            - weights.get("prepotency", 0.20) * prepotency * conflict
        )
        candidates.append({
            "action_id": identifier, "score": score, "goal_fit": goal_fit,
            "relevance": relevance, "distraction": distraction,
            "prepotency": prepotency, "conflict": conflict,
        })
    probabilities = softmax([row["score"] for row in candidates], config.executive_temperature)
    for row, probability in zip(candidates, probabilities):
        row["probability"] = probability
    candidates.sort(key=lambda item: item["probability"], reverse=True)
    rule_switched = bool(event.current_rule and event.previous_rule
                         and event.current_rule != event.previous_rule)
    switch_cost = 0.0
    if rule_switched:
        switch_cost = clamp(0.25 + float(weights.get("switching", 0.25)))
        state.current_rule = event.current_rule
    elif event.current_rule:
        state.current_rule = event.current_rule
    feedback = event.rule_feedback
    error = clamp(float(feedback.get("error", 0.0)))
    if feedback:
        state.executive_error_history.append(error)
        state.executive_error_history = state.executive_error_history[-50:]
        target = 1.0 - error
        for key in ("goal", "relevance", "distraction"):
            current = clamp(float(weights.get(key, 0.3)))
            direction = target if key != "distraction" else error
            weights[key] = clamp(current + config.executive_learning_rate * (direction - current))
    top_conflict = candidates[0]["conflict"] if candidates else 0.0
    control_demand = clamp(
        0.35 * float(working_memory.get("load", 0.0)) / max(1.0, state.working_memory_capacity)
        + 0.35 * float(rule_switched) + 0.30 * top_conflict
    )
    return {
        "goal": goal, "action_scores": candidates,
        "selected_action": candidates[0]["action_id"] if candidates else None,
        "selected_probability": candidates[0]["probability"] if candidates else 0.0,
        "inhibition_required": bool(candidates and candidates[0]["conflict"] > 0.5),
        "rule_switched": rule_switched, "switch_cost_proxy": switch_cost,
        "control_demand": control_demand, "control_weights": dict(weights),
        "mean_recent_error": (
            sum(state.executive_error_history) / len(state.executive_error_history)
            if state.executive_error_history else 0.0
        ),
        "goal_maintenance_and_error_monitoring": True,
    }


def refine_developing_circuits(
    event: DevelopmentalExperience,
    state: DevelopmentalLearnerState,
    config: DevelopmentalConfig,
) -> dict[str, Any]:
    """Apply coactivity learning, cautious underuse pruning, and myelination timing proxies."""
    active = {str(key): clamp(float(value)) for key, value in event.circuit_activations.items()}
    pathways = {str(key): clamp(float(value)) for key, value in event.active_pathways.items()}
    keys = sorted(active)
    updated: list[dict[str, Any]] = []
    visited: set[str] = set()
    modulatory = clamp(float(event.metadata.get("neuromodulatory_context", 0.5)))
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            key = left + "->" + right
            visited.add(key)
            previous = state.circuit_weights.get(key, 0.10)
            coactivity = active[left] * active[right]
            delta = config.circuit_learning_rate * coactivity * modulatory - config.circuit_decay
            weight = clamp(previous + delta)
            state.circuit_weights[key] = weight
            low_count = state.circuit_low_use_counts.get(key, 0)
            low_count = low_count + 1 if coactivity < config.pruning_threshold else max(0, low_count - 1)
            state.circuit_low_use_counts[key] = low_count
            pruned = low_count >= config.pruning_patience and weight < 0.12
            if pruned:
                state.circuit_weights[key] = 0.0
                state.circuit_low_use_counts[key] = 0
            updated.append({
                "connection": key, "before": previous, "after": state.circuit_weights[key],
                "coactivity": coactivity, "low_use_count": low_count, "pruned": pruned,
            })
    # Connections not co-active in this experience receive a small decay and accrue underuse evidence.
    for key, previous in list(state.circuit_weights.items()):
        if key in visited or not active:
            continue
        low_count = state.circuit_low_use_counts.get(key, 0) + 1
        state.circuit_low_use_counts[key] = low_count
        weight = clamp(previous - config.circuit_decay * 0.25)
        pruned = low_count >= config.pruning_patience and weight < 0.12
        state.circuit_weights[key] = 0.0 if pruned else weight
        if pruned:
            state.circuit_low_use_counts[key] = 0
        updated.append({
            "connection": key, "before": previous, "after": state.circuit_weights[key],
            "coactivity": 0.0, "low_use_count": low_count, "pruned": pruned,
        })
    myelin_updates: list[dict[str, Any]] = []
    delays: dict[str, float] = {}
    for path, usage in pathways.items():
        previous = state.myelination.get(path, 0.05)
        updated_myelin = clamp(previous + config.myelination_rate * usage * (1.0 - previous))
        state.myelination[path] = updated_myelin
        length = max(0.001, float(event.path_lengths.get(path, 1.0)))
        velocity = 1.0 + 4.0 * updated_myelin
        delays[path] = length / velocity
        myelin_updates.append({
            "pathway": path, "before": previous, "after": updated_myelin, "usage": usage,
            "conduction_velocity_proxy": velocity, "conduction_delay_proxy": delays[path],
        })
    return {
        "connection_updates": updated, "myelination_updates": myelin_updates,
        "connectivity_count": len(state.circuit_weights),
        "myelination": dict(state.myelination), "conduction_delay_proxy": delays,
        "pruning_is_activity_dependent_and_cautious": True,
        "biological_synapses_not_equated_with_regularization": True,
    }


def integrate_hippocampal_prefrontal_control(
    event: DevelopmentalExperience,
    state: DevelopmentalLearnerState,
    executive: Mapping[str, Any],
    config: DevelopmentalConfig,
) -> dict[str, Any]:
    """Retrieve relevant episodes and gradually coordinate memory with goal-directed control."""
    query = set(event.memory_cues) | set(event.current_goal.lower().split())
    rows: list[dict[str, Any]] = []
    for episode in state.episodic_memory:
        tokens = set(str(episode.get("word", "")).split())
        tokens |= set(episode.get("context_tokens", ()))
        overlap = len(query & tokens) / max(1, len(query | tokens))
        if overlap > 0.0:
            rows.append({
                "experience_id": episode.get("experience_id"),
                "relevance": overlap, "meanings": episode.get("meanings", []),
            })
    rows.sort(key=lambda item: item["relevance"], reverse=True)
    retrieved = rows[:5]
    age = event.age_months if event.age_months is not None else 0.0
    # Smooth contextual weighting only; chronological age is not a maturity diagnosis.
    age_context = age / (age + 72.0) if age > 0.0 else 0.0
    evidence = clamp(
        0.5 * (len(retrieved) / 5.0)
        + 0.5 * float(bool(executive.get("selected_action")))
    )
    target = clamp(0.15 + 0.65 * evidence * age_context)
    before = state.hippocampal_prefrontal_coupling
    state.hippocampal_prefrontal_coupling = clamp(
        before + config.memory_learning_rate * (target - before)
    )
    memory_value = sum(row["relevance"] for row in retrieved)
    integrated_value = clamp(
        0.55 * float(executive.get("selected_probability", 0.0))
        + 0.45 * (memory_value / max(1, len(retrieved)))
    ) if retrieved else float(executive.get("selected_probability", 0.0))
    return {
        "retrieved_memories": retrieved,
        "integrated_action": executive.get("selected_action"),
        "memory_relevance": memory_value, "integrated_value": integrated_value,
        "coupling_before": before, "coupling_after": state.hippocampal_prefrontal_coupling,
        "age_context_weight": age_context,
        "age_is_context_not_a_fixed_maturity_cutoff": True,
    }


def refine_multi_step_plan(
    event: DevelopmentalExperience,
    state: DevelopmentalLearnerState,
) -> dict[str, Any]:
    """Track task dependencies and complete only steps with satisfied prerequisites and evidence."""
    plan = [dict(step) for step in event.plan_steps]
    if not plan:
        return {
            "completed_steps": list(state.multi_step_state.get("completed", ())),
            "pending_steps": list(state.multi_step_state.get("pending", ())),
            "ready_steps": [], "goal_satisfied": False, "dependency_graph": [],
            "premature_action_suppressed": False,
        }
    known_ids = {str(step.get("step_id", index)) for index, step in enumerate(plan)}
    completed = set(str(item) for item in state.multi_step_state.get("completed", ()))
    observations = dict(event.plan_observations)
    invalid: list[str] = []
    for index, step in enumerate(plan):
        identifier = str(step.get("step_id", index))
        dependencies = [str(item) for item in step.get("depends_on", ())]
        if any(item not in known_ids for item in dependencies):
            invalid.append(identifier)
    ready: list[str] = []
    newly_completed: list[str] = []
    for index, step in enumerate(plan):
        identifier = str(step.get("step_id", index))
        dependencies = [str(item) for item in step.get("depends_on", ())]
        if identifier in completed or identifier in invalid:
            continue
        if all(item in completed for item in dependencies):
            ready.append(identifier)
            observation = observations.get(identifier)
            if observation is not None:
                is_complete = (
                    bool(observation.get("completed", True))
                    if isinstance(observation, Mapping) else bool(observation)
                )
                if is_complete:
                    completed.add(identifier)
                    newly_completed.append(identifier)
    pending = sorted(known_ids - completed)
    state.multi_step_state = {
        "completed": sorted(completed), "pending": pending,
        "goal": event.current_goal, "step_count": len(plan),
    }
    dependency_graph = [
        {"step_id": str(step.get("step_id", index)),
         "depends_on": [str(item) for item in step.get("depends_on", ())]}
        for index, step in enumerate(plan)
    ]
    goal_satisfied = bool(plan) and len(completed & known_ids) == len(known_ids) and not invalid
    return {
        "completed_steps": sorted(completed), "newly_completed_steps": newly_completed,
        "pending_steps": pending, "ready_steps": ready,
        "invalid_dependency_steps": invalid, "dependency_graph": dependency_graph,
        "goal_satisfied": goal_satisfied,
        "premature_action_suppressed": bool(pending) or bool(invalid),
        "completed_and_pending_are_separated": True,
    }


def calibrate_developmental_support(
    event: DevelopmentalExperience,
    state: DevelopmentalLearnerState,
    config: DevelopmentalConfig,
) -> dict[str, Any]:
    """Adapt multidomain difficulty and support from repeated performance, not age-only rules."""
    feedback = event.performance_feedback
    observations = feedback.get("domains", {})
    updated: dict[str, Any] = {}
    for domain, raw in observations.items():
        score = clamp(float(raw.get("success", raw.get("accuracy", 0.5))))
        history = state.performance_history.setdefault(str(domain), [])
        history.append(score)
        history[:] = history[-30:]
        mean_score = sum(history) / len(history)
        current = state.developmental_parameters.get(str(domain), 0.35)
        parameter = clamp(current + config.calibration_rate * (score - 0.5))
        difficulty = state.task_difficulty.get(str(domain), 0.45)
        difficulty = clamp(
            difficulty + config.calibration_rate * (mean_score - config.target_success)
        )
        state.developmental_parameters[str(domain)] = parameter
        state.task_difficulty[str(domain)] = difficulty
        if str(domain) == "working_memory":
            # Capacity changes gradually with repeated performance; it is not a fixed slot count.
            state.working_memory_capacity = max(
                0.5,
                min(12.0, state.working_memory_capacity
                    + config.calibration_rate * (mean_score - config.target_success)),
            )
        updated[str(domain)] = {
            "observed_success": score, "recent_mean_success": mean_score,
            "developmental_parameter": parameter, "next_task_difficulty": difficulty,
            "additional_support": mean_score < config.target_success,
            "more_challenge": mean_score >= config.target_success,
        }
    return {
        "domain_updates": updated, "domains_tracked_separately": True,
        "age_is_not_used_as_a_single_maturity_score": True,
        "recommended_support": {
            domain: {
                "repeat_more": row["additional_support"],
                "increase_challenge_gradually": row["more_challenge"],
                "difficulty": row["next_task_difficulty"],
            } for domain, row in updated.items()
        },
        "calibration_target": config.target_success,
        "working_memory_capacity_after_calibration": state.working_memory_capacity,
    }
