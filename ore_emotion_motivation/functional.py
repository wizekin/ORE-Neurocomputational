"""Functional mathematics for Point 13 affect, attention, effort, and memory."""
from __future__ import annotations

from dataclasses import replace
from math import exp, log, sqrt
from typing import Mapping, Sequence

from .contracts import (
    AffectiveAppraisal, AttentionAllocation, EmotionLearningConfig, EventFeature,
    EffortDecision, LearningExperience, LearningState, MemoryTrace, RetrievedMemory,
    RewardFeedback,
)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Bound a finite model value to an inclusive interval."""
    if lower > upper:
        raise ValueError("lower bound cannot exceed upper bound")
    return min(upper, max(lower, float(value)))


def softmax(scores: Mapping[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Convert finite scores into a numerically stable normalized distribution."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not scores:
        return {}
    peak = max(float(value) for value in scores.values())
    values = {str(key): exp((float(value) - peak) / temperature)
              for key, value in scores.items()}
    total = sum(values.values())
    if total <= 0 or not total < float("inf"):
        raise ValueError("scores must yield a finite positive normalization")
    return {key: value / total for key, value in values.items()}


def cosine_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Compute cosine similarity between sparse feature maps and return zero for zero vectors."""
    keys = set(left) | set(right)
    dot = sum(float(left.get(key, 0.0)) * float(right.get(key, 0.0)) for key in keys)
    norm_left = sqrt(sum(float(value) ** 2 for value in left.values()))
    norm_right = sqrt(sum(float(value) ** 2 for value in right.values()))
    if norm_left <= 0.0 or norm_right <= 0.0:
        return 0.0
    return clamp(dot / (norm_left * norm_right), -1.0, 1.0)


def extract_event_vector(experience: LearningExperience) -> dict[str, float]:
    """Extract a sparse event-feature vector while retaining each feature's signed value."""
    return {feature.feature_id: float(feature.value) for feature in experience.features}


def estimate_affective_appraisal(experience: LearningExperience) -> AffectiveAppraisal:
    """Estimate valence and arousal separately from feature cues and contextual appraisal inputs."""
    total_weight = sum(max(0.05, item.salience) for item in experience.features)
    valence = sum(item.valence * max(0.05, item.salience)
                  for item in experience.features) / total_weight
    feature_arousal = sum(item.arousal * max(0.05, item.salience)
                          for item in experience.features) / total_weight
    valence = 0.65 * valence + 0.35 * experience.valence_hint
    arousal = clamp(0.65 * feature_arousal + 0.35 * experience.arousal_hint)
    threat = clamp(experience.threat + max(0.0, -valence) * 0.25)
    reward = clamp(experience.reward_relevance + max(0.0, valence) * 0.20)
    novelty = experience.novelty
    uncertainty = experience.uncertainty
    significance = clamp(
        0.30 * threat + 0.25 * reward + 0.20 * novelty
        + 0.10 * uncertainty + 0.15 * arousal
    )
    labels: list[str] = []
    if threat >= 0.6:
        labels.append("threat_relevant")
    if reward >= 0.6:
        labels.append("reward_relevant")
    if novelty >= 0.6:
        labels.append("novel")
    if arousal >= 0.7:
        labels.append("high_arousal")
    if not labels:
        labels.append("low_or_mixed_significance")
    return AffectiveAppraisal(
        valence=clamp(valence, -1.0, 1.0), arousal=arousal,
        significance=significance, threat=threat,
        reward_relevance=reward, novelty=novelty,
        uncertainty=uncertainty, confidence=0.65, labels=tuple(labels),
    )


def update_arousal_state(
    state: LearningState,
    appraisal: AffectiveAppraisal,
    experience: LearningExperience,
    config: EmotionLearningConfig,
) -> dict[str, float]:
    """Update fast arousal and slow stress proxies separately and estimate an NE gain proxy."""
    previous = state.arousal
    fast_input = clamp(0.55 * appraisal.arousal + 0.30 * appraisal.significance
                       + 0.15 * experience.body_state.get("arousal", 0.0))
    fast = (
        config.fast_arousal_retention * previous.fast_arousal
        + (1.0 - config.fast_arousal_retention) * fast_input
    )
    slow_input = clamp(0.65 * appraisal.threat + 0.20 * appraisal.arousal
                       + 0.15 * experience.body_state.get("stress", 0.0))
    slow = (
        config.slow_stress_retention * previous.slow_stress_load
        + (1.0 - config.slow_stress_retention) * slow_input
    )
    ne_gain = clamp(0.5 + config.ne_gain_strength * (1.0 - exp(-2.0 * fast)))
    glucocorticoid = clamp(0.90 * previous.glucocorticoid_load + 0.10 * slow)
    fatigue = clamp(0.70 * previous.fatigue + 0.30 * experience.fatigue)
    state.arousal.fast_arousal = fast
    state.arousal.slow_stress_load = slow
    state.arousal.norepinephrine_gain = ne_gain
    state.arousal.glucocorticoid_load = glucocorticoid
    state.arousal.fatigue = fatigue
    return {
        "fast_arousal": fast, "slow_stress_load": slow,
        "norepinephrine_gain_proxy": ne_gain,
        "glucocorticoid_load_proxy": glucocorticoid, "fatigue": fatigue,
    }


def compute_attentional_priorities(
    features: Sequence[EventFeature],
    goal_strength: float,
    appraisal: AffectiveAppraisal,
    config: EmotionLearningConfig,
    context_relevance: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Combine bottom-up salience, goal fit, context, and emotional capture into priority scores."""
    context_relevance = context_relevance or {}
    scores: dict[str, float] = {}
    for feature in features:
        goal = feature.goal_relevance * goal_strength
        context = clamp(float(context_relevance.get(feature.feature_id, 0.5)))
        threat_capture = (
            config.threat_capture_gain * appraisal.threat
            * max(0.0, -feature.valence) * feature.salience
        )
        raw = (
            0.38 * feature.salience + 0.34 * goal
            + 0.20 * context + 0.08 * feature.centrality + threat_capture
        )
        scores[feature.feature_id] = clamp(raw)
    return scores


def allocate_attention(
    priorities: Mapping[str, float],
    config: EmotionLearningConfig,
    control_capacity: float = 1.0,
) -> AttentionAllocation:
    """Allocate a finite resource budget across features using normalized priority competition."""
    budget = clamp(config.attention_capacity * control_capacity)
    probabilities = softmax(priorities, temperature=0.35)
    weights = {key: value * budget for key, value in probabilities.items()}
    entropy = -sum(value * log(value) for value in probabilities.values() if value > 0)
    normalized_entropy = entropy / log(len(probabilities)) if len(probabilities) > 1 else 0.0
    selected = tuple(
        key for key, weight in sorted(weights.items(), key=lambda item: (-item[1], item[0]))
        if weight >= max(0.01, budget / max(1, len(weights)) * 0.5)
    )
    return AttentionAllocation(
        priorities=priorities, weights=weights, selected_feature_ids=selected,
        entropy=clamp(normalized_entropy), capacity_budget=budget,
    )


def estimate_neuromodulatory_gain(
    arousal_state: Mapping[str, float],
    config: EmotionLearningConfig,
) -> float:
    """Compute an illustrative inverted-U gain proxy with a separate LC-NE factor."""
    arousal = clamp(float(arousal_state.get("fast_arousal", 0.0)))
    ne = clamp(float(arousal_state.get("norepinephrine_gain_proxy", arousal_state.get("norepinephrine_gain", 0.5))))
    distance = arousal - config.arousal_optimum
    task_gain = exp(-(distance * distance) / (2.0 * config.arousal_sigma ** 2))
    # The floor prevents a single arousal estimate from shutting off all feature processing.
    return clamp((0.45 + 0.75 * task_gain) * (0.70 + 0.60 * ne), 0.25, 1.5) / 1.5


def divisive_normalization(
    activations: Mapping[str, float],
    weights: Mapping[str, float],
    epsilon: float = 1e-8,
) -> dict[str, float]:
    """Normalize non-negative population activations by weighted pooled activity."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    pooled = sum(max(0.0, float(value)) * max(0.0, float(weights.get(key, 0.0)))
                 for key, value in activations.items())
    denominator = epsilon + pooled
    return {
        key: max(0.0, float(value)) * max(0.0, float(weights.get(key, 0.0))) / denominator
        for key, value in activations.items()
    }


def modulate_feature_activity(
    experience: LearningExperience,
    attention: AttentionAllocation,
    gain: float,
) -> dict[str, float]:
    """Apply attention and state-dependent gain to positive feature activity."""
    return {
        feature.feature_id: max(0.0, feature.value) * attention.weights.get(feature.feature_id, 0.0)
        * clamp(gain, 0.0, 1.5) / 1.5
        for feature in experience.features
    }


def estimate_control_capacity(
    arousal_state: Mapping[str, float],
    fatigue: float,
    task_demands: float,
    config: EmotionLearningConfig,
) -> float:
    """Estimate task-specific control capacity with non-monotonic arousal and fatigue costs."""
    arousal = clamp(float(arousal_state.get("fast_arousal", 0.0)))
    slow_stress = clamp(float(arousal_state.get("slow_stress_load", 0.0)))
    distance = abs(arousal - config.arousal_optimum)
    arousal_fit = exp(-(distance * distance) / (2.0 * (config.arousal_sigma * 1.6) ** 2))
    chronic_load = 1.0 - 0.35 * slow_stress
    fatigue_factor = 1.0 - 0.50 * clamp(fatigue)
    demand_factor = 1.0 - 0.15 * clamp(task_demands)
    return clamp(arousal_fit * chronic_load * fatigue_factor * demand_factor, 0.10, 1.0)


def encode_memory_trace(
    experience: LearningExperience,
    attention: AttentionAllocation,
    appraisal: AffectiveAppraisal,
    state: LearningState,
    config: EmotionLearningConfig,
) -> MemoryTrace:
    """Encode an event with attention-weighted features, relation links, and BLA-like modulation."""
    modulation = clamp(appraisal.arousal * appraisal.significance)
    strengths: dict[str, float] = {}
    for feature in experience.features:
        attention_weight = attention.weights.get(feature.feature_id, 0.0)
        bounded_magnitude = abs(feature.value) / (1.0 + abs(feature.value))
        base = (
            0.20 * bounded_magnitude + 0.40 * attention_weight
            + 0.20 * feature.salience + 0.20 * feature.goal_relevance
        )
        emotional_gain = 1.0 + 0.45 * modulation * (0.5 + 0.5 * feature.salience)
        strengths[feature.feature_id] = clamp(base * emotional_gain)
    tags: dict[str, list[str]] = {}
    for feature in experience.features:
        for tag in feature.tags:
            tags.setdefault(tag, []).append(feature.feature_id)
    links: list[tuple[str, str, float]] = []
    for members in tags.values():
        unique = sorted(set(members))
        if len(unique) > 1:
            for index, left in enumerate(unique):
                for right in unique[index + 1:]:
                    links.append((left, right, 1.0))
    central = tuple(item.feature_id for item in experience.features if item.role == "central")
    peripheral = tuple(item.feature_id for item in experience.features if item.role == "peripheral")
    context_tags = tuple(sorted(set(
        tag for feature in experience.features for tag in feature.tags
    ) | {str(key).lower() for key in experience.context}))
    encoding_strength = sum(strengths.values()) / max(1, len(strengths))
    return MemoryTrace(
        event_id=experience.event_id,
        feature_strengths=strengths,
        encoded_feature_ids=tuple(item.feature_id for item in experience.features),
        central_feature_ids=central, peripheral_feature_ids=peripheral,
        relation_links=tuple(links), context_tags=context_tags,
        encoding_strength=encoding_strength,
        consolidation_strength=encoding_strength,
        amygdala_modulation=modulation,
        provenance="selective_episodic_encoding",
    )


def estimate_motivational_value(
    experience: LearningExperience,
    state: LearningState,
) -> tuple[float, float]:
    """Estimate a goal-conditioned expected reward and net utility before effort costs."""
    prior = state.motivation.expected_reward
    reward_value = (
        0.55 * prior + 0.30 * experience.reward_relevance
        + 0.15 * state.motivation.goal_strength
    )
    effort_cost = 0.35 * experience.task_demands + 0.35 * state.arousal.fatigue + 0.30 * state.arousal.slow_stress_load
    utility = reward_value - effort_cost
    state.motivation.expected_reward = reward_value
    return reward_value, utility


def update_reward_values(
    feedback: RewardFeedback | None,
    state: LearningState,
    config: EmotionLearningConfig,
) -> tuple[float, dict[str, float]]:
    """Apply a temporal-difference value update when observed action feedback is available."""
    if feedback is None:
        return 0.0, {
            "feedback_available": 0.0,
            "expected_value": state.motivation.action_values.get("default", 0.0),
            "updated_value": state.motivation.action_values.get("default", 0.0),
        }
    current = state.motivation.state_values.get(feedback.state_id, 0.0)
    next_value = 0.0 if feedback.terminal else state.motivation.state_values.get(feedback.next_state_id, 0.0)
    delta = feedback.reward + config.discount_factor * next_value - current
    updated = current + config.reward_learning_rate * delta
    state.motivation.state_values[feedback.state_id] = updated
    old_action = state.motivation.action_values.get(feedback.action_id, 0.0)
    state.motivation.action_values[feedback.action_id] = (
        old_action + config.reward_learning_rate * (feedback.reward - old_action)
    )
    state.motivation.expected_reward = updated
    return delta, {
        "feedback_available": 1.0, "expected_value": current,
        "next_state_value": next_value, "observed_reward": feedback.reward,
        "updated_value": updated,
        "updated_action_value": state.motivation.action_values[feedback.action_id],
    }


def allocate_effort(
    experience: LearningExperience,
    state: LearningState,
    control_capacity: float,
    expected_value: float,
    config: EmotionLearningConfig,
) -> EffortDecision:
    """Choose a bounded effort level by maximizing expected progress minus effort and fatigue costs."""
    candidates = (0.0, 0.25, 0.5, 0.75, 1.0)
    best: tuple[float, float, float, float] | None = None
    cost_weight = config.effort_cost_weight + config.fatigue_cost_weight * state.arousal.fatigue
    for effort in candidates:
        progress = clamp(
            effort * control_capacity * (0.45 + 0.55 * state.motivation.goal_strength)
            * (1.0 - 0.35 * experience.task_demands)
        )
        cost = clamp(0.25 * effort * effort * cost_weight
                      + effort * 0.05 * experience.task_demands)
        net = expected_value * progress - cost
        if best is None or net > best[0]:
            best = (net, effort, progress, cost)
    assert best is not None
    net, effort, progress, cost = best
    continue_task = effort >= config.elaboration_threshold and net > 0.0
    strategy = "continue_elaborative_processing" if continue_task else (
        "change_strategy_or_seek_support" if expected_value > 0.0
        else "pause_or_disengage"
    )
    return EffortDecision(
        effort=effort, expected_progress=progress, estimated_cost=cost,
        net_utility=net, continue_task=continue_task,
        recommended_strategy=strategy,
    )


def elaborate_processing(
    experience: LearningExperience,
    trace: MemoryTrace,
    state: LearningState,
    effort: EffortDecision,
) -> MemoryTrace:
    """Add retrieval routes by connecting attended event features to prior concepts and episodes."""
    if not effort.continue_task:
        return trace
    previous_tags = set(state.concept_strengths)
    current_tags = set(trace.context_tags)
    links = sorted(previous_tags & current_tags)
    enriched_strengths = dict(trace.feature_strengths)
    for feature in experience.features:
        overlap = len(set(feature.tags) & previous_tags)
        if overlap:
            enriched_strengths[feature.feature_id] = clamp(
                enriched_strengths[feature.feature_id] + 0.08 * effort.effort * overlap
            )
    elaboration = tuple(sorted(set(trace.elaboration_links) | set(links)))
    for feature in experience.features:
        for tag in feature.tags:
            old = state.concept_strengths.get(tag, 0.0)
            state.concept_strengths[tag] = clamp(old + 0.05 * effort.effort * (1.0 - old))
    return replace(trace, feature_strengths=enriched_strengths, elaboration_links=elaboration)


def consolidate_memory_trace(
    trace: MemoryTrace,
    elapsed_hours: float,
    sleep_replay_support: float,
    config: EmotionLearningConfig,
) -> MemoryTrace:
    """Apply time-limited salience modulation and replay support during a separate consolidation phase."""
    if elapsed_hours < 0:
        raise ValueError("elapsed_hours must be non-negative")
    delay_kernel = exp(-elapsed_hours / config.consolidation_time_constant_hours)
    replay = clamp(sleep_replay_support)
    rate = config.consolidation_rate * delay_kernel * (0.50 + 0.50 * replay)
    modulation = 1.0 + 0.35 * trace.amygdala_modulation * delay_kernel
    consolidated = 1.0 - (1.0 - trace.encoding_strength) * exp(-rate * modulation)
    feature_strengths = {
        key: clamp(value + (1.0 - value) * rate * modulation * 0.25)
        for key, value in trace.feature_strengths.items()
    }
    return replace(
        trace, feature_strengths=feature_strengths,
        consolidation_strength=clamp(max(trace.consolidation_strength, consolidated)),
    )


def retrieve_memories(
    cues: Sequence[str],
    state: LearningState,
    temperature: float = 0.35,
    limit: int = 5,
    ground_truth_event_id: str | None = None,
    threshold: float = 0.05,
) -> tuple[RetrievedMemory, ...]:
    """Retrieve episode traces from cue overlap and stored strengths without equating confidence with accuracy."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    query = set(str(item).lower() for item in cues)
    if not query:
        return ()
    scored: list[tuple[float, MemoryTrace]] = []
    for raw in state.memory_traces:
        if isinstance(raw, MemoryTrace):
            trace = raw
        else:
            trace = MemoryTrace(**raw)
        trace_tokens = set(trace.context_tags) | set(trace.encoded_feature_ids)
        overlap = len(query & trace_tokens) / max(1, len(query | trace_tokens))
        feature_support = sum(
            strength for key, strength in trace.feature_strengths.items()
            if key.lower() in query
        ) / max(1, len(trace.feature_strengths))
        similarity = clamp(0.75 * overlap + 0.25 * feature_support)
        score = similarity * (0.50 + 0.50 * trace.consolidation_strength)
        if similarity >= threshold:
            scored.append((score, trace))
    scored.sort(key=lambda item: (-item[0], item[1].event_id))
    selected = scored[:max(0, limit)]
    if not selected:
        return ()
    distribution = softmax({trace.event_id: score for score, trace in selected}, temperature)
    results: list[RetrievedMemory] = []
    for score, trace in selected:
        recalled = tuple(
            key for key, strength in trace.feature_strengths.items()
            if strength >= 0.20
        )
        recalled_central = tuple(key for key in recalled if key in trace.central_feature_ids)
        recalled_peripheral = tuple(key for key in recalled if key in trace.peripheral_feature_ids)
        vividness = clamp(
            0.55 * trace.consolidation_strength
            + 0.25 * trace.amygdala_modulation
            + 0.20 * distribution[trace.event_id]
        )
        confidence = clamp(0.65 * vividness + 0.35 * distribution[trace.event_id])
        accuracy = None
        if ground_truth_event_id == trace.event_id:
            accuracy = len(set(recalled) & set(trace.encoded_feature_ids)) / max(
                1, len(trace.encoded_feature_ids)
            )
        results.append(RetrievedMemory(
            event_id=trace.event_id, similarity=clamp(score),
            vividness=vividness, confidence=confidence,
            recalled_feature_ids=recalled,
            recalled_central_ids=recalled_central,
            recalled_peripheral_ids=recalled_peripheral,
            accuracy=accuracy,
        ))
    return tuple(results)


def measure_attention_memory_tradeoffs(
    experience: LearningExperience,
    attention: AttentionAllocation,
    trace: MemoryTrace,
) -> dict[str, float]:
    """Quantify attention concentration, central/peripheral retention, and context completeness."""
    total = sum(attention.weights.values())
    probabilities = (
        {key: value / total for key, value in attention.weights.items()}
        if total > 0 else {}
    )
    entropy = -sum(value * log(value) for value in probabilities.values() if value > 0)
    normalized_entropy = entropy / log(len(probabilities)) if len(probabilities) > 1 else 0.0
    central = [trace.feature_strengths.get(item.feature_id, 0.0)
               for item in experience.features if item.role == "central"]
    peripheral = [trace.feature_strengths.get(item.feature_id, 0.0)
                  for item in experience.features if item.role == "peripheral"]
    context_features = [item for item in experience.features if item.role == "context"]
    context_stored = sum(
        trace.feature_strengths.get(item.feature_id, 0.0) >= 0.20
        for item in context_features
    )
    central_mean = sum(central) / max(1, len(central))
    peripheral_mean = sum(peripheral) / max(1, len(peripheral))
    completeness = context_stored / max(1, len(context_features))
    return {
        "attention_entropy": clamp(normalized_entropy),
        "attention_concentration": clamp(1.0 - normalized_entropy),
        "central_mean_strength": clamp(central_mean),
        "peripheral_mean_strength": clamp(peripheral_mean),
        "central_peripheral_ratio": clamp(
            central_mean / (peripheral_mean + 1e-8), 0.0, 10.0
        ),
        "context_completeness": clamp(completeness),
        "vividness_accuracy_distinction": 1.0,
    }


def reappraise_and_update_associations(
    experience: LearningExperience,
    appraisal: AffectiveAppraisal,
    prediction_error: float,
    state: LearningState,
    learning_rate: float,
) -> dict[str, float]:
    """Update cue associations from outcome evidence without rewriting the episodic event."""
    if not 0.0 <= learning_rate <= 1.0:
        raise ValueError("learning_rate must be in [0, 1]")
    revised_significance = clamp(appraisal.significance + learning_rate * abs(prediction_error) * 0.20)
    signed_error = clamp(prediction_error, -1.0, 1.0)
    updates: dict[str, float] = {}
    for feature in experience.features:
        for cue in (feature.feature_id, *feature.tags):
            old = state.associative_weights.get(cue, 0.5)
            revised = clamp(old + learning_rate * signed_error * max(0.05, feature.salience))
            state.associative_weights[cue] = revised
            updates[cue] = revised
    state.appraisal_history.append({
        "event_id": experience.event_id,
        "original_significance": appraisal.significance,
        "revised_significance": revised_significance,
        "prediction_error": prediction_error,
        "valence": appraisal.valence,
    })
    return {
        "revised_significance": revised_significance,
        "prediction_error": prediction_error,
        "association_count_updated": float(len(updates)),
    }


def update_generalized_knowledge(
    experience: LearningExperience,
    trace: MemoryTrace,
    state: LearningState,
    learning_rate: float,
) -> dict[str, float]:
    """Update shared feature-tag strengths conservatively while retaining the episodic trace separately."""
    updates: dict[str, float] = {}
    evidence_weight = trace.encoding_strength * trace.consolidation_strength
    for feature in experience.features:
        for tag in feature.tags:
            old = state.concept_strengths.get(tag, 0.0)
            target = trace.feature_strengths.get(feature.feature_id, 0.0) * evidence_weight
            revised = clamp(old + learning_rate * (target - old))
            state.concept_strengths[tag] = revised
            updates[tag] = revised
    return updates
