"""Functional computations for Point 14 social interaction and conversational learning."""
from __future__ import annotations

from math import exp, log
import re
from typing import Any, Mapping, Sequence

from .contracts import (
    LearnerSocialState, PartnerModel, ReferentCandidate, SocialEpisodeMemory,
    SocialInteractionEvent, SocialLearningConfig, SocialSignal,
)


_TOKEN_PATTERN = re.compile(r"[\w'-]+", re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """Normalize text into lowercase lexical tokens for simple model-level matching."""
    return tuple(token.lower() for token in _TOKEN_PATTERN.findall(text or ""))


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Bound a finite model value to an inclusive interval."""
    if lower > upper:
        raise ValueError("lower must not exceed upper")
    return min(upper, max(lower, float(value)))


def sigmoid(value: float) -> float:
    """Compute a numerically stable logistic transform."""
    value = float(value)
    if value >= 0:
        z = exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = exp(max(value, -700.0))
    return z / (1.0 + z)


def softmax(scores: Mapping[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Normalize finite scores with a numerically stable softmax."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not scores:
        return {}
    peak = max(float(value) for value in scores.values())
    values = {str(key): exp((float(value) - peak) / temperature)
              for key, value in scores.items()}
    total = sum(values.values())
    if not total > 0 or not total < float("inf"):
        raise ValueError("scores must yield finite positive normalization")
    return {key: value / total for key, value in values.items()}


def shannon_entropy(probabilities: Mapping[str, float]) -> float:
    """Compute Shannon entropy in nats for a normalized or unnormalized non-negative map."""
    positive = {key: max(0.0, float(value)) for key, value in probabilities.items()}
    total = sum(positive.values())
    if total <= 0:
        return 0.0
    normalized = [value / total for value in positive.values() if value > 0]
    return -sum(value * log(value) for value in normalized)


def kl_divergence(
    learner_belief: Mapping[str, float],
    partner_belief: Mapping[str, float],
    epsilon: float = 1e-8,
) -> float:
    """Compute a smoothed KL divergence between learner and partner categorical beliefs."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    keys = set(learner_belief) | set(partner_belief)
    if not keys:
        return 0.0
    left_total = sum(max(0.0, float(learner_belief.get(key, 0.0))) for key in keys)
    right_total = sum(max(0.0, float(partner_belief.get(key, 0.0))) for key in keys)
    if left_total <= 0 or right_total <= 0:
        return 0.0
    result = 0.0
    for key in keys:
        p = max(epsilon, float(learner_belief.get(key, 0.0)) / left_total)
        q = max(epsilon, float(partner_belief.get(key, 0.0)) / right_total)
        result += p * log(p / q)
    return max(0.0, result)


def get_or_create_partner(
    state: LearnerSocialState,
    partner_id: str,
) -> PartnerModel:
    """Retrieve or initialize the partner-specific probabilistic knowledge and goal model."""
    model = state.partner_models.get(partner_id)
    if model is None:
        model = PartnerModel(partner_id=partner_id)
        state.partner_models[partner_id] = model
    return model


def estimate_social_attention(
    signals: Sequence[SocialSignal],
    learning_goal: str,
    config: SocialLearningConfig,
) -> dict[str, Any]:
    """Prioritize socially relevant cues using directedness, clarity, novelty, and distraction."""
    goal_tokens = set(tokenize(learning_goal))
    priorities: dict[str, float] = {}
    for signal in signals:
        content_tokens = set(tokenize(signal.content))
        goal_fit = (
            len(content_tokens & goal_tokens) / max(1, len(goal_tokens))
            if goal_tokens else 0.5
        )
        distraction = clamp(float(signal.attributes.get("distraction", 0.0)))
        novelty = clamp(float(signal.attributes.get("novelty", signal.salience)))
        directed = signal.directed_to_learner
        clarity = signal.reliability
        priorities[signal.signal_id] = (
            0.28 * goal_fit + 0.25 * directed + 0.22 * clarity
            + 0.15 * novelty + 0.10 * signal.salience - 0.30 * distraction
        )
    distribution = softmax(priorities, config.attention_temperature)
    ranked = sorted(distribution, key=lambda key: (-distribution[key], key))
    return {
        "priority_scores": priorities,
        "attention_weights": distribution,
        "selected_signal_ids": ranked[:max(1, min(5, len(ranked)))],
        "attention_entropy": shannon_entropy(distribution),
        "goal_tokens": sorted(goal_tokens),
    }


def infer_partner_focus(
    signals: Sequence[SocialSignal],
    attention: Mapping[str, Any],
    explicit_hint: str = "",
) -> tuple[str, float, str]:
    """Infer the partner's likely referent focus from gaze, gesture, and speech evidence."""
    weights = attention.get("attention_weights", {})
    scores: dict[str, float] = {}
    evidence_ids: dict[str, list[str]] = {}
    for signal in signals:
        if not signal.target_id:
            continue
        if signal.modality not in {"gaze", "gesture", "speech", "action"}:
            continue
        weight = float(weights.get(signal.signal_id, 0.0))
        modality_weight = {
            "gaze": 1.0, "gesture": 1.15, "speech": 0.9, "action": 1.1,
        }[signal.modality]
        score = weight * signal.reliability * modality_weight
        scores[signal.target_id] = scores.get(signal.target_id, 0.0) + score
        evidence_ids.setdefault(signal.target_id, []).append(signal.signal_id)
    if explicit_hint:
        scores[explicit_hint] = scores.get(explicit_hint, 0.0) + 0.18
        evidence_ids.setdefault(explicit_hint, []).append("explicit_focus_hint")
    if not scores:
        return "", 0.0, "no_partner_focus_evidence"
    posterior = softmax(scores, 0.35)
    focus = max(posterior, key=posterior.get)
    return focus, posterior[focus], ",".join(evidence_ids.get(focus, ()))


def infer_joint_attention(
    event: SocialInteractionEvent,
    attention: Mapping[str, Any],
    config: SocialLearningConfig,
) -> dict[str, Any]:
    """Estimate shared attention while retaining ambiguity and confirmation evidence."""
    partner_focus, partner_confidence, focus_source = infer_partner_focus(
        event.signals, attention, event.partner_focus_hint
    )
    learner_focus = event.learner_focus
    same_focus = bool(learner_focus and partner_focus and learner_focus == partner_focus)
    direct_match = 1.0 if same_focus else 0.0
    corroboration = sum(
        signal.reliability * float(attention.get("attention_weights", {}).get(signal.signal_id, 0.0))
        for signal in event.signals
        if signal.target_id == partner_focus and signal.modality in {"gaze", "gesture", "action"}
    )
    contingent = clamp(corroboration)
    explicit_confirmations = sum(
        1 for item in event.feedback if item.kind in {"confirmation", "recast"}
    )
    confirmation = clamp(explicit_confirmations / 2.0)
    confidence = sigmoid(
        2.8 * direct_match + 1.4 * partner_confidence
        + 1.0 * contingent + 0.8 * confirmation - 2.0
    )
    if not learner_focus or not partner_focus:
        status = "unknown"
    elif same_focus and confidence >= config.joint_attention_threshold:
        status = "coordinated"
    else:
        status = "uncertain_or_misaligned"
    return {
        "learner_focus": learner_focus,
        "partner_focus": partner_focus,
        "partner_focus_confidence": partner_confidence,
        "shared_attention_confidence": confidence,
        "status": status,
        "evidence_source": focus_source,
        "contingency_score": contingent,
        "confirmation_score": confirmation,
        "candidate_referents": [item.referent_id for item in event.referents],
        "clarification_needed": status != "coordinated",
    }


def _intention_cue_scores(utterance: str, signals: Sequence[SocialSignal]) -> dict[str, float]:
    """Create transparent lexical/action cue scores for candidate communicative intentions."""
    tokens = set(tokenize(utterance + " " + " ".join(item.content for item in signals)))
    score = {
        "naming": 0.15, "requesting": 0.10, "correcting": 0.10,
        "informing": 0.20, "joking": 0.05, "questioning": 0.10,
        "warning": 0.05, "directing_attention": 0.10,
    }
    lexicons = {
        "naming": {"called", "name", "this", "that", "means", "known"},
        "requesting": {"please", "give", "want", "bring", "can", "could", "help"},
        "correcting": {"no", "not", "wrong", "actually", "instead", "correction"},
        "informing": {"because", "therefore", "is", "are", "means", "fact"},
        "joking": {"joke", "funny", "kidding", "haha", "seriously"},
        "questioning": {"what", "why", "where", "when", "who", "how"},
        "warning": {"careful", "danger", "warning", "stop", "avoid"},
        "directing_attention": {"look", "see", "watch", "here", "there", "notice"},
    }
    for intention, terms in lexicons.items():
        score[intention] += 0.20 * len(tokens & terms)
    if "?" in utterance:
        score["questioning"] += 0.55
    if any(signal.modality == "gesture" for signal in signals):
        score["directing_attention"] += 0.10
    if any(signal.modality == "action" for signal in signals):
        score["informing"] += 0.10
    return score


def infer_communicative_intention(
    event: SocialInteractionEvent,
    partner: PartnerModel,
    config: SocialLearningConfig,
    learner_knowledge: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Infer partner communicative goals from utterance, social signals, and conversational context."""
    scores = _intention_cue_scores(event.utterance, event.signals)
    candidates = event.candidate_intentions or tuple(scores)
    candidates = tuple(dict.fromkeys(candidates))
    filtered = {key: scores.get(key, 0.05) for key in candidates}
    for goal, prior in partner.goal_beliefs.items():
        if goal in filtered:
            filtered[goal] += 0.35 * prior
    if event.intention_hint and event.intention_hint in filtered:
        filtered[event.intention_hint] += 0.45
    posterior = softmax(filtered, config.intention_temperature)
    top = max(posterior, key=posterior.get) if posterior else ""
    uncertainty = shannon_entropy(posterior)
    partner_knowledge_gap = {
        key: 1.0 - value for key, value in partner.knowledge_state.items()
    }
    belief_distance = kl_divergence(
        learner_knowledge or {}, partner.knowledge_state
    )
    return {
        "candidate_intentions": posterior,
        "most_likely_intention": top,
        "uncertainty_nats": uncertainty,
        "partner_knowledge_gap": partner_knowledge_gap,
        "partner_belief_mismatch_proxy": belief_distance,
        "is_committed": bool(top and posterior[top] >= 0.65 and uncertainty <= 1.2),
        "clarification_needed": not bool(top and posterior[top] >= 0.55),
    }


def integrate_multimodal_signals(
    event: SocialInteractionEvent,
    attention: Mapping[str, Any],
    config: SocialLearningConfig,
) -> dict[str, Any]:
    """Integrate aligned modalities by reliability-weighted precision while preserving provenance."""
    weights: dict[str, float] = {}
    modality_totals: dict[str, float] = {}
    terms_by_modality: dict[str, set[str]] = {}
    timelines: list[dict[str, Any]] = []
    for signal in event.signals:
        attention_weight = float(attention.get("attention_weights", {}).get(signal.signal_id, 0.0))
        variance = max(1e-4, 1.0 - signal.reliability)
        precision = (1.0 / variance) * max(0.05, attention_weight) * signal.salience
        weights[signal.signal_id] = precision
        modality_totals[signal.modality] = modality_totals.get(signal.modality, 0.0) + precision
        terms_by_modality.setdefault(signal.modality, set()).update(tokenize(signal.content))
        timelines.append({
            "signal_id": signal.signal_id,
            "modality": signal.modality,
            "timestamp": signal.timestamp,
            "target_id": signal.target_id,
            "reliability": signal.reliability,
            "provenance": signal.provenance,
        })
    total = sum(modality_totals.values())
    modality_weights = {
        key: value / total for key, value in modality_totals.items()
    } if total else {}
    sorted_timeline = sorted(timelines, key=lambda row: (row["timestamp"], row["signal_id"]))
    aligned_groups: list[list[str]] = []
    for row in sorted_timeline:
        if not aligned_groups:
            aligned_groups.append([row["signal_id"]])
            last_time = row["timestamp"]
        elif row["timestamp"] - last_time <= config.cue_window_seconds:
            aligned_groups[-1].append(row["signal_id"])
            last_time = row["timestamp"]
        else:
            aligned_groups.append([row["signal_id"]])
            last_time = row["timestamp"]
    modality_conflicts: list[dict[str, Any]] = []
    target_by_modality: dict[str, set[str]] = {}
    for signal in event.signals:
        if signal.target_id:
            target_by_modality.setdefault(signal.modality, set()).add(signal.target_id)
    targets = {target for values in target_by_modality.values() for target in values}
    if len(targets) > 1:
        modality_conflicts.append({
            "kind": "cross_modal_target_conflict",
            "targets_by_modality": {k: sorted(v) for k, v in target_by_modality.items()},
            "resolution": "retain competing target hypotheses and weight by reliability",
        })
    content = sorted(set().union(*terms_by_modality.values())) if terms_by_modality else []
    return {
        "modality_weights": modality_weights,
        "signal_precisions": weights,
        "tokens_by_modality": {key: sorted(value) for key, value in terms_by_modality.items()},
        "integrated_tokens": content,
        "temporal_alignment_groups": aligned_groups,
        "modality_conflicts": modality_conflicts,
        "source_preserved": True,
    }


def ground_word_to_shared_referent(
    event: SocialInteractionEvent,
    state: LearnerSocialState,
    joint_attention: Mapping[str, Any],
    intention: Mapping[str, Any],
    config: SocialLearningConfig,
) -> dict[str, Any]:
    """Infer a word-to-referent mapping using cue reliability, shared focus, prior association, and intent."""
    word = event.target_word.lower().strip()
    if not word or not event.referents:
        return {
            "target_word": word, "candidate_posterior": {},
            "selected_referent": None, "confidence": 0.0,
            "stored_association_key": None, "clarification_needed": True,
        }
    intention_name = str(intention.get("most_likely_intention", ""))
    focus = str(joint_attention.get("partner_focus", ""))
    raw_scores: dict[str, float] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for referent in event.referents:
        association_key = word + "::" + referent.referent_id
        prior = state.word_referent_associations.get(association_key, 0.5)
        lexical = 1.0 if word in set(tokenize(referent.description)) | set(referent.features) else 0.0
        target_cues = [
            signal for signal in event.signals
            if signal.target_id == referent.referent_id
        ]
        cue_fit = sum(item.reliability * item.salience for item in target_cues)
        focus_fit = 1.0 if focus == referent.referent_id else 0.0
        context_fit = referent.context_fit
        score = (
            0.28 * prior + 0.17 * lexical + 0.24 * clamp(cue_fit)
            + 0.22 * focus_fit + 0.09 * context_fit
        )
        if intention_name == "naming":
            score += 0.08 * lexical
        if intention_name == "directing_attention":
            score += 0.08 * focus_fit
        if event.candidate_mappings:
            score += 0.20 * event.candidate_mappings.get(referent.referent_id, 0.0)
        raw_scores[referent.referent_id] = score
        evidence[referent.referent_id] = {
            "prior_association": prior, "lexical_fit": lexical,
            "social_cue_fit": clamp(cue_fit), "joint_focus_fit": focus_fit,
            "context_fit": context_fit,
            "signal_ids": [item.signal_id for item in target_cues],
        }
    posterior = softmax(raw_scores, config.referent_temperature)
    selected = max(posterior, key=posterior.get)
    confidence = posterior[selected]
    association_key = word + "::" + selected
    association_before = state.word_referent_associations.get(association_key, 0.5)
    quality = (
        float(joint_attention.get("shared_attention_confidence", 0.0))
        * float(intention.get("candidate_intentions", {}).get("naming", 0.5))
    )
    observed = confidence
    updated = clamp(
        association_before + config.association_learning_rate * quality * (observed - association_before)
    )
    state.word_referent_associations[association_key] = updated
    state.semantic_knowledge[association_key] = clamp(
        0.7 * state.semantic_knowledge.get(association_key, 0.0) + 0.3 * confidence
    )
    committed = (
        confidence >= 0.62
        and joint_attention.get("status") == "coordinated"
    )
    if not committed:
        state.unresolved_questions.append(
            "Clarify whether '" + word + "' refers to " + ", ".join(posterior)
        )
    return {
        "target_word": word,
        "candidate_posterior": posterior,
        "selected_referent": selected if committed else None,
        "provisional_best_referent": selected,
        "confidence": confidence,
        "evidence_by_referent": evidence,
        "association_before": association_before,
        "association_after": updated,
        "interaction_quality": quality,
        "stored_association_key": association_key,
        "clarification_needed": not committed,
    }


def predict_turn_state(
    event: SocialInteractionEvent,
    attention: Mapping[str, Any],
    config: SocialLearningConfig,
) -> dict[str, Any]:
    """Estimate continue/yield/repair turn states from syntax, timing, gaze, and gesture cues."""
    utterance = event.utterance.strip()
    tokens = tokenize(utterance)
    punctuation_yield = 0.75 if utterance.endswith(("?", "!", ".")) else 0.20
    question_cue = 0.18 if utterance.endswith("?") or (
        tokens and tokens[0] in {"what", "why", "where", "when", "who", "how", "can", "could"}
    ) else 0.0
    pause = max(
        [float(signal.attributes.get("pause_seconds", 0.0))
         for signal in event.signals if signal.modality == "timing"] or [0.0]
    )
    gaze_yield = max(
        [signal.reliability for signal in event.signals
         if signal.modality == "gaze" and signal.attributes.get("turn_yielding")] or [0.0]
    )
    gesture_yield = max(
        [signal.reliability for signal in event.signals
         if signal.modality == "gesture" and signal.attributes.get("inviting_response")] or [0.0]
    )
    repair_cue = any(item.kind in {"clarification", "informational_correction"} for item in event.feedback)
    yield_score = (
        0.36 * punctuation_yield + question_cue + 0.24 * clamp(pause / 1.2)
        + 0.22 * gaze_yield + 0.18 * gesture_yield
    )
    scores = {
        "continue": 0.65 - 0.45 * yield_score,
        "yield": yield_score,
        "repair": 0.75 if repair_cue or event.partner_response and event.predicted_partner_response
        and event.partner_response != event.predicted_partner_response else 0.05,
    }
    posterior = softmax(scores, 0.45)
    selected = max(posterior, key=posterior.get)
    ready = bool(event.learner_attempt or event.metadata.get("response_ready", False))
    should_respond = posterior.get("yield", 0.0) >= config.turn_yield_threshold and ready
    return {
        "state_posterior": posterior,
        "predicted_state": selected,
        "yield_probability": posterior.get("yield", 0.0),
        "response_ready": ready,
        "learner_should_respond": should_respond,
        "pause_seconds": pause,
        "turn_hazard_proxy": clamp(pause / (pause + 0.8)) if pause else 0.0,
        "repair_cue": repair_cue,
    }


def design_audience_response(
    event: SocialInteractionEvent,
    partner: PartnerModel,
    intention: Mapping[str, Any],
) -> dict[str, Any]:
    """Select response complexity from partner knowledge and communicative information needs."""
    goal = event.learning_goal or str(intention.get("most_likely_intention", "informing"))
    knowledge = partner.knowledge_state.get(event.target_concept or goal, 0.35)
    missing = 1.0 - knowledge
    if missing >= 0.65:
        detail = "high"
        vocabulary = "plain_language_with_examples"
        response_plan = "define_terms_then_give_concrete_example_and_check_understanding"
    elif missing >= 0.30:
        detail = "moderate"
        vocabulary = "shared_terms_with_brief_explanation"
        response_plan = "explain_key_relation_and_offer_one_example"
    else:
        detail = "concise"
        vocabulary = "partner_familiar_terminology"
        response_plan = "state_result_and_invite_advanced_question"
    return {
        "goal": goal,
        "estimated_partner_knowledge": knowledge,
        "estimated_information_gap": missing,
        "detail_level": detail,
        "vocabulary_strategy": vocabulary,
        "response_plan": response_plan,
        "ambiguity_risk": clamp(0.25 + 0.55 * missing - 0.25 * partner.reliability),
        "monitor_for_understanding": True,
    }


def detect_conversational_mismatch(
    event: SocialInteractionEvent,
    intention: Mapping[str, Any],
    config: SocialLearningConfig,
) -> dict[str, Any]:
    """Detect misunderstanding by comparing expected and observed partner behavior and feedback."""
    predicted = set(tokenize(event.predicted_partner_response))
    observed = set(tokenize(event.partner_response))
    if predicted or observed:
        overlap = len(predicted & observed) / max(1, len(predicted | observed))
        response_error = 1.0 - overlap
    else:
        response_error = 0.0
    negative_feedback = [
        item for item in event.feedback
        if item.kind in {"informational_correction", "clarification"}
    ]
    confirmation_feedback = [
        item for item in event.feedback
        if item.kind == "confirmation"
    ]
    feedback_error = max((item.reliability for item in negative_feedback), default=0.0)
    feedback_support = max((item.reliability for item in confirmation_feedback), default=0.0)
    likelihood = clamp(
        0.55 * response_error + 0.35 * feedback_error - 0.20 * feedback_support
    )
    mismatch = (
        likelihood >= config.repair_probability_threshold
        or response_error * 3.0 >= config.repair_error_threshold
    )
    labels: list[str] = []
    if negative_feedback:
        for item in negative_feedback:
            labels.append("referential" if item.target_key else "lexical_or_pragmatic")
    if response_error > 0.35:
        labels.append("response_prediction_mismatch")
    if not labels and mismatch:
        labels.append("unclassified")
    if mismatch:
        action = "ask_clarification_or_offer_alternative_interpretation"
    elif feedback_support > 0.5:
        action = "retain_provisional_interpretation_with_confirmation"
    else:
        action = "continue_monitoring"
    return {
        "mismatch_detected": mismatch,
        "mismatch_probability": likelihood,
        "response_error": response_error,
        "feedback_error": feedback_error,
        "confirmation_support": feedback_support,
        "mismatch_types": sorted(set(labels)),
        "repair_action": action,
        "alternatives_reopened": mismatch,
        "current_intention_uncertainty": intention.get("uncertainty_nats", 0.0),
    }


def learn_from_social_feedback(
    event: SocialInteractionEvent,
    state: LearnerSocialState,
    partner: PartnerModel,
    config: SocialLearningConfig,
) -> dict[str, Any]:
    """Update factual knowledge, action values, and social reward in separate learning channels."""
    corrections: list[dict[str, Any]] = []
    rewards: list[dict[str, Any]] = []
    for feedback in event.feedback:
        if feedback.kind in {"informational_correction", "recast", "confirmation", "clarification"}:
            key = feedback.target_key or event.target_concept or event.target_word
            if key and feedback.factual_value is not None:
                before = state.semantic_knowledge.get(key, 0.35)
                # The scalar confidence is moved only in proportion to information reliability.
                target_confidence = 1.0 if feedback.confirms is not False else 0.0
                strength = config.knowledge_update_rate * feedback.reliability * partner.reliability
                after = clamp(before + strength * (target_confidence - before))
                state.semantic_knowledge[key] = after
                state.knowledge_claims[key] = feedback.factual_value
                state.source_ledger.append({
                    "key": key, "value": feedback.factual_value,
                    "source_id": feedback.source_id or partner.partner_id,
                    "reliability": feedback.reliability,
                    "evidence_kind": feedback.kind,
                    "event_id": event.event_id,
                    "provenance": feedback.provenance,
                })
                corrections.append({
                    "key": key, "confidence_before": before,
                    "confidence_after": after, "factual_value": feedback.factual_value,
                    "source_id": feedback.source_id or partner.partner_id,
                })
        if feedback.kind in {"outcome", "social_affective"}:
            action = event.learning_goal or event.target_word or "social_interaction"
            before = state.action_values.get(action, 0.5)
            delta = feedback.reward - before
            after = clamp(before + config.reward_learning_rate * delta)
            state.action_values[action] = after
            rewards.append({
                "action": action, "reward": feedback.reward,
                "prediction_error": delta, "value_before": before, "value_after": after,
                "feedback_kind": feedback.kind,
            })
    if event.outcome_reward:
        action = event.learning_goal or "social_interaction"
        before = state.action_values.get(action, 0.5)
        delta = event.outcome_reward - before
        state.action_values[action] = clamp(before + config.reward_learning_rate * delta)
        rewards.append({
            "action": action, "reward": event.outcome_reward,
            "prediction_error": delta, "value_before": before,
            "value_after": state.action_values[action], "feedback_kind": "observed_outcome",
        })
    return {
        "factual_corrections": corrections,
        "social_and_outcome_rewards": rewards,
        "factual_confidence_is_separate_from_approval": True,
        "source_ledger_entries": len(state.source_ledger),
    }


def select_scaffold(
    event: SocialInteractionEvent,
    state: LearnerSocialState,
    config: SocialLearningConfig,
) -> dict[str, Any]:
    """Adapt prompt strength to the estimated competence–difficulty gap and fade support after success."""
    skill = event.learning_goal or event.target_concept or "general_learning"
    competence = state.skill_competence.get(skill, 0.35)
    if event.competence_observation is not None:
        competence = clamp(
            competence + 0.75 * (event.competence_observation - competence)
        )
        state.skill_competence[skill] = competence
    elif event.learner_attempt:
        attempt_quality = clamp(float(event.metadata.get("attempt_quality", 0.5)))
        competence = clamp(competence + 0.10 * (attempt_quality - competence))
        state.skill_competence[skill] = competence
    gap = event.difficulty - competence
    support = clamp(
        config.scaffold_gain * (0.5 + gap),
        config.scaffold_minimum, config.scaffold_maximum,
    )
    if support >= 0.70:
        kind = "worked_example_or_demonstration"
    elif support >= 0.42:
        kind = "targeted_hint_or_guiding_question"
    else:
        kind = "retrieval_prompt_or_independent_attempt"
    if competence >= 0.75 and event.competence_observation is not None:
        kind = "fade_support_and_test_transfer"
        support = config.scaffold_minimum
    return {
        "skill": skill,
        "estimated_competence": competence,
        "task_difficulty": event.difficulty,
        "competence_difficulty_gap": gap,
        "support_level": support,
        "scaffold_type": kind,
        "minimum_help_principle": True,
        "transfer_test_recommended": competence >= 0.65,
    }


def evaluate_explanation(
    event: SocialInteractionEvent,
    state: LearnerSocialState,
    config: SocialLearningConfig,
) -> dict[str, Any]:
    """Score an explanation for weighted term/relation coverage and update retrieval-strength estimates."""
    text = event.learner_attempt
    tokens = set(tokenize(text))
    required_terms = set(event.required_explanation_terms)
    term_hits = sorted(tokens & required_terms)
    term_coverage = len(term_hits) / max(1, len(required_terms)) if required_terms else (
        1.0 if text else 0.0
    )
    relation_hits = [
        relation for relation in event.required_relations
        if all(token in tokens for token in tokenize(relation))
    ]
    relation_coverage = len(relation_hits) / max(1, len(event.required_relations)) if event.required_relations else (
        1.0 if text else 0.0
    )
    completeness = clamp(0.65 * term_coverage + 0.35 * relation_coverage)
    missing_terms = sorted(required_terms - set(term_hits))
    missing_relations = sorted(set(event.required_relations) - set(relation_hits))
    key = event.target_concept or event.target_word or event.learning_goal or "general"
    before = state.retrieval_strengths.get(key, 0.0)
    retrieval_success = completeness
    after = clamp(before + config.explanation_retrieval_rate * retrieval_success * (1.0 - before))
    state.retrieval_strengths[key] = after
    state.explanation_strength[key] = clamp(
        state.explanation_strength.get(key, 0.0) + 0.15 * completeness
    )
    gaps: list[str] = []
    gaps.extend("missing_term:" + item for item in missing_terms)
    gaps.extend("missing_relation:" + item for item in missing_relations)
    return {
        "explanation_text": text,
        "term_coverage": term_coverage,
        "relation_coverage": relation_coverage,
        "completeness": completeness,
        "covered_terms": term_hits,
        "covered_relations": relation_hits,
        "knowledge_gaps": gaps,
        "retrieval_strength_before": before,
        "retrieval_strength_after": after,
        "correction_or_evidence_check_needed": completeness < 0.8,
    }


def compare_perspectives(
    event: SocialInteractionEvent,
    state: LearnerSocialState,
    partner: PartnerModel,
    config: SocialLearningConfig,
) -> dict[str, Any]:
    """Integrate partner perspective by reliability while preserving source dependence and disagreement."""
    updates: dict[str, Any] = {}
    disagreements: list[dict[str, Any]] = []
    for key, partner_value in event.partner_claims.items():
        learner_value = state.semantic_knowledge.get(key, 0.5)
        evidence_quality = event.partner_evidence.get(key, partner.reliability)
        rho = clamp(evidence_quality * partner.reliability)
        revised = clamp(
            learner_value + config.knowledge_update_rate * rho * (partner_value - learner_value)
        )
        state.semantic_knowledge[key] = revised
        updates[key] = {
            "learner_before": learner_value, "partner_claim": partner_value,
            "source_reliability": rho, "learner_after": revised,
        }
        if abs(learner_value - partner_value) >= 0.20:
            disagreements.append({
                "key": key, "learner_belief": learner_value,
                "partner_belief": partner_value, "evidence_quality": evidence_quality,
                "agreement_is_not_truth": True,
            })
        state.source_ledger.append({
            "key": key, "value": partner_value, "source_id": partner.partner_id,
            "reliability": rho, "evidence_kind": "partner_perspective",
            "event_id": event.event_id, "provenance": "reported_partner_claim",
            "independence_assumed": False,
        })
    return {
        "updates": updates,
        "disagreements": disagreements,
        "agreement_is_not_truth": True,
        "source_dependence_is_retained": True,
    }


def build_narrative_model(
    event: SocialInteractionEvent,
    state: LearnerSocialState,
    config: SocialLearningConfig,
) -> dict[str, Any]:
    """Build a temporal-causal event graph and maintain agent-specific knowledge and goals."""
    added: list[str] = []
    prediction_records: list[dict[str, Any]] = []
    for story_event in event.narrative_events:
        node = story_event.to_dict()
        state.narrative_graph[story_event.event_id] = node
        added.append(story_event.event_id)
        if story_event.agent_id:
            agent = state.narrative_agent_states.setdefault(
                story_event.agent_id, {"goals": [], "known_facts": [], "actions": []}
            )
            if story_event.goal and story_event.goal not in agent["goals"]:
                agent["goals"].append(story_event.goal)
            if story_event.action:
                agent["actions"].append(story_event.action)
            for fact_key in story_event.facts:
                recipients = story_event.reveals_to or (story_event.agent_id,)
                for recipient_id in recipients:
                    recipient = state.narrative_agent_states.setdefault(
                        recipient_id, {"goals": [], "known_facts": [], "actions": []}
                    )
                    if fact_key not in recipient["known_facts"]:
                        recipient["known_facts"].append(fact_key)
        if story_event.causal_parents:
            missing_parents = [
                parent for parent in story_event.causal_parents
                if parent not in state.narrative_graph
            ]
        else:
            missing_parents = []
        if missing_parents:
            prediction_records.append({
                "event_id": story_event.event_id,
                "missing_causal_parents": missing_parents,
                "coherence_warning": True,
            })
        else:
            prediction_records.append({
                "event_id": story_event.event_id,
                "predicted_next_event": (
                    "goal_consistent_action_by_" + story_event.agent_id
                    if story_event.agent_id and story_event.goal else "next_event_unspecified"
                ),
                "surprise_proxy": 0.0,
            })
    return {
        "added_event_ids": added,
        "narrative_graph": dict(state.narrative_graph),
        "agent_specific_states": dict(state.narrative_agent_states),
        "prediction_records": prediction_records,
        "graph_node_count": len(state.narrative_graph),
        "graph_edge_count": sum(
            len(item.get("causal_parents", ())) for item in state.narrative_graph.values()
        ),
        "surprise_threshold": config.narrative_surprise_threshold,
    }


def retrieve_social_memories(
    cues: Sequence[str],
    memories: Sequence[SocialEpisodeMemory],
    partner_id: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve social episodes by cue overlap while retaining partner/source information."""
    query = set(tokenize(" ".join(cues)))
    scored: list[tuple[float, Any]] = []
    for memory in memories:
        tokens = set(tokenize(memory.content)) | set(memory.context_tags) | set(memory.concept_keys)
        overlap = len(query & tokens) / max(1, len(query | tokens)) if query else 0.0
        partner_bonus = 0.05 if partner_id and memory.partner_id == partner_id else 0.0
        score = clamp(0.75 * overlap + 0.25 * memory.consolidation_strength + partner_bonus)
        if score > 0:
            scored.append((score, memory))
    scored.sort(key=lambda item: (-item[0], item[1].event_id))
    return [{
        "event_id": memory.event_id,
        "partner_id": memory.partner_id,
        "score": score,
        "content": memory.content,
        "source_claims": dict(memory.source_claims),
        "confidence": memory.confidence,
    } for score, memory in scored[:max(0, limit)]]



def create_social_memory(
    event: SocialInteractionEvent,
    state: LearnerSocialState,
    partner: PartnerModel,
    attention: Mapping[str, Any],
    intention: Mapping[str, Any],
    grounding: Mapping[str, Any],
    feedback_result: Mapping[str, Any],
    config: SocialLearningConfig,
) -> dict[str, Any]:
    """Bind learning content, partner source, context, and confidence into a social episode."""
    content = " ".join(part for part in (
        event.utterance, event.learner_attempt,
        " ".join(signal.content for signal in event.signals),
    ) if part)
    context_tags = sorted(
        set(tokenize(content)) | {str(k).lower() for k in event.context}
        | {event.learning_goal.lower()} if event.learning_goal else
        set(tokenize(content)) | {str(k).lower() for k in event.context}
    )
    joint_quality = float(attention.get("quality", 0.5))
    intention_confidence = max(
        intention.get("candidate_intentions", {}).values(), default=0.0
    )
    grounding_confidence = float(grounding.get("confidence", 0.0))
    feedback_quality = max(
        (row["confidence_after"] for row in feedback_result.get("factual_corrections", ())),
        default=0.0,
    )
    confidence = clamp(
        0.25 * joint_quality + 0.25 * intention_confidence
        + 0.25 * grounding_confidence + 0.25 * feedback_quality
    )
    priority = (
        0.25 * joint_quality + 0.20 * intention_confidence
        + 0.20 * max((signal.salience for signal in event.signals), default=0.0)
        + 0.20 * (1.0 - partner.reliability) + 0.15 * confidence
    )
    elapsed = event.elapsed_hours
    temporal_decay = exp(-elapsed / 24.0)
    consolidation = clamp(
        priority + config.consolidation_rate * temporal_decay
        * (0.5 + 0.5 * float(event.metadata.get("replay_support", 0.0)))
    )
    source_claims = {
        "partner_id": partner.partner_id,
        "communicative_intention": intention.get("most_likely_intention"),
        "feedback_ids": [item.feedback_id for item in event.feedback],
        "grounded_word": grounding.get("target_word"),
        "referent": grounding.get("selected_referent"),
    }
    memory = SocialEpisodeMemory(
        event_id=event.event_id,
        partner_id=partner.partner_id,
        learner_id=event.learner_id,
        content=content,
        source_claims=source_claims,
        context_tags=tuple(context_tags),
        modalities=tuple(sorted({item.modality for item in event.signals})),
        concept_keys=tuple(
            item for item in (event.target_word, event.target_concept, event.learning_goal)
            if item
        ),
        feedback_ids=tuple(item.feedback_id for item in event.feedback),
        confidence=confidence,
        source_reliability=partner.reliability,
        episodic_strength=clamp(priority),
        consolidation_strength=consolidation,
        timestamp=str(event.metadata.get("timestamp", "")),
    )
    prior = retrieve_social_memories(
        context_tags, state.episodic_memories, partner_id=partner.partner_id, limit=3
    )
    state.episodic_memories.append(memory)
    if len(state.episodic_memories) > config.memory_limit:
        state.episodic_memories = state.episodic_memories[-config.memory_limit:]
    return {
        "memory": memory.to_dict(),
        "retrieved_prior_episodes": prior,
        "priority_score": priority,
        "confidence": confidence,
        "consolidation_strength": consolidation,
        "source_binding_preserved": True,
    }


def update_partner_and_cross_partner_models(
    event: SocialInteractionEvent,
    state: LearnerSocialState,
    partner: PartnerModel,
    attention: Mapping[str, Any],
    intention: Mapping[str, Any],
    grounding: Mapping[str, Any],
    config: SocialLearningConfig,
) -> dict[str, Any]:
    """Update one partner cautiously and generalize only patterns observed across multiple partners."""
    prior_reliability = partner.reliability
    source_evidence = [
        item.reliability for item in event.feedback
        if item.kind in {
            "informational_correction", "recast", "confirmation",
            "clarification", "outcome",
        }
    ]
    observed_source_reliability = (
        max(source_evidence) if source_evidence
        else float(attention.get("quality", 0.5))
    )
    observed_reliability = clamp(
        0.70 * partner.reliability + 0.30 * observed_source_reliability
    )
    partner.reliability = clamp(
        prior_reliability + config.partner_model_learning_rate
        * (observed_reliability - prior_reliability)
    )
    partner.interaction_count += 1
    partner.last_seen = str(event.metadata.get("timestamp", ""))
    if event.learning_goal:
        for goal, probability in intention.get("candidate_intentions", {}).items():
            partner.goal_beliefs[goal] = clamp(
                partner.goal_beliefs.get(goal, 0.0)
                + config.partner_model_learning_rate
                * (probability - partner.goal_beliefs.get(goal, 0.0))
            )
    partner.attention_focus = str(grounding.get("selected_referent") or partner.attention_focus)
    partner.vocabulary.update(tokenize(event.utterance))
    word = event.target_word
    referent = grounding.get("selected_referent") or grounding.get("provisional_best_referent")
    cross_partner_result: dict[str, Any] = {}
    if word and referent:
        record = state.cross_partner_patterns.setdefault(
            word, {"partner_ids": [], "referent_counts": {}, "generalizable": False}
        )
        partners = set(record.get("partner_ids", []))
        partners.add(partner.partner_id)
        record["partner_ids"] = sorted(partners)
        counts = dict(record.get("referent_counts", {}))
        counts[referent] = counts.get(referent, 0) + 1
        record["referent_counts"] = counts
        total = sum(counts.values())
        best_ref = max(counts, key=counts.get)
        consistency = counts[best_ref] / max(1, total)
        generalizable = len(partners) >= 2 and consistency >= 0.70
        record["generalizable"] = generalizable
        record["best_referent"] = best_ref if generalizable else None
        record["consistency"] = consistency
        cross_partner_result = {
            "word": word, "partner_count": len(partners),
            "referent_counts": counts, "consistency": consistency,
            "generalizable": generalizable,
        }
    state.interaction_count += 1
    return {
        "partner_id": partner.partner_id,
        "reliability_before": prior_reliability,
        "reliability_after": partner.reliability,
        "interaction_count": partner.interaction_count,
        "estimated_vocabulary_size": len(partner.vocabulary),
        "goal_beliefs": dict(partner.goal_beliefs),
        "cross_partner_pattern": cross_partner_result,
        "generalization_requires_multiple_partners": True,
    }
