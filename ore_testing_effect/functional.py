"""Functional mathematical models for the 22-stage Point 04 testing-effect algorithm."""
from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import (
    Feedback, LearningTarget, RetrievalAttempt, TargetMemoryState, TestingEffectState,
)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Bound a finite computational signal to a closed interval."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("value must be finite")
    return min(high, max(low, number))


def sigmoid(value: float) -> float:
    """Map an unbounded activation or logit to a numerically stable probability."""
    x = float(value)
    if not math.isfinite(x):
        raise ValueError("value must be finite")
    if x >= 60.0:
        return 1.0
    if x <= -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def softmax(scores: Mapping[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Normalize competing scores into a stable categorical probability distribution."""
    if not scores:
        return {}
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    keys = list(scores)
    values = np.asarray([float(scores[key]) for key in keys], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    scaled = (values - np.max(values)) / temperature
    exponents = np.exp(np.clip(scaled, -60.0, 0.0))
    probabilities = exponents / max(float(exponents.sum()), 1e-12)
    return {key: float(probability) for key, probability in zip(keys, probabilities)}


def tokenize(text: str) -> set[str]:
    """Convert text into a normalized set of lexical features for approximate overlap."""
    return {token for token in re.findall(r"[a-zA-Z0-9]+", str(text).casefold()) if len(token) > 1}


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Calculate cosine alignment between cue and target vectors with a zero-norm guard."""
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        raise ValueError("vectors must have matching shapes")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return clamp(float(np.dot(a, b) / denominator), -1.0, 1.0)


def lexical_alignment(cue: str, target_text: str) -> float:
    """Estimate cue relevance from token overlap without claiming semantic understanding."""
    cue_tokens = tokenize(cue)
    target_tokens = tokenize(target_text)
    if not cue_tokens or not target_tokens:
        return 0.0
    return float(len(cue_tokens & target_tokens) / len(cue_tokens | target_tokens))


def _weight(mapping: Mapping[str, float], item: str) -> float:
    """Return an explicit item weight or a neutral unit default."""
    return float(mapping.get(item, 1.0))


def _weighted_ratio(values: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Compute a normalized weighted mean with safe behavior for an empty evidence set."""
    if not values:
        return 1.0
    denominator = sum(_weight(weights, key) for key in values)
    if denominator <= 1e-12:
        return float(np.mean(list(values.values())))
    return clamp(sum(_weight(weights, key) * value for key, value in values.items()) / denominator)


def weighted_coverage(
    target_units: Iterable[str], observed_units: Iterable[str],
    weights: Mapping[str, float],
) -> float:
    """Calculate importance-weighted coverage of retrieved concepts or relationships."""
    observed = set(observed_units)
    units = list(target_units)
    if not units:
        return 1.0
    total = sum(_weight(weights, item) for item in units)
    if total <= 1e-12:
        return sum(item in observed for item in units) / len(units)
    return clamp(sum(_weight(weights, item) for item in units if item in observed) / total)


def _infer_component_status(
    target: LearningTarget, attempt: RetrievalAttempt,
) -> dict[str, str]:
    """Use explicit rubric labels first and conservative token overlap only as a fallback."""
    statuses = {item: "missing" for item in target.components}
    for item in attempt.recalled_components:
        if item in statuses:
            statuses[item] = "correct"
    for item in attempt.inferred_components:
        if item in statuses and statuses[item] == "missing":
            statuses[item] = "uncertain"
    statuses.update({key: value for key, value in attempt.component_status.items()
                     if key in statuses})
    if not attempt.recalled_components and not attempt.component_status:
        response_tokens = tokenize(attempt.response)
        for item in target.components:
            unit_tokens = tokenize(item)
            if unit_tokens and len(unit_tokens & response_tokens) / len(unit_tokens) >= 0.5:
                statuses[item] = "correct"
    return statuses


def _infer_relation_status(
    target: LearningTarget, attempt: RetrievalAttempt,
) -> dict[str, str]:
    """Score relations from explicit evidence and leave unreported relations unresolved/missing."""
    statuses = {item: "missing" for item in target.relations}
    for item in attempt.recalled_relations:
        if item in statuses:
            statuses[item] = "correct"
    statuses.update({key: value for key, value in attempt.relation_status.items()
                     if key in statuses})
    if not attempt.recalled_relations and not attempt.relation_status:
        response_tokens = tokenize(attempt.response)
        for item in target.relations:
            unit_tokens = tokenize(item)
            if unit_tokens and len(unit_tokens & response_tokens) / len(unit_tokens) >= 0.65:
                statuses[item] = "correct"
    return statuses


def score_order(expected: Sequence[str], observed: Sequence[str]) -> float:
    """Calculate normalized pairwise order agreement for sequence or procedure retrieval."""
    if len(expected) < 2:
        return 1.0
    positions = {item: index for index, item in enumerate(observed)}
    required = [item for item in expected if item in positions]
    # Missing units are scored by coverage; order is evaluated only among retrieved units.
    if len(required) < 2:
        return 1.0
    total_pairs = len(required) * (len(required) - 1) / 2
    correct_pairs = 0
    for i, left in enumerate(required):
        for right in required[i + 1:]:
            if positions[left] < positions[right]:
                correct_pairs += 1
    return clamp(correct_pairs / total_pairs)


def assess_cue_quality(target: LearningTarget, attempt: RetrievalAttempt) -> dict[str, float]:
    """Balance cue relevance against answer leakage and ambiguity."""
    relevance = lexical_alignment(attempt.cue, target.prompt + " " + " ".join(target.components))
    if attempt.cue_type not in target.valid_cue_types and attempt.cue_type != "free_recall":
        relevance *= 0.5
    utility = clamp(0.55 * relevance + 0.45 * (1.0 - attempt.ambiguity)
                    - 0.70 * attempt.answer_leakage)
    return {"relevance": relevance, "answer_leakage": attempt.answer_leakage,
            "ambiguity": attempt.ambiguity, "cue_utility": utility}


def assess_test_quality(
    target: LearningTarget, attempt: RetrievalAttempt, cue: Mapping[str, float],
) -> dict[str, float]:
    """Estimate relevance, criterion validity, clarity, ambiguity, and answer leakage."""
    relevance = float(cue["relevance"])
    validity = target.criterion_reliability
    clarity = clamp(1.0 - float(cue["ambiguity"]))
    ambiguity = float(cue["ambiguity"])
    leakage = float(cue["answer_leakage"])
    score = clamp(0.30 * relevance + 0.30 * validity + 0.20 * clarity
                  - 0.10 * ambiguity - 0.10 * leakage)
    return {"relevance": relevance, "validity": validity, "criterion_clarity": clarity,
            "ambiguity": ambiguity, "answer_leakage": leakage, "test_quality": score,
            "criterion_trusted": validity >= 0.70}


def assess_attempt(target: LearningTarget, attempt: RetrievalAttempt) -> dict[str, Any]:
    """Compare a reconstructed response with content, relations, order, and conditions."""
    component_status = _infer_component_status(target, attempt)
    relation_status = _infer_relation_status(target, attempt)
    component_values = {key: 1.0 if value == "correct" else
                        0.5 if value == "uncertain" else 0.0
                        for key, value in component_status.items()}
    relation_values = {key: 1.0 if value == "correct" else
                       0.5 if value == "uncertain" else 0.0
                       for key, value in relation_status.items()}
    component_coverage = _weighted_ratio(component_values, target.component_weights)
    relation_coverage = _weighted_ratio(relation_values, target.relation_weights)
    relation_score = relation_coverage if target.relations else component_coverage
    order_score = score_order(target.expected_order, attempt.recalled_components)
    condition_values = {
        item: 1.0 if attempt.condition_status.get(item, False) else 0.0
        for item in target.conditions
    }
    condition_score = float(np.mean(list(condition_values.values()))) if condition_values else 1.0
    cue = assess_cue_quality(target, attempt)
    test_quality = assess_test_quality(target, attempt, cue)
    coverage_mix = 0.57 * component_coverage + 0.24 * relation_score + 0.09 * condition_score + 0.10 * order_score
    if not target.expected_order:
        coverage_mix = 0.69 * component_coverage + 0.31 * relation_score
    raw_quality = clamp(0.72 * coverage_mix + 0.18 * attempt.confidence
                        + 0.10 * clamp(1.0 - attempt.response_time_ms / 120000.0))
    aid_factor = 1.0 / (1.0 + 0.20 * attempt.aid_level)
    independence = attempt.aid_level == 0 and not attempt.guessed
    evidenced_accuracy = clamp(coverage_mix * aid_factor)
    return {
        "component_status": component_status,
        "relation_status": relation_status,
        "condition_status": condition_values,
        "component_coverage": component_coverage,
        "relation_coverage": relation_coverage,
        "order_accuracy": order_score,
        "condition_accuracy": condition_score,
        "weighted_coverage": coverage_mix,
        "response_quality": raw_quality,
        "evidenced_accuracy": evidenced_accuracy,
        "confidence": attempt.confidence,
        "confidence_error_proxy": abs(attempt.confidence - evidenced_accuracy),
        "cue_quality": cue,
        "test_quality": test_quality,
        "independent_retrieval": independence,
        "aid_level": attempt.aid_level,
        "guessed": attempt.guessed,
        "response_time_ms": attempt.response_time_ms,
        "direct_recollections": list(attempt.recalled_components),
        "inferred_components": list(attempt.inferred_components),
        "criterion_trusted": bool(test_quality["criterion_trusted"]),
    }


def classify_errors(
    target: LearningTarget, attempt: RetrievalAttempt, assessment: Mapping[str, Any],
) -> dict[str, Any]:
    """Estimate uncertainty-aware probabilities for distinct retrieval failure categories."""
    component_status = assessment["component_status"]
    relation_status = assessment["relation_status"]
    missing = sum(value == "missing" for value in component_status.values())
    substitutions = sum(value in {"substitution", "contradicted"}
                         for value in component_status.values())
    relation_errors = sum(value in {"missing", "reversed", "incorrect"}
                          for value in relation_status.values())
    sequence_error = (1.0 - float(assessment["order_accuracy"])) if target.expected_order else 0.0
    condition_error = (1.0 - float(assessment["condition_accuracy"])) if target.conditions else 0.0
    cue_failure = 1.0 - float(assessment["cue_quality"]["relevance"])
    insufficient = 0.0
    if target.initial_encoding_quality is not None:
        insufficient = clamp((0.45 - target.initial_encoding_quality) / 0.45)
    features = {
        "omission": missing / max(1, len(target.components)),
        "substitution_or_confusion": substitutions / max(1, len(target.components)),
        "relation_error": relation_errors / max(1, len(target.relations)),
        "sequence_error": sequence_error,
        "condition_error": condition_error,
        "cue_failure": cue_failure,
        "insufficient_initial_learning": insufficient,
    }
    if target.criterion_reliability < 0.70:
        return {"probabilities": {}, "primary_error": "criterion_uncertain",
                "features": features, "diagnosis_confidence": 0.0,
                "diagnosis_allowed": False}
    probabilities = softmax({key: value * 3.0 for key, value in features.items()}, temperature=0.65)
    primary = max(features, key=features.get) if features else "none"
    if max(features.values(), default=0.0) < 0.10 and assessment["weighted_coverage"] >= 0.85:
        primary = "no_material_error_detected"
    return {"probabilities": probabilities, "primary_error": primary,
            "features": features, "diagnosis_confidence": max(probabilities.values(), default=0.0),
            "diagnosis_allowed": True}


def select_feedback(
    target: LearningTarget, attempt: RetrievalAttempt,
    diagnosis: Mapping[str, Any], feedback: Feedback | None,
) -> dict[str, Any]:
    """Select correction depth and content from diagnosed gaps and source reliability."""
    source_trust = (feedback.source_reliability if feedback else 0.0)
    verified = bool(feedback and feedback.verified and source_trust >= 0.70
                    and target.criterion_reliability >= 0.70)
    error = str(diagnosis["primary_error"])
    emphasis: dict[str, float] = {}
    assessment_labels = diagnosis.get("features", {})
    for component in target.components:
        confidence = attempt.claim_confidence.get(component, attempt.confidence)
        is_wrong = attempt.component_status.get(component) in {"substitution", "contradicted"}
        importance = _weight(target.component_weights, component)
        emphasis[component] = clamp(0.55 * importance * (1.0 - confidence)
                                    + 0.75 * importance * confidence * float(is_wrong))
    if not feedback:
        action = "provide_targeted_feedback" if error != "no_material_error_detected" else "confirm_and_elaborate"
    elif not verified:
        action = "verify_feedback_source_before_correction"
    elif error in {"omission", "relation_error", "sequence_error", "condition_error",
                   "substitution_or_confusion"}:
        action = "correct_specific_gap_and_explain_distinction"
    elif error in {"cue_failure", "insufficient_initial_learning"}:
        action = "re_explain_or_reencode_before_retesting"
    else:
        action = "confirm_and_elaborate"
    priorities = sorted(emphasis, key=emphasis.get, reverse=True)
    return {
        "action": action, "verified_feedback_accepted": verified,
        "feedback_available": feedback is not None,
        "feedback_source": feedback.source if feedback else None,
        "feedback_source_reliability": source_trust,
        "diagnosed_error": error,
        "priority_components": priorities,
        "component_feedback_emphasis": emphasis,
        "correct_representation": feedback.correct_representation if verified else None,
        "explanation": feedback.explanation if verified else None,
        "original_attempt_must_remain_immutable": True,
    }


def update_accessibility(
    state: TargetMemoryState, attempt: RetrievalAttempt,
    assessment: Mapping[str, Any], learning_rate: float = 0.18,
) -> dict[str, Any]:
    """Update cue-conditioned accessibility with an EMA and Beta-binomial evidence."""
    lr = clamp(learning_rate)
    outcome = float(assessment["evidenced_accuracy"])
    previous = state.accessibility_ema
    state.accessibility_ema = clamp(previous + lr * (outcome - previous))
    independent = bool(assessment["independent_retrieval"])
    if independent:
        if outcome >= 0.75:
            state.successes += 1
        else:
            state.failures += 1
    cue_type = attempt.cue_type
    route = state.cue_routes.setdefault(cue_type, {"successes": 0.0, "failures": 0.0,
                                                    "mean_quality": 0.5, "attempts": 0.0})
    route["attempts"] += 1.0
    route["mean_quality"] += lr * (outcome - route["mean_quality"])
    if independent and outcome >= 0.75:
        route["successes"] += 1.0
    elif independent:
        route["failures"] += 1.0
    alpha, beta = 1.0 + state.successes, 1.0 + state.failures
    posterior_mean = alpha / (alpha + beta)
    posterior_variance = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
    if independent:
        state.confidence_pairs.append({"confidence": float(attempt.confidence), "outcome": outcome})
    state.confidence_pairs[:] = state.confidence_pairs[-200:]
    brier = (sum((row["confidence"] - row["outcome"]) ** 2
                 for row in state.confidence_pairs) / len(state.confidence_pairs)
             if state.confidence_pairs else None)
    return {"accessibility_before": previous, "accessibility_ema": state.accessibility_ema,
            "independent_successes": state.successes, "independent_failures": state.failures,
            "beta_alpha": alpha, "beta_beta": beta, "posterior_retrieval_mean": posterior_mean,
            "posterior_variance": posterior_variance, "confidence_brier_score": brier,
            "cue_route": dict(route), "independence_required_for_beta_update": True}


def choose_learning_action(
    assessment: Mapping[str, Any], feedback_decision: Mapping[str, Any],
    fatigue: float, available_time_minutes: float,
) -> dict[str, Any]:
    """Choose retrieval, correction, re-encoding, transfer, restudy, or delay by expected utility."""
    quality = float(assessment["weighted_coverage"])
    confidence = float(assessment["confidence"])
    trusted = bool(assessment["criterion_trusted"])
    options = {
        "retrieve_again": 0.55 * quality + 0.25 * confidence - 0.20 * fatigue,
        "targeted_feedback": 0.75 * (1.0 - quality) + 0.15 * fatigue,
        "restudy_or_reencode": 0.70 * (1.0 - quality) + 0.20 * (not trusted),
        "transfer_test": 0.35 + 0.35 * quality - 0.20 * fatigue,
        "delay_and_retrieve_later": 0.50 + 0.20 * quality - 0.10 * fatigue,
        "pause_or_reduce_load": 0.80 * fatigue,
    }
    if feedback_decision["action"] == "verify_feedback_source_before_correction":
        options["verify_criterion"] = 1.0
        options["targeted_feedback"] = -1.0
    if quality < 0.25:
        options["restudy_or_reencode"] += 0.35
        options["retrieve_again"] -= 0.40
    if available_time_minutes < 5.0:
        options["delay_and_retrieve_later"] += 0.30
    utilities = {key: float(value) for key, value in options.items()}
    selected = max(utilities, key=utilities.get)
    return {"selected_action": selected, "expected_utility_scores": utilities,
            "utility_margin": sorted(utilities.values(), reverse=True)[0]
            - (sorted(utilities.values(), reverse=True)[1] if len(utilities) > 1 else 0.0),
            "fatigue_adjusted": True, "time_budget_minutes": available_time_minutes}


def detect_test_failures(
    target: LearningTarget, attempt: RetrievalAttempt,
    assessment: Mapping[str, Any],
) -> dict[str, Any]:
    """Flag cue mismatch, leakage, ambiguity, guessing, overdifficulty, and criterion mismatch."""
    cue = assessment["cue_quality"]
    flags = {
        "cue_mismatch": float(cue["relevance"]) < 0.12,
        "answer_leakage": attempt.answer_leakage >= 0.35,
        "ambiguous_question": attempt.ambiguity >= 0.35,
        "unreliable_answer_criterion": target.criterion_reliability < 0.70,
        "guessing_possible": attempt.guessed,
        "overly_difficult_or_unsupported": assessment["weighted_coverage"] < 0.15,
        "overly_easy_or_highly_cued": attempt.aid_level > 0 or attempt.answer_leakage >= 0.65,
        "narrow_cue_dependence": False,
        "excessive_repetition_risk": False,
        "confidence_illusion": (attempt.confidence >= 0.80
                                and assessment["evidenced_accuracy"] < 0.50),
        "premature_answer_exposure": attempt.aid_level > 0,
        "criterion_mismatch": assessment["test_quality"]["test_quality"] < 0.35,
        "uncorrected_high_confidence_error": False,
    }
    return {"flags": flags, "test_quality": assessment["test_quality"],
            "valid_for_independent_retrieval_measure": assessment["independent_retrieval"]
            and not any(flags[key] for key in ("cue_mismatch", "answer_leakage",
                                               "ambiguous_question",
                                               "unreliable_answer_criterion"))}


def update_knowledge_graph(
    target: LearningTarget, assessment: Mapping[str, Any],
    feedback_decision: Mapping[str, Any], feedback: Feedback | None,
    state: TargetMemoryState, learning_rate: float = 0.06,
) -> dict[str, Any]:
    """Strengthen only explicitly supported target relations and retain uncertain links separately."""
    verified = bool(feedback_decision["verified_feedback_accepted"])
    updates: dict[str, float] = {}
    if verified:
        for relation in target.relations:
            old = state.verified_relations.get(relation, 0.0)
            delta = clamp(learning_rate * feedback.source_reliability)  # type: ignore[union-attr]
            state.verified_relations[relation] = clamp(old + delta)
            updates[relation] = state.verified_relations[relation]
    return {"graph_nodes": list(target.components), "candidate_edges": list(target.relations),
            "verified_edge_strengths": dict(state.verified_relations),
            "updated_edges": updates, "unverified_edges_strengthened": False,
            "criterion_reliability_gate": verified}


def delayed_retention_assessment(
    target: LearningTarget, attempts: Sequence[RetrievalAttempt],
) -> dict[str, Any]:
    """Measure delayed access separately from immediate correction and cue transfer."""
    scores = [assess_attempt(target, item) for item in attempts]
    independent = [score for score in scores if score["independent_retrieval"]]
    accuracy = [float(score["evidenced_accuracy"]) for score in independent]
    return {"delayed_attempt_count": len(attempts),
            "independent_delayed_attempt_count": len(independent),
            "mean_delayed_accuracy": float(np.mean(accuracy)) if accuracy else None,
            "delayed_accuracy": accuracy,
            "durable_learning_evidence_available": bool(independent),
            "immediate_retest_is_not_delayed_retention": True}


def calculate_future_priority(
    target: LearningTarget, assessment: Mapping[str, Any],
    diagnosis: Mapping[str, Any], state: TargetMemoryState,
    transfer_accuracy: float | None = None,
) -> dict[str, Any]:
    """Rank future practice using inaccuracy, uncertainty, recurrent errors, and transfer weakness."""
    accuracy = float(assessment["evidenced_accuracy"])
    uncertainty = 1.0 - abs(float(assessment["confidence"]) - 0.5) * 2.0
    recurrence = clamp(sum(state.error_counts.values()) / max(1.0, 4.0 * len(target.components)))
    transfer = 1.0 - clamp(transfer_accuracy) if transfer_accuracy is not None else 0.5
    priority = clamp(0.38 * (1.0 - accuracy) + 0.22 * uncertainty
                     + 0.22 * recurrence + 0.18 * transfer)
    return {"priority_score": priority, "accuracy_term": 1.0 - accuracy,
            "uncertainty_term": uncertainty, "recurring_error_term": recurrence,
            "transfer_weakness_term": transfer,
            "recommended_focus": ("transfer_practice" if transfer > 0.65 and accuracy > 0.7
                                  else "targeted_correction" if diagnosis["primary_error"] !=
                                  "no_material_error_detected" else "delayed_retrieval"),
            "target_id": target.target_id}


def update_error_history(state: TargetMemoryState, diagnosis: Mapping[str, Any]) -> None:
    """Increment recurrence counts for the current dominant error category."""
    error = str(diagnosis.get("primary_error", "unknown"))
    if error not in {"none", "no_material_error_detected", "criterion_uncertain"}:
        state.error_counts[error] = state.error_counts.get(error, 0) + 1


def record_attempt(
    state: TargetMemoryState, attempt: RetrievalAttempt,
    assessment: Mapping[str, Any], label: str,
) -> None:
    """Append an audit record without mutating the original immutable response object."""
    state.attempt_history.append({
        "label": label, "cue": attempt.cue, "cue_type": attempt.cue_type,
        "response": attempt.response, "confidence": attempt.confidence,
        "aid_level": attempt.aid_level, "guess": attempt.guessed,
        "evidenced_accuracy": assessment["evidenced_accuracy"],
        "independent_retrieval": assessment["independent_retrieval"],
    })
    state.attempt_history[:] = state.attempt_history[-500:]


def update_cue_diversity(
    target: LearningTarget, attempts: Sequence[RetrievalAttempt],
) -> dict[str, Any]:
    """Measure the number and diversity of valid retrieval cue types used for one target."""
    valid = [item for item in attempts
             if item.cue_type in target.valid_cue_types or item.cue_type == "free_recall"]
    counts = Counter(item.cue_type for item in valid)
    total_types = len(target.valid_cue_types) + 1
    diversity = len(counts) / max(1, total_types)
    return {"valid_attempts": len(valid), "cue_type_counts": dict(counts),
            "tested_cue_types": sorted(counts), "cue_type_coverage": clamp(diversity),
            "multiple_routes_tested": len(counts) > 1}


def schedule_handoff(
    target: LearningTarget, priority: Mapping[str, Any],
    assessment: Mapping[str, Any], delayed: Mapping[str, Any],
    preferred_delay_hours: float | None = None,
) -> dict[str, Any]:
    """Prepare a Point 03-compatible review handoff without duplicating its scheduler."""
    score = float(priority["priority_score"])
    if score >= 0.70:
        delay_hours = 6.0
    elif score >= 0.40:
        delay_hours = 24.0
    else:
        delay_hours = 72.0
    if assessment["evidenced_accuracy"] < 0.35:
        delay_hours = min(delay_hours, 2.0)
    policy_delay_hours = delay_hours
    if preferred_delay_hours is not None:
        requested = float(preferred_delay_hours)
        if not math.isfinite(requested) or requested < 0:
            raise ValueError("preferred_delay_hours must be finite and non-negative")
        delay_hours = requested
    return {"target_id": target.target_id, "operation": "schedule_delayed_retrieval",
            "suggested_delay_hours": delay_hours, "policy_delay_hours": policy_delay_hours,
            "priority_score": score,
            "last_evidenced_accuracy": assessment["evidenced_accuracy"],
            "delayed_evidence_available": delayed["durable_learning_evidence_available"],
            "scheduler_owner": "Point 03 Spaced Repetition",
            "external_scheduler_required": True}


def update_state_for_attempt(
    state: TargetMemoryState, attempt: RetrievalAttempt,
    assessment: Mapping[str, Any], label: str,
) -> dict[str, Any]:
    """Update accessibility estimates and store the observed attempt by cue route."""
    record_attempt(state, attempt, assessment, label)
    return update_accessibility(state, attempt, assessment)
