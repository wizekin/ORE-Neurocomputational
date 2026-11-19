"""Functional scheduling, accessibility, retrieval-scoring, and calibration operations for Point 03."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Any, Mapping, Sequence

from .contracts import LearningItem, ReviewObservation, SpacedRepetitionConfig


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Bound a finite value and reject invalid ranges."""
    number = float(value)
    if not math.isfinite(number) or low > high:
        raise ValueError("value must be finite and low must not exceed high")
    return min(high, max(low, number))


def parse_utc(value: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp used by the scheduler."""
    candidate = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(candidate)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def add_hours(timestamp: str, hours: float) -> str:
    """Return a UTC timestamp after a finite non-negative number of hours."""
    if not math.isfinite(float(hours)) or hours < 0:
        raise ValueError("hours must be finite and non-negative")
    return (parse_utc(timestamp) + timedelta(hours=float(hours))).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def elapsed_hours(start: str, end: str) -> float:
    """Calculate elapsed hours and reject time moving backwards."""
    value = (parse_utc(end) - parse_utc(start)).total_seconds() / 3600.0
    if value < -1e-9:
        raise ValueError("end timestamp cannot precede start timestamp")
    return max(0.0, value)


def estimate_encoding_quality(measurements: Mapping[str, float]) -> dict[str, float]:
    """Estimate initial encoding quality and uncertainty from available behavioral indicators."""
    weights = {
        "attention": 0.25,
        "comprehension": 0.30,
        "relational_organization": 0.30,
        "meaningful_practice": 0.15,
    }
    default_value = 0.5
    supplied_weight = sum(weights[key] for key in measurements)
    quality = sum(weights[key] * float(measurements.get(key, default_value))
                  for key in weights)
    missing_fraction = 1.0 - supplied_weight
    values = [float(measurements.get(key, default_value)) for key in weights]
    variance = sum((value - quality) ** 2 for value in values) / len(values)
    uncertainty = clamp(0.20 + 0.65 * missing_fraction + 0.35 * math.sqrt(variance))
    return {
        "encoding_quality": clamp(quality),
        "encoding_uncertainty": uncertainty,
        "measurement_coverage": clamp(supplied_weight),
        "measurements_used": sorted(measurements),
        "interpretation": "behavioral proxy; not a direct measurement of a neural memory trace",
    }


def initialize_stability(
    item: LearningItem, encoding_quality: float, config: SpacedRepetitionConfig
) -> float:
    """Initialize a bounded durability timescale from quality and difficulty."""
    quality_gain = 0.50 + 1.50 * clamp(encoding_quality)
    difficulty_penalty = 1.0 - 0.55 * item.difficulty
    return min(config.maximum_stability_hours, max(
        config.minimum_stability_hours,
        config.base_stability_hours * quality_gain * difficulty_penalty,
    ))


def choose_initial_interval(
    item: LearningItem, encoding_quality: float, config: SpacedRepetitionConfig
) -> dict[str, Any]:
    """Select a nonzero initial gap from retention horizon, difficulty, and encoding quality."""
    quality_factor = 0.70 + 0.60 * clamp(encoding_quality)
    difficulty_factor = 1.25 - 0.50 * item.difficulty
    proposed = item.target_retention_hours * config.initial_interval_fraction
    proposed *= quality_factor * difficulty_factor
    upper = min(config.maximum_interval_hours, item.target_retention_hours * 0.5)
    interval = min(upper, max(config.minimum_interval_hours, proposed))
    return {
        "interval_hours": interval,
        "formula": "clip(horizon*fraction*(0.70+0.60*encoding_quality)*(1.25-0.50*difficulty))",
        "nonzero_delay": True,
        "parameters_are_calibratable": True,
    }


def estimate_accessibility(
    stability_hours: float, elapsed: float, review_count: int,
    prior_uncertainty: float, outcomes: Sequence[float],
    config: SpacedRepetitionConfig,
) -> dict[str, float | str]:
    """Estimate time-dependent accessibility with an explicit uncertainty diagnostic."""
    if not math.isfinite(stability_hours) or stability_hours <= 0:
        raise ValueError("stability_hours must be finite and positive")
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed must be finite and non-negative")
    if review_count < 0:
        raise ValueError("review_count must be non-negative")
    ratio = elapsed / stability_hours
    if config.forgetting_model == "exponential":
        accessibility = math.exp(-ratio)
        formula = "exp(-elapsed_hours/stability_hours)"
    else:
        accessibility = (1.0 + ratio) ** -0.85
        formula = "(1+elapsed_hours/stability_hours)^(-0.85)"
    variability = 0.0
    if len(outcomes) > 1:
        mean = sum(outcomes) / len(outcomes)
        variability = math.sqrt(sum((x - mean) ** 2 for x in outcomes) / len(outcomes))
    uncertainty = clamp(
        max(config.uncertainty_floor, prior_uncertainty / math.sqrt(1.0 + review_count))
        + 0.20 * variability
    )
    return {
        "memory_accessibility": clamp(accessibility),
        "predicted_recall_probability": clamp(accessibility),
        "memory_uncertainty": uncertainty,
        "elapsed_hours": elapsed,
        "stability_hours": stability_hours,
        "forgetting_model": config.forgetting_model,
        "formula": formula,
    }


def decide_review_action(
    predicted_recall: float, now_utc: str, deadline_utc: str,
    next_due_utc: str, config: SpacedRepetitionConfig,
) -> dict[str, Any]:
    """Choose review/defer/relearn/urgent action from recall band, due time, and deadline."""
    probability = clamp(predicted_recall)
    remaining_deadline = elapsed_hours(now_utc, deadline_utc) if deadline_utc >= now_utc else 0.0
    overdue = parse_utc(now_utc) >= parse_utc(next_due_utc)
    urgent = remaining_deadline <= config.urgency_window_hours
    if urgent:
        action = "urgent_review"
        cue_policy = "minimal_cue_unless_retrieval_fails"
    elif probability < config.target_recall_low:
        action = "relearn_with_cue"
        cue_policy = "supportive_cue_and_corrective_relearning"
    elif probability > config.target_recall_high:
        action = "review" if overdue else "defer"
        cue_policy = "free_recall_when_due"
    else:
        action = "review" if overdue else "defer"
        cue_policy = "minimal_cue_or_free_recall"
    return {
        "action": action, "cue_policy": cue_policy,
        "review_due": overdue, "urgent_deadline": urgent,
        "hours_until_deadline": remaining_deadline,
        "next_due_at_utc": next_due_utc,
        "decision_is_scheduler_policy": True,
    }


def score_review(observation: ReviewObservation) -> dict[str, Any]:
    """Score accuracy, completeness, relations, and fluency while separating effort and aid."""
    weights = {"accuracy": 0.35, "completeness": 0.25,
               "relation_accuracy": 0.25, "fluency": 0.15}
    score = sum(weights[key] * float(getattr(observation, key)) for key in weights)
    independent = (
        observation.response_mode in {"free_recall", "cued_recall", "procedure", "explanation"}
        and observation.hint_level < 0.5
        and observation.outcome != "recognition_only"
    )
    return {
        "quality_score": clamp(score), "accuracy": observation.accuracy,
        "completeness": observation.completeness,
        "relation_accuracy": observation.relation_accuracy,
        "fluency": observation.fluency, "effort": observation.effort,
        "response_mode": observation.response_mode,
        "hint_level": observation.hint_level,
        "independent_retrieval": independent,
        "feedback_provided": observation.feedback_provided,
        "outcome": observation.outcome,
        "event_type": (
            "corrected_relearning" if observation.feedback_provided
            else "independent_retrieval" if independent else "recognition_or_aided_response"
        ),
    }


def update_memory_state(
    stability_hours: float, uncertainty: float, score: Mapping[str, Any],
    config: SpacedRepetitionConfig,
) -> dict[str, float | str]:
    """Update the durability estimate with outcome-specific gains and preserve uncertainty."""
    old = float(stability_hours)
    effort = clamp(float(score["effort"]))
    quality = clamp(float(score["quality_score"]))
    outcome = str(score["outcome"])
    independent = bool(score["independent_retrieval"])
    learning_rate = config.learning_rate

    if outcome == "success" and independent:
        multiplier = 1.0 + learning_rate * (0.65 + 0.70 * effort) * quality
        uncertainty_delta = -0.10 * quality
    elif outcome == "partial" and independent:
        multiplier = 1.0 + learning_rate * 0.35 * quality
        uncertainty_delta = 0.04
    elif outcome == "corrected_relearning" or bool(score["feedback_provided"]):
        multiplier = 1.0 + learning_rate * 0.40 * quality
        uncertainty_delta = 0.03
    elif outcome == "recognition_only" or not independent:
        multiplier = 1.0 + learning_rate * 0.05 * quality
        uncertainty_delta = 0.06
    else:
        multiplier = max(0.55, 1.0 - 0.40 * learning_rate * (1.0 - quality))
        uncertainty_delta = 0.12
    new_stability = min(config.maximum_stability_hours, max(
        config.minimum_stability_hours, old * multiplier
    ))
    new_uncertainty = clamp(
        float(uncertainty) + uncertainty_delta,
        config.uncertainty_floor, 1.0,
    )
    return {
        "previous_stability_hours": old,
        "stability_hours": new_stability,
        "stability_multiplier": multiplier,
        "memory_uncertainty": new_uncertainty,
        "update_type": outcome,
        "recognition_does_not_equal_independent_mastery": outcome == "recognition_only",
    }


def choose_next_interval(
    stability_hours: float, item: LearningItem, observation: ReviewObservation,
    now_utc: str, config: SpacedRepetitionConfig,
) -> dict[str, Any]:
    """Calculate a stability-based interval and adapt it to outcome and remaining horizon."""
    base = -float(stability_hours) * math.log(config.target_recall_probability)
    outcome_factor = {
        "success": 1.0,
        "partial": 0.78,
        "failure": 0.45,
        "corrected_relearning": 0.68,
        "recognition_only": 0.60,
    }[observation.outcome]
    if observation.hint_level >= 0.5:
        outcome_factor *= 0.85
    proposed = base * outcome_factor
    deadline_utc = add_hours(
        item.created_at_utc or now_utc, item.target_retention_hours
    )
    remaining = max(0.0, (parse_utc(deadline_utc) - parse_utc(now_utc)).total_seconds() / 3600.0)
    horizon_cap = remaining * 0.60 if remaining > 0 else config.maximum_interval_hours
    maximum = min(config.maximum_interval_hours, max(config.minimum_interval_hours, horizon_cap))
    interval = min(maximum, max(config.minimum_interval_hours, proposed))
    return {
        "interval_hours": interval, "base_interval_hours": base,
        "outcome_factor": outcome_factor, "horizon_cap_hours": horizon_cap,
        "schedule_reason": (
            f"stability-based interval adjusted by {observation.outcome}, "
            "cue support, and remaining retention horizon"
        ),
    }


def assess_maintenance(
    history: Sequence[Mapping[str, Any]], created_at_utc: str,
    now_utc: str, config: SpacedRepetitionConfig,
) -> dict[str, Any]:
    """Mark an item provisionally stable only after independent successes across separated reviews."""
    successful = [
        item for item in history
        if item.get("independent_retrieval") is True
        and item.get("outcome") == "success"
        and float(item.get("quality_score", 0.0)) >= config.maintenance_accuracy_threshold
    ]
    span = 0.0
    if len(successful) >= 2:
        span = elapsed_hours(str(successful[0]["occurred_at_utc"]),
                             str(successful[-1]["occurred_at_utc"]))
    retention_elapsed = elapsed_hours(created_at_utc, now_utc)
    stable = (len(successful) >= config.maintenance_successes_required
              and span >= config.maintenance_min_span_hours
              and retention_elapsed >= config.maintenance_min_span_hours)
    return {
        "provisionally_stable": stable,
        "independent_success_count": len(successful),
        "success_span_hours": span,
        "elapsed_since_initial_learning_hours": retention_elapsed,
        "required_successes": config.maintenance_successes_required,
        "required_span_hours": config.maintenance_min_span_hours,
        "permanent_mastery_claimed": False,
    }


def apply_offline_consolidation(
    stability_hours: float, uncertainty: float, observation: Mapping[str, Any],
    config: SpacedRepetitionConfig,
) -> dict[str, Any]:
    """Apply only an explicitly observed, evidence-weighted offline influence to the model state."""
    observed = bool(observation.get("observed", False))
    if not observed:
        return {
            "observed": False, "stability_adjustment_hours": 0.0,
            "stability_hours": stability_hours, "memory_uncertainty": uncertainty,
            "guaranteed_sleep_boost": False,
            "reason": "no offline/sleep observation supplied; no automatic adjustment applied",
        }
    duration = max(0.0, float(observation.get("duration_hours", 0.0)))
    quality = clamp(float(observation.get("quality", 0.5)))
    interference = clamp(float(observation.get("interference", 0.0)))
    evidence = clamp(float(observation.get("evidence_confidence", 0.0)))
    exposure = min(1.0, duration / 8.0)
    gain = config.consolidation_learning_rate * exposure * quality * (1.0 - interference) * evidence
    adjustment = float(stability_hours) * gain
    new_stability = min(config.maximum_stability_hours, float(stability_hours) + adjustment)
    new_uncertainty = clamp(float(uncertainty) - 0.04 * evidence * exposure,
                            config.uncertainty_floor, 1.0)
    return {
        "observed": True, "duration_hours": duration, "quality": quality,
        "interference": interference, "evidence_confidence": evidence,
        "stability_adjustment_hours": adjustment, "stability_hours": new_stability,
        "memory_uncertainty": new_uncertainty, "guaranteed_sleep_boost": False,
        "reason": "small configurable influence conditioned on explicit context and evidence",
    }


def assess_calibration(
    prediction_history: Sequence[Mapping[str, float]], config: SpacedRepetitionConfig,
) -> dict[str, Any]:
    """Calculate Brier score and signed prediction error from independent binary recall outcomes."""
    records = list(prediction_history)[-config.calibration_window:]
    if not records:
        return {"sample_count": 0, "brier_score": None, "mean_prediction_error": None,
                "calibration_available": False}
    errors = [float(row["observed"]) - float(row["predicted"]) for row in records]
    brier = sum(error ** 2 for error in errors) / len(errors)
    return {
        "sample_count": len(records), "brier_score": brier,
        "mean_prediction_error": sum(errors) / len(errors),
        "calibration_available": True,
        "interpretation": "behavioral prediction calibration; not a neural measurement",
    }
