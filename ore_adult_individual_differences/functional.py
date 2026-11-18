"""Goal representation, state estimation, segmentation, and numerical operations (Point 17)."""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from .contracts import (
    AdultComprehensionExperience, AdultDifferencesConfig, AdultLearnerState,
)


_TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)?|[^\w\s]", flags=re.UNICODE)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a finite scalar to a closed interval."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("value must be finite")
    return max(low, min(high, number))


def sigmoid(value: float) -> float:
    """Compute a numerically stable logistic response."""
    number = float(value)
    if number >= 0.0:
        z = math.exp(-min(number, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(number, -700.0))
    return z / (1.0 + z)


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Normalize finite scores into a stable probability distribution."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not scores:
        return []
    values = [float(item) / temperature for item in scores]
    peak = max(values)
    exponents = [math.exp(max(-700.0, item - peak)) for item in values]
    total = sum(exponents)
    return [item / total for item in exponents]


def normalize_probabilities(values: Sequence[float]) -> list[float]:
    """Normalize non-negative values, falling back to a uniform distribution."""
    clean = [max(0.0, float(value)) for value in values]
    total = sum(clean)
    if total <= 0.0:
        return [1.0 / len(clean)] * len(clean) if clean else []
    return [value / total for value in clean]


def entropy(probabilities: Sequence[float]) -> float:
    """Compute Shannon entropy in nats for a probability vector."""
    values = normalize_probabilities(probabilities)
    return -sum(value * math.log(value) for value in values if value > 0.0)


def token_set(value: Any) -> set[str]:
    """Extract a normalized set of lexical features from nested data."""
    if isinstance(value, Mapping):
        result = {str(key).lower() for key in value}
        for item in value.values():
            result |= token_set(item)
        return result
    if isinstance(value, (tuple, list, set)):
        result: set[str] = set()
        for item in value:
            result |= token_set(item)
        return result
    if value in (None, ""):
        return set()
    return {match.lower() for match in _TOKEN_PATTERN.findall(str(value)) if match.strip()}


def jaccard_similarity(left: Any, right: Any) -> float:
    """Compute Jaccard overlap between normalized feature sets."""
    a, b = token_set(left), token_set(right)
    return len(a & b) / max(1, len(a | b))


def _safe_number(mapping: Mapping[str, Any], key: str, default: float) -> float:
    """Read a finite numeric mapping value or use a supplied default."""
    value = mapping.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def establish_task_goal(
    event: AdultComprehensionExperience,
) -> dict[str, Any]:
    """Create the explicit objective vector and input-feature priority representation."""
    goal = dict(event.task_goal)
    feature_support = event.metadata.get("feature_goal_support", {})
    priorities: dict[str, float] = {}
    for feature, support in feature_support.items():
        score = sum(
            float(weight) * clamp(_safe_number(support, objective, 0.0))
            for objective, weight in goal.items()
        ) if isinstance(support, Mapping) else 0.0
        priorities[str(feature)] = clamp(score)
    return {
        "objective_vector": goal,
        "desired_output": event.desired_output,
        "feature_priorities": priorities,
        "primary_objective": max(goal, key=goal.get),
        "task_scope": str(event.task_context.get("scope", event.domain)),
        "goal_is_explicit_not_inferred_from_content": True,
    }


def estimate_current_resources(
    event: AdultComprehensionExperience,
    learner: AdultLearnerState,
    config: AdultDifferencesConfig,
) -> dict[str, Any]:
    """Estimate dynamic attention and maintenance support for the current task."""
    task_focus = clamp(_safe_number(event.task_context, "goal_focus", 0.7))
    focus_term = 0.5 * task_focus + 0.5 * event.motivation
    attention_linear = (
        2.0 * focus_term
        - config.attention_distraction_weight * event.distraction
        - config.attention_fatigue_weight * event.fatigue
        - config.attention_stress_weight * event.stress
        + config.motivation_weight * event.motivation
    )
    context_attention = sigmoid(attention_linear)
    attention = clamp(learner.attention_control * context_attention)
    external = event.external_support * config.external_support_gain
    maintenance_mix = (
        (1.0 - config.wm_attention_mix)
        + config.wm_attention_mix * attention
    )
    working_memory = clamp(
        learner.working_memory_support * maintenance_mix + external
    )
    speed_context = clamp(
        1.0 - 0.25 * event.fatigue - 0.15 * event.stress
        + 0.10 * event.motivation
    )
    task_speed = clamp(learner.processing_speed * speed_context)
    return {
        "attention_control": attention,
        "working_memory_availability": working_memory,
        "processing_speed_estimate": task_speed,
        "external_support": event.external_support,
        "distraction": event.distraction,
        "fatigue": event.fatigue,
        "stress": event.stress,
        "motivation": event.motivation,
        "task_specific_estimate": True,
        "permanent_trait_or_diagnosis_inferred": False,
        "resource_equations": {
            "attention": "baseline_attention * sigmoid(focus - weighted_interference + motivation)",
            "working_memory": "baseline_WM * ((1-mix)+mix*attention) + external_support_gain",
            "speed": "baseline_speed * context_factor(fatigue, stress, motivation)",
        },
    }


def _unit_tokens(raw: Any, index: int) -> list[dict[str, Any]]:
    """Convert one supplied unit into ordered token records with source provenance."""
    if isinstance(raw, Mapping):
        content = raw.get("content", raw.get("text", raw.get("value", "")))
        source = str(raw.get("source_id", raw.get("source", f"input-{index}")))
        confidence = clamp(float(raw.get("confidence", 1.0)))
        provenance = str(raw.get("provenance", "observed"))
        explicit_kind = str(raw.get("kind", ""))
    else:
        content = raw
        source = f"input-{index}"
        confidence = 1.0
        provenance = "observed"
        explicit_kind = ""
    tokens = [str(content)] if explicit_kind in {"proposition", "event", "instruction"} else (
        _TOKEN_PATTERN.findall(str(content))
    )
    if not tokens and content not in (None, ""):
        tokens = [str(content)]
    return [
        {
            "unit_id": f"{source}:{token_index}",
            "content": token,
            "source_id": source,
            "source_unit_index": index,
            "token_index": token_index,
            "confidence": confidence,
            "provenance": provenance,
            "kind": explicit_kind or "token",
        }
        for token_index, token in enumerate(tokens)
    ]


def segment_input(
    event: AdultComprehensionExperience,
    config: AdultDifferencesConfig,
) -> dict[str, Any]:
    """Build temporally ordered meaningful units and confidence-scored boundary records."""
    units: list[dict[str, Any]] = []
    for index, raw in enumerate(event.content_units):
        units.extend(_unit_tokens(raw, index))
    cue_records = event.metadata.get("boundary_cues", ())
    boundaries: list[dict[str, Any]] = []
    for index in range(max(0, len(units) - 1)):
        cue = cue_records[index] if index < len(cue_records) and isinstance(cue_records[index], Mapping) else {}
        cue_strength = clamp(_safe_number(cue, "strength", 0.5))
        syntax = clamp(_safe_number(cue, "syntax", 0.5))
        prosody = clamp(_safe_number(cue, "prosody", 0.5))
        context = clamp(_safe_number(cue, "context", 0.5))
        boundary_probability = sigmoid(
            -1.0 + 1.4 * cue_strength + 1.0 * syntax + 0.8 * prosody + 0.8 * context
        )
        boundaries.append({
            "after_unit": index,
            "left_unit_id": units[index]["unit_id"],
            "right_unit_id": units[index + 1]["unit_id"],
            "probability": boundary_probability,
            "ambiguous": 0.35 <= boundary_probability <= 0.65,
            "cues": {"perceptual": cue_strength, "syntax": syntax,
                     "prosody": prosody, "context": context},
        })
    return {
        "units": units,
        "unit_count": len(units),
        "boundary_records": boundaries,
        "ambiguous_boundaries": [
            row["after_unit"] for row in boundaries if row["ambiguous"]
        ],
        "source_order_preserved": True,
        "provenance_retained": True,
        "segmentation_scope": "unit tokens, propositions, events, or instructions according to supplied input",
    }
