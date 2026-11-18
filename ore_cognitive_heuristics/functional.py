"""Functional operations for Point 22 cognitive heuristics and corrective reasoning."""
from __future__ import annotations

from math import exp, isfinite, log, sqrt
from typing import Any, Mapping, Sequence

from .contracts import JudgmentProblem, Point22Config, Point22State


_EPS = 1e-12


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Return a finite value bounded by lower and upper."""
    if not isfinite(float(value)):
        raise ValueError("value must be finite")
    return max(lower, min(upper, float(value)))


def softmax(scores: Mapping[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Convert finite category scores to a stable probability distribution."""
    if temperature <= 0 or not isfinite(float(temperature)):
        raise ValueError("temperature must be positive and finite")
    if not scores:
        return {}
    scaled = {key: float(value) / temperature for key, value in scores.items()}
    if any(not isfinite(value) for value in scaled.values()):
        raise ValueError("softmax scores must be finite")
    peak = max(scaled.values())
    weights = {key: exp(max(-700.0, value - peak)) for key, value in scaled.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def resolve_categories(problem: JudgmentProblem) -> tuple[str, ...]:
    """Resolve category labels from explicit categories and supplied evidence structures."""
    keys = list(problem.categories)
    for source in (problem.base_rates, problem.prototypes, problem.similarity_scores):
        for key in source:
            if key not in keys:
                keys.append(str(key))
    likelihoods = problem.evidence.get("likelihoods", {})
    if isinstance(likelihoods, Mapping):
        for key in likelihoods:
            if key not in keys:
                keys.append(str(key))
    if not keys:
        keys = ["yes", "no"]
    return tuple(keys)


def normalize_distribution(values: Mapping[str, float], categories: Sequence[str]) -> dict[str, float]:
    """Normalize non-negative category weights, falling back to a uniform distribution."""
    raw = {key: max(0.0, float(values.get(key, 0.0))) for key in categories}
    total = sum(raw.values())
    if total <= _EPS:
        return {key: 1.0 / len(categories) for key in categories}
    return {key: value / total for key, value in raw.items()}


def prior_distribution(problem: JudgmentProblem, categories: Sequence[str]) -> dict[str, float]:
    """Return normalized population priors or an explicitly uniform starting prior."""
    if not problem.base_rates:
        return {key: 1.0 / len(categories) for key in categories}
    return normalize_distribution(problem.base_rates, categories)


def receive_problem(problem: JudgmentProblem, categories: Sequence[str]) -> dict[str, Any]:
    """Translate the question into a target, candidate set, stake level, and evidence budget."""
    risk = max(problem.decision_cost, problem.emotional_salience)
    target = {
        "question": problem.question,
        "target_type": problem.target_type,
        "categories": list(categories),
        "uncertainty_present": bool(problem.evidence or len(categories) > 1),
        "time_limited": problem.time_pressure > 0.5,
        "risk_level": risk,
        "required_verification": risk >= 0.5 or problem.target_type in {"cause", "probability"},
        "available_evidence_count": len(problem.examples) + len(problem.evidence),
    }
    return target


def activate_knowledge(problem: JudgmentProblem) -> dict[str, Any]:
    """Retrieve examples and feature prototypes while retaining access and source metadata."""
    memories = []
    for index, raw in enumerate(problem.examples):
        row = dict(raw)
        recency = clamp(float(row.get("recency", 0.0)))
        salience = clamp(float(row.get("salience", 0.0)))
        fluency = clamp(float(row.get("fluency", row.get("strength", 0.0))))
        learned = clamp(float(row.get("strength", 0.0)))
        accessibility = 0.25 * recency + 0.25 * salience + 0.20 * fluency + 0.30 * learned
        memories.append({
            "example_id": str(row.get("example_id", f"example-{index + 1}")),
            "category": row.get("category"),
            "content": str(row.get("content", "")),
            "source_id": row.get("source_id"),
            "independence_group": row.get("independence_group"),
            "accessibility": accessibility,
            "recency": recency,
            "salience": salience,
            "fluency": fluency,
            "strength": learned,
        })
    return {
        "retrieved_examples": memories,
        "prototype_categories": sorted(problem.prototypes),
        "active_memory_count": len(memories),
        "mean_accessibility": (
            sum(item["accessibility"] for item in memories) / len(memories) if memories else 0.0
        ),
        "retrieved_items_are_not_verified_evidence": True,
    }


def select_heuristic_strategy(
    problem: JudgmentProblem,
    knowledge: Mapping[str, Any],
    config: Point22Config,
) -> dict[str, Any]:
    """Select candidate shortcuts and mark the conditions that make them unsafe."""
    candidates: list[str] = []
    if knowledge["retrieved_examples"]:
        candidates.append("availability")
    if problem.prototypes or problem.similarity_scores or problem.features:
        candidates.append("representativeness")
    if problem.candidate_anchors:
        candidates.append("anchoring-and-adjustment")
    if problem.fluency > 0 or problem.evidence.get("exposure_count"):
        candidates.append("processing-fluency")
    if not candidates:
        candidates.append("base-rate/default-prior")
    risks = []
    if len(knowledge["retrieved_examples"]) < 3 and knowledge["retrieved_examples"]:
        risks.append("Few retrieved examples may not represent the population.")
    if problem.base_rates:
        risks.append("Use base rates as a denominator check rather than a mere resemblance cue.")
    if problem.decision_cost >= config.high_risk_threshold:
        risks.append("High decision cost requires a verification pass.")
    if problem.time_pressure >= config.high_risk_threshold:
        risks.append("Time pressure may reduce alternative checking.")
    return {
        "candidate_heuristics": candidates,
        "selected_heuristics": candidates,
        "selection_basis": "available cues, task type, time pressure, and error cost",
        "environmental_assumptions_checked": False,
        "warnings": risks,
    }


def availability_heuristic(
    problem: JudgmentProblem,
    knowledge: Mapping[str, Any],
    categories: Sequence[str],
) -> dict[str, Any]:
    """Estimate accessibility-based support while exposing the gap from population frequency."""
    rows = knowledge["retrieved_examples"]
    totals = {key: 0.0 for key in categories}
    counts = {key: 0 for key in categories}
    for row in rows:
        category = row.get("category")
        if category in totals:
            totals[category] += float(row["accessibility"])
            counts[category] += 1
    category_scores = {
        key: (totals[key] / counts[key] if counts[key] else 0.0) for key in categories
    }
    if rows and sum(counts.values()) > 0:
        distribution = normalize_distribution(
            {key: totals[key] for key in categories}, categories
        )
    else:
        distribution = {key: 1.0 / len(categories) for key in categories}
    overall = (
        sum(float(row["accessibility"]) for row in rows) / len(rows) if rows else 0.0
    )
    return {
        "availability_score": overall,
        "category_accessibility": category_scores,
        "availability_distribution": distribution,
        "retrieved_count": len(rows),
        "population_frequency_known": bool(problem.base_rates),
        "accessibility_is_not_frequency": True,
    }


def _weighted_cosine(
    left: Mapping[str, float],
    right: Mapping[str, float],
    weights: Mapping[str, float],
) -> float | None:
    keys = sorted(set(left) | set(right))
    if not keys:
        return None
    w = {key: max(0.0, float(weights.get(key, 1.0))) for key in keys}
    dot = sum(w[key] * float(left.get(key, 0.0)) * float(right.get(key, 0.0)) for key in keys)
    norm_left = sqrt(sum(w[key] * float(left.get(key, 0.0)) ** 2 for key in keys))
    norm_right = sqrt(sum(w[key] * float(right.get(key, 0.0)) ** 2 for key in keys))
    if norm_left <= _EPS or norm_right <= _EPS:
        return None
    return clamp((dot / (norm_left * norm_right) + 1.0) / 2.0)


def representativeness_heuristic(
    problem: JudgmentProblem,
    categories: Sequence[str],
) -> dict[str, Any]:
    """Compare the case with category prototypes using weighted similarity."""
    scores: dict[str, float] = {}
    for category in categories:
        if category in problem.similarity_scores:
            scores[category] = clamp(float(problem.similarity_scores[category]))
        elif category in problem.prototypes:
            similarity = _weighted_cosine(
                problem.features, problem.prototypes[category], problem.feature_weights
            )
            scores[category] = 0.5 if similarity is None else similarity
        else:
            scores[category] = 0.5
    distribution = softmax({key: value - 0.5 for key, value in scores.items()})
    return {
        "similarity_scores": scores,
        "similarity_distribution": distribution,
        "feature_count": len(problem.features),
        "prototype_count": len(problem.prototypes),
        "similarity_is_not_probability": True,
    }


def bayesian_base_rate_check(
    problem: JudgmentProblem,
    categories: Sequence[str],
) -> dict[str, Any]:
    """Combine supplied priors with category likelihoods, retaining neutral missing likelihoods."""
    prior = prior_distribution(problem, categories)
    raw_likelihoods = problem.evidence.get("likelihoods")
    likelihoods: dict[str, float] = {}
    if isinstance(raw_likelihoods, Mapping):
        for category in categories:
            value = raw_likelihoods.get(category, 1.0)
            likelihoods[category] = clamp(float(value))
    elif "likelihood_if_true" in problem.evidence and len(categories) == 2:
        likelihoods[categories[0]] = clamp(float(problem.evidence["likelihood_if_true"]))
        likelihoods[categories[1]] = clamp(float(problem.evidence.get("likelihood_if_false", 1.0)))
    else:
        likelihoods = {key: 1.0 for key in categories}
    unnormalized = {key: prior[key] * likelihoods[key] for key in categories}
    total = sum(unnormalized.values())
    evidence_available = (
        isinstance(raw_likelihoods, Mapping)
        or "likelihood_if_true" in problem.evidence
    )
    if total <= _EPS:
        posterior = prior
        warning = "Supplied likelihoods assign zero probability to every category; prior retained."
    else:
        posterior = {key: value / total for key, value in unnormalized.items()}
        warning = None
    return {
        "prior_distribution": prior,
        "likelihoods": likelihoods,
        "posterior_distribution": posterior,
        "likelihood_evidence_supplied": evidence_available,
        "posterior_updated": evidence_available and total > _EPS,
        "warnings": [warning] if warning else [],
        "normalization_denominator": total,
        "bayes_rule_applied": evidence_available and total > _EPS,
    }


def adjust_anchor(
    problem: JudgmentProblem,
    config: Point22Config,
) -> dict[str, Any]:
    """Move a starting estimate toward an evidence estimate by a bounded adjustment fraction."""
    if not problem.candidate_anchors:
        return {"applied": False, "estimate": None, "reason": "No candidate anchor supplied."}
    anchor = float(problem.candidate_anchors[0])
    estimate = problem.evidence.get("evidence_estimate")
    if estimate is None:
        return {
            "applied": False, "anchor": anchor, "estimate": None,
            "reason": "An anchor was present but no independent evidence estimate was supplied.",
        }
    target = float(estimate)
    fraction = float(config.adjustment_fraction)
    result = anchor + fraction * (target - anchor)
    return {
        "applied": True,
        "anchor": anchor,
        "evidence_estimate": target,
        "adjustment_fraction": fraction,
        "estimate": result,
        "anchor_influence_remaining": abs(result - target),
    }


def monitor_fluency(problem: JudgmentProblem) -> dict[str, Any]:
    """Represent familiarity as a metacognitive cue separate from corroborating evidence."""
    exposure = max(0, int(problem.evidence.get("exposure_count", 0)))
    independent_sources = max(0, int(problem.evidence.get("independent_source_count", 0)))
    fluency = max(float(problem.fluency), min(1.0, exposure / 10.0))
    evidence_confidence = float(problem.evidence_confidence)
    subjective = clamp(evidence_confidence + 0.2 * fluency)
    repeated_source_warning = exposure > independent_sources and exposure > 1
    warnings = []
    if repeated_source_warning:
        warnings.append("Repeated exposure may increase familiarity without adding independent evidence.")
    return {
        "fluency": fluency,
        "exposure_count": exposure,
        "independent_source_count": independent_sources,
        "evidence_confidence": evidence_confidence,
        "subjective_confidence": subjective,
        "fluency_inflation": subjective - evidence_confidence,
        "warnings": warnings,
        "fluency_separated_from_evidence": True,
    }


def estimate_sample_uncertainty(problem: JudgmentProblem, config: Point22Config) -> dict[str, Any]:
    """Compute standard errors and a simple interval when the supplied sampling design allows it."""
    evidence = problem.evidence
    n_raw = evidence.get("sample_n")
    if n_raw is None:
        return {"available": False, "standard_error": None, "warnings": []}
    n = int(n_raw)
    if n <= 0:
        raise ValueError("sample_n must be positive")
    dependence = float(evidence.get("design_effect", 1.0))
    if not isfinite(dependence) or dependence < 1.0:
        raise ValueError("design_effect must be finite and at least 1")
    effective_n = n / dependence
    if "sample_proportion" in evidence:
        p = clamp(float(evidence["sample_proportion"]))
        se = sqrt(max(0.0, p * (1.0 - p)) / effective_n)
        estimate = p
        interval_type = "normal_approximation_for_sample_proportion"
    elif "sample_sd" in evidence:
        sd = float(evidence["sample_sd"])
        if not isfinite(sd) or sd < 0:
            raise ValueError("sample_sd must be finite and non-negative")
        se = sd / sqrt(effective_n)
        estimate = float(evidence.get("sample_mean", 0.0))
        interval_type = "normal_approximation_for_sample_mean"
    else:
        return {
            "available": True, "sample_n": n, "effective_sample_size": effective_n,
            "standard_error": None,
            "warnings": ["Sample size supplied without sample_sd or sample_proportion."],
        }
    margin = 1.96 * se
    warnings = []
    if n < config.small_sample_threshold:
        warnings.append("Small sample: inspect sampling variability and seek replication.")
    if dependence > 1.0:
        warnings.append("Dependence/design effect reduces the effective sample size.")
    return {
        "available": True,
        "sample_n": n,
        "effective_sample_size": effective_n,
        "estimate": estimate,
        "standard_error": se,
        "interval_95_approx": [estimate - margin, estimate + margin],
        "interval_type": interval_type,
        "warnings": warnings,
    }


def correlation_from_values(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    """Calculate Pearson correlation for paired finite observations."""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x = [float(item) for item in x_values]
    y = [float(item) for item in y_values]
    if any(not isfinite(item) for item in x + y):
        raise ValueError("correlation inputs must be finite")
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    dx = [item - mean_x for item in x]
    dy = [item - mean_y for item in y]
    denom = sqrt(sum(item * item for item in dx) * sum(item * item for item in dy))
    if denom <= _EPS:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def check_correlation_causation(problem: JudgmentProblem) -> dict[str, Any]:
    """List missing causal-identification checks without turning association into a cause claim."""
    evidence = problem.evidence
    correlation = evidence.get("correlation")
    if correlation is None and "x_values" in evidence and "y_values" in evidence:
        correlation = correlation_from_values(evidence["x_values"], evidence["y_values"])
    warnings = []
    observational = bool(evidence.get("observational_association", correlation is not None))
    temporal = bool(evidence.get("temporal_order_known", False))
    confounders = bool(evidence.get("confounders_examined", False))
    reverse = bool(evidence.get("reverse_causality_examined", False))
    selection = bool(evidence.get("selection_bias_possible", False))
    intervention = bool(evidence.get("intervention_evidence", False))
    if observational:
        warnings.append("Observed association alone does not establish a causal effect.")
    if not temporal:
        warnings.append("Temporal precedence has not been established.")
    if not confounders:
        warnings.append("Plausible common causes have not been examined.")
    if not reverse:
        warnings.append("Reverse causation has not been ruled out.")
    if selection:
        warnings.append("Selection effects may distort the observed association.")
    status = "stronger_causal_design_available" if intervention else (
        "causal_identification_incomplete" if observational else "no_association_claim_supplied"
    )
    return {
        "correlation": float(correlation) if correlation is not None else None,
        "observational_association": observational,
        "temporal_order_known": temporal,
        "confounders_examined": confounders,
        "reverse_causality_examined": reverse,
        "selection_bias_possible": selection,
        "intervention_evidence": intervention,
        "causal_status": status,
        "warnings": warnings,
        "association_is_not_causation": True,
    }


def weight_salience_and_urgency(problem: JudgmentProblem, config: Point22Config) -> dict[str, Any]:
    """Compute normalized cue-attention weights from utility and salience signals."""
    cues = {
        "memory_examples": 0.35 * problem.emotional_salience + 0.25 * problem.fluency,
        "base_rate_check": 0.40 * (1.0 - problem.time_pressure) + 0.25 * problem.decision_cost,
        "alternative_search": 0.45 * problem.decision_cost + 0.30 * (1.0 - problem.time_pressure),
        "rapid_response": 0.55 * problem.time_pressure + 0.20 * problem.fluency,
    }
    weights = softmax(cues, temperature=config.attention_temperature)
    risk = max(problem.decision_cost, problem.emotional_salience, problem.time_pressure)
    return {
        "cue_attention_weights": weights,
        "risk_signal": risk,
        "verification_priority": clamp(0.5 * problem.decision_cost + 0.3 * problem.time_pressure
                                       + 0.2 * problem.emotional_salience),
        "high_urgency": problem.time_pressure >= config.high_risk_threshold,
        "high_salience": problem.emotional_salience >= config.high_risk_threshold,
        "emotion_can_help_or_hinder_attention": True,
    }


def build_intuitive_judgment(
    problem: JudgmentProblem,
    categories: Sequence[str],
    availability: Mapping[str, Any],
    representativeness: Mapping[str, Any],
    bayes: Mapping[str, Any],
    fluency: Mapping[str, Any],
    state: Point22State,
    config: Point22Config,
) -> dict[str, Any]:
    """Combine shortcut cues into an initial distribution and a distinct subjective confidence."""
    avail = availability["availability_distribution"]
    rep = representativeness["similarity_scores"]
    prior = bayes["prior_distribution"]
    weights = state.cue_weights
    scores: dict[str, float] = {}
    for category in categories:
        scores[category] = (
            float(weights.get("availability", 1.0)) * 0.25 * log(max(_EPS, float(avail[category])))
            + float(weights.get("representativeness", 1.0)) * 0.45 * float(rep[category])
            + float(weights.get("base_rate", 1.0)) * 0.30 * log(max(_EPS, float(prior[category])))
        )
    probabilities = softmax(scores)
    answer = max(probabilities, key=probabilities.get)
    return {
        "answer": answer,
        "probabilities": probabilities,
        "category_scores": scores,
        "confidence": clamp(0.70 * max(probabilities.values())
                            + 0.30 * float(fluency["subjective_confidence"])),
        "evidence_confidence": float(fluency["evidence_confidence"]),
        "subjective_fluency": float(fluency["fluency"]),
        "cue_weights_used": {key: float(value) for key, value in weights.items()},
        "confidence_is_not_accuracy": True,
    }


def detect_reasoning_conflict(
    problem: JudgmentProblem,
    intuitive: Mapping[str, Any],
    bayes: Mapping[str, Any],
    sampling: Mapping[str, Any],
    causal: Mapping[str, Any],
    config: Point22Config,
) -> dict[str, Any]:
    """Find disagreements, unsupported confidence, and conditions that require slower checking."""
    categories = list(intuitive["probabilities"])
    p = intuitive["probabilities"]
    q = bayes["posterior_distribution"]
    divergence = 0.5 * sum(abs(float(p[key]) - float(q[key])) for key in categories)
    warnings = []
    if divergence >= config.conflict_threshold:
        warnings.append("The intuitive judgment differs from the base-rate/likelihood check.")
    if intuitive["confidence"] - intuitive["evidence_confidence"] > 0.2:
        warnings.append("Subjective confidence exceeds the supplied evidence-confidence estimate.")
    if sampling.get("warnings"):
        warnings.extend(sampling["warnings"])
    if causal.get("warnings") and problem.target_type == "cause":
        warnings.extend(causal["warnings"])
    if problem.candidate_anchors and "evidence_estimate" not in problem.evidence:
        warnings.append("An initial anchor is present without an independent evidence estimate.")
    high_stakes = max(problem.decision_cost, problem.emotional_salience) >= config.high_risk_threshold
    urgent = problem.time_pressure >= config.high_risk_threshold
    trigger = (
        divergence >= config.conflict_threshold
        or high_stakes
        or urgent
        or bool(sampling.get("warnings"))
        or (problem.target_type == "cause" and bool(causal.get("warnings")))
    )
    return {
        "total_variation_distance": divergence,
        "warnings": list(dict.fromkeys(warnings)),
        "high_stakes": high_stakes,
        "time_pressure_high": urgent,
        "verification_triggered": trigger,
        "conflict_threshold": config.conflict_threshold,
    }


def verify_judgment(
    problem: JudgmentProblem,
    categories: Sequence[str],
    bayes: Mapping[str, Any],
    sampling: Mapping[str, Any],
    causal: Mapping[str, Any],
) -> dict[str, Any]:
    """Perform the explicit prior/likelihood pass and preserve uncertainty and causal caveats."""
    posterior = dict(bayes["posterior_distribution"])
    max_probability = max(posterior.values())
    sample_penalty = 0.0
    if sampling.get("standard_error") is not None:
        sample_penalty = min(0.25, float(sampling["standard_error"]))
    confidence = clamp(0.65 * max_probability + 0.35 * float(problem.evidence_confidence)
                       - sample_penalty)
    answer: Any = max(posterior, key=posterior.get)
    assumptions = [
        "Priors are taken from the supplied base_rates or a uniform fallback.",
        "Supplied category likelihoods are treated as conditional on the candidate category.",
        "Missing category likelihoods are neutral (1.0), not treated as negative evidence.",
    ]
    if not bayes["likelihood_evidence_supplied"]:
        assumptions.append("No diagnostic likelihoods were supplied; the prior distribution is retained.")
    if sampling.get("standard_error") is None and sampling.get("available"):
        assumptions.append("Sample size was supplied without enough dispersion information for a standard error.")
    if problem.target_type == "magnitude":
        estimate = problem.evidence.get("evidence_estimate")
        if estimate is not None:
            answer = float(estimate)
    return {
        "answer": answer,
        "probabilities": posterior,
        "confidence": confidence,
        "max_probability": max_probability,
        "sample_uncertainty_penalty": sample_penalty,
        "assumptions": assumptions,
        "warnings": list(bayes.get("warnings", [])) + list(sampling.get("warnings", []))
                    + (list(causal.get("warnings", [])) if problem.target_type == "cause" else []),
        "diagnostic_likelihoods_used": bool(bayes["likelihood_evidence_supplied"]),
        "conditional_probability_checks": dict(problem.evidence.get("conditional_probabilities", {})),
        "repeated_claims_not_counted_as_independent_by_default": True,
    }


def evaluate_ecological_fit(
    problem: JudgmentProblem,
    strategy: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate known cue validity and environmental stability without assuming one strategy is universal."""
    validity = dict(problem.cue_validity)
    selected = list(strategy["selected_heuristics"])
    mapped = {
        "availability": "availability",
        "representativeness": "representativeness",
        "anchoring-and-adjustment": "anchor",
        "processing-fluency": "fluency",
        "base-rate/default-prior": "base_rate",
    }
    used_values = [float(validity[mapped[item]]) for item in selected if mapped[item] in validity]
    known = bool(used_values)
    mean_validity = sum(used_values) / len(used_values) if used_values else None
    score = (
        clamp(float(mean_validity) * float(problem.environment_stability))
        if mean_validity is not None else None
    )
    robust = (score >= 0.55) if score is not None else None
    return {
        "cue_validity": validity,
        "selected_heuristics": selected,
        "mean_selected_cue_validity": mean_validity,
        "environment_stability": float(problem.environment_stability),
        "ecological_fit_score": score,
        "robust_in_supplied_environment": robust,
        "out_of_sample_evaluation_required": True,
        "strategy_cost_tradeoff": "A simpler rule may outperform a complex model in noisy environments; measure risk on held-out cases.",
        "validity_known": known,
    }


def learn_from_feedback(
    problem: JudgmentProblem,
    intuitive: Mapping[str, Any],
    verified: Mapping[str, Any],
    state: Point22State,
    config: Point22Config,
) -> dict[str, Any]:
    """Record prediction error and apply only attributable, gradual cue-weight updates."""
    outcome = problem.evidence.get("observed_outcome")
    probabilities = dict(verified["probabilities"])
    brier = None
    delta = None
    if outcome in probabilities:
        brier = sum((float(probabilities[key]) - (1.0 if key == outcome else 0.0)) ** 2
                    for key in probabilities)
        delta = 1.0 - float(probabilities[outcome])
    cue_feedback = problem.evidence.get("cue_feedback", {})
    updates = {}
    if isinstance(cue_feedback, Mapping):
        for cue, signed_signal in cue_feedback.items():
            old = float(state.cue_weights.get(str(cue), 1.0))
            new = max(0.1, min(2.0, old + config.learning_rate * float(signed_signal)))
            state.cue_weights[str(cue)] = new
            updates[str(cue)] = {"before": old, "after": new}
    row = {
        "problem_id": problem.problem_id,
        "observed_outcome": outcome,
        "prediction_error": delta,
        "brier_score": brier,
        "cue_weights_updated": bool(updates),
        "cue_updates": updates,
    }
    if outcome in probabilities or updates:
        state.prediction_history.append(row)
    state.prediction_history = state.prediction_history[-500:]
    return {
        **row,
        "feedback_available": outcome in probabilities or bool(updates),
        "learning_rate": config.learning_rate,
        "overcorrection_guard": "Cue weights are clipped to [0.1, 2.0]; no cue is reattributed without supplied cue_feedback.",
        "updated_cue_weights": dict(state.cue_weights),
    }


def make_final_decision(
    problem: JudgmentProblem,
    verified: Mapping[str, Any],
    ecological: Mapping[str, Any],
    conflict: Mapping[str, Any],
    feedback: Mapping[str, Any],
    config: Point22Config,
) -> dict[str, Any]:
    """Select by verified probabilities/expected loss or explicitly defer when uncertainty is high."""
    probabilities = dict(verified["probabilities"])
    losses = problem.evidence.get("expected_losses")
    expected_loss: dict[str, float] = {}
    if isinstance(losses, Mapping):
        for decision, row in losses.items():
            if isinstance(row, Mapping):
                expected_loss[str(decision)] = sum(
                    float(probabilities.get(outcome, 0.0)) * float(loss)
                    for outcome, loss in row.items()
                )
            else:
                expected_loss[str(decision)] = float(row)
    if expected_loss:
        choice = min(expected_loss, key=expected_loss.get)
        choice_basis = "minimum expected loss"
    else:
        choice = max(probabilities, key=probabilities.get)
        choice_basis = "highest verified posterior probability"
    max_probability = max(probabilities.values())
    high_cost = max(problem.decision_cost, problem.emotional_salience) >= config.high_risk_threshold
    defer = max_probability < 0.45 and problem.decision_cost >= 0.5
    review = high_cost or conflict["verification_triggered"] or defer
    if defer:
        final_answer: Any = "defer_or_collect_more_evidence"
    elif problem.target_type == "magnitude" and verified["answer"] is not None:
        final_answer = verified["answer"]
    else:
        final_answer = choice
    return {
        "answer": final_answer,
        "choice_basis": choice_basis,
        "expected_losses": expected_loss,
        "review_required": review,
        "deferred": defer,
        "confidence": float(verified["confidence"]),
        "probabilities": probabilities,
        "ecological_fit_known": ecological["validity_known"],
        "feedback_recorded": bool(feedback["feedback_available"]),
        "assumptions_and_warnings_preserved": True,
    }


def validate_point22(
    trace: Sequence[Mapping[str, Any]],
    probabilities: Mapping[str, float],
    confidence: float,
    categories: Sequence[str],
) -> dict[str, Any]:
    """Check operation order, distribution normalization, and bounded confidence."""
    expected = [f"P22-{index:02d}" for index in range(1, 18)]
    observed = [str(row.get("operation_id")) for row in trace]
    failures = []
    if observed != expected:
        failures.append("operation_order_mismatch")
    if set(probabilities) != set(categories):
        failures.append("probability_category_mismatch")
    if not isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
        failures.append("confidence_out_of_bounds")
    if probabilities and abs(sum(float(value) for value in probabilities.values()) - 1.0) > 1e-8:
        failures.append("probabilities_not_normalized")
    return {
        "operation_order_verified": observed == expected,
        "operation_count": len(trace),
        "probabilities_normalized": bool(probabilities)
            and abs(sum(float(value) for value in probabilities.values()) - 1.0) <= 1e-8,
        "confidence_bounded": isfinite(float(confidence)) and 0.0 <= float(confidence) <= 1.0,
        "failure_signatures": failures,
        "valid": not failures,
    }
