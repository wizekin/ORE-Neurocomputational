"""Functional computations for Point 24 feedback and misconception correction."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contracts import (
    ClarificationOption, CorrectionRequest, CorrectiveConfig, FeedbackEvidence,
    KnowledgeHypothesis,
)


EPSILON = 1e-12
DIRECTNESS = {"direct": 1.0, "indirect": 0.75, "hearsay": 0.45}


def normalize(values: Mapping[str, float]) -> dict[str, float]:
    """Normalize non-negative weights, with a uniform fallback for an empty mass."""
    clean = {key: max(0.0, float(value)) for key, value in values.items()}
    total = sum(clean.values())
    if total <= EPSILON:
        return {key: 1.0 / len(clean) for key in clean} if clean else {}
    return {key: value / total for key, value in clean.items()}


def entropy(probabilities: Mapping[str, float]) -> float:
    """Shannon entropy in nats."""
    return -sum(float(p) * math.log(max(EPSILON, float(p))) for p in probabilities.values() if p > 0)


def normalized_entropy(probabilities: Mapping[str, float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    return entropy(probabilities) / math.log(len(probabilities))


def hypothesis_map(request: CorrectionRequest) -> dict[str, KnowledgeHypothesis]:
    return {item.hypothesis_id: item for item in request.hypotheses}


def prior_distribution(request: CorrectionRequest) -> dict[str, float]:
    return normalize({item.hypothesis_id: item.prior for item in request.hypotheses})


def generate_predictions(
    request: CorrectionRequest, probabilities: Mapping[str, float]
) -> dict[str, Any]:
    """Combine candidate predictions, keeping absent predictions explicit."""
    outcomes: dict[str, float] = {}
    for item in request.hypotheses:
        mass = probabilities.get(item.hypothesis_id, 0.0)
        row = normalize(item.predictions)
        for outcome, probability in row.items():
            outcomes[outcome] = outcomes.get(outcome, 0.0) + mass * probability
    distribution = normalize(outcomes)
    numeric_mean = None
    if request.predicted_value_by_hypothesis:
        numeric_mean = sum(
            probabilities.get(key, 0.0) * float(value)
            for key, value in request.predicted_value_by_hypothesis.items()
        )
    return {
        "outcome_distribution": distribution,
        "predicted_numeric_value": numeric_mean,
        "candidate_predictions": {
            item.hypothesis_id: normalize(item.predictions) for item in request.hypotheses
        },
        "distinguishing_outcomes": sorted({
            outcome for item in request.hypotheses for outcome in item.predictions
            if len({round(normalize(h.predictions).get(outcome, 0.0), 6) for h in request.hypotheses}) > 1
        }),
    }


def detect_mismatch(request: CorrectionRequest, prediction: Mapping[str, Any], config: CorrectiveConfig) -> dict:
    """Measure observed/predicted mismatch; surprise alone never proves a misconception."""
    details: dict[str, Any] = {}
    scores: list[float] = []
    distribution = prediction["outcome_distribution"]
    if request.observed_outcome is not None and distribution:
        probability = float(distribution.get(request.observed_outcome, 0.0))
        surprisal = -math.log(max(EPSILON, probability))
        score = 1.0 - probability
        details.update({
            "observed_outcome": request.observed_outcome,
            "predicted_probability": probability,
            "surprisal_nats": surprisal,
            "discrete_mismatch": score,
        })
        scores.append(score)
    residual = None
    if request.observed_value is not None and prediction["predicted_numeric_value"] is not None:
        residual = float(request.observed_value) - float(prediction["predicted_numeric_value"])
        scale = max(1.0, abs(float(request.observed_value)), abs(float(prediction["predicted_numeric_value"])))
        score = min(1.0, abs(residual) / scale)
        details.update({"numeric_residual": residual, "normalized_numeric_mismatch": score})
        scores.append(score)
    score = max(scores, default=0.0)
    return {
        **details,
        "mismatch_score": score,
        "mismatch_detected": score >= config.mismatch_threshold,
        "observation_available": bool(scores),
        "caution": "Surprise or discrepancy is a trigger for evaluation, not proof that the current interpretation is false.",
    }


ERROR_CLASSES = (
    "perceptual_input", "lexical_semantic", "syntactic_relational", "context",
    "knowledge", "inference", "source_monitoring", "confidence_calibration", "procedure",
)


def classify_error(request: CorrectionRequest, mismatch: Mapping[str, Any]) -> dict[str, float]:
    """Retain all hinted causes and normalize them instead of forcing one diagnosis."""
    weights = {key: float(value) for key, value in request.error_hints.items() if key in ERROR_CLASSES}
    if not weights and mismatch["mismatch_detected"]:
        weights = {"knowledge": 0.35, "inference": 0.30, "context": 0.20, "source_monitoring": 0.15}
    return normalize(weights)


def expected_information_gain(
    probabilities: Mapping[str, float], option: ClarificationOption
) -> float:
    """Compute entropy reduction expected under the option's supplied response model."""
    before = entropy(probabilities)
    response_mass: dict[str, float] = {}
    conditional: dict[str, dict[str, float]] = {}
    for hypothesis_id, outcomes in option.response_models.items():
        if hypothesis_id in probabilities:
            conditional[hypothesis_id] = normalize(outcomes)
            for response, value in conditional[hypothesis_id].items():
                response_mass[response] = response_mass.get(response, 0.0) + probabilities[hypothesis_id] * value
    response_mass = normalize(response_mass)
    expected_after = 0.0
    for response, response_probability in response_mass.items():
        unnormalized = {
            hypothesis_id: probabilities.get(hypothesis_id, 0.0) * outcomes.get(response, 0.0)
            for hypothesis_id, outcomes in conditional.items()
        }
        expected_after += response_probability * entropy(normalize(unnormalized))
    return max(0.0, before - expected_after)


def choose_information_action(
    request: CorrectionRequest, probabilities: Mapping[str, float], mismatch: Mapping[str, Any]
) -> dict[str, Any]:
    """Select the most discriminating available action while accounting for its cost."""
    if not mismatch["mismatch_detected"]:
        return {"action": "continue", "selected_option": None, "expected_information_gain": 0.0, "reason": "no actionable mismatch"}
    if len(probabilities) <= 1:
        return {"action": "check", "selected_option": None, "expected_information_gain": 0.0, "reason": "single hypothesis; verify the observation"}
    scored = []
    for option in request.clarification_options:
        gain = expected_information_gain(probabilities, option)
        utility = gain - 0.25 * option.cost
        scored.append({
            "action_id": option.action_id, "kind": option.kind, "description": option.description,
            "expected_information_gain": gain, "cost": option.cost, "utility": utility,
        })
    if scored:
        best = max(scored, key=lambda item: item["utility"])
        return {
            "action": best["kind"], "selected_option": best,
            "expected_information_gain": best["expected_information_gain"],
            "ranked_options": sorted(scored, key=lambda item: item["utility"], reverse=True),
            "reason": "highest expected information gain after action cost",
        }
    fallback = "clarify" if request.task_stakes != "low" else "check"
    return {
        "action": fallback, "selected_option": None, "expected_information_gain": 0.0,
        "reason": "competing hypotheses remain and no clarification response model was supplied",
        "unresolved": True,
    }


def assess_evidence(evidence: Sequence[FeedbackEvidence], hypothesis_ids: Sequence[str]) -> dict[str, Any]:
    """Calculate reliability weights and collapse dependent evidence into one group contribution."""
    items = []
    grouped: dict[str, list[tuple[FeedbackEvidence, float]]] = {}
    for item in evidence:
        weight = (
            item.reliability * item.relevance * item.diagnosticity * DIRECTNESS[item.directness]
        )
        group = item.independence_key.strip() or item.evidence_id
        grouped.setdefault(group, []).append((item, weight))
        items.append({
            "evidence_id": item.evidence_id, "source_name": item.source_name,
            "source_type": item.source_type, "independence_group": group,
            "reliability": item.reliability, "relevance": item.relevance,
            "diagnosticity": item.diagnosticity, "directness": item.directness,
            "effective_weight": weight, "description": item.description,
        })
    contributions: dict[str, dict[str, float]] = {}
    for group, members in grouped.items():
        total = sum(weight for _, weight in members)
        if total <= EPSILON:
            continue
        row = {}
        for hypothesis_id in hypothesis_ids:
            mean_log_likelihood = sum(
                weight * math.log(max(EPSILON, item.likelihoods.get(hypothesis_id, 1.0)))
                for item, weight in members
            ) / total
            row[hypothesis_id] = mean_log_likelihood
        contributions[group] = row
    quality = (
        sum(max(weight for _, weight in members) for members in grouped.values()) / len(grouped)
        if grouped else 0.0
    )
    return {
        "items": items,
        "independent_group_count": len(contributions),
        "group_log_likelihoods": contributions,
        "quality": min(1.0, quality),
        "dependent_sources_collapsed": sum(max(0, len(members) - 1) for members in grouped.values()),
        "source_independence_is_caller_supplied": True,
    }


def bayesian_update(
    priors: Mapping[str, float], assessment: Mapping[str, Any]
) -> dict[str, float]:
    """Update priors with one weighted log-likelihood contribution per independent group."""
    log_scores = {key: math.log(max(EPSILON, value)) for key, value in priors.items()}
    for likelihoods in assessment["group_log_likelihoods"].values():
        for key in log_scores:
            log_scores[key] += float(likelihoods.get(key, 0.0))
    ceiling = max(log_scores.values(), default=0.0)
    scores = {key: math.exp(value - ceiling) for key, value in log_scores.items()}
    return normalize(scores)


def build_correction(
    request: CorrectionRequest, priors: Mapping[str, float], posterior: Mapping[str, float]
) -> dict[str, Any]:
    """Select only a supplied evidence-supported candidate and preserve competing alternatives."""
    ranked = sorted(posterior, key=posterior.get, reverse=True)
    selected = ranked[0] if ranked else request.current_hypothesis_id
    current_probability = posterior.get(request.current_hypothesis_id, 0.0)
    selected_probability = posterior.get(selected, 0.0)
    hypotheses = {item.hypothesis_id: item for item in request.hypotheses}
    return {
        "previous_hypothesis_id": request.current_hypothesis_id,
        "previous_statement": hypotheses[request.current_hypothesis_id].statement,
        "selected_hypothesis_id": selected,
        "selected_statement": hypotheses[selected].statement,
        "probability_before": priors.get(request.current_hypothesis_id, 0.0),
        "probability_after": current_probability,
        "selected_probability": selected_probability,
        "interpretation_changed": selected != request.current_hypothesis_id,
        "ranked_alternatives": [
            {"hypothesis_id": key, "statement": hypotheses[key].statement, "posterior": posterior[key]}
            for key in ranked
        ],
        "candidate_generation_scope": "caller-supplied candidates only; no unsupported facts or hypotheses invented",
        "posterior_support_not_equivalent_to_verified_truth": request.verified_hypothesis_id is None,
    }


def update_components(
    request: CorrectionRequest,
    posterior: Mapping[str, float],
    config: CorrectiveConfig,
) -> dict[str, Any]:
    """Apply reliability- and scope-gated updates only to named components."""
    hypotheses = {item.hypothesis_id: item for item in request.hypotheses}
    selected_id = max(posterior, key=posterior.get)
    selected = hypotheses[selected_id]
    strength = min(1.0, posterior[selected_id] * request.context_relevance * config.learning_rate)
    affected = set(request.affected_components)
    if not affected:
        affected = set(selected.components)
    updated = {}
    preserved = []
    previous_hypothesis = hypotheses[request.current_hypothesis_id]
    for name, value in selected.components.items():
        if name not in affected:
            preserved.append(name)
            continue
        old = previous_hypothesis.components.get(name)
        if isinstance(value, (int, float)) and isinstance(old, (int, float)):
            new_value = float(old) + strength * (float(value) - float(old))
        else:
            new_value = value if strength >= 0.05 else old
        updated[name] = {"previous": old, "target": value, "updated": new_value, "gate": strength}
    return {
        "selected_hypothesis_id": selected_id,
        "update_strength": strength,
        "affected_components": sorted(affected),
        "component_updates": updated,
        "preserved_component_names": preserved,
        "scope_limited": bool(request.affected_components),
    }


def build_explanation(request: CorrectionRequest, selected_id: str) -> dict[str, Any]:
    """Expose graph-like explanation content and its applicability boundaries."""
    selected = next(item for item in request.hypotheses if item.hypothesis_id == selected_id)
    supplied = dict(request.explanation)
    nodes = list(supplied.get("nodes", []))
    edges = list(supplied.get("edges", []))
    return {
        "graph": {"nodes": nodes, "edges": edges, "edge_count": len(edges)},
        "corrected_statement": selected.statement,
        "conditions": list(supplied.get("conditions", [])),
        "example": supplied.get("example"),
        "non_example": supplied.get("non_example"),
        "boundary_cases": list(supplied.get("boundary_cases", [])),
        "learner_explanation": supplied.get("learner_explanation", ""),
        "relational_model_supplied": bool(nodes or edges),
        "scope_warning": "A verbal explanation is not treated as proof of conceptual accuracy.",
    }


def score_retrieval(request: CorrectionRequest, selected_id: str, config: CorrectiveConfig) -> dict[str, Any]:
    attempt = request.retrieval_attempt
    if attempt is None:
        return {"attempted": False, "success": None, "accuracy": None, "delay_days": None, "accessibility": None}
    success = attempt.hypothesis_id == selected_id
    return {
        "attempted": True, "success": success, "accuracy": float(success),
        "response_hypothesis_id": attempt.hypothesis_id, "target_hypothesis_id": selected_id,
        "response_text": attempt.response_text, "reported_confidence": attempt.confidence,
        "delay_days": attempt.delay_days,
        "accessibility_after_delay": math.exp(-config.forgetting_rate * attempt.delay_days),
    }


def retrieval_feedback(retrieval: Mapping[str, Any], selected_id: str) -> dict[str, Any]:
    if not retrieval["attempted"]:
        return {"feedback_required": False, "message": "No retrieval attempt was supplied."}
    if retrieval["success"]:
        return {"feedback_required": False, "message": "Retrieval matches the selected interpretation.", "target_hypothesis_id": selected_id}
    return {
        "feedback_required": True,
        "message": "The response differs from the selected interpretation; restate the discriminating relation and retry.",
        "target_hypothesis_id": selected_id,
        "residual_error": True,
        "old_response_is_not_reinforced_as_truth": True,
    }


def score_transfer(request: CorrectionRequest, threshold: float = 0.75) -> dict[str, Any]:
    probes = []
    for item in request.transfer_probes:
        correct = item.response_hypothesis_id == item.target_hypothesis_id
        probes.append({
            "probe_id": item.probe_id, "context": item.context,
            "target_hypothesis_id": item.target_hypothesis_id,
            "response_hypothesis_id": item.response_hypothesis_id,
            "is_counterexample": item.is_counterexample, "correct": correct,
        })
    score = sum(row["correct"] for row in probes) / len(probes) if probes else None
    return {
        "attempted": bool(probes), "score": score, "passed": None if score is None else score >= threshold,
        "probe_results": probes, "counterexamples_tested": sum(row["is_counterexample"] for row in probes),
        "surface_memorization_warning": "Transfer evidence is stronger when surface features differ from the correction example.",
    }


def brier_score(observations: Sequence[Mapping[str, float]]) -> float | None:
    if not observations:
        return None
    return sum((float(row["confidence"]) - float(row["outcome"])) ** 2 for row in observations) / len(observations)


def consolidate_state(
    state: Any,
    selected_id: str,
    previous_id: str,
    retrieval: Mapping[str, Any],
    posterior: Mapping[str, float],
    request: CorrectionRequest,
    config: CorrectiveConfig,
) -> dict[str, Any]:
    """Decay accessibility, reinforce retrieval, and increment correction consolidation."""
    old_access = state.belief_accessibility.get(previous_id, 0.5)
    correction_access = state.correction_accessibility.get(selected_id, 0.25)
    delay = float(retrieval.get("delay_days") or 0.0)
    old_after_decay = old_access * math.exp(-config.forgetting_rate * delay)
    correction_after_decay = correction_access * math.exp(-config.forgetting_rate * delay)
    retrieval_success = bool(retrieval.get("success")) if retrieval.get("attempted") else False
    if retrieval_success:
        correction_after_decay += config.learning_rate * (1.0 - correction_after_decay)
    encoding = posterior.get(selected_id, 0.0)
    reinforcement = float(retrieval_success) * request.context_relevance
    old_strength = state.consolidation_strength.get(selected_id, 0.0)
    strength = min(1.0, old_strength + config.consolidation_rate * encoding * (0.5 + reinforcement))
    state.belief_accessibility[previous_id] = old_after_decay
    state.correction_accessibility[selected_id] = min(1.0, correction_after_decay)
    state.consolidation_strength[selected_id] = strength
    return {
        "previous_belief_accessibility": old_after_decay,
        "correction_accessibility": min(1.0, correction_after_decay),
        "consolidation_strength": strength,
        "encoding_quality": encoding,
        "retrieval_reinforcement": reinforcement,
        "sleep_or_replay_not_assumed": True,
    }


def detect_regression(
    state: Any, previous_id: str, selected_id: str, retrieval: Mapping[str, Any]
) -> dict[str, Any]:
    old = state.belief_accessibility.get(previous_id, 0.0)
    corrected = state.correction_accessibility.get(selected_id, 0.0)
    returned_to_old = bool(retrieval.get("attempted") and retrieval.get("response_hypothesis_id") == previous_id and previous_id != selected_id)
    risk = min(1.0, max(0.0, old - corrected + (0.35 if returned_to_old else 0.0)))
    return {
        "belief_regression_detected": returned_to_old,
        "old_belief_accessibility": old,
        "corrected_belief_accessibility": corrected,
        "regression_risk": risk,
        "safeguard": "Keep the correction retrievable and label the old interpretation as superseded.",
    }


def decide_stopping(
    request: CorrectionRequest,
    mismatch: Mapping[str, Any],
    posterior: Mapping[str, float],
    assessment: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    transfer: Mapping[str, Any],
    config: CorrectiveConfig,
) -> dict[str, Any]:
    """Use uncertainty, evidence, retrieval, transfer, and task stakes; never force closure."""
    selected_id = max(posterior, key=posterior.get)
    confidence = float(posterior[selected_id])
    uncertainty = normalized_entropy(posterior)
    quality = float(assessment["quality"])
    required_evidence = config.minimum_evidence_quality * (1.20 if request.task_stakes == "high" else 1.0)
    required_posterior = min(0.98, config.correction_posterior_threshold + (0.10 if request.task_stakes == "high" else 0.0))
    retrieval_ok = retrieval.get("success") is True if retrieval["attempted"] else False
    transfer_ok = transfer.get("score") is not None and transfer["score"] >= config.transfer_threshold
    checks = {
        "uncertainty_low_enough": uncertainty <= config.uncertainty_threshold,
        "posterior_high_enough": confidence >= required_posterior,
        "evidence_quality_high_enough": quality >= min(1.0, required_evidence),
        "retrieval_passed_or_not_supplied": retrieval_ok if retrieval["attempted"] else False,
        "transfer_passed_or_not_supplied": transfer_ok if transfer["attempted"] else False,
    }
    # Retrieval and transfer are optional when not attempted, but their absence is not positive evidence.
    retrieval_gate = not retrieval["attempted"] or retrieval_ok
    transfer_gate = not transfer["attempted"] or transfer_ok
    enough = (
        mismatch["mismatch_detected"] and confidence >= required_posterior
        and uncertainty <= config.uncertainty_threshold
        and quality >= min(1.0, required_evidence)
        and retrieval_gate and transfer_gate
    )
    if not mismatch["observation_available"]:
        status = "uncertain"
        reason = "No observed outcome or measurement was supplied; absence of evidence is not a clean result."
    elif not mismatch["mismatch_detected"]:
        status = "no_correction_needed"
        reason = "The supplied observation is sufficiently compatible with the current interpretation."
    elif enough:
        status = "corrected"
        reason = "Evidence, posterior support, uncertainty, and supplied learning checks meet task thresholds."
    else:
        status = "uncertain"
        reason = "At least one critical evidence or learning check remains incomplete."
    return {
        "status": status, "stop": status in {"no_correction_needed", "corrected"},
        "reason": reason, "residual_uncertainty": uncertainty,
        "selected_posterior": confidence, "evidence_quality": quality,
        "checks": checks,
        "earliest_failed_stage": (
            "evaluate_feedback" if quality < min(1.0, required_evidence) else
            "retrieve_corrected_understanding" if not retrieval_gate else
            "test_transfer" if not transfer_gate else
            "compare_hypotheses" if confidence < required_posterior or uncertainty > config.uncertainty_threshold else None
        ),
        "thresholds_are_task_dependent": True,
    }


def validate_result(
    request: CorrectionRequest,
    posterior: Mapping[str, float],
    trace: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values = list(posterior.values())
    valid_probabilities = bool(values) and all(math.isfinite(x) and 0 <= x <= 1 for x in values)
    mass = sum(values)
    ordered = [row["operation_id"] for row in trace]
    return {
        "valid_posterior": valid_probabilities and abs(mass - 1.0) < 1e-8,
        "posterior_mass": mass,
        "operation_order_verified": ordered == [f"P24-{i:02d}" for i in range(1, 21)],
        "source_provenance_preserved": all(item.source for item in request.hypotheses),
        "unsupported_candidate_generation": False,
        "all_outputs_finite": valid_probabilities,
    }
