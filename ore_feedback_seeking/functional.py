"""Functional algorithms for seeking, evaluating, and learning from feedback."""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping

from .contracts import FeedbackConfig, FeedbackRequest, FeedbackState


ERROR_CATEGORIES = (
    "perceptual_input", "vocabulary", "knowledge_gap", "prerequisite_gap",
    "relational_error", "inference_error", "retrieval_failure",
    "working_memory_overload", "attention_control", "misleading_schema",
    "source_feedback_error", "communication_ambiguity",
)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Bound a numeric estimate to an inclusive interval."""
    return min(upper, max(lower, float(value)))


def sigmoid(value: float) -> float:
    """Apply a numerically stable logistic transformation."""
    if value >= 0.0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


def normalize_distribution(values: Mapping[str, float]) -> dict[str, float]:
    """Normalize non-negative weights, returning a uniform prior for a zero-sum input."""
    cleaned = {str(key): max(0.0, float(value)) for key, value in values.items()}
    total = sum(cleaned.values())
    if not cleaned:
        return {}
    if total <= 0.0:
        return {key: 1.0 / len(cleaned) for key in cleaned}
    return {key: value / total for key, value in cleaned.items()}


def entropy(distribution: Mapping[str, float]) -> float:
    """Calculate Shannon entropy in nats while safely ignoring zero-probability items."""
    return -sum(p * math.log(p) for p in normalize_distribution(distribution).values() if p > 0.0)


def brier_score(confidence: float, correctness: float) -> float:
    """Return the squared-probability error for one independently labelled prediction."""
    return (clamp(confidence) - clamp(correctness)) ** 2


def token_similarity(left: str, right: str) -> float:
    """Estimate lexical overlap with Jaccard similarity, not a full semantic judgment."""
    left_tokens = {token.strip(".,;:!?()[]{}\"'`").lower() for token in left.split() if token.strip()}
    right_tokens = {token.strip(".,;:!?()[]{}\"'`").lower() for token in right.split() if token.strip()}
    if not left_tokens and not right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def weighted_completeness(request: FeedbackRequest) -> float:
    """Compute criterion coverage as a normalized weighted sum."""
    if not request.criteria:
        return 1.0
    total_weight = sum(item.weight for item in request.criteria)
    if total_weight <= 0.0:
        return sum(item.achievement for item in request.criteria) / len(request.criteria)
    return clamp(sum(item.weight * item.achievement for item in request.criteria) / total_weight)


def _prior(request: FeedbackRequest) -> dict[str, float]:
    """Return normalized hypothesis priors or a neutral empty distribution."""
    return normalize_distribution(request.hypothesis_priors)


def _source_quality(source: Any, domain: str = "general") -> float:
    """Combine source reliability, relevance, evidence quality, and domain alignment."""
    domain_alignment = 1.0 if domain.lower() == "general" or source.domain.lower() in {"general", domain.lower()} else 0.65
    raw_quality = (
        0.28 * source.reliability + 0.22 * source.relevance
        + 0.18 * source.specificity + 0.22 * source.evidence_quality
        - 0.10 * source.noise
    )
    return clamp(raw_quality * (0.75 + 0.25 * domain_alignment))


def _expected_information_gain(source: Any, prior: Mapping[str, float]) -> float:
    """Compute expected entropy reduction from supplied outcome likelihoods."""
    if not prior or not source.outcome_likelihoods:
        return 0.0
    initial = entropy(prior)
    expected_posterior_entropy = 0.0
    outcome_probabilities: dict[str, float] = {}
    for outcome, likelihoods in source.outcome_likelihoods.items():
        probability = sum(
            prior.get(hypothesis, 0.0) * clamp(likelihoods.get(hypothesis, 0.0))
            for hypothesis in prior
        )
        outcome_probabilities[outcome] = probability
    total_outcome_probability = sum(outcome_probabilities.values())
    if total_outcome_probability <= 0.0:
        return 0.0
    for outcome, probability in outcome_probabilities.items():
        likelihoods = source.outcome_likelihoods[outcome]
        unnormalized = {
            hypothesis: prior_value * clamp(likelihoods.get(hypothesis, 0.0))
            for hypothesis, prior_value in prior.items()
        }
        posterior = normalize_distribution(unnormalized)
        expected_posterior_entropy += (probability / total_outcome_probability) * entropy(posterior)
    return max(0.0, initial - expected_posterior_entropy)


def step_01_define_target(request: FeedbackRequest, outputs: dict[str, Any], state: FeedbackState, config: FeedbackConfig) -> dict[str, Any]:
    """Build the weighted goal vector and measure whether the candidate meets its requirements."""
    total_weight = sum(item.weight for item in request.criteria)
    criteria = [
        {"criterion_id": item.criterion_id, "description": item.description,
         "weight": item.weight / total_weight if total_weight > 0 else 1.0 / max(1, len(request.criteria)),
         "achievement": item.achievement}
        for item in request.criteria
    ]
    completeness = weighted_completeness(request)
    return {"target": request.learning_target, "criteria": criteria,
            "target_vector": [item["weight"] for item in criteria],
            "completeness": completeness,
            "normalized_weights": True,
            "target_sufficiently_met": completeness >= config.goal_completeness_threshold}


def step_02_generate_interpretation(request: FeedbackRequest, outputs: dict[str, Any], state: FeedbackState, config: FeedbackConfig) -> dict[str, Any]:
    """Preserve the proposed response, reasoning chain, and competing interpretation priors."""
    prior = _prior(request)
    return {"candidate_response": request.candidate_response,
            "reasoning_nodes": [
                {"node_id": node.node_id, "content": node.content,
                 "consistency": node.consistency, "parent_ids": list(node.parent_ids)}
                for node in request.reasoning_chain
            ],
            "hypothesis_prior": prior, "hypothesis_entropy": entropy(prior),
            "knowledge_component_count": len(request.knowledge_components)}


def step_03_monitor_comprehension(request: FeedbackRequest, outputs: dict[str, Any], state: FeedbackState, config: FeedbackConfig) -> dict[str, Any]:
    """Estimate online uncertainty from low confidence, reasoning inconsistencies, and ambiguity."""
    inconsistency = 1.0 - min((node.consistency for node in request.reasoning_chain), default=1.0)
    uncertainty = clamp(
        config.uncertainty_confidence_weight * (1.0 - request.confidence)
        + config.uncertainty_inconsistency_weight * inconsistency
        + config.uncertainty_ambiguity_weight * request.ambiguity
    )
    check_required = uncertainty >= config.uncertainty_threshold or request.stakes >= 0.8
    return {"uncertainty": uncertainty, "inconsistency": inconsistency,
            "ambiguity": request.ambiguity, "check_required": check_required,
            "monitoring_signals": [
                label for label, score in (("low_confidence", 1.0-request.confidence),
                                           ("reasoning_conflict", inconsistency),
                                           ("ambiguous_input", request.ambiguity)) if score >= 0.35
            ]}


def step_04_estimate_confidence(request: FeedbackRequest, outputs: dict[str, Any], state: FeedbackState, config: FeedbackConfig) -> dict[str, Any]:
    """Keep subjective confidence distinct from correctness and quantify current hypothesis entropy."""
    prior = _prior(request)
    correctness = request.retrieval_correctness
    score = brier_score(request.confidence, correctness) if correctness is not None else None
    return {"confidence_estimate": request.confidence, "entropy": entropy(prior),
            "brier_score": score, "correctness_label_available": correctness is not None,
            "warning": "Confidence is an estimate, not evidence that the response is correct."}


def step_05_select_feedback_route(request: FeedbackRequest, outputs: dict[str, Any], state: FeedbackState, config: FeedbackConfig) -> dict[str, Any]:
    """Rank feedback sources by expected information gain, quality, and access cost."""
    prior = _prior(request)
    scored = []
    for source in request.feedback_sources:
        eig = _expected_information_gain(source, prior)
        quality = _source_quality(source, request.domain)
        domain_alignment = 1.0 if request.domain.lower() == "general" or source.domain.lower() in {"general", request.domain.lower()} else 0.65
        cost_penalty = config.source_cost_weight * source.access_cost
        score = eig + quality - cost_penalty
        scored.append({"source_id": source.source_id, "name": source.name,
                       "domain": source.domain, "domain_alignment": domain_alignment,
                       "expected_information_gain": eig, "quality": quality,
                       "cost_penalty": cost_penalty, "selection_score": score})
    scored.sort(key=lambda item: item["selection_score"], reverse=True)
    return {"source_scores": scored,
            "selected_source": scored[0]["source_id"] if scored else None,
            "source_selection_available": bool(scored)}


def step_06_obtain_feedback(request: FeedbackRequest, outputs: dict[str, Any], state: FeedbackState, config: FeedbackConfig) -> dict[str, Any]:
    """Collect caller-supplied evidence and quantify its relevance to the current target."""
    selected = outputs.get("step_05_select_feedback_route", {}).get("selected_source")
    items = []
    for evidence in request.feedback_evidence:
        items.append({"evidence_id": evidence.evidence_id, "source_id": evidence.source_id,
                      "statement": evidence.statement, "correction": evidence.correction,
                      "relevance": evidence.relevance, "actionability": evidence.actionability,
                      "evidence_quality": evidence.evidence_quality, "noise": evidence.noise,
                      "independence_key": evidence.independence_key,
                      "affected_components": list(evidence.affected_components),
                      "corrected_component_values": dict(evidence.corrected_component_values),
                      "selected_source": evidence.source_id == selected})
    mismatches = [
        1.0 - token_similarity(request.candidate_response, item["correction"])
        for item in items if item.get("correction")
    ]
    return {"feedback_items": items, "feedback_received": bool(items),
            "discrepancy": sum(mismatches) / len(mismatches) if mismatches else None,
            "discrepancy_method": "one_minus_jaccard_lexical_overlap_proxy" if mismatches else "not_available",
            "missing_feedback_fields": [field for field, present in (
                ("correction", any(item["correction"] for item in items)),
                ("source", bool(items)), ("evidence", any(item["evidence_quality"] > 0 for item in items))) if not present]}


def step_07_verify_feedback(request: FeedbackRequest, outputs: dict[str, Any], state: FeedbackState, config: FeedbackConfig) -> dict[str, Any]:
    """Assess evidence quality, collapse dependent sources, and update hypothesis probabilities conservatively."""
    sources = {source.source_id: source for source in request.feedback_sources}
    evidence_by_group: dict[str, list[Any]] = defaultdict(list)
    item_scores = []
    for evidence in request.feedback_evidence:
        source = sources[evidence.source_id]
        score = clamp(
            0.22 * _source_quality(source) + 0.24 * evidence.relevance
            + 0.20 * evidence.actionability + 0.26 * evidence.evidence_quality
            - 0.08 * evidence.noise
        )
        group = evidence.independence_key or f"source:{evidence.source_id}"
        evidence_by_group[group].append((evidence, score))
        item_scores.append({"evidence_id": evidence.evidence_id, "quality": score,
                            "accepted": score >= config.feedback_quality_threshold})
    accepted_groups = []
    for group, members in evidence_by_group.items():
        best = max(members, key=lambda pair: pair[1])
        if best[1] >= config.feedback_quality_threshold:
            accepted_groups.append({"group": group, "evidence": best[0], "quality": best[1]})
    posterior = _prior(request)
    prior = dict(posterior)
    for group in accepted_groups:
        likelihoods = group["evidence"].hypothesis_likelihoods
        if likelihoods:
            posterior = normalize_distribution({
                hypothesis: posterior_value * max(1e-9, likelihoods.get(hypothesis, 1.0)) ** group["quality"]
                for hypothesis, posterior_value in posterior.items()
            })
    trusted = [group["evidence"] for group in accepted_groups]
    feedback_quality = sum(group["quality"] for group in accepted_groups) / len(accepted_groups) if accepted_groups else 0.0
    return {"item_scores": item_scores, "accepted_evidence_count": len(trusted),
            "dependent_evidence_groups_collapsed": sum(max(0, len(members)-1) for members in evidence_by_group.values()),
            "feedback_quality": feedback_quality, "posterior": posterior,
            "prior": prior, "trusted_feedback": [
                {"evidence_id": item.evidence_id, "source_id": item.source_id,
                 "statement": item.statement, "correction": item.correction,
                 "affected_components": list(item.affected_components),
                 "corrected_component_values": dict(item.corrected_component_values)}
                for item in trusted],
            "feedback_trustworthy": feedback_quality >= config.feedback_quality_threshold,
            "source_independence_respected": True}


def step_08_classify_error(request: FeedbackRequest, outputs: dict[str, Any], state: FeedbackState, config: FeedbackConfig) -> dict[str, Any]:
    """Infer likely error categories and identify the earliest low-consistency reasoning node."""
    hints = {category: 0.0 for category in ERROR_CATEGORIES}
    for category, value in request.error_hints.items():
        canonical = category.strip().lower().replace("-", "_").replace(" ", "_")
        if canonical in hints:
            hints[canonical] = clamp(value)
    if request.ambiguity > 0.5:
        hints["communication_ambiguity"] = max(hints["communication_ambiguity"], request.ambiguity)
    if request.performance.working_memory_load > 0.7:
        hints["working_memory_overload"] = max(hints["working_memory_overload"], request.performance.working_memory_load)
    if request.performance.distraction > 0.7 or request.performance.time_pressure > 0.7:
        hints["attention_control"] = max(hints["attention_control"], max(request.performance.distraction, request.performance.time_pressure))
    if not any(hints.values()):
        uncertainty = outputs.get("step_03_monitor_comprehension", {}).get("uncertainty", 0.0)
        hints["knowledge_gap"] = clamp(uncertainty * 0.75)
        hints["inference_error"] = clamp(uncertainty * 0.25)
    posterior = normalize_distribution(hints)
    first_bad = next((node for node in request.reasoning_chain if node.consistency < 0.7), None)
    return {"error_probabilities": posterior,
            "most_likely_error": max(posterior, key=posterior.get) if posterior else "undetermined",
            "first_divergence": ({"node_id": first_bad.node_id, "content": first_bad.content,
                                  "consistency": first_bad.consistency} if first_bad else None),
            "reasoning_graph_node_count": len(request.reasoning_chain)}


def step_09_separate_performance(request: FeedbackRequest, outputs: dict[str, Any], state: FeedbackState, config: FeedbackConfig) -> dict[str, Any]:
    """Separate knowledge strength from temporary cognitive load and avoid inventing correctness labels."""
    numeric = [clamp(float(value)) for value in request.knowledge_components.values()
               if isinstance(value, (int, float)) and math.isfinite(float(value))]
    knowledge_strength = sum(numeric) / len(numeric) if numeric else weighted_completeness(request)
    performance = request.performance
    load = clamp(0.30 * performance.fatigue + 0.25 * performance.distraction
                 + 0.30 * performance.working_memory_load + 0.15 * performance.time_pressure)
    difficulty = clamp(0.20 + 0.30 * (1.0 - weighted_completeness(request)))
    predicted_correct = sigmoid(3.0 * (knowledge_strength - difficulty) - 2.0 * load)
    if request.retrieval_correctness is not None:
        failure_type = "knowledge_or_model_error" if request.retrieval_correctness < 0.5 else "no_observed_failure"
    elif load >= 0.55 and knowledge_strength >= 0.55:
        failure_type = "performance_limited_possible"
    else:
        failure_type = "undetermined_without_independent_retrieval_label"
    return {"knowledge_strength": knowledge_strength, "estimated_task_difficulty": difficulty,
            "performance_load": load, "predicted_correct_probability": predicted_correct,
            "observed_correctness": request.retrieval_correctness,
            "failure_type": failure_type,
            "label_was_invented": False}


def step_10_apply_targeted_correction(request: FeedbackRequest, outputs: dict[str, Any], state: FeedbackState, config: FeedbackConfig) -> dict[str, Any]:
    """Update only affected knowledge components, blending numeric values and preserving unrelated entries."""
    current = dict(request.knowledge_components)
    current.update(state.knowledge_components)
    trusted = outputs.get("step_07_verify_feedback", {}).get("trusted_feedback", [])
    # Do not apply a caller-provided replacement unless at least one evidence item
    # passed the source/evidence gate in Step 07. Untrusted corrections remain visible
    # in the diagnostic output but do not overwrite the knowledge representation.
    allowed = set(request.affected_components) if trusted else set()
    proposed: dict[str, Any] = dict(request.corrected_components) if trusted else {}
    for evidence in trusted:
        proposed.update(evidence.get("corrected_component_values", {}))
        allowed.update(evidence.get("affected_components", []))
    if not allowed and trusted:
        allowed.update(proposed)
    updates: dict[str, dict[str, Any]] = {}
    for key in sorted(allowed):
        if key not in proposed:
            continue
        previous = current.get(key)
        target = proposed[key]
        if isinstance(previous, (int, float)) and isinstance(target, (int, float)):
            revised = float(previous) + config.correction_learning_rate * (float(target) - float(previous))
        else:
            revised = target
        current[key] = revised
        updates[key] = {"previous": previous, "target": target, "updated": revised,
                        "learning_rate": config.correction_learning_rate,
                        "changed": previous != revised}
    state.knowledge_components = current
    return {"component_updates": updates, "revised_knowledge": current,
            "preserved_component_names": sorted(set(current) - set(updates)),
            "correction_applied": bool(updates),
            "correction_provisional": not bool(trusted),
            "correction_source_ids": sorted({item["source_id"] for item in trusted})}


def step_11_integrate_knowledge(request: FeedbackRequest, outputs: dict[str, Any], state: FeedbackState, config: FeedbackConfig) -> dict[str, Any]:
    """Build an evidence-linked concept graph with explicit support and boundary relations."""
    correction = outputs.get("step_10_apply_targeted_correction", {})
    nodes = {request.learning_target: {"kind": "target", "confidence": request.confidence}}
    edges = []
    for component, details in correction.get("component_updates", {}).items():
        nodes[component] = {"kind": "corrected_component", "value": details.get("updated")}
        edges.append({"source": component, "relation": "addresses", "target": request.learning_target})
    for concept in request.supporting_concepts:
        nodes[concept] = {"kind": "supporting_concept"}
        edges.append({"source": concept, "relation": "supports", "target": request.learning_target})
    for condition in request.conditions:
        nodes[condition] = {"kind": "validity_condition"}
        edges.append({"source": request.learning_target, "relation": "valid_when", "target": condition})
    sources = outputs.get("step_07_verify_feedback", {}).get("trusted_feedback", [])
    provenance = [{"source_id": item["source_id"], "evidence_id": item["evidence_id"]} for item in sources]
    cues = list(dict.fromkeys([request.learning_target, *request.supporting_concepts, *request.conditions]))
    return {"nodes": nodes, "edges": edges, "provenance": provenance,
            "retrieval_cues": cues, "graph_is_provenance_linked": bool(provenance)}


def step_12_explain_correction(request: FeedbackRequest, outputs: dict[str, Any], state: FeedbackState, config: FeedbackConfig) -> dict[str, Any]:
    """Score explanatory accuracy, relational coherence, evidence, and boundary awareness."""
    correction = outputs.get("step_10_apply_targeted_correction", {})
    trusted = outputs.get("step_07_verify_feedback", {}).get("trusted_feedback", [])
    has_corrected_text = bool(request.corrected_response or any(item.get("correction") for item in trusted))
    accuracy = 1.0 if has_corrected_text and trusted else 0.25 if has_corrected_text else 0.0
    relational = clamp(len(request.supporting_concepts) / 2.0) if request.supporting_concepts else 0.0
    evidence = 1.0 if trusted else 0.0
    boundary = 1.0 if request.conditions and (request.boundary_examples or request.counterexample) else 0.5 if request.conditions or request.boundary_examples else 0.0
    quality = clamp(0.35 * accuracy + 0.25 * relational + 0.25 * evidence + 0.15 * boundary)
    return {"corrected_explanation": request.corrected_response or next((item.get("correction") for item in trusted if item.get("correction")), None),
            "counterexample": request.counterexample, "positive_example": request.positive_example,
            "conditions": list(request.conditions), "boundary_examples": list(request.boundary_examples),
            "quality_components": {"accuracy": accuracy, "relational_coherence": relational,
                                   "evidence_support": evidence, "boundary_awareness": boundary},
            "explanation_quality": quality,
            "correction_is_provisional": not bool(trusted),
            "targeted_components_count": len(correction.get("component_updates", {}))}


def step_13_test_retrieval(request: FeedbackRequest, outputs: dict[str, Any], state: FeedbackState, config: FeedbackConfig) -> dict[str, Any]:
    """Evaluate a fresh retrieval response without using the original feedback as its answer."""
    corrected = request.corrected_response or outputs.get("step_12_explain_correction", {}).get("corrected_explanation")
    if request.retrieval_correctness is not None:
        success = request.retrieval_correctness >= 0.5
        post_error = 1.0 - request.retrieval_correctness
        score_source = "caller_verified_label"
    elif request.retrieval_response and corrected:
        similarity = token_similarity(request.retrieval_response, corrected)
        success = similarity >= 0.55
        post_error = 1.0 - similarity
        score_source = "lexical_overlap_proxy"
    else:
        similarity, success, post_error, score_source = None, None, None, "not_available"
    return {"retrieval_response": request.retrieval_response, "retrieval_success": success,
            "retrieval_similarity_proxy": similarity if request.retrieval_response and corrected and request.retrieval_correctness is None else None,
            "post_error": post_error, "score_source": score_source,
            "independent_retrieval_attempt": request.retrieval_response is not None or request.retrieval_correctness is not None}


def step_14_calibrate_confidence(request: FeedbackRequest, outputs: dict[str, Any], state: FeedbackState, config: FeedbackConfig) -> dict[str, Any]:
    """Update confidence calibration history only when an independent correctness label is supplied."""
    correctness = request.retrieval_correctness
    record = {"confidence": request.confidence, "correctness": correctness}
    if correctness is not None:
        state.confidence_history.append(record)
        brier = brier_score(request.confidence, correctness)
        bins: dict[int, list[dict[str, float]]] = defaultdict(list)
        for item in state.confidence_history:
            bins[min(9, int(clamp(item["confidence"]) * 10))].append(item)
        total = len(state.confidence_history)
        ece = 0.0
        for items in bins.values():
            mean_conf = sum(item["confidence"] for item in items) / len(items)
            mean_acc = sum(item["correctness"] for item in items) / len(items)
            ece += len(items) / total * abs(mean_acc - mean_conf)
    else:
        brier, ece = None, None
    return {"current_brier_score": brier, "expected_calibration_error": ece,
            "calibration_sample_count": len(state.confidence_history),
            "calibration_updated": correctness is not None,
            "confidence_not_used_as_evidence": True}


def step_15_control_feedback_load(request: FeedbackRequest, outputs: dict[str, Any], state: FeedbackState, config: FeedbackConfig) -> dict[str, Any]:
    """Estimate feedback complexity and divide high-load feedback into manageable chunks."""
    item_count = len(request.feedback_evidence)
    interaction_density = clamp(sum(len(item.affected_components) for item in request.feedback_evidence) / max(1, item_count * 4))
    terminology_uncertainty = request.ambiguity
    load = 0.25 * clamp(item_count / 8.0) + 0.45 * interaction_density + 0.30 * terminology_uncertainty
    chunk_count = max(1, int(math.ceil(load / max(config.feedback_load_threshold, 1e-6))))
    labels = [item.statement for item in request.feedback_evidence]
    chunk_size = max(1, int(math.ceil(len(labels) / chunk_count)))
    chunks = [labels[index:index + chunk_size] for index in range(0, len(labels), chunk_size)]
    if not chunks:
        chunks = [[]]
    return {"feedback_load": clamp(load), "item_count": item_count,
            "interaction_density": interaction_density,
            "terminology_uncertainty": terminology_uncertainty,
            "chunk_count": chunk_count, "chunks": chunks,
            "chunking_required": load > config.feedback_load_threshold}


def step_16_test_transfer(request: FeedbackRequest, outputs: dict[str, Any], state: FeedbackState, config: FeedbackConfig) -> dict[str, Any]:
    """Measure transfer using observed probe labels and report structural versus surface similarity."""
    observed = [probe for probe in request.transfer_probes if probe.correctness is not None]
    score = sum(probe.correctness for probe in observed) / len(observed) if observed else None
    counterexamples = [probe for probe in observed if probe.is_counterexample]
    return {"transfer_score": score, "observed_probe_count": len(observed),
            "total_probe_count": len(request.transfer_probes),
            "counterexamples_tested": len(counterexamples),
            "mean_structural_similarity": (sum(probe.structural_similarity for probe in request.transfer_probes) / len(request.transfer_probes)) if request.transfer_probes else None,
            "mean_surface_similarity": (sum(probe.surface_similarity for probe in request.transfer_probes) / len(request.transfer_probes)) if request.transfer_probes else None,
            "transfer_status": "passed" if score is not None and score >= 0.8 else "failed" if score is not None else "not_tested",
            "unlabelled_probes_not_scored": len(request.transfer_probes) - len(observed)}


def step_17_schedule_delayed_check(request: FeedbackRequest, outputs: dict[str, Any], state: FeedbackState, config: FeedbackConfig) -> dict[str, Any]:
    """Estimate memory accessibility decay and add a provenance-aware delayed retrieval event."""
    retrieval = outputs.get("step_13_test_retrieval", {})
    initial_accessibility = 1.0 if retrieval.get("retrieval_success") is True else 0.75 if retrieval.get("retrieval_success") is None else 0.45
    accessibility = initial_accessibility * math.exp(-config.retention_decay_rate * request.delayed_check_days)
    target = config.retention_target
    recommended_days = max(0.25, math.log(max(initial_accessibility, 1e-9) / target) / config.retention_decay_rate)
    item = {"request_id": request.request_id, "learning_target": request.learning_target,
            "due_in_days": min(request.delayed_check_days, recommended_days),
            "recommended_interval_days": recommended_days,
            "estimated_accessibility": accessibility,
            "requires_retrieval_without_feedback": True,
            "original_candidate": request.candidate_response,
            "corrected_response": request.corrected_response}
    state.review_queue.append(item)
    return {"accessibility": accessibility, "recommended_interval_days": recommended_days,
            "review_item": item, "review_queue_size": len(state.review_queue),
            "recurrence_count": sum(1 for entry in state.correction_history if entry.get("request_id") == request.request_id and entry.get("regression"))}


def step_18_update_monitoring_strategy(request: FeedbackRequest, outputs: dict[str, Any], state: FeedbackState, config: FeedbackConfig) -> dict[str, Any]:
    """Update rolling error frequencies and retain strategy outcomes for future targeted monitoring."""
    category = outputs.get("step_08_classify_error", {}).get("most_likely_error", "undetermined")
    for known in set(state.error_frequencies) | {category}:
        previous = state.error_frequencies.get(known, 0.0)
        indicator = 1.0 if known == category else 0.0
        state.error_frequencies[known] = config.error_frequency_decay * previous + (1.0 - config.error_frequency_decay) * indicator
    retrieval = outputs.get("step_13_test_retrieval", {})
    outcome = retrieval.get("retrieval_success")
    strategy_key = category
    metrics = state.strategy_outcomes.setdefault(strategy_key, {"attempts": 0.0, "successes": 0.0})
    metrics["attempts"] += 1.0
    if outcome is True:
        metrics["successes"] += 1.0
    state.episode_count += 1
    state.correction_history.append({"request_id": request.request_id, "error_category": category,
                                     "retrieval_success": outcome,
                                     "regression": outcome is False})
    policy = max(state.error_frequencies, key=state.error_frequencies.get) if state.error_frequencies else category
    return {"rolling_error_frequencies": dict(state.error_frequencies),
            "strategy_outcomes": state.strategy_outcomes,
            "monitoring_policy": f"monitor_{policy}_earlier",
            "episode_count": state.episode_count,
            "future_strategy_success_rate": (metrics["successes"] / metrics["attempts"]) if metrics["attempts"] else None}


FUNCTIONAL_STEPS = (
    step_01_define_target, step_02_generate_interpretation,
    step_03_monitor_comprehension, step_04_estimate_confidence,
    step_05_select_feedback_route, step_06_obtain_feedback,
    step_07_verify_feedback, step_08_classify_error,
    step_09_separate_performance, step_10_apply_targeted_correction,
    step_11_integrate_knowledge, step_12_explain_correction,
    step_13_test_retrieval, step_14_calibrate_confidence,
    step_15_control_feedback_load, step_16_test_transfer,
    step_17_schedule_delayed_check, step_18_update_monitoring_strategy,
)
