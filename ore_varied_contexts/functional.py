"""Functional models for contextual variation, interleaving, discrimination, and transfer."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import (
    ConceptDefinition, ConceptMemoryState, LearningExample, TrialObservation,
    VariedContextsConfig, VariedContextsState,
)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Bound a finite model signal to an inclusive numeric range."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("value must be finite")
    return min(high, max(low, number))


def stable_softmax(scores: Mapping[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Convert finite competing concept scores into a stable probability distribution."""
    if not scores:
        return {}
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    keys = list(scores)
    values = np.asarray([float(scores[key]) for key in keys], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    exponents = np.exp(np.clip((values - float(np.max(values))) / temperature, -60.0, 0.0))
    probabilities = exponents / max(float(exponents.sum()), 1e-12)
    return {key: float(value) for key, value in zip(keys, probabilities)}


def entropy(probabilities: Mapping[str, float]) -> float:
    """Calculate Shannon entropy for a normalized non-negative categorical distribution."""
    values = [max(0.0, float(value)) for value in probabilities.values()]
    total = sum(values)
    if total <= 1e-12:
        return 0.0
    normalized = [value / total for value in values]
    return float(-sum(value * math.log(max(value, 1e-12)) for value in normalized if value > 0))


def context_id(context: Mapping[str, str]) -> str:
    """Create a deterministic identifier for a context-feature combination."""
    return json.dumps(dict(sorted(context.items())), ensure_ascii=False, separators=(",", ":"))


def feature_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Calculate Euclidean distance over the union of two named feature dictionaries."""
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    a = np.asarray([float(left.get(key, 0.0)) for key in keys], dtype=float)
    b = np.asarray([float(right.get(key, 0.0)) for key in keys], dtype=float)
    return float(np.linalg.norm(a - b) / math.sqrt(len(keys)))


def cosine_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Calculate cosine similarity between sparse named feature vectors."""
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    a = np.asarray([float(left.get(key, 0.0)) for key in keys], dtype=float)
    b = np.asarray([float(right.get(key, 0.0)) for key in keys], dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return clamp(float(np.dot(a, b) / denominator), -1.0, 1.0)


def weighted_feature_score(
    observed: Mapping[str, float], expected: Mapping[str, float],
) -> float:
    """Compute normalized feature alignment, including explicit penalties for missing features."""
    if not expected:
        return 0.0
    denominator = sum(abs(float(value)) for value in expected.values())
    if denominator <= 1e-12:
        return 0.0
    numerator = sum(float(weight) * float(observed.get(key, 0.0))
                    for key, weight in expected.items())
    return clamp(numerator / denominator, -1.0, 1.0)


def _mean_feature_map(examples: Sequence[LearningExample]) -> dict[str, float]:
    """Calculate a feature-wise sample mean while treating absent features as zero."""
    if not examples:
        return {}
    keys = sorted({key for item in examples for key in item.concept_features})
    return {
        key: float(np.mean([item.concept_features.get(key, 0.0) for item in examples]))
        for key in keys
    }


def _variance_feature_map(
    examples: Sequence[LearningExample], prototype: Mapping[str, float],
) -> dict[str, float]:
    """Calculate feature-wise variance around a concept prototype."""
    keys = sorted({key for item in examples for key in item.concept_features}
                  | set(prototype))
    if not examples:
        return {key: 0.0 for key in keys}
    return {
        key: float(np.mean([
            (item.concept_features.get(key, 0.0) - prototype.get(key, 0.0)) ** 2
            for item in examples
        ]))
        for key in keys
    }


def readiness_report(
    concepts: Sequence[ConceptDefinition], readiness: Mapping[str, float],
) -> dict[str, Any]:
    """Estimate prerequisite readiness and identify concepts needing an anchor or initial block."""
    reports: dict[str, Any] = {}
    for concept in concepts:
        prerequisite_scores = [float(readiness.get(item, 0.0))
                                for item in concept.prerequisite_concepts]
        prerequisite_readiness = (float(np.mean(prerequisite_scores))
                                  if prerequisite_scores else concept.initial_familiarity)
        reports[concept.concept_id] = {
            "prerequisites": list(concept.prerequisite_concepts),
            "prerequisite_readiness": prerequisite_readiness,
            "prerequisite_gaps": [
                item for item in concept.prerequisite_concepts
                if float(readiness.get(item, 0.0)) < 0.60
            ],
            "basic_anchor_ready": prerequisite_readiness >= 0.50,
        }
    return {"concept_readiness": reports,
            "overall_readiness": float(np.mean([
                item["prerequisite_readiness"] for item in reports.values()
            ])) if reports else 0.0,
            "blocked_interleaving_concepts": [
                key for key, value in reports.items() if not value["basic_anchor_ready"]
            ]}


def build_concept_profiles(
    concepts: Sequence[ConceptDefinition],
    examples: Sequence[LearningExample],
) -> dict[str, dict[str, Any]]:
    """Estimate prototypes, within-concept variability, invariants, and feature distributions."""
    profiles: dict[str, dict[str, Any]] = {}
    for concept in concepts:
        selected = [item for item in examples if item.concept_id == concept.concept_id]
        prototype = _mean_feature_map(selected)
        variance = _variance_feature_map(selected, prototype)
        invariant = {
            key: value for key, value in prototype.items()
            if sum(key in item.concept_features for item in selected) / max(1, len(selected)) >= 0.70
            and variance.get(key, 0.0) <= 0.25
        }
        modalities = sorted({item.modality for item in selected})
        contexts = sorted({context_id(item.context_features) for item in selected})
        profiles[concept.concept_id] = {
            "concept_id": concept.concept_id,
            "domain": concept.domain,
            "example_count": len(selected),
            "prototype": prototype,
            "within_category_variance": variance,
            "candidate_invariants": invariant,
            "modalities": modalities,
            "context_count": len(contexts),
            "context_ids": contexts,
            "relations": list(concept.relations),
            "boundary_conditions": dict(concept.boundary_conditions),
            "episodic_examples_preserved": [item.example_id for item in selected],
        }
    return profiles


def context_coverage(examples: Sequence[LearningExample]) -> dict[str, Any]:
    """Measure which contextual dimensions and values are represented in the example set."""
    dimension_values: dict[str, set[str]] = defaultdict(set)
    context_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    for item in examples:
        for key, value in item.context_features.items():
            dimension_values[key].add(value)
        context_counts[context_id(item.context_features)] += 1
        modality_counts[item.modality] += 1
    elapsed = [
        float(item.elapsed_since_previous_hours)
        for item in examples if item.elapsed_since_previous_hours is not None
    ]
    temporal_separation = {
        "observed_interval_count": len(elapsed),
        "mean_hours": float(np.mean(elapsed)) if elapsed else None,
        "minimum_hours": min(elapsed) if elapsed else None,
        "maximum_hours": max(elapsed) if elapsed else None,
        "intervals_supplied_by_caller": bool(elapsed),
        "calendar_scheduler_owned_by_point03": True,
    }
    return {
        "dimension_values": {key: sorted(values) for key, values in dimension_values.items()},
        "dimension_count": len(dimension_values),
        "context_count": len(context_counts),
        "context_episode_counts": dict(context_counts),
        "modality_counts": dict(modality_counts),
        "temporal_separation": temporal_separation,
        "context_variation_is_separate_from_topic_interleaving": True,
        "spacing_is_not_inferred_from_interleaving_order": True,
    }


def infer_context(
    cues: Mapping[str, str],
    examples: Sequence[LearningExample],
    state: VariedContextsState | None = None,
    smoothing: float = 0.5,
) -> dict[str, Any]:
    """Infer a context from categorical cues using a smoothed naive-Bayes/log-transition approximation."""
    if not examples:
        return {"context_probabilities": {}, "best_context_id": None,
                "uncertainty": 0.0, "needs_more_cues": True}
    context_examples: dict[str, list[LearningExample]] = defaultdict(list)
    for item in examples:
        context_examples[context_id(item.context_features)].append(item)
    candidate_ids = sorted(context_examples)
    total = sum(len(values) for values in context_examples.values())
    log_scores: dict[str, float] = {}
    for candidate in candidate_ids:
        records = context_examples[candidate]
        prior = (len(records) + smoothing) / (total + smoothing * len(candidate_ids))
        log_score = math.log(max(prior, 1e-12))
        for key, value in cues.items():
            values = [record.context_features.get(key) for record in records]
            observed_count = sum(item == value for item in values)
            distinct = max(1, len({record.context_features.get(key)
                                   for group in context_examples.values()
                                   for record in group}))
            likelihood = (observed_count + smoothing) / (len(records) + smoothing * distinct)
            log_score += math.log(max(likelihood, 1e-12))
        if state:
            previous_contexts = state.context_transition_counts
            candidate_text = candidate
            incoming = sum(previous_contexts.get(source, {}).get(candidate_text, 0)
                           for source in previous_contexts)
            outgoing_total = sum(sum(row.values()) for row in previous_contexts.values())
            if outgoing_total:
                log_score += 0.15 * math.log((incoming + smoothing) /
                                             (outgoing_total + smoothing * len(candidate_ids)))
        log_scores[candidate] = log_score
    probabilities = stable_softmax(log_scores)
    uncertainty = entropy(probabilities) / max(math.log(max(2, len(probabilities))), 1e-12)
    best = max(probabilities, key=probabilities.get)
    return {"context_probabilities": probabilities, "best_context_id": best,
            "uncertainty": clamp(uncertainty),
            "needs_more_cues": uncertainty >= 0.55,
            "cue_match": cues,
            "model": "smoothed_context_inference_approximation"}


def score_example_against_concept(
    example: LearningExample,
    concept: ConceptDefinition,
    prototype: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Combine defining-feature support, prototype similarity, and conditional-context fit."""
    feature_score = weighted_feature_score(
        example.concept_features, concept.defining_features
    )
    prototype_score = (cosine_similarity(example.concept_features, prototype)
                       if prototype else feature_score)
    condition_matches = 0
    condition_count = len(concept.boundary_conditions)
    violated: list[str] = []
    for key, expected in concept.boundary_conditions.items():
        if example.context_features.get(key) == expected:
            condition_matches += 1
        elif key in example.context_features:
            violated.append(key)
    context_fit = (condition_matches / condition_count if condition_count else 1.0)
    # Conditions are informative constraints, not a universal penalty for missing context metadata.
    context_score = 1.0 if not condition_count else (0.5 + 0.5 * context_fit)
    score = 0.65 * feature_score + 0.25 * prototype_score + 0.10 * context_score
    return {"feature_score": feature_score, "prototype_similarity": prototype_score,
            "context_condition_fit": context_fit, "context_score": context_score,
            "combined_score": score, "violated_conditions": violated}


def classify_example(
    example: LearningExample,
    concepts: Sequence[ConceptDefinition],
    profiles: Mapping[str, Mapping[str, Any]],
    config: VariedContextsConfig,
) -> dict[str, Any]:
    """Generate competing concept scores and normalized probabilities for one example."""
    scores = {
        concept.concept_id: score_example_against_concept(
            example, concept, profiles.get(concept.concept_id, {}).get("prototype")
        )["combined_score"]
        for concept in concepts
    }
    probabilities = stable_softmax(scores, config.probability_temperature)
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    selected = ranked[0][0]
    alternative = ranked[1][0] if len(ranked) > 1 else None
    margin = probabilities[selected] - (probabilities[alternative] if alternative else 0.0)
    return {"example_id": example.example_id, "predicted_concept_id": selected,
            "candidate_scores": scores, "concept_probabilities": probabilities,
            "alternative_concept_id": alternative, "contrastive_margin": margin,
            "prediction_entropy": entropy(probabilities)}


def compare_concepts(
    concepts: Sequence[ConceptDefinition],
    profiles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Estimate between-category separation and feature-level discriminability."""
    pair_reports: dict[str, Any] = {}
    concept_ids = [item.concept_id for item in concepts]
    for index, left in enumerate(concepts):
        for right in concepts[index + 1:]:
            left_prototype = profiles[left.concept_id]["prototype"]
            right_prototype = profiles[right.concept_id]["prototype"]
            distance = feature_distance(left_prototype, right_prototype)
            similarity = cosine_similarity(left_prototype, right_prototype)
            keys = sorted(set(left_prototype) | set(right_prototype))
            differences = {
                key: abs(left_prototype.get(key, 0.0) - right_prototype.get(key, 0.0))
                for key in keys
            }
            diagnostic = sorted(differences, key=differences.get, reverse=True)
            key = f"{left.concept_id}::{right.concept_id}"
            pair_reports[key] = {
                "left": left.concept_id, "right": right.concept_id,
                "prototype_distance": distance, "prototype_similarity": similarity,
                "feature_differences": differences,
                "candidate_discriminative_features": diagnostic[:8],
                "shared_domain": left.domain == right.domain,
                "shared_prerequisites": sorted(set(left.prerequisite_concepts)
                                               & set(right.prerequisite_concepts)),
            }
    return {"concept_ids": concept_ids, "pairs": pair_reports}


def cross_modality_consistency(
    examples: Sequence[LearningExample],
    profiles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare concept-feature representations across modalities while preserving relation-level evidence."""
    result: dict[str, Any] = {}
    for concept_id in profiles:
        selected = [item for item in examples if item.concept_id == concept_id]
        modality_groups: dict[str, list[LearningExample]] = defaultdict(list)
        for item in selected:
            modality_groups[item.modality].append(item)
        means = {modality: _mean_feature_map(group)
                 for modality, group in modality_groups.items()}
        pairs: dict[str, float] = {}
        modalities = sorted(means)
        for index, left in enumerate(modalities):
            for right in modalities[index + 1:]:
                pairs[f"{left}::{right}"] = cosine_similarity(means[left], means[right])
        all_relations = set().union(*(set(item.relations) for item in selected)) if selected else set()
        result[concept_id] = {
            "modalities": modalities,
            "modality_pair_similarity": pairs,
            "shared_relation_count": len(all_relations),
            "cross_modality_consistency": float(np.mean(list(pairs.values())))
            if pairs else None,
            "representation_routes": len(modalities),
        }
    return result


def _confusion_matrix(
    concepts: Sequence[ConceptDefinition],
    examples: Sequence[LearningExample],
    observations: Sequence[TrialObservation],
) -> dict[str, dict[str, int]]:
    """Build a truth-by-selected-concept confusion matrix from observed learner decisions."""
    ids = [item.concept_id for item in concepts]
    matrix = {truth: {predicted: 0 for predicted in ids} for truth in ids}
    example_map = {item.example_id: item for item in examples}
    for observation in observations:
        example = example_map[observation.example_id]
        matrix[example.concept_id][observation.selected_concept_id] += 1
    return matrix


def _precision_recall(matrix: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    """Calculate per-class precision, recall, F1, and balanced accuracy."""
    labels = list(matrix)
    per_class: dict[str, Any] = {}
    recalls: list[float] = []
    for label in labels:
        tp = int(matrix[label].get(label, 0))
        fn = sum(int(value) for value in matrix[label].values()) - tp
        fp = sum(int(row.get(label, 0)) for truth, row in matrix.items() if truth != label)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": tp + fn}
        if tp + fn:
            recalls.append(recall)
    return {"per_class": per_class,
            "balanced_accuracy": float(np.mean(recalls)) if recalls else None}


def _observations_by_example(
    observations: Sequence[TrialObservation],
) -> dict[str, TrialObservation]:
    """Map the latest supplied observation for each example to its typed record."""
    return {item.example_id: item for item in observations}


def determine_interleaving(
    concepts: Sequence[ConceptDefinition],
    examples: Sequence[LearningExample],
    comparisons: Mapping[str, Any],
    observations: Sequence[TrialObservation],
    readiness: Mapping[str, Any],
    config: VariedContextsConfig,
) -> dict[str, Any]:
    """Select related concept pairs for interleaving using domain, prerequisites, and observed confusion."""
    confusion = _confusion_matrix(concepts, examples, observations)
    pair_decisions: dict[str, Any] = {}
    approved: list[tuple[str, str]] = []
    for pair_key, report in comparisons["pairs"].items():
        left, right = report["left"], report["right"]
        left_total = max(1, sum(confusion[left].values()))
        right_total = max(1, sum(confusion[right].values()))
        confusion_rate = (confusion[left].get(right, 0) / left_total
                          + confusion[right].get(left, 0) / right_total) / 2.0
        shared_structure = bool(report["shared_domain"] or report["shared_prerequisites"])
        relatedness = clamp(0.55 * max(0.0, report["prototype_similarity"])
                            + 0.25 * float(report["shared_domain"])
                            + 0.20 * bool(report["shared_prerequisites"]))
        learnable_contrast = clamp(report["prototype_distance"])
        blocked = (left in readiness["blocked_interleaving_concepts"]
                   or right in readiness["blocked_interleaving_concepts"])
        eligible = (
            not blocked and shared_structure
            and relatedness >= config.interleaving_min_relatedness
            and (confusion_rate >= config.interleaving_min_confusion
                 or learnable_contrast >= 0.15)
        )
        pair_decisions[pair_key] = {
            "eligible_for_interleaving": eligible,
            "relatedness": relatedness,
            "observed_confusion_rate": confusion_rate,
            "learnable_contrast": learnable_contrast,
            "shared_domain_or_prerequisite": shared_structure,
            "blocked_by_readiness": blocked,
            "reason": ("meaningful contrast and prerequisite/domain relation"
                       if eligible else "insufficient relation, readiness, or useful contrast"),
        }
        if eligible:
            approved.append((left, right))
    return {"pair_decisions": pair_decisions,
            "approved_pairs": [list(pair) for pair in approved],
            "confusion_matrix": confusion,
            "metrics": _precision_recall(confusion),
            "random_mixing_recommended": False}


def build_interleaving_schedule(
    examples: Sequence[LearningExample],
    decision: Mapping[str, Any],
    available_time_minutes: float,
    config: VariedContextsConfig,
) -> dict[str, Any]:
    """Construct a balanced deterministic schedule that alternates only approved related categories."""
    pool = [item for item in examples if item.transfer_level == "training"]
    if not pool:
        pool = list(examples)
    pairs = {frozenset(pair) for pair in decision["approved_pairs"]}
    counts = Counter(item.concept_id for item in pool)
    selected: list[LearningExample] = []
    remaining = list(pool)
    schedule_trace: list[dict[str, Any]] = []
    served = Counter()
    previous: str | None = None
    while remaining:
        scored: list[tuple[float, str, LearningExample]] = []
        for item in remaining:
            contrast = 0.0
            switch = 0.0
            if previous is not None and item.concept_id != previous:
                pair = frozenset((previous, item.concept_id))
                if pair in pairs:
                    contrast = 1.0
                    switch = 1.0
            elif previous is not None:
                switch = -0.35
            balance = 1.0 - served[item.concept_id] / max(1, counts[item.concept_id])
            anchor_bonus = 0.10 if item.is_anchor and served[item.concept_id] == 0 else 0.0
            score = 0.55 * contrast + 0.35 * balance + anchor_bonus + 0.10 * switch
            scored.append((score, item.example_id, item))
        scored.sort(key=lambda row: (-row[0], row[1]))
        score, _, chosen = scored[0]
        selected.append(chosen)
        served[chosen.concept_id] += 1
        schedule_trace.append({
            "position": len(selected), "example_id": chosen.example_id,
            "concept_id": chosen.concept_id, "schedule_score": score,
            "switch_from_previous": previous is not None and previous != chosen.concept_id,
            "approved_contrast": (previous is not None
                                  and frozenset((previous, chosen.concept_id)) in pairs),
        })
        previous = chosen.concept_id
        remaining.remove(chosen)
    estimated_minutes = len(selected) * 2.0
    overload = clamp(max(0.0, estimated_minutes - available_time_minutes)
                     / max(estimated_minutes, 1.0))
    return {
        "example_order": [item.example_id for item in selected],
        "concept_order": [item.concept_id for item in selected],
        "trace": schedule_trace,
        "category_counts": dict(served),
        "switch_count": sum(row["switch_from_previous"] for row in schedule_trace),
        "approved_contrast_switches": sum(row["approved_contrast"] for row in schedule_trace),
        "interleaving_used": sum(row["approved_contrast"] for row in schedule_trace) > 0,
        "estimated_time_minutes": estimated_minutes,
        "overload_estimate": overload,
        "spacing_not_derived_from_order": True,
    }


def _context_change_count(
    previous: Mapping[str, str], current: Mapping[str, str],
) -> int:
    """Count explicit context dimensions whose values differ between adjacent examples."""
    return sum(previous.get(key) != current.get(key)
               for key in set(previous) | set(current))


def score_practice_trial(
    example: LearningExample,
    concepts: Sequence[ConceptDefinition],
    profiles: Mapping[str, Mapping[str, Any]],
    observation: TrialObservation | None,
    config: VariedContextsConfig,
) -> dict[str, Any]:
    """Execute concept selection and score the observed choice, confidence, context, and effort."""
    classification = classify_example(example, concepts, profiles, config)
    observation = observation
    predicted = classification["predicted_concept_id"]
    selected = observation.selected_concept_id if observation else predicted
    correct = selected == example.concept_id
    confidence = observation.confidence if observation else classification["concept_probabilities"][predicted]
    effort = observation.effort if observation else 0.5
    context_match = None
    if observation:
        total = len(example.context_features)
        context_match = (sum(observation.context_judgment.get(key) == value
                             for key, value in example.context_features.items()) / total
                         if total else 1.0)
    return {
        **classification,
        "ground_truth_concept_id": example.concept_id,
        "selected_concept_id": selected,
        "observed_selection_correct": correct,
        "prediction_correct": predicted == example.concept_id,
        "confidence": confidence,
        "effort": effort,
        "context_match": context_match,
        "observation_available": observation is not None,
        "aid_level": observation.aid_level if observation else 0,
        "cue_type": observation.cue_type if observation else "model_generated",
        "transfer_level": example.transfer_level,
        "modality": example.modality,
    }


def contrastive_trial_analysis(
    trial_results: Sequence[Mapping[str, Any]],
    approved_pairs: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Summarize candidate margins and category confusions for each approved contrast pair."""
    matrix: dict[str, dict[str, Any]] = {}
    for pair in approved_pairs:
        left, right = pair
        relevant = [item for item in trial_results
                    if item["ground_truth_concept_id"] in {left, right}]
        matrix[f"{left}::{right}"] = {
            "left_to_right_errors": sum(
                item["ground_truth_concept_id"] == left
                and item["selected_concept_id"] == right for item in relevant
            ),
            "right_to_left_errors": sum(
                item["ground_truth_concept_id"] == right
                and item["selected_concept_id"] == left for item in relevant
            ),
            "mean_margin": float(np.mean([
                item["contrastive_margin"] for item in relevant
            ])) if relevant else None,
            "trial_count": len(relevant),
        }
    return {"pair_contrasts": matrix,
            "mean_confidence": float(np.mean([item["confidence"] for item in trial_results]))
            if trial_results else None}


def estimate_task_complexity(
    example: LearningExample,
    previous: LearningExample | None,
    known_feature_keys: set[str],
) -> dict[str, Any]:
    """Estimate new context demands, category switches, and unknown feature burden."""
    new_contexts = (_context_change_count(previous.context_features, example.context_features)
                    if previous else len(example.context_features))
    switches = int(previous is not None and previous.concept_id != example.concept_id)
    unknown = len(set(example.concept_features) - known_feature_keys)
    complexity = 0.25 * new_contexts + 0.35 * switches + 0.20 * unknown
    return {"new_context_dimensions": new_contexts, "category_switch": switches,
            "unknown_feature_count": unknown, "complexity_estimate": complexity}


def adapt_variation_level(
    current_level: float,
    accuracy: float,
    effort: float,
    target_accuracy: float,
    config: VariedContextsConfig,
) -> dict[str, float]:
    """Adjust the context-variation level with bounded accuracy and effort feedback."""
    delta = 0.18 * (clamp(accuracy) - target_accuracy) - 0.10 * clamp(effort)
    updated = clamp(current_level + delta, config.minimum_variation,
                    config.maximum_variation)
    return {"previous_variation_level": current_level, "variation_delta": delta,
            "updated_variation_level": updated, "accuracy": accuracy, "effort": effort}


def retrieve_by_cues(
    concept: ConceptDefinition,
    examples: Sequence[LearningExample],
    cue_type: str,
    cue_context: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Estimate concept retrieval through alternate modality, context, and relation cues."""
    selected = [item for item in examples if item.concept_id == concept.concept_id]
    if not selected:
        return {"retrieval_probability": 0.0, "matched_routes": [], "retrieved": False}
    route_matches = []
    for item in selected:
        score = 0.0
        if cue_type == item.modality:
            score += 0.55
        if cue_type == item.representation:
            score += 0.35
        if cue_context:
            overlap = sum(cue_context.get(key) == value
                          for key, value in item.context_features.items())
            score += 0.10 * overlap / max(1, len(item.context_features))
        route_matches.append({"example_id": item.example_id, "route_score": clamp(score),
                              "modality": item.modality,
                              "context_id": context_id(item.context_features)})
    best = max(route_matches, key=lambda row: row["route_score"])
    return {"retrieval_probability": best["route_score"],
            "matched_routes": route_matches,
            "best_route": best,
            "retrieved": best["route_score"] >= 0.25}


def _relation_transfer(
    concept: ConceptDefinition, observation: TrialObservation | None,
) -> dict[str, Any]:
    """Calculate relation preservation and incompatible mapping penalties for analogy tasks."""
    required = set(concept.relations)
    if not required:
        required = set(observation.preserved_relations) if observation else set()
    preserved = set(observation.preserved_relations) if observation else set()
    incompatible = set(observation.incompatible_relations) if observation else set()
    coverage = len(preserved & required) / max(1, len(required))
    incompatible_rate = len(incompatible) / max(1, len(required))
    score = clamp(coverage - 0.5 * incompatible_rate)
    return {"required_relations": sorted(required),
            "preserved_relations": sorted(preserved),
            "incompatible_relations": sorted(incompatible),
            "relation_coverage": coverage,
            "incompatible_relation_rate": incompatible_rate,
            "mapping_score": score,
            "valid_mapping": score >= 0.50}


def transfer_assessment(
    concepts: Sequence[ConceptDefinition],
    examples: Sequence[LearningExample],
    observations: Sequence[TrialObservation],
) -> dict[str, Any]:
    """Report near and far transfer separately, including structural relation-preservation evidence."""
    concept_map = {item.concept_id: item for item in concepts}
    example_map = {item.example_id: item for item in examples}
    observation_map = _observations_by_example(observations)
    reports: dict[str, Any] = {}
    for level in ("near", "far", "delayed"):
        selected = [item for item in examples if item.transfer_level == level]
        scores: list[float] = []
        relation_reports: list[dict[str, Any]] = []
        for example in selected:
            observation = observation_map.get(example.example_id)
            if observation:
                scores.append(float(observation.selected_concept_id == example.concept_id))
                if level == "far":
                    relation_reports.append(_relation_transfer(
                        concept_map[example.concept_id], observation
                    ))
        reports[level] = {
            "trial_count": len(selected),
            "observed_trial_count": len(scores),
            "accuracy": float(np.mean(scores)) if scores else None,
            "relation_mapping": relation_reports,
            "mean_relation_mapping_score": float(np.mean([
                row["mapping_score"] for row in relation_reports
            ])) if relation_reports else None,
            "unseen_examples_only": bool(selected) and all(item.is_novel for item in selected),
        }
    training = [item for item in examples if item.transfer_level == "training"]
    training_observed = [
        float(observation_map[item.example_id].selected_concept_id == item.concept_id)
        for item in training if item.example_id in observation_map
    ]
    training_accuracy = float(np.mean(training_observed)) if training_observed else None
    near_accuracy = reports["near"]["accuracy"]
    reports["near"]["training_accuracy"] = training_accuracy
    reports["near"]["generalization_gap"] = (
        training_accuracy - near_accuracy
        if training_accuracy is not None and near_accuracy is not None else None
    )
    return reports


def apply_context_boundaries(
    example: LearningExample, concepts: Sequence[ConceptDefinition],
) -> dict[str, Any]:
    """Restore contextual exceptions and report when a boundary condition changes concept applicability."""
    results: dict[str, Any] = {}
    for concept in concepts:
        conditions = dict(concept.boundary_conditions)
        matched = [key for key, expected in conditions.items()
                   if example.context_features.get(key) == expected]
        contradicted = [key for key, expected in conditions.items()
                        if key in example.context_features
                        and example.context_features.get(key) != expected]
        unavailable = [key for key in conditions if key not in example.context_features]
        results[concept.concept_id] = {
            "boundary_conditions": conditions,
            "matched_conditions": matched,
            "contradicted_conditions": contradicted,
            "unobserved_conditions": unavailable,
            "condition_satisfied": not contradicted,
            "context_required_for_decision": bool(conditions),
        }
    return {"example_id": example.example_id, "concept_conditions": results}


def consolidate_concept_memory(
    state: VariedContextsState,
    concepts: Sequence[ConceptDefinition],
    examples: Sequence[LearningExample],
    config: VariedContextsConfig,
) -> dict[str, Any]:
    """Update feature prototypes while preserving individual episode and context records."""
    concept_reports: dict[str, Any] = {}
    for concept in concepts:
        memory = state.concepts.setdefault(concept.concept_id,
                                           ConceptMemoryState(concept.concept_id))
        selected = [item for item in examples
                    if item.concept_id == concept.concept_id and item.verified]
        for example in selected:
            if not memory.prototype:
                memory.prototype = dict(example.concept_features)
                memory.feature_variance = {key: 0.0 for key in example.concept_features}
            else:
                keys = set(memory.prototype) | set(example.concept_features)
                old_count = max(1, memory.example_count)
                for key in keys:
                    old = memory.prototype.get(key, 0.0)
                    value = example.concept_features.get(key, 0.0)
                    new_mean = old + config.feature_learning_rate * (value - old)
                    memory.feature_variance[key] = (
                        (1.0 - config.feature_learning_rate) *
                        (memory.feature_variance.get(key, 0.0)
                         + config.feature_learning_rate * (value - old) * (value - new_mean))
                    )
                    memory.prototype[key] = new_mean
            memory.example_count += 1
            memory.modality_counts[example.modality] = memory.modality_counts.get(
                example.modality, 0
            ) + 1
            cid = context_id(example.context_features)
            for key, value in example.context_features.items():
                counts = memory.context_counts.setdefault(key, {})
                counts[value] = counts.get(value, 0) + 1
            state.episode_records[example.example_id] = {
                "concept_id": example.concept_id,
                "context_id": cid,
                "context_features": dict(example.context_features),
                "modality": example.modality,
                "representation": example.representation,
                "relations": list(example.relations),
                "transfer_level": example.transfer_level,
                "is_anchor": example.is_anchor,
                "verified": example.verified,
            }
        invariant = {
            key: value for key, value in memory.prototype.items()
            if memory.feature_variance.get(key, 0.0) <= 0.25
        }
        concept_reports[concept.concept_id] = {
            "prototype": dict(memory.prototype),
            "feature_variance": dict(memory.feature_variance),
            "candidate_invariants": invariant,
            "episode_count": memory.example_count,
            "modality_counts": dict(memory.modality_counts),
            "context_counts": dict(memory.context_counts),
            "episode_detail_preserved": True,
        }
    return {"concepts": concept_reports,
            "total_episodes_preserved": len(state.episode_records),
            "abstraction_did_not_delete_episode_records": True}


def diagnose_errors(
    concepts: Sequence[ConceptDefinition],
    examples: Sequence[LearningExample],
    observations: Sequence[TrialObservation],
    context_inferences: Mapping[str, Mapping[str, Any]],
    config: VariedContextsConfig,
) -> dict[str, Any]:
    """Classify concept confusion, context inference, modality, prerequisite, and overload errors."""
    concept_map = {item.concept_id: item for item in concepts}
    example_map = {item.example_id: item for item in examples}
    errors: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for observation in observations:
        example = example_map[observation.example_id]
        if observation.selected_concept_id == example.concept_id:
            continue
        inferred = context_inferences.get(example.example_id, {})
        if inferred.get("needs_more_cues"):
            category = "context_inference_uncertain"
        elif observation.selected_concept_id in {
            item for item in concept_map[example.concept_id].prerequisite_concepts
        }:
            category = "prerequisite_or_hierarchical_confusion"
        elif observation.cue_type not in {example.modality, example.representation, "original"}:
            category = "cue_or_modality_mismatch"
        elif observation.effort >= 0.85:
            category = "task_complexity_or_overload"
        else:
            category = "concept_feature_confusion"
        counts[category] += 1
        errors.append({
            "example_id": example.example_id, "true_concept": example.concept_id,
            "selected_concept": observation.selected_concept_id,
            "error_category": category, "confidence": observation.confidence,
            "effort": observation.effort,
            "context_uncertainty": inferred.get("uncertainty"),
        })
    return {"errors": errors, "error_counts": dict(counts),
            "priority_errors": sorted(counts, key=counts.get, reverse=True),
            "correction_required": bool(errors)}


def adaptive_practice_plan(
    concepts: Sequence[ConceptDefinition],
    profiles: Mapping[str, Mapping[str, Any]],
    errors: Mapping[str, Any],
    trial_results: Sequence[Mapping[str, Any]],
    current_variation: float,
    fatigue: float,
    available_time_minutes: float,
    config: VariedContextsConfig,
) -> dict[str, Any]:
    """Adapt variation, contrast, and practice priority to accuracy, error frequency, and effort."""
    observed = [item for item in trial_results if item["observation_available"]]
    accuracy = (float(np.mean([item["observed_selection_correct"] for item in observed]))
                if observed else 0.0)
    effort = float(np.mean([item["effort"] for item in observed])) if observed else fatigue
    variation = adapt_variation_level(
        current_variation, accuracy, effort, config.target_accuracy, config
    )
    priorities: dict[str, float] = {}
    for concept in concepts:
        profile = profiles[concept.concept_id]
        context_count = max(1, int(profile["context_count"]))
        error_count = sum(
            row["true_concept"] == concept.concept_id for row in errors["errors"]
        )
        uncertainty = 1.0 / context_count
        priorities[concept.concept_id] = clamp(
            0.35 * uncertainty + 0.40 * error_count / max(1, len(observed))
            + 0.25 * concept.importance
        )
    recommendations: dict[str, str] = {}
    for concept_id, score in priorities.items():
        if fatigue >= 0.80 or effort >= 0.85:
            recommendations[concept_id] = "reduce_context_changes_and_rebuild_anchor"
        elif score >= 0.65:
            recommendations[concept_id] = "targeted_contrastive_examples"
        elif accuracy >= config.target_accuracy:
            recommendations[concept_id] = "increase_valid_context_and_transfer_variety"
        else:
            recommendations[concept_id] = "repeat_clear_anchor_then_add_one_variation"
    return {"accuracy": accuracy, "mean_effort": effort,
            "variation_adjustment": variation,
            "concept_priority": priorities,
            "recommendations": recommendations,
            "available_time_minutes": available_time_minutes,
            "time_pressure": available_time_minutes < len(trial_results) * 1.5}


def evaluate_context_flexibility(
    concepts: Sequence[ConceptDefinition],
    examples: Sequence[LearningExample],
    observations: Sequence[TrialObservation],
    original_context_id: str | None = None,
) -> dict[str, Any]:
    """Measure performance by context, modality, interleaved selection, and delayed retention."""
    example_map = {item.example_id: item for item in examples}
    observations_map = _observations_by_example(observations)
    context_scores: dict[str, list[float]] = defaultdict(list)
    modality_scores: dict[str, list[float]] = defaultdict(list)
    level_scores: dict[str, list[float]] = defaultdict(list)
    for observation in observations:
        example = example_map[observation.example_id]
        score = float(observation.selected_concept_id == example.concept_id)
        context_scores[context_id(example.context_features)].append(score)
        modality_scores[example.modality].append(score)
        level_scores[example.transfer_level].append(score)
    context_accuracy = {key: float(np.mean(values))
                        for key, values in context_scores.items()}
    modality_accuracy = {key: float(np.mean(values))
                         for key, values in modality_scores.items()}
    original = original_context_id
    if original is None:
        anchors = [item for item in examples if item.is_anchor]
        original = context_id(anchors[0].context_features) if anchors else (
            next(iter(context_accuracy)) if context_accuracy else None
        )
    original_accuracy = context_accuracy.get(original) if original else None
    changed_values = [value for key, value in context_accuracy.items() if key != original]
    changed_accuracy = float(np.mean(changed_values)) if changed_values else None
    delayed = level_scores.get("delayed", [])
    near = level_scores.get("near", [])
    far = level_scores.get("far", [])
    train = level_scores.get("training", [])
    confusion = {concept.concept_id: {other.concept_id: 0 for other in concepts}
                 for concept in concepts}
    for observation in observations:
        truth = example_map[observation.example_id].concept_id
        confusion[truth][observation.selected_concept_id] += 1
    metrics = _precision_recall(confusion)
    return {
        "context_accuracy": context_accuracy,
        "modality_accuracy": modality_accuracy,
        "original_context_id": original,
        "original_context_accuracy": original_accuracy,
        "changed_context_accuracy": changed_accuracy,
        "context_robustness_gap": (
            original_accuracy - changed_accuracy
            if original_accuracy is not None and changed_accuracy is not None else None
        ),
        "context_flexibility_mean": float(np.mean(list(context_accuracy.values())))
        if context_accuracy else None,
        "training_accuracy": float(np.mean(train)) if train else None,
        "near_transfer_accuracy": float(np.mean(near)) if near else None,
        "far_transfer_accuracy": float(np.mean(far)) if far else None,
        "delayed_retention_accuracy": float(np.mean(delayed)) if delayed else None,
        "modality_coverage": len(modality_accuracy),
        "confusion_matrix": confusion,
        "discrimination_metrics": metrics,
        "evaluation_sample_count": len(observations),
        "separate_measures_prevent_single_score_masking_weaknesses": True,
    }


def update_state_from_observations(
    state: VariedContextsState,
    examples: Sequence[LearningExample],
    observations: Sequence[TrialObservation],
) -> dict[str, Any]:
    """Record observed cue routes, errors, context transitions, and confidence history."""
    example_map = {item.example_id: item for item in examples}
    transitions = 0
    for observation in observations:
        example = example_map[observation.example_id]
        memory = state.concepts.setdefault(example.concept_id,
                                           ConceptMemoryState(example.concept_id))
        memory.cue_counts[observation.cue_type] = memory.cue_counts.get(
            observation.cue_type, 0
        ) + 1
        if observation.selected_concept_id != example.concept_id:
            error = f"confused_with:{observation.selected_concept_id}"
            memory.error_counts[error] = memory.error_counts.get(error, 0) + 1
        memory.observation_history.append({
            "example_id": example.example_id,
            "selected_concept_id": observation.selected_concept_id,
            "confidence": observation.confidence,
            "effort": observation.effort,
            "cue_type": observation.cue_type,
            "correct": observation.selected_concept_id == example.concept_id,
            "transfer_level": example.transfer_level,
        })
        memory.observation_history[:] = memory.observation_history[-300:]
        cid = context_id(example.context_features)
        if state.interleaving_history:
            previous = state.interleaving_history[-1]
            if previous != example.concept_id:
                transitions += 1
                row = state.context_transition_counts.setdefault(previous, {})
                row[cid] = row.get(cid, 0) + 1
        state.interleaving_history.append(example.concept_id)
        state.interleaving_history[:] = state.interleaving_history[-500:]
    return {"observation_count": len(observations), "concept_switches": transitions,
            "cue_counts": {key: dict(value.cue_counts) for key, value in state.concepts.items()},
            "error_counts": {key: dict(value.error_counts) for key, value in state.concepts.items()}}
