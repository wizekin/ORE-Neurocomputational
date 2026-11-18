"""Functional and mathematical operations for working-memory overload (Point 19)."""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from .contracts import Dependency, Point19Config, Point19State, Representation


_TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)?|[^\w\s]", flags=re.UNICODE)


def clip01(value: float) -> float:
    """Clamp a finite model value to the inclusive unit interval."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("model values must be finite")
    return max(0.0, min(1.0, number))


def sigmoid(value: float) -> float:
    """Compute a numerically stable logistic activation for overload detection."""
    number = float(value)
    if number >= 0.0:
        z = math.exp(-min(number, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(number, -700.0))
    return z / (1.0 + z)


def softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Normalize candidate scores with a temperature-scaled, overflow-safe softmax."""
    if not values:
        return []
    scale = float(temperature)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("temperature must be finite and positive")
    scores = [float(value) for value in values]
    if any(not math.isfinite(value) for value in scores):
        raise ValueError("softmax scores must be finite")
    peak = max(scores)
    exponentials = [math.exp(max(-700.0, (value - peak) / scale)) for value in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def segment_units(text: str) -> list[str]:
    """Split supplied text into words and punctuation without claiming full syntactic parsing."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    return _TOKEN_PATTERN.findall(text)


def _numeric_feature(value: Any) -> float:
    """Convert a feature to a bounded strength while retaining categorical matches elsewhere."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return clip01(float(value))
    return 1.0 if value is not None and str(value) else 0.0


def feature_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Estimate feature overlap using equality for categories and cosine for numeric vectors."""
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    categorical = any(
        not isinstance(value, (int, float, bool))
        for mapping in (left, right) for value in mapping.values()
    )
    if categorical:
        comparable = [key for key in set(left) & set(right)]
        if not comparable:
            return 0.0
        matches = sum(left[key] == right[key] for key in comparable)
        # Extra features on a candidate are not negative evidence when a dependency query
        # only specifies a subset of the candidate representation.
        return matches / max(1, len(comparable))
    a = [_numeric_feature(left.get(key, 0.0)) for key in keys]
    b = [_numeric_feature(right.get(key, 0.0)) for key in keys]
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return clip01(dot / (norm_a * norm_b))


def _normalize_features(features: Mapping[str, Any]) -> dict[str, Any]:
    """Copy feature evidence while rejecting empty keys and non-finite numeric values."""
    normalized: dict[str, Any] = {}
    for key, value in features.items():
        label = str(key).strip()
        if not label:
            raise ValueError("feature keys must be non-empty")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                raise ValueError(f"feature {label!r} must be finite")
            normalized[label] = clip01(float(value))
        else:
            normalized[label] = value
    return normalized


def encode_unit(
    sequence_id: str,
    unit: str,
    position: int,
    annotation: Mapping[str, Any] | None = None,
    goal_relevance: float = 0.5,
) -> Representation:
    """Encode one observed unit, preserving caller-supplied features and source provenance."""
    supplied = dict(annotation or {})
    features = _normalize_features(supplied.get("features", {}))
    features.setdefault("surface_form", unit.casefold())
    if "syntactic_role" in supplied:
        features["syntactic_role"] = supplied["syntactic_role"]
    if "referent" in supplied:
        features["referent"] = supplied["referent"]
    if "semantic_type" in supplied:
        features["semantic_type"] = supplied["semantic_type"]
    identifier = str(supplied.get("identifier", f"{sequence_id}:u{position}"))
    confidence = clip01(float(supplied.get("confidence", 0.65)))
    return Representation(
        identifier=identifier,
        text=unit,
        features=features,
        position=position,
        source=str(supplied.get("source", "observed_input")),
        accessibility=clip01(float(supplied.get("accessibility", 1.0))),
        confidence=confidence,
        task_relevance=clip01(float(supplied.get("task_relevance", goal_relevance))),
        chunk_id=str(supplied["chunk_id"]) if supplied.get("chunk_id") is not None else None,
    )


def create_dependencies(
    specs: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    sequence_id: str,
) -> list[Dependency]:
    """Create only explicitly supplied open relations and dependency cues."""
    collected: list[dict[str, Any]] = [dict(item) for item in specs]
    for index, annotation in enumerate(annotations):
        opened = annotation.get("opens_dependency", ())
        if isinstance(opened, (str, bytes)):
            opened = (opened,)
        for item in opened:
            if isinstance(item, Mapping):
                collected.append(dict(item))
            else:
                collected.append({"required_features": {str(item): 1.0}})
    dependencies: list[Dependency] = []
    for index, spec in enumerate(collected):
        required = _normalize_features(spec.get("required_features", {}))
        cues = _normalize_features(spec.get("cue_features", {}))
        if not required and not cues:
            raise ValueError("each dependency requires required_features or cue_features")
        dependencies.append(Dependency(
            identifier=str(spec.get("identifier", f"{sequence_id}:d{index}")),
            required_features=required,
            cue_features=cues,
            priority=clip01(float(spec.get("priority", 0.7))),
        ))
    return dependencies


def update_accessibility(
    state: Point19State,
    current_position: int,
    config: Point19Config,
    contextual_support: Mapping[str, float] | None = None,
    refreshed_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Update item accessibility from contextual support, refreshing, elapsed distance, and interference."""
    support = contextual_support or {}
    refreshed = refreshed_ids or set()
    rows: dict[str, dict[str, float]] = {}
    active = [item for item in state.representations.values() if item.active]
    for item in active:
        competitors = [other for other in active if other.identifier != item.identifier]
        interference = sum(
            feature_similarity(item.features, other.features) * other.accessibility
            for other in competitors
        ) / max(1, len(competitors))
        distance = max(0, current_position - item.position)
        contextual_gain = clip01(float(support.get(item.identifier, 0.0)))
        refresh_gain = config.refresh_gain if item.identifier in refreshed else 0.0
        decay = config.retention_decay * min(1.0, distance / 12.0)
        before = item.accessibility
        item.interference = clip01(interference)
        item.accessibility = clip01(
            before + contextual_gain + refresh_gain - decay
            - config.interference_gain * item.interference
        )
        rows[item.identifier] = {
            "before": before,
            "after": item.accessibility,
            "interference": item.interference,
            "distance": float(distance),
            "contextual_gain": contextual_gain,
            "refresh_gain": refresh_gain,
            "decay": decay,
        }
    return {"items": rows, "active_item_count": len(active)}


def estimate_load(
    state: Point19State,
    structural_complexity: float,
    extra_task_load: float,
    uncertainty: float,
    context_support: float,
    config: Point19Config,
    current_position: int | None = None,
) -> dict[str, Any]:
    """Estimate normalized demand from dependencies, complexity, distance, interference, and uncertainty."""
    open_dependencies = [item for item in state.dependencies.values() if item.open]
    representations = [item for item in state.representations.values() if item.active]
    dependency_cost = min(1.0, len(open_dependencies) / config.dependency_scale)
    position = state.step_index if current_position is None else int(current_position)
    distances = [max(0, position - item.position) for item in representations]
    retrieval_distance = min(1.0, (sum(distances) / max(1, len(distances))) / 20.0)
    interference = sum(item.interference for item in representations) / max(1, len(representations))
    components = {
        "unresolved_dependencies": clip01(dependency_cost),
        "structural_complexity": clip01(structural_complexity),
        "retrieval_distance": clip01(retrieval_distance),
        "similarity_interference": clip01(interference),
        "additional_task_demand": clip01(extra_task_load),
        "uncertainty": clip01(uncertainty),
    }
    weights = dict(state.adaptive_weights)
    burden = sum(weights.get(name, 0.0) * value for name, value in (
        ("dependency", components["unresolved_dependencies"]),
        ("complexity", components["structural_complexity"]),
        ("distance", components["retrieval_distance"]),
        ("interference", components["similarity_interference"]),
        ("task", components["additional_task_demand"]),
        ("uncertainty", components["uncertainty"]),
    ))
    chunk_relief = min(0.18, 0.035 * len(state.chunks))
    contextual_relief = 0.12 * clip01(context_support)
    state.estimated_load = clip01(burden - chunk_relief - contextual_relief)
    return {
        "load": state.estimated_load,
        "components": components,
        "weights": weights,
        "chunk_relief": chunk_relief,
        "contextual_relief": contextual_relief,
        "open_dependency_count": len(open_dependencies),
        "active_representation_count": len(representations),
        "capacity_interpretation": "Relative task demand; no universal fixed human item or word capacity is assumed.",
    }


def form_chunks(state: Point19State) -> dict[str, Any]:
    """Group only explicitly tagged items into reversible chunks with preserved token links."""
    groups: dict[str, list[Representation]] = {}
    for item in state.representations.values():
        if item.active and item.chunk_id:
            groups.setdefault(item.chunk_id, []).append(item)
    chunks: list[dict[str, Any]] = []
    for chunk_id, items in groups.items():
        if len(items) < 2:
            continue
        ordered = sorted(items, key=lambda item: item.position)
        chunk = {
            "chunk_id": chunk_id,
            "member_ids": [item.identifier for item in ordered],
            "surface": " ".join(item.text for item in ordered),
            "member_count": len(ordered),
            "effective_cost": 1.0 + 0.25 * math.log1p(len(ordered)),
            "grouping_source": "explicit_input_annotation",
            "reversible": True,
        }
        chunks.append(chunk)
    state.chunks = chunks
    return {"chunks": chunks, "chunk_count": len(chunks), "member_count": sum(x["member_count"] for x in chunks)}


def allocate_attention(
    state: Point19State,
    temperature: float,
) -> dict[str, Any]:
    """Allocate a normalized priority distribution between new-processing and maintenance candidates."""
    items = [item for item in state.representations.values() if item.active]
    scores = [
        0.40 * item.task_relevance + 0.35 * item.accessibility
        + 0.15 * (1.0 - item.interference) + 0.10 * (1.0 if item.chunk_id else 0.0)
        for item in items
    ]
    probabilities = softmax(scores, temperature)
    ranked = sorted(
        ({"identifier": item.identifier, "text": item.text, "score": score, "attention": probability}
         for item, score, probability in zip(items, scores, probabilities)),
        key=lambda row: row["attention"], reverse=True,
    )
    return {"ranked_items": ranked, "attention_sum": sum(probabilities), "item_count": len(items)}


def estimate_interference(state: Point19State) -> dict[str, Any]:
    """Quantify similarity-weighted competition for each active item and retrieval cue."""
    items = [item for item in state.representations.values() if item.active]
    estimates: dict[str, float] = {}
    for target in items:
        competitors = [other for other in items if other.identifier != target.identifier]
        value = sum(
            feature_similarity(target.features, other.features) * other.accessibility
            for other in competitors
        ) / max(1, len(competitors))
        target.interference = clip01(value)
        estimates[target.identifier] = target.interference
    return {
        "items": estimates,
        "mean_interference": sum(estimates.values()) / max(1, len(estimates)),
        "interpretation": "Similarity-weighted competitive pressure, not evidence of literal neural inhibition values.",
    }


def retrieve_by_cue(
    dependency: Dependency,
    candidates: Sequence[Representation],
    config: Point19Config,
) -> dict[str, Any]:
    """Rank candidate representations by explicit feature cues, accessibility, relevance, and competition."""
    scored: list[tuple[float, Representation, float, float]] = []
    for item in candidates:
        required_match = feature_similarity(dependency.required_features, item.features)
        cue_match = feature_similarity(dependency.cue_features, item.features) if dependency.cue_features else 0.0
        score = (
            0.58 * required_match + 0.17 * cue_match
            + 0.17 * item.accessibility + 0.08 * item.task_relevance
            - 0.15 * item.interference
        )
        scored.append((score, item, required_match, cue_match))
    if not scored:
        return {"dependency_id": dependency.identifier, "candidates": [], "selected": None, "confidence": 0.0}
    probabilities = softmax([row[0] for row in scored], config.retrieval_temperature)
    ranked = sorted(
        ({
            "identifier": row[1].identifier,
            "text": row[1].text,
            "score": row[0],
            "probability": probability,
            "required_match": row[2],
            "cue_match": row[3],
            "accessibility": row[1].accessibility,
        } for row, probability in zip(scored, probabilities)),
        key=lambda item: item["probability"], reverse=True,
    )
    best = ranked[0]
    second_probability = ranked[1]["probability"] if len(ranked) > 1 else 0.0
    margin = max(0.0, best["probability"] - second_probability)
    # Relative softmax probability alone is not evidence strength: a lone poor candidate
    # would otherwise receive probability 1.0. Require absolute cue/evidence support too.
    evidence_strength = (
        0.58 * float(best["required_match"])
        + 0.17 * float(best["cue_match"])
        + 0.17 * float(best["accessibility"])
        + 0.08 * clip01(float(best.get("score", 0.0)))
    )
    confidence = clip01((0.55 * float(best["probability"]) + 0.45 * margin) * evidence_strength)
    selected = best if confidence >= config.retrieval_threshold else None
    return {
        "dependency_id": dependency.identifier,
        "candidates": ranked,
        "selected": selected,
        "confidence": confidence,
        "margin": margin,
        "status": "retrieved" if selected else "ambiguous_or_weak_retrieval",
    }


def integrate_retrieval(
    state: Point19State,
    dependency: Dependency,
    retrieval: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a retrieved item only when confidence passes the configured upstream threshold."""
    selected = retrieval.get("selected")
    confidence = clip01(float(retrieval.get("confidence", 0.0)))
    if not selected:
        dependency.confidence = confidence
        dependency.failure_type = "unresolved"
        return {"dependency_id": dependency.identifier, "integrated": False, "reason": "no sufficiently supported candidate"}
    candidate_id = str(selected["identifier"])
    candidate = state.representations.get(candidate_id)
    if candidate is None:
        dependency.failure_type = "binding_failure"
        return {"dependency_id": dependency.identifier, "integrated": False, "reason": "candidate reference is missing"}
    dependency.candidate_ids = [candidate_id]
    dependency.confidence = confidence
    dependency.open = False
    dependency.failure_type = "none"
    relation = {
        "dependency_id": dependency.identifier,
        "candidate_id": candidate_id,
        "content": candidate.text,
        "confidence": confidence,
        "provenance": candidate.source,
        "status": "supported_retrieval",
    }
    state.event_model.append(relation)
    candidate.accessibility = clip01(candidate.accessibility + 0.08)
    return {"dependency_id": dependency.identifier, "integrated": True, "relation": relation}


def detect_overload(
    state: Point19State,
    metrics: Mapping[str, float],
    context_support: float,
    threshold: float = 0.62,
) -> dict[str, Any]:
    """Estimate overload from load, slowing, reanalysis, and context, using a calibrated threshold."""
    slowing = clip01(float(metrics.get("processing_slowing", metrics.get("slowing", 0.0))))
    reanalysis = clip01(float(metrics.get("reanalysis_rate", metrics.get("reanalysis", 0.0))))
    errors = clip01(float(metrics.get("error_rate", metrics.get("error", 0.0))))
    z = -1.65 + 1.55 * state.estimated_load + 0.55 * slowing + 0.80 * reanalysis + 0.90 * errors - 0.45 * clip01(context_support)
    probability = sigmoid(z)
    state.overload_probability = probability
    return {
        "probability": probability,
        "detected": probability >= clip01(threshold),
        "threshold": clip01(threshold),
        "signals": {
            "estimated_load": state.estimated_load,
            "processing_slowing": slowing,
            "reanalysis_rate": reanalysis,
            "error_rate": errors,
            "context_support": clip01(context_support),
        },
        "model_note": "A graded model estimate; no single behavioral sign establishes overload.",
    }


def identify_at_risk(state: Point19State) -> dict[str, Any]:
    """Rank at-risk items and relations by low accessibility, interference, and unresolved demand."""
    rows: list[dict[str, Any]] = []
    open_count = sum(item.open for item in state.dependencies.values())
    for item in state.representations.values():
        if not item.active:
            continue
        dependency_demand = max(
            (dep.priority for dep in state.dependencies.values() if dep.open), default=0.0
        )
        uncertainty = 1.0 - item.confidence
        risk = clip01(0.34 * (1.0 - item.accessibility) + 0.30 * item.interference + 0.22 * dependency_demand + 0.14 * uncertainty)
        rows.append({"identifier": item.identifier, "text": item.text, "risk": risk,
                     "accessibility": item.accessibility, "interference": item.interference,
                     "open_dependency_count": open_count})
    return {"at_risk_items": sorted(rows, key=lambda row: row["risk"], reverse=True)}


def choose_recovery_action(
    request_context: Mapping[str, Any],
    load: float,
    at_risk_count: int,
    config: Point19Config,
) -> dict[str, Any]:
    """Choose the action with highest expected accuracy benefit after load and time costs."""
    supplied = request_context.get("recovery_options")
    options = list(supplied) if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)) else []
    if not options:
        options = [
            {"action": "reactivation", "accuracy_gain": 0.68, "extra_load": 0.08, "time_cost": 0.12},
            {"action": "rehearsal", "accuracy_gain": 0.55, "extra_load": 0.16, "time_cost": 0.10},
            {"action": "chunk_restructuring", "accuracy_gain": 0.50, "extra_load": 0.05, "time_cost": 0.14},
            {"action": "reanalysis", "accuracy_gain": 0.60, "extra_load": 0.24, "time_cost": 0.30},
            {"action": "clarification", "accuracy_gain": 0.82, "extra_load": 0.04, "time_cost": 0.42},
        ]
    ranked: list[dict[str, Any]] = []
    for option in options:
        item = dict(option)
        gain = clip01(float(item.get("accuracy_gain", 0.0)))
        extra = clip01(float(item.get("extra_load", 0.0)))
        time = clip01(float(item.get("time_cost", 0.0)))
        net = gain - config.recovery_load_cost * extra * (0.5 + load) - config.recovery_time_cost * time
        ranked.append({"action": str(item.get("action", "clarification")), "accuracy_gain": gain,
                       "extra_load": extra, "time_cost": time, "net_value": net})
    ranked.sort(key=lambda item: item["net_value"], reverse=True)
    selected = ranked[0] if ranked else {"action": "preserve_uncertainty", "net_value": 0.0}
    return {"selected_action": selected, "ranked_actions": ranked,
            "at_risk_count": at_risk_count, "decision_rule": "expected benefit minus load and time costs"}


def apply_recovery(
    state: Point19State,
    selected_action: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a bounded recovery operation to existing evidence without creating missing source facts."""
    action = str(selected_action.get("action", "preserve_uncertainty"))
    open_dependencies = [item for item in state.dependencies.values() if item.open]
    affected: list[str] = []
    if action in {"reactivation", "rehearsal", "reread", "clarification"}:
        for dependency in open_dependencies:
            candidates = [state.representations[key] for key in dependency.candidate_ids if key in state.representations]
            if not candidates:
                candidates = list(state.representations.values())
            for candidate in candidates[:3]:
                candidate.accessibility = clip01(candidate.accessibility + 0.14)
                affected.append(candidate.identifier)
    elif action == "chunk_restructuring":
        for item in state.representations.values():
            if item.active and item.chunk_id:
                item.accessibility = clip01(item.accessibility + 0.06)
                affected.append(item.identifier)
    elif action == "reanalysis":
        for item in state.representations.values():
            if item.active and item.accessibility < 0.8:
                item.accessibility = clip01(item.accessibility + 0.08)
                affected.append(item.identifier)
    return {"action": action, "affected_ids": sorted(set(affected)),
            "remaining_open_dependencies": len(open_dependencies),
            "recovered_from_existing_evidence_only": True}


def diagnose_failures(
    state: Point19State,
    retrievals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify weak dependencies as encoding, maintenance, retrieval, binding, or integration failures."""
    retrieval_by_id = {str(item.get("dependency_id")): item for item in retrievals}
    diagnoses: list[dict[str, Any]] = []
    for dependency in state.dependencies.values():
        if not dependency.open:
            continue
        retrieval = retrieval_by_id.get(dependency.identifier, {})
        ranked = retrieval.get("candidates", [])
        if not ranked:
            failure = "encoding_weak"
        elif max(float(item.get("accessibility", 0.0)) for item in ranked) < 0.25:
            failure = "maintenance_weak"
        elif len(ranked) > 1 and float(ranked[0].get("probability", 0.0)) - float(ranked[1].get("probability", 0.0)) < 0.08:
            failure = "retrieval_interference"
        elif ranked and float(ranked[0].get("required_match", 0.0)) < 0.20:
            failure = "binding_failure"
        else:
            failure = "integration_failure"
        dependency.failure_type = failure
        diagnoses.append({"dependency_id": dependency.identifier, "failure_type": failure,
                          "candidate_count": len(ranked), "confidence": dependency.confidence})
    return {"diagnoses": diagnoses, "failure_count": len(diagnoses)}


def reconstruct_supported(
    state: Point19State,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Separate source-supported reconstructions from unknown details and caller-supplied hypotheses."""
    supplied = context.get("candidate_reconstructions", ())
    if isinstance(supplied, Mapping):
        supplied = (supplied,)
    hypotheses: list[dict[str, Any]] = []
    if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)):
        for candidate in supplied:
            if not isinstance(candidate, Mapping):
                continue
            raw_evidence_ids = candidate.get("evidence_ids", ())
            if isinstance(raw_evidence_ids, (str, bytes)):
                raw_evidence_ids = (raw_evidence_ids,)
            evidence_ids = [str(value) for value in raw_evidence_ids]
            supported_ids = [value for value in evidence_ids if value in state.representations]
            support = len(supported_ids) / max(1, len(evidence_ids))
            hypotheses.append({
                "content": candidate.get("content"),
                "evidence_ids": evidence_ids,
                "supported_evidence_ids": supported_ids,
                "support": clip01(support),
                "provenance": "caller_supplied_hypothesis",
                "status": "inferred" if support > 0.0 else "unsupported",
            })
    supported = [item for item in state.event_model if item.get("status") == "supported_retrieval"]
    return {
        "supported_relations": [dict(item) for item in supported],
        "candidate_reconstructions": hypotheses,
        "unknowns": [item.identifier for item in state.dependencies.values() if item.open],
        "invented_content": False,
    }


def check_coherence(
    state: Point19State,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Score supplied grammatical, semantic, and discourse checks while exposing missing checks."""
    names = ("grammatical_consistency", "semantic_coherence", "discourse_fit")
    values: dict[str, float] = {}
    missing: list[str] = []
    for name in names:
        if name in context:
            values[name] = clip01(float(context[name]))
        else:
            values[name] = 0.5
            missing.append(name)
    contradictions = max(0.0, float(context.get("contradiction_count", 0.0)))
    score = clip01(
        0.34 * values[names[0]] + 0.36 * values[names[1]]
        + 0.30 * values[names[2]] - 0.12 * min(1.0, contradictions)
    )
    open_count = sum(item.open for item in state.dependencies.values())
    return {
        "score": score,
        "component_scores": values,
        "missing_checks": missing,
        "contradiction_count": contradictions,
        "open_dependency_count": open_count,
        "status": "needs_review" if contradictions or open_count else ("provisional" if missing else "coherent"),
    }


def commit_interpretation(
    state: Point19State,
    reconstruction: Mapping[str, Any],
    coherence: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit supported relations and explicit unknowns while keeping inferred content separate."""
    supported = [dict(item) for item in reconstruction.get("supported_relations", [])]
    inferred = [dict(item) for item in reconstruction.get("candidate_reconstructions", []) if item.get("status") == "inferred"]
    unresolved = [
        {"dependency_id": item.identifier, "required_features": dict(item.required_features),
         "failure_type": item.failure_type, "confidence": item.confidence}
        for item in state.dependencies.values() if item.open
    ]
    state.uncertainty = list(unresolved)
    return {
        "supported": supported,
        "inferred": inferred,
        "unresolved": unresolved,
        "coherence": dict(coherence),
        "confidence_policy": "Confidence is attached to each relation; unsupported details remain unknown.",
        "durable_episodic_storage_performed": False,
    }


def adapt_difficulty(
    state: Point19State,
    metrics: Mapping[str, float],
    config: Point19Config,
) -> dict[str, Any]:
    """Update task-specific load weights from supplied prediction error and driver signals."""
    error = float(metrics.get("predicted_difficulty_error", 0.0))
    if not math.isfinite(error):
        raise ValueError("predicted_difficulty_error must be finite")
    drivers = metrics.get("difficulty_drivers", {})
    if not isinstance(drivers, Mapping):
        drivers = {}
    before = dict(state.adaptive_weights)
    for name, weight in list(state.adaptive_weights.items()):
        driver = clip01(float(drivers.get(name, 0.0)))
        direction = 1.0 if error > 0.0 else -1.0 if error < 0.0 else 0.0
        state.adaptive_weights[name] = clip01(weight + config.adaptation_rate * direction * driver * min(1.0, abs(error)))
    return {
        "weights_before": before,
        "weights_after": dict(state.adaptive_weights),
        "prediction_error": error,
        "drivers": {str(key): clip01(float(value)) for key, value in drivers.items()},
        "mean_absolute_weight_change": sum(abs(state.adaptive_weights[key] - before[key]) for key in before) / max(1, len(before)),
        "update_rule": "bounded signed delta update; a computational adaptation hypothesis, not literal neural gradient descent",
    }
