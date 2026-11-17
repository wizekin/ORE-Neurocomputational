"""Functional implementation of the 36-stage distributed semantic algorithm.

The pipeline is sequentially inspectable but recurrent in interpretation: context,
prediction, control, memory, and learning return information to earlier semantic
representations. Its biological track is a separate synthetic LIF reference.
"""
from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Callable, Mapping, Sequence
import math
import re
import unicodedata

import numpy as np

from ore_semantic.contracts import SemanticConcept

from .biology import SemanticLIFSimulator, biological_drives
from .contracts import (
    DistributedSemanticConfig,
    DistributedSemanticRequest,
    DistributedSemanticResult,
    normalize_modalities,
)
from .registry import ALL_OPERATIONS, STAGE_REGISTRY, StageSpec


State = dict[str, Any]
_EPS = 1e-9
_VERBS = {
    "tune", "tuned", "plays", "play", "played", "give", "gave", "gives",
    "eat", "eats", "ate", "drink", "drinks", "drank", "see", "saw",
    "use", "used", "make", "made", "buy", "bought", "take", "took",
    "enter", "entered", "adjust", "adjusted", "raise", "raised",
    "paint", "painted", "hold", "held", "open", "opened", "break", "broke",
}
_SELECTION_FEATURES = {
    "tune": {"MUSICAL_INSTRUMENT": 1.0, "MACHINE": 0.5},
    "tuned": {"MUSICAL_INSTRUMENT": 1.0, "MACHINE": 0.5},
    "play": {"MUSICAL_INSTRUMENT": 0.8, "GAME": 0.7, "SPORT": 0.6},
    "played": {"MUSICAL_INSTRUMENT": 0.8, "GAME": 0.7, "SPORT": 0.6},
    "eat": {"FOOD": 1.0, "EDIBLE": 1.0},
    "eats": {"FOOD": 1.0, "EDIBLE": 1.0},
    "drink": {"LIQUID": 1.0, "BEVERAGE": 1.0},
    "drinks": {"LIQUID": 1.0, "BEVERAGE": 1.0},
    "give": {"PERSON": 0.6, "ANIMATE": 0.6},
    "gave": {"PERSON": 0.6, "ANIMATE": 0.6},
}
_STAGE_HANDLERS: dict[int, Callable[[State, DistributedSemanticRequest, "DistributedSemanticProcessor"], dict[str, Any]]] = {}


def _normalize_text(value: str) -> str:
    """Normalize a lexical label for comparison without changing the user's original text."""
    normalized = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"^\W+|\W+$", "", normalized, flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Split supplied text into Unicode-aware lexical tokens while preserving token order."""
    return re.findall(r"[\w'-]+", text, flags=re.UNICODE)


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Bound a scalar to a finite computational interval."""
    if not math.isfinite(float(value)):
        return lower
    return float(min(upper, max(lower, value)))


def _softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    """Convert finite candidate scores into a numerically stable probability-like distribution."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.size == 0:
        return vector
    scaled = vector / max(float(temperature), _EPS)
    scaled -= float(np.max(scaled))
    exp_values = np.exp(np.clip(scaled, -60.0, 60.0))
    total = float(np.sum(exp_values))
    return exp_values / total if total > _EPS else np.full(vector.size, 1.0 / vector.size)


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Calculate cosine similarity over sparse named feature maps with a defined zero case."""
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    a = np.asarray([float(left.get(key, 0.0)) for key in keys], dtype=np.float64)
    b = np.asarray([float(right.get(key, 0.0)) for key in keys], dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > _EPS else 0.0


def _merge_features(*maps: Mapping[str, float], scale: float = 1.0) -> dict[str, float]:
    """Merge compatible feature maps using a bounded maximum-preserving pooling operation."""
    merged: dict[str, float] = {}
    for feature_map in maps:
        for key, value in feature_map.items():
            merged[str(key)] = max(merged.get(str(key), 0.0), _clip(float(value) * scale))
    return merged


def _entropy(probabilities: Sequence[float]) -> float:
    """Calculate Shannon entropy in nats for a finite probability distribution."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.size == 0:
        return 0.0
    values = np.clip(values, _EPS, 1.0)
    values /= float(np.sum(values))
    return float(-np.sum(values * np.log(values)))


def _concept_modalities(concept: SemanticConcept) -> dict[str, dict[str, float]]:
    """Normalize a concept's known modality maps while ignoring unsupported legacy labels."""
    result: dict[str, dict[str, float]] = {}
    for raw_modality, raw_features in concept.modality_features.items():
        try:
            modality_map = normalize_modalities({raw_modality: raw_features})
        except (TypeError, ValueError):
            continue
        for modality, features in modality_map.items():
            result[modality] = _merge_features(result.get(modality, {}), features)
    return result


def _concept_features(concept: SemanticConcept) -> dict[str, float]:
    """Combine caller-supplied shared and modality-specific concept features into one view."""
    maps = [concept.semantic_features]
    maps.extend(_concept_modalities(concept).values())
    return _merge_features(*maps)


def _find_concept_id(label: str, candidates: Mapping[str, Sequence[Mapping[str, Any]]]) -> str | None:
    """Resolve a surface label to the highest-ranked known concept candidate."""
    values = candidates.get(_normalize_text(label), ())
    return str(values[0]["concept_id"]) if values else None


def _string_similarity(left: str, right: str) -> float:
    """Return normalized character-sequence similarity for bounded lexical fallback matching."""
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(None, left, right).ratio())


def _stage_activity(value: Mapping[str, Any]) -> float:
    """Estimate bounded functional activity from numeric confidence or activation fields."""
    for key in ("activity", "confidence", "score", "coverage", "capacity"):
        raw = value.get(key)
        if isinstance(raw, (int, float)):
            return _clip(float(raw))
    return 0.4


def _candidate_records(state: State) -> list[dict[str, Any]]:
    """Flatten token-indexed candidates into records for downstream integration."""
    flattened: list[dict[str, Any]] = []
    for token, records in state.get("candidates_by_token", {}).items():
        for record in records:
            flattened.append({**record, "token": token})
    return flattened


def _select_top_candidate(state: State, token: str) -> dict[str, Any] | None:
    """Return the currently selected candidate record for a normalized token."""
    normalized = _normalize_text(token)
    selected = state.get("selected_by_token", {}).get(normalized)
    if selected:
        return selected
    records = state.get("candidate_posteriors", {}).get(normalized, ())
    return records[0] if records else None


def _feature_count_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Compute overlap similarity for concept feature maps with graded activation values."""
    return _cosine(left, right)


def _guess_syntax(tokens: Sequence[str]) -> list[dict[str, str]]:
    """Build a small transparent subject-predicate-object heuristic when no parser is supplied."""
    normalized = [_normalize_text(token) for token in tokens]
    for index, token in enumerate(normalized):
        if token not in _VERBS:
            continue
        relations: list[dict[str, str]] = []
        if index > 0:
            relations.append({"head": tokens[index], "dependent": tokens[index - 1], "relation": "subject"})
        if index + 1 < len(tokens):
            relations.append({"head": tokens[index], "dependent": tokens[index + 1], "relation": "object"})
        return relations
    return []


def _build_concept_evidence(processor: "DistributedSemanticProcessor") -> dict[str, dict[str, Any]]:
    """Build an immutable-style lookup view over concept inventory and authorized learned features."""
    evidence: dict[str, dict[str, Any]] = {}
    for concept in processor.concepts:
        learned = processor._learned_features.get(concept.concept_id, {})
        evidence[concept.concept_id] = {
            "concept_id": concept.concept_id,
            "label": concept.label,
            "aliases": tuple(concept.aliases),
            "lexical_category": concept.lexical_category,
            "prototype_id": concept.prototype_id,
            "frequency": float(concept.frequency),
            "features": _merge_features(_concept_features(concept), learned),
            "modalities": _concept_modalities(concept),
            "verified_by": concept.verified_by,
            "source_reference": concept.source_reference,
        }
    return evidence


def _stage_01(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Acquire parallel modality channels and preserve their source and availability metadata."""
    inputs = normalize_modalities(request.modality_inputs)
    if request.text and "linguistic" not in inputs:
        inputs["linguistic"] = {"TEXT_PRESENT": 1.0}
    state["modality_features"] = inputs
    state["tokens"] = list(request.tokens) if request.tokens else _tokenize(request.text)
    state["concept_evidence"] = _build_concept_evidence(processor)
    state["available_modalities"] = sorted(inputs)
    return {"available_modalities": sorted(inputs), "token_count": len(state["tokens"]), "activity": min(1.0, len(inputs) / 5.0 + 0.15)}


def _stage_02(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Identify token candidates from inventory labels and aliases without forcing uncertain matches."""
    candidates: dict[str, list[dict[str, Any]]] = {}
    for token in state["tokens"]:
        normalized = _normalize_text(token)
        matches: list[dict[str, Any]] = []
        for concept_id, evidence in state["concept_evidence"].items():
            aliases = (evidence["label"], *evidence["aliases"])
            lexical = max((_string_similarity(normalized, _normalize_text(alias)) for alias in aliases), default=0.0)
            exact = any(normalized == _normalize_text(alias) for alias in aliases)
            if exact or lexical >= 0.82:
                prior = math.log1p(max(0.0, evidence["frequency"])) / 10.0
                matches.append({"concept_id": concept_id, "label": evidence["label"], "lexical_score": 1.0 if exact else lexical, "prior": prior})
        matches.sort(key=lambda item: (item["lexical_score"] + item["prior"], item["concept_id"]), reverse=True)
        candidates[normalized] = matches[: processor.config.max_candidates]
    state["lexical_candidates"] = candidates
    state["unknown_tokens"] = [token for token in state["tokens"] if not candidates.get(_normalize_text(token))]
    return {"matched_token_count": sum(bool(value) for value in candidates.values()), "unknown_tokens": list(state["unknown_tokens"]), "activity": 0.7 if candidates else 0.2}


def _stage_03(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Activate candidate concepts and spread graded activation over caller-supported associations."""
    active: dict[str, float] = defaultdict(float)
    for records in state["lexical_candidates"].values():
        for record in records:
            active[record["concept_id"]] = max(active[record["concept_id"]], _clip(record["lexical_score"] * 0.8 + record["prior"] * 0.2))
    associations = processor._combined_associations(request)
    for source, strength in list(active.items()):
        for target, link_strength in associations.get(source, {}).items():
            active[target] = max(active[target], _clip(strength * float(link_strength) * 0.35))
    state["candidate_activation"] = dict(active)
    return {"activated_concepts": sorted(active), "activation_strengths": dict(active), "activity": max(active.values(), default=0.0)}


def _stage_04(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Retrieve shared and modality-specific features for every active concept candidate."""
    modality_maps: dict[str, dict[str, float]] = {name: dict(values) for name, values in state["modality_features"].items()}
    candidate_features: dict[str, dict[str, float]] = {}
    for records in state["lexical_candidates"].values():
        for record in records:
            concept_id = record["concept_id"]
            evidence = state["concept_evidence"][concept_id]
            candidate_features[concept_id] = dict(evidence["features"])
            for modality, features in evidence["modalities"].items():
                modality_maps[modality] = _merge_features(modality_maps.get(modality, {}), features, scale=0.65)
    state["candidate_features"] = candidate_features
    state["modality_features"] = modality_maps
    return {"retrieved_concept_count": len(candidate_features), "feature_count": sum(len(value) for value in candidate_features.values()), "activity": min(1.0, len(candidate_features) / 4.0 + 0.2)}


def _stage_05(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Calculate cross-modal associations and apply explicitly requested spoke-connection attenuation."""
    impairment = request.intervention.get("modality_connection_impairment", {})
    impaired: dict[str, float] = {}
    for modality, features in list(state["modality_features"].items()):
        severity = _clip(float(impairment.get(modality, 0.0))) if isinstance(impairment, Mapping) else 0.0
        impaired[modality] = 1.0 - severity
        state["modality_features"][modality] = {key: value * impaired[modality] for key, value in features.items()}
    modalities = sorted(state["modality_features"])
    pair_scores: dict[str, float] = {}
    for index, left in enumerate(modalities):
        for right in modalities[index + 1:]:
            pair_scores[f"{left}|{right}"] = _cosine(state["modality_features"][left], state["modality_features"][right])
    state["cross_modal_scores"] = pair_scores
    state["spoke_connection_gain"] = impaired
    return {"pairwise_compatibility": pair_scores, "connection_gain": impaired, "activity": float(np.mean(list(impaired.values()))) if impaired else 0.2}


def _stage_06(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Converge reliable spoke features in a bilateral ATL-like shared representation."""
    modality_maps = state["modality_features"]
    reliability = {name: float(np.mean(list(features.values()))) if features else 0.0 for name, features in modality_maps.items()}
    weights_array = _softmax(list(reliability.values())) if reliability else np.asarray([], dtype=np.float64)
    weights = {name: float(weights_array[index]) for index, name in enumerate(reliability)}
    integrated: dict[str, float] = {}
    for modality, features in modality_maps.items():
        for feature, strength in features.items():
            integrated[feature] = integrated.get(feature, 0.0) + weights.get(modality, 0.0) * float(strength)
    if not integrated:
        for features in state["candidate_features"].values():
            integrated = _merge_features(integrated, features, scale=0.5)
    intervention = request.intervention
    bilateral = _clip(float(intervention.get("atl_bilateral_degradation", 0.0)))
    side = str(intervention.get("atl_lesion_side", "")).casefold()
    unilateral = _clip(float(intervention.get("atl_lesion_severity", 0.0)))
    temporary = _clip(float(intervention.get("temporary_atl_perturbation", 0.0)))
    left_gain = 1.0 - bilateral
    right_gain = 1.0 - bilateral
    if side == "left":
        left_gain *= 1.0 - unilateral
    elif side == "right":
        right_gain *= 1.0 - unilateral
    temporary_active = bool(intervention.get("temporary_atl_active", True))
    if temporary_active:
        left_gain *= 1.0 - temporary
        right_gain *= 1.0 - temporary
    atl_capacity = _clip((left_gain + right_gain) / 2.0)
    state["semantic_representation"] = {key: _clip(value * atl_capacity) for key, value in integrated.items()}
    state["atl_state"] = {"left_gain": left_gain, "right_gain": right_gain, "capacity": atl_capacity, "bilateral_degradation": bilateral, "unilateral_side": side or None, "temporary_perturbation": temporary if temporary_active else 0.0}
    return {"semantic_representation": dict(state["semantic_representation"]), "modality_weights": weights, "atl_state": dict(state["atl_state"]), "activity": atl_capacity}


def _stage_07(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Extract category prototypes and distinctive feature residuals from related inventory concepts."""
    categories: dict[str, list[str]] = defaultdict(list)
    for concept_id, evidence in state["concept_evidence"].items():
        category = evidence.get("prototype_id") or evidence.get("lexical_category") or "unspecified"
        categories[str(category)].append(concept_id)
    prototypes: dict[str, dict[str, float]] = {}
    distinctive: dict[str, dict[str, float]] = {}
    for category, concept_ids in categories.items():
        maps = [state["concept_evidence"][concept_id]["features"] for concept_id in concept_ids]
        counts: dict[str, int] = defaultdict(int)
        sums: dict[str, float] = defaultdict(float)
        for features in maps:
            for feature, strength in features.items():
                counts[feature] += 1
                sums[feature] += strength
        prototypes[category] = {feature: sums[feature] / max(1, counts[feature]) for feature in sums}
        for concept_id in concept_ids:
            distinctive[concept_id] = {
                feature: _clip(strength - prototypes[category].get(feature, 0.0))
                for feature, strength in state["concept_evidence"][concept_id]["features"].items()
                if strength > prototypes[category].get(feature, 0.0)
            }
    state["category_prototypes"] = prototypes
    state["distinctive_features"] = distinctive
    return {"category_count": len(prototypes), "prototype_features": prototypes, "distinctive_feature_count": sum(len(value) for value in distinctive.values()), "activity": min(1.0, len(categories) / 5.0)}


def _stage_08(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Build a typed association graph and propagate activation without equating association to composition."""
    graph = processor._combined_associations(request)
    active = dict(state["candidate_activation"])
    propagated: dict[str, float] = {}
    for source, strength in active.items():
        for target, link in graph.get(source, {}).items():
            propagated[target] = max(propagated.get(target, 0.0), _clip(strength * link))
    state["association_graph"] = graph
    state["associative_activation"] = propagated
    return {"node_count": len(state["concept_evidence"]), "edge_count": sum(len(edges) for edges in graph.values()), "propagated_concepts": propagated, "activity": min(1.0, len(graph) / 8.0)}


def _stage_09(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Construct a provisional relation frame from supplied syntax or a transparent heuristic."""
    relations = [dict(item) for item in request.syntactic_relations] if request.syntactic_relations else _guess_syntax(state["tokens"])
    state["syntax_relations"] = relations
    predicates = [item["head"] for item in relations if _normalize_text(item["head"]) in _VERBS]
    state["syntax_frame"] = {"relations": relations, "predicates": predicates, "source": "caller_supplied" if request.syntactic_relations else "bounded_surface_heuristic"}
    return {"relations": relations, "source": state["syntax_frame"]["source"], "activity": 0.65 if relations else 0.2}


def _stage_10(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Compose modifier and head concepts while retaining local concept and feature representations."""
    composed = dict(state["semantic_representation"])
    token_pairs: list[dict[str, str]] = []
    tokens = state["tokens"]
    for index, token in enumerate(tokens):
        if index + 1 < len(tokens) and _normalize_text(token) not in _VERBS:
            next_token = tokens[index + 1]
            token_pairs.append({"modifier": token, "head": next_token})
    for pair in token_pairs:
        modifier_id = _find_concept_id(pair["modifier"], state["lexical_candidates"])
        head_id = _find_concept_id(pair["head"], state["lexical_candidates"])
        if modifier_id and head_id:
            modifier_features = state["candidate_features"].get(modifier_id, {})
            head_features = state["candidate_features"].get(head_id, {})
            for feature, strength in head_features.items():
                composed[feature] = max(composed.get(feature, 0.0), strength)
            for feature, strength in modifier_features.items():
                if feature in head_features:
                    composed[feature] = _clip(composed[feature] * (0.5 + 0.5 * strength))
    state["phrase_composition"] = {"features": composed, "modifier_head_pairs": token_pairs}
    return {"modifier_head_pairs": token_pairs, "composed_feature_count": len(composed), "activity": min(1.0, len(composed) / 12.0)}


def _stage_11(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Assign explicit or provisional thematic roles and construct a participant event frame."""
    roles = dict(request.event_roles)
    if not roles:
        for relation in state["syntax_relations"]:
            relation_type = relation["relation"].casefold()
            if relation_type in {"subject", "nsubj", "agent"}:
                roles["Agent"] = relation["dependent"]
                roles["Action"] = relation["head"]
            elif relation_type in {"object", "dobj", "obj", "theme", "patient"}:
                roles["Theme"] = relation["dependent"]
                roles.setdefault("Action", relation["head"])
    event = {"roles": roles, "predicate": roles.get("Action"), "source": "caller_supplied" if request.event_roles else "syntax_or_heuristic"}
    state["event_frame"] = event
    return {"event_frame": event, "role_count": len(roles), "activity": min(1.0, len(roles) / 4.0)}


def _stage_12(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Score event participants against conservative predicate selectional preferences."""
    event = state["event_frame"]
    predicate = _normalize_text(str(event.get("predicate") or ""))
    theme = event.get("roles", {}).get("Theme")
    concept_id = _find_concept_id(str(theme), state["lexical_candidates"]) if theme else None
    preferences = _SELECTION_FEATURES.get(predicate, {})
    features = state["candidate_features"].get(concept_id, {}) if concept_id else {}
    compatibility = _cosine(features, preferences) if preferences else None
    anomaly = (1.0 - compatibility) if compatibility is not None else None
    state["selectional_compatibility"] = {"predicate": predicate or None, "theme": theme, "concept_id": concept_id, "preferences": preferences, "compatibility": compatibility, "anomaly_score": anomaly, "evaluated": bool(preferences and concept_id)}
    return {"selectional_compatibility": state["selectional_compatibility"], "conflict": anomaly is not None and anomaly > 0.65, "activity": compatibility if compatibility is not None else 0.25}


def _stage_13(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Predict plausible continuations from current candidate activation and explicit graph links."""
    graph = state.get("association_graph", {})
    scores: dict[str, float] = {}
    for source, source_activation in state["candidate_activation"].items():
        for target, link in graph.get(source, {}).items():
            scores[target] = max(scores.get(target, 0.0), source_activation * link)
    selected = state.get("event_frame", {}).get("roles", {}).get("Theme")
    if selected:
        selected_id = _find_concept_id(str(selected), state["lexical_candidates"])
        if selected_id:
            for target, link in graph.get(selected_id, {}).items():
                scores[target] = max(scores.get(target, 0.0), 0.6 * link)
    identifiers = sorted(scores, key=scores.get, reverse=True)[: processor.config.max_candidates]
    probabilities = _softmax([scores[item] for item in identifiers])
    predictions = [
        {"concept_id": item, "label": state["concept_evidence"].get(item, {}).get("label", item), "score": scores[item], "probability": float(probabilities[index])}
        for index, item in enumerate(identifiers)
    ]
    state["predictions"] = predictions
    return {"predictions": predictions, "entropy": _entropy(probabilities), "activity": max(scores.values(), default=0.15)}


def _stage_14(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Condition the integrated concept state on explicit context features with a bounded feature gate."""
    conditioned = dict(state.get("semantic_representation", {}))
    context = dict(request.context_features)
    for feature, context_strength in context.items():
        if feature in conditioned:
            gate = 1.0 / (1.0 + math.exp(-float(np.clip(4.0 * context_strength - 2.0, -30.0, 30.0))))
            conditioned[feature] = _clip(conditioned[feature] * (0.5 + 0.5 * gate))
    state["context_conditioned_representation"] = conditioned
    state["context_features"] = context
    return {"conditioned_representation": conditioned, "context_feature_count": len(context), "activity": min(1.0, len(context) / 8.0 + 0.15)}


def _stage_15(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Normalize lexical, contextual, and associative evidence across competing word senses."""
    posteriors: dict[str, list[dict[str, Any]]] = {}
    for token, records in state["lexical_candidates"].items():
        scores: list[float] = []
        for record in records:
            features = state["candidate_features"].get(record["concept_id"], {})
            context_fit = _cosine(features, state["context_features"])
            association = state["associative_activation"].get(record["concept_id"], 0.0)
            scores.append(0.55 * record["lexical_score"] + 0.25 * context_fit + 0.20 * association + 0.05 * record["prior"])
        probabilities = _softmax(scores)
        posteriors[token] = [
            {**record, "score": scores[index], "probability": float(probabilities[index])}
            for index, record in enumerate(records)
        ]
    state["candidate_posteriors"] = posteriors
    ambiguity = max((_entropy([item["probability"] for item in values]) for values in posteriors.values()), default=0.0)
    state["ambiguity_entropy"] = ambiguity
    return {"candidate_posteriors": posteriors, "ambiguity_entropy": ambiguity, "activity": min(1.0, ambiguity / 2.0)}


def _stage_16(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Apply task-sensitive relevance gating and reduce competition without deleting stored knowledge."""
    impairment = _clip(float(request.intervention.get("semantic_control_impairment", 0.0)))
    gate_gain = 1.0 - impairment
    selected: dict[str, dict[str, Any]] = {}
    margins: dict[str, float] = {}
    for token, records in state["candidate_posteriors"].items():
        ranked = sorted(records, key=lambda item: item["probability"] * gate_gain, reverse=True)
        if ranked:
            top = ranked[0]
            runner_up = ranked[1]["probability"] if len(ranked) > 1 else 0.0
            margin = max(0.0, float(top["probability"]) - float(runner_up))
            margins[token] = margin
            if float(top["probability"]) >= processor.config.confidence_threshold and margin >= processor.config.margin_threshold * gate_gain:
                selected[token] = top
    state["selected_by_token"] = selected
    state["control_state"] = {"impairment": impairment, "gain": gate_gain, "selected_count": len(selected)}
    return {"selected_by_token": selected, "margins": margins, "control_state": state["control_state"], "conflict": impairment > 0.35, "activity": gate_gain}


def _stage_17(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Integrate token meanings and their relations into an angular-gyrus-like contextual structure."""
    local_maps: list[Mapping[str, float]] = []
    for candidate in _candidate_records(state):
        local_maps.append(state["candidate_features"].get(candidate["concept_id"], {}))
    integrated = dict(state.get("context_conditioned_representation", {}))
    if local_maps:
        keys = set().union(*(set(item) for item in local_maps))
        for feature in keys:
            integrated[feature] = _clip(0.6 * integrated.get(feature, 0.0) + 0.4 * float(np.mean([item.get(feature, 0.0) for item in local_maps])))
    structure = {
        "features": integrated,
        "token_nodes": [{"token": token, "selected_concept": record.get("concept_id")} for token, record in state["selected_by_token"].items()],
        "event_frame": state["event_frame"],
        "associations": state["association_graph"],
    }
    state["ag_integrated_meaning"] = structure
    return {"integrated_feature_count": len(integrated), "token_node_count": len(structure["token_nodes"]), "activity": min(1.0, len(structure["token_nodes"]) / 5.0 + 0.2)}


def _stage_18(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Construct typed posterior-temporal relations between predicate and participant concepts."""
    event = state["event_frame"]
    roles = event.get("roles", {})
    relations: list[dict[str, Any]] = []
    predicate = event.get("predicate")
    if predicate:
        for role, participant in roles.items():
            if role != "Action":
                relations.append({
                    "source": str(participant), "relation": role.casefold(),
                    "predicate": str(predicate), "target": str(predicate),
                    "confidence": 0.75 if event.get("source") == "caller_supplied" else 0.5,
                })
    state["relational_graph"] = relations
    return {"relations": relations, "relation_count": len(relations), "activity": min(1.0, len(relations) / 4.0)}


def _stage_19(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Compare actual lexical evidence with predictive candidates and calculate semantic surprisal."""
    last_token = _normalize_text(state["tokens"][-1]) if state["tokens"] else ""
    observed = _select_top_candidate(state, last_token) if last_token else None
    prediction_by_id = {item["concept_id"]: item["probability"] for item in state["predictions"]}
    observed_probability = prediction_by_id.get(observed["concept_id"], 0.0) if observed else None
    surprisal = -math.log(max(_EPS, observed_probability)) if observed_probability is not None else None
    prediction_error = (1.0 - observed_probability) if observed_probability is not None else None
    state["prediction_error"] = {"observed_concept": observed.get("concept_id") if observed else None, "predicted_probability": observed_probability, "surprisal_nats": surprisal, "prediction_error": prediction_error}
    return {"prediction_error": state["prediction_error"], "activity": _clip((surprisal or 0.0) / 5.0)}


def _stage_20(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Generate a negative-going N400-like integration difficulty proxy, never a measured EEG value."""
    error = state["prediction_error"]
    difficulty = float(error["surprisal_nats"] or 0.0)
    anomaly = state["selectional_compatibility"].get("anomaly_score")
    if anomaly is not None:
        difficulty = 0.7 * difficulty + 0.3 * float(anomaly)
    amplitude = -0.8 * difficulty
    proxy = {"amplitude_proxy": amplitude, "integration_difficulty": difficulty, "scientific_status": "illustrative_computational_proxy_not_eeg"}
    state["n400_proxy"] = proxy
    return {**proxy, "activity": _clip(difficulty / 5.0)}


def _stage_21(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Re-score alternative interpretations when unresolved ambiguity or semantic conflict is detected."""
    revisions: list[dict[str, Any]] = []
    if state["ambiguity_entropy"] > 0.2 or (state["selectional_compatibility"].get("anomaly_score") or 0.0) > 0.5:
        for token, records in state["candidate_posteriors"].items():
            if len(records) > 1:
                ranked = sorted(records, key=lambda item: item["score"], reverse=True)
                old = state["selected_by_token"].get(token, ranked[0])
                best = ranked[0]
                if best["concept_id"] != old["concept_id"]:
                    revisions.append({"token": token, "previous": old["concept_id"], "revised": best["concept_id"], "reason": "alternative score exceeded prior selection"})
    state["reanalysis"] = {"revisions": revisions, "performed": bool(revisions)}
    return {"reanalysis": state["reanalysis"], "activity": min(1.0, len(revisions) / 3.0 + 0.15)}


def _stage_22(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Retrieve supplied or previously stored episodes by feature similarity and preserve episode provenance."""
    query = state.get("ag_integrated_meaning", {}).get("features", {})
    episodes = [*processor._episodes, *[dict(item) for item in request.episodic_memory]]
    ranked: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes):
        features = episode.get("features", episode.get("semantic_features", {}))
        similarity = _cosine(query, features) if isinstance(features, Mapping) else 0.0
        ranked.append({"episode_id": episode.get("episode_id", f"supplied-{index}"), "similarity": similarity, "source_reference": episode.get("source_reference", "caller_supplied"), "confidence": _clip(float(episode.get("confidence", 1.0)))})
    ranked.sort(key=lambda item: item["similarity"], reverse=True)
    state["retrieved_episodes"] = ranked[:5]
    return {"retrieved_episodes": state["retrieved_episodes"], "activity": ranked[0]["similarity"] if ranked else 0.1}


def _stage_23(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Match only caller-supplied schemas and emit confidence-tagged inferences without inventing facts."""
    features = state.get("ag_integrated_meaning", {}).get("features", {})
    matched: list[dict[str, Any]] = []
    for schema_id, schema in request.schemas.items():
        expected = schema.get("features", schema.get("expected_features", {}))
        similarity = _cosine(features, expected) if isinstance(expected, Mapping) else 0.0
        if similarity > 0.0:
            matched.append({"schema_id": schema_id, "compatibility": similarity, "inferences": list(schema.get("inferences", ())), "source_reference": schema.get("source_reference", "caller_supplied")})
    state["schema_inferences"] = matched
    return {"matched_schemas": matched, "inference_count": sum(len(item["inferences"]) for item in matched), "activity": max((item["compatibility"] for item in matched), default=0.1)}


def _stage_24(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Integrate current propositions with supplied discourse history and resolve simple recent pronouns."""
    history = [dict(item) for item in request.discourse_history]
    entities: list[str] = []
    for item in history:
        entities.extend(str(value) for value in item.get("entities", ()))
    current_entities = [
        state["concept_evidence"][record["concept_id"]]["label"]
        for record in state["selected_by_token"].values()
        if record["concept_id"] in state["concept_evidence"]
    ]
    pronoun_links: dict[str, str] = {}
    antecedent_pool = current_entities or entities
    for token in state["tokens"]:
        if _normalize_text(token) in {"he", "she", "it", "they", "him", "her", "them", "one"} and antecedent_pool:
            pronoun_links[token] = antecedent_pool[-1]
    discourse = {
        "prior_turn_count": len(history),
        "entities": list(dict.fromkeys([*entities, *current_entities])),
        "pronoun_links": pronoun_links,
        "current_event": state["event_frame"],
        "relations": state["relational_graph"],
    }
    state["discourse_state"] = discourse
    return {"discourse_state": discourse, "activity": min(1.0, len(discourse["entities"]) / 5.0 + 0.15)}


def _stage_25(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Allocate an executive-control budget from explicit task demand, ambiguity, and semantic anomaly."""
    entropy = float(state.get("ambiguity_entropy", 0.0))
    anomaly = float(state["selectional_compatibility"].get("anomaly_score") or 0.0)
    task_gain = 0.15 if request.task.casefold() in {"disambiguate", "explain", "compare", "infer"} else 0.0
    demand = _clip(0.5 * min(1.0, entropy / 1.5) + 0.35 * anomaly + task_gain)
    resource = _clip(1.0 - 0.55 * demand)
    state["executive_allocation"] = {"demand": demand, "resource_level": resource, "mode": "controlled" if demand > 0.35 else "mostly_automatic"}
    return {"executive_allocation": state["executive_allocation"], "activity": demand}


def _stage_26(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Prepare confidence-weighted semantic updates and apply them only when learning is authorized."""
    target = next(iter(state["selected_by_token"].values()), None)
    target_id = target["concept_id"] if target else None
    proposed = dict(request.learning_feedback)
    for feature_map in state["modality_features"].values():
        for feature, strength in feature_map.items():
            proposed.setdefault(feature, _clip(strength * 0.25))
    applied = False
    if request.allow_learning and target_id and proposed:
        existing = processor._learned_features.setdefault(target_id, {})
        for feature, strength in proposed.items():
            previous = existing.get(feature, 0.0)
            existing[feature] = _clip(previous + processor.config.learning_rate * float(strength))
        processor._episodes.append({
            "episode_id": f"learned-{request.sequence_id}",
            "features": dict(proposed),
            "source_reference": "explicit_learning_request",
            "confidence": 0.8,
        })
        applied = True
        processor.model_generation += 1
    state["learning_update"] = {"target_concept": target_id, "proposed_features": proposed, "applied": applied, "model_generation": processor.model_generation}
    return {"learning_update": state["learning_update"], "activity": min(1.0, len(proposed) / 10.0)}


def _stage_27(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Estimate a model-only maturation profile from explicitly supplied experience counts."""
    experience_count = int(request.developmental_experience_count) + len(processor._episodes)
    maturity = 1.0 - math.exp(-experience_count / 50.0)
    profile = {"experience_count": experience_count, "maturity_index": _clip(maturity), "hierarchical_refinement": _clip(0.2 + 0.8 * maturity), "compositional_refinement": _clip(0.15 + 0.85 * maturity), "status": "model_parameter_not_personal_developmental_assessment"}
    state["developmental_profile"] = profile
    return {"developmental_profile": profile, "activity": maturity}


def _stage_28(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Apply the caller's explicit aging simulation factor to modeled latency while leaving memory intact."""
    factor = _clip(request.aging_factor)
    baseline_ms = 100.0 + 35.0 * len(state["tokens"]) + 15.0 * len(state["relational_graph"])
    profile = {"baseline_latency_ms_proxy": baseline_ms, "adapted_latency_ms_proxy": baseline_ms * (1.0 + factor), "slowdown_factor": 1.0 + factor, "semantic_memory_preserved": True}
    state["aging_profile"] = profile
    return {"aging_profile": profile, "activity": _clip(factor + 0.15)}


def _stage_29(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Select available language-specific surface forms while keeping concept identities language-independent."""
    mapping = request.multilingual_lexicon.get(request.language, {})
    selected: dict[str, str] = {}
    for token in state["tokens"]:
        concept_id = _find_concept_id(token, state["lexical_candidates"])
        if concept_id:
            label = state["concept_evidence"][concept_id]["label"]
            lexical_form = mapping.get(concept_id, mapping.get(label, mapping.get(token)))
            if lexical_form:
                selected[token] = str(lexical_form)
    state["language_selection"] = {"language": request.language, "selected_forms": selected, "mapping_supplied": bool(mapping)}
    return {"language_selection": state["language_selection"], "activity": 0.7 if mapping else 0.2}


def _stage_30(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Measure supplied educational modality coverage and create a transparent reinforcement proposal."""
    education = normalize_modalities(request.education_inputs)
    channel_count = len(education)
    coverage = channel_count / max(1, len(("visual", "auditory", "spatial_motor", "linguistic")))
    features = _merge_features(*education.values()) if education else {}
    proposal = {"modalities": sorted(education), "coverage": _clip(coverage), "reinforcement_features": features, "applied": False}
    if request.allow_learning and state["learning_update"].get("target_concept") and features:
        target_id = state["learning_update"]["target_concept"]
        target = processor._learned_features.setdefault(target_id, {})
        for feature, strength in features.items():
            target[feature] = _clip(target.get(feature, 0.0) + processor.config.learning_rate * strength)
        proposal["applied"] = True
    state["education_reinforcement"] = proposal
    return {"education_reinforcement": proposal, "activity": coverage}


def _stage_31(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Report modeled effects of bilateral ATL degradation without claiming a clinical diagnosis."""
    severity = _clip(float(request.intervention.get("atl_bilateral_degradation", 0.0)))
    profile = {"severity": severity, "detailed_semantic_retention_proxy": 1.0 - severity, "cross_modal_convergence_retention_proxy": 1.0 - severity, "category_level_retention_proxy": 1.0 - 0.65 * severity, "clinical_reference": "semantic_dementia", "status": "model_intervention_not_patient_prediction"}
    state["bilateral_atl_profile"] = profile
    return {"bilateral_atl_profile": profile, "activity": severity}


def _stage_32(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Model unilateral ATL attenuation and disclose the explicit contralateral compensation assumption."""
    side = str(request.intervention.get("atl_lesion_side", "none")).casefold()
    severity = _clip(float(request.intervention.get("atl_lesion_severity", 0.0)))
    compensation = _clip(float(request.intervention.get("contralateral_compensation", 0.15)))
    residual = 1.0 - severity * (1.0 - compensation) if side in {"left", "right"} else 1.0
    profile = {"lesioned_side": side if side in {"left", "right"} else None, "severity": severity, "compensation_assumption": compensation, "residual_hub_capacity_proxy": _clip(residual), "clinical_reference": "unilateral_temporal_lobe_resection_or_focal_damage"}
    state["unilateral_atl_profile"] = profile
    return {"unilateral_atl_profile": profile, "activity": 1.0 - residual}


def _stage_33(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Model temporary ATL perturbation as reversible attenuation rather than permanent memory deletion."""
    perturbation = _clip(float(request.intervention.get("temporary_atl_perturbation", 0.0)))
    active = bool(request.intervention.get("temporary_atl_active", True))
    profile = {"perturbation": perturbation if active else 0.0, "active": active, "temporary_capacity_proxy": 1.0 - (perturbation if active else 0.0), "recovery_assumed": True, "clinical_reference": "temporary_atl_perturbation_such_as_tms"}
    state["temporary_atl_profile"] = profile
    return {"temporary_atl_profile": profile, "activity": perturbation if active else 0.0}


def _stage_34(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Model semantic-control impairment while explicitly preserving underlying stored representations."""
    severity = _clip(float(request.intervention.get("semantic_control_impairment", 0.0)))
    profile = {"severity": severity, "control_gain_proxy": 1.0 - severity, "stored_semantic_knowledge_preserved": True, "flexible_retrieval_capacity_proxy": 1.0 - severity, "clinical_reference": "semantic_aphasia_or_semantic_control_impairment"}
    state["control_impairment_profile"] = profile
    return {"control_impairment_profile": profile, "conflict": severity > 0.3, "activity": severity}


def _stage_35(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Classify the modeled effects of modality-specific spoke disconnection versus central ATL degradation."""
    impairments = request.intervention.get("modality_connection_impairment", {})
    impairments = dict(impairments) if isinstance(impairments, Mapping) else {}
    profile = {
        "connection_retention": {str(key): 1.0 - _clip(float(value)) for key, value in impairments.items()},
        "central_atl_capacity_proxy": state["atl_state"]["capacity"],
        "interpretation": "spoke_disconnection" if impairments else "no_spoke_disconnection_specified",
        "clinical_reference": "distributed_or_modality_to_concept_integration_impairment",
    }
    state["multimodal_impairment_profile"] = profile
    return {"multimodal_impairment_profile": profile, "activity": min(1.0, len(impairments) / 4.0)}


def _stage_36(state: State, request: DistributedSemanticRequest, processor: "DistributedSemanticProcessor") -> dict[str, Any]:
    """Evaluate available semantic outputs and score only caller-provided independent targets."""
    correct = 0
    total = 0
    for token, expected in request.evaluation_targets.items():
        total += 1
        chosen = _select_top_candidate(state, token)
        expected_normalized = _normalize_text(expected)
        chosen_label = _normalize_text(chosen["label"]) if chosen else ""
        chosen_id = str(chosen["concept_id"]) if chosen else ""
        correct += int(expected_normalized in {chosen_label, chosen_id.casefold()})
    modality_scores = list(state["cross_modal_scores"].values())
    consistency = float(np.mean(modality_scores)) if modality_scores else None
    entropy = float(state.get("ambiguity_entropy", 0.0))
    evaluation = {
        "target_accuracy": correct / total if total else None,
        "target_count": total,
        "cross_modal_consistency_proxy": consistency,
        "candidate_entropy_nats": entropy,
        "relation_count": len(state.get("relational_graph", ())),
        "selected_concept_count": len(state.get("selected_by_token", {})),
        "neural_measurements_supplied": False,
        "biological_validation_status": "not_tested_without_independent_human_data",
    }
    state["evaluation"] = evaluation
    return {"evaluation": evaluation, "activity": evaluation["target_accuracy"] if total else 0.4}


class DistributedSemanticProcessor:
    """Execute the 36-stage semantic architecture over an explicit concept inventory."""

    def __init__(
        self,
        concepts: Sequence[SemanticConcept] = (),
        config: DistributedSemanticConfig | None = None,
        associations: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        """Validate the concept inventory and initialize explicitly separate learning state."""
        self.config = config or DistributedSemanticConfig()
        self.concepts = tuple(concepts)
        self._concept_by_id = {concept.concept_id: concept for concept in self.concepts}
        if len(self._concept_by_id) != len(self.concepts):
            raise ValueError("concept_id values must be unique")
        self.associations = self._validate_associations(associations or {})
        self._learned_features: dict[str, dict[str, float]] = {}
        self._episodes: list[dict[str, Any]] = []
        self.model_generation = 0

    def _validate_associations(
        self, associations: Mapping[str, Mapping[str, float]]
    ) -> dict[str, dict[str, float]]:
        """Validate supplied semantic edges and reject unknown concept identifiers."""
        validated: dict[str, dict[str, float]] = {}
        for source, targets in associations.items():
            if source not in self._concept_by_id:
                raise ValueError(f"Unknown association source: {source}")
            validated[source] = {}
            for target, raw_strength in targets.items():
                strength = float(raw_strength)
                if target not in self._concept_by_id or not math.isfinite(strength) or not 0 <= strength <= 1:
                    raise ValueError(f"Invalid association target or strength: {source} -> {target}")
                validated[source][target] = strength
        return validated

    def _combined_associations(
        self, request: DistributedSemanticRequest
    ) -> dict[str, dict[str, float]]:
        """Merge processor-level and request-level semantic links without mutating either source."""
        combined = {source: dict(targets) for source, targets in self.associations.items()}
        for source, targets in request.associations.items():
            combined.setdefault(source, {})
            for target, strength in targets.items():
                if source in self._concept_by_id and target in self._concept_by_id:
                    combined[source][target] = max(combined[source].get(target, 0.0), float(strength))
        return combined

    def process(self, request: DistributedSemanticRequest) -> DistributedSemanticResult:
        """Run all 36 functional stages, advance optional LIF populations, and assemble diagnostics."""
        state: State = {
            "sequence_id": request.sequence_id,
            "request_text": request.text,
            "stage_outputs": {},
            "operation_trace": [],
            "stage_trace": [],
            "biological_trace": [],
        }
        simulator = SemanticLIFSimulator(self.config) if self.config.simulate_biology else None
        for stage in STAGE_REGISTRY:
            handler = _STAGE_HANDLERS[stage.stage_number]
            stage_output = handler(state, request, self)
            state["stage_outputs"][stage.stage_id] = stage_output
            state["stage_trace"].append({
                "stage_id": stage.stage_id,
                "stage_number": stage.stage_number,
                "name": stage.name,
                "status": "completed",
                "output_summary": stage_output,
                "biological_reference": stage.biological_reference,
            })
            for operation in stage.operations:
                state["operation_trace"].append({
                    "operation_id": operation.operation_id,
                    "stage_id": stage.stage_id,
                    "stage_name": stage.name,
                    "operation": operation.name,
                    "functional_responsibility": operation.functional_responsibility,
                    "mathematical_operation": operation.mathematical_operation,
                    "biological_reference": stage.biological_reference,
                    "functional_status": "executed_as_stage_component",
                    "biological_status": "separate_lif_reference" if simulator else "disabled",
                })
            if simulator:
                drives = biological_drives(stage.stage_number, stage_output)
                state["biological_trace"].append(simulator.step(stage.stage_id, drives))
        selected = {
            token: {"concept_id": item["concept_id"], "label": item["label"], "probability": item["probability"]}
            for token, item in state.get("selected_by_token", {}).items()
        }
        output = {
            "selected_concepts": selected,
            "integrated_meaning": state.get("ag_integrated_meaning", {}),
            "event_frame": state.get("event_frame", {}),
            "relational_graph": state.get("relational_graph", []),
            "predictions": state.get("predictions", []),
            "n400_proxy": state.get("n400_proxy", {}),
            "discourse_state": state.get("discourse_state", {}),
            "reanalysis": state.get("reanalysis", {}),
            "evaluation": state.get("evaluation", {}),
        }
        diagnostics = {
            "valid": len(state["stage_trace"]) == 36 and len(state["operation_trace"]) == len(ALL_OPERATIONS),
            "stage_count": len(state["stage_trace"]),
            "operation_count": len(state["operation_trace"]),
            "biological_stage_count": len(state["biological_trace"]),
            "biology_simulated": simulator is not None,
            "model_generation": self.model_generation,
            "unknown_tokens": state.get("unknown_tokens", []),
            "learning_applied": bool(state.get("learning_update", {}).get("applied", False)),
            "scientific_boundary": "functional and synthetic biological-reference tracks are computational hypotheses, not validated reproductions of human neural processing",
        }
        return DistributedSemanticResult(
            sequence_id=request.sequence_id,
            output=output,
            stage_outputs=state["stage_outputs"],
            stage_trace=state["stage_trace"],
            operation_trace=state["operation_trace"],
            biological_trace=state["biological_trace"],
            diagnostics=diagnostics,
        )


_STAGE_HANDLERS.update({
    1: _stage_01, 2: _stage_02, 3: _stage_03, 4: _stage_04, 5: _stage_05,
    6: _stage_06, 7: _stage_07, 8: _stage_08, 9: _stage_09, 10: _stage_10,
    11: _stage_11, 12: _stage_12, 13: _stage_13, 14: _stage_14, 15: _stage_15,
    16: _stage_16, 17: _stage_17, 18: _stage_18, 19: _stage_19, 20: _stage_20,
    21: _stage_21, 22: _stage_22, 23: _stage_23, 24: _stage_24, 25: _stage_25,
    26: _stage_26, 27: _stage_27, 28: _stage_28, 29: _stage_29, 30: _stage_30,
    31: _stage_31, 32: _stage_32, 33: _stage_33, 34: _stage_34, 35: _stage_35,
    36: _stage_36,
})


def request_from_text(
    sequence_id: str,
    text: str,
    *,
    context_features: Mapping[str, float] | None = None,
    modality_inputs: Mapping[str, Mapping[str, float]] | None = None,
    language: str = "en",
    task: str = "comprehension",
) -> DistributedSemanticRequest:
    """Create a validated request from plain text and optional caller-supplied feature evidence."""
    return DistributedSemanticRequest(
        sequence_id=sequence_id,
        text=text,
        context_features=context_features or {},
        modality_inputs=modality_inputs or {},
        language=language,
        task=task,
    )


def operation_trace_as_dicts(result: DistributedSemanticResult) -> list[dict[str, Any]]:
    """Return the stable ordered mathematical-operation trace for external inspection."""
    return [dict(row) for row in result.operation_trace]
