"""Deterministic functional operations for connecting new information to prior knowledge."""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import KnowledgeNode, KnowledgeRelation, KnowledgeSchema, LearningUnit, RelationInput

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "i", "in", "into", "is", "it", "its",
    "of", "on", "or", "our", "she", "that", "the", "their", "them", "there", "these",
    "they", "this", "those", "to", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "you", "your", "can", "may", "might", "should", "must",
}
_NEGATIONS = {"not", "no", "never", "cannot", "without", "neither", "n't"}
_RELATION_HINTS = {
    "causes": ("cause", "because", "leads to", "results in"),
    "part_of": ("part of", "component of", "within"),
    "used_for": ("used for", "functions as", "enables"),
    "located_in": ("located in", "found in", "inside"),
    "is_a": ("is a", "is an", "belongs to", "is a type of"),
    "contrasts_with": ("unlike", "in contrast to", "differs from"),
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Bound a finite scalar and reject invalid ranges."""
    number = float(value)
    if not math.isfinite(number) or low > high:
        raise ValueError("value must be finite and low must not exceed high")
    return min(high, max(low, number))


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize Unicode word forms without claiming full linguistic analysis."""
    return tuple(token.casefold() for token in re.findall(r"[^\W_]+(?:['’-][^\W_]+)?", text, flags=re.UNICODE))


def softmax(values: Sequence[float], temperature: float = 1.0) -> tuple[float, ...]:
    """Convert finite candidate scores into a numerically stable probability distribution."""
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if not values:
        return ()
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("scores must be finite")
    array = array / temperature
    array -= float(np.max(array))
    weights = np.exp(np.clip(array, -700.0, 0.0))
    total = float(weights.sum())
    return tuple(float(x / total) for x in weights)


def shannon_entropy(probabilities: Sequence[float]) -> float:
    """Measure uncertainty in a normalized non-negative candidate distribution."""
    if not probabilities:
        return 0.0
    values = np.asarray(probabilities, dtype=float)
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(values.sum())
    if total <= 0:
        return 0.0
    values /= total
    positive = values[values > 0]
    return float(-np.sum(positive * np.log(positive)))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity, using zero for a zero-norm vector."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape:
        raise ValueError("vectors must have identical shapes")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator <= 1e-12 else clamp(float(np.dot(a, b) / denominator), -1.0, 1.0)


def vectorize_tokens(tokens: Iterable[str], dimensions: int = 64) -> np.ndarray:
    """Create deterministic signed-hash vectors for transparent semantic feature proxies."""
    if dimensions < 8:
        raise ValueError("dimensions must be at least 8")
    vector = np.zeros(dimensions, dtype=float)
    for token in tokens:
        digest = hashlib.blake2b(str(token).casefold().encode("utf-8"), digest_size=8).digest()
        integer = int.from_bytes(digest, "big")
        index = integer % dimensions
        sign = 1.0 if ((integer >> 8) & 1) else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def extract_new_representation(unit: LearningUnit) -> dict[str, Any]:
    """Extract concepts and attributes, preferring explicit structure over keyword heuristics."""
    tokens = tokenize(unit.text)
    content_tokens = tuple(token for token in tokens if token not in _STOP_WORDS and len(token) > 1)
    counts = Counter(content_tokens)
    concepts = list(unit.entities) or [token for token, _ in counts.most_common(12)]
    features = dict(unit.features) or {token: min(1.0, count / max(counts.values(), default=1))
                                       for token, count in counts.items()}
    if not concepts:
        concepts = [unit.text[:80]]
    relations = [
        {"subject": item.subject.casefold(), "predicate": item.predicate.casefold(),
         "object": item.object.casefold(), "confidence": item.confidence,
         "evidential_status": item.evidential_status}
        for item in unit.relations
    ]
    if not relations:
        lowered = unit.text.casefold()
        for predicate, markers in _RELATION_HINTS.items():
            marker = next((m for m in markers if m in lowered), None)
            if marker:
                left, right = lowered.split(marker, 1)
                left_terms = [t for t in tokenize(left) if t not in _STOP_WORDS]
                right_terms = [t for t in tokenize(right) if t not in _STOP_WORDS]
                if left_terms and right_terms:
                    relations.append({"subject": left_terms[-1], "predicate": predicate,
                                      "object": right_terms[0], "confidence": 0.45,
                                      "evidential_status": "inferred_from_text"})
    return {
        "unit_id": unit.unit_id, "source_text": unit.text, "modality": unit.modality,
        "concepts": concepts, "features": features, "relations": relations,
        "negated": any(token in _NEGATIONS for token in tokens),
        "reliability": unit.reliability, "provenance": unit.provenance,
        "context": dict(unit.context), "content_tokens": list(content_tokens),
    }


def generate_retrieval_cues(
    representation: Mapping[str, Any], context: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Generate semantic, relational, contextual, and contrastive cues with distinct weights."""
    cues: dict[str, dict[str, Any]] = {}
    for concept in representation["concepts"]:
        key = str(concept).casefold()
        cues[key] = {"cue": key, "cue_type": "concept", "weight": 1.0}
    for feature, weight in representation["features"].items():
        key = str(feature).casefold()
        current = cues.get(key, {"cue": key, "cue_type": "feature", "weight": 0.0})
        current["weight"] = max(float(current["weight"]), float(weight) * 0.9)
        cues[key] = current
    for relation in representation["relations"]:
        for key, kind in ((relation["predicate"], "relational"),
                          (relation["subject"], "relation_endpoint"),
                          (relation["object"], "relation_endpoint")):
            cues.setdefault(key, {"cue": key, "cue_type": kind, "weight": 0.85})
    for key, value in context.items():
        if isinstance(value, (str, int, float)) and str(value).strip():
            cue = str(value).casefold()
            if len(cue) <= 80:
                cues.setdefault(cue, {"cue": cue, "cue_type": "context", "weight": 0.55})
    if representation["negated"]:
        cues["contrast:negation"] = {"cue": "contrast:negation", "cue_type": "contrast", "weight": 0.7}
    return sorted(cues.values(), key=lambda item: (-item["weight"], item["cue"]))


def _set_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    """Compute Jaccard overlap with zero similarity for either empty set."""
    a, b = set(str(x).casefold() for x in left), set(str(x).casefold() for x in right)
    return len(a & b) / len(a | b) if a and b else 0.0


def _node_tokens(node: KnowledgeNode) -> set[str]:
    """Combine normalized label terms with explicitly represented semantic features."""
    return {x for x in tokenize(node.label) if x not in _STOP_WORDS} | set(node.features)


def retrieve_candidates(
    cues: Sequence[Mapping[str, Any]], nodes: Mapping[str, KnowledgeNode],
    edges: Sequence[KnowledgeRelation], limit: int = 12, max_hops: int = 2,
    temperature: float = 0.75,
) -> list[dict[str, Any]]:
    """Retrieve candidates through cue overlap and bounded spreading over prior graph edges."""
    cue_weights = {str(item["cue"]): float(item["weight"]) for item in cues}
    candidate_scores: dict[str, float] = {}
    candidate_sources: dict[str, set[str]] = {}
    for node_id, node in nodes.items():
        terms = _node_tokens(node)
        matched = [(cue, weight) for cue, weight in cue_weights.items()
                   if cue in terms or cue in node.label.casefold()]
        if matched:
            score = max(weight for _, weight in matched) * (0.55 + 0.45 * len(matched) / max(1, len(cue_weights)))
            candidate_scores[node_id] = clamp(score)
            candidate_sources[node_id] = {cue for cue, _ in matched}
    adjacency: dict[str, list[tuple[str, float]]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source_id, []).append((edge.target_id, edge.strength * edge.confidence))
        adjacency.setdefault(edge.target_id, []).append((edge.source_id, edge.strength * edge.confidence))
    frontier = dict(candidate_scores)
    for _ in range(max_hops):
        next_frontier: dict[str, float] = {}
        for node_id, activation in frontier.items():
            for neighbor, weight in adjacency.get(node_id, []):
                propagated = activation * weight * 0.65
                next_frontier[neighbor] = max(next_frontier.get(neighbor, 0.0), propagated)
                candidate_sources.setdefault(neighbor, set()).update(candidate_sources.get(node_id, set()))
        for node_id, activation in next_frontier.items():
            candidate_scores[node_id] = max(candidate_scores.get(node_id, 0.0), activation)
        frontier = next_frontier
    rows = [{"node_id": node_id, "label": nodes[node_id].label,
             "memory_type": nodes[node_id].memory_type,
             "activation": clamp(score),
             "retrieval_cues": sorted(candidate_sources.get(node_id, set())),
             "reliability": nodes[node_id].reliability,
             "provenance": nodes[node_id].provenance}
            for node_id, score in candidate_scores.items() if node_id in nodes]
    rows = sorted(rows, key=lambda row: (-row["activation"], row["node_id"]))[:limit]
    probabilities = softmax([row["activation"] for row in rows], temperature=temperature)
    for row, probability in zip(rows, probabilities):
        row["retrieval_probability"] = probability
    return rows

def rank_candidates(
    candidates: Sequence[Mapping[str, Any]], representation: Mapping[str, Any],
    context: Mapping[str, Any], nodes: Mapping[str, KnowledgeNode],
) -> list[dict[str, Any]]:
    """Rank memory candidates by relevance, reliability, and context rather than familiarity alone."""
    output = []
    for candidate in candidates:
        node = nodes[str(candidate["node_id"])]
        relevance = _set_similarity(representation["features"], _node_tokens(node))
        relevance = max(relevance, _set_similarity(representation["concepts"], [node.label]))
        if context and node.context:
            shared = sum(1 for key, value in context.items() if key in node.context and node.context[key] == value)
            context_fit = shared / max(1, len(set(context) | set(node.context)))
        else:
            context_fit = 0.5
        score = (0.35 * float(candidate["activation"]) + 0.35 * relevance
                 + 0.20 * node.reliability + 0.10 * context_fit)
        output.append({**dict(candidate), "relevance": clamp(relevance),
                       "context_fit": clamp(context_fit), "rank_score": clamp(score)})
    return sorted(output, key=lambda row: (-row["rank_score"], row["node_id"]))


def compare_knowledge(
    representation: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
    nodes: Mapping[str, KnowledgeNode], edges: Sequence[KnowledgeRelation],
) -> list[dict[str, Any]]:
    """Compare surface, feature, functional, and explicit relational structure for each candidate."""
    output = []
    new_features = set(representation["features"])
    new_relations = {(x["subject"], x["predicate"], x["object"]) for x in representation["relations"]}
    for candidate in candidates:
        node = nodes[str(candidate["node_id"])]
        prior_features = _node_tokens(node)
        feature_overlap = _set_similarity(new_features, prior_features)
        related = [edge for edge in edges if edge.source_id == node.node_id or edge.target_id == node.node_id]
        prior_relations = {(node.label.casefold(), e.predicate.casefold(),
                            nodes[e.target_id].label.casefold())
                           for e in related if e.source_id == node.node_id and e.target_id in nodes}
        prior_relations |= {(nodes[e.source_id].label.casefold(), e.predicate.casefold(),
                             node.label.casefold())
                            for e in related if e.target_id == node.node_id and e.source_id in nodes}
        relation_overlap = 0.0
        if new_relations and prior_relations:
            normalized_new = {(s.casefold(), p.casefold(), o.casefold()) for s, p, o in new_relations}
            relation_overlap = _set_similarity(normalized_new, prior_relations)
        complementary = sorted(new_features - prior_features)
        contradictions = []
        for edge in related:
            if edge.evidential_status == "contradicted":
                contradictions.append({"predicate": edge.predicate, "target_id": edge.target_id,
                                       "reason": "prior edge is explicitly marked contradicted"})
        analogy = clamp(0.60 * feature_overlap + 0.40 * relation_overlap)
        output.append({"candidate_id": node.node_id, "candidate_label": node.label,
                       "surface_or_feature_overlap": feature_overlap,
                       "relational_overlap": relation_overlap,
                       "complementary_features": complementary,
                       "contradictions": contradictions,
                       "overall_comparison": analogy})
    return output


def align_analogies(
    representation: Mapping[str, Any], comparisons: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build bounded analogy maps and explicitly mark unmapped or speculative relations."""
    rows = []
    for comparison in comparisons:
        candidate = next(x for x in candidates if x["node_id"] == comparison["candidate_id"])
        new_relations = representation["relations"]
        preserved = [dict(item) for item in new_relations if item["confidence"] >= 0.5]
        mapped = [{"new_concept": concept, "prior_candidate": candidate["label"],
                   "mapping_basis": "candidate retrieval and feature/relation comparison",
                   "mapping_confidence": comparison["overall_comparison"]}
                  for concept in representation["concepts"][:3]]
        rows.append({"candidate_id": comparison["candidate_id"], "candidate_label": candidate["label"],
                     "role_correspondences": mapped, "preserved_relations": preserved,
                     "unmapped_relations": [dict(item) for item in new_relations if item["confidence"] < 0.5],
                     "structure_preservation_score": comparison["relational_overlap"],
                     "mapping_cost": 1.0 - comparison["overall_comparison"],
                     "analogy_quality": clamp(0.65 * comparison["overall_comparison"]
                                              + 0.35 * comparison["relational_overlap"]),
                     "limitations": ["analogy is a hypothesis scaffold, not identity or proof",
                                     "unmapped features must not be transferred automatically"]})
    return rows


def evaluate_schema_fit(
    representation: Mapping[str, Any], schemas: Mapping[str, KnowledgeSchema], *,
    conflict_threshold: float = 0.45, novelty_threshold: float = 0.72,
    integration_threshold: float = 0.62,
) -> dict[str, Any]:
    """Estimate schema compatibility and explicit relational conflict without forcing assimilation."""
    for name, value in (("conflict_threshold", conflict_threshold),
                        ("novelty_threshold", novelty_threshold),
                        ("integration_threshold", integration_threshold)):
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    observed_roles = {str(x).casefold() for x in representation["concepts"]}
    observed_features = set(representation["features"])
    observed_relations = sorted({
        (x["subject"], x["predicate"], x["object"]) for x in representation["relations"]
    })
    candidates = []
    for schema in schemas.values():
        expected = set(schema.expectations) | set(schema.roles)
        role_fit = _set_similarity(observed_roles | observed_features, expected)
        expected_relations = set(schema.relation_patterns)
        relation_fit = _set_similarity(observed_relations, expected_relations)
        relation_fit = relation_fit if observed_relations and expected_relations else (0.5 if not observed_relations else 0.0)
        fit = clamp(0.60 * role_fit + 0.40 * relation_fit)
        candidates.append({"schema_id": schema.schema_id, "schema_name": schema.name,
                           "fit": fit, "role_fit": role_fit, "relation_fit": relation_fit,
                           "schema_confidence": schema.confidence})
    candidates.sort(key=lambda x: (-x["fit"], x["schema_id"]))
    best = candidates[0] if candidates else None
    feature_novelty = 1.0 - max((x["role_fit"] for x in candidates), default=0.0)
    relation_novelty = 1.0 - max((x["relation_fit"] for x in candidates), default=0.0)
    novelty = clamp(0.65 * feature_novelty + 0.35 * relation_novelty)

    # A relation with the same subject and predicate but a different object can violate
    # a schema expectation. This is a mismatch signal, not automatic proof of falsity.
    relational_conflicts = []
    best_schema = schemas.get(best["schema_id"]) if best else None
    for observed in observed_relations:
        expected_relations = sorted(best_schema.relation_patterns) if best_schema else ()
        for expected in expected_relations:
            if observed[0] == expected[0] and observed[1] == expected[1] and observed[2] != expected[2]:
                relational_conflicts.append({"observed": observed, "expected": expected,
                                             "reason": "same subject and predicate, different object"})
    contradiction = clamp(0.55 * len(relational_conflicts))
    if representation["negated"]:
        contradiction = max(contradiction, 0.15)
        if best and best["fit"] >= integration_threshold:
            contradiction = max(contradiction, conflict_threshold + 0.05)
    if best and best["fit"] < 0.2:
        contradiction = max(contradiction, min(conflict_threshold, 0.25))

    if contradiction >= conflict_threshold:
        fit_state = "schema_conflict"
    elif best and best["fit"] >= integration_threshold and novelty < 0.5 * novelty_threshold:
        fit_state = "schema_consistent"
    elif best and best["fit"] >= 0.75 * integration_threshold:
        fit_state = "schema_complementary"
    elif novelty >= novelty_threshold:
        fit_state = "high_novelty"
    else:
        fit_state = "ambiguous_or_partial_fit"
    probabilities = softmax([item["fit"] for item in candidates], temperature=0.5)
    entropy = shannon_entropy(probabilities)
    surprisal = -math.log(max(1e-6, best["fit"] if best else 0.05))
    return {"fit_state": fit_state, "best_schema": best, "schema_candidates": candidates,
            "novelty": novelty, "contradiction": contradiction,
            "relational_conflicts": relational_conflicts,
            "surprisal": surprisal, "candidate_entropy": entropy,
            "probabilities": list(probabilities),
            "interpretation": "schema-fit values are computational estimates, not measured neural variables"}


def choose_integration_strategy(
    fit: Mapping[str, Any], reliability: float, minimum_reliability: float,
    revision_reliability_threshold: float = 0.78,
) -> dict[str, Any]:
    """Choose integrate, complement, revise-proposal, separate-episode, or defer."""
    for name, value in (("reliability", reliability),
                        ("minimum_reliability", minimum_reliability),
                        ("revision_reliability_threshold", revision_reliability_threshold)):
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    state = fit["fit_state"]
    if state == "schema_consistent" and reliability >= minimum_reliability:
        strategy = "integrate"
    elif state == "schema_complementary" and reliability >= minimum_reliability:
        strategy = "complement_schema"
    elif state == "schema_conflict" and reliability >= revision_reliability_threshold:
        strategy = "revise_candidate_schema"
    elif state in {"schema_conflict", "high_novelty"}:
        strategy = "store_separately"
    else:
        strategy = "defer"
    return {"strategy": strategy, "fit_state": state,
            "decision_reason": "fit, novelty, contradiction, and source reliability jointly determine the strategy",
            "automatic_schema_overwrite": False}


def build_integrated_representation(
    representation: Mapping[str, Any], candidate_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]], strategy: str,
) -> dict[str, Any]:
    """Create a provenance-preserving representation linked only to ranked candidates."""
    matched = []
    comparison_by_id = {x["candidate_id"]: x for x in comparisons}
    for row in candidate_rows[:5]:
        detail = comparison_by_id.get(row["node_id"], {})
        matched.append({"node_id": row["node_id"], "label": row["label"],
                        "connection_strength": row["rank_score"],
                        "relation_overlap": detail.get("relational_overlap", 0.0),
                        "link_status": "inferred_association"})
    return {"unit_id": representation["unit_id"], "text": representation["source_text"],
            "concepts": list(representation["concepts"]),
            "features": dict(representation["features"]),
            "relations": [dict(x) for x in representation["relations"]],
            "connected_prior_knowledge": matched, "integration_strategy": strategy,
            "source_provenance": representation["provenance"],
            "observed_inference_distinction_preserved": True}


def build_retrieval_routes(
    representation: Mapping[str, Any], cues: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]], analogy_maps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Construct bidirectional, source-tagged retrieval routes for future recall attempts."""
    routes = []
    for cue in cues:
        for candidate in candidate_rows[:5]:
            routes.append({"cue": cue["cue"], "cue_type": cue["cue_type"],
                           "target_unit_id": representation["unit_id"],
                           "prior_node_id": candidate["node_id"],
                           "direction": "prior_to_new_and_new_to_prior",
                           "route_strength": clamp(float(cue["weight"]) * candidate["rank_score"]),
                           "provenance": "modeled_association_not_observed_recall"})
    return sorted(routes, key=lambda item: (-item["route_strength"], item["cue"]))


def test_retrieval_routes(
    representation: Mapping[str, Any], routes: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run a deterministic simulated route check while clearly separating it from human recall."""
    cues = {str(x["cue"]) for x in routes}
    candidate_ids = {str(x["node_id"]) for x in candidates}
    recovered = bool(routes and cues and candidate_ids)
    expected = max(1, len(representation["concepts"]) + len(representation["relations"]))
    score = clamp(min(len(cues), expected) / expected) if recovered else 0.0
    return {"simulated_route_recovery": recovered, "route_coverage": score,
            "cue_count": len(cues), "connected_prior_candidate_count": len(candidate_ids),
            "human_retrieval_test_performed": False,
            "note": "This checks software route construction, not a learner's actual memory retrieval."}


def stable_id(prefix: str, value: str) -> str:
    """Generate a reproducible content-derived identifier for an episode or representation."""
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
