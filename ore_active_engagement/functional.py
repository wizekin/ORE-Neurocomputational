"""Functional operations for the Point 01 active-engagement algorithm.

The text processing is deliberately deterministic and transparent. It is a research
baseline, not a language model and not a substitute for a learner's own explanation.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import ActiveEngagementConfig, ActiveEngagementRequest, LearningSegment


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "i", "in", "into", "is",
    "it", "its", "of", "on", "or", "our", "she", "that", "the", "their",
    "them", "there", "these", "they", "this", "those", "to", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "you", "your",
}
_RELATION_MARKERS = {
    "cause": ("because", "causes", "caused", "due to", "therefore", "so that"),
    "consequence": ("therefore", "as a result", "consequently", "leads to"),
    "condition": ("if", "unless", "when", "provided that"),
    "contrast": ("however", "although", "whereas", "but", "in contrast"),
    "sequence": ("first", "then", "next", "finally", "before", "after"),
    "example": ("for example", "such as", "for instance"),
    "purpose": ("in order to", "so that", "to enable"),
}
_NEGATIONS = {"not", "no", "never", "cannot", "can't", "without", "neither"}


def tokenize(text: str) -> tuple[str, ...]:
    """Split text into normalized word-like tokens while preserving Unicode letters."""
    return tuple(token.casefold() for token in re.findall(r"[^\W_]+(?:['’-][^\W_]+)?", text, flags=re.UNICODE))


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Bound a finite model value to the requested closed interval."""
    if not math.isfinite(float(value)):
        raise ValueError("value must be finite")
    if minimum > maximum:
        raise ValueError("minimum must not exceed maximum")
    return max(minimum, min(maximum, float(value)))


def sigmoid(value: float) -> float:
    """Compute a numerically stable logistic activation for a scalar input."""
    value = float(value)
    if value >= 0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


def softmax(values: Sequence[float], temperature: float = 1.0) -> tuple[float, ...]:
    """Normalize competing activation scores into a stable probability distribution."""
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if not values:
        return ()
    scaled = np.asarray(values, dtype=float) / float(temperature)
    scaled -= float(np.max(scaled))
    weights = np.exp(np.clip(scaled, -700.0, 0.0))
    total = float(weights.sum())
    if total <= 0 or not math.isfinite(total):
        return tuple(1.0 / len(values) for _ in values)
    return tuple(float(item / total) for item in weights)


def shannon_entropy(probabilities: Sequence[float]) -> float:
    """Estimate uncertainty in a normalized candidate distribution using Shannon entropy."""
    if not probabilities:
        return 0.0
    values = np.asarray(probabilities, dtype=float)
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(values.sum())
    if total <= 0:
        return 0.0
    values = values / total
    positive = values[values > 0]
    return float(-np.sum(positive * np.log(positive)))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity, defining similarity as zero for an empty vector."""
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        raise ValueError("vectors must have the same shape")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return clamp(float(np.dot(a, b) / denominator), -1.0, 1.0)


def vectorize_texts(texts: Sequence[str], dimensions: int = 64) -> np.ndarray:
    """Create deterministic signed-hash feature vectors for reproducible semantic proxies."""
    if dimensions < 8:
        raise ValueError("dimensions must be at least 8")
    result = np.zeros(dimensions, dtype=float)
    for text in texts:
        for token in tokenize(text):
            if token in _STOP_WORDS:
                continue
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            result[index] += sign
    norm = float(np.linalg.norm(result))
    if norm > 0:
        result /= norm
    return result


def segment_material(segments: Sequence[LearningSegment]) -> list[dict[str, Any]]:
    """Split source segments into sentence-like units while retaining original source IDs."""
    output: list[dict[str, Any]] = []
    for segment in segments:
        pieces = [
            item.strip() for item in re.split(r"(?<=[.!?])\s+|[\r\n]+", segment.text)
            if item.strip()
        ]
        if not pieces:
            pieces = [segment.text.strip()]
        for part_index, text in enumerate(pieces, start=1):
            output.append({
                "proposition_source_id": f"{segment.segment_id}:{part_index}",
                "segment_id": segment.segment_id,
                "text": text,
                "modality": segment.modality,
                "reliability": segment.reliability,
                "boundary_confidence": 0.85 if re.search(r"[.!?]$", text) else 0.65,
            })
    return output


def extract_propositions(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Extract auditable sentence-level provisional propositions and explicit relation cues."""
    propositions: list[dict[str, Any]] = []
    for index, unit in enumerate(units, start=1):
        text = str(unit["text"])
        tokens = tokenize(text)
        content_tokens = tuple(token for token in tokens if token not in _STOP_WORDS)
        relation_types = [
            relation for relation, markers in _RELATION_MARKERS.items()
            if any(marker in text.casefold() for marker in markers)
        ]
        propositions.append({
            "proposition_id": f"prop-{index:03d}",
            "source_id": str(unit["proposition_source_id"]),
            "segment_id": str(unit["segment_id"]),
            "text": text,
            "tokens": list(tokens),
            "key_terms": list(dict.fromkeys(content_tokens)),
            "relation_types": relation_types,
            "boundary_confidence": float(unit["boundary_confidence"]),
            "source_reliability": float(unit["reliability"]),
            "explicit_source_statement": True,
        })
    return propositions


def _overlap(left: Sequence[str], right: Sequence[str]) -> float:
    """Estimate symmetric lexical overlap between two token sequences."""
    a = set(left)
    b = set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def select_target(
    propositions: Sequence[Mapping[str, Any]],
    goal: str,
    config: ActiveEngagementConfig,
) -> dict[str, Any]:
    """Rank propositions by goal relevance, local term centrality, and uncertainty."""
    if not propositions:
        raise ValueError("at least one proposition is required")
    goal_tokens = [token for token in tokenize(goal) if token not in _STOP_WORDS]
    term_counts = Counter(
        token for item in propositions for token in item.get("key_terms", [])
    )
    scores: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    for item in propositions:
        terms = list(item.get("key_terms", []))
        relevance = _overlap(terms, goal_tokens)
        centrality = (
            sum(term_counts[token] for token in terms) / max(1, len(terms))
        )
        max_count = max(term_counts.values(), default=1)
        centrality = centrality / max_count
        text = str(item.get("text", ""))
        uncertainty = clamp(
            0.25 + 0.25 * ("?" in text)
            + 0.25 * (float(item.get("boundary_confidence", 0.8)) < 0.7)
            + 0.25 * (len(terms) < 3)
        )
        score = (
            config.goal_weight * relevance
            + config.centrality_weight * centrality
            + config.uncertainty_weight * uncertainty
        )
        scores.append(score)
        diagnostics.append({
            "proposition_id": item["proposition_id"],
            "goal_relevance": relevance,
            "local_centrality": centrality,
            "uncertainty": uncertainty,
            "priority": score,
        })
    winner = int(np.argmax(np.asarray(scores, dtype=float)))
    support = [
        dict(item) for index, item in enumerate(propositions)
        if index != winner and _overlap(
            list(item.get("key_terms", [])),
            list(propositions[winner].get("key_terms", [])),
        ) > 0.05
    ]
    return {
        "target": dict(propositions[winner]),
        "supporting_propositions": support,
        "ranking": sorted(diagnostics, key=lambda item: item["priority"], reverse=True),
        "priority_distribution": list(softmax(scores)),
    }


def generate_questions(
    target: Mapping[str, Any],
    propositions: Sequence[Mapping[str, Any]],
    goal: str,
    max_questions: int,
    config: ActiveEngagementConfig,
) -> list[dict[str, Any]]:
    """Generate and rank explanatory questions from missing causes, conditions, and evidence."""
    terms = list(target.get("key_terms", []))
    anchor = " ".join(terms[:5]) or str(target.get("text", "the central idea"))
    relation_types = set(target.get("relation_types", []))
    templates = [
        ("what", f"What is the main claim about {anchor}?"),
        ("how", f"How does the process involving {anchor} work?"),
        ("why", f"Why does the stated relationship involving {anchor} occur?"),
        ("condition", f"What changes if one of the conditions involving {anchor} changes?"),
        ("evidence", f"What evidence in the material supports the claim about {anchor}?"),
        ("prediction", f"What would be expected to happen next given {anchor}?"),
    ]
    source_tokens = [token for item in propositions for token in item.get("tokens", [])]
    questions: list[dict[str, Any]] = []
    for kind, text in templates:
        q_tokens = tokenize(text)
        lexical_fit = _overlap(q_tokens, terms + tokenize(goal))
        relation_bonus = 0.15 if (
            (kind == "why" and "cause" in relation_types)
            or (kind == "condition" and "condition" in relation_types)
            or (kind == "prediction" and "sequence" in relation_types)
        ) else 0.0
        information_gain = clamp(0.25 + 0.65 * lexical_fit + relation_bonus)
        importance = clamp(0.45 + 0.35 * lexical_fit + 0.2 * float(bool(terms)))
        cost = 0.5 + config.question_cost_weight * len(q_tokens) / 12.0
        utility = information_gain * importance / max(cost, 1e-6)
        questions.append({
            "question_id": f"q-{len(questions) + 1:02d}",
            "kind": kind,
            "text": text,
            "information_gain_estimate": information_gain,
            "importance": importance,
            "estimated_cost": cost,
            "utility": utility,
            "prediction_prompt": f"Before checking, predict an answer to: {text}",
            "status": "pending",
        })
    questions.sort(key=lambda item: item["utility"], reverse=True)
    selected = questions[:max_questions]
    for index, question in enumerate(selected, start=1):
        question["question_id"] = f"q-{index:02d}"
        question["status"] = "prioritized"
    return selected


def allocate_attention(
    question: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    config: ActiveEngagementConfig,
) -> dict[str, Any]:
    """Select evidence by question similarity and return normalized attentional weights."""
    q_tokens = tokenize(str(question.get("text", "")))
    scores = [
        _overlap(q_tokens, tokenize(str(unit["text"])))
        + 0.08 * float(unit.get("reliability", 1.0))
        for unit in units
    ]
    weights = softmax(scores, temperature=config.attention_temperature)
    ranked = sorted(
        zip(units, weights, scores),
        key=lambda item: item[2],
        reverse=True,
    )
    selected = [
        {
            "source_id": str(unit["proposition_source_id"]),
            "segment_id": str(unit["segment_id"]),
            "text": str(unit["text"]),
            "relevance_score": float(score),
            "attention_weight": float(weight),
            "reliability": float(unit["reliability"]),
        }
        for unit, weight, score in ranked[:max(1, min(5, len(ranked)))]
    ]
    return {
        "attention_weights": {
            str(unit["proposition_source_id"]): float(weight)
            for unit, weight in zip(units, weights)
        },
        "selected_evidence": selected,
        "suppressed_evidence": [
            str(unit["proposition_source_id"])
            for unit, _, _ in ranked[len(selected):]
        ],
        "weights_sum": float(sum(weights)),
        "question": str(question.get("text", "")),
    }


def generate_candidate_answers(
    question: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Create competing evidence-grounded answer candidates without inventing missing facts."""
    selected = list(evidence.get("selected_evidence", []))
    candidates = []
    for index, item in enumerate(selected[:4], start=1):
        candidates.append({
            "candidate_id": f"answer-{index:02d}",
            "text": str(item["text"]),
            "source_ids": [str(item["source_id"])],
            "evidence_strength": clamp(
                float(item["attention_weight"]) * float(item["reliability"]) * 2.0
            ),
            "support_status": "direct_source_evidence",
            "inference_status": "not_inferred",
            "question_id": str(question.get("question_id", "")),
        })
    if selected:
        combined = " ".join(str(item["text"]) for item in selected[:3])
        candidates.insert(0, {
            "candidate_id": "answer-combined",
            "text": combined,
            "source_ids": [str(item["source_id"]) for item in selected[:3]],
            "evidence_strength": clamp(sum(
                float(item["attention_weight"]) * float(item["reliability"])
                for item in selected[:3]
            )),
            "support_status": "combined_direct_source_evidence",
            "inference_status": "not_inferred",
            "question_id": str(question.get("question_id", "")),
        })
    return candidates


def reconstruct_explanation(
    candidate_answers: Sequence[Mapping[str, Any]],
    learner_explanation: str | None,
) -> dict[str, Any]:
    """Prefer the learner's own text; otherwise return a labeled extractive scaffold."""
    if learner_explanation:
        return {
            "text": learner_explanation.strip(),
            "origin": "learner_generated",
            "is_learner_generated": True,
            "source_ids": sorted({
                str(source_id)
                for candidate in candidate_answers
                for source_id in candidate.get("source_ids", [])
            }),
            "limitations": [],
        }
    combined = next(
        (str(item["text"]) for item in candidate_answers
         if item.get("candidate_id") == "answer-combined"),
        " ".join(str(item.get("text", "")) for item in candidate_answers[:2]),
    )
    return {
        "text": combined,
        "origin": "extractive_scaffold",
        "is_learner_generated": False,
        "source_ids": sorted({
            str(source_id)
            for candidate in candidate_answers
            for source_id in candidate.get("source_ids", [])
        }),
        "limitations": [
            "This is a source-grounded scaffold, not evidence that the learner can explain the idea independently."
        ],
    }


def elaborate_explanation(
    explanation: Mapping[str, Any],
    propositions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a proposition graph from source-supported relations and label uncertain links."""
    text = str(explanation.get("text", ""))
    all_source_text = " ".join(str(item.get("text", "")) for item in propositions)
    relation_edges: list[dict[str, Any]] = []
    for index, item in enumerate(propositions):
        markers = [
            relation for relation, phrases in _RELATION_MARKERS.items()
            if any(phrase in str(item.get("text", "")).casefold() for phrase in phrases)
        ]
        if markers and index + 1 < len(propositions):
            relation_edges.append({
                "from": str(item["proposition_id"]),
                "to": str(propositions[index + 1]["proposition_id"]),
                "relation_types": markers,
                "evidence": str(item["text"]),
                "support_status": "explicit_in_source",
            })
    supported_terms = set(tokenize(all_source_text))
    explanation_terms = set(tokenize(text))
    unsupported_terms = sorted(
        token for token in explanation_terms - supported_terms
        if token not in _STOP_WORDS and len(token) > 3
    )
    nodes = [
        {
            "proposition_id": str(item["proposition_id"]),
            "label": str(item["text"]),
            "key_terms": list(item.get("key_terms", [])),
        }
        for item in propositions
    ]
    return {
        "graph": {"nodes": nodes, "edges": relation_edges},
        "supported_relations": relation_edges,
        "unsupported_or_unmatched_terms": unsupported_terms,
        "relation_coverage": (
            min(1.0, len(relation_edges) / max(1, len(propositions) - 1))
            if len(propositions) > 1 else 1.0
        ),
        "source_relation_types": sorted({
            relation for item in propositions for relation in item.get("relation_types", [])
        }),
    }


def _contradictions(text: str) -> list[dict[str, str]]:
    """Detect simple same-term polarity conflicts as a conservative contradiction proxy."""
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]
    records: list[dict[str, str]] = []
    for left_index, left in enumerate(sentences):
        left_tokens = set(tokenize(left))
        left_negated = bool(left_tokens & _NEGATIONS)
        for right in sentences[left_index + 1:]:
            right_tokens = set(tokenize(right))
            right_negated = bool(right_tokens & _NEGATIONS)
            shared = sorted((left_tokens & right_tokens) - _STOP_WORDS - _NEGATIONS)
            if shared and left_negated != right_negated:
                records.append({
                    "shared_terms": ", ".join(shared[:5]),
                    "left": left,
                    "right": right,
                    "status": "possible_polarity_conflict",
                })
    return records


def evaluate_explanation(
    explanation: Mapping[str, Any],
    propositions: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    config: ActiveEngagementConfig,
    required_concepts: Sequence[str] = (),
) -> dict[str, Any]:
    """Score explanation fidelity, term coverage, relation coherence, evidence, and uncertainty."""
    text = str(explanation.get("text", ""))
    source = " ".join(str(item.get("text", "")) for item in propositions)
    source_vector = vectorize_texts([source], config.embedding_dimensions)
    explanation_vector = vectorize_texts([text], config.embedding_dimensions)
    fidelity = clamp(cosine_similarity(source_vector, explanation_vector), 0.0, 1.0)
    target_terms = list(target.get("key_terms", []))
    required = list(required_concepts) or target_terms[: min(6, len(target_terms))]
    explanation_tokens = set(tokenize(text))
    missing_terms = [term for term in required if term.casefold() not in explanation_tokens]
    coverage = (
        (len(required) - len(missing_terms)) / len(required) if required else 1.0
    )
    selected_ids = {
        str(item["proposition_id"])
        for item in propositions
        if _overlap(tokenize(str(item["text"])), tokenize(text)) >= 0.35
    }
    proposition_coverage = (
        len(selected_ids) / max(1, len(propositions))
    )
    contradictions = _contradictions(text)
    relation_count = sum(len(item.get("relation_types", [])) for item in propositions)
    explanation_relation_count = sum(
        1 for marker_group in _RELATION_MARKERS.values()
        if any(marker in text.casefold() for marker in marker_group)
    )
    coherence = clamp(
        0.7 * proposition_coverage
        + 0.3 * min(1.0, explanation_relation_count / max(1, relation_count))
        if relation_count else 0.7 + 0.3 * proposition_coverage
    )
    source_terms = set(tokenize(source))
    explained_content_terms = {
        token for token in explanation_tokens
        if token not in _STOP_WORDS and len(token) > 2
    }
    evidence_support = (
        len(explained_content_terms & source_terms) / max(1, len(explained_content_terms))
    )
    contradiction_penalty = min(1.0, len(contradictions) / 3.0)
    uncertainty = clamp(
        0.65 * (1.0 - coverage) + 0.35 * (1.0 - proposition_coverage)
    )
    weighted = (
        config.fidelity_weight * fidelity
        + config.coverage_weight * coverage
        + config.coherence_weight * coherence
        + config.evidence_weight * evidence_support
        - config.contradiction_penalty * contradiction_penalty
        - config.uncertainty_penalty * uncertainty
    )
    total_weight = (
        config.fidelity_weight + config.coverage_weight
        + config.coherence_weight + config.evidence_weight
    )
    score = clamp(weighted / max(total_weight, 1e-9))
    gaps = [f"missing_concept:{term}" for term in missing_terms]
    if contradictions:
        gaps.extend("possible_contradiction:" + item["shared_terms"] for item in contradictions)
    return {
        "quality_score": score,
        "fidelity": fidelity,
        "concept_coverage": coverage,
        "proposition_coverage": proposition_coverage,
        "relational_coherence": coherence,
        "evidence_support": clamp(evidence_support),
        "contradictions": contradictions,
        "missing_concepts": missing_terms,
        "unresolved_gaps": gaps,
        "uncertainty": uncertainty,
        "accepted_provisionally": score >= config.minimum_quality and not contradictions,
        "measurement_note": (
            "Lexical and structural proxies are not a validated measure of human comprehension."
        ),
        "explanation_origin": str(explanation.get("origin", "unknown")),
    }


def repair_explanation(
    explanation: Mapping[str, Any],
    quality: Mapping[str, Any],
    propositions: Sequence[Mapping[str, Any]],
    maximum_passes: int = 1,
) -> dict[str, Any]:
    """Generate targeted, source-linked repair prompts without silently fabricating an answer."""
    gaps = list(quality.get("unresolved_gaps", []))
    repair_questions: list[dict[str, Any]] = []
    suggestions: list[dict[str, Any]] = []
    for gap in gaps:
        kind, _, value = str(gap).partition(":")
        if kind == "missing_concept":
            term = value.strip()
            candidates = [
                item for item in propositions
                if term in set(item.get("key_terms", []))
            ]
            repair_questions.append({
                "gap": gap,
                "question": f"Explain what '{term}' means in this material and how it supports the main point.",
            })
            if candidates:
                suggestions.append({
                    "gap": gap,
                    "source_ids": [str(item["source_id"]) for item in candidates],
                    "source_text": " ".join(str(item["text"]) for item in candidates),
                    "support_status": "direct_source_evidence",
                })
        elif kind == "possible_contradiction":
            repair_questions.append({
                "gap": gap,
                "question": "Check whether these statements refer to the same conditions and reconcile their polarity.",
            })
        else:
            repair_questions.append({
                "gap": gap,
                "question": "Return to the source evidence for this unresolved issue and revise only what it supports.",
            })
    return {
        "repair_passes_requested": min(maximum_passes, 1 if gaps else 0),
        "repair_questions": repair_questions,
        "source_supported_suggestions": suggestions,
        "unresolved_gaps": gaps,
        "explanation_preserved": str(explanation.get("text", "")),
        "no_unsupported_facts_added": True,
    }


def prepare_teaching_output(
    explanation: Mapping[str, Any],
    propositions: Sequence[Mapping[str, Any]],
    listener_level: str,
    enabled: bool,
) -> dict[str, Any]:
    """Prepare a level-adjusted teaching scaffold and predict listener questions."""
    if not enabled:
        return {
            "enabled": False,
            "script": "",
            "anticipated_questions": [],
            "listener_level": listener_level,
        }
    source_order = sorted(
        propositions,
        key=lambda item: (len(item.get("key_terms", [])), str(item["proposition_id"])),
    )
    key_terms = list(dict.fromkeys(
        term for item in propositions for term in item.get("key_terms", [])
    ))[:8]
    script = str(explanation.get("text", ""))
    if listener_level == "novice":
        introduction = "Start with the central idea, define necessary terms, then explain the supporting details."
    elif listener_level == "intermediate":
        introduction = "State the central idea, connect the main relations, and clarify important conditions."
    else:
        introduction = "State the claim and its strongest supporting relations, then identify boundary conditions."
    anticipated = [
        {
            "question": f"What does '{term}' mean here?",
            "answerable_from_material": any(
                term in set(item.get("key_terms", [])) for item in propositions
            ),
        }
        for term in key_terms[:4]
    ]
    return {
        "enabled": True,
        "listener_level": listener_level,
        "introduction": introduction,
        "script": script,
        "ordered_proposition_ids": [str(item["proposition_id"]) for item in source_order],
        "anticipated_questions": anticipated,
        "listener_gap_count": sum(
            1 for item in anticipated if not item["answerable_from_material"]
        ),
        "delivery_modes": ["speech", "writing", "drawing", "imagined_rehearsal"],
        "note": "This is a teaching outline; actual communication requires a human or language-generation component.",
    }


def construct_imagery(
    propositions: Sequence[Mapping[str, Any]],
    elaboration: Mapping[str, Any],
    explanation: Mapping[str, Any],
    enabled: bool,
    dimensions: int = 64,
) -> dict[str, Any]:
    """Create an optional symbolic scene/process graph when imagery is explicitly enabled."""
    if not enabled:
        return {
            "enabled": False,
            "representation_type": "none",
            "nodes": [],
            "edges": [],
            "alignment_score": None,
        }
    nodes = [
        {
            "node_id": str(item["proposition_id"]),
            "label": str(item["text"]),
            "sequence": index,
        }
        for index, item in enumerate(propositions, start=1)
    ]
    edges = list(elaboration.get("supported_relations", []))
    source_vector = vectorize_texts(
        [" ".join(str(item["text"]) for item in propositions)], dimensions
    )
    verbal_vector = vectorize_texts([str(explanation.get("text", ""))], dimensions)
    return {
        "enabled": True,
        "representation_type": "symbolic_process_or_relation_graph",
        "nodes": nodes,
        "edges": edges,
        "alignment_score": clamp(cosine_similarity(source_vector, verbal_vector), 0.0, 1.0),
        "rendered_image_available": False,
        "note": "The output is structured scene data, not a generated visual image.",
    }


def bind_episode(
    request: ActiveEngagementRequest,
    propositions: Sequence[Mapping[str, Any]],
    questions: Sequence[Mapping[str, Any]],
    explanation: Mapping[str, Any],
    imagery: Mapping[str, Any],
    dimensions: int = 64,
) -> dict[str, Any]:
    """Bind source, goal, learner actions, questions, context, and optional imagery."""
    source_text = " ".join(str(item["text"]) for item in propositions)
    question_text = " ".join(str(item["text"]) for item in questions)
    context_text = " ".join(f"{key}:{value}" for key, value in request.context.items())
    fields = {
        "source": source_text,
        "goal": request.goal,
        "questions": question_text,
        "explanation": str(explanation.get("text", "")),
        "context": context_text,
        "imagery": " ".join(str(item.get("label", "")) for item in imagery.get("nodes", [])),
    }
    vector = vectorize_texts(list(fields.values()), dimensions)
    retrieval_cues = list(dict.fromkeys(
        tokenize(request.goal)[:3]
        + tokenize(question_text)[:5]
        + [
            str(item.get("key_terms", [""])[0])
            for item in propositions if item.get("key_terms")
        ][:5]
    ))
    return {
        "episode_id": str(request.metadata.get("episode_id", f"{request.learner_id}-episode-{request.metadata.get('sequence', 1)}")),
        "learner_id": request.learner_id,
        "content": source_text,
        "goal": request.goal,
        "questions": [dict(item) for item in questions],
        "explanation": dict(explanation),
        "context": dict(request.context),
        "modalities": sorted({str(item.get("modality", "text")) for item in propositions}),
        "imagery_bound": bool(imagery.get("enabled")),
        "retrieval_cues": retrieval_cues,
        "embedding": [float(value) for value in vector.tolist()],
        "source_proposition_ids": [str(item["proposition_id"]) for item in propositions],
        "provenance": "active_engagement_episode",
        "binding_is_distributed_model": True,
    }


def monitor_comprehension(
    quality: Mapping[str, Any],
    explanation: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Estimate calibrated comprehension only when independent correctness labels are supplied."""
    quality_score = clamp(float(quality.get("quality_score", 0.0)))
    unresolved = list(quality.get("unresolved_gaps", []))
    confidence = clamp(quality_score * (1.0 - 0.25 * min(1.0, len(unresolved) / 4.0)))
    labels = metadata.get("proposition_accuracy")
    brier: float | None = None
    if isinstance(labels, Mapping) and labels:
        predictions = metadata.get("proposition_confidence", {})
        if isinstance(predictions, Mapping):
            pairs = [
                (clamp(float(predictions[key])), clamp(float(value)))
                for key, value in labels.items() if key in predictions
            ]
            if pairs:
                brier = sum((prediction - actual) ** 2 for prediction, actual in pairs) / len(pairs)
    return {
        "comprehension_estimate": quality_score,
        "confidence_estimate": confidence,
        "unresolved_gap_count": len(unresolved),
        "unresolved_gaps": unresolved,
        "explanation_origin": str(explanation.get("origin", "unknown")),
        "independent_calibration_available": brier is not None,
        "brier_score": brier,
        "calibration_note": (
            "No independent correctness labels were supplied; confidence is not empirically calibrated."
            if brier is None else "Brier score computed from externally supplied proposition labels."
        ),
    }


def decide_closure(
    quality: Mapping[str, Any],
    comprehension: Mapping[str, Any],
    minimum_quality: float,
) -> dict[str, Any]:
    """Close a learning cycle only when the quality criterion and critical-gap check pass."""
    gaps = list(quality.get("unresolved_gaps", []))
    score = float(comprehension.get("comprehension_estimate", 0.0))
    critical_gaps = [
        item for item in gaps
        if str(item).startswith(("possible_contradiction:", "missing_concept:"))
    ]
    complete = score >= minimum_quality and not critical_gaps
    return {
        "episode_status": "provisionally_complete" if complete else "incomplete",
        "stop_condition_met": complete,
        "quality_threshold": minimum_quality,
        "quality_score": score,
        "critical_gaps": critical_gaps,
        "next_action": (
            "close_current_unit_and_pass_episode_to_memory_system"
            if complete else "return_to_targeted_question_and_gap_repair"
        ),
        "no_spaced_schedule_created": True,
        "no_formal_retrieval_test_run": True,
    }
