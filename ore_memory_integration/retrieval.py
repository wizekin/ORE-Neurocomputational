"""Recall-only adapters and transparent weighted memory ranking."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from .contracts import MemoryBatch, MemoryEvidence, MemoryIntegrationConfig, MemoryQuery


def _text(value: Any) -> str:
    """Flatten nested record fields into stable text for inspectable token matching."""
    if isinstance(value, Mapping):
        return " ".join(f"{key} {_text(item)}" for key, item in value.items())
    if isinstance(value, (tuple, list, set)):
        return " ".join(_text(item) for item in value)
    return "" if value is None else str(value)


def _tokens(value: Any) -> set[str]:
    """Tokenize case-insensitive cues without external NLP dependencies or semantic claims."""
    return {
        token.casefold().strip(".,!?;:()[]{}\"'")
        for token in _text(value).split()
        if token.strip(".,!?;:()[]{}\"'")
    }


def _similarity(left: Any, right: Any) -> float:
    """Calculate Jaccard overlap; zero means no shared observed token cues."""
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def _unit(value: Any, default: float = 0.5) -> float:
    """Bound a finite evidence feature, substituting a documented neutral fallback."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, result)) if math.isfinite(result) else default


def _weighted(features: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Compute a normalized non-negative weighted mean over configured retrieval features."""
    denominator = sum(float(value) for value in weights.values())
    if denominator <= 0.0:
        return 0.0
    return max(0.0, min(1.0, sum(
        float(weights.get(key, 0.0)) * _unit(value)
        for key, value in features.items()
    ) / denominator))


def _event_roles(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract only stored role bindings from a Point 7 event-core record."""
    event = record.get("event_core", {})
    if not isinstance(event, Mapping):
        return {}
    roles = event.get("participants_and_roles", {})
    return roles if isinstance(roles, Mapping) else {}


def _temporal_relevance(query: MemoryQuery, record_time: Any) -> float:
    """Estimate temporal cue compatibility, using a neutral score when time is unavailable."""
    if query.current_timestamp is None or not isinstance(record_time, (int, float)):
        return 0.5
    delta = abs(float(query.current_timestamp) - float(record_time))
    scale = float(query.temporal_cues.get("decay_seconds", 86400.0))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("temporal_cues.decay_seconds must be positive and finite")
    return math.exp(-delta / scale)


class EpisodicRetrievalAdapter:
    """Read previously stored Point 7 episodes without invoking its encode-first processor."""

    def __init__(self, memory_store: Any, config: MemoryIntegrationConfig | None = None) -> None:
        """Bind the existing Point7MemoryStore and configurable episodic ranking weights."""
        if not callable(getattr(memory_store, "episodes", None)):
            raise TypeError("memory_store must expose episodes()")
        self.store = memory_store
        self.config = config or MemoryIntegrationConfig()

    def retrieve(self, query: MemoryQuery) -> tuple[MemoryEvidence, ...]:
        """Rank prior episodes; temporal strength alone cannot retrieve a cue-unrelated event."""
        results = []
        lexical_query = query.lexical_cues or tuple(sorted(_tokens(query.text)))
        for raw in self.store.episodes():
            if str(raw.get("sequence_id", "")) == query.sequence_id:
                continue
            event_core = raw.get("event_core", {})
            content = event_core if isinstance(event_core, Mapping) else raw
            context = raw.get("context", {})
            roles = _event_roles(raw)
            relations = raw.get("relations", ())
            if not isinstance(context, Mapping):
                context = {}
            lexical = _similarity(lexical_query, content)
            relation_score = _similarity(query.event_cues, {"roles": roles, "relations": relations}) if query.event_cues else 0.0
            context_score = _similarity(query.context_cues, context) if query.context_cues else _similarity(query.text, context)
            goal_score = _similarity(query.goal, raw.get("goal", context.get("goal", "")))
            if max(lexical, relation_score, context_score, goal_score) <= 0.0:
                continue
            temporal = _temporal_relevance(query, raw.get("time", context.get("time")))
            strength = _unit(raw.get("trace_strength", 0.25))
            score = _weighted({
                "lexical": lexical, "relations": relation_score, "context": context_score,
                "goal": goal_score, "temporal": temporal, "strength": strength,
            }, self.config.episodic_weights)
            if score < self.config.minimum_relevance:
                continue
            trace_id = str(raw.get("trace_id") or raw.get("memory_id") or "")
            if not trace_id:
                continue
            confidence_values = raw.get("confidence_by_component", ())
            confidence = min((_unit(item) for item in confidence_values), default=_unit(raw.get("confidence", 0.5)))
            results.append(MemoryEvidence(
                memory_id=trace_id,
                memory_type="episodic",
                content=content,
                entities=roles,
                relations=relations if isinstance(relations, Sequence) else (),
                context=context,
                provenance={
                    "source": "point7_episodic_traces",
                    "sequence_id": raw.get("sequence_id"),
                    "source_provenance": raw.get("source_provenance", ()),
                    "encoding_status": raw.get("encoding_status", "stored_episode"),
                },
                source_reference=str(raw.get("source_reference") or raw.get("speaker_or_source") or "point7.episodic_traces"),
                verification_status="stored_episode",
                valid_time=raw.get("time", context.get("time")),
                independence_group=str(raw.get("independence_group") or f"episode:{trace_id}"),
                parent_memory_ids=tuple(raw.get("parent_memory_ids", ())),
                sequence_id=str(raw.get("sequence_id", "")),
                relevance_score=score,
                confidence=confidence,
            ))
        return tuple(results)


class SemanticRetrievalAdapter:
    """Read Point 7 semantic claims while preserving accepted and pending verification states."""

    def __init__(self, memory_store: Any, config: MemoryIntegrationConfig | None = None) -> None:
        """Bind the existing Point7MemoryStore and semantic retrieval weights."""
        if not callable(getattr(memory_store, "semantic_records", None)):
            raise TypeError("memory_store must expose semantic_records()")
        self.store = memory_store
        self.config = config or MemoryIntegrationConfig()

    def retrieve(self, query: MemoryQuery) -> tuple[MemoryEvidence, ...]:
        """Rank claims only when a concept, relation, goal, or context cue is actually compatible."""
        results = []
        for raw in self.store.semantic_records():
            claim = raw.get("claim", raw)
            relations = raw.get("relations", raw.get("semantic_relations", ()))
            context = raw.get("context", {})
            status = str(raw.get("status", raw.get("verification_status", "unknown")))
            quality = {
                "accepted": 1.0, "verified": 1.0, "pending_verification": 0.25,
                "candidate_requires_prior_knowledge_and_reliability_check": 0.25,
                "rejected": 0.0, "contradicted": 0.0,
            }.get(status.casefold(), 0.5)
            concept = _similarity(query.text, claim)
            relation_score = _similarity(query.event_cues, relations) if query.event_cues else 0.0
            goal_score = _similarity(query.goal, raw.get("goal", context.get("goal", ""))) if isinstance(context, Mapping) else 0.0
            context_score = _similarity(query.context_cues, context) if query.context_cues else 0.0
            if max(concept, relation_score, goal_score, context_score) <= 0.0:
                continue
            score = _weighted({
                "concept": concept, "relations": relation_score, "goal": goal_score,
                "quality": quality, "context": context_score,
            }, self.config.semantic_weights)
            if score < self.config.minimum_relevance:
                continue
            memory_id = str(raw.get("knowledge_id") or raw.get("memory_id") or "")
            if not memory_id:
                continue
            provenance = raw.get("provenance", {})
            if not isinstance(provenance, Mapping):
                provenance = {"source_provenance": provenance}
            results.append(MemoryEvidence(
                memory_id=memory_id,
                memory_type="semantic",
                content=claim,
                entities=raw.get("entities", {}) if isinstance(raw.get("entities", {}), Mapping) else {},
                relations=relations if isinstance(relations, Sequence) and not isinstance(relations, (str, bytes)) else (),
                context=context if isinstance(context, Mapping) else {},
                provenance={
                    **dict(provenance),
                    "source": "point7_semantic_knowledge",
                    "claim_key": raw.get("claim_key"),
                },
                source_reference=str(raw.get("source") or "point7.semantic_knowledge"),
                verification_status=status,
                valid_time=raw.get("valid_time"),
                independence_group=str(raw.get("independence_group") or raw.get("source_id") or f"semantic:{memory_id}"),
                parent_memory_ids=tuple(raw.get("parent_memory_ids", ())),
                sequence_id=str(raw.get("sequence_id", "")),
                relevance_score=score,
                confidence=_unit(raw.get("confidence", 0.5)),
            ))
        return tuple(results)


class MemoryRankingEngine:
    """Combine memory types into a stable cross-store ranking without treating score as truth."""

    def rank(self, evidence: Sequence[MemoryEvidence], limit: int) -> tuple[MemoryEvidence, ...]:
        """Sort by relevance then stable ID, dropping duplicate IDs without merging source quality."""
        if limit < 1:
            raise ValueError("limit must be positive")
        unique: dict[str, MemoryEvidence] = {}
        for item in evidence:
            current = unique.get(item.memory_id)
            if current is None or item.relevance_score > current.relevance_score:
                unique[item.memory_id] = item
        ordered = sorted(unique.values(), key=lambda item: (-item.relevance_score, item.memory_type, item.memory_id))
        return tuple(ordered[:limit])


class MemoryRetrievalCoordinator:
    """Coordinate read-only episodic and semantic recall from the existing Point 7 store."""

    def __init__(self, memory_store: Any, config: MemoryIntegrationConfig | None = None) -> None:
        """Construct typed adapters around one shared, persistent Point 7 database."""
        self.config = config or MemoryIntegrationConfig()
        self.episodic = EpisodicRetrievalAdapter(memory_store, self.config)
        self.semantic = SemanticRetrievalAdapter(memory_store, self.config)
        self.ranking = MemoryRankingEngine()

    def retrieve(self, query: MemoryQuery) -> MemoryBatch:
        """Recall previous memories only, then publish a bounded ranked batch and diagnostics."""
        episodic = self.episodic.retrieve(query)
        semantic = self.semantic.retrieve(query)
        ranked = self.ranking.rank(
            (*episodic, *semantic),
            min(query.retrieval_limit, self.config.max_retrieval_items),
        )
        return MemoryBatch(
            query=query,
            items=ranked,
            diagnostics={
                "episodic_candidates": len(episodic),
                "semantic_candidates": len(semantic),
                "returned_count": len(ranked),
                "same_sequence_excluded": True,
                "recall_mutated_long_term_store": False,
                "scores_are_probabilities": False,
                "ranking": "normalized weighted feature means with deterministic tie breaking",
            },
        )
