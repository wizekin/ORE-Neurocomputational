"""Hybrid retrieval with provenance and reranking (Step 41).

Exact IDs, lexical retrieval, USG semantic relations, graph traversal and
metadata filters are combined as R(d, q) = a*R_lex + b*R_sem + g*R_graph +
d*R_fresh + e*R_auth, then reranked by task intent.  Results carry evidence
and provenance, never bare generated summaries, and ranking quality is
measurable with Recall@k, Precision@k, MRR and nDCG.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Protocol, Sequence

from ..config import RetrievalWeights
from ..contracts import utc_now
from ..functional import (
    hybrid_retrieval_score,
    mrr,
    ndcg,
    precision_at_k,
    recall_at_k,
    unit_scale,
)

__all__ = (
    "GraphProvider",
    "HybridResult",
    "HybridRetriever",
    "LexicalProvider",
    "RankedDocument",
    "RankingMetrics",
    "RetrievedDocument",
    "RetrievalProvider",
    "SemanticProvider",
    "evaluate_ranking",
    "token_overlap",
)

_MAX_AGE_SECONDS = 30 * 24 * 3600.0


@dataclass(frozen=True, slots=True)
class RetrievedDocument:
    """One retrievable document with provenance metadata (Step 26)."""

    doc_id: str
    content: str
    source_id: str
    locator: str = ""
    timestamp: str = field(default_factory=utc_now)
    version: str = ""
    authority: float = 0.5
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the document."""
        return {"doc_id": self.doc_id, "content": self.content,
                "source_id": self.source_id, "locator": self.locator,
                "timestamp": self.timestamp, "version": self.version,
                "authority": self.authority, "metadata": dict(self.metadata)}


class RetrievalProvider(Protocol):
    """A retrieval channel returning provenance-carrying documents (Step 41)."""

    def search(self, query: str, limit: int) -> tuple[RetrievedDocument, ...]: ...


def token_overlap(query: str, content: str) -> float:
    """Return the lexical token overlap ratio used by the lexical channel."""
    query_tokens = {token for token in query.lower().split() if token}
    content_tokens = {token for token in content.lower().split() if token}
    if not query_tokens or not content_tokens:
        return 0.0
    return len(query_tokens & content_tokens) / len(query_tokens)


class LexicalProvider:
    """Token-overlap lexical retrieval (Step 41, lexical channel)."""

    def __init__(self, documents: Sequence[RetrievedDocument]) -> None:
        """Index the given documents."""
        self._documents = tuple(documents)

    def search(self, query: str, limit: int) -> tuple[RetrievedDocument, ...]:
        """Return documents ranked by token overlap."""
        scored = sorted(
            ((token_overlap(query, doc.content), doc) for doc in self._documents),
            key=lambda pair: (-pair[0], pair[1].doc_id))
        return tuple(doc for score, doc in scored[:limit] if score > 0.0)


class SemanticProvider:
    """USG-backed semantic retrieval through a duck-typed index (Step 41).

    The index satisfies the ``lookup(query) -> result`` and
    ``relations_for(record_id)`` shape of ``ore_usg.retrieval.USGKnowledgeIndex``
    without importing it, keeping the package boundary replaceable.
    """

    def __init__(self, index: Any, *, authority: float = 0.9) -> None:
        """Wrap a USG-like knowledge index."""
        self._index = index
        self._authority = authority

    def search(self, query: str, limit: int) -> tuple[RetrievedDocument, ...]:
        """Return exact-match semantic records with provenance."""
        try:
            result = self._index.lookup(query)
        except Exception:  # noqa: BLE001 - external index is untrusted input
            return ()
        items = list(getattr(result, "items", ()) or ())[:limit]
        documents = []
        for item in items:
            documents.append(RetrievedDocument(
                doc_id=f"usg:{item.record_id}", content=item.label,
                source_id=getattr(item, "source_reference", "usg") or "usg",
                locator=item.record_id,
                version="usg-bundle", authority=self._authority,
                metadata={"kind": item.kind, "exact_match": True}))
        return tuple(documents)


class GraphProvider:
    """Graph traversal over USG-style relations (Step 41, graph channel)."""

    def __init__(self, index: Any, *, authority: float = 0.8) -> None:
        """Wrap a USG-like index for relation traversal."""
        self._index = index
        self._authority = authority

    def search(self, query: str, limit: int) -> tuple[RetrievedDocument, ...]:
        """Return documents reachable through relations from an exact match."""
        try:
            result = self._index.lookup(query)
            items = list(getattr(result, "items", ()) or ())[:1]
        except Exception:  # noqa: BLE001 - external index is untrusted input
            return ()
        documents: list[RetrievedDocument] = []
        for item in items:
            try:
                relations = self._index.relations_for(item.record_id)
            except Exception:  # noqa: BLE001
                relations = ()
            for relation in list(relations)[:limit]:
                documents.append(RetrievedDocument(
                    doc_id=f"usg-relation:{relation.relation_id}",
                    content=f"{relation.subject_id} {relation.predicate} "
                            f"{relation.object_id}",
                    source_id=getattr(relation, "source_reference", "usg") or "usg",
                    locator=relation.relation_id, version="usg-bundle",
                    authority=self._authority,
                    metadata={"predicate": relation.predicate,
                              "subject": relation.subject_id,
                              "object": relation.object_id}))
        return tuple(documents)


@dataclass(frozen=True, slots=True)
class RankedDocument:
    """One reranked document with component scores and provenance (Step 41)."""

    document: RetrievedDocument
    score: float
    lexical: float
    semantic: float
    graph: float
    freshness: float
    authority: float
    exact_id_match: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize with full provenance, not a bare summary."""
        return {
            "doc_id": self.document.doc_id,
            "source_id": self.document.source_id,
            "locator": self.document.locator,
            "score": self.score,
            "components": {
                "lexical": self.lexical, "semantic": self.semantic,
                "graph": self.graph, "freshness": self.freshness,
                "authority": self.authority},
            "exact_id_match": self.exact_id_match,
        }


@dataclass(frozen=True, slots=True)
class HybridResult:
    """The result of one hybrid retrieval query (Step 41)."""

    query: str
    ranked: tuple[RankedDocument, ...]

    @property
    def doc_ids(self) -> tuple[str, ...]:
        """Return the ranked document IDs."""
        return tuple(item.document.doc_id for item in self.ranked)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result."""
        return {"query": self.query,
                "ranked": [item.to_dict() for item in self.ranked]}


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    """Ranking evaluation metrics (math foundations)."""

    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize the metrics."""
        return {"recall_at_k": self.recall_at_k,
                "precision_at_k": self.precision_at_k,
                "mrr": self.mrr, "ndcg": self.ndcg}


def _age_seconds(timestamp: str) -> float:
    """Approximate document age in seconds from an ISO timestamp."""
    try:
        import time as _time
        struct = _time.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
        import calendar as _calendar
        return max(0.0, _time.time() - _calendar.timegm(struct))
    except (ValueError, TypeError):
        return _MAX_AGE_SECONDS


class HybridRetriever:
    """Combine retrieval channels and rerank by intent (Step 41)."""

    def __init__(self, providers: Sequence[RetrievalProvider],
                 weights: RetrievalWeights | None = None,
                 *, metadata_filter: Mapping[str, Any] | None = None) -> None:
        """Compose the retriever from channels and governed weights."""
        self._providers = tuple(providers)
        self._weights = weights or RetrievalWeights()
        self._metadata_filter = dict(metadata_filter or {})

    def retrieve(self, query: str, limit: int = 8,
                 intent: str = "") -> HybridResult:
        """Run every channel, blend component scores and rerank (Step 41)."""
        channels: dict[str, dict[str, RetrievedDocument]] = {
            "lexical": {}, "semantic": {}, "graph": {}}
        for provider in self._providers:
            channel = ("semantic" if isinstance(provider, SemanticProvider)
                       else "graph" if isinstance(provider, GraphProvider)
                       else "lexical")
            for document in provider.search(query, limit):
                channels[channel][document.doc_id] = document
        all_documents: dict[str, RetrievedDocument] = {}
        for channel_documents in channels.values():
            all_documents.update(channel_documents)
        query_lower = query.strip().lower()
        ranked: list[RankedDocument] = []
        for doc_id, document in all_documents.items():
            if self._metadata_filter and not self._matches_filter(document):
                continue
            lexical = token_overlap(query, document.content)
            semantic = 1.0 if doc_id in channels["semantic"] else 0.0
            graph = 1.0 if doc_id in channels["graph"] else 0.0
            freshness = 1.0 - unit_scale(
                _age_seconds(document.timestamp), _MAX_AGE_SECONDS)
            authority = max(0.0, min(1.0, document.authority))
            score = hybrid_retrieval_score(
                lexical, semantic, graph, freshness, authority,
                weights={"lexical": self._weights.lexical,
                         "semantic": self._weights.semantic,
                         "graph": self._weights.graph,
                         "freshness": self._weights.freshness,
                         "authority": self._weights.authority})
            exact = doc_id.lower() == query_lower or query_lower in doc_id.lower()
            if exact:
                score += 1.0
            if intent and intent.lower() in document.content.lower():
                score += 0.25  # intent rerank boost
            ranked.append(RankedDocument(
                document=document, score=round(score, 6),
                lexical=round(lexical, 6), semantic=semantic, graph=graph,
                freshness=round(freshness, 6), authority=authority,
                exact_id_match=exact))
        ranked.sort(key=lambda item: (-item.score, item.document.doc_id))
        return HybridResult(query=query, ranked=tuple(ranked[:limit]))

    def _matches_filter(self, document: RetrievedDocument) -> bool:
        """Apply metadata filters (Step 41)."""
        for key, expected in self._metadata_filter.items():
            if document.metadata.get(key) != expected:
                return False
        return True


def evaluate_ranking(result: HybridResult, relevant_ids: Sequence[str],
                     k: int = 5) -> RankingMetrics:
    """Measure Recall@k, Precision@k, MRR and nDCG of one result (Step 41)."""
    ranked = list(result.doc_ids)
    relevant = list(relevant_ids)
    relevance = [1.0 if doc_id in set(relevant) else 0.0 for doc_id in ranked]
    return RankingMetrics(
        recall_at_k=recall_at_k(ranked, relevant, k),
        precision_at_k=precision_at_k(ranked, relevant, k),
        mrr=mrr(ranked, relevant),
        ndcg=ndcg(relevance))


def freshness_score(timestamp: str, max_age_seconds: float = _MAX_AGE_SECONDS
                    ) -> float:
    """Return the freshness component on [0, 1] with exponential decay."""
    age = _age_seconds(timestamp)
    return math.exp(-age / max_age_seconds) if max_age_seconds > 0 else 0.0
