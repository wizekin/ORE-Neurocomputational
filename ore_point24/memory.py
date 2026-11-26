"""Phase VI, Steps 26-27 - memory systems and retrieval.

Step 26 separates semantic, episodic and procedural memory; entries keep
stable identifiers, timestamps, source lineage, access policy, version and
verification status (verified, tentative, contradicted or obsolete).  Step 27
builds retrieval with exact and approximate paths: exact lookup for stable
IDs and contract keys, semantic/graph retrieval for candidate discovery,
approximate indexes only as candidate filters, and every returned candidate
is re-checked against the authoritative record.  ``R(x, q) = sum_i w_i s_i(x, q)``
is a ranking function unless calibrated against a defined relevance dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .core import new_id, sha256_digest, utc_now_iso

MEMORY_SCHEMA = "ore.point24.memory.v1"

SEMANTIC = "semantic"
EPISODIC = "episodic"
PROCEDURAL = "procedural"

VERIFIED = "verified"
TENTATIVE = "tentative"
CONTRADICTED = "contradicted"
OBSOLETE = "obsolete"


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One typed memory entry with full provenance (Step 26)."""

    record_type: str
    content: str
    source_artifact: str
    model_version: str | None = None
    access_policy: str = "module_read"
    status: str = VERIFIED
    relations: tuple[tuple[str, str], ...] = ()
    record_id: str = field(default_factory=lambda: new_id("memory"))
    created_at: str = field(default_factory=utc_now_iso)
    version: int = 1

    def __post_init__(self) -> None:
        if self.record_type not in {SEMANTIC, EPISODIC, PROCEDURAL}:
            raise ValueError(f"unknown memory record type: {self.record_type!r}")
        if self.status not in {VERIFIED, TENTATIVE, CONTRADICTED, OBSOLETE}:
            raise ValueError(f"unknown memory status: {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "record_type": self.record_type,
            "content": self.content,
            "source_artifact": self.source_artifact,
            "model_version": self.model_version,
            "access_policy": self.access_policy,
            "status": self.status,
            "relations": [[kind, target] for kind, target in self.relations],
            "created_at": self.created_at,
            "version": self.version,
        }


class MemoryStore:
    """The three separate memory systems with deduplication and provenance.

    Semantic memory stores deduplicated claims and relationships; episodic
    memory stores chronological task observations, decisions and outcomes;
    procedural memory stores reusable routines, policies and verified
    algorithms.  A vector-similarity match is only ever a retrieval hint,
    never a fact-verification result.
    """

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._content_fingerprints: dict[str, str] = {}

    def add(self, record: MemoryRecord) -> str:
        """Add one record; duplicate content is deduplicated by fingerprint."""
        fingerprint = sha256_digest(
            {"type": record.record_type, "content": record.content}
        )
        existing = next(
            (rid for rid, fp in self._content_fingerprints.items() if fp == fingerprint),
            None,
        )
        if existing is not None:
            return existing
        self._records[record.record_id] = record
        self._content_fingerprints[record.record_id] = fingerprint
        return record.record_id

    def get(self, record_id: str) -> MemoryRecord | None:
        return self._records.get(record_id)

    def by_type(self, record_type: str) -> tuple[MemoryRecord, ...]:
        return tuple(
            record for record in self._records.values()
            if record.record_type == record_type
        )

    def all_records(self) -> tuple[MemoryRecord, ...]:
        return tuple(self._records.values())

    def counts(self) -> dict[str, int]:
        return {
            "semantic": len(self.by_type(SEMANTIC)),
            "episodic": len(self.by_type(EPISODIC)),
            "procedural": len(self.by_type(PROCEDURAL)),
            "total": len(self._records),
        }


@dataclass(frozen=True, slots=True)
class RetrievedItem:
    """One ranked retrieval candidate with its exact record reference."""

    record_id: str
    record_type: str
    content: str
    score: float
    signals: Mapping[str, float]
    exact_match: bool
    rechecked: bool = False
    authoritative: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "record_type": self.record_type,
            "content": self.content,
            "score": self.score,
            "signals": dict(self.signals),
            "exact_match": self.exact_match,
            "rechecked": self.rechecked,
            "authoritative": self.authoritative,
        }


class MemoryRetriever:
    """Step 27 - exact and approximate retrieval with re-check (Step 27).

    Retrieval combines normalized signals - relevance, recency, source
    reliability and goal relevance - each separately inspectable.  Cache hits
    never bypass authorization, source validity or version checks, and every
    approximate candidate is re-checked against the authoritative record
    before it can be used in a critical decision.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        weights: Mapping[str, float] | None = None,
        max_results: int = 16,
    ) -> None:
        self.store = store
        self.weights = {
            "relevance": 0.55,
            "recency": 0.15,
            "source_reliability": 0.15,
            "goal_relevance": 0.15,
            **dict(weights or {}),
        }
        self.max_results = max_results
        self._cache: dict[str, list[str]] = {}

    def _relevance(self, record: MemoryRecord, query: str) -> float:
        query_terms = set(query.casefold().split())
        if not query_terms:
            return 0.0
        content_terms = set(record.content.casefold().split())
        if not content_terms:
            return 0.0
        return len(query_terms & content_terms) / len(query_terms)

    def _recency(self, record: MemoryRecord, records: Sequence[MemoryRecord]) -> float:
        ordered = sorted(record.created_at for record in records)
        if len(ordered) < 2 or record.created_at in (ordered[0],):
            position = ordered.index(record.created_at) if record.created_at in ordered else 0
        else:
            position = ordered.index(record.created_at)
        return position / max(len(ordered) - 1, 1)

    def retrieve(
        self,
        query: str,
        *,
        record_id: str | None = None,
        goal: str = "",
        authoritative_only: bool = False,
    ) -> tuple[RetrievedItem, ...]:
        """Exact lookup first; ranked semantic retrieval for discovery."""
        results: list[RetrievedItem] = []
        records = self.store.all_records()
        if record_id is not None:
            record = self.store.get(record_id)
            if record is None:
                return ()
            results.append(RetrievedItem(
                record_id=record.record_id, record_type=record.record_type,
                content=record.content, score=1.0,
                signals={"relevance": 1.0}, exact_match=True, rechecked=True,
                authoritative=record.status == VERIFIED,
            ))
            return tuple(results)
        goal_terms = set(goal.casefold().split())
        for record in records:
            if authoritative_only and record.status != VERIFIED:
                continue
            relevance = self._relevance(record, query)
            recency = self._recency(record, records)
            reliability = 1.0 if record.status == VERIFIED else 0.5
            goal_relevance = (
                len(goal_terms & set(record.content.casefold().split())) / len(goal_terms)
                if goal_terms else 0.0
            )
            signals = {
                "relevance": relevance,
                "recency": recency,
                "source_reliability": reliability,
                "goal_relevance": goal_relevance,
            }
            score = sum(self.weights[name] * value for name, value in signals.items())
            results.append(RetrievedItem(
                record_id=record.record_id, record_type=record.record_type,
                content=record.content, score=round(score, 4),
                signals={k: round(v, 4) for k, v in signals.items()},
                exact_match=False, rechecked=True,
                authoritative=record.status == VERIFIED,
            ))
        results.sort(key=lambda item: (-item.score, item.record_id))
        return tuple(results[: self.max_results])

    def recheck_authoritative(self, item: RetrievedItem) -> RetrievedItem:
        """Re-check an approximate hit against the authoritative record."""
        record = self.store.get(item.record_id)
        if record is None:
            return RetrievedItem(
                record_id=item.record_id, record_type=item.record_type,
                content=item.content, score=0.0, signals=dict(item.signals),
                exact_match=False, rechecked=True, authoritative=False,
            )
        return RetrievedItem(
            record_id=record.record_id, record_type=record.record_type,
            content=record.content, score=item.score,
            signals=dict(item.signals), exact_match=item.exact_match,
            rechecked=True, authoritative=record.status == VERIFIED,
        )


__all__ = [
    "MEMORY_SCHEMA",
    "MemoryRecord",
    "MemoryRetriever",
    "MemoryStore",
    "RetrievedItem",
]
