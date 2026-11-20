"""Deterministic exact retrieval over a unified USG-derived ORE resource."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .knowledge import USGKnowledgeResource


@dataclass(frozen=True, slots=True)
class USGRetrievedItem:
    """One exact retrieval hit with source identity and original provenance."""

    kind: str
    record_id: str
    label: str
    language: str
    source_reference: str
    lifecycle_state: str | None = None
    details: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class USGRetrievalResult:
    """A deterministic lookup result; no probabilistic confidence is fabricated."""

    query: str
    normalized_query: str
    items: tuple[USGRetrievedItem, ...]
    exact_match: bool
    bundle_id: str
    bundle_digest: str


class USGKnowledgeIndex:
    """Provide exact lexical, concept, class, frame, and relation lookup."""

    def __init__(self, resource: USGKnowledgeResource) -> None:
        """Index one immutable unified resource without modifying its records."""
        if not isinstance(resource, USGKnowledgeResource):
            raise TypeError("resource must be a USGKnowledgeResource")
        self.resource = resource

    @staticmethod
    def _normalize(value: str) -> str:
        """Normalize only surrounding whitespace and Unicode case; do not stem."""
        return value.strip().casefold()

    def lookup(
        self,
        query: str,
        *,
        kind: str | None = None,
        language: str | None = None,
    ) -> USGRetrievalResult:
        """Return exact case-insensitive matches; optionally constrain kind/language."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty text")
        allowed = {
            "LEXICAL_ENTRY", "SEMANTIC_CONCEPT", "SEMANTIC_CLASS", "PREDICATE_FRAME",
        }
        if kind is not None and kind not in allowed:
            raise ValueError(f"kind must be one of {sorted(allowed)}")
        normalized = self._normalize(query)
        language_filter = language.strip().casefold() if language else None
        hits: list[USGRetrievedItem] = []

        if kind in (None, "LEXICAL_ENTRY"):
            for entry in self.resource.lexical.entries:
                if self._normalize(entry.orthography) == normalized and (
                    language_filter is None or entry.language.casefold() == language_filter
                ):
                    details = tuple(
                        f"meaning:{meaning.meaning_id}:{meaning.gloss}"
                        for meaning in entry.meanings
                    )
                    hits.append(USGRetrievedItem(
                        "LEXICAL_ENTRY", entry.entry_id, entry.orthography,
                        entry.language, entry.source_reference, None, details,
                    ))

        if kind in (None, "SEMANTIC_CONCEPT"):
            for concept in self.resource.semantic.concepts:
                labels = (concept.label, *concept.aliases)
                if any(self._normalize(label) == normalized for label in labels) and (
                    language_filter is None or concept.language.casefold() == language_filter
                ):
                    hits.append(USGRetrievedItem(
                        "SEMANTIC_CONCEPT", concept.concept_id, concept.label,
                        concept.language, concept.source_reference,
                    ))

        if kind in (None, "SEMANTIC_CLASS"):
            for item in self.resource.structures.classes:
                if self._normalize(item.class_key) == normalized:
                    hits.append(USGRetrievedItem(
                        "SEMANTIC_CLASS", item.class_id, item.class_key,
                        "und", item.source_reference, item.lifecycle_state,
                        (item.definition,),
                    ))

        if kind in (None, "PREDICATE_FRAME"):
            for item in self.resource.structures.predicate_frames:
                if self._normalize(item.frame_key) == normalized:
                    hits.append(USGRetrievedItem(
                        "PREDICATE_FRAME", item.frame_id, item.frame_key,
                        "und", item.source_reference, item.lifecycle_state,
                        (item.roles_json,),
                    ))

        hits.sort(key=lambda item: (item.kind, item.record_id))
        return USGRetrievalResult(
            query=query,
            normalized_query=normalized,
            items=tuple(hits),
            exact_match=bool(hits),
            bundle_id=self.resource.bundle_id,
            bundle_digest=self.resource.bundle_digest,
        )

    def relations_for(self, record_id: str) -> tuple[Any, ...]:
        """Return explicit directed relations incident to an exact ORE record ID."""
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id must be non-empty text")
        return tuple(
            relation for relation in self.resource.relations
            if relation.subject_id == record_id or relation.object_id == record_id
        )

    def frames_by_key(self, frame_key: str) -> tuple[Any, ...]:
        """Return all exact frame-key matches without selecting a preferred version."""
        result = self.lookup(frame_key, kind="PREDICATE_FRAME")
        ids = {item.record_id for item in result.items}
        return tuple(
            frame for frame in self.resource.structures.predicate_frames
            if frame.frame_id in ids
        )


__all__ = ["USGRetrievedItem", "USGRetrievalResult", "USGKnowledgeIndex"]
