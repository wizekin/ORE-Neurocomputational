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
            "LEXICAL_ENTRY", "SEMANTIC_CONCEPT", "SEMANTIC_CLASS",
            "SEMANTIC_CATEGORY", "PREDICATE_FRAME",
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
            semantic_resource = self.resource.semantic_knowledge
            concepts = (
                semantic_resource.concepts if semantic_resource is not None
                else self.resource.semantic.concepts
            )
            concept_records = (
                {item.concept.concept_id: item for item in semantic_resource.concept_records}
                if semantic_resource is not None else {}
            )
            for concept in concepts:
                labels = (concept.label, *concept.aliases)
                if any(self._normalize(label) == normalized for label in labels) and (
                    language_filter is None or concept.language.casefold() == language_filter
                ):
                    imported = concept_records.get(concept.concept_id)
                    details = () if imported is None else (
                        f"record_kind:{imported.record_kind}",
                        f"definition:{imported.definition or ''}",
                        f"active:{imported.active}",
                        f"lemma_record_id:{imported.lemma_record_id or ''}",
                    )
                    hits.append(USGRetrievedItem(
                        "SEMANTIC_CONCEPT", concept.concept_id, concept.label,
                        concept.language, concept.source_reference,
                        imported.lifecycle_state if imported is not None else None,
                        details,
                    ))

        if kind in (None, "SEMANTIC_CLASS", "SEMANTIC_CATEGORY"):
            semantic_resource = self.resource.semantic_knowledge
            if semantic_resource is not None:
                class_items = semantic_resource.categories
                for item in class_items:
                    if self._normalize(item.class_key) == normalized:
                        hits.append(USGRetrievedItem(
                            "SEMANTIC_CATEGORY" if kind == "SEMANTIC_CATEGORY" else "SEMANTIC_CLASS",
                            item.category_id, item.class_key,
                            "und", item.source_reference, item.lifecycle_state,
                            (item.definition, f"parents:{','.join(item.parent_category_ids)}",
                             f"unresolved_parents:{','.join(item.unresolved_parent_keys)}"),
                        ))
            else:
                for item in self.resource.structures.classes:
                    if self._normalize(item.class_key) == normalized:
                        hits.append(USGRetrievedItem(
                            "SEMANTIC_CLASS", item.class_id, item.class_key,
                            "und", item.source_reference, item.lifecycle_state,
                            (item.definition,),
                        ))

        if kind in (None, "PREDICATE_FRAME"):
            semantic_resource = self.resource.semantic_knowledge
            if semantic_resource is not None:
                frame_items = semantic_resource.predicate_frames
                for item in frame_items:
                    if self._normalize(item.frame_key) == normalized:
                        hits.append(USGRetrievedItem(
                            "PREDICATE_FRAME", item.frame_id, item.frame_key,
                            "und", item.source_reference, item.lifecycle_state,
                            tuple(role.raw_json for role in item.roles) + (
                                f"status:{item.status}",
                                f"unresolved:{','.join(item.unresolved_fields)}",
                            ),
                        ))
            else:
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
        relations = (
            self.resource.semantic_knowledge.relations
            if self.resource.semantic_knowledge is not None
            else self.resource.relations
        )
        return tuple(
            relation for relation in relations
            if relation.subject_id == record_id or relation.object_id == record_id
        )

    def memberships_for(self, record_id: str) -> tuple[Any, ...]:
        """Return explicit semantic-category membership assertions incident to one ID."""
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id must be non-empty text")
        semantic_resource = self.resource.semantic_knowledge
        if semantic_resource is None:
            return ()
        return tuple(
            membership for membership in semantic_resource.memberships
            if membership.member_id == record_id or membership.category_id == record_id
        )


    def context_by_id(self, context_id: str) -> Any | None:
        """Retrieve one exact preserved USG context summary by source ID."""
        if not isinstance(context_id, str) or not context_id.strip():
            raise ValueError("context_id must be non-empty text")
        return next(
            (item for item in self.resource.contexts if item.context_id == context_id),
            None,
        )

    def evidence_by_id(self, evidence_id: str) -> Any | None:
        """Retrieve one exact preserved USG evidence summary by source ID."""
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise ValueError("evidence_id must be non-empty text")
        return next(
            (item for item in self.resource.evidence if item.evidence_id == evidence_id),
            None,
        )

    def source_by_id(self, source_id: str) -> Any | None:
        """Retrieve one exact preserved source-rights summary by source ID."""
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("source_id must be non-empty text")
        return next(
            (item for item in self.resource.sources if item.source_id == source_id),
            None,
        )

    def frames_by_key(self, frame_key: str) -> tuple[Any, ...]:
        """Return all exact frame-key matches without selecting a preferred version."""
        result = self.lookup(frame_key, kind="PREDICATE_FRAME")
        ids = {item.record_id for item in result.items}
        return tuple(
            frame for frame in self.resource.structures.predicate_frames
            if frame.frame_id in ids
        )

    def semantic_frames_by_key(self, frame_key: str) -> tuple[Any, ...]:
        """Return complete ORE predicate structures, including preserved incomplete frames."""
        if not isinstance(frame_key, str) or not frame_key.strip():
            raise ValueError("frame_key must be non-empty text")
        if self.resource.semantic_knowledge is None:
            return ()
        normalized = self._normalize(frame_key)
        return tuple(
            frame for frame in self.resource.semantic_knowledge.predicate_frames
            if self._normalize(frame.frame_key) == normalized
        )


__all__ = ["USGRetrievedItem", "USGRetrievalResult", "USGKnowledgeIndex"]
