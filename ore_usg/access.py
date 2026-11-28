"""Typed, read-only knowledge access over the unified USG resource.

The access layer exposes exact, direct-related, and explicit-context retrieval.
It never infers inverse/transitive relations or invents calibrated confidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .knowledge import USGKnowledgeResource
from .retrieval import USGKnowledgeIndex, USGRetrievedItem


_QUERY_MODES = frozenset({"EXACT", "RELATED_CONCEPT", "CONTEXTUAL"})
_QUERY_KINDS = frozenset({
    "LEXICAL_ENTRY", "SEMANTIC_CONCEPT", "SEMANTIC_CLASS",
    "SEMANTIC_CATEGORY", "PREDICATE_FRAME",
})
_UNCERTAIN = frozenset({
    "AMBIGUOUS", "MISSING", "CONFLICTED", "CONTRADICTED",
    "CONTRADICTS", "UNRESOLVED", "CANDIDATE", "NEUTRAL", "UNASSESSED",
})


@dataclass(frozen=True, slots=True)
class USGKnowledgeQuery:
    """A validated request for exact, related-concept, or contextual knowledge."""

    mode: str
    query: str | None = None
    record_id: str | None = None
    kind: str | None = None
    language: str | None = None
    context_type: str | None = None
    context_value: str | None = None
    max_results: int = 50

    def __post_init__(self) -> None:
        if self.mode not in _QUERY_MODES:
            raise ValueError(f"mode must be one of {sorted(_QUERY_MODES)}")
        if self.kind is not None and self.kind not in _QUERY_KINDS:
            raise ValueError(f"kind must be one of {sorted(_QUERY_KINDS)}")
        if self.mode == "EXACT" and not (
            (isinstance(self.query, str) and self.query.strip())
            or (isinstance(self.record_id, str) and self.record_id.strip())
        ):
            raise ValueError("EXACT mode requires query or record_id")
        if self.mode == "RELATED_CONCEPT" and not (
            (isinstance(self.record_id, str) and self.record_id.strip())
            or (isinstance(self.query, str) and self.query.strip())
        ):
            raise ValueError("RELATED_CONCEPT mode requires record_id or query")
        if self.mode == "CONTEXTUAL" and not (
            (isinstance(self.record_id, str) and self.record_id.strip())
            or (isinstance(self.query, str) and self.query.strip())
        ):
            raise ValueError("CONTEXTUAL mode requires record_id or query")
        for name in ("language", "context_type", "context_value"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be non-empty text when provided")
        if isinstance(self.max_results, bool) or not isinstance(self.max_results, int):
            raise TypeError("max_results must be an integer")
        if not 1 <= self.max_results <= 500:
            raise ValueError("max_results must lie between 1 and 500")


@dataclass(frozen=True, slots=True)
class USGKnowledgeCandidate:
    """One retrieved record plus its uncertainty and source-lineage information."""

    kind: str
    record_id: str
    label: str
    language: str
    source_reference: str
    lifecycle_state: str | None
    assertion_status: str
    evidence_ids: tuple[str, ...] = ()
    evidence_statuses: tuple[str, ...] = ()
    context_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    source_references: tuple[str, ...] = ()
    relationship_id: str | None = None
    relationship_predicate: str | None = None
    relationship_direction: str | None = None
    confidence_value: float | None = None
    confidence_basis: str = "NOT_PROVIDED_BY_SOURCE"

    def __post_init__(self) -> None:
        if not self.kind or not self.record_id or not self.label:
            raise ValueError("candidate kind, record_id and label must be non-empty")
        if self.confidence_value is not None and not 0.0 <= self.confidence_value <= 1.0:
            raise ValueError("confidence_value must be in [0, 1] when explicitly provided")


@dataclass(frozen=True, slots=True)
class USGKnowledgeQueryResult:
    """Retrieval result with explicit availability, uncertainty, and artifact lineage."""

    mode: str
    query: str | None
    record_id: str | None
    candidates: tuple[USGKnowledgeCandidate, ...]
    relations: tuple[Any, ...]
    contexts: tuple[Any, ...]
    evidence: tuple[Any, ...]
    source_provenance: tuple[Any, ...]
    available_domains: tuple[str, ...]
    unavailable_domains: tuple[str, ...]
    bundle_id: str
    bundle_digest: str
    exact_match: bool
    uncertainty_present: bool
    confidence_available: bool = False
    confidence_note: str = (
        "No calibrated confidence value is declared by the imported USG source contract."
    )


class USGKnowledgeAccess:
    """Stable common read-only interface for ORE lexical, semantic and predicate modules."""

    def __init__(self, resource: USGKnowledgeResource) -> None:
        if not isinstance(resource, USGKnowledgeResource):
            raise TypeError("resource must be a USGKnowledgeResource")
        self.resource = resource
        self.index = USGKnowledgeIndex(resource)

    @property
    def available_domains(self) -> tuple[str, ...]:
        """Return domains with imported records; empty or absent domains remain unavailable."""
        available: list[str] = []
        if self.resource.lexical.entries:
            available.append("LEXICAL")
        if self.resource.semantic_concepts or self.resource.semantic_categories:
            available.append("SEMANTIC")
        if self.resource.semantic_relations or self.resource.semantic_memberships:
            available.append("RELATIONAL")
        if self.resource.predicate_frames:
            available.append("PREDICATE")
        if self.resource.contextual_records:
            available.append("CONTEXT")
        if self.resource.evidence_records:
            available.append("EVIDENCE")
        if self.resource.source_provenance:
            available.append("SOURCE_PROVENANCE")
        return tuple(available)

    @property
    def unavailable_domains(self) -> tuple[str, ...]:
        all_domains = {
            "LEXICAL", "SEMANTIC", "RELATIONAL", "PREDICATE",
            "CONTEXT", "EVIDENCE", "SOURCE_PROVENANCE",
        }
        return tuple(sorted(all_domains - set(self.available_domains)))

    def lexical_entries(self, orthography: str, *, language: str | None = None) -> tuple[Any, ...]:
        """Return all exact matching lexical entries; preserve homographs as alternatives."""
        result = self.index.lookup(orthography, kind="LEXICAL_ENTRY", language=language)
        wanted = {item.record_id for item in result.items}
        return tuple(item for item in self.resource.lexical.entries if item.entry_id in wanted)

    def lexical_meanings(self, orthography: str, *, language: str | None = None) -> tuple[Any, ...]:
        """Return every meaning of every exact matching lexical entry without selecting a sense."""
        return tuple(
            meaning
            for entry in self.lexical_entries(orthography, language=language)
            for meaning in entry.meanings
        )

    def semantic_concepts(self, label_or_alias: str, *, language: str | None = None) -> tuple[Any, ...]:
        """Return every exact concept label/alias match, retaining polysemy and ambiguity."""
        result = self.index.lookup(label_or_alias, kind="SEMANTIC_CONCEPT", language=language)
        wanted = {item.record_id for item in result.items}
        concepts = (
            self.resource.semantic_knowledge.concepts
            if self.resource.semantic_knowledge is not None
            else self.resource.semantic_concepts
        )
        return tuple(item for item in concepts if item.concept_id in wanted)

    def source_provenance_by_id(self, source_id: str) -> Any | None:
        """Return licensing and redistribution metadata for an exact source ID."""
        return self.index.source_provenance_by_id(source_id)

    def semantic_relations(self, record_id: str) -> tuple[Any, ...]:
        """Return explicitly asserted directed semantic relations incident to record_id."""
        return self.index.relations_for(record_id)

    def predicate_frames(
        self, predicate: str, *, executable_only: bool = False,
    ) -> tuple[Any, ...]:
        """Return all exact frame versions; optionally limit to explicitly executable frames."""
        if self.resource.semantic_knowledge is not None:
            frames = self.index.semantic_frames_by_key(predicate)
            if executable_only:
                return tuple(
                    item.ore_frame for item in frames
                    if item.active and item.status == "EXECUTABLE" and item.ore_frame is not None
                )
            return frames
        frames = self.index.frames_by_key(predicate)
        if executable_only:
            from .processors import to_ore_predicate_frame
            converted = []
            for frame in frames:
                try:
                    converted.append(to_ore_predicate_frame(frame))
                except ValueError:
                    continue
            return tuple(converted)
        return frames

    def query(self, request: USGKnowledgeQuery) -> USGKnowledgeQueryResult:
        """Execute one bounded lookup while preserving provenance and unresolved states."""
        if not isinstance(request, USGKnowledgeQuery):
            raise TypeError("request must be a USGKnowledgeQuery")
        if request.mode == "EXACT":
            return self._exact(request)
        if request.mode == "RELATED_CONCEPT":
            return self._related(request)
        return self._contextual(request)

    def _exact(self, request: USGKnowledgeQuery) -> USGKnowledgeQueryResult:
        if request.record_id:
            item = self._item_by_id(request.record_id)
            kind_matches = (
                item is not None
                and (
                    request.kind is None
                    or request.kind == item.kind
                    or (request.kind == "SEMANTIC_CLASS" and item.kind == "SEMANTIC_CATEGORY")
                )
            )
            candidates = (
                (self._candidate(item),)
                if kind_matches and item is not None else ()
            )
        else:
            found = self.index.lookup(
                request.query or "", kind=request.kind, language=request.language,
            )
            candidates = tuple(
                self._candidate(item)
                for item in found.items[:request.max_results]
            )
        context_ids = {
            context_id for candidate in candidates
            for context_id in candidate.context_ids
        }
        contexts = {
            item.context_id: item
            for candidate in candidates
            for item in self.index.contexts_for(candidate.record_id)
        }
        contexts.update({
            item.context_id: item
            for context_id in context_ids
            if (item := self.index.context_by_id(context_id)) is not None
        })
        evidence_ids = {
            evidence_id for candidate in candidates
            for evidence_id in candidate.evidence_ids
        }
        evidence = tuple(
            item for evidence_id in sorted(evidence_ids)
            if (item := self.index.evidence_by_id(evidence_id)) is not None
        )
        evidence += tuple(
            item for candidate in candidates
            for item in self.index.evidence_for(candidate.record_id)
        )
        return self._result(
            request, candidates=candidates, relations=(),
            contexts=tuple(contexts[key] for key in sorted(contexts)),
            evidence=evidence,
            exact_match=bool(candidates),
        )

    def _related(self, request: USGKnowledgeQuery) -> USGKnowledgeQueryResult:
        anchor_ids: list[str] = []
        if request.record_id:
            anchor_ids = [request.record_id]
        elif request.query:
            anchors = self.index.lookup(
                request.query, kind=request.kind, language=request.language,
            )
            anchor_ids = [item.record_id for item in anchors.items]

        candidates: list[USGKnowledgeCandidate] = []
        relations: list[Any] = []
        seen: set[tuple[str, str | None]] = set()
        for anchor_id in anchor_ids:
            for relation in self.index.relations_for(anchor_id):
                target_id = (
                    relation.object_id if relation.subject_id == anchor_id
                    else relation.subject_id
                )
                relationship_id = getattr(relation, "relation_id", None)
                key = (target_id, relationship_id)
                if key in seen:
                    continue
                seen.add(key)
                item = self._item_by_id(target_id)
                if item is not None:
                    candidates.append(self._candidate(
                        item,
                        assertion_status=getattr(relation, "status", "UNASSESSED"),
                        evidence_ids=tuple(getattr(relation, "evidence_ids", ())),
                        context_ids=tuple(getattr(relation, "context_ids", ())),
                        relationship_id=relationship_id,
                        relationship_predicate=getattr(relation, "predicate", None),
                        relationship_direction=getattr(relation, "direction", "DIRECTED"),
                    ))
                    relations.append(relation)

            semantic = self.resource.semantic_knowledge
            if semantic is not None:
                for membership in self.index.memberships_for(anchor_id):
                    target_id = (
                        membership.category_id if membership.member_id == anchor_id
                        else membership.member_id
                    )
                    relationship_id = getattr(membership, "membership_id", None)
                    key = (target_id, relationship_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    item = self._item_by_id(target_id)
                    if item is not None:
                        candidates.append(self._candidate(
                            item,
                            assertion_status=getattr(membership, "status", "UNASSESSED"),
                            evidence_ids=tuple(getattr(membership, "evidence_ids", ())),
                            context_ids=tuple(getattr(membership, "context_ids", ())),
                            relationship_id=relationship_id,
                            relationship_predicate=getattr(membership, "predicate", None),
                            relationship_direction=getattr(membership, "direction", "DIRECTED"),
                        ))
                        relations.append(membership)

            integration = self.resource.language_integration
            if integration is not None:
                for link in integration.lexical_concept_links:
                    if link.lexical_entry_id == anchor_id or link.lexical_meaning_id == anchor_id:
                        target_id = link.semantic_concept_id
                    elif link.semantic_concept_id == anchor_id:
                        target_id = link.lexical_entry_id
                    else:
                        continue
                    if target_id is None:
                        continue
                    key = (target_id, link.link_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    item = self._item_by_id(target_id)
                    if item is not None:
                        candidates.append(self._candidate(
                            item,
                            assertion_status=link.status,
                            evidence_ids=link.evidence_ids,
                            context_ids=link.context_ids,
                            relationship_id=link.link_id,
                            relationship_predicate=link.predicate,
                            relationship_direction=link.direction,
                        ))
                        relations.append(link)

        candidates.sort(key=lambda item: (item.kind, item.record_id, item.relationship_id or ""))
        selected = tuple(candidates[:request.max_results])
        relation_ids = {item.relationship_id for item in selected}
        selected_relations = tuple(
            item for item in relations
            if getattr(item, "relation_id", getattr(item, "membership_id", getattr(item, "link_id", None)))
            in relation_ids
        )
        contexts = tuple(
            context for candidate in selected
            for context in self.index.contexts_for(candidate.record_id)
        )
        evidence = tuple(
            item for candidate in selected
            for item in self.index.evidence_for(candidate.record_id)
        )
        evidence += tuple(
            item
            for relation in selected_relations
            for evidence_id in getattr(relation, "evidence_ids", ())
            if (item := self.index.evidence_by_id(evidence_id)) is not None
        )
        return self._result(
            request, candidates=selected, relations=selected_relations,
            contexts=contexts, evidence=evidence, exact_match=bool(anchor_ids),
        )

    def _contextual(self, request: USGKnowledgeQuery) -> USGKnowledgeQueryResult:
        integration = self.resource.language_integration
        if integration is None:
            return self._result(
                request, candidates=(), relations=(), contexts=(), evidence=(),
                exact_match=False,
            )

        anchor_ids: list[str] = []
        if request.record_id:
            anchor_ids.append(request.record_id)
        elif request.query:
            anchors = self.index.lookup(
                request.query, kind=request.kind, language=request.language,
            )
            for anchor in anchors.items:
                anchor_ids.append(anchor.record_id)
                # A lexical entry may contain multiple source-backed meanings.
                # Include each meaning as a separate context anchor so a query
                # for a polysemous form can return distinct domain readings.
                if anchor.kind == "LEXICAL_ENTRY":
                    entry = next(
                        (item for item in self.resource.lexical.entries if item.entry_id == anchor.record_id),
                        None,
                    )
                    if entry is not None:
                        anchor_ids.extend(item.meaning_id for item in entry.meanings)
        anchor_ids = list(dict.fromkeys(anchor_ids))
        if not anchor_ids:
            return self._result(
                request, candidates=(), relations=(), contexts=(), evidence=(),
                exact_match=False,
            )

        contexts_by_id = {
            context.context_id: context
            for anchor_id in anchor_ids
            for context in self.index.contexts_for(anchor_id)
        }
        contexts = tuple(contexts_by_id[key] for key in sorted(contexts_by_id))
        if request.context_type:
            expected = request.context_type.casefold()
            contexts = tuple(item for item in contexts if item.context_type.casefold() == expected)
        if request.context_value:
            expected = request.context_value.casefold()
            contexts = tuple(item for item in contexts if item.value.casefold() == expected)

        context_ids = {item.context_id for item in contexts}
        assignments = tuple(
            assignment for assignment in integration.context_assignments
            if assignment.context_id in context_ids
        )
        anchor_set = set(anchor_ids)
        candidate_ids = {
            assignment.target_id for assignment in assignments
            if assignment.target_id not in anchor_set
        }
        candidates = []
        for candidate_id in sorted(candidate_ids):
            item = self._item_by_id(candidate_id)
            if item is not None:
                candidates.append(self._candidate(item))
        candidates = candidates[:request.max_results]
        evidence = tuple(
            item for candidate in candidates
            for item in self.index.evidence_for(candidate.record_id)
        ) + tuple(
            item for anchor_id in anchor_ids
            for item in self.index.evidence_for(anchor_id)
        )
        return self._result(
            request, candidates=tuple(candidates), relations=(),
            contexts=contexts, evidence=evidence,
            exact_match=bool(contexts),
        )

    def _result(
        self,
        request: USGKnowledgeQuery,
        *,
        candidates: tuple[USGKnowledgeCandidate, ...],
        relations: tuple[Any, ...],
        contexts: tuple[Any, ...],
        evidence: tuple[Any, ...],
        exact_match: bool,
    ) -> USGKnowledgeQueryResult:
        evidence_by_id = {
            item.evidence_id: item for item in evidence
            if getattr(item, "evidence_id", None)
        }
        source_ids = {item.source_id for item in evidence if getattr(item, "source_id", None)}
        for candidate in candidates:
            source_ids.update(candidate.source_ids)
        sources = tuple(
            source for source_id in sorted(source_ids)
            if (source := self.index.source_provenance_by_id(source_id)) is not None
        )
        statuses = {
            candidate.assertion_status for candidate in candidates
        } | {
            item.support_status for item in evidence
            if getattr(item, "support_status", None)
        }
        uncertainty = bool(statuses & _UNCERTAIN)
        unique_evidence = tuple(evidence_by_id[key] for key in sorted(evidence_by_id))
        unique_contexts = {
            item.context_id: item for item in contexts
            if getattr(item, "context_id", None)
        }
        return USGKnowledgeQueryResult(
            mode=request.mode,
            query=request.query,
            record_id=request.record_id,
            candidates=candidates,
            relations=relations,
            contexts=tuple(unique_contexts[key] for key in sorted(unique_contexts)),
            evidence=unique_evidence,
            source_provenance=sources,
            available_domains=self.available_domains,
            unavailable_domains=self.unavailable_domains,
            bundle_id=self.resource.bundle_id,
            bundle_digest=self.resource.bundle_digest,
            exact_match=exact_match,
            uncertainty_present=uncertainty,
            confidence_available=any(item.confidence_value is not None for item in candidates),
        )

    def _item_by_id(self, record_id: str) -> USGRetrievedItem | None:
        # The exact index intentionally exposes name lookup only; this fallback
        # resolves IDs by scanning the immutable typed resources.
        for entry in self.resource.lexical.entries:
            if entry.entry_id == record_id:
                return USGRetrievedItem(
                    "LEXICAL_ENTRY", entry.entry_id, entry.orthography, entry.language,
                    entry.source_reference,
                )
        semantic = self.resource.semantic_knowledge
        concepts = semantic.concepts if semantic is not None else self.resource.semantic.concepts
        for concept in concepts:
            if concept.concept_id == record_id:
                record = next(
                    (item for item in semantic.concept_records if item.concept.concept_id == record_id),
                    None,
                ) if semantic is not None else None
                return USGRetrievedItem(
                    "SEMANTIC_CONCEPT", concept.concept_id, concept.label,
                    concept.language, concept.source_reference,
                    record.lifecycle_state if record is not None else None,
                )
        categories = semantic.categories if semantic is not None else ()
        for category in categories:
            if category.category_id == record_id:
                return USGRetrievedItem(
                    "SEMANTIC_CATEGORY", category.category_id, category.class_key,
                    "und", category.source_reference, category.lifecycle_state,
                )
        frames = semantic.predicate_frames if semantic is not None else ()
        for frame in frames:
            if frame.frame_id == record_id:
                return USGRetrievedItem(
                    "PREDICATE_FRAME", frame.frame_id, frame.frame_key, "und",
                    frame.source_reference, frame.lifecycle_state,
                )
        for item in self.resource.structures.classes:
            if item.class_id == record_id:
                return USGRetrievedItem(
                    "SEMANTIC_CLASS", item.class_id, item.class_key, "und",
                    item.source_reference, item.lifecycle_state,
                )
        for item in self.resource.structures.predicate_frames:
            if item.frame_id == record_id:
                return USGRetrievedItem(
                    "PREDICATE_FRAME", item.frame_id, item.frame_key, "und",
                    item.source_reference, item.lifecycle_state,
                )
        return None

    def _candidate(
        self,
        item: USGRetrievedItem,
        *,
        assertion_status: str | None = None,
        evidence_ids: tuple[str, ...] = (),
        context_ids: tuple[str, ...] = (),
        relationship_id: str | None = None,
        relationship_predicate: str | None = None,
        relationship_direction: str | None = None,
    ) -> USGKnowledgeCandidate:
        source_reference = item.source_reference
        lifecycle_state = item.lifecycle_state
        source_ids: tuple[str, ...] = ()
        semantic = self.resource.semantic_knowledge
        if item.kind == "SEMANTIC_CONCEPT" and semantic is not None:
            record = next(
                (value for value in semantic.concept_records if value.concept.concept_id == item.record_id),
                None,
            )
            if record is not None:
                lifecycle_state = record.lifecycle_state
                evidence_ids = evidence_ids or record.evidence_ids
                context_ids = context_ids or record.context_ids
                source_ids = record.source_ids
                assertion_status = assertion_status or (
                    "ACTIVE" if record.active else record.lifecycle_state
                )
        if item.kind == "PREDICATE_FRAME" and semantic is not None:
            frame = next((value for value in semantic.predicate_frames if value.frame_id == item.record_id), None)
            if frame is not None:
                lifecycle_state = frame.lifecycle_state
                assertion_status = assertion_status or frame.status
        if item.kind == "LEXICAL_ENTRY":
            entry = next(
                (value for value in self.resource.lexical.entries if value.entry_id == item.record_id),
                None,
            )
            if entry is not None:
                assertion_status = assertion_status or entry.semantic_link_status
                meaning_evidence_ids = {
                    evidence.evidence_id
                    for meaning in entry.meanings
                    for evidence in self.index.evidence_for(meaning.meaning_id)
                }
                evidence_ids = tuple(sorted(set(evidence_ids) | meaning_evidence_ids))
                meaning_context_ids = {
                    context.context_id
                    for meaning in entry.meanings
                    for context in self.index.contexts_for(meaning.meaning_id)
                }
                context_ids = tuple(sorted(set(context_ids) | meaning_context_ids))
        attached = tuple(
            evidence for evidence_id in evidence_ids
            if (evidence := self.index.evidence_by_id(evidence_id)) is not None
        )
        if not context_ids:
            context_ids = tuple(
                value.context_id for value in self.index.contexts_for(item.record_id)
            )
        if not evidence_ids and attached:
            evidence_ids = tuple(value.evidence_id for value in attached)
        statuses = tuple(sorted({
            value.support_status for value in attached
        }))
        return USGKnowledgeCandidate(
            kind=item.kind,
            record_id=item.record_id,
            label=item.label,
            language=item.language,
            source_reference=source_reference,
            lifecycle_state=lifecycle_state,
            assertion_status=assertion_status or lifecycle_state or "UNASSESSED",
            evidence_ids=tuple(sorted(set(evidence_ids))),
            evidence_statuses=statuses,
            context_ids=tuple(sorted(set(context_ids))),
            source_ids=tuple(sorted(set(source_ids) | {value.source_id for value in attached})),
            source_references=(source_reference,) if source_reference else (),
            relationship_id=relationship_id,
            relationship_predicate=relationship_predicate,
            relationship_direction=relationship_direction,
            confidence_value=None,
            confidence_basis="NOT_PROVIDED_BY_SOURCE",
        )


__all__ = [
    "USGKnowledgeQuery", "USGKnowledgeCandidate", "USGKnowledgeQueryResult",
    "USGKnowledgeAccess",
]
