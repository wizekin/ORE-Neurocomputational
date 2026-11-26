"""Read-only, demand-routed USG knowledge and feedback for the Main Brain.

The coordinator keeps one digest-verified immutable snapshot per active bundle.
Specialists receive only the lexical/semantic/contextual records routed to their
declared needs. Feedback is a separate ORE-side channel and never mutates USG.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import threading
import unicodedata
from typing import Any, Iterable, Mapping

from ore_contracts import ContractEnvelope, EvidenceKind
from ore_contracts.module_artifacts import ContractArtifact, content_hash
from ore_usg import USGKnowledgeAccess, USGKnowledgeAdapter, USGKnowledgeIndex, load_knowledge_bundle
from ore_usg.contracts import (
    ORE_USG_KNOWLEDGE_CONTRACT,
    ORE_USG_LEXICON_CONTRACT,
)
from ore_workspace.models import Provenance, utc_now

from .contracts import COGNITIVE_CONTROL_PRODUCER, USGSnapshotReference, sha256_json
from .bridge import BLACKBOARD_USG_MODULE_KNOWLEDGE_CONTRACT, BLACKBOARD_CONTRACT_VERSION


@dataclass(frozen=True, slots=True)
class USGKnowledgeFeedback:
    """One ORE-side request for missing, ambiguous or conflicting source knowledge."""

    feedback_id: str
    source_module: str
    target_module: str
    feedback_kind: str
    query: str | None
    record_id: str | None
    reason: str
    source_finding_id: str | None
    workspace_revision: int
    bundle_id: str
    bundle_digest: str
    status: str = "OPEN"

    def __post_init__(self) -> None:
        if not all((self.feedback_id, self.source_module, self.target_module, self.feedback_kind)):
            raise ValueError("feedback identity, modules and kind must be non-empty")
        if not self.reason.strip() or self.workspace_revision < 0:
            raise ValueError("feedback reason and non-negative revision are required")
        if self.status not in {"OPEN", "CONSUMED", "RESOLVED", "SUPERSEDED"}:
            raise ValueError("unsupported USG feedback lifecycle state")

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "source_module": self.source_module,
            "target_module": self.target_module,
            "feedback_kind": self.feedback_kind,
            "query": self.query,
            "record_id": self.record_id,
            "reason": self.reason,
            "source_finding_id": self.source_finding_id,
            "workspace_revision": self.workspace_revision,
            "bundle_id": self.bundle_id,
            "bundle_digest": self.bundle_digest,
            "status": self.status,
            "source_mutation_performed": False,
        }


@dataclass(frozen=True, slots=True)
class USGModuleKnowledge:
    """Bounded module-specific projection of one USG snapshot."""

    module_id: str
    domains: tuple[str, ...]
    requested_terms: tuple[str, ...]
    predicate_terms: tuple[str, ...]
    bundle_id: str
    bundle_digest: str
    resource_reference: str
    lexical_entries: tuple[Any, ...] = ()
    semantic_concepts: tuple[Any, ...] = ()
    semantic_categories: tuple[Any, ...] = ()
    relations: tuple[Any, ...] = ()
    memberships: tuple[Any, ...] = ()
    lexical_concept_links: tuple[Any, ...] = ()
    morphology_links: tuple[Any, ...] = ()
    predicate_frames: tuple[Any, ...] = ()
    contexts: tuple[Any, ...] = ()
    context_assignments: tuple[Any, ...] = ()
    evidence: tuple[Any, ...] = ()
    evidence_attachments: tuple[Any, ...] = ()
    source_provenance: tuple[Any, ...] = ()
    feedback: tuple[USGKnowledgeFeedback, ...] = ()
    evidence_by_record: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    context_by_record: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0

    def __post_init__(self) -> None:
        if not self.module_id.strip() or not self.bundle_id.strip():
            raise ValueError("module_id and bundle_id must be non-empty")
        if len(self.bundle_digest) != 64:
            raise ValueError("bundle_digest must be a SHA-256 digest")
        if self.cache_hits < 0 or self.cache_misses < 0:
            raise ValueError("cache counters cannot be negative")
        if len(self.requested_terms) != len(set(self.requested_terms)):
            raise ValueError("requested terms must be unique")

    @property
    def record_count(self) -> int:
        return (
            len(self.lexical_entries) + len(self.semantic_concepts)
            + len(self.semantic_categories) + len(self.predicate_frames)
        )


class BlackboardUSGRouter:
    """Thread-safe, cached, one-hop router over the read-only Phase 4 access API."""

    MAX_CACHE_ENTRIES = 2048

    _IGNORED_TERMS = frozenset({
        "a", "an", "the", "and", "or", "but", "if", "to", "of", "in", "on",
        "at", "by", "for", "from", "with", "is", "are", "was", "were", "be",
        "been", "being", "he", "she", "it", "they", "we", "i", "you",
    })

    def __init__(self, access: USGKnowledgeAccess) -> None:
        if not isinstance(access, USGKnowledgeAccess):
            raise TypeError("access must be a USGKnowledgeAccess")
        self.access = access
        self._lock = threading.RLock()
        self._entry_cache: dict[tuple[str, str | None], tuple[Any, ...]] = {}
        self._frame_cache: dict[str, tuple[Any, ...]] = {}
        self._relation_cache: dict[str, tuple[tuple[Any, ...], tuple[Any, ...]]] = {}
        self._feedback: dict[str, USGKnowledgeFeedback] = {}
        self._consumed_feedback: set[tuple[str, str, str]] = set()
        self._published_feedback: set[tuple[str, str]] = set()
        self._route_cache: dict[tuple[Any, ...], USGModuleKnowledge] = {}
        # Snapshot-scoped exact indexes: build lazily once, reuse across all tasks
        # sharing this digest-verified snapshot rather than scanning the inventories
        # for every new query.
        self._lexical_form_index: dict[str, tuple[Any, ...]] | None = None
        self._semantic_alias_index: dict[str, tuple[Any, ...]] | None = None
        self._predicate_frame_index: dict[str, tuple[Any, ...]] | None = None
        self._relations_by_record: dict[str, tuple[Any, ...]] | None = None
        self._memberships_by_record: dict[str, tuple[Any, ...]] | None = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.route_calls = 0

    @staticmethod
    def _normalise(value: str) -> str:
        return unicodedata.normalize("NFC", str(value)).strip().casefold()

    def _ensure_lexical_index(self) -> None:
        """Build the exact surface-form index once for this immutable snapshot."""
        with self._lock:
            if self._lexical_form_index is not None:
                return
            rows: dict[str, list[Any]] = {}
            for entry in self.access.resource.lexical.entries:
                rows.setdefault(self._normalise(entry.orthography), []).append(entry)
            self._lexical_form_index = {
                key: tuple(sorted(values, key=lambda item: item.entry_id))
                for key, values in rows.items()
            }

    def _ensure_semantic_index(self) -> None:
        """Build a one-time label/alias index without merging equal labels."""
        with self._lock:
            if self._semantic_alias_index is not None:
                return
            rows: dict[str, dict[str, Any]] = {}
            for concept in self.access.resource.semantic_concepts:
                for label in (concept.label, *concept.aliases):
                    rows.setdefault(self._normalise(label), {})[concept.concept_id] = concept
            self._semantic_alias_index = {
                key: tuple(values[item] for item in sorted(values))
                for key, values in rows.items()
            }

    def _ensure_frame_index(self) -> None:
        """Build an exact predicate-key index once, retaining every source version."""
        with self._lock:
            if self._predicate_frame_index is not None:
                return
            rows: dict[str, dict[str, Any]] = {}
            for frame in self.access.resource.predicate_frames:
                frame_id = str(getattr(frame, "frame_id", ""))
                frame_key = self._normalise(getattr(frame, "frame_key", ""))
                if frame_id and frame_key:
                    rows.setdefault(frame_key, {})[frame_id] = frame
            self._predicate_frame_index = {
                key: tuple(values[item] for item in sorted(values))
                for key, values in rows.items()
            }

    def _ensure_related_index(self) -> None:
        """Build one-hop incident-edge indexes once without inferring new edges."""
        with self._lock:
            if self._relations_by_record is not None and self._memberships_by_record is not None:
                return
            relation_rows: dict[str, list[Any]] = {}
            membership_rows: dict[str, list[Any]] = {}
            for relation in self.access.resource.semantic_relations:
                relation_rows.setdefault(relation.subject_id, []).append(relation)
                if relation.object_id != relation.subject_id:
                    relation_rows.setdefault(relation.object_id, []).append(relation)
            for membership in self.access.resource.semantic_memberships:
                membership_rows.setdefault(membership.member_id, []).append(membership)
                if membership.category_id != membership.member_id:
                    membership_rows.setdefault(membership.category_id, []).append(membership)
            self._relations_by_record = {
                key: tuple(sorted(values, key=lambda item: item.relation_id))
                for key, values in relation_rows.items()
            }
            self._memberships_by_record = {
                key: tuple(sorted(values, key=lambda item: item.membership_id))
                for key, values in membership_rows.items()
            }

    def _semantic_concepts(self, term: str, language: str | None) -> tuple[Any, ...]:
        self._ensure_semantic_index()
        assert self._semantic_alias_index is not None
        concepts = self._semantic_alias_index.get(self._normalise(term), ())
        if language is None:
            return concepts
        return tuple(item for item in concepts if item.language.casefold() == language.casefold())

    def _entries(self, term: str, language: str | None) -> tuple[tuple[Any, ...], bool]:
        key = (self._normalise(term), language.casefold() if language else None)
        with self._lock:
            if key in self._entry_cache:
                self.cache_hits += 1
                return self._entry_cache[key], True
        self._ensure_lexical_index()
        assert self._lexical_form_index is not None
        entries = self._lexical_form_index.get(key[0], ())
        if language is not None:
            entries = tuple(item for item in entries if item.language.casefold() == key[1])
        with self._lock:
            existing = self._entry_cache.setdefault(key, tuple(entries))
            if len(self._entry_cache) > self.MAX_CACHE_ENTRIES:
                self._entry_cache.pop(next(iter(self._entry_cache)))
            self.cache_misses += 1
        return existing, False

    def _frames(self, term: str) -> tuple[tuple[Any, ...], bool]:
        key = self._normalise(term)
        with self._lock:
            if key in self._frame_cache:
                self.cache_hits += 1
                return self._frame_cache[key], True
        self._ensure_frame_index()
        assert self._predicate_frame_index is not None
        frames = self._predicate_frame_index.get(key, ())
        with self._lock:
            existing = self._frame_cache.setdefault(key, frames)
            if len(self._frame_cache) > self.MAX_CACHE_ENTRIES:
                self._frame_cache.pop(next(iter(self._frame_cache)))
            self.cache_misses += 1
        return existing, False

    def _related(self, concept_id: str) -> tuple[tuple[Any, ...], tuple[Any, ...], bool]:
        with self._lock:
            if concept_id in self._relation_cache:
                self.cache_hits += 1
                relations, memberships = self._relation_cache[concept_id]
                return relations, memberships, True
        self._ensure_related_index()
        assert self._relations_by_record is not None
        assert self._memberships_by_record is not None
        relations = self._relations_by_record.get(concept_id, ())
        memberships = self._memberships_by_record.get(concept_id, ())
        with self._lock:
            existing = self._relation_cache.setdefault(concept_id, (relations, memberships))
            if len(self._relation_cache) > self.MAX_CACHE_ENTRIES:
                self._relation_cache.pop(next(iter(self._relation_cache)))
            self.cache_misses += 1
        return existing[0], existing[1], False

    def record_feedback(
        self,
        *,
        source_module: str,
        target_module: str,
        feedback_kind: str,
        reason: str,
        query: str | None = None,
        record_id: str | None = None,
        source_finding_id: str | None = None,
        workspace_revision: int = 0,
    ) -> tuple[USGKnowledgeFeedback, bool]:
        """Append one deduplicated feedback request outside the imported USG data."""
        payload = {
            "source_module": source_module,
            "target_module": target_module,
            "feedback_kind": feedback_kind,
            "query": query,
            "record_id": record_id,
            "reason": reason,
            "source_finding_id": source_finding_id,
            "bundle_id": self.access.resource.bundle_id,
            "bundle_digest": self.access.resource.bundle_digest,
        }
        # Same source/target/query/reason is one outstanding retrieval request,
        # even when multiple downstream findings report it in different rounds.
        identity_payload = {
            key: value for key, value in payload.items()
            if key != "source_finding_id"
        }
        feedback_id = sha256_json(identity_payload)
        feedback = USGKnowledgeFeedback(
            feedback_id=feedback_id,
            source_module=source_module,
            target_module=target_module,
            feedback_kind=feedback_kind,
            query=query,
            record_id=record_id,
            reason=reason,
            source_finding_id=source_finding_id,
            workspace_revision=workspace_revision,
            bundle_id=self.access.resource.bundle_id,
            bundle_digest=self.access.resource.bundle_digest,
        )
        with self._lock:
            created = feedback_id not in self._feedback
            if created:
                self._feedback[feedback_id] = feedback
        return feedback, created

    def record_finding_feedback(
        self,
        *,
        source_module: str,
        finding_id: str,
        workspace_revision: int,
        payload: Mapping[str, Any],
    ) -> tuple[USGKnowledgeFeedback, ...]:
        """Convert explicit specialist requests into detached feedback records."""
        requests = payload.get("usg_feedback_requests", ())
        if not isinstance(requests, (tuple, list)):
            return ()
        created: list[USGKnowledgeFeedback] = []
        for request in requests:
            if not isinstance(request, Mapping):
                continue
            target = str(request.get("target_module", "ore.blackboard.semantic_specialist"))
            kind = str(request.get("feedback_kind", "AMBIGUOUS_KNOWLEDGE"))
            reason = str(request.get("reason", "downstream module requested additional knowledge"))
            query = request.get("query")
            record_id = request.get("record_id")
            feedback, is_new = self.record_feedback(
                source_module=source_module,
                target_module=target,
                feedback_kind=kind,
                reason=reason,
                query=str(query) if query is not None else None,
                record_id=str(record_id) if record_id is not None else None,
                source_finding_id=finding_id,
                workspace_revision=workspace_revision,
            )
            if is_new:
                created.append(feedback)
        return tuple(created)

    def feedback_for(
        self, module_id: str, *, delivery_scope: str = "default",
    ) -> tuple[USGKnowledgeFeedback, ...]:
        """Return open feedback not yet delivered to this module in this task scope."""
        with self._lock:
            return tuple(
                item for key, item in sorted(self._feedback.items())
                if item.target_module == module_id
                and (module_id, key, delivery_scope) not in self._consumed_feedback
                and item.status == "OPEN"
            )

    def unannounced_feedback(
        self, *, publication_scope: str = "default",
    ) -> tuple[USGKnowledgeFeedback, ...]:
        """Return feedback not yet published as a question in this workspace scope."""
        with self._lock:
            return tuple(
                item for key, item in sorted(self._feedback.items())
                if item.status == "OPEN"
                and (key, publication_scope) not in self._published_feedback
            )

    def resolve_feedback(
        self, feedback_id: str, *, status: str = "RESOLVED",
    ) -> USGKnowledgeFeedback:
        """Close one ORE-side request without changing its source USG records."""
        if status not in {"RESOLVED", "SUPERSEDED"}:
            raise ValueError("feedback status must be RESOLVED or SUPERSEDED")
        with self._lock:
            current = self._feedback.get(feedback_id)
            if current is None:
                raise KeyError(f"unknown USG feedback ID: {feedback_id}")
            updated = replace(current, status=status)
            self._feedback[feedback_id] = updated
            return updated

    def mark_feedback_published(
        self, feedback_id: str, *, publication_scope: str = "default",
    ) -> None:
        with self._lock:
            if feedback_id in self._feedback:
                self._published_feedback.add((feedback_id, publication_scope))

    def route(
        self,
        module_id: str,
        *,
        domains: Iterable[str],
        lexical_terms: Iterable[str] = (),
        predicate_terms: Iterable[str] = (),
        language: str | None = None,
        workspace_revision: int = 0,
        delivery_scope: str = "default",
    ) -> USGModuleKnowledge:
        """Return only exact lexical entries and their explicit one-hop linked knowledge."""
        domain_set = tuple(sorted({str(item).upper() for item in domains}))
        terms = tuple(sorted({
            str(item).strip() for item in lexical_terms
            if isinstance(item, str) and item.strip()
        }, key=lambda item: (self._normalise(item), item)))
        predicates = tuple(sorted({
            str(item).strip() for item in predicate_terms
            if isinstance(item, str) and item.strip()
        }, key=lambda item: (self._normalise(item), item)))
        pending_feedback = self.feedback_for(
            module_id, delivery_scope=delivery_scope,
        )
        route_key = (
            module_id, domain_set, tuple(self._normalise(item) for item in terms),
            tuple(self._normalise(item) for item in predicates), language,
            delivery_scope,
            tuple(item.feedback_id for item in pending_feedback),
        )
        with self._lock:
            cached = self._route_cache.get(route_key)
            if cached is not None:
                self.cache_hits += 1
                self.route_calls += 1
                for item in pending_feedback:
                    self._consumed_feedback.add((
                        module_id, item.feedback_id, delivery_scope,
                    ))
                return cached

        resource = self.access.resource
        entry_map: dict[str, Any] = {}
        frame_map: dict[str, Any] = {}
        concept_map: dict[str, Any] = {}
        relation_map: dict[str, Any] = {}
        membership_map: dict[str, Any] = {}
        link_map: dict[str, Any] = {}
        morphology_map: dict[str, Any] = {}
        context_map: dict[str, Any] = {}
        evidence_map: dict[str, Any] = {}
        attachment_map: dict[tuple[str, str], Any] = {}
        source_map: dict[str, Any] = {}
        evidence_by_record: dict[str, tuple[str, ...]] = {}
        context_by_record: dict[str, tuple[str, ...]] = {}
        before_hits, before_misses = self.cache_hits, self.cache_misses

        if "LEXICAL" in domain_set or "SEMANTIC" in domain_set or "MORPHOLOGY" in domain_set:
            for term in terms:
                entries, _ = self._entries(term, language)
                for entry in entries:
                    entry_map[entry.entry_id] = entry

        if "PREDICATE" in domain_set:
            for term in predicates:
                frames, _ = self._frames(term)
                for frame in frames:
                    frame_map[frame.frame_id] = frame

        semantic = resource.semantic_knowledge
        integration = resource.language_integration
        concept_by_id = {
            item.concept.concept_id: item
            for item in semantic.concept_records
        } if semantic is not None else {}

        if semantic is not None and "SEMANTIC" in domain_set:
            for term in terms:
                for concept in self._semantic_concepts(term, language):
                    concept_map[concept.concept_id] = concept
                    imported = concept_by_id.get(concept.concept_id)
                    if imported is not None:
                        evidence_by_record[concept.concept_id] = tuple(imported.evidence_ids)
                        context_by_record[concept.concept_id] = tuple(imported.context_ids)
                        if "EVIDENCE" in domain_set:
                            for evidence_id in imported.evidence_ids:
                                evidence = self.access.index.evidence_by_id(evidence_id)
                                if evidence is not None:
                                    evidence_map[evidence_id] = evidence
                        if "CONTEXT" in domain_set:
                            for context_id in imported.context_ids:
                                context = self.access.index.context_by_id(context_id)
                                if context is not None:
                                    context_map[context_id] = context
                        for source_id in imported.source_ids:
                            source = self.access.source_provenance_by_id(source_id)
                            if source is not None:
                                source_map[source_id] = source

        for entry in entry_map.values():
            if "MORPHOLOGY" in domain_set:
                for link in self.access.index.morphology_for_form(entry.entry_id):
                    morphology_map[link.link_id] = link
            meaning_ids = tuple(item.meaning_id for item in entry.meanings)
            for meaning in entry.meanings:
                if "EVIDENCE" in domain_set:
                    evidence_by_record[meaning.meaning_id] = tuple(
                        item.evidence_id for item in self.access.index.evidence_for(meaning.meaning_id)
                    )
                if "CONTEXT" in domain_set:
                    context_by_record[meaning.meaning_id] = tuple(
                        item.context_id for item in self.access.index.contexts_for(meaning.meaning_id)
                    )
                if integration is not None and (
                    "SEMANTIC" in domain_set or "RELATIONAL" in domain_set
                ):
                    for link in self.access.index.lexical_concept_links_for(
                        entry.entry_id, meaning_id=meaning.meaning_id,
                    ):
                        link_map[link.link_id] = link
                        if link.semantic_concept_id:
                            record = concept_by_id.get(link.semantic_concept_id)
                            if record is not None:
                                concept_map[link.semantic_concept_id] = record.concept
                                evidence_by_record[link.semantic_concept_id] = tuple(record.evidence_ids)
                                context_by_record[link.semantic_concept_id] = tuple(record.context_ids)
                                if "EVIDENCE" in domain_set:
                                    for evidence_id in record.evidence_ids:
                                        evidence = self.access.index.evidence_by_id(evidence_id)
                                        if evidence is not None:
                                            evidence_map[evidence_id] = evidence
                                if "CONTEXT" in domain_set:
                                    for context_id in record.context_ids:
                                        context = self.access.index.context_by_id(context_id)
                                        if context is not None:
                                            context_map[context_id] = context
                                for source_id in record.source_ids:
                                    source = self.access.source_provenance_by_id(source_id)
                                    if source is not None:
                                        source_map[source_id] = source
                if "CONTEXT" in domain_set:
                    for context in self.access.index.contexts_for(meaning.meaning_id):
                        context_map[context.context_id] = context
                if "EVIDENCE" in domain_set:
                    for evidence in self.access.index.evidence_for(meaning.meaning_id):
                        evidence_map[evidence.evidence_id] = evidence
            if "EVIDENCE" in domain_set:
                evidence_by_record[entry.entry_id] = tuple(
                    item.evidence_id for item in self.access.index.evidence_for(entry.entry_id)
                )
            if "CONTEXT" in domain_set:
                context_by_record[entry.entry_id] = tuple(
                    item.context_id for item in self.access.index.contexts_for(entry.entry_id)
                )
                for context in self.access.index.contexts_for(entry.entry_id):
                    context_map[context.context_id] = context
            if "EVIDENCE" in domain_set:
                for evidence in self.access.index.evidence_for(entry.entry_id):
                    evidence_map[evidence.evidence_id] = evidence

        if semantic is not None and ("RELATIONAL" in domain_set or "SEMANTIC" in domain_set):
            for concept_id in tuple(concept_map):
                relations, memberships, _ = self._related(concept_id)
                for relation in relations:
                    relation_map[relation.relation_id] = relation
                    for endpoint_id in (relation.subject_id, relation.object_id):
                        imported_endpoint = concept_by_id.get(endpoint_id)
                        if imported_endpoint is not None:
                            concept_map[endpoint_id] = imported_endpoint.concept
                            evidence_by_record[endpoint_id] = tuple(imported_endpoint.evidence_ids)
                            context_by_record[endpoint_id] = tuple(imported_endpoint.context_ids)
                            if "EVIDENCE" in domain_set:
                                for evidence_id in imported_endpoint.evidence_ids:
                                    evidence = self.access.index.evidence_by_id(evidence_id)
                                    if evidence is not None:
                                        evidence_map[evidence_id] = evidence
                            if "CONTEXT" in domain_set:
                                for context_id in imported_endpoint.context_ids:
                                    context = self.access.index.context_by_id(context_id)
                                    if context is not None:
                                        context_map[context_id] = context
                            for source_id in imported_endpoint.source_ids:
                                source = self.access.source_provenance_by_id(source_id)
                                if source is not None:
                                    source_map[source_id] = source
                    if "EVIDENCE" in domain_set:
                        for evidence_id in relation.evidence_ids:
                            evidence = self.access.index.evidence_by_id(evidence_id)
                            if evidence is not None:
                                evidence_map[evidence_id] = evidence
                for membership in memberships:
                    membership_map[membership.membership_id] = membership
                    if membership.member_id in concept_by_id:
                        concept_map[membership.member_id] = concept_by_id[membership.member_id].concept
                    if "EVIDENCE" in domain_set:
                        for evidence_id in membership.evidence_ids:
                            evidence = self.access.index.evidence_by_id(evidence_id)
                            if evidence is not None:
                                evidence_map[evidence_id] = evidence

        # Predicate frames retain source IDs even when a role schema is not executable.
        # Resolve their explicit evidence/context attachments without inventing frame roles.
        if semantic is not None:
            frame_records = {item.frame_id: item for item in semantic.predicate_frames}
            for frame_id in tuple(frame_map):
                frame = frame_records.get(frame_id)
                if frame is not None:
                    raw_id = str(getattr(frame, "usg_record_id", ""))
                    if "EVIDENCE" in domain_set:
                        frame_evidence = (
                            tuple(getattr(frame, "evidence_ids", ()))
                            or tuple(item.evidence_id for item in self.access.index.evidence_for(raw_id))
                        )
                        evidence_by_record[frame_id] = frame_evidence
                        for evidence_id in frame_evidence:
                            evidence = self.access.index.evidence_by_id(evidence_id)
                            if evidence is not None:
                                evidence_map[evidence_id] = evidence
                    if "CONTEXT" in domain_set:
                        frame_contexts = (
                            tuple(getattr(frame, "context_ids", ()))
                            or tuple(item.context_id for item in self.access.index.contexts_for(raw_id))
                        )
                        context_by_record[frame_id] = frame_contexts
                        for context_id in frame_contexts:
                            context = self.access.index.context_by_id(context_id)
                            if context is not None:
                                context_map[context_id] = context

        selected_record_ids = set(entry_map) | set(frame_map) | set(concept_map)
        selected_record_ids.update(item.meaning_id for entry in entry_map.values() for item in entry.meanings)
        selected_record_ids.update(relation_map)
        selected_record_ids.update(membership_map)
        selected_record_ids.update(link_map)
        selected_record_ids.update(item.form_entry_id for item in morphology_map.values())
        selected_record_ids.update(
            item.lemma_entry_id for item in morphology_map.values()
            if item.lemma_entry_id is not None
        )

        if integration is not None:
            if "EVIDENCE" in domain_set:
                for attachment in integration.evidence_attachments:
                    if (
                        attachment.evidence_id in evidence_map
                        and attachment.target_id in selected_record_ids
                    ):
                        attachment_map[(attachment.target_kind, attachment.target_id)] = attachment
            if "CONTEXT" in domain_set:
                selected_assignments = tuple(
                    assignment for assignment in integration.context_assignments
                    if (
                        assignment.context_id in context_map
                        and assignment.target_id in selected_record_ids
                    )
                )
                for assignment in selected_assignments:
                    context_map[assignment.context_id] = (
                        self.access.index.context_by_id(assignment.context_id)
                        or context_map[assignment.context_id]
                    )
            for evidence in evidence_map.values():
                source = self.access.source_provenance_by_id(evidence.source_id)
                if source is not None:
                    source_map[source.source_id] = source

        # Missing exact lexical entries remain explicit feedback, never a fabricated match.
        if "LEXICAL" in domain_set:
            for term in terms:
                if len(self._normalise(term)) > 1 and self._normalise(term) not in self._IGNORED_TERMS:
                    if not any(self._normalise(entry.orthography) == self._normalise(term)
                               for entry in entry_map.values()):
                        self.record_feedback(
                            source_module="ore.blackboard.usg_router",
                            target_module=module_id,
                            feedback_kind="MISSING_LEXICAL_ENTRY",
                            query=term,
                            reason="exact USG lexical lookup returned no entry; preserve the token as unknown",
                            workspace_revision=workspace_revision,
                        )

        current_feedback = self.feedback_for(
            module_id, delivery_scope=delivery_scope,
        )
        consumed_feedback = current_feedback
        with self._lock:
            for item in current_feedback:
                self._consumed_feedback.add((
                    module_id, item.feedback_id, delivery_scope,
                ))
            self.route_calls += 1

        result = USGModuleKnowledge(
            module_id=module_id,
            domains=domain_set,
            requested_terms=terms,
            predicate_terms=predicates,
            bundle_id=resource.bundle_id,
            bundle_digest=resource.bundle_digest,
            resource_reference=resource.resource_id,
            lexical_entries=tuple(entry_map[key] for key in sorted(entry_map)),
            semantic_concepts=tuple(concept_map[key] for key in sorted(concept_map)),
            semantic_categories=tuple(
                item for item in (semantic.categories if semantic is not None else ())
                if item.category_id in {
                    membership.category_id for membership in membership_map.values()
                }
            ),
            relations=tuple(relation_map[key] for key in sorted(relation_map)),
            memberships=tuple(membership_map[key] for key in sorted(membership_map)),
            lexical_concept_links=tuple(link_map[key] for key in sorted(link_map)),
            morphology_links=tuple(morphology_map[key] for key in sorted(morphology_map)),
            predicate_frames=tuple(frame_map[key] for key in sorted(frame_map)),
            contexts=tuple(context_map[key] for key in sorted(context_map)),
            context_assignments=tuple(
                item for item in (integration.context_assignments if integration else ())
                if item.context_id in context_map
            ),
            evidence=tuple(evidence_map[key] for key in sorted(evidence_map)),
            evidence_attachments=tuple(
                attachment_map[key] for key in sorted(attachment_map)
            ),
            source_provenance=tuple(source_map[key] for key in sorted(source_map)),
            feedback=consumed_feedback,
            evidence_by_record=evidence_by_record,
            context_by_record=context_by_record,
            cache_hits=self.cache_hits - before_hits,
            cache_misses=self.cache_misses - before_misses,
        )
        # A query that itself created a missing-knowledge feedback record must
        # not cache that feedback as if it were pending on every future replay.
        result_key = (
            route_key + (tuple(item.feedback_id for item in consumed_feedback),)
            if consumed_feedback else route_key
        )
        with self._lock:
            self._route_cache[result_key] = result
            if len(self._route_cache) > self.MAX_CACHE_ENTRIES:
                self._route_cache.pop(next(iter(self._route_cache)))
        return result

    def statistics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cached_lexical_terms": len(self._entry_cache),
                "cached_predicate_terms": len(self._frame_cache),
                "lexical_surface_index_built": self._lexical_form_index is not None,
                "semantic_alias_index_built": self._semantic_alias_index is not None,
                "predicate_frame_index_built": self._predicate_frame_index is not None,
                "related_edge_index_built": (
                    self._relations_by_record is not None
                    and self._memberships_by_record is not None
                ),
                "cached_related_concepts": len(self._relation_cache),
                "cached_module_routes": len(self._route_cache),
                "cache_entry_limit": self.MAX_CACHE_ENTRIES,
                "feedback_count": len(self._feedback),
                "feedback_open": sum(
                    1 for item in self._feedback.values() if item.status == "OPEN"
                ),
                "feedback_pending_default_scope": sum(
                    1 for item in self._feedback.values()
                    if item.status == "OPEN"
                    and (item.target_module, item.feedback_id, "default") not in self._consumed_feedback
                ),
                "route_calls": self.route_calls,
            }


class BlackboardUSGSnapshot:
    """Hold one digest-verified USG snapshot and share routed, read-only views."""

    MAX_RUN_ARTIFACTS = 64

    def __init__(self, bundle: Mapping[str, Any], *, expected_digest: str | None) -> None:
        snapshot = load_knowledge_bundle(bundle, expected_digest=expected_digest)
        self.bundle_snapshot = snapshot
        self.resource = USGKnowledgeAdapter().convert(snapshot.bundle)
        self.access = USGKnowledgeAccess(self.resource)
        self.index = USGKnowledgeIndex(self.resource)
        self.router = BlackboardUSGRouter(self.access)
        self._knowledge_artifacts: dict[str, ContractArtifact] = {}
        lexical = self.resource.lexical
        semantic = self.resource.semantic_knowledge
        self.reference = USGSnapshotReference(
            namespace=self.resource.namespace,
            bundle_id=self.resource.bundle_id,
            bundle_digest=self.resource.bundle_digest,
            adapter_version=lexical.adapter_version,
            resource_categories={
                "lexical_entries": len(lexical.entries),
                "meanings": lexical.report.meanings_created,
                "concepts": len(semantic.concepts) if semantic is not None else len(self.resource.semantic.concepts),
                "semantic_classes": len(semantic.categories) if semantic is not None else len(self.resource.structures.classes),
                "predicate_frames": len(semantic.predicate_frames) if semantic is not None else len(self.resource.structures.predicate_frames),
                "relations": len(semantic.relations) if semantic is not None else len(self.resource.structures.relations),
                "context_records": len(self.resource.contextual_records),
                "evidence_records": len(self.resource.evidence_records),
            },
        )

    def lexicon_artifact(self, run_id: str) -> ContractArtifact:
        """Wrap the old whole-lexicon input for backwards-compatible specialist ports."""
        return ContractArtifact(
            envelope=ContractEnvelope(
                contract_id=ORE_USG_LEXICON_CONTRACT,
                schema_version="1.0.0",
                run_id=run_id,
                producer_id="ore.blackboard.usg_reference",
                config_hash=sha256_json({"artifact": "usg_lexicon"}),
                created_at=utc_now(),
                evidence_kind=EvidenceKind.EXTERNAL_REFERENCE,
                data_hash=self.resource.bundle_digest,
            ),
            payload=self.resource.lexical,
            metadata={
                "usg_bundle_id": self.reference.bundle_id,
                "usg_bundle_digest": self.reference.bundle_digest,
                "read_only": True,
                "source_database_mutation_performed": False,
            },
        )

    def knowledge_artifact(self, run_id: str) -> ContractArtifact:
        """Return a cached identity artifact used only as the lineage parent of routed slices."""
        cached = self._knowledge_artifacts.get(run_id)
        if cached is None:
            cached = ContractArtifact(
                envelope=ContractEnvelope(
                    contract_id=ORE_USG_KNOWLEDGE_CONTRACT,
                    schema_version="1.0.0",
                    run_id=run_id,
                    producer_id="ore.blackboard.usg_reference",
                    config_hash=sha256_json({"artifact": "usg_knowledge", "read_only": True}),
                    created_at=utc_now(),
                    evidence_kind=EvidenceKind.EXTERNAL_REFERENCE,
                    data_hash=self.resource.bundle_digest,
                ),
                payload=self.resource,
                metadata={
                    "usg_bundle_id": self.reference.bundle_id,
                    "usg_bundle_digest": self.reference.bundle_digest,
                    "read_only": True,
                    "source_database_mutation_performed": False,
                },
            )
            self._knowledge_artifacts[run_id] = cached
            if len(self._knowledge_artifacts) > self.MAX_RUN_ARTIFACTS:
                self._knowledge_artifacts.pop(next(iter(self._knowledge_artifacts)))
        return cached

    def module_knowledge_artifact(
        self,
        run_id: str,
        knowledge: USGModuleKnowledge,
    ) -> ContractArtifact:
        """Wrap one bounded specialist projection and link it to the original snapshot."""
        parent = self.knowledge_artifact(run_id)
        return ContractArtifact(
            envelope=ContractEnvelope(
                contract_id=BLACKBOARD_USG_MODULE_KNOWLEDGE_CONTRACT,
                schema_version=BLACKBOARD_CONTRACT_VERSION,
                run_id=run_id,
                producer_id="ore.blackboard.usg_router",
                config_hash=sha256_json({
                    "module_id": knowledge.module_id,
                    "domains": list(knowledge.domains),
                    "requested_terms": list(knowledge.requested_terms),
                    "predicate_terms": list(knowledge.predicate_terms),
                }),
                created_at=utc_now(),
                evidence_kind=EvidenceKind.DERIVED,
                parent_artifact_ids=(parent.payload_hash,),
                data_hash=content_hash(knowledge),
                model_version=BLACKBOARD_CONTRACT_VERSION,
            ),
            payload=knowledge,
            metadata={
                "read_only": True,
                "bundle_id": knowledge.bundle_id,
                "bundle_digest": knowledge.bundle_digest,
                "module_id": knowledge.module_id,
                "domains": list(knowledge.domains),
                "record_count": knowledge.record_count,
                "feedback_count": len(knowledge.feedback),
                "source_database_mutation_performed": False,
            },
        )


def publish_usg_reference(workspace, reference: USGSnapshotReference) -> Any:
    """Publish only a compact bundle identity into the shared workspace."""
    from ore_workspace.publication import FindingPublisher

    publisher = FindingPublisher(workspace, COGNITIVE_CONTROL_PRODUCER)
    finding = publisher.create_finding(
        "artifact_references",
        reference.to_dict(),
        key=f"usg_bundle:{reference.bundle_id}",
        finding_type="usg_knowledge_bundle_reference",
        provenance=Provenance(
            source=f"usg_knowledge_bundle:{reference.bundle_id}",
            rules_applied=("bundle_digest_verification", "read_only_import"),
            evidence=({
                "bundle_id": reference.bundle_id,
                "bundle_digest": reference.bundle_digest,
                "namespace": reference.namespace,
            },),
            notes="USG remains an independently owned, read-only knowledge source.",
        ),
    )
    return workspace.publish(finding)


__all__ = [
    "BlackboardUSGSnapshot",
    "BlackboardUSGRouter",
    "USGKnowledgeFeedback",
    "USGModuleKnowledge",
    "publish_usg_reference",
]
