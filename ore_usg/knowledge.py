"""Unified, read-only USG knowledge import for ORE consumers."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .bundle_io import USGBundleSnapshot, load_knowledge_bundle
from .contracts import USGLexiconResource
from .lexical import USGKnowledgeLexicalAdapter
from .semantic import USGConceptAdapter, USGSemanticResource
from .structures import USGSemanticStructureAdapter, USGStructureResource
from .semantic_knowledge import USGSemanticKnowledgeAdapter, USGSemanticKnowledgeResource
from .language_integration import USGLanguageIntegrationAdapter, USGLanguageIntegrationResource


@dataclass(frozen=True, slots=True)
class USGKnowledgeResource:
    """One linked snapshot exposing lexical, semantic, and structural USG imports."""

    lexical: USGLexiconResource
    semantic: USGSemanticResource
    structures: USGStructureResource
    namespace: str
    bundle_id: str
    bundle_digest: str
    semantic_knowledge: USGSemanticKnowledgeResource | None = None
    language_integration: USGLanguageIntegrationResource | None = None

    def __post_init__(self) -> None:
        """Ensure every subresource came from exactly the same source snapshot."""
        identities = {
            (self.lexical.namespace, self.lexical.bundle_id, self.lexical.bundle_digest),
            (self.semantic.namespace, self.semantic.bundle_id, self.semantic.bundle_digest),
            (self.structures.namespace, self.structures.bundle_id, self.structures.bundle_digest),
        }
        if len(identities) != 1:
            raise ValueError("unified USG resources must share namespace, bundle ID, and digest")
        if (self.namespace, self.bundle_id, self.bundle_digest) not in identities:
            raise ValueError("unified USG identity does not match its component resources")
        expected_identity = (self.namespace, self.bundle_id, self.bundle_digest)
        if self.semantic_knowledge is not None and (
            self.semantic_knowledge.namespace, self.semantic_knowledge.bundle_id,
            self.semantic_knowledge.bundle_digest,
        ) != expected_identity:
            raise ValueError("semantic knowledge does not match the unified USG snapshot")
        if self.language_integration is not None and (
            self.language_integration.namespace, self.language_integration.bundle_id,
            self.language_integration.bundle_digest,
        ) != expected_identity:
            raise ValueError("language integration does not match the unified USG snapshot")

    @property
    def contexts(self) -> tuple[Any, ...]:
        """Return the context summaries preserved by the lexical import report."""
        return self.lexical.report.context_summaries

    @property
    def evidence(self) -> tuple[Any, ...]:
        """Return the evidence summaries preserved by the lexical import report."""
        return self.lexical.report.evidence_summaries

    @property
    def sources(self) -> tuple[Any, ...]:
        """Return the source-rights summaries preserved by the lexical import report."""
        return self.lexical.report.source_summaries

    @property
    def relations(self) -> tuple[Any, ...]:
        """Return the legacy structural relation records for backward compatibility."""
        return self.structures.relations

    @property
    def lexical_entries(self) -> tuple[Any, ...]:
        """Expose imported ORE lexical entries without discarding provenance report."""
        return self.lexical.entries

    @property
    def concepts(self) -> tuple[Any, ...]:
        """Expose the legacy CONCEPT-only semantic inventory."""
        return self.semantic.concepts

    @property
    def semantic_concepts(self) -> tuple[Any, ...]:
        """Expose sense-level and concept-level ORE SemanticConcept records."""
        if self.semantic_knowledge is not None:
            return self.semantic_knowledge.concepts
        return self.semantic.concepts

    @property
    def contextual_records(self) -> tuple[Any, ...]:
        """Return typed ORE context records imported from the USG bundle."""
        return self.language_integration.contexts if self.language_integration is not None else ()

    @property
    def lexical_concept_links(self) -> tuple[Any, ...]:
        return self.language_integration.lexical_concept_links if self.language_integration is not None else ()

    @property
    def lexical_concept_groups(self) -> tuple[Any, ...]:
        return self.language_integration.lexical_concept_groups if self.language_integration is not None else ()

    @property
    def morphology_links(self) -> tuple[Any, ...]:
        return self.language_integration.morphology_links if self.language_integration is not None else ()

    @property
    def evidence_records(self) -> tuple[Any, ...]:
        return self.language_integration.evidence if self.language_integration is not None else ()

    @property
    def source_provenance(self) -> tuple[Any, ...]:
        return self.language_integration.sources if self.language_integration is not None else ()

    def contexts_for(self, record_id: str) -> tuple[Any, ...]:
        """Get explicit context records attached to a lexical, semantic or USG record ID."""
        if self.language_integration is None:
            return ()
        return self.language_integration.contexts_for(record_id)

    def evidence_for(self, record_id: str) -> tuple[Any, ...]:
        """Get source-backed evidence summaries attached to an exact record ID."""
        if self.language_integration is None:
            return ()
        return self.language_integration.evidence_for(record_id)


class USGKnowledgeAdapter:
    """Coordinate all current USG-to-ORE read-only adapters over one bundle."""

    def __init__(
        self,
        *,
        include_candidate_states: bool = False,
    ) -> None:
        """Create coordinated adapters with one consistent lifecycle policy."""
        self.lexical_adapter = USGKnowledgeLexicalAdapter(
            include_candidate_states=include_candidate_states,
        )
        self.concept_adapter = USGConceptAdapter(
            include_candidate_states=include_candidate_states,
        )
        self.structure_adapter = USGSemanticStructureAdapter(
            include_candidate_states=include_candidate_states,
        )
        self.semantic_knowledge_adapter = USGSemanticKnowledgeAdapter(
            include_candidate_states=include_candidate_states,
        )
        self.language_integration_adapter = USGLanguageIntegrationAdapter(
            include_candidate_states=include_candidate_states,
        )

    def convert(self, bundle: Mapping[str, Any]) -> USGKnowledgeResource:
        """Convert one decoded USG bundle into a linked immutable ORE snapshot."""
        lexical = self.lexical_adapter.convert(bundle)
        semantic = self.concept_adapter.convert(bundle)
        structures = self.structure_adapter.convert(bundle)
        semantic_knowledge = self.semantic_knowledge_adapter.convert(bundle)
        language_integration = self.language_integration_adapter.convert(
            bundle, lexical, semantic_knowledge,
        )

        # Project the separate Phase 3 link records onto the ORE lexical entries
        # and meanings so existing lexical processors can inspect the explicit IDs.
        links_by_entry: dict[str, list[Any]] = {}
        links_by_meaning: dict[str, list[Any]] = {}
        for link in language_integration.lexical_concept_links:
            links_by_entry.setdefault(link.lexical_entry_id, []).append(link)
            if link.lexical_meaning_id:
                links_by_meaning.setdefault(link.lexical_meaning_id, []).append(link)
        groups_by_entry = {
            item.lexical_entry_id: item
            for item in language_integration.lexical_concept_groups if item.scope == "ENTRY"
        }
        morphology_by_entry: dict[str, list[Any]] = {}
        for link in language_integration.morphology_links:
            morphology_by_entry.setdefault(link.form_entry_id, []).append(link)

        def link_status(rows: list[Any], concept_ids: tuple[str, ...]) -> str:
            statuses = {item.status for item in rows}
            if "CONFLICTED" in statuses:
                return "CONFLICTED"
            if "CONTRADICTED" in statuses:
                return "CONTRADICTED"
            if len(concept_ids) > 1:
                return "AMBIGUOUS"
            for status in ("UNRESOLVED", "CANDIDATE", "NEUTRAL", "UNASSESSED"):
                if status in statuses:
                    return status
            return "LINKED" if concept_ids else ("MISSING" if "MISSING" in statuses else "UNLINKED")

        enriched_entries = []
        for entry in lexical.entries:
            entry_links = links_by_entry.get(entry.entry_id, [])
            concept_ids = tuple(sorted({
                item.semantic_concept_id for item in entry_links
                if item.semantic_concept_id is not None
            }))
            entry_group = groups_by_entry.get(entry.entry_id)
            entry_status = entry_group.status if entry_group is not None else link_status(entry_links, concept_ids)
            enriched_meanings = []
            for meaning in entry.meanings:
                meaning_links = links_by_meaning.get(meaning.meaning_id, [])
                meaning_concept_ids = tuple(sorted({
                    item.semantic_concept_id for item in meaning_links
                    if item.semantic_concept_id is not None
                }))
                enriched_meanings.append(replace(
                    meaning,
                    semantic_concept_ids=meaning_concept_ids,
                    semantic_link_status=link_status(meaning_links, meaning_concept_ids),
                ))
            morphology = morphology_by_entry.get(entry.entry_id, [])
            lemma_ids = tuple(sorted({
                item.lemma_entry_id for item in morphology if item.lemma_entry_id is not None
            }))
            morphology_status = "NOT_APPLICABLE"
            lemma_entry_id = None
            script = None
            grammatical_information_json = None
            morphology_source_reference = None
            if morphology:
                morphology_statuses = {item.status for item in morphology}
                if "CONFLICTED" in morphology_statuses:
                    morphology_status = "CONFLICTED"
                elif "CONTRADICTED" in morphology_statuses:
                    morphology_status = "CONTRADICTED"
                elif "AMBIGUOUS" in morphology_statuses or len(lemma_ids) > 1:
                    morphology_status = "AMBIGUOUS"
                elif "UNRESOLVED" in morphology_statuses:
                    morphology_status = "UNRESOLVED"
                elif "CANDIDATE" in morphology_statuses:
                    morphology_status = "CANDIDATE"
                elif "NEUTRAL" in morphology_statuses:
                    morphology_status = "NEUTRAL"
                elif "UNASSESSED" in morphology_statuses:
                    morphology_status = "UNASSESSED"
                elif lemma_ids:
                    morphology_status = "LINKED"
                else:
                    morphology_status = "MISSING"
                if morphology_status == "LINKED" and len(lemma_ids) == 1:
                    lemma_entry_id = lemma_ids[0]
                first_morphology = morphology[0]
                script = first_morphology.script
                grammatical_information_json = first_morphology.grammatical_information_json
                morphology_source_reference = first_morphology.source_reference
            enriched_entries.append(replace(
                entry,
                meanings=tuple(enriched_meanings),
                semantic_concept_ids=concept_ids,
                semantic_link_status=entry_status,
                lemma_entry_id=lemma_entry_id,
                morphology_link_status=morphology_status,
                script=script,
                grammatical_information_json=grammatical_information_json,
                morphology_source_reference=morphology_source_reference,
            ))
        lexical = replace(lexical, entries=tuple(enriched_entries))

        return USGKnowledgeResource(
            lexical=lexical,
            semantic=semantic,
            structures=structures,
            namespace=lexical.namespace,
            bundle_id=lexical.bundle_id,
            bundle_digest=lexical.bundle_digest,
            semantic_knowledge=semantic_knowledge,
            language_integration=language_integration,
        )

    def load_and_convert(
        self,
        source: str | Path | Mapping[str, Any],
        *,
        expected_digest: str | None = None,
        max_bytes: int = 256 * 1024 * 1024,
    ) -> tuple[USGBundleSnapshot, USGKnowledgeResource]:
        """Load a bundle through the hardened loader, then build one ORE resource."""
        snapshot = load_knowledge_bundle(
            source,
            expected_digest=expected_digest,
            max_bytes=max_bytes,
        )
        return snapshot, self.convert(snapshot.bundle)


__all__ = ["USGKnowledgeResource", "USGKnowledgeAdapter"]
