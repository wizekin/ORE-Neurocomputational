"""Unified, read-only USG knowledge import for ORE consumers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .bundle_io import USGBundleSnapshot, load_knowledge_bundle
from .contracts import USGLexiconResource
from .lexical import USGKnowledgeLexicalAdapter
from .semantic import USGConceptAdapter, USGSemanticResource
from .structures import USGSemanticStructureAdapter, USGStructureResource


@dataclass(frozen=True, slots=True)
class USGKnowledgeResource:
    """One linked snapshot exposing lexical, semantic, and structural USG imports."""

    lexical: USGLexiconResource
    semantic: USGSemanticResource
    structures: USGStructureResource
    namespace: str
    bundle_id: str
    bundle_digest: str

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
        """Return explicit cross-resource relations from the structural adapter."""
        return self.structures.relations

    @property
    def lexical_entries(self) -> tuple[Any, ...]:
        """Expose imported ORE lexical entries without discarding provenance report."""
        return self.lexical.entries

    @property
    def concepts(self) -> tuple[Any, ...]:
        """Expose imported ORE semantic concepts."""
        return self.semantic.concepts


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

    def convert(self, bundle: Mapping[str, Any]) -> USGKnowledgeResource:
        """Convert one decoded USG bundle into a linked immutable ORE snapshot."""
        lexical = self.lexical_adapter.convert(bundle)
        semantic = self.concept_adapter.convert(bundle)
        structures = self.structure_adapter.convert(bundle)
        return USGKnowledgeResource(
            lexical=lexical,
            semantic=semantic,
            structures=structures,
            namespace=lexical.namespace,
            bundle_id=lexical.bundle_id,
            bundle_digest=lexical.bundle_digest,
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
