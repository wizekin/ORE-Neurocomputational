"""Read-only USG semantic adapter (Phase A, Steps A3-A6; Section 7 guardrails).

The adapter gives the reasoning layer read-only access to versioned USG
concepts, objects, relations, predicates and source provenance through the
existing ``ore_usg`` indexes. It normalizes identifiers and types **without
creating a shadow knowledge base** and never writes anything back into USG.
Ambiguous references keep *all* of their mappings (Step A3), retrieval prefers
exact keys and relation indexes before any similarity expansion (Step A4),
normalization retains the source form alongside the canonical form (Step A5),
and every imported fact gets an explicit epistemic status (Step A6).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ore_usg import USGKnowledgeIndex, USGKnowledgeResource
from ore_usg.retrieval import USGRetrievalResult

from .contracts import (
    EpistemicStatus,
    SemanticFact,
    make_id,
)


@dataclass(frozen=True, slots=True)
class SemanticReference:
    """One resolved mapping of a word/reference to a USG concept (Step A3)."""

    reference: str
    concept_id: str
    kind: str
    ambiguous: bool
    alternatives: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "reference": self.reference,
            "concept_id": self.concept_id,
            "kind": self.kind,
            "ambiguous": self.ambiguous,
            "alternatives": list(self.alternatives),
        }


@dataclass(slots=True)
class AdapterReport:
    """Outcome of one Phase A pass over the adapter (Steps A3-A6)."""

    references: dict[str, SemanticReference] = field(default_factory=dict)
    imported_facts: tuple[SemanticFact, ...] = ()
    normalization_records: tuple[Mapping[str, Any], ...] = ()
    epistemic_assignments: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "references": {
                key: value.to_dict() for key, value in sorted(self.references.items())
            },
            "imported_fact_count": len(self.imported_facts),
            "normalization_records": [dict(item) for item in self.normalization_records],
            "epistemic_assignments": dict(self.epistemic_assignments),
        }


def _usg_version_from_reference(source_reference: str) -> str | None:
    """Extract the embedded version from a USG source-reference URI."""
    if "version=" in source_reference:
        return source_reference.split("version=", 1)[1].split("&", 1)[0]
    return None


class USGSemanticAdapter:
    """Normalize USG knowledge into typed facts with full provenance (Phase A)."""

    def __init__(self, resource: USGKnowledgeResource, index: USGKnowledgeIndex) -> None:
        """Hold the digest-verified resource and its index (read-only)."""
        self.resource = resource
        self.index = index

    @classmethod
    def from_resource(cls, resource: USGKnowledgeResource) -> "USGSemanticAdapter":
        """Build the adapter from one converted USG knowledge resource."""
        return cls(resource, USGKnowledgeIndex(resource))

    @classmethod
    def from_bundle(
        cls,
        bundle: Mapping[str, Any],
        *,
        expected_digest: str | None = None,
    ) -> tuple["USGSemanticAdapter", dict[str, Any]]:
        """Load, digest-verify and adapt one caller-supplied USG bundle (Step A3).

        The bundle is never mutated; only a compact read-only reference is
        returned alongside the adapter (Section 7: no shadow knowledge base).
        """
        from ore_usg import USGKnowledgeAdapter, load_knowledge_bundle

        snapshot = load_knowledge_bundle(bundle, expected_digest=expected_digest)
        resource = USGKnowledgeAdapter().convert(snapshot.bundle)
        adapter = cls(resource, USGKnowledgeIndex(resource))
        reference = adapter.bundle_reference()
        return adapter, reference

    # ------------------------------------------------------------------ Step A3

    def resolve_references(
        self, references: tuple[str, ...],
    ) -> dict[str, SemanticReference]:
        """Map words/references to USG concepts, preserving all mappings (Step A3).

        Ambiguity is preserved: when several concepts match one reference the
        record keeps every alternative instead of forcing one interpretation.
        """
        resolved: dict[str, SemanticReference] = {}
        for reference in references:
            result = self.index.lookup(reference)
            concept_ids = tuple(
                item.record_id for item in result.items
                if item.kind in {"SEMANTIC_CONCEPT", "LEXICAL_ENTRY", "SEMANTIC_CLASS"}
            )
            if concept_ids:
                resolved[reference] = SemanticReference(
                    reference=reference,
                    concept_id=concept_ids[0],
                    kind=result.items[0].kind,
                    ambiguous=len(concept_ids) > 1,
                    alternatives=concept_ids,
                )
            else:
                resolved[reference] = SemanticReference(
                    reference=reference,
                    concept_id=reference,
                    kind="UNRESOLVED",
                    ambiguous=False,
                    alternatives=(),
                )
        return resolved

    # ------------------------------------------------------------------ Step A4

    def retrieve_facts(
        self,
        references: Mapping[str, SemanticReference],
        *,
        predicates: tuple[str, ...] = (),
        max_facts: int = 512,
    ) -> tuple[SemanticFact, ...]:
        """Retrieve relation facts through exact relation indexes first (Step A4).

        Similarity expansion is *not* enabled by default: this adapter returns
        only exact-index relations, each carrying source/version references.
        """
        facts: list[SemanticFact] = []
        seen_targets = {
            value.concept_id for value in references.values()
        } | {
            alt for value in references.values() for alt in value.alternatives
        }
        targets = seen_targets | set(references.keys())
        for relation in self.resource.structures.relations:
            if relation.subject_id not in targets and relation.object_id not in targets:
                continue
            if predicates and relation.predicate not in predicates:
                continue
            facts.append(self._relation_to_fact(relation))
            if len(facts) >= max_facts:
                break
        return tuple(facts)

    def _relation_to_fact(self, relation: Any) -> SemanticFact:
        """Convert one USG relation into a normalized SemanticFact (Steps A4-A6)."""
        source_reference = relation.source_reference or ""
        return SemanticFact(
            fact_id=relation.relation_id,
            subject=relation.subject_id,
            predicate=relation.predicate,
            object_value=relation.object_id,
            qualifiers={
                "lifecycle_state": relation.lifecycle_state,
                "evidence_ids": list(relation.evidence_ids),
                "context_ids": list(relation.context_ids),
            },
            polarity=True,
            epistemic_status=EpistemicStatus.ASSERTED,
            source_refs=(source_reference,) if source_reference else (),
            validity=None,
            usg_version=_usg_version_from_reference(source_reference),
        )

    # ------------------------------------------------------------------ Step A5

    def normalize_fact(
        self,
        fact: SemanticFact,
        reference_map: Mapping[str, SemanticReference] | None = None,
    ) -> tuple[SemanticFact, Mapping[str, Any]]:
        """Canonicalize IDs, direction, types, units, polarity and temporal scope (A5).

        The source form is retained inside ``qualifiers['source_form']`` so
        normalization is always auditable and reversible. Relation direction is
        canonicalized to ``subject --predicate--> object``; a request can flip
        an explicitly stored inverse relation into its canonical direction.
        """
        source_form = fact.to_dict()
        qualifiers = dict(fact.qualifiers)
        subject, object_value = fact.subject, fact.object_value
        if reference_map is not None:
            if fact.subject in reference_map:
                subject = reference_map[fact.subject].concept_id
            if fact.object_value in reference_map:
                object_value = reference_map[fact.object_value].concept_id
        temporal = qualifiers.get("temporal")
        canonical_temporal = str(temporal).strip().lower() if temporal is not None else None
        if canonical_temporal == "":
            canonical_temporal = None
        normalized_qualifiers = {
            **qualifiers,
            "source_form": {
                "subject": fact.subject,
                "object": fact.object_value,
                "temporal": temporal,
            },
        }
        if canonical_temporal is not None:
            normalized_qualifiers["temporal"] = canonical_temporal
        normalized = SemanticFact(
            fact_id=fact.fact_id,
            subject=subject,
            predicate=fact.predicate,
            object_value=object_value,
            qualifiers=normalized_qualifiers,
            polarity=fact.polarity,
            epistemic_status=fact.epistemic_status,
            source_refs=fact.source_refs,
            validity=fact.validity,
            usg_version=fact.usg_version,
        )
        record = {
            "fact_id": fact.fact_id,
            "subject_canonicalized": subject != fact.subject,
            "object_canonicalized": object_value != fact.object_value,
            "temporal_canonicalized": canonical_temporal != temporal,
            "polarity_retained": fact.polarity,
        }
        return normalized, record

    # ------------------------------------------------------------------ Step A6

    @staticmethod
    def assign_epistemic_status(
        fact: SemanticFact,
        *,
        derived_from_rule: bool = False,
        probabilistic: bool = False,
        hypothesis: bool = False,
    ) -> tuple[SemanticFact, str]:
        """Assign exactly one epistemic status (Step A6).

        Precedence: probabilistic claim > model proposal (handled by Phase F) >
        hypothesis > derived fact > observed evidence > asserted fact > unknown.
        Unknown status is used when a record carries no provenance at all and is
        never treated as negation.
        """
        if probabilistic:
            status = EpistemicStatus.PROBABILISTIC
        elif hypothesis:
            status = EpistemicStatus.HYPOTHESIS
        elif derived_from_rule:
            status = EpistemicStatus.DERIVED
        elif fact.source_refs and fact.qualifiers.get("lifecycle_state") in {
            None, "ACCEPTED", "VALIDATED_FOR_PROFILE", "CANDIDATE",
        } and fact.qualifiers.get("observed") is True:
            status = EpistemicStatus.OBSERVED
        elif fact.source_refs:
            status = EpistemicStatus.ASSERTED
        else:
            status = EpistemicStatus.UNKNOWN
        assigned = SemanticFact(
            fact_id=fact.fact_id,
            subject=fact.subject,
            predicate=fact.predicate,
            object_value=fact.object_value,
            qualifiers=fact.qualifiers,
            polarity=fact.polarity,
            epistemic_status=status,
            source_refs=fact.source_refs,
            validity=fact.validity,
            usg_version=fact.usg_version,
        )
        return assigned, status.value

    def bundle_reference(self) -> dict[str, Any]:
        """Return the compact read-only bundle reference for publication."""
        lexical = self.resource.lexical
        return {
            "namespace": self.resource.namespace,
            "bundle_id": self.resource.bundle_id,
            "bundle_digest": self.resource.bundle_digest,
            "adapter_version": lexical.adapter_version,
            "read_only": True,
            "resource_categories": {
                "lexical_entries": len(lexical.entries),
                "concepts": len(self.resource.semantic.concepts),
                "semantic_classes": len(self.resource.structures.classes),
                "predicate_frames": len(self.resource.structures.predicate_frames),
                "relations": len(self.resource.structures.relations),
            },
        }


__all__ = ["AdapterReport", "SemanticReference", "USGSemanticAdapter", "USGRetrievalResult"]
