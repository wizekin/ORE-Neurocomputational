"""ORE language-context, lexical-link, morphology, evidence and source contracts.

These records retain source assertions and uncertainty. Evidence is not promoted
to fact, and source licensing restrictions remain attached to every assertion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_LINK_STATUSES = frozenset({
    "LINKED", "AMBIGUOUS", "MISSING", "CONFLICTED", "CONTRADICTED",
    "UNRESOLVED", "CANDIDATE", "NEUTRAL", "UNASSESSED",
})
_EVIDENCE_STATUSES = frozenset({"SUPPORTS", "CONTRADICTS", "NEUTRAL", "UNASSESSED"})


@dataclass(frozen=True, slots=True)
class ContextualRecord:
    """A typed USG context value without converting it to invented numeric features."""

    context_id: str
    context_type: str
    value: str
    language_profile: str | None = None
    scope: str | None = None
    source_record_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.context_id.strip() or not self.context_type.strip() or not self.value.strip():
            raise ValueError("context ID, type and value must be non-empty")
        if len(self.source_record_ids) != len(set(self.source_record_ids)):
            raise ValueError("context source record IDs must be unique")


@dataclass(frozen=True, slots=True)
class ContextAssignment:
    """An explicit context_ids attachment between a context and an ORE record."""

    context_id: str
    target_id: str
    target_kind: str
    source_record_id: str

    def __post_init__(self) -> None:
        if not all((self.context_id, self.target_id, self.target_kind, self.source_record_id)):
            raise ValueError("context assignment identifiers must be non-empty")


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """Source attribution, license and use restrictions retained for ORE consumers."""

    source_id: str
    provider: str
    source_type: str
    rights_status: str
    redistribution: str
    attribution: str
    usable_for_local_import: bool
    license_id: str | None = None
    source_locator: str | None = None
    retrieved_at: str = ""
    content_digest: str | None = None

    @property
    def redistributable(self) -> bool:
        return self.redistribution == "ALLOWED" and self.rights_status == "KNOWN_PERMITTED"


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """An evidence summary, explicitly not a verified fact or independent truth claim."""

    evidence_id: str
    source_id: str
    evidence_type: str
    support_status: str
    locator: str | None
    text_digest: str | None
    captured_at: str
    transformation_ids: tuple[str, ...]
    source_rights_status: str
    source_redistribution: str
    source_license_id: str | None
    usable_for_local_reasoning: bool
    verified_fact: bool = False
    interpretation: str = "EVIDENCE_SUMMARY_NOT_VERIFIED_FACT"

    def __post_init__(self) -> None:
        if not self.evidence_id.strip() or not self.source_id.strip():
            raise ValueError("evidence and source IDs must be non-empty")
        if self.support_status not in _EVIDENCE_STATUSES:
            raise ValueError(f"unsupported evidence support status: {self.support_status}")
        if self.verified_fact:
            raise ValueError("imported evidence summaries cannot be asserted as verified facts")


@dataclass(frozen=True, slots=True)
class EvidenceAttachment:
    """An explicit evidence_ids attachment to a record, relation or mapping."""

    evidence_id: str
    target_id: str
    target_kind: str
    source_record_id: str

    def __post_init__(self) -> None:
        if not all((self.evidence_id, self.target_id, self.target_kind, self.source_record_id)):
            raise ValueError("evidence attachment identifiers must be non-empty")


@dataclass(frozen=True, slots=True)
class LexicalConceptLink:
    """One source-backed lexical-entry/meaning to semantic-concept edge."""

    link_id: str
    lexical_entry_id: str
    lexical_meaning_id: str | None
    semantic_concept_id: str | None
    source_lexical_record_id: str
    source_concept_record_id: str | None
    predicate: str
    relationship_kind: str
    direction: str
    status: str
    lifecycle_state: str
    evidence_ids: tuple[str, ...]
    evidence_statuses: tuple[str, ...]
    context_ids: tuple[str, ...]
    source_reference: str
    source_kind: str
    source_subject_record_id: str | None = None
    source_object_record_id: str | None = None

    def __post_init__(self) -> None:
        if not self.link_id.strip() or not self.lexical_entry_id.strip():
            raise ValueError("lexical concept link identifiers must be non-empty")
        if self.status not in _LINK_STATUSES:
            raise ValueError(f"unsupported lexical-concept link status: {self.status}")
        if self.direction != "DIRECTED":
            raise ValueError("lexical-concept links preserve directed source orientation")
        if self.semantic_concept_id is None and self.status != "MISSING":
            raise ValueError("a link without a concept target must have MISSING status")


@dataclass(frozen=True, slots=True)
class LexicalConceptLinkGroup:
    """A group of candidate concepts for one lexical entry or one lexical meaning."""

    group_id: str
    lexical_entry_id: str
    lexical_meaning_id: str | None
    scope: str
    candidate_concept_ids: tuple[str, ...]
    link_ids: tuple[str, ...]
    status: str

    def __post_init__(self) -> None:
        if self.scope not in {"ENTRY", "MEANING"}:
            raise ValueError("link group scope must be ENTRY or MEANING")
        if self.status not in {"LINKED", "AMBIGUOUS", "MISSING", "CONFLICTED", "UNRESOLVED"}:
            raise ValueError("unsupported lexical concept link group status")
        if len(self.candidate_concept_ids) != len(set(self.candidate_concept_ids)):
            raise ValueError("candidate concept IDs must be unique")


@dataclass(frozen=True, slots=True)
class MorphologicalFormLink:
    """An explicitly declared surface-form-to-lemma relationship or unresolved gap."""

    link_id: str
    form_entry_id: str
    lemma_entry_id: str | None
    form_record_id: str
    lemma_record_id: str | None
    predicate: str | None
    direction: str
    status: str
    lifecycle_state: str
    language: str
    script: str | None
    grammatical_information_json: str
    evidence_ids: tuple[str, ...]
    evidence_statuses: tuple[str, ...]
    context_ids: tuple[str, ...]
    source_reference: str
    source_kind: str
    source_subject_record_id: str | None = None
    source_object_record_id: str | None = None

    def __post_init__(self) -> None:
        if not self.link_id.strip() or not self.form_entry_id.strip():
            raise ValueError("morphological link identifiers must be non-empty")
        if self.status not in {"LINKED", "AMBIGUOUS", "MISSING", "CONFLICTED", "CONTRADICTED", "UNRESOLVED", "CANDIDATE", "NEUTRAL", "UNASSESSED"}:
            raise ValueError("unsupported morphological link status")
        if self.direction not in {"FORM_TO_LEMMA", "LEMMA_TO_FORM", "UNRESOLVED"}:
            raise ValueError("unsupported morphological link direction")
        if self.lemma_entry_id is None and self.status == "LINKED":
            raise ValueError("a linked morphology record requires a lemma entry")


__all__ = [
    "ContextualRecord", "ContextAssignment", "SourceProvenance", "EvidenceRecord",
    "EvidenceAttachment", "LexicalConceptLink", "LexicalConceptLinkGroup",
    "MorphologicalFormLink",
]
