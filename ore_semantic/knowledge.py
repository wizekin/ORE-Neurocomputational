"""Typed ORE semantic knowledge structures for categories, membership, and relations.

These records model imported knowledge assertions. They do not promote source
claims to verified facts and they never imply missing inverse or transitive links.
"""
from __future__ import annotations

from dataclasses import dataclass


_RELATION_TYPES = frozenset({"EQUIVALENCE", "HIERARCHY", "ASSOCIATION", "UNRESOLVED", "OTHER"})
_MEMBERSHIP_STATES = frozenset({
    "ACTIVE", "CANDIDATE", "UNRESOLVED", "CONFLICTED", "CONTRADICTED", "UNASSESSED",
})


@dataclass(frozen=True, slots=True)
class SemanticCategory:
    """An ORE semantic category derived from an explicit source class record."""

    category_id: str
    class_key: str
    definition: str
    parent_category_ids: tuple[str, ...] = ()
    unresolved_parent_keys: tuple[str, ...] = ()
    lifecycle_state: str = "ACCEPTED"
    source_reference: str = "unspecified"
    payload_json: str = "{}"

    def __post_init__(self) -> None:
        if not self.category_id.strip() or not self.class_key.strip():
            raise ValueError("category_id and class_key must be non-empty")
        if not self.definition.strip():
            raise ValueError("category definition must be non-empty")
        if len(self.parent_category_ids) != len(set(self.parent_category_ids)):
            raise ValueError("parent_category_ids must be unique")
        if self.category_id in self.parent_category_ids:
            raise ValueError("a semantic category cannot be its own parent")


@dataclass(frozen=True, slots=True)
class SemanticCategoryMembership:
    """A source-backed membership/classification assertion, including uncertainty."""

    membership_id: str
    member_id: str
    category_id: str
    predicate: str
    direction: str
    status: str
    lifecycle_state: str
    evidence_ids: tuple[str, ...]
    evidence_statuses: tuple[str, ...]
    context_ids: tuple[str, ...]
    source_reference: str
    source_subject_id: str
    source_object_id: str

    def __post_init__(self) -> None:
        if not all((self.membership_id, self.member_id, self.category_id, self.predicate)):
            raise ValueError("membership identifiers and predicate must be non-empty")
        if self.direction != "DIRECTED":
            raise ValueError("membership direction must preserve the source DIRECTED orientation")
        if self.status not in _MEMBERSHIP_STATES:
            raise ValueError(f"unsupported membership status: {self.status}")
        if not self.evidence_ids:
            raise ValueError("a semantic membership must retain at least one evidence ID")


@dataclass(frozen=True, slots=True)
class SemanticRelation:
    """A typed ORE relation that retains the source predicate and orientation."""

    relation_id: str
    subject_id: str
    object_id: str
    predicate: str
    relation_type: str
    direction: str
    lifecycle_state: str
    evidence_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    source_reference: str
    source_kind: str = "RELATION"
    status: str = "ACTIVE"

    def __post_init__(self) -> None:
        if not all((self.relation_id, self.subject_id, self.object_id, self.predicate)):
            raise ValueError("semantic relation identifiers and predicate must be non-empty")
        if self.subject_id == self.object_id:
            raise ValueError("semantic relations cannot connect an item to itself")
        if self.relation_type not in _RELATION_TYPES:
            raise ValueError(f"unsupported semantic relation type: {self.relation_type}")
        if self.direction != "DIRECTED":
            raise ValueError("semantic relation direction must be DIRECTED")
        if self.status not in _MEMBERSHIP_STATES:
            raise ValueError(f"unsupported semantic relation status: {self.status}")
        if not self.evidence_ids:
            raise ValueError("semantic relations must retain at least one evidence ID")


__all__ = [
    "SemanticCategory",
    "SemanticCategoryMembership",
    "SemanticRelation",
]
