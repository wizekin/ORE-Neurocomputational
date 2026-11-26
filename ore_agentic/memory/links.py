"""Typed links between the three memory tiers (Step 40).

Semantic, episodic and procedural memory share storage infrastructure but
remain logically distinct: links use stable IDs and typed relations
(derived_from, uses_concept, validated_by, observed_in, supersedes).  An
observation is not automatically a fact; a procedure is not universally
valid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..contracts import new_id, utc_now

__all__ = ("MemoryKind", "MemoryLink", "MemoryLinkStore", "MemoryRelation")


class MemoryKind(str, Enum):
    """The three logically distinct memory categories (Step 40)."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class MemoryRelation(str, Enum):
    """The typed relations between memory records (Step 40)."""

    DERIVED_FROM = "derived_from"
    USES_CONCEPT = "uses_concept"
    VALIDATED_BY = "validated_by"
    OBSERVED_IN = "observed_in"
    SUPERSEDES = "supersedes"


@dataclass(frozen=True, slots=True)
class MemoryLink:
    """One typed, directed link between two memory records (Step 40)."""

    link_id: str
    source_kind: MemoryKind
    source_id: str
    relation: MemoryRelation
    target_kind: MemoryKind
    target_id: str
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate the link envelope."""
        if not self.source_id or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if not self.target_id or not self.target_id.strip():
            raise ValueError("target_id must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the link."""
        return {"link_id": self.link_id, "source_kind": self.source_kind.value,
                "source_id": self.source_id, "relation": self.relation.value,
                "target_kind": self.target_kind.value, "target_id": self.target_id,
                "created_at": self.created_at}


class MemoryLinkStore:
    """Store and traverse typed memory links without conflating kinds (Step 40)."""

    def __init__(self) -> None:
        """Start with an empty link set."""
        self._links: dict[str, MemoryLink] = {}

    def link(self, source_kind: MemoryKind, source_id: str,
             relation: MemoryRelation, target_kind: MemoryKind, target_id: str
             ) -> MemoryLink:
        """Create one typed link; identity pairs are idempotent."""
        for existing in self._links.values():
            if (existing.source_id == source_id
                    and existing.target_id == target_id
                    and existing.relation is relation):
                return existing
        record = MemoryLink(
            link_id=new_id("memlink"), source_kind=source_kind, source_id=source_id,
            relation=relation, target_kind=target_kind, target_id=target_id)
        self._links[record.link_id] = record
        return record

    def links_for(self, record_id: str) -> tuple[MemoryLink, ...]:
        """Return all links touching a record in either direction."""
        return tuple(link for link in self._links.values()
                     if link.source_id == record_id or link.target_id == record_id)

    def neighbors(self, record_id: str, *, relation: MemoryRelation | None = None
                  ) -> tuple[str, ...]:
        """Return the IDs linked from or to a record, optionally by relation."""
        neighbors = []
        for link in self.links_for(record_id):
            if relation is not None and link.relation is not relation:
                continue
            neighbors.append(link.target_id if link.source_id == record_id
                             else link.source_id)
        return tuple(neighbors)

    def links(self) -> tuple[MemoryLink, ...]:
        """Return all links."""
        return tuple(self._links.values())

    def describe(self) -> dict[str, Any]:
        """Return a serializable summary."""
        relations: dict[str, int] = {}
        for link in self._links.values():
            relations[link.relation.value] = relations.get(link.relation.value, 0) + 1
        return {"links": len(self._links), "relations": relations}
