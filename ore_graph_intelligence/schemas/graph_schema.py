"""Typed USG graph schema: relation signatures with domain/range, direction, and roles.

A :class:`GraphSchema` is the authority a projection and every downstream service consult
before neural scoring: subject/object node kinds, cardinality, direction, declared roles,
symmetry, inverse links, and transitivity are preserved as explicit constraints, exactly as
required by ORE Point 8 steps 6, 26, and 33.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

CARDINALITIES = frozenset({
    "one_to_one", "one_to_many", "many_to_one", "many_to_many",
})

Cardinality = str


@dataclass(frozen=True, slots=True)
class RoleDeclaration:
    """One declared argument role of a relation type (from USG predicate frames)."""

    role: str
    description: str = ""
    required: bool = False

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("role name must be non-empty")


@dataclass(frozen=True, slots=True)
class RelationSignature:
    """The complete typed signature of one directed USG relation type."""

    relation_type: str
    domain_kinds: frozenset[str]
    range_kinds: frozenset[str]
    cardinality: Cardinality = "many_to_many"
    direction: str = "DIRECTED"
    symmetric: bool = False
    inverse_of: str | None = None
    transitive: bool = False
    roles: tuple[RoleDeclaration, ...] = ()

    def __post_init__(self) -> None:
        if not self.relation_type.strip():
            raise ValueError("relation_type must be non-empty")
        if self.cardinality not in CARDINALITIES:
            raise ValueError(f"cardinality must be one of {sorted(CARDINALITIES)}")
        if self.direction not in {"DIRECTED", "UNDIRECTED"}:
            raise ValueError("direction must be DIRECTED or UNDIRECTED")
        if not self.domain_kinds or not self.range_kinds:
            raise ValueError("domain_kinds and range_kinds must be non-empty")
        if self.symmetric and self.direction != "DIRECTED":
            raise ValueError("symmetric relations are stored directed with symmetric=True")
        role_names = [item.role for item in self.roles]
        if len(role_names) != len(set(role_names)):
            raise ValueError("role declarations must be unique per relation type")

    def domain_allows(self, node_kind: str) -> bool:
        """Return whether a subject node kind is inside this signature's domain."""
        return node_kind in self.domain_kinds

    def range_allows(self, node_kind: str) -> bool:
        """Return whether an object node kind is inside this signature's range."""
        return node_kind in self.range_kinds

    def required_roles(self) -> tuple[str, ...]:
        """Return the roles that every well-formed instance of this type must bind."""
        return tuple(item.role for item in self.roles if item.required)


@dataclass(frozen=True, slots=True)
class GraphSchema:
    """Immutable set of relation signatures plus node-kind registry for one projection."""

    relation_signatures: tuple[RelationSignature, ...] = ()
    node_kinds: frozenset[str] = frozenset(
        {"LEMMA", "LEXICAL_SENSE", "LEXICAL_FORM", "CONCEPT", "SEMANTIC_CLASS",
         "PREDICATE_FRAME", "RELATION_EVENT"},
    )
    source_frame_ids: tuple[str, ...] = ()
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        names = [item.relation_type for item in self.relation_signatures]
        if len(names) != len(set(names)):
            raise ValueError("relation signatures must be unique per relation type")
        inverses = {
            item.inverse_of for item in self.relation_signatures if item.inverse_of is not None
        }
        unknown = inverses.difference(names)
        if unknown:
            raise ValueError(f"inverse_of references unknown relation types: {sorted(unknown)}")

    def signature(self, relation_type: str) -> RelationSignature | None:
        """Return the declared signature for one relation type, or None when undeclared."""
        for item in self.relation_signatures:
            if item.relation_type == relation_type:
                return item
        return None

    def declared_relation_types(self) -> tuple[str, ...]:
        """Return declared relation types in stable declaration order."""
        return tuple(item.relation_type for item in self.relation_signatures)

    def with_observed_defaults(self, observed: Mapping[str, tuple[frozenset[str], frozenset[str]]]) -> "GraphSchema":
        """Fill in permissive default signatures for observed but undeclared relation types.

        Observed domain/range node kinds come from the projection itself; defaults stay
        permissive (many_to_many, directed, no inverse) because the USG never declared them.
        """
        existing = {item.relation_type for item in self.relation_signatures}
        additions = []
        for relation_type in sorted(observed):
            if relation_type in existing:
                continue
            domain_kinds, range_kinds = observed[relation_type]
            additions.append(RelationSignature(
                relation_type=relation_type,
                domain_kinds=frozenset(domain_kinds or self.node_kinds),
                range_kinds=frozenset(range_kinds or self.node_kinds),
            ))
        return GraphSchema(
            relation_signatures=self.relation_signatures + tuple(additions),
            node_kinds=self.node_kinds,
            source_frame_ids=self.source_frame_ids,
            schema_version=self.schema_version,
        )

    @classmethod
    def from_predicate_frames(
        cls,
        frames: Iterable[Mapping[str, str]],
        *,
        node_kinds: Iterable[str] | None = None,
    ) -> "GraphSchema":
        """Build relation signatures from USG predicate-frame records.

        Each frame mapping must expose ``frame_key`` and optionally ``roles_json``; role
        declarations are preserved without inventing ORE execution semantics. Domain and
        range default to all node kinds because USG frames declare roles, not node types.
        """
        signatures: list[RelationSignature] = []
        frame_ids: list[str] = []
        kinds = frozenset(node_kinds or cls().node_kinds)
        for frame in frames:
            frame_key = str(frame.get("frame_key", "")).strip()
            if not frame_key:
                continue
            roles: list[RoleDeclaration] = []
            for role in frame.get("role_declarations", ()):
                roles.append(RoleDeclaration(
                    role=str(role.get("role", "")).strip(),
                    description=str(role.get("description", "")),
                    required=bool(role.get("required", False)),
                ))
            signatures.append(RelationSignature(
                relation_type=frame_key,
                domain_kinds=kinds,
                range_kinds=kinds,
                roles=tuple(roles),
            ))
            frame_ids.append(str(frame.get("frame_id", frame_key)))
        return cls(
            relation_signatures=tuple(signatures),
            node_kinds=kinds,
            source_frame_ids=tuple(sorted(frame_ids)),
        )


__all__ = [
    "CARDINALITIES", "Cardinality", "GraphSchema", "RelationSignature", "RoleDeclaration",
]
