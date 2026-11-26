"""Typed, loss-aware USG graph projection model (ORE Point 8 steps 5-7).

A :class:`TypedGraphSnapshot` is a *versioned projection of USG, never a replacement*:
stable concept identities, typed directed relations with argument roles, qualifiers
(negation, modality, attribution, conditionals, temporal scope, uncertainty) and
provenance are preserved; unsupported logical forms are quarantined instead of flattened.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..schemas.graph_schema import GraphSchema
from .provenance import GraphProvenance, QuarantineRecord


@dataclass(frozen=True, slots=True)
class EdgeQualifiers:
    """Explicit logical scope of one relation instance (ORE Point 8 step 7)."""

    polarity: str = "positive"
    modality: str | None = None
    attribution: str | None = None
    condition: str | None = None
    exceptions: tuple[str, ...] = ()
    temporal_scope: str | None = None
    uncertainty: float | None = None
    source_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.polarity not in {"positive", "negative", "unknown"}:
            raise ValueError("polarity must be positive, negative, or unknown")
        if self.uncertainty is not None and not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must lie in [0, 1]")
        object.__setattr__(self, "exceptions", tuple(self.exceptions))
        object.__setattr__(self, "source_notes", tuple(self.source_notes))

    def scope_summary(self) -> str:
        parts = [f"polarity={self.polarity}"]
        for name in ("modality", "attribution", "condition", "temporal_scope"):
            value = getattr(self, name)
            if value is not None:
                parts.append(f"{name}={value}")
        if self.exceptions:
            parts.append(f"exceptions={len(self.exceptions)}")
        if self.uncertainty is not None:
            parts.append(f"uncertainty={self.uncertainty}")
        return ";".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "polarity": self.polarity, "modality": self.modality,
            "attribution": self.attribution, "condition": self.condition,
            "exceptions": list(self.exceptions), "temporal_scope": self.temporal_scope,
            "uncertainty": self.uncertainty, "source_notes": list(self.source_notes),
        }


@dataclass(frozen=True, slots=True)
class RoleAssignment:
    """One bound argument role of a reified n-ary relation event (step 6)."""

    role: str
    node_id: str

    def __post_init__(self) -> None:
        if not self.role.strip() or not self.node_id.strip():
            raise ValueError("role and node_id must be non-empty")


@dataclass(frozen=True, slots=True)
class TypedNode:
    """One stable concept/object node: one node per semantic identity, not spelling."""

    node_id: str
    node_kind: str
    label: str
    aliases: tuple[str, ...] = ()
    usg_record_id: str = ""
    provenance: GraphProvenance | None = None
    payload_digest: str = ""
    lifecycle_state: str = "ACCEPTED"

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise ValueError("node_id must be non-empty")
        if self.node_kind not in {
            "LEMMA", "LEXICAL_SENSE", "LEXICAL_FORM", "CONCEPT", "SEMANTIC_CLASS",
            "PREDICATE_FRAME", "RELATION_EVENT",
        }:
            raise ValueError(f"unsupported node kind: {self.node_kind}")
        object.__setattr__(self, "aliases", tuple(self.aliases))

    def alias_set(self) -> frozenset[str]:
        """Return label plus aliases; used for anchor resolution without semantic merging."""
        return frozenset({self.label, *self.aliases} if self.label else self.aliases)


@dataclass(frozen=True, slots=True)
class TypedEdge:
    """One typed directed relation instance preserving roles, qualifiers, and provenance."""

    edge_id: str
    relation_type: str
    source_id: str
    target_id: str
    direction: str = "DIRECTED"
    qualifiers: EdgeQualifiers = field(default_factory=EdgeQualifiers)
    roles: tuple[RoleAssignment, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    context_ids: tuple[str, ...] = ()
    provenance: GraphProvenance | None = None
    lifecycle_state: str = "ACCEPTED"

    def __post_init__(self) -> None:
        if not self.edge_id.strip():
            raise ValueError("edge_id must be non-empty")
        if not self.relation_type.strip():
            raise ValueError("relation_type must be non-empty")
        if self.direction not in {"DIRECTED", "UNDIRECTED"}:
            raise ValueError("direction must be DIRECTED or UNDIRECTED")
        object.__setattr__(self, "roles", tuple(self.roles))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "context_ids", tuple(self.context_ids))

    def reversed_signature(self) -> tuple[str, str, str]:
        """Return the reversed (target, relation, source) triple; direction is preserved."""
        return (self.target_id, self.relation_type, self.source_id)


@dataclass(frozen=True, slots=True)
class PartitionInfo:
    """Recorded partition strategy for bounded computation (ORE Point 8 step 10)."""

    strategy: str
    seed: int
    partition_ids: tuple[str, ...] = ()
    max_partition_size: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy, "seed": self.seed,
            "partitions": len(self.partition_ids),
            "max_partition_size": self.max_partition_size,
        }


@dataclass(frozen=True, slots=True)
class TypedGraphSnapshot:
    """Immutable typed projection of one USG bundle under one projection scope."""

    nodes: tuple[TypedNode, ...]
    edges: tuple[TypedEdge, ...]
    schema: GraphSchema
    namespace: str
    bundle_id: str
    bundle_digest: str
    graph_hash: str
    provenance: GraphProvenance
    quarantine: tuple[QuarantineRecord, ...] = ()
    partition: PartitionInfo | None = None
    scope: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("projected node ids must be unique")
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("projected edge ids must be unique")
        if len(self.graph_hash) != 64:
            raise ValueError("graph_hash must be a SHA-256 hex digest")

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def node_by_id(self, node_id: str) -> TypedNode | None:
        """Return one node by stable id without building an index."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def node_id_by_alias(self, alias: str) -> tuple[str, ...]:
        """Resolve an alias to every node whose USG identity carries that alias."""
        matches = tuple(
            node.node_id for node in self.nodes
            if alias in node.alias_set()
        )
        return matches

    def edges_by_relation(self, relation_type: str) -> tuple[TypedEdge, ...]:
        """Return every projected edge instance of one relation type."""
        return tuple(edge for edge in self.edges if edge.relation_type == relation_type)

    def relation_types(self) -> tuple[str, ...]:
        """Distinct relation types present in the projection, preserving type identity."""
        return tuple(sorted({edge.relation_type for edge in self.edges}))

    def degree_of(self, node_id: str) -> tuple[int, int]:
        """Return (out-degree, in-degree) for one node over directed edges."""
        out_degree = sum(1 for edge in self.edges if edge.source_id == node_id)
        in_degree = sum(1 for edge in self.edges if edge.target_id == node_id)
        return (out_degree, in_degree)

    def existing_triples(self) -> frozenset[tuple[str, str, str]]:
        """Return the set of projected (subject, relation, object) triples."""
        return frozenset((edge.source_id, edge.relation_type, edge.target_id) for edge in self.edges)

    def to_serializable(self) -> dict[str, Any]:
        """Canonical, deterministic serialization used for the graph hash."""
        return {
            "namespace": self.namespace,
            "bundle_id": self.bundle_id,
            "bundle_digest": self.bundle_digest,
            "scope": dict(self.scope),
            "schema": {
                "version": self.schema.schema_version,
                "relations": [
                    {
                        "type": sig.relation_type,
                        "domain": sorted(sig.domain_kinds),
                        "range": sorted(sig.range_kinds),
                        "cardinality": sig.cardinality,
                        "direction": sig.direction,
                        "symmetric": sig.symmetric,
                        "inverse_of": sig.inverse_of,
                        "transitive": sig.transitive,
                        "roles": [{"role": r.role, "required": r.required} for r in sig.roles],
                    }
                    for sig in self.schema.relation_signatures
                ],
            },
            "nodes": [
                {
                    "node_id": node.node_id, "kind": node.node_kind, "label": node.label,
                    "aliases": sorted(node.aliases), "usg_record_id": node.usg_record_id,
                    "payload_digest": node.payload_digest, "lifecycle_state": node.lifecycle_state,
                    "provenance": node.provenance.as_dict() if node.provenance else None,
                }
                for node in sorted(self.nodes, key=lambda item: item.node_id)
            ],
            "edges": [
                {
                    "edge_id": edge.edge_id, "relation_type": edge.relation_type,
                    "source": edge.source_id, "target": edge.target_id,
                    "direction": edge.direction, "qualifiers": edge.qualifiers.as_dict(),
                    "roles": [{"role": r.role, "node": r.node_id} for r in edge.roles],
                    "evidence_ids": sorted(edge.evidence_ids),
                    "context_ids": sorted(edge.context_ids),
                    "lifecycle_state": edge.lifecycle_state,
                    "provenance": edge.provenance.as_dict() if edge.provenance else None,
                }
                for edge in sorted(self.edges, key=lambda item: item.edge_id)
            ],
            "quarantine": [record.as_dict() for record in sorted(
                self.quarantine, key=lambda item: (item.record_id, item.reason),
            )],
        }


__all__ = [
    "EdgeQualifiers", "PartitionInfo", "RoleAssignment", "TypedEdge", "TypedGraphSnapshot",
    "TypedNode",
]
