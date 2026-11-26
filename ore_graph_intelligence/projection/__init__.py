"""Read-only, typed, provenance-preserving USG graph projection (ORE Point 8 Phase A)."""
from .integrity import GraphPartitioner, IntegrityCheck, IntegrityReport, IntegrityValidator
from .provenance import (
    GraphProvenance,
    QuarantineRecord,
    canonical_graph_json,
    compute_graph_hash,
    snapshot_digest,
)
from .typed_graph import (
    EdgeQualifiers,
    PartitionInfo,
    RoleAssignment,
    TypedEdge,
    TypedGraphSnapshot,
    TypedNode,
)
from .usg_snapshot import GraphProjectionScope, PROJECTION_METHOD, USGGraphProjector

__all__ = [
    "GraphPartitioner", "IntegrityCheck", "IntegrityReport", "IntegrityValidator",
    "GraphProvenance", "QuarantineRecord", "canonical_graph_json", "compute_graph_hash",
    "snapshot_digest",
    "EdgeQualifiers", "PartitionInfo", "RoleAssignment", "TypedEdge", "TypedGraphSnapshot",
    "TypedNode",
    "GraphProjectionScope", "PROJECTION_METHOD", "USGGraphProjector",
]
