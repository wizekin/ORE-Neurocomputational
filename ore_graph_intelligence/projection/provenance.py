"""Provenance attachment and deterministic graph hashing for USG graph projections.

Every projected node, edge, and quarantined record carries a :class:`GraphProvenance`
record with source ids, evidence ids, extraction method, timestamp, trust label, and the
USG snapshot hash (ORE Point 8 step 8). The graph hash is a deterministic digest over the
canonical serialization so identical snapshot/config pairs reproduce identical hashes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping


def canonical_graph_json(value: Mapping[str, Any]) -> str:
    """Serialize graph data deterministically (sorted keys, compact separators)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _source_locator(
    namespace: str, bundle_id: str, collection: str, record_id: str, state: str,
    source_ids: tuple[str, ...],
) -> str:
    from urllib.parse import quote

    return (
        f"usg-graph://{quote(namespace, safe='')}/{quote(bundle_id, safe='')}/"
        f"{collection}/{quote(record_id, safe='')}?state={quote(state, safe='')}"
        f"&sources={','.join(quote(item, safe='') for item in source_ids)}"
    )


@dataclass(frozen=True, slots=True)
class GraphProvenance:
    """Where one projected graph element came from (ORE Point 8 step 8)."""

    source_ids: tuple[str, ...]
    extraction_method: str
    snapshot_hash: str
    timestamp: str = ""
    trust_label: str = "UNKNOWN"
    evidence_ids: tuple[str, ...] = ()
    source_reference: str = ""

    def __post_init__(self) -> None:
        if not self.extraction_method.strip():
            raise ValueError("extraction_method must be non-empty")
        if len(self.snapshot_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.snapshot_hash
        ):
            raise ValueError("snapshot_hash must be a lowercase SHA-256 hex digest")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_ids": list(self.source_ids),
            "evidence_ids": list(self.evidence_ids),
            "extraction_method": self.extraction_method,
            "timestamp": self.timestamp,
            "trust_label": self.trust_label,
            "snapshot_hash": self.snapshot_hash,
            "source_reference": self.source_reference,
        }


@dataclass(frozen=True, slots=True)
class QuarantineRecord:
    """One malformed or unsupported record excluded from the projection (step 9)."""

    record_id: str
    record_kind: str
    reason: str
    details: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id, "record_kind": self.record_kind,
            "reason": self.reason, "details": self.details,
        }


def snapshot_digest(bundle: Mapping[str, Any]) -> str:
    """Return the USG bundle digest used as the authoritative snapshot hash."""
    hasher = sha256()
    hasher.update(canonical_graph_json(bundle).encode("utf-8"))
    return hasher.hexdigest()


def compute_graph_hash(payload: Mapping[str, Any]) -> str:
    """Hash the canonical projection serialization, including scope and schema config."""
    hasher = sha256()
    hasher.update(canonical_graph_json(payload).encode("utf-8"))
    return hasher.hexdigest()


__all__ = [
    "GraphProvenance", "QuarantineRecord", "canonical_graph_json", "compute_graph_hash",
    "snapshot_digest",
]
