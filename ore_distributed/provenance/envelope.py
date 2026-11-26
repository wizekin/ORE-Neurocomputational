"""Immutable provenance envelopes with digests and causal parents (Step 36).

Every published event carries event ID, actor/node ID, action, object or schema
ID, payload digest, source references, causal parents, HLC, protocol version,
and an authenticated origin signature. Original source payloads stay
content-addressed under retention policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Iterable, Mapping

from ore_distributed.core.identity import NodeCredential, canonical_bytes, digest_hex
from ore_distributed.time.hlc import HLCTimestamp


def content_address(payload: Any) -> str:
    """Return the durable content-addressed reference of a source payload."""
    return "sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class ProvenanceEnvelope:
    """One immutable, authenticated event record (Step 36)."""

    event_id: str
    actor: str
    action: str
    object_id: str
    payload_digest: str
    hlc: HLCTimestamp
    cluster_id: str = "ore"
    protocol_version: str = "ORE.DIST/0.1.0"
    schema_version: str = "0.1.0"
    causal_parents: tuple[str, ...] = ()
    source_references: tuple[str, ...] = ()
    content_reference: str = ""
    signature: str = ""
    origin_node: str = ""

    def __post_init__(self) -> None:
        """Validate identity fields and payload digest format."""
        for name in ("event_id", "actor", "action", "object_id"):
            object.__setattr__(self, name, _non_empty(name, getattr(self, name)))
        if len(self.payload_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.payload_digest
        ):
            raise ValueError("payload_digest must be a lowercase SHA-256 hex digest")
        if not isinstance(self.hlc, HLCTimestamp):
            raise ValueError("hlc must be an HLCTimestamp")
        object.__setattr__(self, "causal_parents", tuple(self.causal_parents))
        object.__setattr__(self, "source_references", tuple(self.source_references))

    def signing_payload(self) -> dict[str, Any]:
        """Return the exact fields the authenticated origin signature covers."""
        return {
            "event_id": self.event_id,
            "actor": self.actor,
            "action": self.action,
            "object_id": self.object_id,
            "payload_digest": self.payload_digest,
            "hlc": self.hlc.to_dict(),
            "cluster_id": self.cluster_id,
            "protocol_version": self.protocol_version,
            "schema_version": self.schema_version,
            "causal_parents": sorted(self.causal_parents),
        }

    def sign(self, credential: NodeCredential) -> "ProvenanceEnvelope":
        """Return a copy carrying the authenticated origin signature."""
        signature = credential.sign(self.signing_payload())
        return ProvenanceEnvelope(
            event_id=self.event_id,
            actor=self.actor,
            action=self.action,
            object_id=self.object_id,
            payload_digest=self.payload_digest,
            hlc=self.hlc,
            cluster_id=self.cluster_id,
            protocol_version=self.protocol_version,
            schema_version=self.schema_version,
            causal_parents=self.causal_parents,
            source_references=self.source_references,
            content_reference=self.content_reference,
            signature=signature,
            origin_node=credential.node_id,
        )

    def verify_signature(self, credential: NodeCredential) -> bool:
        """Verify the authenticated origin signature of this envelope."""
        return credential.verify(self.signing_payload(), self.signature)

    def with_content_reference(self, payload: Any) -> "ProvenanceEnvelope":
        """Return a copy with the durable content-addressed source reference."""
        return ProvenanceEnvelope(
            event_id=self.event_id,
            actor=self.actor,
            action=self.action,
            object_id=self.object_id,
            payload_digest=self.payload_digest,
            hlc=self.hlc,
            cluster_id=self.cluster_id,
            protocol_version=self.protocol_version,
            schema_version=self.schema_version,
            causal_parents=self.causal_parents,
            source_references=self.source_references,
            content_reference=content_address(payload),
            signature=self.signature,
            origin_node=self.origin_node,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the envelope as a plain dictionary."""
        return {
            "event_id": self.event_id,
            "actor": self.actor,
            "action": self.action,
            "object_id": self.object_id,
            "payload_digest": self.payload_digest,
            "hlc": self.hlc.to_dict(),
            "cluster_id": self.cluster_id,
            "protocol_version": self.protocol_version,
            "schema_version": self.schema_version,
            "causal_parents": list(self.causal_parents),
            "source_references": list(self.source_references),
            "content_reference": self.content_reference,
            "signature": self.signature,
            "origin_node": self.origin_node,
        }


def build_envelope(
    *,
    event_id: str,
    actor: str,
    action: str,
    object_id: str,
    payload: Mapping[str, Any],
    hlc: HLCTimestamp,
    causal_parents: Iterable[str] = (),
    source_references: Iterable[str] = (),
    cluster_id: str = "ore",
    credential: NodeCredential | None = None,
) -> ProvenanceEnvelope:
    """Build (and optionally sign) one provenance envelope for a payload."""
    envelope = ProvenanceEnvelope(
        event_id=event_id,
        actor=actor,
        action=action,
        object_id=object_id,
        payload_digest=digest_hex(dict(payload)),
        hlc=hlc,
        cluster_id=cluster_id,
        causal_parents=tuple(causal_parents),
        source_references=tuple(source_references),
    ).with_content_reference(payload)
    if credential is not None:
        return envelope.sign(credential)
    return envelope


def supersede_envelope(
    previous: ProvenanceEnvelope,
    replacement: ProvenanceEnvelope,
) -> ProvenanceEnvelope:
    """Return a replacement envelope that records its predecessor as a parent."""
    return ProvenanceEnvelope(
        event_id=replacement.event_id,
        actor=replacement.actor,
        action=replacement.action,
        object_id=replacement.object_id,
        payload_digest=replacement.payload_digest,
        hlc=replacement.hlc,
        cluster_id=replacement.cluster_id,
        protocol_version=replacement.protocol_version,
        schema_version=replacement.schema_version,
        causal_parents=tuple(set(replacement.causal_parents) | {previous.event_id}),
        source_references=replacement.source_references,
        content_reference=replacement.content_reference,
        signature=replacement.signature,
        origin_node=replacement.origin_node,
    )


__all__ = [
    "ProvenanceEnvelope",
    "content_address",
    "build_envelope",
    "supersede_envelope",
]
