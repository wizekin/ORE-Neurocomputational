"""Versioned message envelopes and compatibility rules (Step 5).

Every network message carries protocol version, schema version, cluster identifier,
sender identity, and message type. Incompatible or unauthenticated messages are
rejected before any state mutation, and rolling upgrades follow explicit rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ore_distributed.core.identity import NodeCredential, canonical_bytes, digest_hex

#: Protocol version implemented by this package.
PROTOCOL_VERSION = "ORE.DIST/0.1.0"

#: Protocol versions this package accepts, ordered oldest to newest.
_SUPPORTED_PROTOCOLS = frozenset({"ORE.DIST/0.1.0"})

#: Message classes understood by the coordination layer.
MESSAGE_TYPES = frozenset(
    {
        "membership.hello",
        "membership.capabilities",
        "membership.admission",
        "consensus.request_vote",
        "consensus.request_vote_reply",
        "consensus.append_entries",
        "consensus.append_entries_reply",
        "consensus.client_command",
        "consensus.client_command_reply",
        "replication.crdt_update",
        "replication.reconcile_request",
        "replication.reconcile_reply",
        "gossip.digest",
        "gossip.repair",
        "provenance.event",
        "execution.task_assign",
        "execution.task_result",
        "recovery.join",
        "recovery.checkpoint",
        "control.ping",
    }
)

#: Message classes allowed to mutate authoritative consensus state.
_STATE_MUTATING_TYPES = frozenset(
    {
        "membership.admission",
        "consensus.request_vote",
        "consensus.append_entries",
        "consensus.client_command",
        "execution.task_assign",
        "execution.task_result",
    }
)


class ProtocolViolation(ValueError):
    """Raised when an envelope fails validation before any state mutation."""


@dataclass(frozen=True, slots=True)
class MessageEnvelope:
    """The single wire format shared by every coordination message (Step 5)."""

    message_type: str
    sender: str
    cluster_id: str
    protocol_version: str = PROTOCOL_VERSION
    schema_version: str = "0.1.0"
    nonce: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
    payload_digest: str = ""
    signature: str = ""

    def __post_init__(self) -> None:
        """Compute the payload digest and validate identity fields."""
        if self.message_type not in MESSAGE_TYPES:
            raise ProtocolViolation(f"unsupported message type {self.message_type!r}")
        for name in ("sender", "cluster_id", "protocol_version", "schema_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ProtocolViolation(f"{name} must be a non-empty string")
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "payload_digest", digest_hex(self.payload))

    def mutates_state(self) -> bool:
        """Return whether this message class may mutate authoritative state."""
        return self.message_type in _STATE_MUTATING_TYPES

    def signing_payload(self) -> dict[str, Any]:
        """Return the exact fields covered by the signature."""
        return {
            "message_type": self.message_type,
            "sender": self.sender,
            "cluster_id": self.cluster_id,
            "protocol_version": self.protocol_version,
            "schema_version": self.schema_version,
            "nonce": self.nonce,
            "payload_digest": self.payload_digest,
        }

    def sign(self, credential: NodeCredential) -> "MessageEnvelope":
        """Return a copy of this envelope signed by the sender's credential."""
        signature = credential.sign(self.signing_payload())
        return MessageEnvelope(
            message_type=self.message_type,
            sender=self.sender,
            cluster_id=self.cluster_id,
            protocol_version=self.protocol_version,
            schema_version=self.schema_version,
            nonce=self.nonce,
            payload=self.payload,
            signature=signature,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the envelope as a JSON-compatible dictionary."""
        return {
            "message_type": self.message_type,
            "sender": self.sender,
            "cluster_id": self.cluster_id,
            "protocol_version": self.protocol_version,
            "schema_version": self.schema_version,
            "nonce": self.nonce,
            "payload": dict(self.payload),
            "payload_digest": self.payload_digest,
            "signature": self.signature,
        }


def protocol_compatible(local: str, remote: str) -> bool:
    """Return whether two protocol version strings are mutually acceptable."""
    return local in _SUPPORTED_PROTOCOLS and remote in _SUPPORTED_PROTOCOLS


def check_rolling_upgrade(old: str, new: str) -> bool:
    """Return whether a rolling upgrade keeps at least one common protocol version.

    A node running ``new`` must still accept ``old`` during the upgrade window;
    downgrades to an unsupported version are rejected.
    """
    return old in _SUPPORTED_PROTOCOLS or new in _SUPPORTED_PROTOCOLS


class ProtocolGate:
    """Validate envelopes before state mutation using identity and replay rules."""

    def __init__(
        self,
        cluster_id: str,
        known_credentials: Mapping[str, NodeCredential] | None = None,
        seen_nonces: int = 2048,
    ) -> None:
        """Create a gate for one cluster with an optional known-credential set."""
        if not cluster_id.strip():
            raise ValueError("cluster_id must be a non-empty string")
        self.cluster_id = cluster_id.strip()
        self._credentials = dict(known_credentials or {})
        self._seen_nonces: set[str] = set()
        self._seen_limit = seen_nonces
        self.rejected: list[str] = []

    def register_credential(self, credential: NodeCredential) -> None:
        """Trust a credential for message verification from its node."""
        if credential.cluster_id != self.cluster_id:
            raise ValueError("credential belongs to a different cluster")
        self._credentials[credential.node_id] = credential

    def _reject(self, envelope: MessageEnvelope, reason: str) -> None:
        """Record a rejection for observability and raise a ProtocolViolation."""
        self.rejected.append(f"{envelope.message_type}:{reason}")
        raise ProtocolViolation(reason)

    def validate(self, envelope: MessageEnvelope) -> MessageEnvelope:
        """Validate compatibility, authentication, and replay before mutation (Step 5)."""
        if envelope.cluster_id != self.cluster_id:
            self._reject(envelope, "cluster identifier mismatch")
        if not protocol_compatible(PROTOCOL_VERSION, envelope.protocol_version):
            self._reject(envelope, f"incompatible protocol version {envelope.protocol_version!r}")
        if envelope.payload_digest != digest_hex(envelope.payload):
            self._reject(envelope, "payload digest mismatch")
        if envelope.mutates_state():
            if not envelope.signature:
                self._reject(envelope, "state-mutating message lacks a signature")
            credential = self._credentials.get(envelope.sender)
            if credential is None:
                self._reject(envelope, "sender is not an authenticated cluster member")
            assert credential is not None
            if not credential.verify(envelope.signing_payload(), envelope.signature):
                self._reject(envelope, "signature verification failed")
            if envelope.nonce:
                if envelope.nonce in self._seen_nonces:
                    self._reject(envelope, "replayed nonce rejected")
                self._seen_nonces.add(envelope.nonce)
                if len(self._seen_nonces) > self._seen_limit:
                    self._seen_nonces.clear()
        return envelope


def make_envelope(
    message_type: str,
    sender: str,
    cluster_id: str,
    payload: Mapping[str, Any] | None = None,
    credential: NodeCredential | None = None,
    nonce: str = "",
    schema_version: str = "0.1.0",
) -> MessageEnvelope:
    """Build a signed envelope, signing automatically for state-mutating types."""
    envelope = MessageEnvelope(
        message_type=message_type,
        sender=sender,
        cluster_id=cluster_id,
        schema_version=schema_version,
        nonce=nonce,
        payload=dict(payload or {}),
    )
    if envelope.mutates_state():
        if credential is None:
            raise ProtocolViolation(
                f"message type {message_type!r} requires a credential to sign"
            )
        envelope = envelope.sign(credential)
    return envelope


def verify_envelope_signature(
    envelope: MessageEnvelope, credential: NodeCredential
) -> bool:
    """Return whether an envelope signature matches the given credential."""
    return credential.verify(envelope.signing_payload(), envelope.signature)


__all__ = [
    "PROTOCOL_VERSION",
    "MESSAGE_TYPES",
    "ProtocolViolation",
    "MessageEnvelope",
    "ProtocolGate",
    "protocol_compatible",
    "check_rolling_upgrade",
    "make_envelope",
    "verify_envelope_signature",
]
