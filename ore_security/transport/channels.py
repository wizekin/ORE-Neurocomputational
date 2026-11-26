"""Secure communication gateway: policy validation and authenticated channels (step 15).

``TransportPolicy`` enforces approved protocol versions and cipher suites and rejects
obsolete configurations. ``SecureChannelEndpoint`` pairs model mutually authenticated
channels between ORE components: the handshake authenticates the peer credential for
the *intended* service identity, derives keys with X25519 + HKDF bound to both
identities, and protects messages with AES-256-GCM under strictly increasing sequence
numbers. In-process endpoints exercise the same contract a TLS 1.3 / mTLS deployment
enforces; production transports substitute the wire layer, not the policy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..canonical import canonical_bytes
from ..crypto import aes_gcm_decrypt, aes_gcm_encrypt, hkdf_derive, x25519_shared_secret
from ..exceptions import ConfigurationRejectedError, PeerValidationError, TransportError
from ..identity.identities import IdentityService

Clock = Callable[[], float]

OBSOLETE_TLS_VERSIONS = frozenset({"1.0", "1.1", "SSL 3.0"})
WEAK_CIPHER_MARKERS = ("RC4", "3DES", "DES", "NULL", "EXPORT", "MD5", "RC2", "anon")
DEFAULT_ALLOWED_TLS_VERSIONS = frozenset({"1.2", "1.3"})
DEFAULT_ALLOWED_CIPHERS = frozenset({
    "TLS_AES_256_GCM_SHA384",
    "TLS_CHACHA20_POLY1305_SHA256",
    "TLS_AES_128_GCM_SHA256",
    "ECDHE-RSA-AES256-GCM-SHA384",
    "ECDHE-ECDSA-AES256-GCM-SHA384",
})


@dataclass(frozen=True, slots=True)
class TransportPolicy:
    """Approved transport configurations; unknown or obsolete settings are rejected."""

    min_tls_version: str = "1.2"
    allowed_tls_versions: frozenset[str] = DEFAULT_ALLOWED_TLS_VERSIONS
    allowed_cipher_suites: frozenset[str] = DEFAULT_ALLOWED_CIPHERS
    require_peer_authentication: bool = True

    def validate_connection_request(self, tls_version: str,
                                    cipher_suites: tuple[str, ...] = ()) -> dict[str, Any]:
        """Accept or reject a proposed connection configuration (acceptance test)."""
        version = str(tls_version).strip()
        if version in OBSOLETE_TLS_VERSIONS:
            raise ConfigurationRejectedError(
                f"obsolete protocol version rejected: TLS {version}"
            )
        if version not in self.allowed_tls_versions:
            raise ConfigurationRejectedError(
                f"protocol version {version!r} is not in the approved set {sorted(self.allowed_tls_versions)}"
            )
        ordered = sorted(self.allowed_tls_versions, key=lambda item: float(item))
        if float(version) < float(self.min_tls_version):
            raise ConfigurationRejectedError(
                f"protocol version {version} is below the minimum {self.min_tls_version}"
            )
        if not cipher_suites:
            return {"accepted": True, "tls_version": version, "cipher_suites": []}
        weak = [suite for suite in cipher_suites
                if any(marker in suite.upper() for marker in WEAK_CIPHER_MARKERS)]
        if weak:
            raise ConfigurationRejectedError(f"weak or obsolete cipher suites rejected: {weak}")
        unknown = [suite for suite in cipher_suites if suite not in self.allowed_cipher_suites]
        if unknown:
            raise ConfigurationRejectedError(
                f"cipher suites not in the approved set: {unknown}"
            )
        return {"accepted": True, "tls_version": version,
                "cipher_suites": list(cipher_suites)}


@dataclass
class ChannelHandshake:
    """Peer-presented handshake material for one endpoint."""

    token: str
    x25519_public: bytes
    proposed_tls_version: str = "1.3"
    proposed_cipher: str = "TLS_AES_256_GCM_SHA384"


class SecureChannelEndpoint:
    """One end of a mutually authenticated, encrypted channel between services."""

    def __init__(self, service_name: str, identity_service: IdentityService,
                 credential_token: str, *, policy: TransportPolicy | None = None,
                 clock: Clock = time.time) -> None:
        self.service_name = service_name
        self.identity_service = identity_service
        self.credential_token = credential_token
        self.policy = policy or TransportPolicy()
        self.clock = clock
        from ..crypto import generate_x25519_keypair
        self._private, self.public = generate_x25519_keypair()
        self._session_key: bytes | None = None
        self._peer_id: str | None = None
        self._send_sequence = 0
        self._receive_sequence = 0
        self._channel_id: str | None = None

    def handshake_material(self) -> ChannelHandshake:
        """Produce this endpoint's handshake presentation."""
        return ChannelHandshake(token=self.credential_token, x25519_public=self.public)

    def establish(self, peer: ChannelHandshake, *, expected_peer_service: str,
                  local_is_initiator: bool) -> dict[str, Any]:
        """Authenticate the peer as the intended service and derive channel keys."""
        report = self.policy.validate_connection_request(
            peer.proposed_tls_version, (peer.proposed_cipher,),
        )
        principal = self.identity_service.authenticate_token(
            peer.token, expected_audience=self.service_name,
        )
        if principal.principal_id != expected_peer_service:
            raise PeerValidationError(
                f"peer authenticated as {principal.principal_id!r} but the intended "
                f"service is {expected_peer_service!r}"
            )
        secret = x25519_shared_secret(self._private, peer.x25519_public)
        initiator = self.service_name if local_is_initiator else principal.principal_id
        responder = principal.principal_id if local_is_initiator else self.service_name
        channel_id = f"{initiator}->{responder}"
        self._session_key = hkdf_derive(
            secret,
            salt=canonical_bytes({"initiator": initiator, "responder": responder}),
            info=canonical_bytes({
                "channel_binding": channel_id,
                "purpose": "ore-security-authenticated-channel",
                "tls_version": report["tls_version"],
            }),
        )
        self._peer_id = principal.principal_id
        self._channel_id = channel_id
        self._send_sequence = 0
        self._receive_sequence = 0
        return {
            "established": True,
            "peer": principal.principal_id,
            "tls_version": report["tls_version"],
            "cipher": peer.proposed_cipher,
            "channel_id": channel_id,
        }

    def send(self, plaintext: bytes) -> dict[str, Any]:
        """Protect one message: AEAD with a sequence-derived nonce and channel AAD."""
        if self._session_key is None:
            raise TransportError("channel is not established")
        self._send_sequence += 1
        nonce = self._sequence_nonce(self._send_sequence, direction=b"C")
        aad = canonical_bytes({
            "channel_id": self._channel_id, "sequence": self._send_sequence,
            "sender": self.service_name,
        })
        sealed = aes_gcm_encrypt(self._session_key, plaintext, nonce, aad)
        return {
            "channel_id": self._channel_id,
            "sequence": self._send_sequence,
            "sender": self.service_name,
            "sealed": sealed.hex(),
        }

    def receive(self, frame: dict[str, Any]) -> bytes:
        """Open one protected message; replays and out-of-order frames are rejected."""
        if self._session_key is None:
            raise TransportError("channel is not established")
        if frame.get("channel_id") != self._channel_id:
            raise TransportError("frame belongs to a different channel")
        sequence = int(frame["sequence"])
        if sequence <= self._receive_sequence:
            raise TransportError(
                f"replayed or out-of-order frame: sequence {sequence} <= {self._receive_sequence}"
            )
        aad = canonical_bytes({
            "channel_id": self._channel_id, "sequence": sequence,
            "sender": frame.get("sender"),
        })
        nonce = self._sequence_nonce(sequence, direction=b"C")
        plaintext = aes_gcm_decrypt(
            self._session_key, bytes.fromhex(frame["sealed"]), nonce, aad,
        )
        self._receive_sequence = sequence
        return plaintext

    @staticmethod
    def _sequence_nonce(sequence: int, direction: bytes) -> bytes:
        """Deterministic 96-bit nonce from the sequence number and direction label."""
        return direction + sequence.to_bytes(11, "big")

    def is_established(self) -> bool:
        return self._session_key is not None
