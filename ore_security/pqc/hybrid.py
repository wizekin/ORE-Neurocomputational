"""Hybrid classical/PQC envelopes with algorithm agility (steps 54, 56).

Hybrid signatures bind two independent signature legs (classical Ed25519 plus a
post-quantum leg) into one versioned envelope; verification requires both legs to
pass. Hybrid key establishment combines a classical X25519 agreement with a PQC KEM
leg through the standard KDF combiner, so the result is secure if either leg holds.

Honest scope: the PQC legs here are reference vehicles for envelope plumbing and
interoperability tests. The hash-based one-time signature demonstrates the SLH-DSA
family's design principle (sign once, verify many) using SHA-256, and the mock KEM
is a labeled hash-based construction. Production requires ML-KEM/ML-DSA/SLH-DSA
from a vetted provider; the envelope format and policy stay unchanged.
"""
from __future__ import annotations

import secrets
from typing import Any

from ..canonical import canonical_bytes, digest_bytes
from ..crypto import (
    ed25519_sign,
    ed25519_verify,
    generate_ed25519_keypair,
    hkdf_derive,
    x25519_shared_secret,
    generate_x25519_keypair,
)
from ..exceptions import PQCError, SecurityError

ENVELOPE_VERSION = "hybrid-envelope-v1"


# ---------------------------------------------------------------------------
# Reference hash-based one-time signature (Lamport) -- PQC signature leg
# ---------------------------------------------------------------------------

class ReferenceHashBasedOTS:
    """Lamport-style one-time signature over SHA-256 (educational reference).

    Security: one-time EUF-CMA under SHA-256 preimage resistance. The private key
    signs exactly one message; attempting a second signature raises. This leg
    demonstrates hash-based signing (the SLH-DSA family principle) for envelope
    interop tests. It is NOT SLH-DSA and must not protect production data.
    """

    algorithm_id = "SLH-DSA-128s (reference-Lamport leg)"
    stateless_note = "reference implementation is strictly one-time per keypair"

    def __init__(self) -> None:
        self._private = [[secrets.token_bytes(32) for _ in range(256)] for _ in range(2)]
        self._public = [[digest_bytes(value) for value in row] for row in self._private]
        self._used = False

    @property
    def public_key(self) -> list[list[str]]:
        return [[digest for digest in row] for row in self._public]

    def sign(self, message: bytes) -> dict[str, Any]:
        """Sign one message; a second use of the key is refused."""
        if self._used:
            raise PQCError("hash-based one-time key already consumed; generate a new keypair")
        self._used = True
        digest = bytes.fromhex(digest_bytes(message))
        signature: list[str] = []
        bit_index = 0
        for byte in digest:
            for position in range(8):
                bit = (byte >> position) & 1
                signature.append(self._private[bit][bit_index].hex())
                bit_index += 1
        return {"algorithm": "Reference-Lamport-SHA256", "values": signature}

    @staticmethod
    def verify(public_key: list[list[str]], message: bytes,
               signature: dict[str, Any]) -> bool:
        """Verify by hashing the revealed values and comparing to the public key."""
        if signature.get("algorithm") != "Reference-Lamport-SHA256":
            return False
        values = signature.get("values")
        if not isinstance(values, list) or len(values) != 256:
            return False
        if len(public_key) != 2 or any(len(row) != 256 for row in public_key):
            return False
        digest = bytes.fromhex(digest_bytes(message))
        bits = [(byte >> position) & 1 for byte in digest for position in range(8)]
        for index, (bit, value_hex) in enumerate(zip(bits, values)):
            try:
                value = bytes.fromhex(value_hex)
            except (TypeError, ValueError):
                return False
            if digest_bytes(value) != public_key[bit][index]:
                return False
        return True


# ---------------------------------------------------------------------------
# Hybrid signature envelope
# ---------------------------------------------------------------------------

class HybridSignatureEnvelope:
    """Sign and verify payloads with two independent signature legs."""

    def __init__(self) -> None:
        self._classical_private, self.classical_public = generate_ed25519_keypair()
        self.ots = ReferenceHashBasedOTS()
        self.ots_public = self.ots.public_key

    def sign(self, payload: bytes) -> dict[str, Any]:
        """Produce the versioned hybrid signature envelope over the payload."""
        classical = ed25519_sign(self._classical_private, payload)
        pq_leg = self.ots.sign(payload)
        return {
            "envelope_version": ENVELOPE_VERSION,
            "payload_digest": digest_bytes(payload),
            "classical_leg": {
                "algorithm": "Ed25519",
                "value": classical.hex(),
            },
            "pqc_leg": {
                "algorithm": pq_leg["algorithm"],
                "note": "reference leg for envelope testing; use SLH-DSA/ML-DSA in production",
                "values": pq_leg["values"],
            },
        }

    def verify(self, payload: bytes, envelope: dict[str, Any]) -> dict[str, Any]:
        """Verify both legs; a failure in either leg fails the envelope."""
        if envelope.get("envelope_version") != ENVELOPE_VERSION:
            raise PQCError("unknown hybrid envelope version")
        if envelope.get("payload_digest") != digest_bytes(payload):
            return {"verified": False, "failed_leg": "payload_digest",
                    "reason": "payload does not match the envelope digest"}
        classical_ok = ed25519_verify(
            self.classical_public, payload,
            bytes.fromhex(envelope["classical_leg"]["value"]),
        )
        if not classical_ok:
            return {"verified": False, "failed_leg": "classical",
                    "reason": "Ed25519 leg failed"}
        pq_ok = ReferenceHashBasedOTS.verify(
            self.ots_public, payload, {"algorithm": envelope["pqc_leg"]["algorithm"],
                                       "values": envelope["pqc_leg"]["values"]},
        )
        if not pq_ok:
            return {"verified": False, "failed_leg": "pqc",
                    "reason": "hash-based reference leg failed"}
        return {"verified": True, "legs": ["classical", "pqc"],
                "note": "hybrid verification requires every leg to pass"}


# ---------------------------------------------------------------------------
# Hybrid key establishment
# ---------------------------------------------------------------------------

PQC_KEM_BACKEND = "mock-hash-kem (envelope testing only; substitute ML-KEM)"


def _mock_pqc_kem_encapsulate(peer_public: bytes, context: bytes) -> tuple[bytes, bytes]:
    """Mock PQC KEM leg: labeled hash-based encapsulation for plumbing tests."""
    secret = secrets.token_bytes(32)
    ciphertext = bytes.fromhex(digest_bytes(peer_public + b"|" + context)) + secret
    return secret, ciphertext


def _mock_pqc_kem_decapsulate(ciphertext: bytes) -> bytes:
    """Recover the mock KEM shared secret (the trailing 32 bytes)."""
    if len(ciphertext) < 32:
        raise PQCError("malformed mock KEM ciphertext")
    return ciphertext[-32:]


class HybridKeyEstablishment:
    """Hybrid classical+PQC key establishment with the standard KDF combiner."""

    def __init__(self) -> None:
        self._x25519_private, self.x25519_public = generate_x25519_keypair()
        self._kem_seed = secrets.token_bytes(32)
        self.kem_public = bytes.fromhex(digest_bytes(self._kem_seed))

    def initiate(self, peer_classical_public: bytes, peer_kem_public: bytes,
                 context: bytes) -> dict[str, Any]:
        """Produce the hybrid handshake message and derive the shared key."""
        classical_secret = x25519_shared_secret(self._x25519_private, peer_classical_public)
        pq_secret, kem_ciphertext = _mock_pqc_kem_encapsulate(peer_kem_public, context)
        shared = _hybrid_combine(classical_secret, pq_secret, context)
        return {
            "envelope_version": ENVELOPE_VERSION,
            "classical_public": self.x25519_public.hex(),
            "kem_ciphertext": kem_ciphertext.hex(),
            "pqc_backend": PQC_KEM_BACKEND,
            "shared_key_binding": digest_bytes(shared),
        }

    def respond(self, message: dict[str, Any], peer_classical_public: bytes,
                context: bytes) -> dict[str, Any]:
        """Respond to a hybrid handshake and derive the same shared key."""
        classical_secret = x25519_shared_secret(self._x25519_private, peer_classical_public)
        pq_secret = _mock_pqc_kem_decapsulate(bytes.fromhex(message["kem_ciphertext"]))
        shared = _hybrid_combine(classical_secret, pq_secret, context)
        return {
            "envelope_version": ENVELOPE_VERSION,
            "shared_key_binding": digest_bytes(shared),
            "pqc_backend": PQC_KEM_BACKEND,
        }

    def derive_shared_key(self, classical_secret: bytes, pq_secret: bytes,
                          context: bytes) -> bytes:
        return _hybrid_combine(classical_secret, pq_secret, context)


def _hybrid_combine(classical_secret: bytes, pq_secret: bytes, context: bytes) -> bytes:
    """Standard KDF combiner: HKDF over the concatenation of both secrets.

    The combiner is the reviewed pattern (labeled concatenation into a KDF); the
    security holds if either contributing agreement remains secure.
    """
    return hkdf_derive(
        classical_secret + pq_secret,
        salt=b"ore-hybrid-establishment",
        info=canonical_bytes({"context": context.hex(), "version": ENVELOPE_VERSION}),
    )


def interoperability_test() -> dict[str, Any]:
    """Run the hybrid establishment and signature interoperability checks (step 56)."""
    initiator = HybridKeyEstablishment()
    responder = HybridKeyEstablishment()
    context = b"interop-test-context"
    message = initiator.initiate(responder.x25519_public, responder.kem_public, context)
    response = responder.respond(message, bytes.fromhex(message["classical_public"]), context)
    establishment_ok = message["shared_key_binding"] == response["shared_key_binding"]

    envelope_holder = HybridSignatureEnvelope()
    payload = b"ore-hybrid-interop-payload"
    envelope = envelope_holder.sign(payload)
    signature_ok = envelope_holder.verify(payload, envelope)["verified"]
    tampered = envelope_holder.verify(b"other payload", envelope)["verified"] is False

    return {
        "hybrid_establishment": establishment_ok,
        "hybrid_signature": signature_ok,
        "tamper_detection": tampered,
        "pqc_backend": PQC_KEM_BACKEND,
        "passed": establishment_ok and signature_ok and tampered,
    }
