"""Facade over the vetted ``cryptography`` library for all security primitives.

Spec Point 20 forbids hand-written cryptographic primitives: signatures, AEAD, and key
agreement must come from a maintained library. This module is the single import point,
exposes the provider identity for the acceptance test on vetted libraries, and raises a
clear error instead of silently falling back to unvetted code when the library is
missing. Higher-level protocols (hash chains, Merkle trees, Schnorr proofs, secret
sharing, Paillier, masking) are built on top of these primitives plus ``hashlib``.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Any

from .exceptions import SecurityError

PROVIDER_NAME = "cryptography"
REFERENCE_PROTOCOLS_NOTE = (
    "Protocol-level constructions (hash chain, Merkle tree, Schnorr PoK, additive "
    "secret sharing, Paillier PHE, secure-aggregation masking) are composed from these "
    "vetted primitives and stdlib big-integer arithmetic; production deployments should "
    "swap in maintained proof systems and FHE libraries through the adapter contracts."
)


def provider_report() -> dict[str, Any]:
    """Report the primitive provider identity for audits and acceptance tests."""
    try:
        from cryptography import __version__ as cryptography_version
    except Exception as exc:  # pragma: no cover - environment specific
        raise SecurityError(
            "the vetted 'cryptography' library is required for ore_security primitives; "
            "install it before using signing, transport, or AEAD features"
        ) from exc
    return {
        "library": PROVIDER_NAME,
        "version": cryptography_version,
        "primitives": ["Ed25519", "X25519", "AES-256-GCM", "HKDF-SHA256", "SHA-2"],
        "reference_protocols_note": REFERENCE_PROTOCOLS_NOTE,
    }


def require_provider() -> Any:
    """Import and return the cryptography module or fail with an actionable error."""
    try:
        import cryptography  # noqa: F401
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        return cryptography
    except Exception as exc:
        raise SecurityError(
            "the vetted 'cryptography' library is required but not importable; "
            "install it before using ore_security cryptographic adapters"
        ) from exc


def _ed25519():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
    return Ed25519PrivateKey, serialization


def generate_ed25519_keypair() -> tuple[bytes, bytes]:
    """Return (private_bytes, public_bytes) for a fresh Ed25519 keypair."""
    Ed25519PrivateKey, serialization = _ed25519()
    private = Ed25519PrivateKey.generate()
    private_bytes = private.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()
    )
    public_bytes = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return private_bytes, public_bytes


def ed25519_sign(private_bytes: bytes, message: bytes) -> bytes:
    """Sign a message with a raw Ed25519 private key."""
    Ed25519PrivateKey, serialization = _ed25519()
    private = Ed25519PrivateKey.from_private_bytes(private_bytes)
    return private.sign(message)


def ed25519_verify(public_bytes: bytes, message: bytes, signature: bytes) -> bool:
    """Verify an Ed25519 signature; return False instead of raising on mismatch."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.exceptions import InvalidSignature
    try:
        Ed25519PublicKey.from_public_bytes(public_bytes).verify(signature, message)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def generate_x25519_keypair() -> tuple[bytes, bytes]:
    """Return (private_bytes, public_bytes) for a fresh X25519 keypair."""
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives import serialization
    private = X25519PrivateKey.generate()
    private_bytes = private.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()
    )
    public_bytes = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return private_bytes, public_bytes


def x25519_shared_secret(private_bytes: bytes, peer_public_bytes: bytes) -> bytes:
    """Compute the raw X25519 Diffie-Hellman shared secret."""
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    private = X25519PrivateKey.from_private_bytes(private_bytes)
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey
    return private.exchange(X25519PublicKey.from_public_bytes(peer_public_bytes))


def hkdf_derive(input_key_material: bytes, *, salt: bytes, info: bytes, length: int = 32) -> bytes:
    """Derive key material with HKDF-SHA256."""
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(input_key_material)


def aes_gcm_encrypt(key: bytes, plaintext: bytes, nonce: bytes, associated_data: bytes = b"") -> bytes:
    """AES-256-GCM seal returning ciphertext||tag."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(key).encrypt(nonce, plaintext, associated_data)


def aes_gcm_decrypt(key: bytes, sealed: bytes, nonce: bytes, associated_data: bytes = b"") -> bytes:
    """AES-256-GCM open; raises InvalidTag mapped to SecurityError on failure."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.exceptions import InvalidTag
    try:
        return AESGCM(key).decrypt(nonce, sealed, associated_data)
    except InvalidTag as exc:
        raise SecurityError("AES-GCM authentication failed: ciphertext or AAD was modified") from exc


def new_nonce(length: int = 12) -> bytes:
    """Fresh random nonce from the stdlib CSPRNG."""
    return secrets.token_bytes(length)


def random_bytes(length: int) -> bytes:
    return secrets.token_bytes(length)


def random_int(modulus: int) -> int:
    """Uniform random integer in [0, modulus) from the stdlib CSPRNG."""
    return secrets.randbelow(modulus)


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def keyed_hash(key: bytes, data: bytes) -> bytes:
    """HMAC-SHA256 used as a PRG core and binding tag."""
    return hmac.new(key, data, hashlib.sha256).digest()


def prg_bytes(seed: bytes, context: bytes, length: int) -> bytes:
    """Deterministic pseudorandom stream: HMAC-PRG in counter mode.

    Used for secure-aggregation masks. The construction is the standard NIST
    HMAC-DRBG style counter mode over a secret seed and context label.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    blocks = (length + 31) // 32
    return b"".join(keyed_hash(seed, context + b"|" + block_index.to_bytes(8, "big"))
                    for block_index in range(blocks))[:length]


def prg_integers(seed: bytes, context: bytes, count: int, modulus: int) -> list[int]:
    """Deterministic pseudorandom field elements in [0, modulus)."""
    stream = prg_bytes(seed, context, count * 16)
    values: list[int] = []
    for index in range(count):
        chunk = stream[index * 16:(index + 1) * 16]
        values.append(int.from_bytes(chunk, "big") % modulus)
    return values
