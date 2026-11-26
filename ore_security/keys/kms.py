"""Local software KMS adapter: lifecycle, envelope encryption, recovery (steps 13-19).

This adapter demonstrates the full key-management contract in software: generation
per purpose, use auditing, rotation with historical verification, revocation, DEK/KEK
envelope encryption with rewrap-on-rotation, and passphrase-protected backups. Key
material never leaves this class except inside encrypted backup bundles; descriptors
exposed to callers contain metadata only. Production deployments must substitute an
HSM- or OS-keystore-backed adapter with the same interface.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from ..canonical import canonical_bytes, digest_bytes
from ..contracts import KeyDescriptor, KeyPurpose, KeyStatus
from ..crypto import (
    aes_gcm_decrypt,
    aes_gcm_encrypt,
    ed25519_sign,
    ed25519_verify,
    generate_ed25519_keypair,
    hkdf_derive,
    new_nonce,
    random_bytes,
)
from ..exceptions import (
    KeyManagementError,
    KeyNotActiveError,
    KeyNotFoundError,
    KMSUnavailableError,
)

Clock = Callable[[], float]
DEFAULT_KEY_VALIDITY_SECONDS = 90 * 24 * 3600.0


def _validity_end(not_before: float, desired_end: float) -> float:
    """Clamp a validity end so intervals stay non-degenerate under frozen clocks."""
    if desired_end > not_before:
        return desired_end
    return not_before + 1e-6

PURPOSE_ALGORITHMS: dict[KeyPurpose, frozenset[str]] = {
    KeyPurpose.SIGNING: frozenset({"Ed25519"}),
    KeyPurpose.CREDENTIAL: frozenset({"Ed25519"}),
    KeyPurpose.ENCRYPTION: frozenset({"AES-256-GCM"}),
    KeyPurpose.KEY_ENCRYPTION: frozenset({"AES-256-GCM"}),
    KeyPurpose.KEY_AGREEMENT: frozenset({"X25519"}),
    KeyPurpose.AUTHENTICATION: frozenset({"Ed25519", "AES-256-GCM"}),
}


class SecretCanaryRegistry:
    """In-memory registry of secret byte strings used to block log leakage.

    The registry exists to enforce the acceptance test that private keys never appear
    in logs or task records: audit events are scanned against registered canaries
    before persistence. Canary values live only in process memory.
    """

    def __init__(self) -> None:
        self._canaries: set[bytes] = set()

    def register(self, secret: bytes) -> None:
        """Register a secret value that must never appear in persisted text."""
        if not isinstance(secret, (bytes, bytearray)) or len(secret) == 0:
            raise ValueError("canary must be non-empty bytes")
        self._canaries.add(bytes(secret))

    def find_in_text(self, text: str) -> bytes | None:
        """Return the registered canary found in the text, or None."""
        if not text:
            return None
        for canary in self._canaries:
            if canary.hex() in text:
                return canary
            try:
                if canary.decode("utf-8") in text:
                    return canary
            except UnicodeDecodeError:
                continue
        return None

    def canary_count(self) -> int:
        return len(self._canaries)


class LocalKMS:
    """Software key-management service with audited use and protected backups."""

    def __init__(self, *, clock: Clock = time.time,
                 canary_registry: SecretCanaryRegistry | None = None,
                 kms_ref: str = "local-software-kms") -> None:
        self.clock = clock
        self.canary_registry = canary_registry or SecretCanaryRegistry()
        self.kms_ref = kms_ref
        self._keys: dict[str, dict[str, Any]] = {}
        self._use_audit: list[dict[str, Any]] = []
        self._outage = False

    # -- lifecycle -----------------------------------------------------------

    def generate_key(self, purpose: KeyPurpose | str, algorithm: str | None = None,
                     *, validity_seconds: float = DEFAULT_KEY_VALIDITY_SECONDS,
                     key_id: str | None = None) -> KeyDescriptor:
        """Generate a key for one purpose; algorithm must be compatible with it."""
        resolved_purpose = KeyPurpose(purpose) if isinstance(purpose, str) else purpose
        if algorithm is None:
            algorithm = next(iter(PURPOSE_ALGORITHMS[resolved_purpose]))
        if algorithm not in PURPOSE_ALGORITHMS[resolved_purpose]:
            raise KeyManagementError(
                f"algorithm {algorithm} is not permitted for purpose {resolved_purpose.value}"
            )
        self._fail_if_outage()
        now = self.clock()
        resolved_id = key_id or ("key-" + uuid.uuid4().hex[:16])
        if resolved_id in self._keys:
            raise KeyManagementError(f"key id already exists: {resolved_id}")
        if algorithm == "Ed25519":
            private_bytes, public_bytes = generate_ed25519_keypair()
            material: tuple[bytes, bytes | None] = (private_bytes, None)
            public_commitment = digest_bytes(public_bytes)
        elif algorithm == "X25519":
            from ..crypto import generate_x25519_keypair
            private_bytes, public_bytes = generate_x25519_keypair()
            material = (private_bytes, None)
            public_commitment = digest_bytes(public_bytes)
        elif algorithm == "AES-256-GCM":
            private_bytes = random_bytes(32)
            material = (private_bytes, None)
            public_commitment = digest_bytes(private_bytes)  # commitment, not the key
        else:  # pragma: no cover - guarded above
            raise KeyManagementError(f"unsupported algorithm: {algorithm}")
        self.canary_registry.register(private_bytes)
        descriptor = KeyDescriptor(
            key_id=resolved_id, purpose=resolved_purpose, algorithm=algorithm,
            status=KeyStatus.ACTIVE, not_before=now, not_after=now + float(validity_seconds),
            kms_ref=self.kms_ref, public_commitment=public_commitment,
        )
        self._keys[resolved_id] = {"descriptor": descriptor, "material": material,
                                   "public": public_bytes if algorithm != "AES-256-GCM" else None,
                                   "revoked_at": None}
        self._use_audit.append({"key_id": resolved_id, "operation": "generate",
                                "algorithm": algorithm, "at": now})
        return descriptor

    def get_descriptor(self, key_id: str) -> KeyDescriptor:
        """Return key metadata; never secret material."""
        entry = self._keys.get(key_id)
        if entry is None:
            raise KeyNotFoundError(f"unknown key: {key_id}")
        return entry["descriptor"]

    def list_descriptors(self) -> list[KeyDescriptor]:
        """Return all key descriptors ordered by key id."""
        return [self._keys[key_id]["descriptor"] for key_id in sorted(self._keys)]

    def rotate(self, key_id: str) -> KeyDescriptor:
        """Rotate a key: mark it ROTATED and create a linked successor.

        Historical verification with the rotated key remains possible so old
        signatures stay verifiable (spec step 18).
        """
        self._fail_if_outage()
        entry = self._require(key_id)
        descriptor = entry["descriptor"]
        if descriptor.status not in (KeyStatus.ACTIVE,):
            raise KeyNotActiveError(f"only ACTIVE keys can rotate: {key_id} is {descriptor.status.value}")
        now = self.clock()
        successor = self.generate_key(descriptor.purpose, descriptor.algorithm)
        rotated = KeyDescriptor(
            key_id=key_id, purpose=descriptor.purpose, algorithm=descriptor.algorithm,
            status=KeyStatus.ROTATED, not_before=descriptor.not_before,
            not_after=_validity_end(descriptor.not_before, now),
            kms_ref=descriptor.kms_ref, public_commitment=descriptor.public_commitment,
            successor_key_id=successor.key_id,
        )
        entry["descriptor"] = rotated
        successor_entry = self._keys[successor.key_id]
        successor_entry["descriptor"] = KeyDescriptor(
            key_id=successor.key_id, purpose=successor.purpose, algorithm=successor.algorithm,
            status=KeyStatus.ACTIVE, not_before=now, not_after=successor.not_after,
            kms_ref=successor.kms_ref, public_commitment=successor.public_commitment,
            predecessor_key_id=key_id,
        )
        self._use_audit.append({"key_id": key_id, "operation": "rotate",
                                "successor": successor.key_id, "at": now})
        return successor_entry["descriptor"]

    def revoke(self, key_id: str, *, reason: str = "unspecified") -> KeyDescriptor:
        """Revoke a key: future signing is refused, historical verification context is kept."""
        self._fail_if_outage()
        entry = self._require(key_id)
        descriptor = entry["descriptor"]
        now = self.clock()
        entry["descriptor"] = KeyDescriptor(
            key_id=key_id, purpose=descriptor.purpose, algorithm=descriptor.algorithm,
            status=KeyStatus.REVOKED, not_before=descriptor.not_before,
            not_after=_validity_end(descriptor.not_before, min(descriptor.not_after, now)),
            kms_ref=descriptor.kms_ref,
            public_commitment=descriptor.public_commitment,
            successor_key_id=descriptor.successor_key_id,
            predecessor_key_id=descriptor.predecessor_key_id,
        )
        entry["revoked_at"] = now
        entry["revoke_reason"] = reason
        self._use_audit.append({"key_id": key_id, "operation": "revoke",
                                "reason": reason, "at": now})
        return entry["descriptor"]

    def mark_compromised(self, key_id: str, *, reason: str = "suspected compromise") -> KeyDescriptor:
        """Mark a key compromised: neither new signing nor verification is trusted."""
        self._fail_if_outage()
        entry = self._require(key_id)
        descriptor = entry["descriptor"]
        entry["descriptor"] = KeyDescriptor(
            key_id=key_id, purpose=descriptor.purpose, algorithm=descriptor.algorithm,
            status=KeyStatus.COMPROMISED, not_before=descriptor.not_before,
            not_after=descriptor.not_after, kms_ref=descriptor.kms_ref,
            public_commitment=descriptor.public_commitment,
            successor_key_id=descriptor.successor_key_id,
            predecessor_key_id=descriptor.predecessor_key_id,
        )
        entry["revoked_at"] = self.clock()
        entry["revoke_reason"] = reason
        self._use_audit.append({"key_id": key_id, "operation": "mark_compromised",
                                "reason": reason, "at": self.clock()})
        return entry["descriptor"]

    # -- signing and verification ---------------------------------------------

    def sign(self, key_id: str, message: bytes) -> tuple[bytes, KeyDescriptor]:
        """Sign with an ACTIVE, in-validity-window signing key; every use is audited."""
        self._fail_if_outage()
        entry = self._require(key_id)
        descriptor = entry["descriptor"]
        now = self.clock()
        if descriptor.purpose not in (KeyPurpose.SIGNING, KeyPurpose.CREDENTIAL):
            raise KeyNotActiveError(f"key {key_id} purpose {descriptor.purpose.value} cannot sign")
        if descriptor.status != KeyStatus.ACTIVE:
            raise KeyNotActiveError(
                f"key {key_id} is {descriptor.status.value}; only ACTIVE keys may sign"
            )
        if not descriptor.not_before <= now <= descriptor.not_after:
            raise KeyNotActiveError(f"key {key_id} is outside its validity interval")
        signature = ed25519_sign(entry["material"][0], message)
        self._use_audit.append({"key_id": key_id, "operation": "sign", "at": now,
                                "message_digest": digest_bytes(message)[:16]})
        return signature, descriptor

    def verify(self, key_id: str, message: bytes, signature: bytes,
               *, signature_time: float | None = None) -> bool:
        """Verify a signature; REVOKED and ROTATED keys keep historical validity.

        A signature made before revocation remains verifiable; a signature whose
        claimed time falls after revocation is rejected.
        """
        entry = self._keys.get(key_id)
        if entry is None:
            raise KeyNotFoundError(f"unknown key: {key_id}")
        descriptor = entry["descriptor"]
        if descriptor.status == KeyStatus.COMPROMISED:
            return False
        if descriptor.status == KeyStatus.REVOKED and signature_time is not None:
            if signature_time > (entry.get("revoked_at") or 0.0):
                return False
        ok = ed25519_verify(entry["public"], message, signature)
        self._use_audit.append({"key_id": key_id, "operation": "verify",
                                "at": self.clock(), "result": ok})
        return ok

    # -- symmetric protection --------------------------------------------------

    def aead_encrypt(self, key_id: str, plaintext: bytes,
                     associated_data: bytes = b"") -> tuple[bytes, bytes]:
        """Encrypt under an ACTIVE ENCRYPTION-purpose key; returns (nonce, sealed)."""
        self._fail_if_outage()
        entry = self._require(key_id)
        descriptor = entry["descriptor"]
        if descriptor.algorithm != "AES-256-GCM":
            raise KeyManagementError("aead_encrypt requires an AES-256-GCM key")
        if descriptor.status != KeyStatus.ACTIVE:
            raise KeyNotActiveError(f"key {key_id} is not ACTIVE")
        nonce = new_nonce()
        sealed = aes_gcm_encrypt(entry["material"][0], plaintext, nonce, associated_data)
        self._use_audit.append({"key_id": key_id, "operation": "aead_encrypt",
                                "at": self.clock()})
        return nonce, sealed

    def aead_decrypt(self, key_id: str, nonce: bytes, sealed: bytes,
                     associated_data: bytes = b"") -> bytes:
        """Decrypt under any non-compromised AES key (historical decryption preserved)."""
        entry = self._require(key_id)
        descriptor = entry["descriptor"]
        if descriptor.algorithm != "AES-256-GCM":
            raise KeyManagementError("aead_decrypt requires an AES-256-GCM key")
        if descriptor.status == KeyStatus.COMPROMISED:
            raise KeyNotActiveError(f"key {key_id} is compromised; decryption refused")
        plaintext = aes_gcm_decrypt(entry["material"][0], sealed, nonce, associated_data)
        self._use_audit.append({"key_id": key_id, "operation": "aead_decrypt",
                                "at": self.clock()})
        return plaintext

    # -- envelope encryption (step 17) ------------------------------------------

    def ensure_kek(self, kek_id: str = "kek-primary") -> KeyDescriptor:
        """Create the key-encryption key on first use."""
        if kek_id in self._keys:
            return self._keys[kek_id]["descriptor"]
        return self.generate_key(KeyPurpose.KEY_ENCRYPTION, "AES-256-GCM", key_id=kek_id)

    def envelope_encrypt(self, plaintext: bytes, associated_data: bytes = b"",
                         *, kek_id: str = "kek-primary") -> dict[str, Any]:
        """Encrypt with a fresh data-encryption key wrapped under the KEK.

        Returns an EnvelopeRecord: ciphertext, nonce, wrapped DEK and KEK reference.
        The DEK material exists only inside the wrapped structure.
        """
        self._fail_if_outage()
        self.ensure_kek(kek_id)
        dek = random_bytes(32)
        self.canary_registry.register(dek)
        data_nonce = new_nonce()
        sealed = aes_gcm_encrypt(dek, plaintext, data_nonce, associated_data)
        kek_entry = self._keys[kek_id]
        wrap_nonce = new_nonce()
        wrapped = aes_gcm_encrypt(kek_entry["material"][0], dek, wrap_nonce,
                                  b"kek-wrap|" + kek_id.encode("utf-8"))
        record = {
            "schema_version": "1.0.0",
            "kek_id": kek_id,
            "algorithm": "AES-256-GCM+AES-256-GCM-wrap",
            "data_nonce": data_nonce.hex(),
            "ciphertext": sealed.hex(),
            "wrapped_dek": wrapped.hex(),
            "wrap_nonce": wrap_nonce.hex(),
            "aad_digest": digest_bytes(associated_data),
        }
        self._use_audit.append({"key_id": kek_id, "operation": "envelope_encrypt",
                                "at": self.clock()})
        return record

    def envelope_decrypt(self, record: dict[str, Any],
                         associated_data: bytes = b"") -> bytes:
        """Open an envelope record with the referenced KEK."""
        self._fail_if_outage()
        kek_id = record["kek_id"]
        kek_entry = self._require(kek_id)
        if kek_entry["descriptor"].status == KeyStatus.COMPROMISED:
            raise KeyNotActiveError(f"KEK {kek_id} is compromised; envelopes must be re-encrypted")
        dek = aes_gcm_decrypt(
            kek_entry["material"][0], bytes.fromhex(record["wrapped_dek"]),
            bytes.fromhex(record["wrap_nonce"]), b"kek-wrap|" + kek_id.encode("utf-8"),
        )
        plaintext = aes_gcm_decrypt(
            dek, bytes.fromhex(record["ciphertext"]), bytes.fromhex(record["data_nonce"]),
            associated_data,
        )
        self._use_audit.append({"key_id": kek_id, "operation": "envelope_decrypt",
                                "at": self.clock()})
        return plaintext

    def rotate_kek_and_rewrap(self, envelope_record: dict[str, Any]) -> dict[str, Any]:
        """Rotate the KEK and rewrap the DEK without re-encrypting the payload."""
        old_kek_id = envelope_record["kek_id"]
        old_entry = self._require(old_kek_id)
        dek = aes_gcm_decrypt(
            old_entry["material"][0], bytes.fromhex(envelope_record["wrapped_dek"]),
            bytes.fromhex(envelope_record["wrap_nonce"]), b"kek-wrap|" + old_kek_id.encode("utf-8"),
        )
        successor = self.rotate(old_kek_id)
        new_kek_id = successor.key_id
        new_entry = self._keys[new_kek_id]
        wrap_nonce = new_nonce()
        wrapped = aes_gcm_encrypt(new_entry["material"][0], dek, wrap_nonce,
                                  b"kek-wrap|" + new_kek_id.encode("utf-8"))
        rewrapped = dict(envelope_record)
        rewrapped["kek_id"] = new_kek_id
        rewrapped["wrapped_dek"] = wrapped.hex()
        rewrapped["wrap_nonce"] = wrap_nonce.hex()
        self._use_audit.append({"key_id": new_kek_id, "operation": "rewrap",
                                "previous_kek": old_kek_id, "at": self.clock()})
        return rewrapped

    # -- backup and recovery (step 19) -------------------------------------------

    def export_backup(self, passphrase: str) -> bytes:
        """Export every key in a passphrase-encrypted backup bundle."""
        self._fail_if_outage()
        if not passphrase or len(passphrase) < 8:
            raise KeyManagementError("backup passphrase must be at least 8 characters")
        payload = {
            "schema_version": "1.0.0",
            "kms_ref": self.kms_ref,
            "exported_at": self.clock(),
            "keys": [
                {
                    "key_id": key_id,
                    "descriptor": entry["descriptor"].to_dict(),
                    "private": entry["material"][0].hex(),
                    "public": entry["public"].hex() if entry.get("public") else None,
                    "revoked_at": entry.get("revoked_at"),
                    "revoke_reason": entry.get("revoke_reason"),
                }
                for key_id, entry in sorted(self._keys.items())
            ],
        }
        salt = random_bytes(16)
        key = hkdf_derive(passphrase.encode("utf-8"), salt=salt,
                          info=b"ore-security-kms-backup", length=32)
        nonce = new_nonce()
        sealed = aes_gcm_encrypt(key, canonical_bytes(payload), nonce, b"kms-backup-v1")
        return b"OREKMS1." + salt + nonce + sealed

    @classmethod
    def restore_backup(cls, bundle: bytes, passphrase: str, *,
                       clock: Clock = time.time) -> "LocalKMS":
        """Restore a KMS from a backup bundle; wrong passphrases fail closed."""
        if not bundle.startswith(b"OREKMS1."):
            raise KeyManagementError("unrecognized KMS backup format")
        body = bundle[len(b"OREKMS1."):]
        salt, nonce, sealed = body[:16], body[16:28], body[28:]
        key = hkdf_derive(passphrase.encode("utf-8"), salt=salt,
                          info=b"ore-security-kms-backup", length=32)
        plaintext = aes_gcm_decrypt(key, sealed, nonce, b"kms-backup-v1")
        import json
        payload = json.loads(plaintext.decode("utf-8"))
        restored = cls(clock=clock)
        for item in payload["keys"]:
            descriptor_dict = dict(item["descriptor"])
            descriptor = KeyDescriptor(
                key_id=descriptor_dict["key_id"],
                purpose=KeyPurpose(descriptor_dict["purpose"]),
                algorithm=descriptor_dict["algorithm"],
                status=KeyStatus(descriptor_dict["status"]),
                not_before=descriptor_dict["not_before"],
                not_after=descriptor_dict["not_after"],
                kms_ref=descriptor_dict.get("kms_ref", "local-software-kms"),
                public_commitment=descriptor_dict.get("public_commitment", ""),
                successor_key_id=descriptor_dict.get("successor_key_id"),
                predecessor_key_id=descriptor_dict.get("predecessor_key_id"),
            )
            private_bytes = bytes.fromhex(item["private"])
            public_bytes = bytes.fromhex(item["public"]) if item.get("public") else None
            restored.canary_registry.register(private_bytes)
            restored._keys[descriptor.key_id] = {
                "descriptor": descriptor,
                "material": (private_bytes, None),
                "public": public_bytes,
                "revoked_at": item.get("revoked_at"),
                "revoke_reason": item.get("revoke_reason"),
            }
        return restored

    # -- outage simulation (step 62) ----------------------------------------------

    def simulate_outage(self, enabled: bool = True) -> None:
        """Simulate KMS unavailability; protected operations must fail closed."""
        self._outage = bool(enabled)

    def is_available(self) -> bool:
        return not self._outage

    def use_audit_log(self) -> tuple[dict[str, Any], ...]:
        """Return the key-use audit log: key ids and operations, never material."""
        return tuple(self._use_audit)

    # -- internals ------------------------------------------------------------------

    def _require(self, key_id: str) -> dict[str, Any]:
        entry = self._keys.get(key_id)
        if entry is None:
            raise KeyNotFoundError(f"unknown key: {key_id}")
        return entry

    def _fail_if_outage(self) -> None:
        if self._outage:
            raise KMSUnavailableError(
                "KMS is unavailable; protected operations fail closed rather than bypass the store"
            )
