"""Artifact Signing Service: manifests, signatures, layered verification (steps 20-21).

Signing binds an artifact to a canonical manifest (digest, version, build provenance,
signer, intended use) and an Ed25519 signature envelope. Verification recomputes the
digest, checks the signature, signer authorization and intended-use scope, and key
revocation timing. A valid signature authenticates the signer, never the truth or
safety of the content: semantic validation remains a separate, mandatory step.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from ..canonical import CANONICALIZATION_ID, DEFAULT_HASH_ALGORITHM, canonical_bytes, digest_bytes
from ..contracts import KeyDescriptor, SignedArtifactManifest
from ..exceptions import ArtifactVerificationError
from ..keys.kms import LocalKMS

Clock = Callable[[], float]


class ArtifactSigningService:
    """Sign approved artifacts and verify them before loading or execution."""

    def __init__(self, kms: LocalKMS, *, clock: Clock = time.time) -> None:
        self.kms = kms
        self.clock = clock

    def build_manifest(self, artifact_id: str, artifact_version: str, artifact_bytes: bytes,
                       *, intended_use: str, build_provenance: str = "") -> dict[str, Any]:
        """Build the unsigned canonical manifest for an artifact (step 20)."""
        return {
            "artifact_id": artifact_id,
            "artifact_version": artifact_version,
            "digest": digest_bytes(artifact_bytes),
            "hash_algorithm": DEFAULT_HASH_ALGORITHM,
            "canonicalization": CANONICALIZATION_ID,
            "intended_use": intended_use,
            "build_provenance": build_provenance,
        }

    def sign(self, manifest_fields: dict[str, Any], signer_key_id: str) -> SignedArtifactManifest:
        """Sign a manifest with an ACTIVE signing key held by the KMS."""
        payload = {
            "artifact_id": manifest_fields["artifact_id"],
            "artifact_version": manifest_fields["artifact_version"],
            "digest": manifest_fields["digest"],
            "hash_algorithm": manifest_fields.get("hash_algorithm", DEFAULT_HASH_ALGORITHM),
            "canonicalization": manifest_fields.get("canonicalization", CANONICALIZATION_ID),
            "intended_use": manifest_fields["intended_use"],
            "signer_key_id": signer_key_id,
            "build_provenance": manifest_fields.get("build_provenance", ""),
        }
        signature, descriptor = self.kms.sign(signer_key_id, canonical_bytes(payload))
        return SignedArtifactManifest(
            artifact_id=payload["artifact_id"],
            artifact_version=payload["artifact_version"],
            digest=payload["digest"],
            hash_algorithm=payload["hash_algorithm"],
            canonicalization=payload["canonicalization"],
            intended_use=payload["intended_use"],
            signer_key_id=signer_key_id,
            signature_envelope={
                "algorithm": "Ed25519",
                "key_id": signer_key_id,
                "value": signature.hex(),
                "key_status_at_signing": descriptor.status.value,
            },
            signed_at=self.clock(),
            build_provenance=payload["build_provenance"],
        )

    def verify_artifact(self, manifest: SignedArtifactManifest, artifact_bytes: bytes,
                        *, expected_use: str | None = None,
                        authorized_signers: frozenset[str] = frozenset(),
                        signature_time: float | None = None) -> dict[str, Any]:
        """Layered verification returning a report; failures raise or mark not verified.

        Layer 1 recomputes the content digest; layer 2 verifies the signature against
        the recorded key; layer 3 checks signer authorization; layer 4 checks the
        intended-use scope; layer 5 checks revocation timing. The report states
        explicitly that cryptographic validity does not establish semantic truth.
        """
        checks: list[dict[str, Any]] = []

        computed = digest_bytes(artifact_bytes)
        digest_ok = computed == manifest.digest
        checks.append({"check": "content_digest", "passed": digest_ok,
                       "detail": "recomputed SHA-256 over artifact bytes"})
        if not digest_ok:
            raise ArtifactVerificationError(
                "artifact digest mismatch: content changed after signing"
            )

        signature_ok = self.kms.verify(
            manifest.signer_key_id, canonical_bytes(manifest.signed_payload()),
            bytes.fromhex(manifest.signature_envelope["value"]),
            signature_time=manifest.signed_at if signature_time is None else signature_time,
        )
        checks.append({"check": "signature", "passed": signature_ok,
                       "detail": f"Ed25519 via key {manifest.signer_key_id}"})
        if not signature_ok:
            raise ArtifactVerificationError(
                "artifact signature verification failed (invalid, revoked-after-signing, or compromised key)"
            )

        signer_authorized = not authorized_signers or manifest.signer_key_id in authorized_signers
        checks.append({"check": "signer_authorization", "passed": signer_authorized,
                       "detail": "signer key must appear in the authorized signer set"})
        if not signer_authorized:
            raise ArtifactVerificationError(
                f"correctly signed artifact rejected: signer {manifest.signer_key_id} is not authorized"
            )

        use_ok = expected_use is None or manifest.intended_use == expected_use
        checks.append({"check": "intended_use_scope", "passed": use_ok,
                       "detail": f"manifest use {manifest.intended_use!r} vs expected {expected_use!r}"})
        if not use_ok:
            raise ArtifactVerificationError(
                f"artifact intended use {manifest.intended_use!r} does not match expected {expected_use!r}"
            )

        key_descriptor = self.kms.get_descriptor(manifest.signer_key_id)
        checks.append({"check": "key_status", "passed": key_descriptor.status.value != "COMPROMISED",
                       "detail": f"key status {key_descriptor.status.value}"})

        return {
            "verified": True,
            "checks": checks,
            "manifest": manifest.to_dict(),
            "semantic_validation_required": True,
            "note": "signature authenticates the signer and content integrity, not the truth or safety of the content",
        }
