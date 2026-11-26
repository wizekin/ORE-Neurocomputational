"""Acceptance tests: key management, artifact signing, integrity and audit chains (Phases C-D)."""
from __future__ import annotations

import pytest

from ore_security.audit import AuditLedger, RedactionEngine, SecurityEvent, security_event
from ore_security.exceptions import (
    ArtifactVerificationError,
    CheckpointError,
    IntegrityError,
    KeyNotActiveError,
    KMSUnavailableError,
    RedactionError,
)
from ore_security.integrity import (
    CheckpointService,
    ContentHasher,
    ExternalAnchorVault,
    HashChain,
    MerkleTree,
)
from ore_security.keys import LocalKMS
from ore_security.signing import ArtifactSigningService


class FrozenClock:
    def __init__(self, start: float = 2000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def clock():
    return FrozenClock()


@pytest.fixture()
def kms(clock):
    return LocalKMS(clock=clock)


@pytest.fixture()
def signing(kms, clock):
    return ArtifactSigningService(kms, clock=clock)


def _signed_manifest(signing: ArtifactSigningService, key_id: str, artifact_id: str = "art-1",
                     version: str = "1.0.0", payload: bytes = b"artifact-bytes",
                     intended_use: str = "model-load"):
    fields = signing.build_manifest(artifact_id, version, payload, intended_use=intended_use)
    return signing.sign(fields, key_id)


def test_sensitive_data_at_rest_is_encrypted_under_policy(kms):
    """Acceptance: sensitive data at rest is encrypted under policy (envelope encryption)."""
    plaintext = b"highly sensitive USG record content"
    record = kms.envelope_encrypt(plaintext, b"usg-record")
    assert bytes.fromhex(record["ciphertext"]) != plaintext
    assert "dek" not in record or not record.get("dek")  # no plaintext DEK ever stored
    assert kms.envelope_decrypt(record, b"usg-record") == plaintext
    with pytest.raises(Exception):
        kms.envelope_decrypt(record, b"wrong-aad")  # AAD binding fails closed


def test_envelope_rewrap_on_kek_rotation(kms):
    """KEK rotation rewraps the DEK without re-encrypting the payload (step 17-18)."""
    plaintext = b"archived telemetry blob"
    record = kms.envelope_encrypt(plaintext, b"telemetry")
    original_ciphertext = record["ciphertext"]
    rewrapped = kms.rotate_kek_and_rewrap(record)
    assert rewrapped["ciphertext"] == original_ciphertext
    assert rewrapped["kek_id"] != record["kek_id"]
    assert kms.envelope_decrypt(rewrapped, b"telemetry") == plaintext


def test_private_keys_never_appear_in_logs_or_task_records(kms, clock):
    """Acceptance: private keys never appear in logs, source control or task records."""
    descriptor = kms.generate_key("SIGNING", "Ed25519")
    assert kms.canary_registry.canary_count() >= 1
    forbidden = {"private_key", "material", "secret", "seed", "private_bytes"}
    assert forbidden & set(KeyDescriptor_fields()) == set()
    engine = RedactionEngine(kms.canary_registry)
    leaked = kms._keys[descriptor.key_id]["material"][0].hex()
    with pytest.raises(RedactionError):
        engine.sanitize({"note": leaked})
    with pytest.raises(RedactionError):
        engine.sanitize({"config": "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----"})
    cleaned = engine.sanitize({"public_info": "safe-value", "passphrase": "hidden"})
    assert cleaned["passphrase"] == "[REDACTED]"


def KeyDescriptor_fields():
    from ore_security.contracts import KeyDescriptor
    return tuple(KeyDescriptor.__dataclass_fields__)


def test_key_rotation_preserves_historical_verification(kms):
    """Acceptance: key rotation preserves historical signature verification."""
    descriptor = kms.generate_key("SIGNING", "Ed25519")
    signature, _ = kms.sign(descriptor.key_id, b"release-manifest")
    successor = kms.rotate(descriptor.key_id)
    assert successor.key_id != descriptor.key_id
    assert successor.predecessor_key_id == descriptor.key_id
    assert kms.get_descriptor(descriptor.key_id).status.value == "ROTATED"
    assert kms.verify(descriptor.key_id, b"release-manifest", signature)
    with pytest.raises(KeyNotActiveError):
        kms.sign(descriptor.key_id, b"new-release")


def test_revoked_signing_keys_cannot_sign_new_releases(kms, clock):
    """Acceptance: revoked signing keys cannot sign new trusted releases."""
    descriptor = kms.generate_key("SIGNING", "Ed25519")
    signature, _ = kms.sign(descriptor.key_id, b"older-release")
    clock.now += 100.0
    kms.revoke(descriptor.key_id, reason="key rotation policy")
    with pytest.raises(KeyNotActiveError):
        kms.sign(descriptor.key_id, b"new-release")
    # historical verification context is preserved for signatures made before revocation
    assert kms.verify(descriptor.key_id, b"older-release", signature,
                      signature_time=clock.now - 100.0)
    # a signature claimed after revocation is rejected
    assert not kms.verify(descriptor.key_id, b"older-release", signature,
                          signature_time=clock.now + 1.0)


def test_compromised_keys_fail_verification(kms):
    """Marked-compromised keys are refused for both signing and verification."""
    descriptor = kms.generate_key("SIGNING", "Ed25519")
    signature, _ = kms.sign(descriptor.key_id, b"payload")
    kms.mark_compromised(descriptor.key_id)
    assert not kms.verify(descriptor.key_id, b"payload", signature)
    with pytest.raises(KeyNotActiveError):
        kms.sign(descriptor.key_id, b"payload")


def test_modified_artifact_fails_verification(signing, kms):
    """Acceptance: modified artifact fails digest/signature verification."""
    kms.generate_key("SIGNING", "Ed25519", key_id="key-signer")
    manifest = _signed_manifest(signing, "key-signer")
    with pytest.raises(ArtifactVerificationError, match="digest"):
        signing.verify_artifact(manifest, b"artifact-bytes-tampered",
                                expected_use="model-load")


def test_incorrect_signature_fails_verification(signing, kms):
    """A corrupted signature envelope is rejected."""
    import dataclasses
    kms.generate_key("SIGNING", "Ed25519", key_id="key-signer")
    manifest = _signed_manifest(signing, "key-signer")
    broken = dataclasses.replace(
        manifest,
        signature_envelope={"algorithm": "Ed25519", "key_id": "key-signer",
                            "value": "00" * 64},
    )
    with pytest.raises(ArtifactVerificationError, match="signature"):
        signing.verify_artifact(broken, b"artifact-bytes", expected_use="model-load")


def test_signed_but_unauthorized_artifact_is_rejected(signing, kms):
    """Acceptance: correctly signed but unauthorized artifact is rejected."""
    kms.generate_key("SIGNING", "Ed25519", key_id="key-authorized")
    kms.generate_key("SIGNING", "Ed25519", key_id="key-rogue")
    manifest = _signed_manifest(signing, "key-rogue")
    with pytest.raises(ArtifactVerificationError, match="not authorized"):
        signing.verify_artifact(manifest, b"artifact-bytes", expected_use="model-load",
                                authorized_signers=frozenset({"key-authorized"}))


def test_manifest_binds_version_digest_and_intended_use(signing, kms):
    """Acceptance: artifact manifest binds signature to version, digest and intended use."""
    import dataclasses
    kms.generate_key("SIGNING", "Ed25519", key_id="key-signer")
    manifest = _signed_manifest(signing, "key-signer")
    # digest binding: same content, different claimed version must not verify
    tampered = dataclasses.replace(manifest, artifact_version="9.9.9")
    with pytest.raises(ArtifactVerificationError):
        signing.verify_artifact(tampered, b"artifact-bytes", expected_use="model-load")
    # intended-use scope binding
    with pytest.raises(ArtifactVerificationError, match="intended use"):
        signing.verify_artifact(manifest, b"artifact-bytes", expected_use="firmware-flash")
    report = signing.verify_artifact(manifest, b"artifact-bytes", expected_use="model-load")
    assert report["verified"]
    assert report["semantic_validation_required"] is True


def test_content_hasher_stores_algorithm_and_canonicalization():
    """Hash records carry algorithm and canonicalization identifiers (step 22)."""
    hasher = ContentHasher()
    record = hasher.hash_bytes(b"content")
    assert record["algorithm"] == "SHA-256"
    assert record["canonicalization"] == "raw-bytes-v1"
    assert hasher.verify_bytes(b"content", record["digest"])
    assert not hasher.verify_bytes(b"content2", record["digest"])
    with pytest.raises(ValueError):
        ContentHasher("MD5")
    with pytest.raises(ValueError):
        ContentHasher("SHA-1")


def test_hash_chain_alteration_is_detected_against_checkpoint():
    """Acceptance: hash-chain alteration is detected against a trusted checkpoint."""
    chain = HashChain()
    for index in range(5):
        chain.append({"event": f"e{index}", "seq_hint": index})
    trusted_head = chain.head
    chain.append({"event": "e5-tampered", "seq_hint": 5})
    with pytest.raises(IntegrityError):
        chain.verify(expected_head=trusted_head)


def test_chain_event_modification_is_detected():
    """Modifying a stored event breaks its chained digest."""
    chain = HashChain()
    for index in range(4):
        chain.append({"event": f"e{index}"})
    entries = list(chain.entries())
    entries[2]["event"]["event"] = "rewritten-history"
    tampered = HashChain.from_entries(entries)
    with pytest.raises(IntegrityError):
        tampered.verify()


def test_deletion_and_reordering_of_committed_events_is_detected():
    """Acceptance: deletion/reordering of committed audit events is detected."""
    chain = HashChain()
    for index in range(6):
        chain.append({"event": f"e{index}"})
    checkpoint_head = chain.head
    # deletion of the last two events (truncation against the checkpoint)
    truncated = HashChain.from_entries(list(chain.entries())[:4])
    with pytest.raises(CheckpointError):
        truncated.verify(expected_head=checkpoint_head)
    # reordering: swap two middle events
    entries = list(chain.entries())
    entries[1], entries[2] = entries[2], entries[1]
    reordered = HashChain.from_entries(entries)
    with pytest.raises(IntegrityError):
        reordered.verify()


def test_merkle_inclusion_proof_verifies_for_included_record():
    """Acceptance: Merkle inclusion proof verifies for an included record."""
    records = [{"record": f"r{index}"} for index in range(7)]
    tree = MerkleTree(records)
    proof = tree.inclusion_proof(3)
    assert MerkleTree.verify_inclusion(records[3], 3, proof, tree.root)
    for index in range(len(records)):
        assert MerkleTree.verify_inclusion(
            records[index], index, tree.inclusion_proof(index), tree.root)


def test_modified_leaf_or_incorrect_proof_fails_merkle_verification():
    """Acceptance: modified leaf or incorrect proof fails Merkle verification."""
    records = [{"record": f"r{index}"} for index in range(5)]
    tree = MerkleTree(records)
    proof = tree.inclusion_proof(2)
    assert not MerkleTree.verify_inclusion({"record": "modified"}, 2, proof, tree.root)
    assert not MerkleTree.verify_inclusion(records[2], 4, proof, tree.root)
    wrong_root = "f" * 64
    assert not MerkleTree.verify_inclusion(records[2], 2, proof, wrong_root)


def test_signed_checkpoints_verify_and_anchor_in_separate_domain(kms, clock):
    """Checkpoints sign the chain head and Merkle root and anchor externally (step 23-24)."""
    key = kms.generate_key("SIGNING", "Ed25519", key_id="key-checkpoint")
    service = CheckpointService(kms, key.key_id, clock=clock)
    vault = ExternalAnchorVault(clock=clock)
    ledger = AuditLedger(checkpoint_service=service, anchor_vault=vault, clock=clock)
    for index in range(4):
        ledger.record(security_event(f"op-{index}", result="allowed",
                                     correlation_id="corr-x", timestamp=clock.now))
    info = ledger.create_checkpoint()
    assert vault.anchor_count() == 1
    report = service.verify_checkpoint(
        _checkpoint_from(info), anchor_vault=vault, anchor_id=info["anchor_id"],
    )
    assert report["signature_valid"] and report["anchor_valid"] is True
    # an unanchored checkpoint reference fails closed
    with pytest.raises(CheckpointError):
        service.verify_checkpoint(_checkpoint_from(info), anchor_vault=vault,
                                  anchor_id="missing-anchor")


def _checkpoint_from(info):
    from ore_security.contracts import IntegrityCheckpoint
    payload = info["checkpoint"]
    return IntegrityCheckpoint(
        checkpoint_id=payload["checkpoint_id"], range_start=payload["range_start"],
        range_end=payload["range_end"], chain_head=payload["chain_head"],
        merkle_root=payload["merkle_root"], tree_version=payload["tree_version"],
        signature=payload["signature"], signer_key_id=payload["signer_key_id"],
        timestamp=payload["timestamp"],
    )


def test_audit_checkpoints_verify_after_backup_restoration(kms, clock):
    """Acceptance: audit checkpoints verify after backup restoration."""
    key = kms.generate_key("SIGNING", "Ed25519", key_id="key-checkpoint")
    service = CheckpointService(kms, key.key_id, clock=clock)
    ledger = AuditLedger(checkpoint_service=service, clock=clock)
    for index in range(3):
        ledger.record(security_event(f"op-{index}", result="allowed",
                                     correlation_id="corr-restore", timestamp=clock.now))
    ledger.create_checkpoint()
    backup = ledger.export_backup()
    restored = AuditLedger.restore_backup(backup, clock=clock)
    verification = restored.verify_history()
    assert verification["valid"] and verification.get("checkpoint_verified") is True
    assert restored.event_count() == ledger.event_count()


def test_security_events_require_correlation_id():
    """Events without correlation ids or timestamps are refused by schema validation."""
    with pytest.raises(ValueError):
        SecurityEvent(action="op", result="allowed", correlation_id="",
                      timestamp=1000.0)
    with pytest.raises(ValueError):
        SecurityEvent(action="op", result="allowed", correlation_id="corr",
                      timestamp=0.0)
    event = security_event("op", result="allowed")
    assert event.correlation_id.startswith("corr-")
