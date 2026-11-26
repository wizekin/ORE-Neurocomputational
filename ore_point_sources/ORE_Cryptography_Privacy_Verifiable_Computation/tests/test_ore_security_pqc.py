"""Acceptance tests: post-quantum readiness (Phase I)."""
from __future__ import annotations

import pytest

from ore_security.exceptions import AlgorithmRejectedError
from ore_security.pqc import (
    CryptoInventory,
    CryptoPolicy,
    HybridSignatureEnvelope,
    MigrationManager,
    ReferenceHashBasedOTS,
    parse_algorithm_identifier,
    registered_algorithm_ids,
)
from ore_security.pqc.hybrid import interoperability_test
from ore_security.pqc.identifiers import PQC_FAMILIES


def test_pqc_algorithm_identifiers_parse_unambiguously():
    """Acceptance: PQC algorithm identifiers and parameter sets are parsed unambiguously."""
    for family, parameter_sets in PQC_FAMILIES.items():
        for parameter in parameter_sets:
            identifier = parse_algorithm_identifier(f"{family}-{parameter}")
            assert identifier.algorithm_id == f"{family}-{parameter}"
            assert identifier.is_pqc
            # round trip through serialization
            again = parse_algorithm_identifier(identifier.serialize())
            assert again.algorithm_id == identifier.algorithm_id
            assert again.security_category == identifier.security_category
    # normalization accepts case/separator variants, canonicalizing to one form
    assert parse_algorithm_identifier("ml_kem 768").algorithm_id == "ML-KEM-768"
    assert parse_algorithm_identifier("  ml-dsa-65 ").algorithm_id == "ML-DSA-65"
    assert parse_algorithm_identifier("slh_dsa 192S").algorithm_id == "SLH-DSA-192s"
    # classical identifiers remain parseable for inventory purposes
    classical = parse_algorithm_identifier("Ed25519")
    assert not classical.is_pqc and classical.kind == "signature"


def test_unknown_or_ambiguous_identifiers_are_rejected():
    """Anything outside the registry is rejected rather than guessed."""
    for bad in ("ML-KEM-999", "ML-KEM", "FrodoKEM-640", "RSA-2048", "SHA-1",
                "Rainbow-Classic", "", "ML-"):
        with pytest.raises(AlgorithmRejectedError):
            parse_algorithm_identifier(bad)


def test_unsupported_or_downgraded_algorithms_are_rejected():
    """Acceptance: unsupported or downgraded cryptographic algorithms are rejected."""
    policy = CryptoPolicy(min_security_category=3)
    with pytest.raises(AlgorithmRejectedError, match="security category"):
        policy.check("ML-KEM-512", purpose="key-establishment")
    with pytest.raises(AlgorithmRejectedError, match="not in the allowed set"):
        CryptoPolicy(allowed_algorithm_ids=frozenset({"ML-KEM-768"})).check(
            "ML-DSA-65", purpose="signature")
    # classical establishment alone cannot protect long-lived confidentiality
    # (harvest-now/decrypt-later), even at the default security category
    with pytest.raises(AlgorithmRejectedError, match="hybrid"):
        CryptoPolicy().check("X25519", purpose="key-establishment",
                             data_lifetime_years=20.0)
    # purpose mismatch is a downgrade attempt and is refused
    with pytest.raises(AlgorithmRejectedError, match="cannot serve purpose"):
        CryptoPolicy().check("Ed25519", purpose="encryption")
    # short-lived classical establishment remains permitted
    assert CryptoPolicy().check("X25519", purpose="key-establishment",
                                data_lifetime_years=1.0).algorithm_id == "X25519"


def test_pqc_interoperability_tests_cover_establishment_and_signatures():
    """Acceptance: PQC interoperability tests cover key establishment/signature verification."""
    report = interoperability_test()
    assert report["hybrid_establishment"] is True
    assert report["hybrid_signature"] is True
    assert report["tamper_detection"] is True
    assert report["passed"]


def test_hybrid_signature_envelope_requires_both_legs():
    """Hybrid verification fails when either the classical or the PQC leg fails."""
    holder = HybridSignatureEnvelope()
    payload = b"release-bundle-v1"
    envelope = holder.sign(payload)
    assert holder.verify(payload, envelope)["verified"]
    tampered = dict(envelope)
    tampered["classical_leg"] = dict(envelope["classical_leg"])
    tampered["classical_leg"]["value"] = "00" * 64
    failed = holder.verify(payload, tampered)
    assert not failed["verified"] and failed["failed_leg"] == "classical"
    tampered_pq = dict(envelope)
    tampered_pq["pqc_leg"] = dict(envelope["pqc_leg"])
    tampered_pq["pqc_leg"]["values"] = ["00" * 32] * 256
    failed_pq = holder.verify(payload, tampered_pq)
    assert not failed_pq["verified"] and failed_pq["failed_leg"] == "pqc"


def test_hash_based_reference_ots_is_strictly_one_time():
    """The reference Lamport leg enforces one-time use like a hash-based signature."""
    ots = ReferenceHashBasedOTS()
    signature = ots.sign(b"first message")
    assert ReferenceHashBasedOTS.verify(ots.public_key, b"first message", signature)
    with pytest.raises(Exception):
        ots.sign(b"second message")


def test_inventory_prioritizes_harvest_now_decrypt_later():
    """Migration priority puts long-lived classical establishment first (step 53)."""
    inventory = CryptoInventory()
    inventory.record_usage("kms.transport", "X25519", purpose="key-establishment",
                           data_lifetime_years=25.0)
    inventory.record_usage("kms.signing", "Ed25519", purpose="signature")
    inventory.record_usage("signing.release", "ML-DSA-65", purpose="signature")
    report = inventory.report()
    assert report["pq_algorithms_in_use"] == ["ML-DSA-65"]
    assert report["long_lived_classical_establishment"][0]["component"] == "kms.transport"
    manager = MigrationManager(inventory)
    priorities = manager.prioritize()
    assert priorities[0]["priority"] == "HIGH"
    assert "harvest-now" in priorities[0]["driver"]
    manager.set_status("X25519", "MIGRATING", note="hybrid rollout in progress")
    assert manager.status_report()["plans"]["X25519"]["status"] == "MIGRATING"


def test_registered_algorithm_ids_cover_standardized_families():
    """The registry tracks the NIST-standardized families (step 54)."""
    identifiers = registered_algorithm_ids()
    for expected in ("ML-KEM-512", "ML-KEM-768", "ML-KEM-1024",
                     "ML-DSA-44", "ML-DSA-65", "ML-DSA-87",
                     "SLH-DSA-128s", "SLH-DSA-256f"):
        assert expected in identifiers
