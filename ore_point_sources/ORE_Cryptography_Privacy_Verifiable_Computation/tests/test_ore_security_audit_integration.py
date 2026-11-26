"""Acceptance tests: audit, incident response, ORE integration and the pipeline (Phase J)."""
from __future__ import annotations

import sqlite3

import pytest

from ore_security import build_default_trust_stack, provider_report
from ore_security.audit import AuditLedger, security_event
from ore_security.audit.service import AuditService, IncidentResponse
from ore_security.contracts import PrincipalKind
from ore_security.exceptions import (
    CredentialRevokedError,
    KMSUnavailableError,
    PolicyEngineFailure,
    SecurityError,
)
from ore_security.integrity.checkpoints import CheckpointService
from ore_security.keys import LocalKMS
from ore_security.pipeline import CoordinatorGuard, SecuredRecordStore
from ore_security.policy import DataClassification, PolicyRule, ResourceDescriptor


class FrozenClock:
    def __init__(self, start: float = 7000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def clock():
    return FrozenClock()


@pytest.fixture()
def stack(clock):
    stack = build_default_trust_stack(clock=clock)
    ids = stack["identity_service"]
    engine = stack["policy_engine"]
    kms = stack["kms"]
    ids.register_identity("svc-writer", PrincipalKind.SERVICE, roles=("WRITER",))
    ids.register_identity("svc-reader", PrincipalKind.SERVICE, roles=("READER",))
    ids.register_identity("svc-agent", PrincipalKind.SERVICE, roles=("AGENT",))
    kms.generate_key("SIGNING", "Ed25519", key_id="key-usg-signer")
    engine.resource_registry.register(ResourceDescriptor(
        "usg://records", DataClassification.CONFIDENTIAL, "usg-owner",
        allowed_purposes=frozenset({"record-ingest"}),
    ))
    engine.add_rule(PolicyRule(
        "allow-writer-ingest", "svc-writer", frozenset({"usg.record.ingest"}),
        ("usg://records",), DataClassification.CONFIDENTIAL,
        required_purpose="record-ingest",
    ))
    token, _ = ids.issue_credential("svc-writer", "ore-service", ("usg:write",), ttl=900)
    reader_token, _ = ids.issue_credential("svc-reader", "ore-service", ("usg:read",), ttl=900)
    stack["writer_token"] = token
    stack["reader_token"] = reader_token
    return stack


def test_key_compromise_drill_revokes_credentials_and_blocks_new_use(stack, clock):
    """Acceptance: key compromise drill revokes credentials and blocks new use."""
    ids = stack["identity_service"]
    kms = stack["kms"]
    # the agent holds a credential and a key named after its identity
    agent_token, _ = ids.issue_credential("svc-agent", "ore-service", ("agent:use",), ttl=900)
    agent_key = kms.generate_key("SIGNING", "Ed25519", key_id="key-svc-agent-signing")
    signature, _ = kms.sign(agent_key.key_id, b"agent artifact")

    incident = IncidentResponse(ids, kms, stack["audit"])
    report = incident.compromise_principal("svc-agent", correlation_id="incident-1")
    assert "identity_revoked" in report["actions"]
    assert agent_key.key_id in report["affected_keys"]

    with pytest.raises((CredentialRevokedError, SecurityError)):
        ids.authenticate_token(agent_token, expected_audience="ore-service")
    with pytest.raises(SecurityError):
        kms.sign(agent_key.key_id, b"new artifact")
    assert not kms.verify(agent_key.key_id, b"agent artifact", signature)  # compromised


def test_kms_outage_recovery_does_not_bypass_access_policy(stack, clock):
    """Acceptance: KMS outage recovery does not bypass access policy."""
    kms = stack["kms"]
    revoked = kms.generate_key("SIGNING", "Ed25519", key_id="key-revoked-before-outage")
    kms.revoke("key-revoked-before-outage", reason="routine rotation")

    incident = IncidentResponse(stack["identity_service"], kms, stack["audit"])
    drill = incident.kms_outage_recovery_drill("recovery passphrase 1234")
    assert drill["outage_failed_closed"] is True
    assert drill["restored_key_count"] > 0

    # during an outage, protected operations fail closed
    kms.simulate_outage(True)
    with pytest.raises(KMSUnavailableError):
        kms.sign("key-usg-signer", b"payload")
    kms.simulate_outage(False)

    # restore from backup and confirm revocations survived restoration
    backup = kms.export_backup("recovery passphrase 1234")
    restored = LocalKMS.restore_backup(backup, "recovery passphrase 1234")
    with pytest.raises(SecurityError):
        restored.sign("key-revoked-before-outage", b"payload")  # still revoked
    wrong_passphrase = pytest.raises(SecurityError)
    with wrong_passphrase:
        LocalKMS.restore_backup(backup, "incorrect passphrase")


def test_sqlite_audit_annotations_contain_no_private_keys(stack, clock, tmp_path):
    """Acceptance: SQLite audit annotations contain no private keys or raw secrets."""
    database_path = tmp_path / "audit.sqlite3"
    ledger = stack["ledger"]
    ledger._sqlite_path = database_path
    ledger._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        from ore_security.audit.ledger import AUDIT_SCHEMA
        connection.executescript(AUDIT_SCHEMA)

    stack["audit"].record_decision(
        action="usg.record.ingest", principal_id="svc-writer",
        resource_ref="usg://records/r1", policy_decision="ALLOWED_BY_RULE",
        decision_reason="rule matched", result="allowed",
        correlation_id="corr-sqlite-1", key_id="key-usg-signer",
        algorithm="Ed25519",
    )
    report = ledger.sqlite_integrity_report()
    assert report["sqlite_configured"]
    assert report["persisted_events"] >= 1
    assert report["secrets_in_storage"] is False
    assert report["forbidden_columns_present"] == []

    # scan the raw database file for the registered key-material canaries
    raw = database_path.read_bytes()
    canaries = stack["kms"].canary_registry
    found = canaries.find_in_text(raw.hex())
    assert found is None, "private key material leaked into the SQLite audit store"


def test_security_events_include_correlation_ids_and_policy_decisions(stack):
    """Acceptance: security events include correlation IDs and policy decisions."""
    pipeline = stack["pipeline"]
    result = pipeline.execute(
        action="usg.record.ingest", resource="usg://records/x",
        purpose="record-ingest", credential_token=stack["reader_token"],
        expected_audience="ore-service",
    )
    assert not result.allowed
    events = stack["audit"].query(result.correlation_id)
    assert len(events) == 1
    event = events[0]["event"]
    assert event["correlation_id"] == result.correlation_id
    assert event["policy_decision"] == "NO_MATCHING_POLICY"
    assert event["action"] == "usg.record.ingest"
    assert event["result"] == "denied"


def test_audit_checkpoints_verify_after_sqlite_backup_restoration(stack, clock):
    """Acceptance: audit checkpoints verify after backup restoration (SQLite-backed)."""
    ledger = stack["ledger"]
    for index in range(3):
        ledger.record(security_event(f"op-{index}", result="allowed",
                                     correlation_id="corr-bk", timestamp=clock.now))
    ledger.create_checkpoint()
    backup = ledger.export_backup()
    restored = AuditLedger.restore_backup(backup, clock=clock)
    verification = restored.verify_history()
    assert verification["valid"] is True
    assert verification["checkpoint_verified"] is True


def test_coordinator_cannot_override_security_policy_by_priority(stack):
    """Acceptance: ORE coordinator cannot override security policy by raising task priority."""
    guard = CoordinatorGuard(stack["pipeline"])
    for priority in (1, 5, 99, 1000):
        result = guard.request_with_priority(
            task_priority=priority,
            action="usg.record.ingest", resource="usg://records/x",
            purpose="record-ingest", credential_token=stack["reader_token"],
            expected_audience="ore-service",
        )
        assert not result.allowed
    # the rejected override attempts are audited
    override_events = stack["audit"].query(result.correlation_id)
    assert any(event["event"]["action"] == "coordinator.priority_override_attempt"
               for event in override_events)
    # and a properly authorized caller succeeds regardless of priority marking
    allowed = guard.request_with_priority(
        task_priority=1,
        action="usg.record.ingest", resource="usg://records/x",
        purpose="record-ingest", credential_token=stack["writer_token"],
        expected_audience="ore-service",
    )
    assert allowed.allowed


def test_usg_updates_require_security_checks_and_semantic_validation(stack):
    """Acceptance: USG updates require security checks and semantic validation."""
    signing = stack["signing"]
    pipeline = stack["pipeline"]
    store = SecuredRecordStore(pipeline)

    payload = b'{"claim": "supported-by-evidence"}'
    fields = signing.build_manifest("rec-ok", "1.0.0", payload, intended_use="usg-record")
    manifest = signing.sign(fields, "key-usg-signer")

    accepted = store.ingest(
        "rec-ok", payload, manifest, stack["writer_token"],
        purpose="record-ingest", resource_uri="usg://records/rec-ok",
        semantic_validator=lambda data: b"supported-by-evidence" in data,
        expected_use="usg-record", authorized_signers=frozenset({"key-usg-signer"}),
    )
    assert accepted["stored"]
    assert store.sink["rec-ok"] == payload

    # intact signature, but the content is semantically false for this validator
    false_payload = b'{"claim": "the-moon-is-cheese"}'
    false_fields = signing.build_manifest("rec-false", "1.0.0", false_payload,
                                          intended_use="usg-record")
    false_manifest = signing.sign(false_fields, "key-usg-signer")
    refused = store.ingest(
        "rec-false", false_payload, false_manifest, stack["writer_token"],
        purpose="record-ingest", resource_uri="usg://records/rec-false",
        semantic_validator=lambda data: b"supported-by-evidence" in data,
        expected_use="usg-record", authorized_signers=frozenset({"key-usg-signer"}),
    )
    assert not refused["stored"]
    assert "semantic validation failed" in refused["reason"]
    assert "rec-false" not in store.sink

    # unauthorized writer cannot store even valid content
    unauthorized = store.ingest(
        "rec-noauth", payload, manifest, stack["reader_token"],
        purpose="record-ingest", resource_uri="usg://records/rec-noauth",
        semantic_validator=lambda data: True, expected_use="usg-record",
        authorized_signers=frozenset({"key-usg-signer"}),
    )
    assert not unauthorized["stored"]


def test_proof_verifier_failure_prevents_operation_marked_verified(stack, clock):
    """Acceptance: proof verifier failure prevents operation from being marked verified."""
    from ore_security.proofs import SchnorrWitness

    proof_service = stack["proof_service"]
    witness = SchnorrWitness(555)
    y = witness.public_value(proof_service.adapter.group)
    statement = proof_service.build_statement(
        "stmt-v", {"y": y}, task_id="task-v", task_nonce="n-v", ttl=100.0,
    )
    record = proof_service.prove(statement, witness)
    record["proof"] = dict(record["proof"])
    record["proof"]["s"] = str(int(record["proof"]["s"]) + 1)  # corrupt the response

    result = stack["pipeline"].execute(
        action="usg.record.ingest", resource="usg://records/proof-1",
        purpose="record-ingest", credential_token=stack["writer_token"],
        expected_audience="ore-service", operation=lambda: "computed",
        proof=(record, statement),
    )
    assert result.allowed            # authorization itself passed
    assert result.verified is False  # but the operation is not marked verified
    assert result.proof_verification is not None
    assert result.proof_verification["verified"] is False
    assert "not marked verified" in result.failure_reason


def test_cryptographic_adapters_use_vetted_libraries():
    """Acceptance: cryptographic adapters use vetted libraries rather than hand-written primitives."""
    report = provider_report()
    assert report["library"] == "cryptography"
    assert report["version"]
    assert "Ed25519" in report["primitives"] and "AES-256-GCM" in report["primitives"]
    # the Schnorr, Merkle, sharing and Paillier protocols are composed on vetted
    # primitives; the note documents that boundary explicitly
    assert "vetted primitives" in report["reference_protocols_note"]


def test_policy_engine_failure_fails_closed_through_pipeline(stack):
    """A simulated policy outage denies protected operations end to end."""
    engine = stack["policy_engine"]
    engine.simulate_engine_failure(True)
    result = stack["pipeline"].execute(
        action="usg.record.ingest", resource="usg://records/x",
        purpose="record-ingest", credential_token=stack["writer_token"],
        expected_audience="ore-service",
    )
    assert not result.allowed
    assert result.error_category == "POLICY_ENGINE_FAILURE"
    engine.simulate_engine_failure(False)


def test_register_annotations_with_ore_logger(tmp_path):
    """ore_logger integration: annotations index every ore_security symbol (step 59)."""
    pytest.importorskip("ore_logger")
    from ore_security.annotations import register_annotations

    database_path = tmp_path / "ore_logger.sqlite3"
    report = register_annotations(database_path)
    assert report["scope"] == "ore_security"
    assert report["annotation_table"] == "annotations"
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT COUNT(*) FROM annotations WHERE module_name LIKE 'ore_security%'"
        ).fetchone()
    assert rows[0] > 0


def test_record_pipeline_run_to_ore_logger_sqlite(stack, tmp_path):
    """ore_logger integration: protected-operation runs persist as linked events."""
    pytest.importorskip("ore_logger")
    from ore_security.annotations import record_pipeline_run_to_sqlite

    result = stack["pipeline"].execute(
        action="usg.record.ingest", resource="usg://records/x",
        purpose="record-ingest", credential_token=stack["writer_token"],
        expected_audience="ore-service", operation=lambda: "value",
    )
    assert result.allowed and result.executed
    database_path = tmp_path / "runs.sqlite3"
    report = record_pipeline_run_to_sqlite(result, database_path)
    assert report["events_written"] == 1
    assert report["run_id"]
    with sqlite3.connect(database_path) as connection:
        run = connection.execute(
            "SELECT status, entry_kind FROM runs WHERE run_id = ?",
            (report["run_id"],),
        ).fetchone()
    assert run == ("completed", "point20_trust_layer")


def test_register_with_ore_contracts_catalog_and_registry():
    """ore_contracts integration: typed interfaces and the module manifest register cleanly."""
    pytest.importorskip("ore_contracts")
    from ore_contracts import ContractCatalog, ModuleRegistry
    from ore_security.integration import register_with_ore

    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    report = register_with_ore(catalog=catalog, module_registry=registry)
    assert len(report["contracts_registered"]) == 12
    assert report["module_registered"] is True
    assert report["module_id"] == "ore.security.trust"
    # contract lookups resolve with the exact version
    definition = catalog.get_contract("ore.security.policy_decision", "1.0.0")
    assert definition is not None
    assert definition.model_type is not None
