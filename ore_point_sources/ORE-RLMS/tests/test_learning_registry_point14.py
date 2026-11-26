"""Acceptance tests: registry admission, lifecycle, lineage, and audit (Steps 6, 51-56)."""
from __future__ import annotations

import pytest

from ore_learning.audit import AUDIT_FORBIDDEN_KEYS, AuditStore, sanitize_audit_metadata
from ore_learning.contracts import (
    ApprovalRecord,
    CheckOutcome,
    DeterministicCheck,
    LifecycleEventType,
    LifecycleState,
    Modality,
    ModelManifest,
    SchemaDescriptor,
    ValidationReport,
)
from ore_learning.registry import (
    ALLOWED_TRANSITIONS,
    LifecycleTransitionError,
    ModelRegistry,
    RegistryRejection,
)
from ore_learning.registry.lineage import LineageGraph

from learning_point14_helpers import digest_for, make_manifest


@pytest.fixture()
def registry() -> ModelRegistry:
    """A registry with a validated admission path."""
    return ModelRegistry()


def test_registry_rejects_missing_artifact_digest():
    """Acceptance: registry rejects a missing artifact digest."""
    manifest = make_manifest("model.nodigest")
    object.__setattr__(manifest, "artifact_digest", "")  # simulate a missing digest
    with pytest.raises(RegistryRejection, match="digest"):
        ModelRegistry().register(manifest)


def test_registry_rejects_incompatible_schemas():
    """Acceptance: registry rejects incompatible input/output schemas."""
    manifest = make_manifest("model.badschema")
    incompatible = SchemaDescriptor("structured", "unknown-schema-version")
    broken = ModelManifest(
        model_id="model.badschema", semantic_name="x", version="1.0.0",
        artifact_digest=digest_for("bad"), architecture_family=manifest.architecture_family,
        framework="numpy", input_schema=incompatible, output_schema=incompatible,
        schema_version="unknown-schema-version",
        supported_tasks=manifest.supported_tasks,
        modalities=(Modality.STRUCTURED,),
        training_data_provenance=manifest.training_data_provenance)
    with pytest.raises(RegistryRejection, match="schema"):
        ModelRegistry().register(broken)


def test_registry_rejects_provenance_incomplete_entries():
    """Acceptance: registry rejects provenance-incomplete entries (Section 2.2)."""
    manifest = make_manifest("model.noprovenance")
    object.__setattr__(manifest, "training_data_provenance", None)
    with pytest.raises(RegistryRejection, match="provenance"):
        ModelRegistry().register(manifest)


def test_registry_records_provenance_and_license_metadata(registry: ModelRegistry):
    """Acceptance: registry records provenance and license metadata."""
    from ore_learning.contracts import CapabilityRequest, Capability

    manifest = make_manifest("model.meta", license_id="apache-2.0", dataset="ds-42")
    registry.register(manifest)
    stored = registry.require("model.meta", "1.0.0")
    assert stored.license == "apache-2.0"
    assert stored.training_data_provenance.dataset_id == "ds-42"
    assert stored.training_data_provenance.license == "apache-2.0"
    assert stored.training_data_provenance.collection_period == "2023-Q3..2024-Q2"
    assert stored.security_review.verdict == "approved"
    payload = stored.to_dict()
    assert payload["license"] == "apache-2.0"
    assert payload["training_data_provenance"]["dataset_id"] == "ds-42"
    # and the query path works with the recorded metadata
    from learning_point14_helpers import make_request
    result = registry.query(make_request())
    assert any(item.model_id == "model.meta" for item in result.matched)


def test_registry_rejects_revoked_or_quarantined_models():
    """Acceptance: registry rejects revoked or quarantined models."""
    from learning_point14_helpers import make_request

    registry = ModelRegistry()
    active = registry.register(make_manifest("model.active"))
    quarantined = registry.register(make_manifest("model.quar"))
    revoked = registry.register(make_manifest("model.revoked"))
    validated_report = ValidationReport(checks=(
        DeterministicCheck(check_id="c", validator="unit", outcome=CheckOutcome.PASS),))
    registry.transition("model.quar", "1.0.0", LifecycleState.QUARANTINED,
                        reason="drift")
    registry.transition("model.revoked", "1.0.0", LifecycleState.REVOKED, reason="license")
    result = registry.query(make_request())
    matched = {manifest.model_id for manifest in result.matched}
    assert "model.active" in matched
    assert "model.quar" not in matched and "model.revoked" not in matched
    assert "model.quar@1.0.0" in result.rejected
    assert "model.revoked@1.0.0" in result.rejected
    # manifests are immutable: transitions store a new record, so re-query
    assert active.lifecycle_state is LifecycleState.ACTIVE
    assert registry.require("model.quar", "1.0.0").lifecycle_state \
        is LifecycleState.QUARANTINED
    assert registry.require("model.revoked", "1.0.0").lifecycle_state \
        is LifecycleState.REVOKED


def test_registry_identity_is_immutable_against_digest_substitution():
    """Artifact digests prevent silent substitution (Section 9)."""
    registry = ModelRegistry()
    registry.register(make_manifest("model.stable"))
    tampered = make_manifest("model.stable", quality=0.99)
    object.__setattr__(tampered, "artifact_digest", digest_for("other"))
    with pytest.raises(RegistryRejection, match="silent substitution"):
        registry.register(tampered)
    assert registry.verify_artifact("model.stable", "1.0.0",
                                    digest_for("model.stable:1.0.0"))
    assert not registry.verify_artifact("model.stable", "1.0.0", digest_for("tamper"))


def test_lifecycle_state_machine_rejects_invalid_transitions(registry: ModelRegistry):
    """Only the documented transitions are allowed (Section 2.2)."""
    registry.register(make_manifest("model.sm", state=LifecycleState.PROPOSED))
    with pytest.raises(LifecycleTransitionError):
        registry.transition("model.sm", "1.0.0", LifecycleState.CANARY,
                            reason="cannot enter canary from proposed")
    assert LifecycleState.RETIRED not in ALLOWED_TRANSITIONS[LifecycleState.RETIRED]


def test_entering_validated_requires_verified_report(registry: ModelRegistry):
    """Validation gates: unverified reports cannot validate a model."""
    registry.register(make_manifest("model.gate", state=LifecycleState.PROPOSED))
    with pytest.raises(LifecycleTransitionError, match="verified"):
        registry.transition("model.gate", "1.0.0", LifecycleState.VALIDATED,
                            reason="no report")
    good = ValidationReport(checks=(
        DeterministicCheck(check_id="c", validator="unit", outcome=CheckOutcome.PASS),))
    registry.transition("model.gate", "1.0.0", LifecycleState.VALIDATED,
                        reason="independently validated", validation=good)
    assert registry.require("model.gate", "1.0.0").lifecycle_state \
        is LifecycleState.VALIDATED


def test_high_risk_promotion_requires_recorded_approval(registry: ModelRegistry):
    """Acceptance: human approval is required for configured high-risk promotion."""
    registry.register(make_manifest("model.promo", state=LifecycleState.VALIDATED))
    with pytest.raises(LifecycleTransitionError, match="approval"):
        registry.promote("model.promo", "1.0.0", risk_class="critical",
                         reason="production replacement")
    approval = ApprovalRecord(approval_id="appr-1", approver="reviewer@ore",
                              scope="production model replacement",
                              reviewed_evidence=("bench:held-out",))
    event = registry.promote("model.promo", "1.0.0", risk_class="critical",
                             reason="production replacement", approval=approval)
    assert event.event_type is LifecycleEventType.ACTIVATED
    assert registry.require("model.promo", "1.0.0").lifecycle_state \
        is LifecycleState.ACTIVE
    # low-risk promotion does not need approval
    registry.register(make_manifest("model.lowrisk", state=LifecycleState.VALIDATED))
    registry.promote("model.lowrisk", "1.0.0", risk_class="low", reason="routine")
    assert registry.require("model.lowrisk", "1.0.0").lifecycle_state \
        is LifecycleState.ACTIVE


def test_quarantine_reentry_requires_reevaluation_and_approval(registry: ModelRegistry):
    """Step 51: re-entry requires reevaluation and approval."""
    registry.register(make_manifest("model.reentry", state=LifecycleState.ACTIVE))
    registry.quarantine("model.reentry", "1.0.0", reason="degraded quality")
    report = ValidationReport(checks=(
        DeterministicCheck(check_id="c", validator="unit", outcome=CheckOutcome.PASS),))
    with pytest.raises(LifecycleTransitionError, match="re-evaluation"):
        registry.transition("model.reentry", "1.0.0", LifecycleState.VALIDATED,
                            reason="re-entry", validation=report)
    approval = ApprovalRecord("appr-2", "reviewer@ore", "quarantine re-entry")
    registry.transition("model.reentry", "1.0.0", LifecycleState.VALIDATED,
                        reason="re-entry after re-evaluation", validation=report,
                        approval=approval)
    assert registry.require("model.reentry", "1.0.0").lifecycle_state \
        is LifecycleState.VALIDATED


def test_retirement_preserves_lineage_and_evaluation_evidence(registry: ModelRegistry):
    """Acceptance: retirement preserves historical lineage and evaluation evidence."""
    manifest = registry.register(make_manifest("model.retire"))
    record = registry.lineage.record("model.retire", "1.0.0")
    record.add_link("ds-open-1", "training_dataset")
    record.add_link("bench:v2", "evaluation_suite", metadata={"metric": "accuracy"})
    record.add_evaluation("held-out", {"accuracy": 0.91})
    registry.retire("model.retire", "1.0.0", reason="dominated by a better model")
    assert registry.require("model.retire", "1.0.0").lifecycle_state \
        is LifecycleState.RETIRED
    history = registry.lineage.history("model.retire", "1.0.0")
    held_out = [entry for entry in history if entry["suite"] == "held-out"]
    assert held_out and held_out[0]["metrics"]["accuracy"] == 0.91
    links = registry.lineage.record("model.retire", "1.0.0").links
    assert {link.relation for link in links} >= {"training_dataset",
                                                 "evaluation_suite"}


def test_lifecycle_events_persist_and_link_to_task_and_model(tmp_path):
    """Acceptance: lifecycle events persist and link to task and model."""
    from sqlite3 import connect

    database_path = tmp_path / "audit.sqlite3"
    store = AuditStore(database_path)
    registry = ModelRegistry(event_listener=lambda event: store.record(
        "ore_learning.registry", event.event_type.value, "lifecycle",
        model_id=event.model_id, model_version=event.model_version,
        task_id=event.task_id,
        metadata={"from_state": event.from_state.value if event.from_state else None,
                  "to_state": event.to_state.value, "reason": event.reason}))
    registry.register(make_manifest("model.audited"))
    registry.quarantine("model.audited", "1.0.0", reason="drift", task_id="task-9")
    store.close()
    # reopen from disk and query by model and task
    reopened = AuditStore(database_path)
    by_model = reopened.query(model_id="model.audited")
    by_task = reopened.query(task_id="task-9")
    assert len(by_model) >= 2
    assert {event["operation"] for event in by_model} >= {"registered", "quarantined"}
    assert by_task and by_task[-1]["model_id"] == "model.audited"
    assert [event["sequence"] for event in by_model] == \
        sorted(event["sequence"] for event in by_model)
    reopened.close()


def test_audit_records_do_not_leak_raw_sensitive_payloads(tmp_path):
    """Acceptance: audit records do not leak raw sensitive payloads."""
    store = AuditStore(tmp_path / "audit.sqlite3")
    store.record("ore_learning.inference", "execute", "execution",
                 metadata={"raw_payload": {"secret": "s"},
                           "inputs": [1, 2, 3],
                           "secret": "hunter2",
                           "api_key": "xyz",
                           "generated_code": "import os",
                           "model_id": "m",
                           "task_id": "t",
                           "duration_ms": 1.2})
    violations = store.assert_no_forbidden_keys()
    assert violations == []
    stored = store.query(event_kind="execution")[0]["metadata"]
    assert "raw_payload" not in stored and "secret" not in stored
    assert "api_key" not in stored and "generated_code" not in stored
    assert stored["model_id"] == "m"  # allowlisted references survive
    # the sanitizer drops forbidden keys even outside the store
    cleaned = sanitize_audit_metadata({key: 1 for key in AUDIT_FORBIDDEN_KEYS})
    assert cleaned == {}
    store.close()


def test_audit_corrections_become_new_linked_events(tmp_path):
    """Step 54: corrections become new linked events, never in-place rewrites."""
    store = AuditStore(tmp_path / "audit.sqlite3")
    original = store.record("ore_learning", "select", "selection",
                            metadata={"detail": "utility=0.5"})
    corrected = store.correct(original.event_id, "ore_learning", "select",
                              metadata={"detail": "utility=0.7 (corrected weights)"})
    events = [event for event in store.query() if event["operation"] == "select"]
    assert len(events) == 2
    assert events[1]["event_kind"] == "correction"
    assert corrected.correction_of == original.event_id
    assert events[0]["metadata"]["detail"].startswith("utility=0.5")  # untouched
    store.close()


def test_lineage_graph_tracks_ancestry():
    """Step 53: lineage links models to base artifacts and datasets."""
    graph = LineageGraph()
    base = make_manifest("model.base")
    child = make_manifest("model.child", version="2.0.0")
    graph.register(base)
    child_record = graph.register(child, base_artifact=base.model_id)
    child_record.add_link("ds-adapt-1", "fine_tuning_dataset")
    ancestors = graph.ancestors("model.child", "2.0.0")
    assert base.model_id in ancestors
    summary = graph.describe()
    assert summary["records"] == 2 and summary["links"] >= 2
