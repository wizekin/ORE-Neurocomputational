"""Phase A/B acceptance tests: contracts, canonical identity, ingestion, index registry (Steps 1-12)."""
from __future__ import annotations

import pytest

from ore_data.contracts import (
    DataClass,
    DataContractRegistry,
    DataFamilyContract,
    DataStoreInventory,
    ErrorBudget,
    ExactnessClass,
    FreshnessContract,
    MutabilityContract,
    StoreDescriptor,
    UpdateMode,
    canonical_key,
    canonicalize,
    content_id,
    default_data_contracts,
    default_store_inventory,
    required_exactness_for,
    verify_content_id,
)
from ore_data.ingestion import (
    AuthoritativeStore,
    ChangeEventType,
    DataChangeEvent,
    EventReceipt,
    IngestionService,
    Outbox,
    ProvenanceInfo,
    Quarantine,
    RecordStatus,
    StorageRouter,
    ValidationPolicy,
    validate_record,
)
from ore_data.index_registry import (
    DerivedIndexRegistry,
    DerivedIndexSpec,
    IndexHealth,
    IndexKind,
    reconcile_derived_indexes,
)


# ---------------------------------------------------------------------------
# Phase A (Steps 1-6)


def test_store_inventory_records_current_stores_before_proposing_structures():
    """Step 1: the inventory audits current stores; unmeasured entries stay marked."""
    inventory = default_store_inventory()
    owners = inventory.owners()
    assert "ore_usg" in owners
    assert "ore_logger" in owners
    assert "ore_workspace" in owners
    assert all(store.measured is False for store in inventory.stores)
    with pytest.raises(ValueError):
        StoreDescriptor(
            store_id="", store_kind="x", authoritative_owner="o",
            schema_summary="", approximate_write_rate_per_s=-1,
            approximate_read_rate_per_s=0, data_volume_bytes=0,
            retention_rule="", backup_behavior="",
        )


def test_data_families_are_classified_with_contracts():
    """Step 2: every family gets a data-class contract; approximate never replaces authority."""
    registry = default_data_contracts()
    families = {contract.family_id: contract for contract in registry.families()}
    assert families["ore.usg.semantic_records"].data_class is DataClass.EXACT_TRANSACTIONAL
    assert families["ore.logger.execution_events"].data_class is DataClass.APPEND_ONLY_EVENT
    assert families["ore.data.immutable_artifacts"].data_class is DataClass.IMMUTABLE_ARTIFACT
    assert families["ore.data.bloom_membership"].data_class is DataClass.DERIVED_INDEX
    assert families["ore.data.retrieval_telemetry"].data_class is DataClass.APPROXIMATE_TELEMETRY
    for contract in families.values():
        assert contract.approximate_may_replace_authority() is False
    with pytest.raises(KeyError):
        registry.require("ore.unclassified.family")


def test_canonicalization_is_deterministic_and_domain_separated():
    """Step 3: field order and framing produce one identity for byte-identical artifacts."""
    first = {"b": 2, "a": 1, "nested": {"z": True, "n": None}}
    second = {"a": 1, "b": 2, "nested": {"n": None, "z": True}}
    assert canonicalize(first) == canonicalize(second)
    identity = content_id(canonicalize(first), domain="ore.usg.concept")
    assert verify_content_id(canonicalize(second), identity, domain="ore.usg.concept")
    # Domain separation: a different domain never shares the hash namespace.
    assert content_id(b"x", domain="a") != content_id(b"x", domain="ab")
    assert content_id(canonicalize(first), domain="ore.usg.concept") != content_id(
        canonicalize(first), domain="ore.data.artifact"
    )
    assert canonical_key("CONCEPT", "alpha") == "CONCEPT:alpha"
    with pytest.raises(ValueError):
        canonicalize({"bad": float("nan")})
    with pytest.raises(ValueError):
        canonicalize({"bad": object()})


def test_exactness_classes_are_declared_per_operation():
    """Step 4: USG identity resolution is exact; similarity only nominates candidates."""
    assert required_exactness_for("concept_identity_resolution") is ExactnessClass.EXACT_MEMBERSHIP
    assert required_exactness_for("usage_frequency_estimate") is ExactnessClass.BOUNDED_ERROR_ESTIMATE
    assert required_exactness_for("duplicate_candidate_search") is ExactnessClass.CANDIDATE_SIMILARITY
    assert required_exactness_for("artifact_integrity") is ExactnessClass.CRYPTOGRAPHIC_INTEGRITY
    with pytest.raises(ValueError):
        required_exactness_for("undeclared_operation")


def test_freshness_and_error_budgets_are_enforced():
    """Steps 5-6: freshness contracts bound lag; error budgets validate their inputs."""
    freshness = FreshnessContract("bloom", UpdateMode.QUEUED, 1_000)
    assert freshness.within_lag(999) and freshness.within_lag(1_000)
    assert not freshness.within_lag(1_001)
    budget = ErrorBudget("bloom", 1 << 21, 50_000, 0.01, 100.0, 50.0, 3.0)
    assert budget.target_error_rate == 0.01
    with pytest.raises(ValueError):
        ErrorBudget("bad", 0, 1, 0.5, 1, 1, 1)
    with pytest.raises(ValueError):
        FreshnessContract("bad", UpdateMode.QUEUED, -1)
    # Derived families must declare freshness; approximate families a budget.
    with pytest.raises(ValueError):
        DataFamilyContract(
            family_id="bad.derived", data_class=DataClass.DERIVED_INDEX,
            exactness=ExactnessClass.BOUNDED_ERROR_ESTIMATE, authoritative_owner="x",
            mutability=MutabilityContract(
                deletable=False, correctable=False, versioned=True, replicable=False
            ),
        )


# ---------------------------------------------------------------------------
# Phase B (Steps 7-12)


def _policy() -> ValidationPolicy:
    return ValidationPolicy(
        family_id="ore.workspace.findings",
        allowed_schema_versions=frozenset({"0.1.0"}),
        required_fields=("entity_key", "record_type"),
        max_record_bytes=256,
        authorized_sources=frozenset({"ore_test"}),
    )


def test_invalid_records_are_rejected_or_quarantined_and_index_nothing():
    """Step 7: invalid records update no index; quarantine holds them."""
    policy = _policy()
    provenance = ProvenanceInfo(source="ore_test", actor="agent-1")
    bad_schema = {"schema_version": "9.9.9", "entity_key": "k", "record_type": "CONCEPT"}
    result = validate_record(bad_schema, provenance, policy)
    assert result.valid is False
    assert result.quarantined is True
    unauthorized = validate_record(
        {"schema_version": "0.1.0", "entity_key": "k", "record_type": "X"},
        ProvenanceInfo(source="intruder", actor="agent-2"),
        policy,
    )
    assert not unauthorized.valid
    missing = validate_record(
        {"schema_version": "0.1.0"}, provenance, policy
    )
    assert not missing.valid
    oversized = validate_record(
        {"schema_version": "0.1.0", "entity_key": "k", "record_type": "X", "payload": "x" * 400},
        provenance,
        policy,
    )
    assert not oversized.valid
    quarantine = Quarantine()
    quarantine.hold(bad_schema, result.errors, policy.family_id)
    assert quarantine.size() == 1


def test_retries_do_not_double_apply_authoritative_mutations():
    """Steps 8-9: the same idempotency key applies the authoritative mutation once."""
    store = AuthoritativeStore()
    record = {"schema_version": "0.1.0", "entity_key": "k-1", "v": 1}
    first = store.commit_authoritative(record, "idem-1")
    retry = store.commit_authoritative(record, "idem-1")
    assert first.status is RecordStatus.COMMITTED
    assert retry.status is RecordStatus.DUPLICATE_APPLIED
    assert retry.sequence == first.sequence
    assert store.watermark == first.sequence
    assert len(store.outbox) == 1  # only one change event published
    different = store.commit_authoritative(
        {"schema_version": "0.1.0", "entity_key": "k-1", "v": 2}, "idem-2"
    )
    assert different.status is RecordStatus.COMMITTED
    assert different.sequence > first.sequence
    assert store.lookup("k-1")["v"] == 2
    deleted = store.delete("k-1", idempotency_key="idem-3")
    assert deleted.status is RecordStatus.COMMITTED
    assert store.lookup("k-1") is None
    replay = store.delete("k-1", idempotency_key="idem-3")
    assert replay.status is RecordStatus.DUPLICATE_APPLIED


def test_change_events_are_versioned_and_replay_safe():
    """Step 10: events carry sequence metadata; duplicate event IDs are detected."""
    outbox = Outbox()
    event = DataChangeEvent(
        event_id="evt-1", event_type=ChangeEventType.INSERT, entity_key="k",
        source="s", schema_version="0.1.0", sequence=1,
    )
    receipt = outbox.append(event)
    assert isinstance(receipt, EventReceipt) and receipt.duplicate is False
    replay = outbox.append(event)
    assert replay.duplicate is True
    second = outbox.append(
        DataChangeEvent(
            event_id="evt-2", event_type=ChangeEventType.UPDATE, entity_key="k",
            source="s", schema_version="0.1.0", sequence=2,
        )
    )
    assert [e.sequence for e in outbox.pull(0)] == [1, 2]
    assert [e.event_id for e in outbox.pull(1)] == ["evt-2"]
    assert second.duplicate is False


def test_ingestion_service_routes_by_storage_class():
    """Step 11: routing follows the data-family contract; LSM stays dormant until justified."""
    contracts = {c.family_id: c for c in default_data_contracts().families()}
    dormant = StorageRouter(lsm_justified=False)
    assert dormant.route(contracts["ore.workspace.findings"]) == "transactional"
    assert dormant.route(contracts["ore.usg.semantic_records"]) == "transactional"
    assert dormant.route(contracts["ore.data.immutable_artifacts"]) == "content_addressed"
    assert dormant.route(contracts["ore.logger.execution_events"]) == "transactional"
    justified = StorageRouter(lsm_justified=True)
    assert justified.route(contracts["ore.logger.execution_events"]) == "lsm_backed"
    assert justified.route(contracts["ore.data.bloom_membership"]) == "derived_index"


def test_ingestion_service_end_to_end():
    """Steps 7-10 as a service: valid commit + event, invalid quarantine."""
    service = IngestionService(policies={_policy().family_id: _policy()})
    provenance = ProvenanceInfo(source="ore_test", actor="agent-1")
    receipt = service.ingest(
        {"schema_version": "0.1.0", "entity_key": "concept-a", "record_type": "CONCEPT"},
        provenance,
        "idem-a",
    )
    assert receipt is not None and receipt.applied
    assert service.store.contains("concept-a")
    invalid = service.ingest(
        {"schema_version": "0.0.0", "entity_key": "bad", "record_type": "X"},
        provenance,
        "idem-b",
    )
    assert invalid is None
    assert service.quarantine.size() == 1
    assert service.store.watermark == 1  # invalid record committed nothing
    delivered = []
    assert service.replay_events(delivered.append) == 1
    assert delivered[0].entity_key == "concept-a"


def test_derived_index_registry_tracks_watermarks_and_rebuild_plans():
    """Step 12: derived structures declare source, watermark and rebuild path."""
    registry = DerivedIndexRegistry()
    spec = DerivedIndexSpec(
        index_id="ore.data.index.test", kind=IndexKind.BLOOM_FILTER,
        authoritative_source="store", schema_version="0.1.0", build_watermark=0,
        parameters={"m": 1024}, checksum="abc", rebuild_procedure="rebuild",
        memory_budget_bytes=1024,
    )
    registry.register(spec)
    assert registry.require("ore.data.index.test").kind is IndexKind.BLOOM_FILTER
    with pytest.raises(ValueError):
        registry.register(spec)
    registry.update_watermark("ore.data.index.test", 50)
    report = registry.health("ore.data.index.test", authoritative_watermark=50)
    assert report.health is IndexHealth.FRESH and report.usable_for_precheck
    stale_report = registry.health("ore.data.index.test", authoritative_watermark=200)
    assert stale_report.health is IndexHealth.STALE  # no freshness contract: watermark equality required
    assert stale_report.usable_for_precheck is False
    registry.mark_incomplete("ore.data.index.test")
    incomplete = registry.health("ore.data.index.test", authoritative_watermark=200)
    assert incomplete.health is IndexHealth.INCOMPLETE
    assert incomplete.usable_for_precheck is False


def test_reconciliation_reports_lag_and_plans_rebuilds():
    """Step 19: reconciliation exposes watermark lag and deterministic rebuild plans."""
    from ore_data.contracts import FreshnessContract, UpdateMode

    registry = DerivedIndexRegistry()
    spec = DerivedIndexSpec(
        index_id="ore.data.index.bloom", kind=IndexKind.BLOOM_FILTER,
        authoritative_source="store", schema_version="0.1.0", build_watermark=10,
        parameters={}, checksum="abc", rebuild_procedure="rebuild_bloom",
        memory_budget_bytes=1024,
        freshness=FreshnessContract("bloom", UpdateMode.QUEUED, 5),
    )
    registry.register(spec)
    report = reconcile_derived_indexes(registry, authoritative_watermark=100)
    assert report.authoritative_watermark == 100
    assert report.indexes_stale == 1
    assert report.rebuild_plans[0].to_watermark == 100
    assert report.rebuild_plans[0].rebuild_procedure == "rebuild_bloom"
    registry.update_watermark("ore.data.index.bloom", 100)
    healthy = reconcile_derived_indexes(registry, authoritative_watermark=100)
    assert healthy.indexes_stale == 0 and not healthy.rebuild_plans


def test_registry_covers_56_steps_in_10_phases():
    """The operation registry maps all 56 steps across the 10 phases."""
    from ore_data.registry import OPERATION_REGISTRY, operation_count, phases

    assert operation_count() == 56
    assert phases() == tuple(f"Phase {letter}" for letter in "ABCDEFGHIJ")
    steps = [spec.source_step for spec in OPERATION_REGISTRY]
    assert steps == list(range(1, 57))
    modules = {spec.implementation_module for spec in OPERATION_REGISTRY}
    assert "ore_data.contracts" in modules
    assert "ore_data.probabilistic.bloom" in modules
    assert "ore_data.probabilistic.count_min" in modules
    assert "ore_data.probabilistic.hyperloglog" in modules
    assert "ore_data.streaming.quantiles" in modules
    assert "ore_data.streaming.heavy_hitters" in modules
    assert "ore_data.similarity.minhash" in modules
    assert "ore_data.similarity.lsh" in modules
    assert "ore_data.similarity.simhash" in modules
    assert "ore_data.storage.lsm_adapter" in modules
    assert "ore_data.storage.content_addressed" in modules
    assert "ore_data.integrity.merkle" in modules
    assert "ore_data.recovery.erasure" in modules
