"""Integration acceptance tests: pipeline end to end, coordinator, CROS, workspace, USG (section 6)."""
from __future__ import annotations

from pathlib import Path

import pytest

from ore_data.contracts import ExactnessClass, canonicalize, content_id
from ore_data.controller_adapter import (
    BackpressureController,
    CROSDataBudgetAdapter,
    ConceptIdentity,
    InMemoryConceptResolver,
    MainBrainCoordinatorAdapter,
    StorageTelemetry,
    USGSemanticVerificationAdapter,
    UsgIndexResolver,
    WorkspaceDataPublisher,
    cros_cannot_relax_correctness,
    telemetry_to_capability_demand,
)
from ore_data.ingestion import ProvenanceInfo
from ore_data.observability import (
    IndexHealthMonitor,
    assert_estimates_carry_metadata,
    approximate_analytics_degrade_gracefully,
    estimate_metadata,
)
from ore_data.pipeline import (
    DataInfrastructurePipeline,
    PipelineConfig,
    build_pipeline,
)
from ore_data.probabilistic.bloom import MembershipVerdict
from ore_data.registry import operation_count, phases
from ore_data.similarity.minhash import verify_byte_identity, verify_candidate_similarity


def _provenance() -> ProvenanceInfo:
    return ProvenanceInfo(source="ore_test", actor="agent-1")


def _record(key: str, record_type: str = "CONCEPT") -> dict[str, object]:
    return {
        "schema_version": "0.1.0",
        "entity_key": key,
        "record_type": record_type,
        "family": "ore.workspace.findings",
    }


def test_pipeline_processes_records_end_to_end(tmp_path: Path):
    """Source section 2: validation, commit, events, routing, indexes, candidates."""
    pipeline = build_pipeline(workspace_root=tmp_path)
    text_a = "the quick brown fox jumps over the lazy dog again and again"
    text_b = "the quick brown fox jumps over the lazy dog again and more"
    text_c = "completely unrelated words about distributed storage systems"
    first = pipeline.process_record(
        _record("concept-a"), idempotency_key="id-a", provenance=_provenance(), text=text_a,
        observation=11.0,
    )
    assert first.status == "committed"
    assert first.commit_receipt is not None and first.commit_receipt.sequence == 1
    assert first.event_receipts[0]["event_type"] == "insert"
    second = pipeline.process_record(
        _record("concept-b"), idempotency_key="id-b", provenance=_provenance(), text=text_b,
        observation=13.0,
    )
    third = pipeline.process_record(
        _record("concept-c"), idempotency_key="id-c", provenance=_provenance(), text=text_c,
        observation=17.0,
    )
    # Similar candidates were nominated (Step 40) but nothing merged (Step 42).
    assert "concept-a" in second.similarity_candidates
    assert second.verification_required is True
    verification = pipeline.verify_similarity(
        "concept-a", "concept-b", texts={"concept-a": text_a, "concept-b": text_b}
    )
    assert verification.exact_value is not None and verification.exact_value > 0.5
    # Membership two-stage path (Step 17).
    present = pipeline.membership("concept-a")
    assert present["exact_member"] is True
    assert present["hint"]["verdict"] == MembershipVerdict.POSSIBLY_PRESENT.value
    absent = pipeline.membership("missing-key")
    assert absent["exact_member"] is False
    assert absent["hint"]["verdict"] == MembershipVerdict.DEFINITELY_ABSENT.value
    assert absent["hint"]["safe_to_skip_exact_lookup"] is True
    # Estimates with metadata (Step 5.1).
    frequency = pipeline.frequency("concept-a")
    assert frequency.exact is False and frequency.estimate >= 1.0
    cardinality = pipeline.cardinality()
    assert cardinality.exact is False
    quantiles = pipeline.quantiles((0.5, 0.9, 0.99))
    assert len(quantiles) == 3 and all(q.exact is False for q in quantiles)
    heavy = pipeline.heavy_hitters(3)
    assert heavy and heavy[0].exact is False


def test_pipeline_rejects_invalid_records_and_updates_no_index(tmp_path: Path):
    """Step 7: invalid records never update indexes or integrity commitments."""
    pipeline = build_pipeline(
        workspace_root=tmp_path, authorized_sources=frozenset({"ore_test"})
    )
    invalid = dict(_record("bad-key"), schema_version="9.9.9")
    outcome = pipeline.process_record(invalid, idempotency_key="bad-1", provenance=_provenance())
    assert outcome.status == "rejected_or_quarantined"
    assert pipeline.ingestion.store.watermark == 0
    assert pipeline.bloom.insertions == 0
    assert pipeline.ingestion.quarantine.size() == 1
    assert pipeline.count_min.total_weight == 0.0
    unauthorized = pipeline.process_record(
        _record("bad-2"),
        idempotency_key="bad-2",
        provenance=ProvenanceInfo(source="intruder", actor="x"),
    )
    assert unauthorized.status == "rejected_or_quarantined"


def test_pipeline_event_retries_never_duplicate_authoritative_mutations(tmp_path: Path):
    """Step 8/10: retries with one idempotency key apply exactly once."""
    pipeline = build_pipeline(workspace_root=tmp_path)
    first = pipeline.process_record(
        _record("concept-x"), idempotency_key="idem-x", provenance=_provenance()
    )
    retry = pipeline.process_record(
        _record("concept-x"), idempotency_key="idem-x", provenance=_provenance()
    )
    assert first.status == "committed"
    assert retry.status == "duplicate_applied"
    assert pipeline.ingestion.store.watermark == 1
    assert pipeline.frequency("concept-x").estimate == 1.0  # sketches not double-updated
    assert len(pipeline.ingestion.store.outbox) == 1


def test_pipeline_artifacts_route_to_content_addressed_storage(tmp_path: Path):
    """Steps 11, 50-53: immutable artifacts go to the object store, deduplicated."""
    pipeline = build_pipeline(workspace_root=tmp_path)
    payload = b"validated algoir specification payload"
    first = pipeline.process_record(
        _record("artifact-1", record_type="ALGOIR"),
        idempotency_key="art-1",
        provenance=_provenance(),
        artifact=payload,
        artifact_class="algo_ir_specification",
    )
    assert first.content_id is not None
    assert first.content_reused is False
    artifact = pipeline.content_store.get_verified(first.content_id)
    assert artifact.content == payload
    canonical = canonicalize(_record("artifact-1", record_type="ALGOIR"))
    expected = content_id(payload, pipeline.content_store.domain)
    assert expected == first.content_id
    deduplicated = pipeline.process_record(
        _record("artifact-2", record_type="ALGOIR"),
        idempotency_key="art-2",
        provenance=_provenance(),
        artifact=payload,
        artifact_class="algo_ir_specification",
    )
    assert deduplicated.content_reused is True
    assert deduplicated.content_id == first.content_id


def test_pipeline_integrity_snapshots_and_verification(tmp_path: Path):
    """Steps 54-55 through the pipeline: commitments, proofs, tamper detection."""
    pipeline = build_pipeline(workspace_root=tmp_path)
    records = [f"record-{i}".encode("utf-8") for i in range(7)]
    commitment = pipeline.build_integrity_snapshot(records, "snap-1")
    assert len(commitment.root) == 64
    from ore_data.integrity.merkle import MerkleTree

    tree = MerkleTree(records)
    verified = pipeline.verify_record("snap-1", records[4], 4, tree)
    assert verified["consistent"] is True
    tampered = pipeline.verify_record("snap-1", b"record-999", 4, tree)
    assert tampered["consistent"] is False


def test_pipeline_reconciliation_and_index_health(tmp_path: Path):
    """Steps 19, 12 + section 8.4: health reports, graceful degradation, rebuild plans."""
    pipeline = build_pipeline(workspace_root=tmp_path)
    for index in range(10):
        pipeline.process_record(
            _record(f"concept-{index}"),
            idempotency_key=f"id-{index}",
            provenance=_provenance(),
        )
    health = pipeline.index_health()
    assert set(health) == {
        "ore.data.index.bloom_membership",
        "ore.data.index.count_min_frequency",
        "ore.data.index.hyperloglog_cardinality",
        "ore.data.index.similarity_candidates",
    }
    for report in health.values():
        assert report["health"] in ("fresh", "stale", "incomplete")
        assert "watermark" in report and "freshness_lag" in report
    report = pipeline.reconcile()
    assert report.authoritative_watermark == 10
    # Simulate a stale index: mark incomplete, health degrades gracefully.
    pipeline.index_registry.mark_incomplete("ore.data.index.bloom_membership")
    monitor = IndexHealthMonitor(pipeline.index_registry)
    assert monitor.usable_for_precheck(
        "ore.data.index.bloom_membership", 10
    ) is False
    assert approximate_analytics_degrade_gracefully(monitor, 10)
    report = pipeline.reconcile()
    assert report.indexes_stale == 1
    assert report.rebuild_plans[0].index_id == "ore.data.index.bloom_membership"


def test_lsm_dormant_until_benchmarks_justify(tmp_path: Path):
    """Step 44: the LSM path stays off by default; benchmarks activate it."""
    dormant = build_pipeline(workspace_root=tmp_path)
    assert dormant.capabilities.lsm_active is False
    assert dormant.lsm is None
    justified = build_pipeline(workspace_root=tmp_path / "lsm", lsm_justified=True)
    assert justified.capabilities.lsm_active is True
    outcome = justified.process_record(
        _record("lsm-key"), idempotency_key="lsm-1", provenance=_provenance()
    )
    assert outcome.status == "committed"
    assert justified.lsm.get("record:lsm-key") is not None


def test_estimates_reach_the_coordinator_with_metadata(tmp_path: Path):
    """Section 6: the coordinator receives estimate metadata and the exactness class."""
    pipeline = build_pipeline(workspace_root=tmp_path)
    pipeline.process_record(
        _record("concept-m"), idempotency_key="id-m", provenance=_provenance()
    )
    frequency = pipeline.frequency("concept-m")
    cardinality = pipeline.cardinality()
    coordinator = pipeline.coordinator
    answer = coordinator.request(
        "usage_frequency_estimate",
        ExactnessClass.BOUNDED_ERROR_ESTIMATE,
        frequency.estimate,
        frequency,
    )
    assert answer.estimate_metadata["algorithm"] == "count_min_sketch"
    assert answer.exactness is ExactnessClass.BOUNDED_ERROR_ESTIMATE
    service = coordinator.service_for(
        "duplicate_candidate_search", ExactnessClass.CANDIDATE_SIMILARITY
    )
    assert "verification" in service
    with pytest.raises(ValueError):
        coordinator.request("bounded", ExactnessClass.BOUNDED_ERROR_ESTIMATE, 1.0, None)
    assert assert_estimates_carry_metadata([frequency, cardinality, *pipeline.quantiles([0.5])])
    assert estimate_metadata(frequency)["exact"] is False


def test_cros_adapts_within_bounds_and_protects_correctness():
    """Section 6: telemetry adapts scheduling; correctness budgets are protected."""
    adapter = CROSDataBudgetAdapter()
    baseline = adapter.parameters()
    adapted = adapter.adapt(StorageTelemetry(0.9, 0.9, 50, 0.9))
    assert adapted["compaction_batch_bytes"] < baseline["compaction_batch_bytes"]
    relaxed = adapter.adapt(StorageTelemetry(0.1, 0.1, 0, 0.1))
    assert relaxed["bloom_rebuild_interval_events"] <= adapted["bloom_rebuild_interval_events"]
    with pytest.raises(ValueError):
        adapter.set_parameter("unknown_parameter", 1)
    with pytest.raises(ValueError):
        adapter.set_parameter("compaction_batch_bytes", 1)  # below the safe bound
    assert cros_cannot_relax_correctness(adapter)
    demand = telemetry_to_capability_demand(StorageTelemetry(0.5, 0.6, 10, 0.4))
    assert getattr(demand, "capability_id", "ore.data.storage_pressure") == (
        "ore.data.storage_pressure"
    )


def test_backpressure_prevents_unbounded_event_queues():
    """Section 8.4: the queue is bounded; overflow defers instead of growing."""
    controller = BackpressureController(queue_limit=4)
    accepted = [controller.offer(f"event-{i}") for i in range(4)]
    assert all(accepted)
    assert controller.offer("event-overflow") is False
    assert controller.deferred == 1
    assert controller.depth() == 4
    drained = controller.drain()
    assert len(drained) == 4
    assert controller.stats()["deferred"] == 1


def test_workspace_publisher_emits_typed_events_and_derived_views(tmp_path: Path):
    """Section 6: versioned events reach the workspace; derived views carry watermarks."""
    pipeline = build_pipeline(workspace_root=tmp_path)
    pipeline.process_record(
        _record("concept-w"), idempotency_key="id-w", provenance=_provenance()
    )
    publisher = pipeline.workspace
    assert len(publisher.published_event_ids) >= 1
    views = publisher.view_status()
    assert views["authority"].startswith("authoritative stores only")
    assert "ore.data.index.bloom_membership" in views["derived_views"]
    # A dedicated publisher converts change events directly.
    from ore_data.ingestion import ChangeEventType, DataChangeEvent

    event = DataChangeEvent(
        event_id="evt-77", event_type=ChangeEventType.SNAPSHOT, entity_key="snapshot",
        source="ore_test", schema_version="0.1.0", sequence=77,
    )
    published = WorkspaceDataPublisher().publish_change_event(event)
    assert str(getattr(published, "event_type", "")) == "data.snapshot"


def test_usg_exact_resolution_gates_similarity_merges():
    """Section 6: USG identity resolves candidates; similarity alone never merges."""
    resolver = InMemoryConceptResolver(
        [
            ConceptIdentity("concept_1", "neural network", "en", "usg"),
            ConceptIdentity("concept_2", "connectionist model", "en", "usg"),
        ]
    )
    adapter = USGSemanticVerificationAdapter(resolver)
    verified = verify_candidate_similarity(
        ("rec-a", "rec-b"), candidate_score=0.9,
        first_features={f"a{i}" for i in range(100)},
        second_features={f"a{i}" for i in range(99)},
        threshold=0.5,
    )
    same_concept = adapter.verify_candidate_merge(
        verified, first_label="neural network", second_label="Neural Network"
    )
    assert same_concept.may_merge is True  # same authoritative identity
    distinct = adapter.verify_candidate_merge(
        verified, first_label="neural network", second_label="connectionist model"
    )
    assert distinct.may_merge is False
    assert "distinct authoritative concepts" in distinct.reason
    unresolved = adapter.verify_candidate_merge(
        verified, first_label="neural network", second_label="unknown label"
    )
    assert unresolved.may_merge is False  # no exact identity evidence -> no merge
    # Byte-identical duplicates may merge without USG identity (Step 42/43).
    byte_identical = verify_byte_identity(("rec-a", "rec-b"), b"same", b"same")
    byte_merge = adapter.verify_candidate_merge(
        byte_identical, first_label="unknown label", second_label="also unknown"
    )
    assert byte_merge.may_merge is True
    assert "byte-identical" in byte_merge.reason
    assert adapter.verifications[0].to_dict()["pair"] == ["rec-a", "rec-b"]


def test_usg_index_resolver_adapts_the_repo_retrieval_interface():
    """Section 6: the ore_usg retrieval index plugs in behind the resolver protocol."""
    class FakeIndex:
        def lookup(self, query: str, kind: str | None = None):
            class Item:
                record_id = "usg_1"
                label = "Neural Network"
                language = "en"
                source_reference = "usg://test"

            class Result:
                items = [Item()]
                exact_match = True

            return Result() if kind == "SEMANTIC_CONCEPT" else Result()

    resolver = UsgIndexResolver(FakeIndex())
    identities = resolver.lookup_concept("Neural Network")
    assert identities and identities[0].concept_id == "usg_1"
    assert identities[0].label == "Neural Network"
    with pytest.raises(TypeError):
        UsgIndexResolver(object())  # requires the ore_usg retrieval interface


def test_pipeline_describe_and_registry(tmp_path: Path):
    """The pipeline describes the deployment and maps all 56 steps."""
    pipeline = build_pipeline(workspace_root=tmp_path)
    description = pipeline.describe()
    assert description["contract"] == "ORE.DATA.INFRASTRUCTURE/0.1.0"
    assert description["capabilities"]["lsm_active"] is False
    assert description["operation_count"] == 56
    summary = DataInfrastructurePipeline.registry_summary()
    assert summary["operation_count"] == 56
    assert len(summary["operations"]) == 56
    checks = DataInfrastructurePipeline.interface_checks()
    assert all(checks.values()), checks
    assert operation_count() == 56
    assert phases() == tuple(f"Phase {letter}" for letter in "ABCDEFGHIJ")


def test_pipeline_config_validation():
    """PipelineConfig validates every declared budget (Steps 5-6)."""
    with pytest.raises(ValueError):
        PipelineConfig(bloom_target_fp_rate=2.0)
    with pytest.raises(ValueError):
        PipelineConfig(count_min_delta=0.0)
    with pytest.raises(ValueError):
        PipelineConfig(hll_precision=99)
    with pytest.raises(ValueError):
        PipelineConfig(kll_k=4)
    config = PipelineConfig()
    assert config.expected_membership_keys > 0


def test_pipeline_metrics_snapshot_covers_the_metric_families(tmp_path: Path):
    """Section 8.5: the metrics snapshot exposes every metric family."""
    pipeline = build_pipeline(workspace_root=tmp_path)
    pipeline.process_record(
        _record("concept-k"), idempotency_key="id-k", provenance=_provenance()
    )
    pipeline.telemetry(StorageTelemetry(0.8, 0.7, 20, 0.5))
    snapshot = pipeline.metrics_snapshot()
    for family in (
        "bloom", "count_min", "hyperloglog", "quantiles", "heavy_hitters",
        "similarity", "storage", "integrity", "recovery", "indexes",
    ):
        assert family in snapshot
    assert snapshot["coordinator"]["requests"] >= 0
    assert snapshot["backpressure"]["queue_limit"] > 0


def test_python_module_summary_runs():
    """``python -m ore_data`` prints the deployment and registry summary."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "ore_data"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ORE.DATA.INFRASTRUCTURE/0.1.0" in result.stdout
    assert '"operation_count": 56' in result.stdout
