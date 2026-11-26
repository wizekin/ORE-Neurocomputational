"""Large-bundle USG import and exact lookup measurements."""
from __future__ import annotations

from copy import deepcopy

from ore_usg.performance import (
    benchmark_usg_import,
    benchmark_usg_lookups,
    measure_usg_resource_footprint,
)
from test_usg_semantic_knowledge_adapter import _phase2_bundle


def _large_bundle(extra_forms: int = 1200):
    bundle = deepcopy(_phase2_bundle())
    timestamp = bundle["created_at"]
    extra = []
    for index in range(extra_forms):
        extra.append({
            "id": f"form:en:synthetic-{index:06d}",
            "kind": "LEXICAL_FORM",
            "version": 1,
            "state": "ACCEPTED",
            "created_at": timestamp,
            "source_ids": ["source:dictionary"],
            "evidence_ids": [],
            "context_ids": [],
            "payload": {"form": f"synthetic-{index:06d}", "language": "en", "script": "Latn"},
        })
    bundle["records"].extend(extra)
    bundle["bundle_id"] = "phase6-large-performance"
    return bundle


def test_large_bundle_import_measures_time_memory_and_resource_counts():
    resource, report = benchmark_usg_import(_large_bundle(), repetitions=2)
    assert report.repetitions == 2
    assert report.bundle_bytes > 50000
    assert report.median_import_ms > 0
    assert report.maximum_import_ms >= report.minimum_import_ms
    assert report.peak_traced_memory_bytes > 0
    assert report.lexical_entries >= 1200
    assert report.bundle_digest == resource.bundle_digest
    footprint = measure_usg_resource_footprint(resource)
    assert footprint.unique_ore_record_ids > 1200
    assert footprint.duplicate_ore_record_ids == 0


def test_exact_lexical_semantic_and_predicate_lookup_metrics_are_reported():
    resource, _ = benchmark_usg_import(_large_bundle(600), repetitions=1)
    metrics = benchmark_usg_lookups(
        resource,
        lexical_queries=("synthetic-000001", "synthetic-000599"),
        semantic_queries=("Finance",),
        predicate_queries=("transfer",),
        repetitions=3,
    )
    assert metrics.lexical_queries == 2
    assert metrics.semantic_queries == 1
    assert metrics.predicate_queries == 1
    assert metrics.results_by_kind["lexical"] >= 2
    assert metrics.median_lookup_us >= 0
    assert metrics.p95_lookup_us >= metrics.median_lookup_us


def test_repeated_footprint_measurement_does_not_duplicate_snapshot_records():
    resource, _ = benchmark_usg_import(_large_bundle(100), repetitions=1)
    assert measure_usg_resource_footprint(resource) == measure_usg_resource_footprint(resource)
