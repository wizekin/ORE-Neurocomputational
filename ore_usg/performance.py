"""Reproducible import, lookup and resource-footprint measurements for USG bundles."""
from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
import time
import tracemalloc
from typing import Any, Iterable

from .access import USGKnowledgeAccess
from .knowledge import USGKnowledgeAdapter, USGKnowledgeResource


@dataclass(frozen=True, slots=True)
class USGImportBenchmark:
    bundle_digest: str
    bundle_bytes: int
    repetitions: int
    median_import_ms: float
    minimum_import_ms: float
    maximum_import_ms: float
    peak_traced_memory_bytes: int
    lexical_entries: int
    semantic_concepts: int
    relation_count: int


@dataclass(frozen=True, slots=True)
class USGLookupBenchmark:
    bundle_digest: str
    repetitions: int
    lexical_queries: int
    semantic_queries: int
    predicate_queries: int
    median_lookup_us: float
    p95_lookup_us: float
    maximum_lookup_us: float
    results_by_kind: dict[str, int]


@dataclass(frozen=True, slots=True)
class USGResourceFootprint:
    bundle_digest: str
    lexical_entries: int
    semantic_concepts: int
    semantic_categories: int
    semantic_relations: int
    predicate_frames: int
    evidence_records: int
    source_records: int
    unique_ore_record_ids: int
    duplicate_ore_record_ids: int


def benchmark_usg_import(
    bundle: Any,
    *,
    adapter: USGKnowledgeAdapter | None = None,
    repetitions: int = 3,
) -> tuple[USGKnowledgeResource, USGImportBenchmark]:
    """Measure repeated import wall time and peak traced Python allocations."""
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    active_adapter = adapter or USGKnowledgeAdapter()
    encoded_bytes = len(repr(bundle).encode("utf-8"))
    timings: list[float] = []
    peak_memory = 0
    resource: USGKnowledgeResource | None = None
    tracing_before = tracemalloc.is_tracing()
    for _ in range(repetitions):
        if not tracing_before:
            tracemalloc.start()
        else:
            tracemalloc.reset_peak()
        started = time.perf_counter()
        try:
            resource = active_adapter.convert(bundle)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            _, peak = tracemalloc.get_traced_memory()
            if not tracing_before:
                tracemalloc.stop()
        timings.append(elapsed_ms)
        peak_memory = max(peak_memory, int(peak))
    assert resource is not None
    report = USGImportBenchmark(
        bundle_digest=resource.bundle_digest,
        bundle_bytes=encoded_bytes,
        repetitions=repetitions,
        median_import_ms=statistics.median(timings),
        minimum_import_ms=min(timings),
        maximum_import_ms=max(timings),
        peak_traced_memory_bytes=peak_memory,
        lexical_entries=len(resource.lexical.entries),
        semantic_concepts=len(resource.semantic_concepts),
        relation_count=len(resource.semantic_relations),
    )
    return resource, report


def benchmark_usg_lookups(
    resource: USGKnowledgeResource,
    *,
    lexical_queries: Iterable[str] = (),
    semantic_queries: Iterable[str] = (),
    predicate_queries: Iterable[str] = (),
    repetitions: int = 5,
) -> USGLookupBenchmark:
    """Measure exact lexical, semantic and predicate lookup latency on one snapshot."""
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    queries = (
        ("lexical", tuple(lexical_queries), "LEXICAL_ENTRY"),
        ("semantic", tuple(semantic_queries), "SEMANTIC_CONCEPT"),
        ("predicate", tuple(predicate_queries), "PREDICATE_FRAME"),
    )
    access = USGKnowledgeAccess(resource)
    durations: list[float] = []
    results = {"lexical": 0, "semantic": 0, "predicate": 0}
    counts = {"lexical": 0, "semantic": 0, "predicate": 0}
    for kind, values, record_kind in queries:
        counts[kind] = len(values)
        for query in values:
            for _ in range(repetitions):
                started = time.perf_counter()
                result = access.index.lookup(query, kind=record_kind)
                durations.append((time.perf_counter() - started) * 1_000_000.0)
                results[kind] += len(result.items)
    ordered = sorted(durations)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1)) if ordered else 0
    return USGLookupBenchmark(
        bundle_digest=resource.bundle_digest,
        repetitions=repetitions,
        lexical_queries=counts["lexical"],
        semantic_queries=counts["semantic"],
        predicate_queries=counts["predicate"],
        median_lookup_us=statistics.median(ordered) if ordered else 0.0,
        p95_lookup_us=ordered[p95_index] if ordered else 0.0,
        maximum_lookup_us=max(ordered) if ordered else 0.0,
        results_by_kind=results,
    )


def measure_usg_resource_footprint(resource: USGKnowledgeResource) -> USGResourceFootprint:
    """Count unified resource records and detect duplicate stable IDs across inventories."""
    inventories = [
        [item.entry_id for item in resource.lexical.entries],
        [item.concept_id for item in resource.semantic_concepts],
        [item.category_id for item in resource.semantic_categories],
        [item.relation_id for item in resource.semantic_relations],
        [item.frame_id for item in resource.predicate_frames],
        [item.evidence_id for item in resource.evidence_records],
        [item.source_id for item in resource.source_provenance],
    ]
    all_ids = [str(value) for inventory in inventories for value in inventory]
    unique_ids = set(all_ids)
    return USGResourceFootprint(
        bundle_digest=resource.bundle_digest,
        lexical_entries=len(resource.lexical.entries),
        semantic_concepts=len(resource.semantic_concepts),
        semantic_categories=len(resource.semantic_categories),
        semantic_relations=len(resource.semantic_relations),
        predicate_frames=len(resource.predicate_frames),
        evidence_records=len(resource.evidence_records),
        source_records=len(resource.source_provenance),
        unique_ore_record_ids=len(unique_ids),
        duplicate_ore_record_ids=len(all_ids) - len(unique_ids),
    )


__all__ = [
    "USGImportBenchmark", "USGLookupBenchmark", "USGResourceFootprint",
    "benchmark_usg_import", "benchmark_usg_lookups", "measure_usg_resource_footprint",
]
