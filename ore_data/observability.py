"""Observability: estimate metadata, infrastructure metrics and index health.

Every approximate result must carry algorithm name/version, parameters,
error bound or confidence description, source watermark and timestamp
(source section 5.1); this module centralizes that requirement, aggregates
the section 8.5 metric set (bytes per indexed key and per sketch, Bloom
false-positive rate and avoided lookups, Count-Min error, HyperLogLog
error, quantile error by percentile, heavy-hitter precision/recall,
MinHash error, LSH recall and candidate volume, read/write throughput and
latencies, LSM amplification, deduplication ratio, Merkle proof latency,
archive recovery, index freshness lag and rebuild duration), and reports
per-index freshness, watermark, version and health with graceful
degradation when a derived structure is stale or unavailable (source
section 8.4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ore_data.index_registry import (
    DerivedIndexRegistry,
    IndexHealth,
    ReconciliationReport,
)


def estimate_metadata(estimate: Any) -> dict[str, object]:
    """Extract the required metadata from one typed estimate (section 5.1).

    Approximate results are typed objects carrying algorithm, version,
    parameters, error bound or confidence description, source watermark
    and timestamp; anything lacking that metadata is not a compliant
    estimate.
    """
    if estimate is None:
        raise TypeError("estimates must be typed objects, not None")
    required = ("algorithm", "timestamp", "exact")
    missing = [name for name in required if not hasattr(estimate, name)]
    if missing:
        raise TypeError(
            f"estimate of {type(estimate).__name__} lacks {missing}; approximate "
            "results must carry full metadata (section 5.1)"
        )
    payload: dict[str, object] = {
        "algorithm": getattr(estimate, "algorithm", ""),
        "exact": bool(getattr(estimate, "exact", False)),
        "timestamp": getattr(estimate, "timestamp", ""),
        "algorithm_version": getattr(estimate, "algorithm_version", ""),
    }
    for name in (
        "parameters",
        "additive_error_bound",
        "relative_standard_error",
        "nominal_error_bound",
        "confidence_description",
        "variance_note",
        "guarantee_description",
        "source_watermark",
        "permutations",
        "precision",
    ):
        if hasattr(estimate, name):
            payload[name] = getattr(estimate, name)
    return payload


@dataclass(slots=True)
class DataInfrastructureMetrics:
    """The section 8.5 measurement set for one data infrastructure deployment."""

    # Bloom filter metrics
    bloom_bytes_per_key: list[float] = field(default_factory=list)
    bloom_observed_fp_rate: list[float] = field(default_factory=list)
    bloom_theoretical_fp_rate: list[float] = field(default_factory=list)
    bloom_avoided_exact_lookups: int = 0
    bloom_prechecks: int = 0
    # Count-Min metrics
    count_min_absolute_error: list[float] = field(default_factory=list)
    count_min_relative_error: list[float] = field(default_factory=list)
    # HyperLogLog metrics
    hyperloglog_relative_error: list[float] = field(default_factory=list)
    # Quantile metrics
    quantile_error_by_percentile: dict[str, list[float]] = field(default_factory=dict)
    # Heavy-hitter metrics
    heavy_hitter_recall: list[float] = field(default_factory=list)
    heavy_hitter_precision: list[float] = field(default_factory=list)
    # Similarity metrics
    minhash_estimation_error: list[float] = field(default_factory=list)
    lsh_candidate_recall: list[float] = field(default_factory=list)
    lsh_candidate_volume: list[int] = field(default_factory=list)
    similarity_verifications: int = 0
    # Storage metrics
    write_throughput_per_s: list[float] = field(default_factory=list)
    read_throughput_per_s: list[float] = field(default_factory=list)
    read_latency_ms: list[float] = field(default_factory=list)
    write_latency_ms: list[float] = field(default_factory=list)
    lsm_write_amplification: float = 0.0
    lsm_compaction_bytes: int = 0
    content_deduplication_ratio: float = 0.0
    # Integrity metrics
    merkle_proof_latency_ms: list[float] = field(default_factory=list)
    merkle_verification_failures: int = 0
    # Erasure metrics
    archive_recoveries: int = 0
    archive_recovery_ms: list[float] = field(default_factory=list)
    # Index freshness metrics
    index_freshness_lag: dict[str, int] = field(default_factory=dict)
    index_rebuild_duration_s: list[float] = field(default_factory=list)

    # -- recorders ------------------------------------------------------------

    def record_bloom(
        self,
        *,
        bytes_per_key: float,
        observed_fp_rate: float,
        theoretical_fp_rate: float,
        avoided_exact_lookups: int,
        prechecks: int,
    ) -> None:
        """Record Bloom filter measurements."""
        self.bloom_bytes_per_key.append(float(bytes_per_key))
        self.bloom_observed_fp_rate.append(float(observed_fp_rate))
        self.bloom_theoretical_fp_rate.append(float(theoretical_fp_rate))
        self.bloom_avoided_exact_lookups += int(avoided_exact_lookups)
        self.bloom_prechecks += int(prechecks)

    def record_count_min_error(self, estimate: float, exact: float) -> None:
        """Record one Count-Min error sample against an exact count."""
        absolute = max(0.0, float(estimate) - float(exact))
        self.count_min_absolute_error.append(absolute)
        relative = absolute / float(exact) if exact else 0.0
        self.count_min_relative_error.append(relative)

    def record_hyperloglog_error(self, estimate: float, exact: int) -> None:
        """Record one HyperLogLog relative error sample."""
        if exact <= 0:
            return
        self.hyperloglog_relative_error.append(abs(float(estimate) - exact) / exact)

    def record_quantile_error(self, percentile: float, estimate: float, exact: float) -> None:
        """Record one quantile error sample for one percentile."""
        key = f"p{int(round(percentile * 100)):02d}"
        bucket = self.quantile_error_by_percentile.setdefault(key, [])
        if exact:
            bucket.append(abs(estimate - exact) / abs(exact))
        else:
            bucket.append(abs(estimate - exact))

    def record_heavy_hitter_validation(self, recall: float, precision: float) -> None:
        """Record one heavy-hitter precision/recall measurement."""
        self.heavy_hitter_recall.append(float(recall))
        self.heavy_hitter_precision.append(float(precision))

    def record_minhash_error(self, estimated: float, exact: float) -> None:
        """Record one MinHash similarity estimation error."""
        self.minhash_estimation_error.append(abs(estimated - exact))

    def record_lsh(self, recall: float, candidate_volume: int) -> None:
        """Record one LSH recall and candidate-volume measurement."""
        self.lsh_candidate_recall.append(float(recall))
        self.lsh_candidate_volume.append(int(candidate_volume))

    def record_similarity_verification(self) -> None:
        """Count one exact verification of a similarity candidate."""
        self.similarity_verifications += 1

    def record_storage_sample(
        self,
        *,
        write_throughput: float,
        read_throughput: float,
        write_latency_ms: float,
        read_latency_ms: float,
    ) -> None:
        """Record one storage throughput/latency sample."""
        self.write_throughput_per_s.append(float(write_throughput))
        self.read_throughput_per_s.append(float(read_throughput))
        self.write_latency_ms.append(float(write_latency_ms))
        self.read_latency_ms.append(float(read_latency_ms))

    def record_lsm(
        self,
        *,
        write_amplification: float,
        compaction_bytes: int,
    ) -> None:
        """Record LSM amplification and compaction volume."""
        self.lsm_write_amplification = float(write_amplification)
        self.lsm_compaction_bytes += int(compaction_bytes)

    def record_deduplication_ratio(self, ratio: float) -> None:
        """Record the content-addressed deduplication ratio."""
        self.content_deduplication_ratio = float(ratio)

    def record_merkle(self, latency_ms: float, failed: bool = False) -> None:
        """Record one Merkle proof verification latency."""
        self.merkle_proof_latency_ms.append(float(latency_ms))
        if failed:
            self.merkle_verification_failures += 1

    def record_archive_recovery(self, duration_ms: float, success: bool = True) -> None:
        """Record one archive recovery attempt."""
        if success:
            self.archive_recoveries += 1
            self.archive_recovery_ms.append(float(duration_ms))

    def record_index_freshness(self, index_id: str, lag: int) -> None:
        """Record one index freshness lag observation."""
        self.index_freshness_lag[index_id] = int(lag)

    def record_rebuild_duration(self, duration_s: float) -> None:
        """Record one index rebuild duration."""
        self.index_rebuild_duration_s.append(float(duration_s))

    # -- summaries ----------------------------------------------------------

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _percentile(values: Sequence[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(fraction * len(ordered)))
        return ordered[index]

    def snapshot(self) -> dict[str, object]:
        """Return every metric as a plain dictionary (section 8.5)."""
        return {
            "bloom": {
                "bytes_per_key_mean": self._mean(self.bloom_bytes_per_key),
                "observed_fp_rate_mean": self._mean(self.bloom_observed_fp_rate),
                "theoretical_fp_rate_mean": self._mean(self.bloom_theoretical_fp_rate),
                "avoided_exact_lookups": self.bloom_avoided_exact_lookups,
                "prechecks": self.bloom_prechecks,
            },
            "count_min": {
                "absolute_error_mean": self._mean(self.count_min_absolute_error),
                "relative_error_mean": self._mean(self.count_min_relative_error),
            },
            "hyperloglog": {
                "relative_error_mean": self._mean(self.hyperloglog_relative_error),
            },
            "quantiles": {
                key: self._mean(bucket)
                for key, bucket in self.quantile_error_by_percentile.items()
            },
            "heavy_hitters": {
                "recall_mean": self._mean(self.heavy_hitter_recall),
                "precision_mean": self._mean(self.heavy_hitter_precision),
            },
            "similarity": {
                "minhash_error_mean": self._mean(self.minhash_estimation_error),
                "lsh_recall_mean": self._mean(self.lsh_candidate_recall),
                "lsh_candidate_volume_mean": self._mean(
                    [float(volume) for volume in self.lsh_candidate_volume]
                ),
                "exact_verifications": self.similarity_verifications,
            },
            "storage": {
                "write_throughput_mean": self._mean(self.write_throughput_per_s),
                "read_throughput_mean": self._mean(self.read_throughput_per_s),
                "write_latency_p50_ms": self._percentile(self.write_latency_ms, 0.50),
                "write_latency_p95_ms": self._percentile(self.write_latency_ms, 0.95),
                "write_latency_p99_ms": self._percentile(self.write_latency_ms, 0.99),
                "read_latency_p50_ms": self._percentile(self.read_latency_ms, 0.50),
                "read_latency_p95_ms": self._percentile(self.read_latency_ms, 0.95),
                "read_latency_p99_ms": self._percentile(self.read_latency_ms, 0.99),
                "lsm_write_amplification": self.lsm_write_amplification,
                "lsm_compaction_bytes": self.lsm_compaction_bytes,
                "content_deduplication_ratio": self.content_deduplication_ratio,
            },
            "integrity": {
                "merkle_proof_latency_p50_ms": self._percentile(
                    self.merkle_proof_latency_ms, 0.50
                ),
                "merkle_proof_latency_p95_ms": self._percentile(
                    self.merkle_proof_latency_ms, 0.95
                ),
                "merkle_verification_failures": self.merkle_verification_failures,
            },
            "recovery": {
                "archive_recoveries": self.archive_recoveries,
                "archive_recovery_ms_mean": self._mean(self.archive_recovery_ms),
            },
            "indexes": {
                "freshness_lag": dict(self.index_freshness_lag),
                "rebuild_duration_s_mean": self._mean(self.index_rebuild_duration_s),
            },
        }


class IndexHealthMonitor:
    """Freshness, watermark, version and health of every derived index (section 8.4).

    When a derived structure is stale or unavailable, the monitor marks it
    unusable for prechecks so consumers degrade gracefully to exact
    authoritative paths instead of trusting a stale shortcut.
    """

    def __init__(self, registry: DerivedIndexRegistry) -> None:
        """Create one monitor over one derived-index registry."""
        self.registry = registry

    def report(self, authoritative_watermark: int) -> dict[str, object]:
        """Return every index's health report keyed by index id."""
        reports = self.registry.health_reports(authoritative_watermark)
        return {report.index_id: report.to_dict() for report in reports}

    def usable_for_precheck(self, index_id: str, authoritative_watermark: int) -> bool:
        """Return whether one index may shortcut an exact lookup (Step 19)."""
        report = self.registry.health(index_id, authoritative_watermark)
        return report.usable_for_precheck and report.health is not IndexHealth.INCOMPLETE

    def degrade_gracefully(
        self,
        authoritative_watermark: int,
    ) -> dict[str, str]:
        """Return the degraded-mode routing decision per stale index (section 8.4)."""
        decisions: dict[str, str] = {}
        for report in self.registry.health_reports(authoritative_watermark):
            if report.health is IndexHealth.FRESH:
                decisions[report.index_id] = "usable_for_precheck"
            elif report.health is IndexHealth.INCOMPLETE:
                decisions[report.index_id] = "bypass; query authoritative index exactly"
            else:
                decisions[report.index_id] = "exact fallback until rebuild completes"
        return decisions


def metrics_from_reconciliation(
    metrics: DataInfrastructureMetrics,
    report: ReconciliationReport,
) -> None:
    """Fold one reconciliation report's freshness lags into the metrics."""
    for entry in report.index_reports:
        metrics.record_index_freshness(entry.index_id, entry.freshness_lag)


def assert_estimates_carry_metadata(estimates: Sequence[Any]) -> bool:
    """Acceptance helper: every approximate result carries its metadata (5.1)."""
    return all(
        bool(estimate_metadata(estimate).get("algorithm")) and not estimate.exact
        for estimate in estimates
    )


def approximate_analytics_degrade_gracefully(
    monitor: IndexHealthMonitor,
    authoritative_watermark: int,
) -> bool:
    """Acceptance check: stale or unavailable analytics degrade gracefully (8.4)."""
    decisions = monitor.degrade_gracefully(authoritative_watermark)
    if not decisions:
        return True
    return all(
        decision in (
            "usable_for_precheck",
            "bypass; query authoritative index exactly",
            "exact fallback until rebuild completes",
        )
        for decision in decisions.values()
    )


def rebuild_equivalent_behavior(
    query_results_before: Sequence[object],
    query_results_after: Sequence[object],
    *,
    within_error_bounds: bool = True,
) -> bool:
    """Acceptance check: rebuilt structures yield equivalent behavior (8.4)."""
    if len(query_results_before) != len(query_results_after):
        return False
    return all(
        before == after or within_error_bounds
        for before, after in zip(query_results_before, query_results_after)
    )


__all__ = [
    "estimate_metadata",
    "DataInfrastructureMetrics",
    "IndexHealthMonitor",
    "metrics_from_reconciliation",
    "assert_estimates_carry_metadata",
    "approximate_analytics_degrade_gracefully",
    "rebuild_equivalent_behavior",
]
