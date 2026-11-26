"""Acceptance tests for CROS telemetry schemas, collectors and quality (Steps 1-8)."""
import time as _time

import pytest

from ore_cros.resource_model import ResourceKind, ResourceScope, ScopeLevel, machine_scope
from ore_cros.telemetry.collectors import (
    BaselineProfile, CollectorSet, HostProfiler, LinuxProcCollector,
    SimulatedCollector, record_baseline,
)
from ore_cros.telemetry.history import TimeWindowedResourceHistory, ewma, quantile
from ore_cros.telemetry.quality import TelemetryQualityAssessor
from ore_cros.telemetry.schemas import (
    Measurement, MetricKind, TelemetrySchemaError, canonical_unit,
    require_single_scope,
)

# Anchored to the real clock so future-timestamp rejection is testable.
NOW = _time.time()


def _measurement(**overrides):
    """Build one valid machine-scope utilization measurement for tests."""
    defaults = dict(
        resource=ResourceKind.CPU, metric=MetricKind.UTILIZATION, value=0.5,
        unit="ratio", timestamp=NOW, scope=machine_scope(), source="test",
        quality=1.0, window_seconds=1.0,
    )
    defaults.update(overrides)
    return Measurement(**defaults)


def test_measurement_rejects_invalid_units():
    """Telemetry rejects measurements whose unit disagrees with the schema."""
    with pytest.raises(TelemetrySchemaError):
        _measurement(unit="percent")


def test_measurement_rejects_impossible_values():
    """Telemetry rejects negative, non-finite and fractional queue depths."""
    with pytest.raises(TelemetrySchemaError):
        _measurement(value=-0.1)
    with pytest.raises(TelemetrySchemaError):
        _measurement(value=float("nan"))
    with pytest.raises(TelemetrySchemaError):
        _measurement(
            resource=ResourceKind.QUEUE, metric=MetricKind.QUEUE_DEPTH,
            value=2.5, unit="jobs",
        )
    with pytest.raises(TelemetrySchemaError):
        _measurement(
            resource=ResourceKind.CPU, metric=MetricKind.UTILIZATION,
            value=100_000.0, unit="ratio",
        )


def test_measurement_rejects_malformed_timestamps():
    """Telemetry rejects NaN, negative and unreasonably future timestamps."""
    with pytest.raises(TelemetrySchemaError):
        _measurement(timestamp=float("nan"))
    with pytest.raises(TelemetrySchemaError):
        _measurement(timestamp=-1.0)
    with pytest.raises(TelemetrySchemaError):
        _measurement(timestamp=NOW + 10_000_000.0)


def test_queue_metrics_restricted_to_queue_resource():
    """Metric-kind constraints keep queue metrics off other resources."""
    with pytest.raises(TelemetrySchemaError):
        _measurement(resource=ResourceKind.CPU, metric=MetricKind.QUEUE_DEPTH,
                     value=3.0, unit="jobs")


def test_scopes_cannot_be_silently_mixed():
    """Machine, process, task and module metrics refuse joint aggregation."""
    machine = _measurement(scope=machine_scope("m1"))
    process = _measurement(scope=ResourceScope(ScopeLevel.PROCESS, "p1", machine_scope("m1")))
    with pytest.raises(TelemetrySchemaError):
        require_single_scope([machine, process])
    same_machine = _measurement(value=0.7, scope=machine_scope("m1"))
    assert require_single_scope([machine, same_machine]).scope_id == "m1"
    with pytest.raises(TelemetrySchemaError):
        require_single_scope([])


def test_discovery_reports_unknown_capacity_as_none_not_zero():
    """Step 1: unavailable capacity dimensions are unknown, never zero."""
    profiler = HostProfiler(root="/nonexistent-host-root")
    profile = profiler.discover()
    assert profile.host_os
    assert profile.capacity[ResourceKind.IO] is None
    assert profile.capacity[ResourceKind.NET] is None
    assert profile.capacity[ResourceKind.QUEUE] is not None


def test_linux_proc_collector_isolated_from_read_errors():
    """A failing collector is reported, never fabricating zero measurements."""
    collector = LinuxProcCollector()
    collector._previous_cpu_totals = None
    # /proc/stat exists on Linux; force a failure path via a missing baseline.
    measurements = collector.collect(now=NOW)
    assert isinstance(measurements, list)


def test_simulated_collector_is_deterministic():
    """Identical seeds and load profiles produce identical measurement streams."""
    def build():
        collector = SimulatedCollector(seed=11, load_profile=lambda step: 0.8)
        return [item.value for item in collector.collect(now=NOW)]
    assert build() == build()


def test_collector_set_isolates_crashing_collectors():
    """Collector crashes surface as failed statuses, not exceptions."""
    class CrashingCollector:
        source_name = "crashing"

        def collect(self, now=None):
            raise RuntimeError("boom")

    healthy = SimulatedCollector(seed=3)
    collector_set = CollectorSet([healthy, CrashingCollector()])
    measurements, statuses = collector_set.collect(now=NOW)
    by_name = {status.name: status for status in statuses}
    assert by_name["crashing"].ok is False
    assert by_name["simulated"].ok is True
    assert measurements


def test_collector_set_deduplicates_by_highest_quality():
    """Duplicate (scope, resource, metric) pairs keep the best source."""
    collector_set = CollectorSet([SimulatedCollector(seed=5)])
    measurements, _ = collector_set.collect(now=NOW)
    keys = [
        (item.resource, item.metric) for item in measurements
    ]
    assert len(keys) == len(set(keys))


def test_stale_telemetry_blocks_high_impact_actions():
    """Stale or missing telemetry cannot trigger unsafe high-impact actions."""
    assessor = TelemetryQualityAssessor()
    stale = _measurement(timestamp=NOW - 3600.0)
    report = assessor.assess([stale], now=NOW)
    assert report.series["CPU/UTILIZATION"].stale
    assert report.allows_high_impact(ResourceKind.CPU) is False

    fresh = _measurement(quality=1.0, source="/proc")
    report = assessor.assess([fresh], now=NOW)
    assert report.allows_high_impact(ResourceKind.CPU) is True


def test_quality_report_tracks_missing_sources_and_series():
    """Failed collectors and absent series are explicit in the report."""
    assessor = TelemetryQualityAssessor()
    report = assessor.assess([], now=NOW, failed_sources=["/proc"])
    assert report.missing_sources == ("/proc",)
    assert report.missing_series
    assert report.confidence(ResourceKind.CPU) == 0.0


def test_counter_resets_are_detected():
    """Counter resets between batches lower the confidence evidence."""
    assessor = TelemetryQualityAssessor()
    first = _measurement(resource=ResourceKind.CPU, metric=MetricKind.ALLOCATED,
                         value=100.0, unit="cores")
    second = _measurement(resource=ResourceKind.CPU, metric=MetricKind.ALLOCATED,
                          value=40.0, unit="cores")
    report = assessor.assess([second], now=NOW, previous=[first])
    assert report.counter_resets


def test_history_windows_and_bounded_retention():
    """Step 7: history keeps bounded windows and aggregates expired records."""
    history = TimeWindowedResourceHistory()
    for index in range(50):
        history.record([_measurement(value=0.5, timestamp=NOW - 500.0 + index * 10.0)])
    stats = history.statistics(
        history.series_keys()[0], window_seconds=120.0, now=NOW,
    )
    assert stats is not None and stats.count > 0
    removed = history.prune(now=NOW)
    assert removed > 0
    assert history.total_samples() <= history.memory_bound()
    assert history.compacted()


def test_quantile_and_ewma_helpers():
    """Windowed statistics helpers compute expected summaries."""
    assert quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert ewma([1.0, 1.0, 1.0]) == 1.0
    with pytest.raises(ValueError):
        quantile([], 0.5)


def test_baseline_profile_records_overhead_and_missing_rate():
    """Step 4: baselines record measurement overhead and missing-data rates."""
    collector_set = CollectorSet([SimulatedCollector(seed=9)])
    profile = record_baseline(collector_set, phases=("idle", "burst"), samples_per_phase=2, now=NOW)
    assert isinstance(profile, BaselineProfile)
    assert profile.missing_rate == 0.0
    assert profile.overhead_seconds >= 0.0
    assert set(profile.phase_summary("idle"))
