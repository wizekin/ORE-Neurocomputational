"""Typed telemetry: schemas, platform collectors, quality and history (Steps 3-8)."""
from __future__ import annotations

from .collectors import (
    BaselineProfile, CollectorSet, CollectorStatus, HostProfiler,
    LinuxProcCollector, SimulatedCollector, record_baseline,
)
from .history import TimeWindowedResourceHistory, quantile, ewma
from .quality import QualityReport, TelemetryQualityAssessor
from .schemas import (
    Measurement, MetricKind, TelemetrySchemaError, canonical_unit,
    require_single_scope,
)

__all__ = [
    "BaselineProfile", "CollectorSet", "CollectorStatus", "HostProfiler",
    "LinuxProcCollector", "SimulatedCollector", "record_baseline",
    "TimeWindowedResourceHistory", "quantile", "ewma",
    "QualityReport", "TelemetryQualityAssessor",
    "Measurement", "MetricKind", "TelemetrySchemaError", "canonical_unit",
    "require_single_scope",
]
