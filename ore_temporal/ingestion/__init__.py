"""Ingestion, time alignment, and quality (specification Phase B, Steps 8-16)."""
from __future__ import annotations

from .missing import MissingDataHandler, MissingDataReport
from .normalize import VersionedNormalizer
from .ordering import ReorderBuffer, apply_ordering_and_deltas, deduplicate_observations
from .quality import IngestionReport, QualityValidator, validate_quality
from .timestamps import TimestampValidator, validate_timestamp_semantics
from .units import UnitConversionError, UnitRegistry, convert_units
from .irregularity import SamplingProfile, profile_sampling
from .windows import build_event_batch, build_temporal_window

__all__ = (
    "IngestionReport",
    "MissingDataHandler",
    "MissingDataReport",
    "QualityValidator",
    "ReorderBuffer",
    "SamplingProfile",
    "TimestampValidator",
    "UnitConversionError",
    "UnitRegistry",
    "VersionedNormalizer",
    "apply_ordering_and_deltas",
    "build_event_batch",
    "build_temporal_window",
    "convert_units",
    "deduplicate_observations",
    "profile_sampling",
    "validate_quality",
    "validate_timestamp_semantics",
)
