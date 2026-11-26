"""Streaming algorithms of the Point 18 data infrastructure (Phase F).

KLL quantile sketches for operational percentile telemetry and
Space-Saving heavy-hitter summaries for frequent-key detection; both return
typed estimates with documented error bounds and both feed capacity
analysis and observability, never correctness claims.
"""
from ore_data.streaming.quantiles import KLLQuantileSketch, QuantileEstimate
from ore_data.streaming.heavy_hitters import (
    HeavyHitterCandidate,
    HeavyHitterValidation,
    SpaceSavingSummary,
)

__all__ = [
    "KLLQuantileSketch",
    "QuantileEstimate",
    "HeavyHitterCandidate",
    "HeavyHitterValidation",
    "SpaceSavingSummary",
]
