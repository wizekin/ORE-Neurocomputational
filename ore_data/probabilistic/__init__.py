"""Probabilistic structures of the Point 18 data infrastructure (Phases C-E).

Bloom filters for membership acceleration, Count-Min sketches for frequency
estimation and HyperLogLog for cardinality estimation: every structure
returns typed estimates that carry algorithm, version, parameters, error
bound, source watermark and timestamp, and none of them ever becomes an
authority for exact truth.
"""
from ore_data.probabilistic.bloom import (
    BloomFilter,
    BloomParameters,
    BloomPrecheckRouter,
    BloomReconciliationReport,
    CountingBloomFilter,
    MembershipHint,
    MembershipVerdict,
    estimate_false_positive_rate,
    reconcile_bloom_filter,
    theoretical_false_positive_rate,
)
from ore_data.probabilistic.count_min import (
    CountMinParameters,
    CountMinSketch,
    FrequencyEstimate,
)
from ore_data.probabilistic.hyperloglog import (
    CardinalityEstimate,
    HyperLogLog,
    HyperLogLogError,
    required_precision_for,
)

__all__ = [
    "BloomFilter",
    "BloomParameters",
    "BloomPrecheckRouter",
    "BloomReconciliationReport",
    "CountingBloomFilter",
    "MembershipHint",
    "MembershipVerdict",
    "estimate_false_positive_rate",
    "reconcile_bloom_filter",
    "theoretical_false_positive_rate",
    "CountMinParameters",
    "CountMinSketch",
    "FrequencyEstimate",
    "CardinalityEstimate",
    "HyperLogLog",
    "HyperLogLogError",
    "required_precision_for",
]
