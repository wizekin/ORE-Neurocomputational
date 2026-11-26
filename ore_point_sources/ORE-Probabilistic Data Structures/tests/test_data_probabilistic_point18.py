"""Phase C/D/E acceptance tests: Bloom filters, Count-Min sketches, HyperLogLog (Steps 13-31)."""
from __future__ import annotations

import math

import pytest

from ore_data.probabilistic.bloom import (
    BloomFilter,
    BloomParameters,
    BloomPrecheckRouter,
    BloomSizingError,
    CountingBloomFilter,
    MembershipVerdict,
    bloom_use_case_justified,
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


# ---------------------------------------------------------------------------
# Phase C (Steps 13-19)


def test_bloom_sizing_matches_the_formulas():
    """Steps 14-15: m = ceil(-n ln p / (ln 2)^2) and k = round(m/n ln 2)."""
    parameters = BloomParameters.from_cardinality_and_error(10_000, 0.01)
    expected_bits = math.ceil(-10_000 * math.log(0.01) / (math.log(2) ** 2))
    assert abs(parameters.bits - expected_bits) <= 8  # byte-rounded allocation
    expected_probes = round(parameters.bits / 10_000 * math.log(2))
    assert parameters.hash_probes == expected_probes
    assert parameters.bytes_required() == parameters.bits // 8
    with pytest.raises(BloomSizingError):
        BloomParameters.from_cardinality_and_error(0, 0.01)
    with pytest.raises(BloomSizingError):
        BloomParameters.from_cardinality_and_error(100, 1.0)


def test_bloom_filter_has_no_false_negatives_and_bounded_fp_rate():
    """Steps 13-14: negatives are certain for members; the observed FP rate stays near target."""
    parameters = BloomParameters.from_cardinality_and_error(1_000, 0.01)
    bloom = BloomFilter(parameters)
    keys = [f"key-{index}" for index in range(1_000)]
    for key in keys:
        bloom.add(key)
    # No false negatives, ever.
    assert all(bloom.might_contain(key) for key in keys)
    # Measured false-positive rate on a representative absent-key sample.
    absent = [f"absent-{index}" for index in range(20_000)]
    observed = estimate_false_positive_rate(bloom, absent)
    assert observed <= 0.02  # within 2x the 1% target on this sample
    theoretical = theoretical_false_positive_rate(
        parameters.bits, 1_000, parameters.hash_probes
    )
    assert observed <= theoretical * 3 + 0.005


def test_bloom_two_stage_membership_respects_watermarks():
    """Step 17: a negative shortcut only for a complete filter covering the watermark."""
    parameters = BloomParameters.from_cardinality_and_error(64, 0.01)
    bloom = BloomFilter(parameters)
    bloom.add("present-key", watermark=5)
    incomplete = bloom.membership_hint("present-key", authoritative_watermark=5)
    assert incomplete.verdict is MembershipVerdict.UNKNOWN_INCOMPLETE
    assert incomplete.safe_to_skip_exact_lookup is False
    bloom.mark_complete(5)
    present = bloom.membership_hint("present-key", authoritative_watermark=5)
    assert present.verdict is MembershipVerdict.POSSIBLY_PRESENT
    assert present.safe_to_skip_exact_lookup is False
    absent = bloom.membership_hint("absent-key", authoritative_watermark=5)
    assert absent.verdict is MembershipVerdict.DEFINITELY_ABSENT
    assert absent.safe_to_skip_exact_lookup is True
    # A watermark beyond the filter's completeness reverts to unknown.
    future = bloom.membership_hint("absent-key", authoritative_watermark=9)
    assert future.verdict is MembershipVerdict.UNKNOWN_INCOMPLETE
    assert future.safe_to_skip_exact_lookup is False


def test_standard_bloom_refuses_deletion_and_counting_filter_guards_underflow():
    """Step 18: standard filters never delete; the counting filter refuses underflow."""
    parameters = BloomParameters.from_cardinality_and_error(64, 0.01)
    bloom = BloomFilter(parameters)
    bloom.add("k")
    with pytest.raises(NotImplementedError):
        bloom.delete("k")
    counting = CountingBloomFilter(parameters)
    counting.add("k")
    assert counting.might_contain("k")
    counting.remove("k")
    with pytest.raises(ValueError):
        counting.remove("k")  # underflow would corrupt membership behavior
    rebuilt = bloom.rebuild(["a", "b"], watermark=3)
    assert rebuilt.complete and rebuilt.insertion_watermark == 3


def test_bloom_reconciliation_measures_fp_rate_and_recommends_rebuild():
    """Step 19: reconciliation verifies watermarks and rebuilds beyond policy."""
    parameters = BloomParameters.from_cardinality_and_error(256, 0.02)
    bloom = BloomFilter(parameters)
    keys = [f"key-{i}" for i in range(256)]
    for key in keys:
        bloom.add(key)
    bloom.mark_complete(256)
    healthy = reconcile_bloom_filter(
        bloom,
        present_keys=keys[:50],
        absent_keys=[f"absent-{i}" for i in range(4_000)],
        authoritative_watermark=256,
    )
    assert healthy.watermark_verified
    assert healthy.recommendation == "healthy"
    assert healthy.observed_false_positive_rate <= 0.05
    # Stale watermark: the negative shortcut must be bypassed.
    stale = reconcile_bloom_filter(
        bloom,
        present_keys=keys[:10],
        absent_keys=[f"absent-{i}" for i in range(1_000)],
        authoritative_watermark=5_000,
    )
    assert stale.watermark_verified is False
    assert "bypass_negative_shortcut" in stale.recommendation


def test_bloom_precheck_reduces_expensive_lookups():
    """Step 13 + metric: the precheck router avoids exact lookups only on safe negatives."""
    parameters = BloomParameters.from_cardinality_and_error(256, 0.01)
    bloom = BloomFilter(parameters)
    exact = {f"key-{i}": {"value": i} for i in range(256)}
    for key in exact:
        bloom.add(key)
    bloom.mark_complete(256)
    router = BloomPrecheckRouter(bloom, exact)
    for index in range(1_000):
        key = f"key-{index % 512}" if index % 2 else f"absent-{index}"
        router.lookup(key, authoritative_watermark=256)
    stats = router.stats()
    assert stats["avoided_exact_lookups"] > 400  # absent keys skipped the exact store
    assert stats["prechecks"] == 1_000
    assert bloom_use_case_justified(
        exact_lookup_rate_per_s=500.0, exact_lookup_cost_ms=2.0, target_fp_rate=0.01
    )
    assert not bloom_use_case_justified(
        exact_lookup_rate_per_s=1.0, exact_lookup_cost_ms=0.001, target_fp_rate=0.01
    )


# ---------------------------------------------------------------------------
# Phase D (Steps 20-25)


def test_count_min_dimensions_match_the_formulas():
    """Step 21: w = ceil(e/epsilon), d = ceil(ln(1/delta))."""
    parameters = CountMinParameters.from_error_bounds(0.01, 0.01)
    assert parameters.width == math.ceil(math.e / 0.01)
    assert parameters.depth == math.ceil(math.log(1 / 0.01))
    with pytest.raises(ValueError):
        CountMinParameters.from_error_bounds(0, 0.01)
    with pytest.raises(ValueError):
        CountMinParameters.from_error_bounds(0.01, 1.0)


def test_count_min_estimates_upper_bound_frequency_within_error():
    """Steps 22-23: estimates never undershoot and stay within epsilon*N on a controlled stream."""
    parameters = CountMinParameters.from_error_bounds(0.001, 0.001)
    sketch = CountMinSketch(parameters, namespace="test.frequency")
    exact: dict[str, int] = {}
    import random

    rng = random.Random(42)
    keys = [f"key-{i}" for i in range(200)]
    total = 0
    for _ in range(20_000):
        key = keys[rng.randrange(len(keys))]
        sketch.update(key)
        exact[key] = exact.get(key, 0) + 1
        total += 1
    bound = parameters.additive_error_bound(total)
    over = 0
    for key, true in exact.items():
        estimate = sketch.estimate(key)
        assert isinstance(estimate, FrequencyEstimate)
        assert estimate.exact is False
        assert estimate.estimate >= true  # upper-biased approximation
        assert estimate.estimate - true <= bound
        over += estimate.estimate - true
    assert over / total < 0.01  # tiny average overshoot on this controlled stream


def test_count_min_validates_weights_and_separates_namespaces():
    """Step 20 + 22: negative weights are rejected; namespaces do not contaminate."""
    parameters = CountMinParameters.from_error_bounds(0.01, 0.01)
    sketch = CountMinSketch(parameters, namespace="a")
    sketch.update("k", 5)
    assert sketch.estimate("k").estimate == 5.0
    with pytest.raises(ValueError):
        sketch.update("k", -1)
    with pytest.raises(ValueError):
        sketch.update("k", float("nan"))
    other = CountMinSketch(parameters, namespace="b")
    other.update("k", 100)
    assert other.estimate("k").estimate == 100.0
    assert sketch.estimate("k").estimate == 5.0  # namespace b never contaminated a


def test_count_min_estimates_carry_full_metadata():
    """Section 5.1: every estimate carries algorithm, parameters, bound and watermark."""
    parameters = CountMinParameters.from_error_bounds(0.01, 0.05)
    sketch = CountMinSketch(parameters, namespace="meta")
    sketch.update("k", 3, watermark=12)
    estimate = sketch.estimate("k")
    payload = estimate.to_dict()
    assert payload["algorithm"] == "count_min_sketch"
    assert payload["exact"] is False
    assert payload["additive_error_bound"] == parameters.additive_error_bound(3.0)
    assert payload["source_watermark"] == 12
    assert payload["confidence_description"]
    assert payload["timestamp"]


def test_count_min_rotation_separates_recent_from_lifetime_frequency():
    """Step 25: epoch buckets keep lifetime and recent frequency distinct."""
    parameters = CountMinParameters.from_error_bounds(0.005, 0.005)
    sketch = CountMinSketch(parameters, namespace="aging")
    for _ in range(100):
        sketch.update("old-key")
    sketch.rotate()
    for _ in range(10):
        sketch.update("new-key")
    recent_new = sketch.recent_estimate("new-key", epochs=1).estimate
    recent_old = sketch.recent_estimate("old-key", epochs=1).estimate
    assert recent_new == 10.0
    assert recent_old == 0.0
    lifetime_old = sketch.lifetime_estimate("old-key").estimate
    assert lifetime_old >= 100.0
    assert sketch.epoch == 1
    assert sketch.parameters.epsilon == parameters.epsilon


# ---------------------------------------------------------------------------
# Phase E (Steps 26-31)


def test_hyperloglog_accuracy_across_cardinalities():
    """Steps 26-30: relative error stays near 1.04/sqrt(m) across magnitudes."""
    for precision, exact_count in ((10, 1_000), (12, 20_000), (14, 100_000)):
        sketch = HyperLogLog(precision)
        items = {f"item-{index}" for index in range(exact_count)}
        for item in items:
            sketch.add(item)
        estimate = sketch.estimate()
        assert isinstance(estimate, CardinalityEstimate)
        assert estimate.exact is False
        relative_error = abs(estimate.estimate - exact_count) / exact_count
        assert relative_error < 3.0 * estimate.relative_standard_error
        payload = estimate.to_dict()
        assert payload["registers"] == 1 << precision
        assert payload["hash_algorithm"] == "sha256-64"
        assert payload["precision"] == precision


def test_hyperloglog_small_range_correction_applies():
    """Step 30: small cardinalities use linear counting with near-zero error."""
    sketch = HyperLogLog(12)
    for index in range(50):
        sketch.add(f"tiny-{index}")
    estimate = sketch.estimate()
    assert abs(estimate.estimate - 50) / 50 < 0.05
    assert sketch.estimate().correction_applied in (
        "small_range_linear_counting",
        "none",
    )


def test_hyperloglog_merges_compatible_and_rejects_incompatible():
    """Step 31: register-wise max merge; incompatible merges are rejected."""
    first = HyperLogLog(10)
    second = HyperLogLog(10)
    for index in range(500):
        first.add(f"a-{index}")
        second.add(f"b-{index}")
    merged = first.merge(second)
    exact = {f"a-{i}" for i in range(500)} | {f"b-{i}" for i in range(500)}
    estimate = merged.estimate()
    assert abs(estimate.estimate - len(exact)) / len(exact) < 0.10
    other = HyperLogLog(11)
    with pytest.raises(HyperLogLogError):
        first.merge(other)
    mixed_hash = HyperLogLog(10, hash_algorithm="other")
    with pytest.raises(HyperLogLogError):
        first.merge(mixed_hash)


def test_hyperloglog_precision_selection():
    """Step 27: precision p gives m = 2^p registers; target errors map to precisions."""
    assert required_precision_for(0.065) <= 8
    assert required_precision_for(0.01) == 14  # 1.04/sqrt(2^14) ~ 0.008 <= 0.01
    sketch = HyperLogLog(4)
    assert sketch.m == 16
    with pytest.raises(HyperLogLogError):
        HyperLogLog(20)
