"""Phase F acceptance tests: streaming quantiles (KLL) and heavy hitters (Space-Saving) (Steps 32-36)."""
from __future__ import annotations

import random

import pytest

from ore_data.streaming.heavy_hitters import (
    HeavyHitterCandidate,
    SpaceSavingSummary,
    heavy_hitter_candidates_are_not_authoritative,
)
from ore_data.streaming.quantiles import KLLQuantileSketch, QuantileEstimate


# ---------------------------------------------------------------------------
# Steps 32-34


def test_kll_quantile_estimates_track_exact_percentiles():
    """Steps 32-34: estimates stay within the nominal rank-error bound of KLL."""
    import bisect

    rng = random.Random(11)
    observations = [rng.gauss(100.0, 15.0) for _ in range(20_000)]
    k = 1024
    sketch = KLLQuantileSketch(k=k, retain_samples=len(observations))
    for value in observations:
        sketch.update(value)
    ordered = sorted(observations)
    nominal_bound = sketch.nominal_error_bound()  # ~ 2/sqrt(k) = 6.25%
    for percentile in (0.50, 0.90, 0.95, 0.99):
        estimate = sketch.query(percentile)
        assert isinstance(estimate, QuantileEstimate)
        assert estimate.exact is False
        rank = bisect.bisect_left(ordered, estimate.value) / len(ordered)
        assert abs(rank - percentile) < nominal_bound, (percentile, rank)
        payload = estimate.to_dict()
        assert payload["algorithm"] == "kll"
        assert payload["unit"] == "ms"
        assert payload["nominal_error_bound"] > 0
        assert payload["observations"] == len(observations)
    # Value-level sanity for the median of a Gaussian.
    median = sketch.query(0.5).value
    assert abs(median - 100.0) < 2.5


def test_kll_validates_observations_and_keeps_exact_distinct():
    """Step 33: non-finite observations are rejected; exact samples stay separate."""
    sketch = KLLQuantileSketch(k=64)
    sketch.update(1.0)
    sketch.update(2.0)
    with pytest.raises(ValueError):
        sketch.update(float("nan"))
    with pytest.raises(ValueError):
        sketch.update(float("inf"))
    estimate = sketch.query(0.5)
    assert estimate.exact is False  # approximate summary values stay qualified
    with pytest.raises(ValueError):
        sketch.query(1.5)


def test_kll_sketch_stays_bounded_under_large_streams():
    """Step 32: the sketch does not store every sample."""
    sketch = KLLQuantileSketch(k=128)
    rng = random.Random(5)
    for index in range(50_000):
        sketch.update(rng.random() * 1_000)
    assert sketch.observation_count == 50_000
    assert sketch.memory_bytes() < 50_000 * 8  # far below exact storage
    assert sketch.query(0.5).value != 0


def test_kll_deterministic_with_seeded_compaction():
    """Compaction randomness is seeded: identical streams give identical estimates."""
    values = [random.Random(3).random() for _ in range(5_000)]
    first = KLLQuantileSketch(k=64, seed=7)
    second = KLLQuantileSketch(k=64, seed=7)
    for value in values:
        first.update(value)
        second.update(value)
    for percentile in (0.5, 0.9, 0.99):
        assert first.query(percentile).value == second.query(percentile).value


# ---------------------------------------------------------------------------
# Steps 35-36


def test_space_saving_finds_heavy_hitters_with_documented_bounds():
    """Step 35: estimates bound the true frequency within W/k."""
    summary = SpaceSavingSummary(capacity=32)
    # Hot keys carry clearly more weight than the error bound admits.
    for key_index in range(10):
        for _ in range(500):
            summary.update(f"hot-{key_index}")
    for key_index in range(30):
        for _ in range(50):
            summary.update(f"cold-{key_index}")
    exact = {f"hot-{i}": 500 for i in range(10)}
    exact.update({f"cold-{i}": 50 for i in range(30)})
    candidates = summary.top(10)
    assert candidates and all(isinstance(c, HeavyHitterCandidate) for c in candidates)
    error_bound = summary.error_bound()
    assert error_bound < 450  # the hot/cold gap exceeds the error bound
    for candidate in candidates:
        true = exact.get(candidate.key, 0)
        assert candidate.estimate >= true
        assert candidate.estimate - true <= error_bound + 1e-9
        assert candidate.exact is False
        assert "W/k" in candidate.guarantee_description()
    top_exact = {k for k, _ in sorted(exact.items(), key=lambda p: p[1], reverse=True)[:10]}
    top_reported = {candidate.key for candidate in candidates}
    assert top_reported == top_exact


def test_space_saving_rejects_negative_weights():
    """Step 35: weighted streams validate non-negative weights."""
    summary = SpaceSavingSummary(8)
    summary.update("k", 3.5)
    assert summary.total_weight == 3.5
    with pytest.raises(ValueError):
        summary.update("k", -1)
    with pytest.raises(ValueError):
        summary.update("k", float("inf"))


def test_heavy_hitter_candidates_are_validated_against_exact_aggregation():
    """Step 36: recall/precision are measured against exact top-k over a retained window."""
    summary = SpaceSavingSummary(capacity=16, retain_window=7_000)
    # Hot keys clearly exceed the W/k error bound for this capacity.
    for key_index in range(8):
        for _ in range(600):
            summary.update(f"hot-{key_index}")
    for key_index in range(22):
        for _ in range(100):
            summary.update(f"cold-{key_index}")
    exact = summary.exact_aggregation()
    validation = summary.validate_against_exact(exact, top_k=8)
    assert validation.recall == 1.0
    assert validation.precision == 1.0
    assert validation.to_dict()["requested_top_k"] == 8
    message = heavy_hitter_candidates_are_not_authoritative(summary.top(5))
    assert "high-impact decisions require exact validation" in message


def test_unmonitored_keys_are_bounded_by_w_over_k():
    """Step 35 guarantee: every unmonitored key's true frequency is at most W/k."""
    summary = SpaceSavingSummary(capacity=8)
    exact: dict[str, int] = {}
    rng = random.Random(77)
    for _ in range(2_000):
        key = f"key-{rng.randrange(24)}"
        summary.update(key)
        exact[key] = exact.get(key, 0) + 1
    bound = summary.total_weight / summary.capacity
    monitored = {candidate.key for candidate in summary.top()}
    for key, true in exact.items():
        if key not in monitored:
            assert true <= bound + 1e-9


def test_similarity_of_streams_with_retained_window_exact_percentile():
    """KLL keeps an exact retained window only for validation (Step 36 aid)."""
    sketch = KLLQuantileSketch(k=128, retain_samples=500)
    values = [float(index) for index in range(500)]
    for value in values:
        sketch.update(value)
    assert sketch.exact_percentile(0.5) == 250.0
    approximate = sketch.query(0.5).value
    assert abs(approximate - 250.0) < 5.0
