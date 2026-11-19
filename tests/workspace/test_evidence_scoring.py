"""Tests for weighted evidence contribution and conservative handling of correlation."""
import pytest

from ore_workspace import EvidenceContribution, HypothesisScorer


def test_weighted_evidence_score_is_not_misreported_as_probability():
    """Evidence contributions are weighted and returned as an uncalibrated score."""
    result = HypothesisScorer().score((
        EvidenceContribution("lexicon", 0.8, weight=0.9),
        EvidenceContribution("syntax", 0.5, weight=0.7),
    ))
    assert result.score == pytest.approx(1.07)
    assert result.calibrated_probability is False
    assert len(result.used_evidence) == 2
    assert len(result.all_evidence) == 2


def test_correlated_sources_are_not_double_counted_but_remain_traceable():
    """Only the strongest source contribution in a correlation group enters the score."""
    result = HypothesisScorer().score((
        EvidenceContribution("model_a", 0.7, 1.0, correlation_group="same_model_family"),
        EvidenceContribution("model_b", 0.9, 1.0, correlation_group="same_model_family"),
        EvidenceContribution("human_rule", 0.4, 0.5),
    ))
    assert result.score == pytest.approx(1.1)
    assert result.suppressed_correlated_sources == ("model_a",)
    assert len(result.used_evidence) == 2
    assert len(result.all_evidence) == 3


def test_non_finite_evidence_is_rejected():
    """Scoring cannot be poisoned by NaN or infinite values."""
    with pytest.raises(ValueError):
        EvidenceContribution("bad", float("nan"))