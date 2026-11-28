"""Pure mathematical functions of the Point 15 agentic layer.

Every formula cited in the specification lives here as a small, testable
function: the risk score, the worker-selection objective, bounded backoff,
queue utilization, retry reliability, correlated-evidence aggregation, the
hybrid retrieval score, and ranking metrics.  None of these functions mutates
state or treats a ranking score as a probability.
"""
from __future__ import annotations

import math
import random
from typing import Iterable, Mapping, Sequence

__all__ = (
    "bayes_posterior",
    "backoff_delay",
    "hybrid_retrieval_score",
    "independent_confirmation_count",
    "inverted_cdf",
    "mean",
    "ndcg",
    "precision_at_k",
    "quorum_satisfied",
    "recall_at_k",
    "risk_score",
    "retry_success_probability",
    "unit_scale",
    "utilization",
    "worker_score",
    "mrr",
)


def mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty sequence."""
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def unit_scale(value: float, maximum: float) -> float:
    """Scale a non-negative value by a positive maximum into [0, 1]."""
    if maximum <= 0.0:
        return 0.0
    return max(0.0, min(1.0, value / maximum))


def inverted_cdf(value: float, maximum: float) -> float:
    """Return 1 - value/maximum clamped to [0, 1]; smaller is better."""
    return 1.0 - unit_scale(value, maximum)


def risk_score(factors: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Return the configurable risk score R = sum(w_i * r_i) (Step 3).

    Factor values and weights are clamped to [0, 1]; the result is a bounded
    ordinal score for threshold comparison, not a probability.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    total = 0.0
    weight_sum = 0.0
    for name, weight in weights.items():
        factor = max(0.0, min(1.0, float(factors.get(name, 0.0))))
        clamped_weight = max(0.0, float(weight))
        total += clamped_weight * factor
        weight_sum += clamped_weight
    if weight_sum <= 0.0:
        raise ValueError("weights must have positive mass")
    return max(0.0, min(1.0, total / weight_sum))


def worker_score(fit: float, quality: float, latency: float, cost: float,
                 uncertainty: float, load: float,
                 weights: Mapping[str, float]) -> float:
    """Return S(a, t) = w_f*F + w_q*Q - w_l*L - w_c*C - w_u*U - w_r*R (Step 11).

    Inputs are pre-normalized to [0, 1]: latency, cost, uncertainty and load
    enter negatively.  The score is a ranking objective calibrated on
    evaluation data, not a probability.
    """
    return (
        float(weights.get("fit", 0.0)) * max(0.0, min(1.0, fit))
        + float(weights.get("quality", 0.0)) * max(0.0, min(1.0, quality))
        - float(weights.get("latency", 0.0)) * max(0.0, min(1.0, latency))
        - float(weights.get("cost", 0.0)) * max(0.0, min(1.0, cost))
        - float(weights.get("uncertainty", 0.0)) * max(0.0, min(1.0, uncertainty))
        - float(weights.get("load", 0.0)) * max(0.0, min(1.0, load))
    )


def backoff_delay(attempt: int, initial_delay: float, maximum_delay: float,
                  jitter_seconds: float,
                  rng: random.Random | None = None) -> float:
    """Return d_k = min(d_max, d_0 * 2^k) + J_k with bounded jitter (Step 16)."""
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    deterministic = min(maximum_delay, initial_delay * (2 ** attempt))
    generator = rng if rng is not None else random.Random(attempt)
    jitter = generator.uniform(0.0, max(0.0, jitter_seconds))
    return deterministic + jitter


def utilization(arrival_rate: float, service_rate: float) -> float | None:
    """Return queue utilization rho = lambda / mu, or None when service is unmeasured.

    A sustained value at or above 1.0 signals instability in the simple
    single-server approximation; real queue distributions must be measured. A
    zero service rate has no finite utilization estimate and is represented as
    unavailable rather than infinity.
    """
    arrival_rate = float(arrival_rate)
    service_rate = float(service_rate)
    if (
        not math.isfinite(arrival_rate) or arrival_rate < 0.0
        or not math.isfinite(service_rate) or service_rate < 0.0
    ):
        raise ValueError("arrival and service rates must be finite and non-negative")
    if service_rate == 0.0:
        return None
    return arrival_rate / service_rate


def retry_success_probability(per_attempt: float, attempts: int) -> float:
    """Return the idealized 1 - (1 - p)^n under independent attempts (Step 16).

    Real failures are correlated; this is an upper-bound model, not a
    guarantee.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if not 0.0 <= per_attempt <= 1.0:
        raise ValueError("per_attempt must be in [0, 1]")
    return 1.0 - (1.0 - per_attempt) ** attempts


def independent_confirmation_count(correlation_groups: Sequence[str | None]) -> int:
    """Count independent confirmations among supporting evidence items.

    Sources sharing a non-None correlation group are correlated copies and
    collapse to a single confirmation; an ungrouped (``None``) item counts as
    its own potentially independent source (Step 27).
    """
    ungrouped = sum(1 for group in correlation_groups if group is None)
    grouped = len({group for group in correlation_groups if group is not None})
    return ungrouped + grouped


def bayes_posterior(prior: float, likelihood_given_h: float,
                    likelihood_given_not_h: float) -> float:
    """Return P(H|E) = P(E|H)P(H) / P(E) for a binary piece of evidence.

    Callers must justify independence before combining evidence pieces; this
    function performs a single update only and never averages away conflict.
    """
    for name, value in (("prior", prior), ("likelihood_given_h", likelihood_given_h),
                        ("likelihood_given_not_h", likelihood_given_not_h)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    denominator = likelihood_given_h * prior + likelihood_given_not_h * (1.0 - prior)
    if denominator <= 0.0:
        raise ValueError("evidence has zero probability under both hypotheses")
    return likelihood_given_h * prior / denominator


def hybrid_retrieval_score(lexical: float, semantic: float, graph: float,
                           freshness: float, authority: float,
                           weights: Mapping[str, float]) -> float:
    """Return R(d, q) = a*R_lex + b*R_sem + g*R_graph + d*R_fresh + e*R_auth (Step 41).

    Components are expected in [0, 1]; the blend is a ranking score that
    requires calibration before any probabilistic interpretation.
    """
    components = (lexical, semantic, graph, freshness, authority)
    if any(value < 0.0 or value > 1.0 for value in components):
        raise ValueError("retrieval components must be in [0, 1]")
    return (
        float(weights.get("lexical", 0.0)) * lexical
        + float(weights.get("semantic", 0.0)) * semantic
        + float(weights.get("graph", 0.0)) * graph
        + float(weights.get("freshness", 0.0)) * freshness
        + float(weights.get("authority", 0.0)) * authority
    )


def quorum_satisfied(votes: Sequence[bool], semantics: str,
                     quorum: float = 0.5,
                     weights: Mapping[str, float] | None = None) -> bool:
    """Evaluate all / any / quorum / weighted-quorum join semantics (Step 19)."""
    if not votes:
        return False
    if semantics == "all":
        return all(votes)
    if semantics == "any":
        return any(votes)
    if semantics == "quorum":
        affirmative = sum(1 for vote in votes if vote)
        return affirmative / len(votes) >= quorum
    if semantics == "weighted_quorum":
        if weights is None:
            raise ValueError("weighted_quorum requires weights")
        total = sum(float(value) for value in weights.values())
        if total <= 0.0:
            raise ValueError("weights must have positive mass")
        affirmative = sum(float(list(weights.values())[index])
                          for index, vote in enumerate(votes) if vote)
        return affirmative / total >= quorum
    raise ValueError(f"unknown join semantics: {semantics}")


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Return Recall@k of a ranked list against a labeled relevant set."""
    if not relevant_ids:
        return 0.0
    top = list(ranked_ids[:k])
    hits = sum(1 for item in relevant_ids if item in top)
    return hits / len(relevant_ids)


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Return Precision@k of a ranked list."""
    if k <= 0:
        return 0.0
    top = list(ranked_ids[:k])
    relevant = set(relevant_ids)
    hits = sum(1 for item in top if item in relevant)
    return hits / len(top)


def mrr(ranked_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Return mean reciprocal rank of the first relevant hit."""
    relevant = set(relevant_ids)
    for rank, item in enumerate(ranked_ids, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg(relevance_scores: Sequence[float]) -> float:
    """Return normalized discounted cumulative gain of a ranked relevance list."""
    if not relevance_scores:
        return 0.0
    def _dcg(scores: Sequence[float]) -> float:
        return sum(score / math.log2(index + 2)
                   for index, score in enumerate(scores))
    ideal = sorted(relevance_scores, reverse=True)
    denominator = _dcg(ideal)
    if denominator <= 0.0:
        return 0.0
    return _dcg(relevance_scores) / denominator
