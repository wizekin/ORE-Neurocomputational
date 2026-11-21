"""ACT-R-inspired declarative-memory activation and candidate ranking."""
from __future__ import annotations

import math
from datetime import datetime
from typing import Iterable

from .models import RetrievalCandidate, RetrievalQuery, RetrievalResult


def base_level_activation(
    access_timestamps: Iterable[datetime],
    now: datetime,
    decay: float = 0.5,
) -> float:
    """Compute B = ln(sum((t_now - t_i)^(-d))) with safe age handling.

    Timestamps in the future are treated as age zero and clamped to a small
    positive interval. No access history yields negative infinity.
    """
    if not 0 < decay <= 2:
        raise ValueError("decay must be in (0, 2]")
    ages = []
    for accessed in access_timestamps:
        seconds = max((now - accessed).total_seconds(), 1e-6)
        ages.append(seconds)
    if not ages:
        return float("-inf")
    return math.log(sum(age ** (-decay) for age in ages))


def _normalized_activation(value: float) -> float:
    if value == float("-inf"):
        return 0.0
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def rank_candidates(
    candidates: Iterable[RetrievalCandidate],
    query: RetrievalQuery,
    task_tags: frozenset[str] = frozenset(),
    decay: float = 0.5,
    cost_weight: float = 0.05,
) -> list[RetrievalResult]:
    """Rank only available candidates; missing provenance is not fabricated."""
    cue_set = {x.casefold() for x in query.cues}
    results: list[RetrievalResult] = []
    for candidate in candidates:
        if not candidate.available:
            continue
        activation = base_level_activation(candidate.access_timestamps, query.now, decay)
        cue_match = (
            len(cue_set & {x.casefold() for x in candidate.cue_terms}) / len(cue_set)
            if cue_set else 0.0
        )
        task_relevance = (
            len(task_tags & candidate.task_tags) / len(task_tags)
            if task_tags else 0.0
        )
        provenance = candidate.provenance_score
        score = (
            0.45 * _normalized_activation(activation)
            + 0.30 * cue_match
            + 0.20 * task_relevance
            + (0.05 * provenance if provenance is not None else 0.0)
            - cost_weight * candidate.estimated_cost
        )
        results.append(RetrievalResult(
            candidate, activation, cue_match, task_relevance, provenance, score
        ))
    results.sort(key=lambda item: (-item.total_score, item.candidate.memory_id))
    return results[:query.limit]
