"""Operation 5: posterior-to-anterior visual word-form processing and lexical sequence matching."""
from __future__ import annotations

import math
import unicodedata
from typing import Sequence

import numpy as np

from .contracts import LexicalEntry


def normalize_orthography(value: str) -> str:
    """Apply Unicode compatibility normalization and case folding without discarding letter order."""
    return unicodedata.normalize("NFKC", value).casefold().strip()


def ordered_similarity(left: str, right: str) -> float:
    """Calculate position-sensitive normalized edit similarity for two orthographic sequences."""
    a, b = normalize_orthography(left), normalize_orthography(right)
    if not a and not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + cost))
        previous = current
    return 1.0 - previous[-1] / max(len(a), len(b), 1)


def overlapping_ngrams(sequence: Sequence[str], sizes: Sequence[int] = (2, 3)) -> list[dict[str, object]]:
    """Extract overlapping ordered sublexical units while retaining each unit's start position."""
    units: list[dict[str, object]] = []
    for size in sizes:
        for start in range(max(0, len(sequence) - size + 1)):
            units.append({"n": size, "start": start, "unit": "".join(sequence[start:start + size])})
    return units


class VWFAHierarchyProcessor:
    """Model posterior letter, intermediate n-gram and anterior lexical populations separately."""

    def process(
        self,
        grapheme_sequence: Sequence[str],
        visual_summary: np.ndarray | None = None,
        lexicon: Sequence[LexicalEntry] = (),
    ) -> dict[str, object]:
        """Create ordered orthographic representations and rank only caller-supplied lexical entries."""
        sequence = [str(item) for item in grapheme_sequence]
        joined = "".join(sequence)
        ngrams = overlapping_ngrams(sequence)
        candidates = []
        normalized = normalize_orthography(joined)
        for entry in lexicon:
            similarity = ordered_similarity(normalized, entry.word)
            frequency_prior = math.log1p(float(entry.frequency))
            score = similarity + 0.025 * frequency_prior
            candidates.append({
                "word": entry.word,
                "similarity": float(similarity),
                "score": float(score),
                "source_reference": entry.source_reference,
            })
        scores = np.asarray([item["score"] for item in candidates], dtype=np.float64)
        if scores.size:
            exponentials = np.exp(scores - np.max(scores))
            probabilities = exponentials / max(float(np.sum(exponentials)), 1e-12)
            for item, probability in zip(candidates, probabilities):
                item["probability"] = float(probability)
        candidates.sort(key=lambda item: float(item.get("probability", 0.0)), reverse=True)
        return {
            "posterior_orthography": sequence,
            "intermediate_ngrams": ngrams,
            "anterior_lexical_candidates": candidates,
            "best_lexical_candidate": candidates[0]["word"] if candidates else None,
            "lexical_confidence": float(candidates[0].get("probability", 0.0)) if candidates else 0.0,
            "order_preserved": True,
            "position_tolerant_transformation": "Unicode NFKC + casefold; ordinal sequence remains intact",
            "visual_summary_dimensions": int(np.asarray(visual_summary).size) if visual_summary is not None else 0,
            "biological_analogue": (
                "Posterior visual-feature, intermediate sublexical and anterior lexical hierarchy "
                "in left ventral occipitotemporal/VWFA circuitry; exact stages remain model hypotheses."
            ),
        }
