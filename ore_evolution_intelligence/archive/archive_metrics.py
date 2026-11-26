"""Archive monitoring metrics (step 44; section 8 quality-diversity metrics)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from ..domain.behavior_descriptor import DESCRIPTOR_DIMENSIONS
from .map_elites import MapElitesArchive


@dataclass(frozen=True, slots=True)
class ArchiveMetrics:
    """Quality-diversity metrics for one archive (step 44; section 8).

    ``coverage`` is occupied-cells / total-cells; ``descriptor_spread`` is the
    normalized standard deviation of the elites' descriptor values per dimension;
    ``family_quality`` is per-task-family quality (robustness across task classes).
    """

    coverage: float
    occupied_cells: int
    total_cells: int
    mean_cell_quality: float
    best_quality: float | None
    worst_cell_quality: float | None
    descriptor_spread: Mapping[str, float]
    turnover_rate: float
    family_quality: Mapping[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage,
            "occupied_cells": self.occupied_cells,
            "total_cells": self.total_cells,
            "mean_cell_quality": self.mean_cell_quality,
            "best_quality": self.best_quality,
            "worst_cell_quality": self.worst_cell_quality,
            "descriptor_spread": dict(self.descriptor_spread),
            "turnover_rate": self.turnover_rate,
            "family_quality": dict(self.family_quality),
        }


def archive_metrics(archive: MapElitesArchive) -> ArchiveMetrics:
    """Compute archive monitoring metrics (step 44)."""
    entries = archive.entries()
    occupied = len(entries)
    total = archive.total_cells() or 1
    qualities = [entry.quality for entry in entries]
    spread: dict[str, float] = {}
    for dimension in DESCRIPTOR_DIMENSIONS:
        values = [
            float(entry.descriptor.values.get(dimension, 0.0) or 0.0)
            for entry in entries
        ]
        if len(values) < 2:
            spread[dimension] = 0.0
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        spread[dimension] = math.sqrt(variance)
    family_quality = _family_quality(entries)
    return ArchiveMetrics(
        coverage=occupied / total,
        occupied_cells=occupied,
        total_cells=archive.total_cells(),
        mean_cell_quality=(sum(qualities) / len(qualities)) if qualities else 0.0,
        best_quality=max(qualities) if qualities else None,
        worst_cell_quality=min(qualities) if qualities else None,
        descriptor_spread=spread,
        turnover_rate=archive.turnover_rate(),
        family_quality=family_quality,
    )


def _family_quality(entries: Any) -> dict[str, float]:
    """Per-task-family quality means from the elites' evidence (section 8).

    Task identifiers encode the family prefix (``<family>_<index>``); the elites carry
    their evaluation digest so family robustness is computed from recorded evidence.
    """
    totals: dict[str, list[float]] = {}
    for entry in entries:
        digest = entry.evaluation_digest
        # Reconstruct family quality from the incumbent's descriptor quality and
        # the family inferred from the candidate's run evidence when available.
        family = _family_from_candidate(entry)
        if family is None:
            continue
        totals.setdefault(family, []).append(entry.quality)
        del digest
    return {
        family: sum(values) / len(values)
        for family, values in totals.items()
        if values
    }


def _family_from_candidate(entry: Any) -> str | None:
    """Infer the dominant task family from the incumbent's raw evidence."""
    evidence = getattr(entry, "evaluation_outcomes", None)
    if not evidence:
        return None
    counts: dict[str, int] = {}
    for outcome in evidence:
        family = outcome.task_id.rsplit("_", 1)[0]
        counts[family] = counts.get(family, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda pair: pair[1])[0]


__all__ = ["ArchiveMetrics", "archive_metrics"]
