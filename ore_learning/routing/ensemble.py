"""Bounded ensembles with explicit disagreement handling (Point 14, Step 10).

Prefer one suitable specialist; use an ensemble only when independent evidence shows
a benefit worth its cost.  Ensemble size, time, and disagreement handling are
bounded, and correlated models (shared training provenance) are marked as providing
no independent confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..config import LearningConfig
from ..contracts import ModelManifest

__all__ = ("BoundedEnsemble", "EnsembleMember", "EnsembleReport",
           "ensemble_justified", "independence_of")


@dataclass(frozen=True, slots=True)
class EnsembleMember:
    """One member of an ensemble with its output and provenance."""

    identity: str
    dataset_id: str
    output: Any

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"identity": self.identity, "dataset_id": self.dataset_id,
                "output": _jsonable(self.output)}


@dataclass(slots=True)
class EnsembleReport:
    """The bounded ensemble's aggregate and its disagreement record."""

    aggregate: Any
    disagreement: float
    conflict: bool
    members: tuple[EnsembleMember, ...]
    independent_confirmations: int
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"aggregate": _jsonable(self.aggregate),
                "disagreement": round(self.disagreement, 6), "conflict": self.conflict,
                "members": [member.to_dict() for member in self.members],
                "independent_confirmations": self.independent_confirmations,
                "note": self.note}


def independence_of(manifests: Sequence[ModelManifest]) -> int:
    """Count members with distinct training-data provenance (correlated models
    trained on the same dataset do not provide independent confirmation)."""
    dataset_ids = {manifest.training_data_provenance.dataset_id
                   if manifest.training_data_provenance else manifest.model_id
                   for manifest in manifests}
    return len(dataset_ids)


def ensemble_justified(candidates: Sequence[Any], config: LearningConfig,
                       top_utility: float, second_utility: float) -> tuple[bool, str]:
    """Decide whether a bounded ensemble is worth its cost (Step 10).

    An ensemble is justified only when the top candidates are within the configured
    utility margin (evidence that they disagree usefully), the set contains at least
    two independently trained models, and the size stays within the bound.
    """
    if len(candidates) < 2:
        return False, "fewer than two candidates"
    if len(candidates) > config.max_ensemble_size:
        return False, f"more members than the bound {config.max_ensemble_size}"
    if top_utility - second_utility > max(config.min_utility_margin_for_ensemble, 1e-9):
        return False, ("a single specialist dominates; the ensemble cost is not "
                       "worth the marginal benefit")
    manifests = [getattr(candidate, "manifest", candidate) for candidate in candidates]
    if independence_of(manifests) < 2:
        return False, "members share training provenance; no independent confirmation"
    return True, "independent candidates within the utility margin; ensemble justified"


class BoundedEnsemble:
    """Aggregate member outputs with bounded disagreement handling."""

    def __init__(self, config: LearningConfig | None = None,
                 disagreement_threshold: float = 0.30) -> None:
        """Configure bounds."""
        self.config = config or LearningConfig()
        self.disagreement_threshold = float(disagreement_threshold)

    def aggregate(self, members: Sequence[EnsembleMember]) -> EnsembleReport:
        """Aggregate outputs; disagreement is surfaced, never hidden by averaging.

        For numeric outputs the aggregate is the mean and disagreement the mean
        absolute deviation; for label outputs the aggregate is the majority label and
        disagreement the share of dissenting members.  When disagreement exceeds the
        threshold the report is flagged as a conflict that requires more evidence.
        """
        if not members:
            raise ValueError("an ensemble requires at least one member")
        if len(members) > self.config.max_ensemble_size:
            raise ValueError(
                f"ensemble size {len(members)} exceeds the bound "
                f"{self.config.max_ensemble_size}")
        outputs = [member.output for member in members]
        if all(isinstance(output, (int, float, np.floating)) for output in outputs):
            values = np.asarray([float(output) for output in outputs])
            aggregate = float(np.mean(values))
            disagreement = float(np.mean(np.abs(values - aggregate)))
        else:
            labels = [str(output) for output in outputs]
            counts: dict[str, int] = {}
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
            aggregate = max(sorted(counts), key=lambda key: counts[key])
            disagreement = 1.0 - counts[aggregate] / len(labels)
        conflict = disagreement > self.disagreement_threshold
        unique_datasets = len({member.dataset_id for member in members})
        note = ("disagreement exceeds the threshold; retain competing outputs and "
                "request more evidence" if conflict else
                "members agree within the configured bound")
        if unique_datasets < len(members):
            note += "; correlated members provide no independent confirmation"
        return EnsembleReport(
            aggregate=aggregate, disagreement=disagreement, conflict=conflict,
            members=tuple(members),
            independent_confirmations=min(unique_datasets, len(members)), note=note)


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of a value for JSON serialization."""
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    if isinstance(value, np.ndarray):
        return value.round(6).tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)
