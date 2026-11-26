"""Novelty search and Pareto archives (section 2.5).

Novelty (section 2.5) is the mean distance to the k nearest behaviour descriptors:
``N(s) = (1/k) sum_{j in kNN(s)} d(b(s), b(s_j))``. Novelty encourages exploration; it
does *not* measure correctness, so novelty scores are only ever combined with quality
gates — infeasible candidates can never be selected through novelty (acceptance test
31: novelty cannot bypass quality/safety gates).

For conflicting objectives the Pareto archive retains the non-dominated set: candidate
``a`` dominates ``b`` when it is no worse on every objective and strictly better on at
least one.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..domain.behavior_descriptor import BehaviorDescriptor

#: Default k for novelty computation.
DEFAULT_K = 5


@dataclass(frozen=True, slots=True)
class NoveltyReport:
    """Novelty measurement for one descriptor."""

    novelty: float
    neighbors: int
    archive_size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "novelty": self.novelty, "neighbors": self.neighbors,
            "archive_size": self.archive_size,
        }


class NoveltyArchive:
    """Bounded FIFO of observed behavior descriptors (novelty reference set)."""

    def __init__(self, *, k: int = DEFAULT_K, capacity: int = 64) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        if capacity < k:
            raise ValueError("capacity must hold at least k descriptors")
        self.k = k
        self.capacity = capacity
        self._descriptors: deque[BehaviorDescriptor] = deque(maxlen=capacity)

    def observe(self, descriptor: BehaviorDescriptor) -> None:
        self._descriptors.append(descriptor)

    def novelty(self, descriptor: BehaviorDescriptor) -> NoveltyReport:
        """Mean distance to the k nearest observed descriptors (section 2.5)."""
        if not self._descriptors:
            return NoveltyReport(novelty=0.0, neighbors=0, archive_size=0)
        distances = sorted(
            (descriptor.distance(other) for other in self._descriptors),
        )
        neighbors = distances[: self.k]
        if not neighbors:
            return NoveltyReport(novelty=0.0, neighbors=0, archive_size=len(self._descriptors))
        novelty = sum(neighbors) / len(neighbors)
        return NoveltyReport(
            novelty=novelty, neighbors=len(neighbors),
            archive_size=len(self._descriptors),
        )

    def __len__(self) -> int:
        return len(self._descriptors)


class NoveltyGuidedSelector:
    """Quality-gated novelty selection (acceptance test 31).

    Combines constrained fitness with a novelty bonus **only among feasible
    candidates**; hard-constraint violations always rank below every feasible
    candidate, so novelty can never bypass quality or safety gates.
    """

    def __init__(self, *, novelty_weight: float = 0.3, k: int = DEFAULT_K) -> None:
        if novelty_weight < 0:
            raise ValueError("novelty weight must be non-negative")
        self.novelty_weight = novelty_weight
        self.archive = NoveltyArchive(k=k)

    def combined_score(
        self, fitness: float | None, descriptor: BehaviorDescriptor,
    ) -> float:
        """Selection score: fitness plus novelty bonus, feasible candidates only."""
        if fitness is None:
            return float("-inf")  # infeasible: never selected through novelty
        report = self.archive.novelty(descriptor)
        return fitness + self.novelty_weight * report.novelty

    def observe(self, descriptor: BehaviorDescriptor) -> None:
        self.archive.observe(descriptor)


#: Objective directions: True = maximize, False = minimize.
ObjectiveDirections = Mapping[str, bool]

DEFAULT_OBJECTIVES: dict[str, bool] = {
    "quality": True, "reliability": True,
    "latency_mean_ms": False, "cost": False,
}


def dominates(
    objectives_a: Mapping[str, float],
    objectives_b: Mapping[str, float],
    directions: ObjectiveDirections = DEFAULT_OBJECTIVES,
) -> bool:
    """Pareto domination (section 2.5): no worse everywhere, strictly better once."""
    no_worse = True
    strictly_better = False
    for name, maximize in directions.items():
        a, b = objectives_a[name], objectives_b[name]
        if maximize:
            if a < b:
                no_worse = False
            if a > b:
                strictly_better = True
        else:
            if a > b:
                no_worse = False
            if a < b:
                strictly_better = True
    return no_worse and strictly_better


@dataclass(frozen=True, slots=True)
class ParetoEntry:
    """One non-dominated archive member."""

    key: str
    objectives: Mapping[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "objectives": dict(self.objectives)}


@dataclass(frozen=True, slots=True)
class ParetoUpdate:
    """Result of one Pareto insertion attempt."""

    accepted: bool
    dominated_removed: tuple[str, ...] = ()
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "dominated_removed": list(self.dominated_removed),
            "reason": self.reason,
        }


class ParetoArchive:
    """Pareto set over conflicting objectives (section 2.5, step 22).

    The archive only ever contains non-dominated candidates: inserting a dominated
    candidate is rejected, and inserting a dominator removes the dominated members.
    """

    def __init__(
        self,
        *,
        directions: ObjectiveDirections = DEFAULT_OBJECTIVES,
        capacity: int = 64,
    ) -> None:
        self.directions = dict(directions)
        self.capacity = capacity
        self._entries: dict[str, Mapping[str, float]] = {}

    def insert(self, key: str, objectives: Mapping[str, float]) -> ParetoUpdate:
        """Insert one candidate; returns the update receipt (step 22)."""
        for name in self.directions:
            if name not in objectives:
                return ParetoUpdate(
                    accepted=False,
                    reason=f"missing objective {name!r}",
                )
        for existing_key, existing in self._entries.items():
            if dominates(existing, objectives, self.directions):
                return ParetoUpdate(
                    accepted=False,
                    reason=f"dominated by {existing_key}",
                )
        dominated_removed: list[str] = []
        for existing_key, existing in list(self._entries.items()):
            if dominates(objectives, existing, self.directions):
                dominated_removed.append(existing_key)
                del self._entries[existing_key]
        if key in self._entries:
            del self._entries[key]
        if len(self._entries) >= self.capacity and key not in self._entries:
            return ParetoUpdate(
                accepted=False,
                dominated_removed=tuple(dominated_removed),
                reason="archive capacity reached",
            )
        self._entries[key] = dict(objectives)
        return ParetoUpdate(
            accepted=True, dominated_removed=tuple(dominated_removed),
        )

    def non_dominated(self) -> list[ParetoEntry]:
        """All non-dominated members (acceptance test 32)."""
        return [
            ParetoEntry(key=key, objectives=objectives)
            for key, objectives in self._entries.items()
        ]

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self) -> list[str]:
        return sorted(self._entries)


def pareto_front_is_consistent(
    entries: Sequence[ParetoEntry],
    directions: ObjectiveDirections = DEFAULT_OBJECTIVES,
) -> bool:
    """Helper for acceptance test 32: verify mutual non-domination."""
    for i, entry_a in enumerate(entries):
        for entry_b in entries[i + 1:]:
            if dominates(entry_a.objectives, entry_b.objectives, directions):
                return False
            if dominates(entry_b.objectives, entry_a.objectives, directions):
                return False
    return True


__all__ = [
    "DEFAULT_K", "DEFAULT_OBJECTIVES", "NoveltyArchive", "NoveltyGuidedSelector",
    "NoveltyReport", "ParetoArchive", "ParetoEntry", "ParetoUpdate", "dominates",
    "pareto_front_is_consistent",
]
