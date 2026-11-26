"""MAP-Elites quality-diversity archive and search driver (Phase E, steps 39-44).

MAP-Elites (section 2.4) discretizes the behavior descriptor into archive cells and
stores the best *feasible* candidate per cell: ``A[c] <- s`` if the cell is empty or
``Q(s) > Q(A[c])`` — insertions happen only when gates pass *and* cell quality improves
(step 43; acceptance tests 29, 30). Parents are sampled from the archive with a
documented strategy (step 42): uniform over occupied cells with a frontier bias toward
cells adjacent to unexplored regions. Archive coverage, per-cell quality, descriptor
spread and turnover are tracked (step 44) and only independently validated candidates
are handed off to registry approval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..domain.behavior_descriptor import BehaviorDescriptor, DescriptorCalibration
from ..domain.evaluation_record import EvaluationRecord
from ..domain.strategy import StrategyCandidate
from ..evaluation.evaluator import SeededRandom, StrategyEvaluator
from .cell_index import CellIndex, neighboring_cells

#: Parent-selection policy (step 42, documented): uniform over occupied cells with
#: frontier bias — with probability ``FRONTIER_BIAS`` a cell adjacent to an unexplored
#: cell is preferred over a uniformly drawn one.
FRONTIER_BIAS = 0.3


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    """One incumbent strategy in one archive cell."""

    cell_id: str
    coordinates: tuple[int, ...]
    candidate_id: str
    candidate: StrategyCandidate
    quality: float
    fitness: float | None
    descriptor: BehaviorDescriptor
    evaluation_digest: str
    generation_inserted: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id, "coordinates": list(self.coordinates),
            "candidate_id": self.candidate_id, "quality": self.quality,
            "fitness": self.fitness,
            "descriptor": self.descriptor.as_dict(),
            "evaluation_digest": self.evaluation_digest,
            "generation_inserted": self.generation_inserted,
        }


@dataclass(frozen=True, slots=True)
class ArchiveUpdate:
    """Result of one insertion attempt (step 43)."""

    accepted: bool
    cell_id: str | None
    reason: str = ""
    replaced_incumbent: str | None = None
    candidate_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted, "cell_id": self.cell_id, "reason": self.reason,
            "replaced_incumbent": self.replaced_incumbent,
            "candidate_id": self.candidate_id,
        }


@dataclass(frozen=True, slots=True)
class ArchiveSnapshot:
    """Immutable view of the archive for reporting (step 44)."""

    version: str
    total_cells: int
    occupied_cells: int
    entries: tuple[Mapping[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version, "total_cells": self.total_cells,
            "occupied_cells": self.occupied_cells,
            "entries": [dict(entry) for entry in self.entries],
        }


class MapElitesArchive:
    """Quality-diversity archive over versioned behavior descriptors (step 41-44)."""

    def __init__(self, calibration: DescriptorCalibration) -> None:
        self.cell_index = CellIndex(calibration)
        self._entries: dict[str, ArchiveEntry] = {}
        self._coordinates: dict[str, tuple[int, ...]] = {}
        self._turnover: dict[str, int] = {}
        self._seen_hashes: set[str] = set()
        self.insertions = 0
        self.rejections: dict[str, int] = {
            "infeasible": 0, "no_cell": 0, "not_better": 0, "duplicate": 0,
        }

    @property
    def version(self) -> str:
        return self.cell_index.version

    def total_cells(self) -> int:
        return self.cell_index.total_cells()

    def occupied(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------ insertion
    def insert(
        self,
        candidate: StrategyCandidate,
        evaluation: EvaluationRecord,
        descriptor: BehaviorDescriptor,
        *,
        generation: int = 0,
    ) -> ArchiveUpdate:
        """Insert one candidate; gates and cell-improvement rules apply (step 43)."""
        if not evaluation.gate.feasible:
            self.rejections["infeasible"] += 1
            return ArchiveUpdate(
                accepted=False, cell_id=None, candidate_id=candidate.candidate_id,
                reason="candidate is infeasible: hard-constraint gates failed "
                       f"({evaluation.gate.reasons[0] if evaluation.gate.reasons else 'unspecified'})",
            )
        if candidate.content_hash in self._seen_hashes:
            self.rejections["duplicate"] += 1
            return ArchiveUpdate(
                accepted=False, cell_id=None, candidate_id=candidate.candidate_id,
                reason="duplicate candidate hash",
            )
        cell = self.cell_index.cell_id(descriptor)
        if cell is None:
            self.rejections["no_cell"] += 1
            _ok, problems = self.cell_index.descriptor_is_in_range(descriptor)
            reason = "; ".join(problems) or "descriptor cannot be binned"
            return ArchiveUpdate(
                accepted=False, cell_id=None, candidate_id=candidate.candidate_id,
                reason=f"descriptor rejected: {reason}",
            )
        coordinates, cell_id = cell
        incumbent = self._entries.get(cell_id)
        if incumbent is not None and evaluation.metrics.quality <= incumbent.quality:
            self.rejections["not_better"] += 1
            return ArchiveUpdate(
                accepted=False, cell_id=cell_id, candidate_id=candidate.candidate_id,
                reason=f"cell quality {evaluation.metrics.quality:.4f} does not improve "
                       f"incumbent {incumbent.quality:.4f}",
                replaced_incumbent=None,
            )
        self._entries[cell_id] = ArchiveEntry(
            cell_id=cell_id,
            coordinates=coordinates,
            candidate_id=candidate.candidate_id,
            candidate=candidate,
            quality=evaluation.metrics.quality,
            fitness=evaluation.fitness,
            descriptor=descriptor,
            evaluation_digest=evaluation.content_digest(),
            generation_inserted=generation,
        )
        self._coordinates[cell_id] = coordinates
        self._seen_hashes.add(candidate.content_hash)
        self.insertions += 1
        if incumbent is not None:
            self._turnover[cell_id] = self._turnover.get(cell_id, 0) + 1
        return ArchiveUpdate(
            accepted=True, cell_id=cell_id, candidate_id=candidate.candidate_id,
            replaced_incumbent=incumbent.candidate_id if incumbent else None,
        )

    # ----------------------------------------------------------- parent selection
    def select_parents(
        self,
        rng: SeededRandom,
        *,
        count: int = 1,
    ) -> list[ArchiveEntry]:
        """Documented parent selection: uniform occupied cells + frontier bias (step 42)."""
        if not self._entries:
            return []
        entries = list(self._entries.values())
        bins_per_dimension = [d.bins for d in self.cell_index.calibration.dimensions]
        occupied_coordinates = set(self._coordinates.values())
        frontier: list[ArchiveEntry] = []
        for entry in entries:
            for neighbor in neighboring_cells(entry.coordinates, bins_per_dimension):
                if neighbor not in occupied_coordinates:
                    frontier.append(entry)
                    break
        pool = frontier if (frontier and rng.uniform() < FRONTIER_BIAS) else entries
        if not pool:
            pool = entries
        count = min(count, len(pool))
        if count <= 0:
            return []
        return rng.sample(pool, count)

    def entries(self) -> list[ArchiveEntry]:
        return list(self._entries.values())

    def entry(self, cell_id: str) -> ArchiveEntry | None:
        return self._entries.get(cell_id)

    def turnover_rate(self) -> float:
        """Replacements per insertion (step 44: archive turnover)."""
        if self.insertions == 0:
            return 0.0
        replacements = sum(self._turnover.values())
        return replacements / self.insertions

    def snapshot(self) -> ArchiveSnapshot:
        return ArchiveSnapshot(
            version=self.version,
            total_cells=self.total_cells(),
            occupied_cells=self.occupied(),
            entries=tuple(entry.as_dict() for entry in self._entries.values()),
        )

    def handoff(self) -> list[ArchiveEntry]:
        """Hand off incumbents for independent validation (step 44)."""
        return sorted(
            self._entries.values(), key=lambda entry: -entry.quality,
        )

    def rejections_summary(self) -> dict[str, int]:
        return dict(self.rejections)


@dataclass(frozen=True, slots=True)
class QDSearchReport:
    """MAP-Elites search driver report (steps 41-44)."""

    stop_reason: str
    generations: int
    evaluations: int
    insertions: int
    coverage: float
    best_quality: float | None
    turnover_rate: float
    archive_snapshot: Mapping[str, Any]
    handoff_candidate_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stop_reason": self.stop_reason, "generations": self.generations,
            "evaluations": self.evaluations, "insertions": self.insertions,
            "coverage": self.coverage, "best_quality": self.best_quality,
            "turnover_rate": self.turnover_rate,
            "archive": dict(self.archive_snapshot),
            "handoff_candidate_ids": list(self.handoff_candidate_ids),
        }


@dataclass(frozen=True, slots=True)
class QDSearchConfig:
    """Documented MAP-Elites search policy (steps 42-43)."""

    parents_per_generation: int = 4
    offspring_per_parent: int = 2
    generation_limit: int = 6
    seed: int = 20261122
    mutation_probability: float = 0.6
    crossover_probability: float = 0.3

    def as_dict(self) -> dict[str, Any]:
        return {
            "parents_per_generation": self.parents_per_generation,
            "offspring_per_parent": self.offspring_per_parent,
            "generation_limit": self.generation_limit, "seed": self.seed,
            "mutation_probability": self.mutation_probability,
            "crossover_probability": self.crossover_probability,
        }


class MapElitesSearch:
    """MAP-Elites evolutionary loop over the archive (steps 41-44).

    Archive parents produce bounded offspring through the shared genome operators;
    every offspring is sandboxed and evaluated, then inserted only if gates pass and
    cell quality improves. The loop never touches the held-out test split.
    """

    def __init__(
        self,
        evaluator: StrategyEvaluator,
        *,
        config: QDSearchConfig | None = None,
        grammar: Any | None = None,
        event_recorder: Any | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or QDSearchConfig()
        self.grammar = grammar
        self.event_recorder = event_recorder

    def run(
        self,
        archive: MapElitesArchive,
        seed_candidates: Sequence[StrategyCandidate],
        *,
        run_id: str = "qd-run",
        budget: Any | None = None,
    ) -> QDSearchReport:
        """Run archive initialization and the QD loop (steps 41-43)."""
        from ..evaluation.evaluator import BudgetExhausted
        from ..search import operators

        rng = SeededRandom(self.config.seed)
        seeds = tuple(
            self.config.seed + 13 * i + 3 for i in range(self.evaluator.trials_per_task)
        )
        evaluations = 0
        stop_reason = "generation_limit"

        # Step 41: initialize the archive with diverse seed strategies.
        for candidate in seed_candidates:
            record = self.evaluator.evaluate_candidate(
                candidate, split="train", seeds=seeds, budget=budget,
            )
            evaluations += 1
            descriptor = BehaviorDescriptor.from_evaluation(
                record.behavior_measurement, archive.version,
            )
            archive.insert(candidate, record, descriptor, generation=0)
        if self.event_recorder is not None:
            self.event_recorder.emit(
                "archive_initialized", run_id,
                {"occupied_cells": archive.occupied(), "seeds": len(seed_candidates)},
            )

        try:
            for generation in range(1, self.config.generation_limit + 1):
                parents = archive.select_parents(
                    rng, count=self.config.parents_per_generation,
                )
                if not parents:
                    stop_reason = "empty_archive"
                    break
                offspring: list[StrategyCandidate] = []
                for parent in parents:
                    for _ in range(self.config.offspring_per_parent):
                        genome = parent.candidate.genome
                        if rng.uniform() < self.config.crossover_probability:
                            partner = archive.select_parents(rng, count=1)
                            if partner and partner[0].candidate_id != parent.candidate_id:
                                crossed = operators.crossover_genomes(
                                    genome, partner[0].candidate.genome, rng,
                                )
                                if crossed is not None:
                                    genome = crossed
                        if rng.uniform() < self.config.mutation_probability:
                            mutated = operators.mutate_genome(genome, rng)
                            if mutated is not None:
                                genome = mutated
                        from ..domain.strategy import LineageRecord
                        candidate = StrategyCandidate.from_genome(
                            genome,
                            lineage=LineageRecord(
                                parents=(parent.candidate_id,),
                                operation="mutation", generation=generation,
                                run_id=run_id, notes="map-elites offspring",
                            ),
                            run_id=run_id,
                        )
                        offspring.append(candidate)
                for candidate in offspring:
                    record = self.evaluator.evaluate_candidate(
                        candidate, split="train", seeds=seeds, budget=budget,
                    )
                    evaluations += 1
                    descriptor = BehaviorDescriptor.from_evaluation(
                        record.behavior_measurement, archive.version,
                    )
                    update = archive.insert(
                        candidate, record, descriptor, generation=generation,
                    )
                    if self.event_recorder is not None and update.accepted:
                        self.event_recorder.emit(
                            "archive_cell_updated", run_id,
                            {"cell_id": update.cell_id, "candidate_id": candidate.candidate_id},
                        )
                if self.event_recorder is not None:
                    self.event_recorder.emit(
                        "archive_metrics_recorded", run_id,
                        {
                            "generation": generation,
                            "occupied": archive.occupied(),
                            "total": archive.total_cells(),
                        },
                    )
        except BudgetExhausted:
            stop_reason = "budget_exhausted"

        from .archive_metrics import archive_metrics
        metrics = archive_metrics(archive)
        handoff = archive.handoff()
        return QDSearchReport(
            stop_reason=stop_reason,
            generations=self.config.generation_limit,
            evaluations=evaluations,
            insertions=archive.insertions,
            coverage=metrics.coverage,
            best_quality=metrics.best_quality,
            turnover_rate=archive.turnover_rate(),
            archive_snapshot=archive.snapshot().as_dict(),
            handoff_candidate_ids=tuple(entry.candidate_id for entry in handoff),
        )


__all__ = [
    "FRONTIER_BIAS", "ArchiveEntry", "ArchiveSnapshot", "ArchiveUpdate",
    "MapElitesArchive", "MapElitesSearch", "QDSearchConfig", "QDSearchReport",
]
