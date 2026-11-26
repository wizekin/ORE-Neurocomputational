"""Archive-layer unit tests (Phase E, steps 39-44; acceptance tests 19-21)."""
from __future__ import annotations

import pytest

from ore_evolution_intelligence.archive.archive_metrics import archive_metrics
from ore_evolution_intelligence.archive.cell_index import CellIndex, neighboring_cells
from ore_evolution_intelligence.archive.map_elites import (
    MapElitesArchive, MapElitesSearch, QDSearchConfig,
)
from ore_evolution_intelligence.domain.behavior_descriptor import (
    BehaviorDescriptor, DescriptorCalibration, default_calibration,
)
from ore_evolution_intelligence.domain.evaluation_record import (
    AggregateMetrics, EvaluationRecord, GateDecision,
)
from ore_evolution_intelligence.evaluation.evaluator import SeededRandom

from .fixtures import make_baseline_candidate, make_candidate, make_evaluator


def _metrics(quality: float = 0.8, **overrides) -> AggregateMetrics:
    values = dict(
        quality=quality, reliability=0.9, latency_mean_ms=200.0,
        latency_p95_ms=240.0, memory_peak_mb=40.0, cpu_total_ms=1000.0,
        cost=60.0, cost_mean=15.0, safety_violations=0, failure_rate=0.0,
        timeout_rate=0.0, invalid_rate=0.0, abstain_rate=0.0,
        quality_std=0.02, quality_ci95=0.02, worst_quality=quality,
        n_trials=9, n_tasks=3, mean_active_modules=3.0, cooperation=0.8,
    )
    values.update(overrides)
    return AggregateMetrics(**values)


def _record(quality: float = 0.8, feasible: bool = True, **overrides) -> EvaluationRecord:
    return EvaluationRecord(
        candidate_id="cand_test", candidate_version=1, genome_hash="h" * 64,
        benchmark_id="bench", benchmark_version="v1", split="train",
        seeds=(1,), metrics=_metrics(quality, **overrides),
        outcomes=(), gate=GateDecision(feasible=feasible),
        fitness=0.9 if feasible else None,
    )


def _descriptor(speed: float = 250.0, depth: float = 4.0) -> BehaviorDescriptor:
    return BehaviorDescriptor(
        {
            "speed": speed, "reasoning_depth": depth, "resource_tier": 40.0,
            "reliability": 0.9, "module_cooperation": 0.7,
        },
        "calibration-v1",
    )


class TestCellIndex:
    """Step 40: versioned boundaries and explicit descriptor handling."""

    def test_boundaries_reproducible(self):
        # Acceptance test 21: descriptor boundaries reproducible.
        one = CellIndex(default_calibration())
        two = CellIndex(default_calibration(version="calibration-v1"))
        assert one.boundaries() == two.boundaries()
        assert one.version == two.version

    def test_new_calibration_is_new_version(self):
        different = default_calibration(version="calibration-v2")
        assert CellIndex(different).version != CellIndex(
            default_calibration()
        ).version

    def test_total_cells_positive(self):
        index = CellIndex(default_calibration())
        assert index.total_cells() > 1

    def test_missing_descriptor_rejected(self):
        index = CellIndex(default_calibration())
        descriptor = BehaviorDescriptor(
            {"speed": None, "reasoning_depth": None, "resource_tier": None,
             "reliability": None, "module_cooperation": None},
            "calibration-v1",
        )
        assert index.cell_id(descriptor) is None

    def test_neighboring_cells_bounded(self):
        neighbors = neighboring_cells((0, 0, 0, 0, 0), (4, 4, 4, 4, 4))
        assert all(len(n) == 5 for n in neighbors)
        assert all(0 <= c < 4 for n in neighbors for c in n)


class TestMapElitesArchive:
    """Steps 41-44: MAP-Elites insertion, incumbents, selection (acceptance 19-20)."""

    def _archive(self) -> MapElitesArchive:
        return MapElitesArchive(default_calibration())

    def test_infeasible_candidates_never_inserted(self):
        # Acceptance test 19: MAP-Elites inserts only feasible candidates.
        archive = self._archive()
        candidate = make_candidate()
        update = archive.insert(candidate, _record(0.9, feasible=False), _descriptor())
        assert not update.accepted
        assert "infeasible" in update.reason
        assert archive.rejections_summary()["infeasible"] == 1

    def test_better_incumbent_preserved(self):
        # Acceptance test 20: archive update preserves a better incumbent.
        archive = self._archive()
        strong = make_candidate({"cpu_budget_share": 0.95})
        weak = make_candidate({"cpu_budget_share": 0.85})
        descriptor = _descriptor()
        first = archive.insert(strong, _record(0.9), descriptor)
        assert first.accepted
        rejected = archive.insert(weak, _record(0.5), descriptor)
        assert not rejected.accepted
        assert "does not improve incumbent" in rejected.reason
        # A better incumbent replaces the old one (distinct candidate, same cell).
        even_better = make_candidate({"memory_budget_share": 0.75})
        replacement = archive.insert(even_better, _record(0.95), descriptor)
        assert replacement.accepted
        assert replacement.replaced_incumbent == strong.candidate_id
        assert archive.turnover_rate() > 0.0

    def test_duplicate_hashes_deduplicated(self):
        archive = self._archive()
        candidate = make_candidate()
        descriptor = _descriptor()
        archive.insert(candidate, _record(0.9), descriptor)
        update = archive.insert(candidate, _record(0.95), descriptor)
        assert not update.accepted
        assert "duplicate" in update.reason

    def test_missing_descriptor_rejected_explicitly(self):
        archive = self._archive()
        candidate = make_candidate()
        descriptor = BehaviorDescriptor(
            {"speed": None, "reasoning_depth": None, "resource_tier": None,
             "reliability": None, "module_cooperation": None},
            "calibration-v1",
        )
        update = archive.insert(candidate, _record(0.9), descriptor)
        assert not update.accepted
        assert "descriptor rejected" in update.reason

    def test_parent_selection_returns_archive_members(self):
        # Step 42: documented parent-selection strategy.
        archive = self._archive()
        descriptor = _descriptor(speed=250.0)
        descriptor_two = _descriptor(speed=420.0)
        archive.insert(make_candidate(), _record(0.8), descriptor)
        archive.insert(make_candidate({"cpu_budget_share": 0.7}), _record(0.75), descriptor_two)
        rng = SeededRandom(3)
        for _ in range(20):
            parents = archive.select_parents(rng, count=2)
            assert len(parents) <= 2
            for parent in parents:
                assert archive.entry(parent.cell_id) is not None

    def test_handoff_returns_sorted_incumbents(self):
        archive = self._archive()
        archive.insert(make_candidate(), _record(0.7), _descriptor(speed=200.0))
        archive.insert(
            make_candidate({"cpu_budget_share": 0.7}), _record(0.9),
            _descriptor(speed=400.0),
        )
        handoff = archive.handoff()
        qualities = [entry.quality for entry in handoff]
        assert qualities == sorted(qualities, reverse=True)
        assert qualities[0] == pytest.approx(0.9)

    def test_archive_metrics_coverage_and_spread(self):
        # Step 44: coverage, per-cell quality, descriptor spread and turnover.
        archive = self._archive()
        archive.insert(make_candidate(), _record(0.8), _descriptor(speed=200.0))
        archive.insert(
            make_candidate({"cpu_budget_share": 0.7}), _record(0.85),
            _descriptor(speed=420.0),
        )
        metrics = archive_metrics(archive)
        assert 0.0 < metrics.coverage < 1.0
        assert metrics.occupied_cells == 2
        assert metrics.best_quality == pytest.approx(0.85)
        assert metrics.descriptor_spread["speed"] > 0.0


class TestMapElitesSearch:
    """Steps 41-43: archive initialization and offspring insertion."""

    def test_qd_search_runs(self):
        evaluator = make_evaluator(trials_per_task=2)
        archive = MapElitesArchive(default_calibration())
        qd = MapElitesSearch(
            evaluator,
            config=QDSearchConfig(
                parents_per_generation=2, offspring_per_parent=1,
                generation_limit=2, seed=31,
            ),
        )
        seeds = [make_baseline_candidate(), make_candidate()]
        report = qd.run(archive, seeds, run_id="qd-unit")
        assert report.stop_reason in ("generation_limit", "budget_exhausted", "empty_archive")
        assert report.evaluations >= 2
        assert 0.0 <= report.coverage <= 1.0
        assert report.archive_snapshot["occupied_cells"] == archive.occupied()

    def test_qd_search_budget_exhaustion(self):
        from ore_evolution_intelligence.evaluation.evaluator import EvaluationBudget
        evaluator = make_evaluator(trials_per_task=2)
        archive = MapElitesArchive(default_calibration())
        qd = MapElitesSearch(
            evaluator,
            config=QDSearchConfig(
                parents_per_generation=2, offspring_per_parent=1,
                generation_limit=5, seed=32,
            ),
        )
        budget = EvaluationBudget(total_evaluations=4)
        # A feasible seed fills the archive; the tiny budget then exhausts cleanly
        # while generating offspring.
        report = qd.run(
            archive, [make_candidate(), make_baseline_candidate()], budget=budget,
        )
        assert report.stop_reason == "budget_exhausted"
        assert report.evaluations <= 5
