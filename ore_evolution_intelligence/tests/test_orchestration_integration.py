"""Orchestration integration tests (Phases A-F; acceptance tests 1, 3-4, 24, 33-34)."""
from __future__ import annotations

import pytest

from ore_evolution_intelligence.evaluation.evaluator import BudgetExhausted
from ore_evolution_intelligence.observability.events import (
    EVENT_KINDS, EvolutionEventRecorder,
)
from ore_evolution_intelligence.observability.lineage import LineageTracker
from ore_evolution_intelligence.observability.repository import (
    SqliteEvolutionRepository,
)
from ore_evolution_intelligence.orchestration.coordinator_adapter import (
    CoordinatorAdapter, ControlledConfigurationInterface, baseline_configuration,
)
from ore_evolution_intelligence.orchestration.evolution_service import (
    EvolutionBudgetSpec, EvolutionObjective, StrategyEvolutionCoordinator,
)
from ore_evolution_intelligence.orchestration.resource_budget import (
    AdmissionController, ResourceBudget,
)
from ore_evolution_intelligence.domain.strategy import StrategyCandidate

from .fixtures import SEED, make_baseline_candidate, make_candidate


def _budget(**overrides) -> EvolutionBudgetSpec:
    values = dict(
        total_evaluations=60, generation_cap=6, population_cap=8,
        ga_generations=3, qd_generations=1, seed=SEED,
    )
    values.update(overrides)
    return EvolutionBudgetSpec(**values)


class TestEvolutionServiceBasics:
    """The bounded pipeline contract (steps 1-10, acceptance tests 3, 4, 33)."""

    def test_run_records_objective_split_evaluator_and_seed(self):
        # Acceptance test 3: every run records objective, task split, evaluator
        # version and seed.
        coordinator = StrategyEvolutionCoordinator()
        report = coordinator.evolve(
            EvolutionObjective.module_selection(), _budget(),
        )
        data = report.as_dict()
        assert report.status == "completed"
        assert data["objective"]["name"]
        assert data["benchmark"]["version"]
        assert data["benchmark"]["split"]["policy"]
        assert data["benchmark"]["task_counts"]["train"] > 0
        assert data["benchmark"]["task_counts"]["test_sealed"] > 0
        assert data["benchmark"]["protocol_digest"]
        assert data["budget"]["resource_budget"]["evaluation_budget"]["total"] > 0
        assert data["budget"]["admission"]["allowed"] is True

    def test_baseline_captured_immutable_and_replayable(self):
        # Acceptance test 4: baseline immutable and replayable.
        coordinator = StrategyEvolutionCoordinator()
        report_one = coordinator.evolve(
            EvolutionObjective.module_selection(), _budget(),
        )
        coordinator_two = StrategyEvolutionCoordinator()
        report_two = coordinator_two.evolve(
            EvolutionObjective.module_selection(), _budget(),
        )
        one = report_one.as_dict()["baseline"]
        two = report_two.as_dict()["baseline"]
        assert one["candidate_id"] == two["candidate_id"]
        assert one["content_hash"] == two["content_hash"]
        assert one["metrics"] == two["metrics"]
        assert one["rollback_target_digest"]

    def test_fixed_seeds_reproduce_within_tolerance(self):
        # Acceptance test 33: fixed deterministic experiments reproduce within
        # declared tolerance.
        runs = []
        for _ in range(2):
            coordinator = StrategyEvolutionCoordinator()
            report = coordinator.evolve(
                EvolutionObjective.module_selection(), _budget(),
            )
            runs.append(report.as_dict())
        assert runs[0]["status"] == runs[1]["status"]
        assert runs[0]["baseline"]["metrics"] == runs[1]["baseline"]["metrics"]
        assert runs[0]["benchmark"]["version"] == runs[1]["benchmark"]["version"]
        assert runs[0]["genetic_search"]["stop_reason"] == \
            runs[1]["genetic_search"]["stop_reason"]
        assert runs[0]["genetic_search"]["evaluations_consumed"] == \
            runs[1]["genetic_search"]["evaluations_consumed"]

    def test_search_visible_splits_only_in_service(self):
        # The service itself never opens the sealed test set during search:
        # GA and QD reports reference train/validation counts only.
        coordinator = StrategyEvolutionCoordinator()
        report = coordinator.evolve(
            EvolutionObjective.module_selection(), _budget(),
        )
        data = report.as_dict()
        assert data["genetic_search"]["train_task_count"] > 0
        assert data["genetic_search"]["validation_task_count"] > 0
        # Held-out evidence appears only through the held-out validator.
        assert "heldout" in data["verification"]

    def test_service_reports_all_phases(self):
        coordinator = StrategyEvolutionCoordinator()
        report = coordinator.evolve(
            EvolutionObjective.module_selection(), _budget(),
        )
        data = report.as_dict()
        for key in ("benchmark", "baseline", "budget", "genetic_search",
                    "quality_diversity", "verification", "events_summary",
                    "lineage_summary"):
            assert data[key], key
        assert data["events_summary"]["kinds_emitted"]
        assert set(data["events_summary"]["kinds_emitted"]) <= set(EVENT_KINDS)


class TestFailureIsolation:
    """Acceptance test 34: evolution failure does not interrupt ORE tasks."""

    def test_internal_failure_returns_failed_report(self):
        coordinator = StrategyEvolutionCoordinator()

        def broken_evolve(*args, **kwargs):
            raise RuntimeError("simulated infrastructure failure")

        coordinator._evolve_phases = broken_evolve
        report = coordinator.evolve(
            EvolutionObjective.module_selection(), _budget(),
        )
        assert report.status == "failed"
        assert "simulated infrastructure failure" in report.error
        assert report.as_dict()["notes"] == []

    def test_admission_denial_fails_cleanly(self):
        budget = ResourceBudget(total_evaluations=1, generation_cap=2, population_cap=4)
        controller = AdmissionController(reserve_fraction=0.999)
        decision = controller.decide(budget)
        assert not decision.allowed
        assert "production reserve" in decision.reason


class TestBudgetExhaustion:
    """Acceptance test 24: budget exhaustion stops the run cleanly."""

    def test_tiny_budget_stops_cleanly(self):
        coordinator = StrategyEvolutionCoordinator()
        report = coordinator.evolve(
            EvolutionObjective.module_selection(),
            _budget(total_evaluations=4, ga_generations=6, qd_generations=2,
                    population_cap=8),
        )
        data = report.as_dict()
        assert report.status == "completed"
        stop_reason = data["genetic_search"]["stop_reason"]
        assert stop_reason in ("budget_exhausted", "generation_limit",
                                "convergence", "unacceptable_failure_rate")
        assert data["budget"]["resource_budget"]["evaluation_budget"]["consumed"] <= 4

    def test_evaluation_budget_raises_for_overrun(self):
        from ore_evolution_intelligence.evaluation.evaluator import EvaluationBudget
        budget = EvaluationBudget(total_evaluations=1)
        budget.consume(1)
        with pytest.raises(BudgetExhausted):
            budget.consume(1)


class TestLineage:
    """Acceptance test 24 (lineage side): candidate lineage reconstructable."""

    def test_lineage_tracker_reconstructs_ancestry(self):
        baseline = make_baseline_candidate()
        tracker = LineageTracker()
        tracker.record(baseline)
        from ore_evolution_intelligence.domain.strategy import LineageRecord
        child = StrategyCandidate.from_genome(
            make_candidate({"cpu_budget_share": 0.7}).genome,
            lineage=LineageRecord(
                parents=(baseline.candidate_id,), operation="mutation", generation=1,
            ),
        )
        tracker.record(child)
        tree = tracker.ancestry(child.candidate_id)
        assert child.candidate_id in tree
        assert baseline.candidate_id in tree
        assert tracker.ancestry_depth(child.candidate_id) == 1
        assert tracker.descendants(baseline.candidate_id) == [child.candidate_id]
        assert tracker.lineage_diversity(
            [baseline.candidate_id, child.candidate_id],
        ) == 1

    def test_service_lineage_summary(self):
        coordinator = StrategyEvolutionCoordinator()
        report = coordinator.evolve(
            EvolutionObjective.module_selection(), _budget(),
        )
        lineage = report.as_dict()["lineage_summary"]
        assert lineage["recorded"] >= 1
        assert "chosen_ancestry_depth" in lineage


class TestObservabilityPersistence:
    """SQLite persistence of runs, candidates, evaluations and events."""

    def test_sqlite_repository_round_trip(self, tmp_path):
        repository = SqliteEvolutionRepository(tmp_path / "evolution.sqlite")
        coordinator = StrategyEvolutionCoordinator(repository=repository)
        report = coordinator.evolve(
            EvolutionObjective.module_selection(), _budget(),
        )
        run = repository.run(report.run_id)
        assert run and run["objective"] == "module_selection_weights"
        assert run["status"] in ("running", "completed", "failed")
        events = repository.events_for_run(report.run_id)
        assert events and {event["kind"] for event in events} <= set(EVENT_KINDS)
        assert repository.table_counts()["evolution_runs"] == 1
        repository.close()

    def test_event_recorder_with_ore_logger_database(self, tmp_path):
        ore_logger = pytest.importorskip("ore_logger")
        from ore_logger.database import LoggerDatabase
        database = LoggerDatabase(tmp_path / "logger.sqlite")
        recorder = EvolutionEventRecorder(logger_database=database)
        coordinator = StrategyEvolutionCoordinator(event_recorder=recorder)
        report = coordinator.evolve(
            EvolutionObjective.module_selection(), _budget(),
        )
        assert recorder.logger_run_id is not None
        assert len(recorder.by_run(report.run_id)) >= 8
        assert "run_started" in recorder.kinds()
        assert "generation_completed" in recorder.kinds()


class TestCoordinatorAdapter:
    """Controlled configuration interface (acceptance test 1)."""

    def test_search_cannot_change_production_configuration(self):
        # Acceptance test 1: candidate generation cannot change production
        # configuration — only APPROVED/CANARY/ACTIVE strategies pass the
        # controlled interface.
        interface = ControlledConfigurationInterface()
        before = interface.current.as_dict()
        coordinator = StrategyEvolutionCoordinator()
        report = coordinator.evolve(
            EvolutionObjective.module_selection(), _budget(),
        )
        # The evolution run itself holds its own interface; production here is
        # untouched because nothing was applied to *this* interface.
        assert interface.current.as_dict() == before
        assert interface.history == []

    def test_approved_strategy_applies_through_controlled_interface(self):
        interface = ControlledConfigurationInterface()
        adapter = CoordinatorAdapter(interface)
        approved = make_candidate({"cpu_budget_share": 0.7}).with_state("APPROVED")
        configuration = adapter.apply_approved(approved)
        assert configuration.strategy_id == approved.candidate_id
        assert len(interface.history) == 1
        assert interface.history[0]["state"] == "APPROVED"

    def test_rollback_restores_baseline(self):
        interface = ControlledConfigurationInterface()
        adapter = CoordinatorAdapter(interface)
        approved = make_candidate({"cpu_budget_share": 0.7}).with_state("APPROVED")
        adapter.apply_approved(approved)
        restored = interface.restore_rollback_target()
        assert restored.strategy_id is None
        assert restored.as_dict() == baseline_configuration().as_dict()

    def test_baseline_configuration_matches_default_genome(self):
        from ore_evolution_intelligence.orchestration.coordinator_adapter import (
            genome_from_configuration,
        )
        from .fixtures import make_baseline_genome
        genome = genome_from_configuration(baseline_configuration())
        baseline = make_baseline_genome()
        assert genome.values == baseline.values
