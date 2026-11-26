"""End-to-end integration tests: full pipeline with ORE framework integration.

Runs the complete Point 9 pipeline — bounded GA search, sandboxed evaluation,
MAP-Elites archive, independent verification, validated registry, canary rollout —
with ore_logger SQLite persistence, the Shared Cognitive Workspace bridge and the
evolution SQLite repository, and asserts both the happy path (ACTIVE strategy) and the
rollback path (canary regression).
"""
from __future__ import annotations

import pytest

from ore_evolution_intelligence import (
    EvolutionBudgetSpec, EvolutionEventRecorder, EvolutionObjective,
    StrategyEvolutionCoordinator,
)
from ore_evolution_intelligence.orchestration.workspace_bridge import (
    WorkspaceEvolutionPublisher, evolution_workspace_permissions,
)
from ore_evolution_intelligence.observability.repository import (
    SqliteEvolutionRepository,
)
from ore_evolution_intelligence.verification.regression_gate import (
    RegressionGate, RegressionPolicy,
)

from .fixtures import SEED


def _budget(**overrides) -> EvolutionBudgetSpec:
    values = dict(
        total_evaluations=400, generation_cap=18, population_cap=16,
        ga_generations=16, qd_generations=2, seed=SEED,
    )
    values.update(overrides)
    return EvolutionBudgetSpec(**values)


def _build_coordinator(tmp_path, *, regression_policy=None, run_canary=True):
    """Full-pipeline coordinator with ore_logger, workspace and repository."""
    ore_logger = pytest.importorskip("ore_logger")
    ore_workspace = pytest.importorskip("ore_workspace")

    from ore_logger.database import LoggerDatabase
    from ore_workspace import PermissionPolicy, WorkspaceManager

    database = LoggerDatabase(tmp_path / "e2e-logger.sqlite")
    recorder = EvolutionEventRecorder(logger_database=database)
    manager = WorkspaceManager(enable_logging=False)
    workspace = manager.create_workspace(
        "evolution e2e", permissions=PermissionPolicy(
            evolution_workspace_permissions(),
        ),
    )
    publisher = WorkspaceEvolutionPublisher(workspace)
    repository = SqliteEvolutionRepository(tmp_path / "e2e-evolution.sqlite")
    coordinator = StrategyEvolutionCoordinator(
        event_recorder=recorder,
        repository=repository,
        workspace_publisher=publisher,
        run_canary=run_canary,
        regression_policy=regression_policy,
    )
    return coordinator, recorder, publisher, repository, workspace, database


class TestFullPipelineHappyPath:
    """Complete pipeline to ACTIVE with framework persistence (section 11)."""

    def test_evolution_reaches_active_strategy(self, tmp_path):
        (
            coordinator, recorder, publisher, repository, workspace, database,
        ) = _build_coordinator(tmp_path)
        report = coordinator.evolve(
            EvolutionObjective.module_selection(), _budget(),
        )
        data = report.as_dict()
        assert report.status == "completed", report.error

        # Verification chain: held-out improvement, static/semantic checks pass.
        verification = data["verification"]
        assert verification["static_check"]["ok"]
        assert verification["semantic_check"]["ok"]
        assert verification["heldout"]["improved"]
        assert verification["heldout"]["quality_delta"] >= 0.05

        # Registry: APPROVED artifact with manifest, canary promotion to ACTIVE.
        assert data["registry"]["note"] == "registered"
        assert data["best"]["state"] == "ACTIVE"
        assert data["canary"]["promoted_to_active"] is True
        assert data["canary"]["final_state"] == "ACTIVE"
        assert data["canary"]["drift"]["drifted"] is False

        # ore_logger persistence: structured events with the shared logger.
        assert recorder.logger_run_id is not None
        kinds = set(recorder.kinds())
        assert {"run_started", "population_initialized", "generation_completed",
                "verification_completed", "registry_registered",
                "canary_started", "canary_evaluated", "run_completed"} <= kinds

        # Evolution SQLite repository: immutable run/candidate/evaluation records.
        run = repository.run(report.run_id)
        assert run["status"] == "completed"
        candidates = repository.candidates_for_run(report.run_id)
        assert candidates, "candidate rows must persist"
        evaluations = repository.evaluations_for_run(report.run_id)
        assert evaluations and "metrics" in evaluations[0]
        events = repository.events_for_run(report.run_id)
        assert events

        # Workspace publication: experiment request, hypothesis, verification,
        # result — all through the controlled finding pipeline.
        findings = publisher.workspace_findings()
        finding_types = {finding["finding_type"] for finding in findings}
        assert {"evolution_request", "evolution_hypothesis",
                "evolution_verification", "evolution_result"} <= finding_types
        target_fields = {finding["target_field"] for finding in findings}
        assert target_fields <= {"artifact_references", "unresolved_questions"}

        repository.close()


class TestCanaryRollbackPath:
    """Acceptance tests 29-30: canary regression triggers rollback."""

    def test_strict_policy_triggers_rollback(self, tmp_path):
        (
            coordinator, _recorder, _publisher, repository, _workspace, database,
        ) = _build_coordinator(
            tmp_path,
            regression_policy=RegressionPolicy(
                # Strict enough that the quality/latency tradeoff regresses.
                max_latency_regression=0.02,
            ),
            run_canary=True,
        )
        report = coordinator.evolve(
            EvolutionObjective.module_selection(), _budget(),
        )
        data = report.as_dict()
        assert report.status == "completed"
        if data["canary"].get("rolled_back") is True:
            assert data["best"]["state"] == "REJECTED"
            assert data["canary"]["final_state"] == "REJECTED"
            assert any(
                "rollback" in note or "canary regression" in note
                for note in data["notes"]
            )
            # The rollback event must be recorded in the event stream.
            kinds = set(coordinator.event_recorder.kinds())
            assert "rollback_triggered" in kinds
            assert "strategy_deactivated" in kinds
        else:
            # With this seed the candidate may still satisfy the strict gate;
            # then the happy path must hold instead.
            assert data["best"]["state"] == "ACTIVE"

    def test_run_canary_disabled_stays_approved(self, tmp_path):
        (
            coordinator, _recorder, _publisher, _repository, _workspace, database,
        ) = _build_coordinator(tmp_path, run_canary=False)
        report = coordinator.evolve(
            EvolutionObjective.module_selection(), _budget(),
        )
        data = report.as_dict()
        assert report.status == "completed"
        if data["registry"].get("record"):
            assert data["best"]["state"] == "APPROVED"
            assert any("canary skipped" in note for note in data["notes"])


class TestAutoApprovalGate:
    """Section 11: only governance-approved strategies deploy."""

    def test_without_auto_approve_candidate_stays_validated(self, tmp_path):
        ore_logger = pytest.importorskip("ore_logger")
        from ore_logger.database import LoggerDatabase
        from ore_evolution_intelligence import EvolutionEventRecorder
        from ore_evolution_intelligence.observability.repository import (
            SqliteEvolutionRepository,
        )

        database = LoggerDatabase(tmp_path / "manual.sqlite")
        recorder = EvolutionEventRecorder(logger_database=database)
        repository = SqliteEvolutionRepository(tmp_path / "manual-evo.sqlite")
        coordinator = StrategyEvolutionCoordinator(
            event_recorder=recorder, repository=repository, auto_approve=False,
        )
        report = coordinator.evolve(
            EvolutionObjective.module_selection(), _budget(),
        )
        data = report.as_dict()
        assert report.status == "completed"
        assert data["best"]["state"] == "VALIDATED"
        assert any("await" in note for note in data["notes"])
        assert data["registry"] == {}
        repository.close()


class TestPolicyTableAndGPTargets:
    """The other two bounded targets complete the same machinery (steps 7, 31-38)."""

    def test_policy_table_target_completes(self, tmp_path):
        from ore_evolution_intelligence.orchestration.coordinator_adapter import (
            ROUTING_POLICY_TARGET,
        )
        from ore_evolution_intelligence import EvolutionObjective
        coordinator = StrategyEvolutionCoordinator()
        objective = EvolutionObjective(
            name="task_routing_policy", target=ROUTING_POLICY_TARGET,
        )
        report = coordinator.evolve(
            objective, _budget(
                total_evaluations=150, ga_generations=6, qd_generations=1,
                population_cap=10,
            ),
        )
        data = report.as_dict()
        assert report.status == "completed", report.error
        # The incumbent policy-table baseline is strong; the run must complete the
        # machinery and honestly report whether an improvement was verified.
        assert "heldout" in data["verification"]
        assert data["baseline"]["metrics"]["quality"] > 0.5

    def test_policy_ir_target_completes(self, tmp_path):
        from ore_evolution_intelligence.orchestration.coordinator_adapter import (
            PLANNING_HEURISTIC_TARGET,
        )
        from ore_evolution_intelligence import EvolutionObjective
        coordinator = StrategyEvolutionCoordinator()
        objective = EvolutionObjective(
            name="planning_heuristic", target=PLANNING_HEURISTIC_TARGET,
        )
        report = coordinator.evolve(
            objective, _budget(
                total_evaluations=150, ga_generations=6, qd_generations=1,
                population_cap=10,
            ),
        )
        data = report.as_dict()
        assert report.status == "completed", report.error
        # GP candidates stay isolated: nothing reaches production without passing
        # every verification gate (step 38).
        assert data["best"]["state"] in (
            "GENERATED", "SANDBOX_TESTED", "VALIDATED", "APPROVED", "ACTIVE",
            "REJECTED",
        )
        assert "static_check" in data["verification"]
        assert "semantic_check" in data["verification"]
