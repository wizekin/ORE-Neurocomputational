"""Benchmark, ablation, and policy-learning tests (Steps 29-31, acceptances 13-14)."""
from __future__ import annotations

import pytest

from ore_blackboard import (
    BLACKBOARD_OPERATION_IDS,
    ABLATION_MECHANISMS,
    BlackboardCoordinator,
    BlackboardPolicy,
    BlackboardTaskRequest,
    SequentialBaselineRunner,
    evaluation_operation_records,
    paired_bootstrap,
    run_ablation,
    run_benchmark,
)
from ore_blackboard.config import AblationSwitches
from ore_blackboard.policy_learning import (
    PolicyProposalBuilder,
    PromotionGateEvaluator,
    RunMeasurementRecorder,
)
from ore_blackboard.resources import BlackboardResourceCoordinator
from test_usg_lexical_adapter import _bundle


def _requests(count: int) -> list[BlackboardTaskRequest]:
    """Build a small representative task set for the benchmark harness."""
    tasks = [
        BlackboardTaskRequest(
            input="The captain saw the bank.", goal="understand_sentence",
            task_id=f"bench-{index}", usg_bundle=_bundle(),
            memory_store=(
                {"memory_id": "m1", "text": "The fisherman moored the boat.",
                 "keywords": ("river", "land", "water")},
            ),
        )
        for index in range(count)
    ]
    tasks.append(BlackboardTaskRequest(
        input="The boy opened the door.", goal="understand_sentence",
        task_id="bench-plain",
    ))
    return tasks


class TestBenchmarkHarness:
    """Step 29: blackboard coordination versus a sequential baseline."""

    def test_benchmark_reports_all_required_metrics(self):
        """SR, UAR, latency percentiles, conflicts, and provenance are reported."""
        report, blackboard, baseline = run_benchmark(
            lambda: BlackboardCoordinator(), _requests(2), iterations=200,
        )
        assert report.task_count >= 2
        for summary in (report.blackboard, report.baseline):
            for metric in (
                "success_rate", "mean_goal_score", "mean_module_calls",
                "unnecessary_activation_rate", "mean_latency_ms", "median_latency_ms",
                "p95_latency_ms", "mean_conflicts_preserved", "recovery_rate",
                "mean_provenance_completeness", "mean_usg_contribution",
            ):
                assert metric in summary
                assert 0.0 <= float(summary[metric]) or metric.endswith("latency_ms")
        assert len(blackboard) == len(baseline) == report.task_count

    def test_blackboard_matches_baseline_goal_completion(self):
        """Both paths reach the same mandatory coverage on the same inputs."""
        report, blackboard, baseline = run_benchmark(
            lambda: BlackboardCoordinator(), _requests(1), iterations=100,
        )
        blackboard_score = report.blackboard["mean_goal_score"]
        baseline_score = report.baseline["mean_goal_score"]
        assert blackboard_score == pytest.approx(baseline_score)
        assert report.blackboard["success_rate"] == report.baseline["success_rate"]

    def test_blackboard_reduces_unnecessary_activations(self):
        """Selective activation produces fewer unnecessary module calls."""
        report, _blackboard, _baseline = run_benchmark(
            lambda: BlackboardCoordinator(), _requests(2), iterations=100,
        )
        # The baseline runs every module unconditionally: abstentions and other
        # no-output calls are unnecessary activations there.
        assert report.blackboard["unnecessary_activation_rate"] < (
            report.baseline["unnecessary_activation_rate"]
        )

    def test_paired_bootstrap_returns_mean_and_interval(self):
        """Paired differences come with a bootstrap confidence interval."""
        mean, low, high = paired_bootstrap(
            [0.9, 0.8, 0.7], [0.7, 0.6, 0.5], metric_name="score", iterations=200,
        )
        assert mean == pytest.approx(0.2)
        assert low <= mean <= high

    def test_benchmark_notes_disclaim_single_run_claims(self):
        """The report explicitly disclaims improvement claims from one run."""
        report, _blackboard, _baseline = run_benchmark(
            lambda: BlackboardCoordinator(), _requests(1), iterations=50,
        )
        assert any("never establishes an improvement" in note for note in report.notes)


class TestAblationHarness:
    """Step 30: disabling one integration mechanism at a time shows its value."""

    def test_ablation_covers_every_documented_mechanism(self):
        """All seven integration mechanisms are ablated one at a time."""
        report = run_ablation(
            lambda policy: BlackboardCoordinator(policy=policy),
            BlackboardTaskRequest(
                input="The captain saw the bank.", goal="understand_sentence",
                task_id="abl-coverage", usg_bundle=_bundle(),
            ),
        )
        assert set(report.ablations) == set(ABLATION_MECHANISMS)
        assert set(report.full_configuration) == set(report.metric_names)

    def test_disabling_selective_subscriptions_increases_module_calls(self):
        """Broadcast activation multiplies module calls (Step 23's value)."""
        report = run_ablation(
            lambda policy: BlackboardCoordinator(policy=policy),
            BlackboardTaskRequest(
                input="The captain saw the bank.", goal="understand_sentence",
                task_id="abl-subscriptions", usg_bundle=_bundle(),
            ),
        )
        ablated = report.ablations["selective_subscriptions"]
        assert ablated["module_calls"] > report.full_configuration["module_calls"]

    def test_disabling_usg_retrieval_removes_imported_knowledge(self):
        """Without USG retrieval, polysemy, conflicts, and provenance disappear."""
        report = run_ablation(
            lambda policy: BlackboardCoordinator(policy=policy),
            BlackboardTaskRequest(
                input="The captain saw the bank.", goal="understand_sentence",
                task_id="abl-usg", usg_bundle=_bundle(),
            ),
        )
        ablated = report.ablations["usg_retrieval"]
        assert ablated["usg_contribution"] == 0.0
        assert ablated["conflict_count"] < report.full_configuration["conflict_count"]

    def test_disabling_dependency_deferral_destroys_provenance(self):
        """Dropping dependency declaration breaks trace completeness."""
        report = run_ablation(
            lambda policy: BlackboardCoordinator(policy=policy),
            BlackboardTaskRequest(
                input="The captain saw the bank.", goal="understand_sentence",
                task_id="abl-deferral", usg_bundle=_bundle(),
            ),
        )
        ablated = report.ablations["dependency_deferral"]
        assert ablated["provenance_completeness"] < (
            report.full_configuration["provenance_completeness"]
        )

    def test_contribution_table_reports_deltas_per_mechanism(self):
        """Each ablation's contribution is recorded as full minus ablated."""
        report = run_ablation(
            lambda policy: BlackboardCoordinator(policy=policy),
            BlackboardTaskRequest(
                input="The captain saw the bank.", goal="understand_sentence",
                task_id="abl-contrib", usg_bundle=_bundle(),
            ),
        )
        payload = report.to_dict()
        assert "contributions" in payload
        for mechanism in ABLATION_MECHANISMS:
            assert mechanism in payload["contributions"]

    def test_evaluation_records_cover_steps_twenty_nine_and_thirty(self):
        """The evaluation harness emits the BBA-029 and BBA-030 records."""
        records = evaluation_operation_records()
        assert {item["operation_id"] for item in records} == {"BBA-029", "BBA-030"}
        combined = set(BLACKBOARD_OPERATION_IDS) - {"BBA-026"}
        combined |= {item["operation_id"] for item in records}
        assert len(combined) == len(BLACKBOARD_OPERATION_IDS) - 1


class TestSequentialBaseline:
    """The sequential baseline must be a fair, fixed-order comparison path."""

    def test_baseline_runs_each_module_once_in_dependency_order(self):
        """Every specialist executes exactly once with accumulated state."""
        coordinator = BlackboardCoordinator()
        baseline = SequentialBaselineRunner(coordinator)
        metrics = baseline.run(BlackboardTaskRequest(
            input="The boy opened the door.", goal="understand_sentence",
            task_id="baseline-once",
        ))
        assert metrics.path == "sequential_baseline"
        assert metrics.module_calls == len(coordinator.specialists)
        assert metrics.success is True
        assert metrics.provenance_completeness == pytest.approx(1.0)

    def test_baseline_preserves_usg_conflicts_without_coordination(self):
        """The baseline still preserves USG polysemy conflicts through the
        shared validation path, but without ranking or recovery machinery."""
        coordinator = BlackboardCoordinator()
        baseline = SequentialBaselineRunner(coordinator)
        metrics = baseline.run(BlackboardTaskRequest(
            input="The captain saw the bank.", goal="understand_sentence",
            task_id="baseline-usg", usg_bundle=_bundle(),
        ))
        assert metrics.conflict_count >= 2
        assert metrics.usg_contribution >= 3


class TestPolicyLearningSeparation:
    """Step 31: learned updates stay outside ordinary inference."""

    def test_recording_measurements_does_not_change_the_answer(self):
        """Runs with identical inputs produce identical accepted answers whether
        or not measurement transactions were recorded in between."""
        policy = BlackboardPolicy(enable_resource_history=True)
        coordinator = BlackboardCoordinator(policy=policy)
        first = coordinator.run(BlackboardTaskRequest(
            input="The boy opened the door.", goal="understand_sentence",
            task_id="learn-1",
        ))
        # The completed run's measurement transaction is recorded separately and
        # never mutates the workspace answer already returned to the caller.
        recorder = RunMeasurementRecorder(BlackboardResourceCoordinator(
            policy, task_family="sentence_understanding",
        ))
        try:
            recorder.record(
                "learn-1",
                measurements=[{"module_id": "m", "latency_ms": 5.0}],
                outcomes=[{"module_id": "m", "success": True}],
            )
        finally:
            recorder.resource_coordinator.close()
        second = BlackboardCoordinator(policy=policy).run(BlackboardTaskRequest(
            input="The boy opened the door.", goal="understand_sentence",
            task_id="learn-2",
        ))
        assert first.status == second.status
        assert first.goal_score == pytest.approx(second.goal_score)
        assert first.accepted_finding_count == second.accepted_finding_count

    def test_run_measurement_recorder_writes_separate_transactions(self):
        """Measurements and outcomes are recorded as explicit transactions."""
        policy = BlackboardPolicy()
        resource = BlackboardResourceCoordinator(policy, task_family="learn_test")
        recorder = RunMeasurementRecorder(resource)
        try:
            transaction = recorder.record(
                "task-1",
                measurements=[{"module_id": "m", "latency_ms": 12.5}],
                outcomes=[{"module_id": "m", "success": True}],
            )
            assert transaction.inference_answer_unchanged is True
            assert transaction.task_id == "task-1"
            assert len(recorder.transactions) == 1
            assert recorder.transactions[0].measurements[0]["latency_ms"] == 12.5
        finally:
            resource.close()

    def test_policy_proposal_is_never_applied_automatically(self):
        """Proposals carry provenance and remain unapplied until gates pass."""
        builder = PolicyProposalBuilder("sentence_understanding")
        proposal = builder.propose(
            activation_statistics={
                "module_a": {
                    "activations": 2, "suppressed_computations": 8,
                    "activation_threshold": 0.5,
                },
            },
            measurement_provenance={"runs": 1, "source": "test"},
        )
        assert proposal.status == "proposed"
        assert proposal.provenance["origin"] == "ore_blackboard.policy_learning"
        assert "not applied automatically" in proposal.provenance["note"]
        assert proposal.proposed_changes["module_a"]["activation_threshold"] > 0.5

    def test_promotion_requires_every_gate_to_pass(self):
        """Promote = and_j Gate_j; a failing regression gate blocks promotion."""
        def passing_regression(proposal):
            return {"goal_score": 0.1, "provenance_completeness": 0.2}

        def failing_regression(proposal):
            return {"goal_score": -0.2}

        evaluator = PromotionGateEvaluator(regression_runner=passing_regression)
        proposal = PolicyProposalBuilder("family").propose(
            activation_statistics={
                "m": {"activations": 4, "suppressed_computations": 3,
                      "activation_threshold": 0.5},
            },
            measurement_provenance={},
        )
        decision = evaluator.evaluate(proposal)
        assert decision.promoted is True

        failing = PromotionGateEvaluator(regression_runner=failing_regression)
        rejected = failing.evaluate(proposal)
        assert rejected.promoted is False
        assert rejected.gate_results["regression"]["passed"] is False

    def test_safety_gate_rejects_unbounded_thresholds(self):
        """Proposed thresholds outside [0, 1] fail the safety gate."""
        from ore_blackboard.policy_learning import PolicyProposal

        proposal = PolicyProposal(
            proposal_id="p", task_family="family",
            proposed_changes={"m": {"activation_threshold": 1.5}},
            provenance={"origin": "test"},
        )
        evaluator = PromotionGateEvaluator(
            regression_runner=lambda proposal: {"goal_score": 0.1},
        )
        decision = evaluator.evaluate(proposal)
        assert decision.promoted is False
        assert decision.gate_results["safety"]["passed"] is False


class TestAblationSwitchesComposition:
    """Ablation switches compose with the coordinator policy."""

    def test_disabled_mechanism_names_are_reported(self):
        """Disabled switch names are reported by the policy object."""
        switches = AblationSwitches(
            selective_subscriptions=False, bounded_parallelism=False,
        )
        assert switches.disabled_names() == (
            "selective_subscriptions", "bounded_parallelism",
        )
