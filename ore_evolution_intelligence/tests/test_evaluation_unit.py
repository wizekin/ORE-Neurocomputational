"""Evaluation-layer unit tests (Phase B, steps 9-17; acceptance tests 1-14)."""
from __future__ import annotations

import pytest

from ore_evolution_intelligence.domain.actions import (
    PROHIBITED_ACTIONS, ActionCall, validate_action,
)
from ore_evolution_intelligence.domain.genome import content_digest
from ore_evolution_intelligence.evaluation.benchmark_manifest import (
    build_task_suite, split_tasks,
)
from ore_evolution_intelligence.evaluation.evaluator import (
    BudgetExhausted, EvaluationBudget, SeededRandom, task_stream_seed,
)
from ore_evolution_intelligence.evaluation.sandbox import (
    GRANTED_CAPABILITIES, SandboxLimits, SandboxedStrategyExecutor,
)
from ore_evolution_intelligence.evaluation.task_simulators import (
    CoordinatorSimulator,
)

from .fixtures import (
    SEED, make_aggressive_genome, make_baseline_candidate, make_baseline_genome,
    make_benchmark, make_candidate, make_evaluator,
)


class TestBenchmarkManifest:
    """Steps 4-5, 9-10: versioned protocol, sealed splits (acceptance 3, 12, 13)."""

    def test_split_is_disjoint_and_declared(self):
        benchmark = make_benchmark()
        train_ids = {t.task_id for t in benchmark.training_tasks()}
        validation_ids = {t.task_id for t in benchmark.validation_tasks()}
        test_ids = {t.task_id for t in benchmark.open_sealed_test_set()}
        # Acceptance test 12: training and held-out tasks follow the declared split.
        assert train_ids and validation_ids and test_ids
        assert not (train_ids & validation_ids)
        assert not (train_ids & test_ids)
        assert not (validation_ids & test_ids)
        manifest = benchmark.split_manifest()
        assert manifest["policy"].startswith("by_source")

    def test_search_facing_api_hides_test_tasks(self):
        benchmark = make_benchmark()
        # Acceptance test 13: only digests are visible before the seal is opened.
        digests = benchmark.test_task_digests()
        assert digests and all(len(d) == 64 for d in digests)

    def test_task_suite_covers_representative_structure(self):
        tasks = build_task_suite(tasks_per_family=10)
        families = {task.family for task in tasks}
        assert len(families) == 6
        assert any(task.safety_critical for task in tasks)  # rare safety-critical
        assert any(task.is_failure_case for task in tasks)  # failure cases
        assert any(task.resource_state == "pressured" for task in tasks)

    def test_changed_scoring_creates_new_benchmark_version(self):
        # Acceptance test 3: a changed evaluator creates a new benchmark version.
        from ore_evolution_intelligence.domain.evaluation_record import HardConstraints
        from ore_evolution_intelligence.evaluation.benchmark_manifest import (
            ScoringConfig, build_benchmark,
        )
        from ore_evolution_intelligence.domain.evaluation_record import ScalarFitness
        base = make_benchmark()
        changed = build_benchmark(
            benchmark_id=base.benchmark_id, seed=base.seed,
            scoring=ScoringConfig(
                fitness_config=ScalarFitness(w_quality=0.5).as_dict(),
                constraints_config=HardConstraints().as_dict(),
            ),
        )
        assert changed.version != base.version

    def test_protocol_digest_covers_environment(self):
        benchmark = make_benchmark()
        assert benchmark.environment["python"]
        assert benchmark.evaluator_version
        assert len(benchmark.protocol_digest()) == 64


class TestSandbox:
    """Steps 11-13, 16: sandboxed execution and budgets (acceptance 2, 7-10)."""

    def _executor(self, **limits) -> SandboxedStrategyExecutor:
        return SandboxedStrategyExecutor(
            simulator=CoordinatorSimulator(), limits=SandboxLimits(**limits),
        )

    def _task(self):
        benchmark = make_benchmark()
        return benchmark.training_tasks()[0]

    def test_granted_capabilities_are_explicit(self):
        # Acceptance test 7: no network or filesystem grants by default.
        executor = self._executor()
        granted = executor.granted_capabilities()
        assert granted == GRANTED_CAPABILITIES
        joined = " ".join(granted)
        assert "net" not in joined and "fs" not in joined and "shell" not in joined

    def test_unknown_action_is_rejected_and_recorded(self):
        # Acceptance test 2: unknown or privileged actions are rejected.
        class SmugglingSimulator:
            def simulate(self, genome, task, rng):
                from ore_evolution_intelligence.evaluation.task_simulators import (
                    SimulationResult,
                )
                return SimulationResult(
                    task_id=task.task_id, actions=("shell_execute", "net_connect"),
                    active_modules=("ore_semantic",), quality=0.9,
                    outcome="SUCCESS", latency_ms=80.0, memory_mb=20.0,
                    cpu_ms=70.0, cost=10.0, safety_violations=0,
                    evidence={"active_modules": 1},
                )

        executor = SandboxedStrategyExecutor(simulator=SmugglingSimulator())
        record, trace = executor.execute(
            make_baseline_genome(), self._task(), SeededRandom(1),
        )
        assert record.outcome == "INVALID"
        assert any("prohibited action attempted" in e for e in trace.exceptions)
        # The vocabulary also rejects construction outright (structural guard).
        with pytest.raises(ValueError):
            ActionCall("net_http_get", {})
        with pytest.raises(ValueError):
            ActionCall("unknown_action", {})
        for name in PROHIBITED_ACTIONS:
            assert validate_action(name, {})

    def test_output_budget_enforced(self):
        # Acceptance test 8: output limits enforced.
        genome = make_baseline_genome()
        task = self._task()
        executor = self._executor(max_output_bytes=8)
        record, trace = executor.execute(genome, task, SeededRandom(1))
        assert record.outcome == "INVALID"
        assert trace.outputs["output_bytes"] > 8

    def test_memory_budget_enforced(self):
        # Acceptance test 8: memory limits enforced.
        genome = make_baseline_genome()
        task = self._task()
        executor = self._executor(max_memory_mb=1.0)
        record, trace = executor.execute(genome, task, SeededRandom(1))
        assert record.outcome == "INVALID"
        assert trace.exceptions or trace.notes

    def test_latency_hard_cap_enforced(self):
        executor = self._executor(max_latency_ms=5.0)
        genome = make_aggressive_genome()
        task = self._task()
        record, _trace = executor.execute(genome, task, SeededRandom(1))
        assert record.outcome == "INVALID"

    def test_nonterminating_policy_ir_is_terminated_and_recorded(self):
        # Acceptance test 9: nonterminating candidate terminated + recorded.
        from ore_evolution_intelligence.domain.actions import ActionCall
        from ore_evolution_intelligence.search.genetic_programming import (
            DEFAULT_GRAMMAR, ActionNode, CombineActionNode, PolicyIRGenome,
        )
        # A combine of two actions needs 3 interpreter steps; a budget of 1
        # forces the nontermination guard to fire and be recorded.
        tree = CombineActionNode(
            ActionNode(ActionCall("activate_module", {"module": "ore_semantic"})),
            ActionNode(ActionCall("activate_module", {"module": "ore_syntax"})),
        )
        genome = PolicyIRGenome(tree, DEFAULT_GRAMMAR, max_steps=1)
        executor = SandboxedStrategyExecutor(simulator=CoordinatorSimulator())
        record, trace = executor.execute(genome, self._task(), SeededRandom(1))
        assert trace.terminated or record.terminated
        assert record.outcome == "INVALID"
        assert record.steps >= 1
        assert any(
            "step budget" in note or "step budget" in exception
            for note in trace.notes + tuple(trace.exceptions)
        )

    def test_malformed_genome_is_invalid(self):
        # Acceptance test 10: candidate output schema validated.
        class BrokenGenome:
            kind = "parameters"

            def validate(self):
                return ["malformed genome: values out of range"]

            def capabilities(self):
                return frozenset()

            def as_dict(self):
                return {}

            def content_payload(self):
                return {}

        executor = SandboxedStrategyExecutor(simulator=CoordinatorSimulator())
        record, trace = executor.execute(BrokenGenome(), self._task(), SeededRandom(1))
        assert record.outcome == "INVALID"
        assert trace.exceptions


class TestEvaluator:
    """Steps 14-17, 20-22: independent evaluation (acceptance 4-6, 11, 14, 30)."""

    def test_evaluator_independent_of_candidates(self):
        # Acceptance test 6: evaluator cannot be modified by candidate code —
        # candidates are data; the evaluator's benchmark digest stays untouched.
        evaluator = make_evaluator()
        digest_before = content_digest(evaluator.benchmark.as_dict())
        candidate = make_candidate()
        evaluator.evaluate_candidate(candidate, split="train")
        assert content_digest(evaluator.benchmark.as_dict()) == digest_before

    def test_multi_seed_trials_record_seeds_and_dispersion(self):
        # Acceptance test 11: stochastic trials record seeds and dispersion.
        evaluator = make_evaluator()
        record = evaluator.evaluate_candidate(
            make_candidate(), split="train", seeds=(11, 22, 33),
        )
        assert record.seeds == (11, 22, 33)
        assert record.metrics.n_trials == len(record.seeds) * len(
            evaluator.benchmark.training_tasks()
        )
        assert record.metrics.quality_std >= 0.0
        assert record.metrics.quality_ci95 >= 0.0

    def test_raw_component_metrics_preserved(self):
        # Acceptance test 14: raw component metrics preserved, never only a scalar.
        evaluator = make_evaluator(trials_per_task=2)
        record = evaluator.evaluate_candidate(make_candidate(), split="train")
        assert len(record.outcomes) == record.metrics.n_trials
        outcome = record.outcomes[0]
        assert outcome.task_id and outcome.seed is not None
        assert outcome.latency_ms > 0 and outcome.cost > 0
        assert "active_modules" in outcome.evidence
        assert "trace" in outcome.evidence  # step 13 structured traces

    def test_safety_violation_blocks_despite_quality(self):
        # Acceptance test 5: hard safety failure makes candidate ineligible.
        evaluator = make_evaluator()
        # Baseline (balanced escalation) triggers safety violations on
        # low-complexity safety-critical tasks.
        record = evaluator.evaluate_candidate(make_baseline_candidate(), split="train")
        if record.metrics.safety_violations > 0:
            assert not record.gate.feasible
            assert any("safety" in r for r in record.gate.reasons)

    def test_search_facing_splits_only(self):
        # Acceptance test 13: search cannot read held-out test labels.
        evaluator = make_evaluator()
        with pytest.raises(ValueError):
            evaluator.evaluate_candidate(make_candidate(), split="heldout_test")
        with pytest.raises(ValueError):
            evaluator.evaluate_candidate(make_candidate(), split="test")

    def test_evaluation_budget_enforced(self):
        budget = EvaluationBudget(total_evaluations=2)
        evaluator = make_evaluator(trials_per_task=1)
        budget.consume(1)
        budget.consume(1)
        with pytest.raises(BudgetExhausted):
            budget.consume(1)

    def test_baseline_is_replayable(self):
        # Acceptance test 4: baseline is immutable and replayable (same seeds).
        evaluator = make_evaluator()
        first = evaluator.evaluate_candidate(
            make_baseline_candidate(), split="train", seeds=(5, 6),
        )
        second = evaluator.evaluate_candidate(
            make_baseline_candidate(), split="train", seeds=(5, 6),
        )
        assert first.metrics.quality == pytest.approx(second.metrics.quality)
        assert first.metrics.as_dict() == second.metrics.as_dict()

    def test_adjudicated_outcomes_come_from_evaluator_credit(self):
        # Acceptance test 30: adjudicated outcomes distinguished from self-reports —
        # quality is the evaluator's verified measurement with noise, not a value the
        # candidate could self-report, and outcome bands are adjudicated evaluator-side.
        evaluator = make_evaluator(trials_per_task=2)
        record = evaluator.evaluate_candidate(make_candidate(), split="train")
        seen_outcomes = set()
        for outcome in record.outcomes:
            seen_outcomes.add(outcome.outcome)
            if outcome.outcome == "SUCCESS":
                assert outcome.quality >= 0.75 - 1e-6
            elif outcome.outcome == "PARTIAL":
                assert 0.35 - 1e-6 <= outcome.quality < 0.75
            elif outcome.outcome == "FAILURE":
                assert outcome.quality < 0.35 + 1e-6
            elif outcome.outcome == "ABSTAIN":
                assert outcome.quality <= 0.1 + 1e-6
            else:  # TIMEOUT / INVALID
                assert outcome.quality == 0.0
        # Seeded noise keeps measured quality varying across trials.
        qualities = {o.quality for o in record.outcomes if o.outcome == "SUCCESS"}
        assert len(qualities) > 1 or not qualities

    def test_hardcoding_detection_flags_constant_quality(self):
        class HackGenome:
            """A genome hardcoding the benchmark: identical quality on every task."""

            kind = "parameters"

            def validate(self):
                return []

            def capabilities(self):
                return frozenset({"activate_module"})

            def as_dict(self):
                return {"kind": self.kind}

            def content_payload(self):
                return {"kind": self.kind}

        class HackSimulator:
            def simulate(self, genome, task, rng):
                from ore_evolution_intelligence.evaluation.task_simulators import (
                    SimulationResult,
                )
                return SimulationResult(
                    task_id=task.task_id, actions=("activate_module",),
                    active_modules=("ore_semantic",), quality=0.95,
                    outcome="SUCCESS", latency_ms=80.0, memory_mb=20.0,
                    cpu_ms=70.0, cost=10.0, safety_violations=0,
                    evidence={"active_modules": 2},
                )

        evaluator = make_evaluator()
        evaluator.sandbox.simulator = HackSimulator()
        record = evaluator.evaluate_candidate(
            make_candidate(), split="train", seeds=(1, 2, 3),
        )
        assert "benchmark hardcoding suspected" in " ".join(record.invalid_reasons)
        assert not record.gate.feasible


class TestSeededRandom:
    def test_task_stream_seed_is_stable(self):
        assert task_stream_seed(1, "reading_01") == task_stream_seed(1, "reading_01")
        assert task_stream_seed(1, "reading_01") != task_stream_seed(2, "reading_01")

    def test_seeded_random_deterministic(self):
        a = SeededRandom(42)
        b = SeededRandom(42)
        assert [a.uniform() for _ in range(5)] == [b.uniform() for _ in range(5)]
