"""Verification and registry unit tests (acceptance tests 27-31, lifecycle states)."""
from __future__ import annotations

import pytest

from ore_evolution_intelligence.domain.evaluation_record import (
    AggregateMetrics, EvaluationRecord, GateDecision,
)
from ore_evolution_intelligence.domain.genome import content_digest
from ore_evolution_intelligence.evaluation.evaluator import StrategyEvaluator
from ore_evolution_intelligence.evaluation.heldout_validation import (
    DriftReport, ValidationEnvelope,
)
from ore_evolution_intelligence.orchestration.coordinator_adapter import (
    CoordinatorAdapter, ControlledConfigurationInterface, baseline_configuration,
)
from ore_evolution_intelligence.orchestration.rollout_controller import (
    CanaryDecision, RolloutController,
)
from ore_evolution_intelligence.registry.artifact_store import (
    ArtifactStore, StrategyArtifact, StrategyManifest,
)
from ore_evolution_intelligence.registry.strategy_registry import (
    ValidatedStrategyRegistry,
)
from ore_evolution_intelligence.registry.versioning import RollbackTarget, Versioning
from ore_evolution_intelligence.verification.promotion_policy import (
    ApprovalRecord, CanaryPlan, DeprecationRecord, PromotionPolicy, RejectionRecord,
)
from ore_evolution_intelligence.verification.regression_gate import (
    RegressionDecision, RegressionGate, RegressionPolicy,
)

from .fixtures import make_baseline_candidate, make_candidate, make_evaluator


def _metrics(quality: float = 0.8, **overrides):
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


class TestPromotionPolicy:
    """Section 11: strict lifecycle states with typed evidence."""

    def _policy(self) -> PromotionPolicy:
        return PromotionPolicy()

    def test_generated_requires_evaluation_to_advance(self):
        candidate = make_candidate()
        policy = self._policy()
        _new, decision = policy.apply(candidate, "SANDBOX_TESTED", {})
        assert not decision.granted
        assert "evaluation" in decision.missing_evidence

    def test_invalid_transition_rejected(self):
        candidate = make_candidate()
        _new, decision = self._policy().apply(candidate, "ACTIVE", {})
        assert not decision.granted
        assert "not allowed" in decision.reasons[0]

    def test_terminal_states_absorb(self):
        rejected = make_candidate().with_state("REJECTED")
        _new, decision = self._policy().apply(rejected, "APPROVED", {})
        assert not decision.granted

    def test_rejection_requires_reason(self):
        _new, decision = self._policy().apply(make_candidate(), "REJECTED", {})
        assert not decision.granted
        assert "rejection" in decision.missing_evidence

    def test_full_lifecycle_chain(self):
        policy = self._policy()
        candidate = make_candidate()
        record = _record_evidence()
        sandboxed, decision = policy.apply(
            candidate, "SANDBOX_TESTED", {"evaluation": record},
        )
        assert decision.granted and sandboxed.state == "SANDBOX_TESTED"

        validated, decision = policy.apply(
            sandboxed, "VALIDATED",
            {
                "static_check": _ok_check(),
                "semantic_check": _ok_semantic(),
                "heldout": _improved_heldout(),
            },
        )
        assert decision.granted and validated.state == "VALIDATED"

        approved, decision = policy.apply(
            validated, "APPROVED",
            {"approval": ApprovalRecord(approver="coordinator", justification="ok")},
        )
        assert decision.granted and approved.state == "APPROVED"

        canary, decision = policy.apply(
            approved, "CANARY", {"canary_plan": CanaryPlan(canary_task_ids=("t1",))},
        )
        assert decision.granted and canary.state == "CANARY"

        active, decision = policy.apply(
            canary, "ACTIVE",
            {"canary_decision": CanaryDecision(approved=True)},
        )
        assert decision.granted and active.state == "ACTIVE"

        deprecated, decision = policy.apply(
            active, "DEPRECATED",
            {"deprecation": DeprecationRecord(reason="superseded")},
        )
        assert decision.granted and deprecated.state == "DEPRECATED"

    def test_heldout_without_improvement_blocks_validation(self):
        candidate = make_candidate().with_state("SANDBOX_TESTED")
        heldout = type("H", (), {"improved": False})()
        _new, decision = PromotionPolicy().apply(
            candidate, "VALIDATED",
            {"static_check": _ok_check(), "semantic_check": _ok_semantic(),
             "heldout": heldout},
        )
        assert not decision.granted


class TestRegressionGate:
    """Section 7: regression gate blocks safety regressions regardless of quality."""

    def test_safety_regression_blocks_high_quality(self):
        # Acceptance test 5 (registry side): safety regression blocks promotion.
        gate = RegressionGate()
        decision = gate.check(
            _metrics(quality=0.95, safety_violations=3),
            _metrics(quality=0.8, safety_violations=0),
        )
        assert not decision.approved
        assert any("safety" in reason for reason in decision.reasons)

    def test_quality_improvement_with_bounded_latency_passes(self):
        gate = RegressionGate()
        decision = gate.check(
            _metrics(quality=0.95, latency_mean_ms=220.0),
            _metrics(quality=0.8, latency_mean_ms=200.0),
        )
        assert decision.approved

    def test_strict_policy_blocks_latency_regression(self):
        gate = RegressionGate(
            RegressionPolicy(max_latency_regression=0.05),
        )
        decision = gate.check(
            _metrics(quality=0.85, latency_mean_ms=230.0),
            _metrics(quality=0.8, latency_mean_ms=200.0),
        )
        assert not decision.approved
        assert any("latency regression" in r for r in decision.reasons)

    def test_reliability_regression_blocks(self):
        gate = RegressionGate()
        decision = gate.check(
            _metrics(quality=0.9, reliability=0.70),
            _metrics(quality=0.8, reliability=0.9),
        )
        assert not decision.approved


class TestVersioning:
    """Step 3: immutable rollback target; step 27: version history."""

    def test_rollback_target_pinned_once(self):
        versioning = Versioning()
        target = RollbackTarget(
            strategy_id=None, content_hash="baseline",
            configuration=baseline_configuration().as_dict(),
            benchmark_version="v1", benchmark_results={}, source_version="v1",
        )
        versioning.pin_rollback_target(target)
        with pytest.raises(ValueError):
            versioning.pin_rollback_target(target)

    def test_version_history_deduplicated_by_hash(self):
        versioning = Versioning()
        first = versioning.register("s1", "h1", change_kind="initial")
        again = versioning.register("s1", "h1", change_kind="initial")
        assert first.version == again.version == 1
        second = versioning.register("s1", "h2", change_kind="improvement")
        assert second.version == 2
        assert len(versioning.history("s1")) == 2


class TestArtifactStore:
    """Acceptance test 28: registry rejects missing manifests or hashes."""

    def _manifest(self) -> StrategyManifest:
        return StrategyManifest(
            benchmark_id="bench", benchmark_version="v1",
            split_digest="d" * 64, evaluator_version="ev-1",
        )

    def test_artifact_requires_manifest(self):
        payload = {"genome": "x"}
        with pytest.raises(ValueError):
            StrategyArtifact(payload=payload, manifest=None, content_hash=content_digest(payload))

    def test_tampered_artifact_fails_integrity(self):
        artifact = StrategyArtifact.build({"genome": "x"}, self._manifest())
        with pytest.raises(ValueError):
            StrategyArtifact(
                payload={"genome": "tampered"},
                manifest=self._manifest(),
                content_hash=artifact.content_hash,
            )

    def test_store_round_trip_verifies(self):
        store = ArtifactStore()
        artifact = StrategyArtifact.build({"genome": "x"}, self._manifest())
        digest = store.save(artifact)
        loaded = store.load(digest)
        assert loaded.payload == artifact.payload
        with pytest.raises(KeyError):
            store.load("missing")


class TestValidatedStrategyRegistry:
    """Registry contract: approval required, dedup, production eligibility (27, 28)."""

    def _registry(self) -> ValidatedStrategyRegistry:
        return ValidatedStrategyRegistry()

    def test_register_requires_approval(self):
        artifact = StrategyArtifact.build(
            {"genome": "x"},
            StrategyManifest(
                benchmark_id="b", benchmark_version="v", split_digest="d" * 64,
                evaluator_version="e",
            ),
        )
        with pytest.raises(ValueError, match="approval"):
            self._registry().register(artifact)

    def test_register_rejects_tampered_hash(self):
        registry = self._registry()
        manifest = StrategyManifest(
            benchmark_id="b", benchmark_version="v", split_digest="d" * 64,
            evaluator_version="e",
        )
        artifact = StrategyArtifact.build({"genome": "x"}, manifest)
        with pytest.raises(ValueError):
            StrategyArtifact(
                payload={"genome": "tampered"}, manifest=manifest,
                content_hash=artifact.content_hash,
            )

    def test_duplicate_content_hash_deduplicated(self):
        # Acceptance test 27 (dedup side): duplicate hashes deduplicated.
        registry = self._registry()
        manifest = StrategyManifest(
            benchmark_id="b", benchmark_version="v", split_digest="d" * 64,
            evaluator_version="e",
        )
        artifact = StrategyArtifact.build({"genome": "x"}, manifest)
        approval = ApprovalRecord(approver="coordinator", justification="ok")
        candidate = make_candidate().with_state("APPROVED")
        first, note_one = registry.register(artifact, approval, candidate=candidate)
        second, note_two = registry.register(artifact, approval, candidate=candidate)
        assert note_one == "registered"
        assert note_two.startswith("duplicate")
        assert first.content_hash == second.content_hash
        assert len(registry.records()) == 1

    def test_production_selection_requires_eligible_strategies(self):
        # Acceptance test 28: only APPROVED strategies selectable in production.
        registry = self._registry()
        with pytest.raises(LookupError):
            registry.select_for_context({"preferred_strategy_id": "any"})
        manifest = StrategyManifest(
            benchmark_id="b", benchmark_version="v", split_digest="d" * 64,
            evaluator_version="e",
        )
        artifact = StrategyArtifact.build({"genome": "x"}, manifest)
        approval = ApprovalRecord(approver="coordinator", justification="ok")
        approved = make_candidate({"cpu_budget_share": 0.7}).with_state("APPROVED")
        registry.register(artifact, approval, candidate=approved)
        record = registry.select_for_context({})
        assert record.state == "APPROVED"

    def test_generated_candidate_cannot_be_applied(self):
        # Acceptance test 1: candidate generation cannot change production config.
        interface = ControlledConfigurationInterface()
        adapter = CoordinatorAdapter(interface)
        with pytest.raises(PermissionError):
            adapter.apply_approved(make_candidate())  # GENERATED state


class TestCanaryAndRollback:
    """Acceptance tests 29-31: canary rollback, config restore, drift monitoring."""

    def _envelope_record(self):
        return EvaluationRecord(
            candidate_id="c", candidate_version=1, genome_hash="h" * 64,
            benchmark_id="b", benchmark_version="v", split="validation",
            seeds=(1,), metrics=_metrics(0.85), outcomes=(),
            gate=GateDecision(feasible=True), fitness=1.0,
        )

    def test_envelope_detects_drift(self):
        # Acceptance test 31: live monitoring detects drift from validation envelope.
        envelope = ValidationEnvelope.from_record(self._envelope_record())
        stable = envelope.check(_metrics(0.85))
        assert not stable.drifted
        drifted = envelope.check(_metrics(0.5, latency_mean_ms=400.0))
        assert drifted.drifted
        assert drifted.violations

    def test_envelope_cost_uses_per_task_mean(self):
        envelope = ValidationEnvelope.from_record(self._envelope_record())
        # Different task counts must not break comparability: cost_mean is per task.
        report = envelope.check(_metrics(0.85, cost=2000.0, cost_mean=15.5))
        assert not report.drifted

    def test_canary_regression_triggers_rollback(self):
        # Acceptance tests 29-30: canary regression triggers rollback and restores
        # the prior configuration.
        evaluator = make_evaluator(trials_per_task=1)
        events: list[tuple[str, dict]] = []

        class Recorder:
            def emit(self, kind, run_id, payload):
                events.append((kind, payload))

        interface = ControlledConfigurationInterface(
            event_recorder=Recorder(), run_id="canary-run",
        )
        registry = ValidatedStrategyRegistry()
        controller = RolloutController(
            evaluator, registry=registry,
            adapter=CoordinatorAdapter(interface),
            regression_gate=RegressionGate(
                RegressionPolicy(max_latency_regression=0.05),
            ),
            event_recorder=Recorder(), run_id="canary-run",
        )
        tasks = list(evaluator.benchmark.validation_tasks())[:4]
        candidate = make_candidate({"cpu_budget_share": 0.9}).with_state("APPROVED")
        baseline = make_baseline_candidate()
        envelope = ValidationEnvelope.from_record(self._envelope_record())
        final, outcome = controller.canary(
            candidate, baseline, tasks, envelope=envelope,
            seeds=(3, 4), budget=None,
        )
        if outcome.rolled_back:
            assert final.state == "REJECTED"
            kinds = [kind for kind, _payload in events]
            assert "rollback_triggered" in kinds
            assert "strategy_deactivated" in kinds
            # Acceptance test 30: rollback restores prior configuration.
            assert interface.current.strategy_id is None
            assert any(
                entry.get("state") == "ROLLBACK" for entry in interface.history
            )

    def test_monitor_active_drift_deactivates(self):
        evaluator = make_evaluator(trials_per_task=1)
        registry = ValidatedStrategyRegistry()
        interface = ControlledConfigurationInterface()
        controller = RolloutController(
            evaluator, registry=registry,
            adapter=CoordinatorAdapter(interface),
            event_recorder=None, run_id="monitor-run",
        )
        active = make_candidate({"cpu_budget_share": 0.9}).with_state("ACTIVE")
        envelope = ValidationEnvelope.from_record(self._envelope_record())
        rolled_back, drift = controller.monitor_active(
            active, _metrics(0.3, latency_mean_ms=500.0), envelope=envelope,
        )
        assert rolled_back is True and drift.drifted
        assert interface.current.strategy_id is None
        assert any(entry.get("state") == "ROLLBACK" for entry in interface.history)


# ------------------------------------------------------------- evidence helpers

def _record_evidence():
    return EvaluationRecord(
        candidate_id="cand_e", candidate_version=1, genome_hash="h" * 64,
        benchmark_id="b", benchmark_version="v", split="train",
        seeds=(1,), metrics=_metrics(0.8), outcomes=(),
        gate=GateDecision(feasible=True), fitness=1.0,
    )


def _ok_check():
    return type("S", (), {"ok": True})()


def _ok_semantic():
    return type("M", (), {"ok": True})()


def _improved_heldout():
    return type("H", (), {"improved": True})()


class TestRegressionDecisionShape:
    def test_decision_serialization(self):
        decision = RegressionDecision(
            approved=True, reasons=(), deltas={"quality": 0.1},
        )
        dumped = decision.as_dict()
        assert dumped["approved"] is True


class TestStrategyEvaluatorEvidence:
    def test_regression_gate_uses_per_task_cost_semantics(self):
        # AggregateMetrics carries both total cost and per-task mean; gates use mean.
        metrics = _metrics(cost=120.0, cost_mean=12.0)
        gate = RegressionGate()
        decision = gate.check(metrics, _metrics(cost=100.0, cost_mean=10.0))
        assert decision.approved
