"""Procedure-learning tests (Point 02 Phase H, §7).

A single successful run never promotes a procedure; held-out and
counterexample tests are required; every promoted procedure is versioned and
reversible; a degraded procedure can be disabled without corrupting
authoritative knowledge.
"""
from __future__ import annotations

import pytest

from dataclasses import replace

from ore_control.config import default_policy
from ore_control.contracts import RiskClass, TriggerEnvelope, TriggerKind
from ore_control.cycle import CognitiveControlCoordinator
from ore_control.procedures import (
    CandidateProcedure,
    GeneralizationReport,
    ProcedureRecord,
    ProcedureRegistry,
    ProcedureSequenceStep,
    VerifiedRunRecord,
    evaluate_promotion_gates,
    extract_candidate_procedure,
    wrap_procedure,
)
from ore_control.operators import OperatorRegistry, register_default_operators

DEMO_TEXT = "The captain saw the boat."


def _trigger(key: str, text: str = DEMO_TEXT) -> TriggerEnvelope:
    """Build one typed user-task trigger."""
    return TriggerEnvelope(
        kind=TriggerKind.USER_TASK, idempotency_key=key,
        event_reference=f"evt:{key}", payload={"text": text},
    )


def _verified_run(run_id: str, text: str, sequence=("control.hypothesize_interpretations",)) -> VerifiedRunRecord:
    """Build one verified-run record for extraction tests."""
    return VerifiedRunRecord(
        run_id=run_id,
        goal_fingerprint="goal_fp",
        task_fingerprint=f"task:{text}",
        operator_sequence=tuple(
            ProcedureSequenceStep(operator_id=operator_id, normalized_arguments={})
            for operator_id in sequence
        ),
        status="SUCCEEDED",
        mean_latency_ms=10.0,
    )


class TestExtraction:
    """Candidate extraction requires repeatability (Step 37)."""

    def test_single_successful_run_is_insufficient(self) -> None:
        candidate = extract_candidate_procedure(
            [_verified_run("run_1", DEMO_TEXT)], policy=default_policy(),
        )
        assert candidate is None

    def test_repeatable_sequence_is_extracted(self) -> None:
        candidate = extract_candidate_procedure(
            [
                _verified_run("run_1", "Text A."),
                _verified_run("run_2", "Text B."),
            ],
            policy=default_policy(),
        )
        assert candidate is not None
        assert candidate.repeatable_runs == 2
        assert "single successful execution" not in " ".join(candidate.assumptions)
        assert len(candidate.extraction_provenance["run_ids"]) == 2

    def test_different_sequences_are_not_merged(self) -> None:
        candidate = extract_candidate_procedure(
            [
                _verified_run("run_1", "A.", sequence=("control.hypothesize_interpretations",)),
                _verified_run("run_2", "B.", sequence=("control.retrieve_knowledge",)),
            ],
            policy=default_policy(),
        )
        assert candidate is None

    def test_failed_runs_are_not_extraction_evidence(self) -> None:
        runs = [_verified_run("run_1", "A."), _verified_run("run_2", "B.")]
        failed = VerifiedRunRecord(
            run_id="run_3", goal_fingerprint="other_goal",
            task_fingerprint="task:C.",
            operator_sequence=runs[0].operator_sequence,
            status="FAILED", mean_latency_ms=1.0,
        )
        candidate = extract_candidate_procedure(
            [runs[0], failed], policy=default_policy(),
        )
        assert candidate is None


class TestGeneralizationGates:
    """Promotion requires held-out and counterexample evidence (Step 38-39)."""

    def _candidate(self, runs: int = 2) -> CandidateProcedure:
        return CandidateProcedure(
            procedure_id="procedure:test",
            goal_fingerprint="goal_fp",
            operator_sequence=(
                ProcedureSequenceStep(
                    operator_id="control.hypothesize_interpretations",
                    normalized_arguments={},
                ),
            ),
            repeatable_runs=runs,
            extraction_provenance={
                "run_ids": [f"run_{i}" for i in range(runs)],
                "reference_latency_ms": 10.0,
            },
        )

    def _report(self, **overrides) -> GeneralizationReport:
        defaults = dict(
            held_out_successes=3, held_out_total=3,
            counterexample_rejections=2, counterexample_total=2,
            mean_replay_latency_ms=8.0, reference_latency_ms=10.0,
            resource_regression_ratio=0.8, memorization_detected=False,
        )
        defaults.update(overrides)
        return GeneralizationReport(**defaults)

    def test_held_out_and_counterexamples_are_required(self) -> None:
        policy = default_policy()
        no_held_out = self._report(held_out_successes=0, held_out_total=0)
        decision = evaluate_promotion_gates(
            self._candidate(), no_held_out, policy=policy,
        )
        assert not decision.promoted
        assert not decision.gate_results["held_out_success"]["passed"]

        accepted_counterexample = self._report(counterexample_rejections=1)
        decision = evaluate_promotion_gates(
            self._candidate(), accepted_counterexample, policy=policy,
        )
        assert not decision.promoted
        assert not decision.gate_results["counterexamples_rejected"]["passed"]

    def test_resource_regression_blocks_promotion(self) -> None:
        policy = default_policy()
        regression = self._report(
            mean_replay_latency_ms=20.0, resource_regression_ratio=2.0,
        )
        decision = evaluate_promotion_gates(
            self._candidate(), regression, policy=policy,
        )
        assert not decision.promoted
        assert not decision.gate_results["resource_regression"]["passed"]

    def test_memorization_blocks_promotion(self) -> None:
        policy = default_policy()
        memorizer = self._report(memorization_detected=True)
        decision = evaluate_promotion_gates(
            self._candidate(), memorizer, policy=policy,
        )
        assert not decision.promoted
        assert not decision.gate_results["no_memorization"]["passed"]

    def test_high_risk_requires_manual_approval(self) -> None:
        policy = default_policy()
        good = self._report()
        auto_low = evaluate_promotion_gates(
            self._candidate(), good, policy=policy, risk_class=RiskClass.LOW,
        )
        assert auto_low.promoted
        high_denied = evaluate_promotion_gates(
            self._candidate(), good, policy=policy, risk_class=RiskClass.HIGH,
        )
        assert not high_denied.promoted
        assert not high_denied.gate_results["risk_approval"]["passed"]
        high_approved = evaluate_promotion_gates(
            self._candidate(), good, policy=policy, risk_class=RiskClass.HIGH,
            manual_approval=True,
        )
        assert high_approved.promoted

    def test_positive_mean_score_alone_is_insufficient(self) -> None:
        policy = default_policy()
        candidate = self._candidate(runs=1)  # below repeatability gate
        decision = evaluate_promotion_gates(
            candidate, self._report(), policy=policy,
        )
        assert not decision.promoted
        assert not decision.gate_results["repeatable_runs"]["passed"]


class TestProcedureRegistry:
    """Versioned, reversible, monitored procedure storage (Steps 39-40)."""

    def _record(self, procedure_id: str = "procedure:unit") -> ProcedureRecord:
        return ProcedureRecord(
            procedure_id=procedure_id, version="1.0.0",
            operator_sequence=(
                ProcedureSequenceStep(
                    operator_id="control.hypothesize_interpretations",
                    normalized_arguments={},
                ),
            ),
            goal_fingerprint="goal_fp",
            preconditions=("task_text_available",),
            parameters={"sequence": ["control.hypothesize_interpretations"]},
            provenance={"run_ids": ["r1", "r2"], "non_authoritative": True},
            test_results={"success_rate": 1.0},
            limitations=("reference replay only",),
            rollback_information={"prior_version": None},
            risk_class=RiskClass.LOW,
            approved=True,
        )

    def test_promoted_procedure_is_versioned_and_reversible(self) -> None:
        registry = ProcedureRegistry()
        record = self._record()
        registry.register(record)
        assert registry.get("procedure:unit").version == "1.0.0"
        assert registry.rollback("procedure:unit") is True
        events = [event["event"] for event in registry.events]
        assert "procedure_rolled_back" in events

    def test_duplicate_registration_is_rejected(self) -> None:
        registry = ProcedureRegistry()
        registry.register(self._record())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(self._record())

    def test_degraded_procedure_is_disabled_without_corrupting_knowledge(self) -> None:
        policy = default_policy()
        policy = replace(
            policy,
            procedures=replace(
                policy.procedures, monitoring_window=3, disable_success_rate=0.5,
            ),
        )
        registry = ProcedureRegistry()
        record = self._record("procedure:degrade")
        registry.register(record)
        for _ in range(3):
            registry.record_execution(
                "procedure:degrade", success=False, latency_ms=1.0,
            )
        healthy, reason = registry.check_health(
            "procedure:degrade", policy=policy,
        )
        assert not healthy
        assert reason == "degraded_success_rate"
        assert record.enabled is False
        # The registry holds no workspace or USG references to corrupt.
        assert record.to_dict()["authoritative"] is False

    def test_quarantine_after_safety_violation(self) -> None:
        registry = ProcedureRegistry()
        record = self._record("procedure:unsafe")
        registry.register(record)
        registry.quarantine("procedure:unsafe", reason="safety_condition_violated")
        assert record.quarantined is True
        assert record.enabled is False
        assert any(
            event["event"] == "procedure_quarantined" for event in registry.events
        )

    def test_procedure_reuse_is_wrapped_as_candidate_operator(self) -> None:
        operators = OperatorRegistry()
        register_default_operators(operators)
        registry = ProcedureRegistry()
        record = self._record("procedure:reuse")
        registry.register(record)
        spec, handler = wrap_procedure(
            record, registry=operators, procedure_registry=registry,
        )
        assert spec.operator_id == "procedure:reuse"
        assert spec.version == record.version
        assert callable(handler)
        assert "artifact_references" in spec.write_fields


class TestEndToEndProcedureLearning:
    """The coordinator promotes procedures only after verification."""

    def test_end_to_end_promotion_requires_two_verified_runs(self) -> None:
        coordinator = CognitiveControlCoordinator()
        first = coordinator.process_trigger(_trigger("p1", "The captain saw the boat."))
        events = [event["event"] for event in first.procedure_events]
        assert events == ["procedure_extraction"]
        assert first.procedure_events[0]["result"] == "no_repeatable_sequence"

        second = coordinator.process_trigger(_trigger("p2", "A boy opened the door."))
        events = [event["event"] for event in second.procedure_events]
        assert "procedure_registered" in events
        assert coordinator.procedure_registry.records()
        record = coordinator.procedure_registry.records()[0]
        assert record.version == "1.0.0"
        assert record.to_dict()["authoritative"] is False

        third = coordinator.process_trigger(_trigger("p3", "The fisherman sat near the river."))
        events = [event["event"] for event in third.procedure_events]
        assert "procedure_already_registered" in events

    def test_promotion_gates_visible_in_run_events(self) -> None:
        coordinator = CognitiveControlCoordinator()
        coordinator.process_trigger(_trigger("g1", "The captain saw the boat."))
        second = coordinator.process_trigger(_trigger("g2", "A boy opened the door."))
        promotion = next(
            event for event in second.procedure_events
            if event["event"] == "procedure_promotion"
        )
        gates = promotion["decision"]["gate_results"]
        for gate_name in (
            "repeatable_runs", "held_out_success", "counterexamples_rejected",
            "no_memorization", "resource_regression", "risk_approval",
        ):
            assert gate_name in gates
            assert gates[gate_name]["passed"] is True


__all__ = []
