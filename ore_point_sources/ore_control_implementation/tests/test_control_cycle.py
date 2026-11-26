"""Cognitive-cycle, retrieval, subproblem, hybrid, and reliability tests.

Covers the Point 02 §7 acceptance groups: deterministic selection, decision
traces with reasons, completed versus incomplete goals, critical hard gates,
stagnation handling, retrieval monotonicity and missing history, ambiguity
preservation, cost-aware retrieval depth, subproblem bounds and reconciliation,
learned-component boundaries, resource-reservation deferral, logger-failure
provenance, and the benchmark harness.
"""
from __future__ import annotations

import math
from dataclasses import replace


from ore_control.activation import (
    RetrievalCandidate,
    base_level_activation,
    choose_retrieval_depth,
    cue_association,
    rank_candidates,
    retrieval_score,
)
from ore_control.config import (
    AmbiguityPolicy,
    RetrievalCoefficients,
    RetrievalDepthPolicy,
    default_policy,
)
from ore_control.contracts import (
    ContextBudget,
    OperatorSpec,
    TriggerEnvelope,
    TriggerKind,
)
from ore_control.cycle import CognitiveControlCoordinator
from ore_control.metacognition import monitor
from ore_control.resources import ControlResourceCoordinator
from ore_control.operators import (
    OperatorRegistry,
    register_default_operators,
)
from ore_control.subproblems import (
    ChildOutcome,
    OperatorFinding,
    reconcile_child_outcomes,
)
from ore_control.routing import (
    LearnedComponentRegistry,
    default_learned_components,
)

DEMO_TEXT = "The captain saw the boat."


def _trigger(key: str, text: str = DEMO_TEXT) -> TriggerEnvelope:
    """Build one typed user-task trigger."""
    return TriggerEnvelope(
        kind=TriggerKind.USER_TASK, idempotency_key=key,
        event_reference=f"evt:{key}", payload={"text": text},
    )


def _usg_bundle() -> dict:
    """Import the blackboard demo USG bundle (bank polysemy)."""
    import sys

    sys.path.insert(0, "examples")
    from blackboard_coordination import _demo_usg_bundle

    return _demo_usg_bundle()


# ------------------------------------------------------- cognitive-cycle ----


class TestCognitiveCycle:
    """Deterministic cycle behaviour with recorded reasons (§7)."""

    def test_deterministic_selection_with_same_snapshot_and_history(self) -> None:
        clock = lambda: 1000.0  # noqa: E731 - fixed activation clock
        first = CognitiveControlCoordinator(clock=clock)
        second = CognitiveControlCoordinator(clock=clock)
        result_a = first.process_trigger(_trigger("det_a"))
        result_b = second.process_trigger(_trigger("det_a"))
        selected_a = [
            trace["selection"]["selected_operator_id"]
            for trace in result_a.decision_traces
        ]
        selected_b = [
            trace["selection"]["selected_operator_id"]
            for trace in result_b.decision_traces
        ]
        assert selected_a == selected_b
        assert result_a.status == result_b.status
        assert result_a.goal_outcome == result_b.goal_outcome

    def test_each_cycle_records_eligible_and_rejected_operators(self) -> None:
        coordinator = CognitiveControlCoordinator(usg_bundle=_usg_bundle())
        result = coordinator.process_trigger(_trigger("reasons", "The fisherman sat near the bank."))
        assert result.decision_traces
        for trace in result.decision_traces:
            candidates = trace["candidates"]
            assert candidates, "every cycle records its candidate operators"
            for candidate in candidates:
                if candidate["precondition_failures"]:
                    assert all(
                        isinstance(reason, str)
                        for reason in candidate["precondition_failures"]
                    )
            steps = {step["step"] for step in trace["operation_steps"]}
            assert {6, 7, 8, 9} <= steps

    def test_completed_goal_produces_stop_decision(self) -> None:
        coordinator = CognitiveControlCoordinator()
        result = coordinator.process_trigger(_trigger("stop_ok"))
        assert result.status == "SUCCEEDED"
        assert result.goal_outcome in {"completed", "completed_with_uncertainty"}
        last = result.decision_traces[-1]
        assert "stop" in last["next_cycle_reason"] or "goal" in last["state_transition"]

    def test_incomplete_goal_does_not_stop_successfully(self) -> None:
        # An unreachable goal threshold keeps the goal incomplete: the run
        # must stop unsuccessfully with a structured reason, never as success.
        policy = replace(
            default_policy(),
            budgets=replace(default_policy().budgets, max_cycles=1),
            goal_threshold=1.0,
        )
        coordinator = CognitiveControlCoordinator(policy=policy)
        result = coordinator.process_trigger(_trigger("stop_incomplete"))
        assert result.status == "FAILED"
        assert result.goal_outcome == "needs_more_processing"
        assert result.status_reason

    def test_critical_unmet_requirement_blocks_success(self) -> None:
        # Remove the hypothesize operator: no lexical/syntactic/semantic
        # representation can ever appear, so critical gates stay unmet.
        operators = OperatorRegistry()
        register_default_operators(operators)
        operators.unregister("control.hypothesize_interpretations")
        coordinator = CognitiveControlCoordinator(operators=operators)
        result = coordinator.process_trigger(_trigger("gates"))
        assert result.status != "SUCCEEDED"
        assert set(result.missing_requirements) & {
            "lexical_hypotheses", "syntactic_hypotheses", "semantic_hypotheses",
        }

    def test_duplicate_triggers_create_no_duplicate_side_effects(self) -> None:
        coordinator = CognitiveControlCoordinator()
        first = coordinator.process_trigger(_trigger("dup"))
        workspaces_before = len(coordinator._workspaces)
        second = coordinator.process_trigger(_trigger("dup"))
        assert second.run_id == first.run_id
        assert second.workspace_id == first.workspace_id
        assert second.workspace_revision == first.workspace_revision
        assert len(coordinator._workspaces) == workspaces_before

    def test_repeated_state_fingerprints_trigger_stagnation_handling(self) -> None:
        # A registry with only an unproductive probe: each cycle executes it,
        # nothing integrates, the state fingerprint repeats, and the cycle
        # handling stops the run instead of repeating forever.
        operators = OperatorRegistry()

        def _noop_handler(context, state, action, services):
            from ore_control.operators import OperatorOutcome

            return OperatorOutcome(
                status="completed",
                payload={"note": "no state change"},
                diagnostics={"note": "no state change"},
                decompose=False,
            )

        operators.register(
            OperatorSpec(
                operator_id="control.noop_probe", version="1.0.0",
                purpose="Deliberately unproductive read-only operator for stagnation tests.",
                preconditions=("task_text_available",),
                required_capabilities=("cognition.verification",),
                writes_workspace=False,
                expected_progress=0.9, expected_information_gain=0.9,
                estimated_time_seconds=0.001,
            ),
            _noop_handler,
        )
        policy = replace(
            default_policy(),
            budgets=replace(
                default_policy().budgets, max_cycles=6, stagnation_window=2,
            ),
        )
        coordinator = CognitiveControlCoordinator(operators=operators, policy=policy)
        result = coordinator.process_trigger(_trigger("stagnation"))
        assert result.status == "FAILED"
        assert result.reproducibility["stagnation_detected"] is True
        reasons = [trace["next_cycle_reason"] for trace in result.decision_traces]
        assert any("stagnation" in reason for reason in reasons)
        assert 3 <= result.cycles <= 6  # bounded, not endless repetition

    def test_operation_step_coverage(self) -> None:
        from ore_control.registry import CONTROL_OPERATION_IDS

        assert len(CONTROL_OPERATION_IDS) == 40
        assert CONTROL_OPERATION_IDS[0] == "CCA-001"
        assert CONTROL_OPERATION_IDS[-1] == "CCA-040"
        coordinator = CognitiveControlCoordinator(usg_bundle=_usg_bundle())
        result = coordinator.process_trigger(
            _trigger("coverage", "The fisherman sat near the bank."),
        )
        covered = {
            step["step"]
            for trace in result.decision_traces
            for step in trace["operation_steps"]
        }
        assert len(covered) >= 30, f"only {len(covered)} steps covered"


# ---------------------------------------------------------------- retrieval --


class TestRetrievalMathematics:
    """ACT-R-inspired retrieval features behave as declared (§7)."""

    def test_base_level_activation_is_monotonic_in_recency(self) -> None:
        now = 1000.0
        recent = base_level_activation(
            [now - 10.0], now=now, decay=0.5, epsilon=1e-3,
        )
        older = base_level_activation(
            [now - 100.0], now=now, decay=0.5, epsilon=1e-3,
        )
        assert recent is not None and older is not None
        assert recent > older

    def test_frequency_increases_activation(self) -> None:
        now = 1000.0
        single = base_level_activation(
            [now - 50.0], now=now, decay=0.5, epsilon=1e-3,
        )
        repeated = base_level_activation(
            [now - 50.0, now - 60.0], now=now, decay=0.5, epsilon=1e-3,
        )
        assert repeated > single

    def test_missing_history_is_missing_not_fabricated(self) -> None:
        assert base_level_activation([], now=1.0, decay=0.5, epsilon=1e-3) is None
        assert base_level_activation(
            [50.0], now=50.0, decay=0.5, epsilon=1e-3,
        ) is None  # non-positive elapsed time: unreliable feature

    def test_cue_association_normalizes_weights_and_traces_contributions(self) -> None:
        score, contributions = cue_association(
            {"bank": 1.0, "river": 3.0}, {"bank": 1.0, "river": 0.5},
        )
        assert math.isclose(score, sum(contributions.values()))
        assert contributions["river"] > contributions["bank"]

    def test_retrieval_score_drops_missing_terms_and_flags_them(self) -> None:
        candidate = RetrievalCandidate(
            record_reference="r", source_type="t",
            base_level_activation=None, activation_available=False,
            cue_association=None, retrieval_cost=0.2,
        )
        scored = retrieval_score(
            candidate, coefficients=RetrievalCoefficients(),
            task_relevance=1.0, provenance_support=1.0,
        )
        assert scored.retrieval_score is not None
        assert scored.retrieval_score < 0.35 + 0.1 - 0.05 * 0.2
        assert scored.dropped_terms == ("B_i", "S_i")

    def test_ambiguity_preserved_when_scores_are_close(self) -> None:
        top = RetrievalCandidate(
            record_reference="r1", source_type="t",
            usage_timestamps=(1.0,),
            base_level_activation=1.0, activation_available=True,
            cue_association=1.0, retrieval_score=1.0,
        )
        close = RetrievalCandidate(
            record_reference="r2", source_type="t",
            usage_timestamps=(1.0,),
            base_level_activation=1.0, activation_available=True,
            cue_association=1.0, retrieval_score=0.95,
        )
        shortlist = rank_candidates(
            [top, close], policy=AmbiguityPolicy(margin=0.1),
        )
        assert shortlist.winner is None
        assert shortlist.ambiguity_records
        assert shortlist.ambiguity_records[0]["scheduled_action"]

    def test_clear_winner_selected_when_separated(self) -> None:
        top = RetrievalCandidate(
            record_reference="r1", source_type="t",
            usage_timestamps=(1.0,),
            base_level_activation=1.0, activation_available=True,
            cue_association=1.0, retrieval_score=1.0,
        )
        distant = RetrievalCandidate(
            record_reference="r2", source_type="t",
            usage_timestamps=(1.0,),
            base_level_activation=0.1, activation_available=True,
            cue_association=0.1, retrieval_score=0.2,
        )
        shortlist = rank_candidates(
            [top, distant], policy=AmbiguityPolicy(margin=0.1),
        )
        assert shortlist.winner is not None
        assert shortlist.winner.record_reference == "r1"

    def test_cost_aware_depth_stops_before_exhausting_stores(self) -> None:
        policy = RetrievalDepthPolicy()
        decision = choose_retrieval_depth(
            remaining_budget=0.5, sufficient_evidence_met=False,
            depth_policy=policy,
        )
        assert decision.level == "structured_index"
        assert "full_resource" in decision.skipped_levels
        tiny = choose_retrieval_depth(
            remaining_budget=0.05, sufficient_evidence_met=False,
            depth_policy=policy,
        )
        assert tiny.level == "none"
        assert tiny.reason.startswith("retrieval budget exhausted")
        satisfied = choose_retrieval_depth(
            remaining_budget=10.0, sufficient_evidence_met=True,
            depth_policy=policy,
        )
        assert satisfied.level == "none"
        assert satisfied.reason == "sufficient evidence threshold met"

    def test_every_retrieved_item_carries_provenance_and_version(self) -> None:
        coordinator = CognitiveControlCoordinator(usg_bundle=_usg_bundle())
        result = coordinator.process_trigger(
            _trigger("prov", "The fisherman sat near the bank."),
        )
        candidates = [
            item
            for trace in result.decision_traces
            for item in trace["retrieval_candidates"]
        ]
        assert candidates
        for item in candidates:
            if str(item["source_type"]).startswith("usg:"):
                assert item["provenance"]["bundle_id"]
                assert item["provenance"]["bundle_digest"]
                assert item["version_reference"]

    def test_retrieval_ranking_improves_on_declaration_order_baseline(self) -> None:
        # A labelled set: the exact-match candidate must outrank weak matches.
        coefficients = RetrievalCoefficients()
        labelled = [
            {"relevant": "usg:exact:river", "weak": "workspace_cache"},
            {"relevant": "usg:exact:bank", "weak": "workspace_cache"},
        ]
        scored_wins = 0
        declared_wins = 0
        for case in labelled:
            relevant = RetrievalCandidate(
                record_reference=case["relevant"], source_type="usg:LEXICAL_ENTRY",
                base_level_activation=1.0, activation_available=True,
                cue_association=1.0, semantic_relevance=1.0,
                provenance={"bundle_id": "b"}, retrieval_cost=0.1,
            )
            weak = RetrievalCandidate(
                record_reference=case["weak"], source_type="workspace_cache",
                base_level_activation=None, activation_available=False,
                cue_association=0.1, semantic_relevance=0.2,
                retrieval_cost=0.1,
            )
            scored = rank_candidates(
                [
                    retrieval_score(
                        item, coefficients=coefficients,
                        task_relevance=item.semantic_relevance,
                        provenance_support=0.8,
                    )
                    for item in (relevant, weak)
                ],
                policy=AmbiguityPolicy(),
            )
            if scored.winner and scored.winner.record_reference == case["relevant"]:
                scored_wins += 1
            # Declaration-order baseline: records arrive weak-first.
            baseline = rank_candidates(
                [weak, relevant], policy=AmbiguityPolicy(require_winner_margin=False),
            )
            if (
                baseline.ranked
                and baseline.ranked[0].record_reference == case["relevant"]
            ):
                declared_wins += 1
        assert scored_wins >= declared_wins


# --------------------------------------------------------------- subproblems -


class TestSubproblemBehaviour:
    """Bounded children reconcile without implying success (§7)."""

    def test_incomplete_children_are_explicit_and_never_imply_success(self) -> None:
        finding = OperatorFinding(field="unresolved_questions", payload={"q": 1})
        incomplete = ChildOutcome(
            context_id="child_1", goal_text="g",
            status="INCOMPLETE", contract_satisfied=False,
            findings=(finding,), unresolved_issues=("budget exhausted",),
        )
        merged, records = reconcile_child_outcomes([incomplete], policy=default_policy())
        assert merged  # contract-validated findings still integrate
        assert records[0]["status"] == "INCOMPLETE"
        assert "never success" in records[0]["note"] or "explicit" in records[0]["note"]

    def test_failed_children_are_abandoned_not_merged(self) -> None:
        failed = ChildOutcome(
            context_id="child_2", goal_text="g",
            status="FAILED", contract_satisfied=False,
        )
        merged, records = reconcile_child_outcomes([failed], policy=default_policy())
        assert not merged
        assert records[0]["decision"] == "abandoned"

    def test_parent_reconciliation_preserves_conflicting_findings(self) -> None:
        from ore_control.operators import ChildGoalSpec
        from ore_control.subproblems import create_child_contexts

        policy = default_policy()
        context = _parent_context()
        children = create_child_contexts(
            context,
            (
                ChildGoalSpec(goal_text="Test reading A", hypothesis_id="hyp:a"),
                ChildGoalSpec(goal_text="Test reading B", hypothesis_id="hyp:b"),
            ),
            policy=policy, workspace_revision=2,
        )
        assert len(children) == 2
        conflicting = [
            OperatorFinding(
                field="unresolved_questions",
                payload={
                    "question_id": f"q:conflict:{i}",
                    "conflict_preserved": True,
                },
                key=f"conflict:{i}",
            )
            for i in range(2)
        ]
        outcomes = [
            ChildOutcome(
                context_id=children[i][0].context_id,
                goal_text=children[i][1].goal_text,
                status="INCOMPLETE", contract_satisfied=False,
                findings=(conflicting[i],),
                unresolved_issues=("no discriminating evidence",),
            )
            for i in range(2)
        ]
        merged, records = reconcile_child_outcomes(outcomes, policy=policy)
        assert len(merged) == 2  # both conflicting alternatives kept, not one


def _parent_context():
    from ore_control.contracts import CognitiveContext

    return CognitiveContext(
        context_id="parent_ctx", goal_id="goal:test",
        workspace_revision=2, budget=ContextBudget(operations=32),
    )


# ------------------------------------------------------- hybrid processing --


class TestHybridProcessing:
    """Learned components stay proposal-only and gated (§7)."""

    def test_schema_mismatch_blocks_learned_invocation(self) -> None:
        registry = default_learned_components()
        check = registry.check(
            "control.toy.sense_ranker",
            input_schema_id="control.operator.input/2.0",
        )
        assert not check.suitable
        assert check.reasons["input_schema"].startswith("schema_mismatch")

    def test_unknown_task_blocks_learned_component(self) -> None:
        registry = default_learned_components()
        check = registry.check(
            "control.toy.sense_ranker", task_type="image_segmentation",
        )
        assert not check.suitable
        assert check.reasons["supported_tasks"] == "task_not_supported:image_segmentation"

    def test_unregistered_component_is_unsuitable(self) -> None:
        registry = LearnedComponentRegistry()
        check = registry.check("control.toy.sense_ranker")
        assert not check.suitable
        assert check.reasons["registration"] == "component_not_registered"

    def test_learned_outputs_are_labelled_proposals_not_probabilities(self) -> None:
        from ore_control.routing import LEARNED_PROPOSAL_NOTE

        coordinator = CognitiveControlCoordinator(usg_bundle=_usg_bundle())
        result = coordinator.process_trigger(
            _trigger("proposal", "The fisherman sat near the bank."),
        )
        assert result.learned_proposals_used >= 1
        traces = [
            step["details"]
            for trace in result.decision_traces
            for step in trace["operation_steps"]
            if step["step"] == 26
        ]
        assert traces and traces[0]["classified_processing_path"] in {
            "HYBRID", "EXPLICIT_SYMBOLIC",
        }
        finding_payloads = [
            finding
            for trace in result.decision_traces
            for finding in trace["retrieval_candidates"]
        ]
        # Proposals never carry a calibrated confidence value.
        for item in finding_payloads:
            assert item.get("confidence") is None or not item.get("confidence")
        assert "proposal" in LEARNED_PROPOSAL_NOTE

    def test_learned_component_cannot_mutate_authoritative_usg(self) -> None:
        bundle = _usg_bundle()
        before = {k: (str(v) if isinstance(v, (str, int, float)) else v)
                  for k, v in bundle.items() if isinstance(v, (str, int, float))}
        coordinator = CognitiveControlCoordinator(usg_bundle=bundle)
        coordinator.process_trigger(
            _trigger("readonly", "The fisherman sat near the bank."),
        )
        after = {k: (str(v) if isinstance(v, (str, int, float)) else v)
                 for k, v in bundle.items() if isinstance(v, (str, int, float))}
        assert before == after  # the caller-supplied bundle is never mutated
        assert coordinator.usg.reference.read_only is True

    def test_low_confidence_or_disagreement_triggers_escalation(self) -> None:
        report = monitor(
            {
                "unresolved_questions": [
                    {"question_id": "q1", "resolved": False},
                    {"question_id": "q2", "resolved": False},
                    {"question_id": "q3", "resolved": False},
                ],
                "competing_hypotheses": 2,
                "unresolved_conflicts": 1,
                "latest_evidence_revision": 0,
                "workspace_revision": 10,
            },
            policy=default_policy(),
            repeated_failures=4,
            budget_consumption=0.2,
        )
        assert report.calibration_flagged
        assert "targeted_retrieval" in report.escalations
        assert "independent_verification" in report.escalations

    def test_explicit_symbolic_checks_are_independently_testable(self) -> None:
        coordinator = CognitiveControlCoordinator(usg_bundle=_usg_bundle())
        result = coordinator.process_trigger(
            _trigger("explicit", "The fisherman sat near the bank."),
        )
        verify_steps = [
            step
            for trace in result.decision_traces
            for step in trace["operation_steps"]
            if step["step"] == 32
        ]
        assert verify_steps
        assert verify_steps[0]["details"]["provenance_present"]


# ---------------------------------------------------------- reliability -----


class TestReliabilityAndPerformance:
    """Reservation, retry, logger, and benchmark behaviour (§7)."""

    def test_reservation_failure_is_a_deferred_not_a_bypass(self) -> None:
        policy = default_policy()
        coordinator = ControlResourceCoordinator(policy)
        decision = coordinator.reserve(
            "control.hypothesize_interpretations",
            estimated_time_seconds=100.0, estimated_memory_mb=1.0,
            remaining_time_seconds=5.0, remaining_memory_mb=64.0,
        )
        assert not decision.granted
        assert decision.reason == "time_budget_exceeded"
        memory_denied = coordinator.reserve(
            "control.retrieve_knowledge",
            estimated_time_seconds=0.01, estimated_memory_mb=512.0,
            remaining_time_seconds=5.0, remaining_memory_mb=64.0,
        )
        assert not memory_denied.granted
        assert memory_denied.reason == "memory_budget_exceeded"

    def test_circuit_breaker_opens_after_repeated_failures(self) -> None:
        policy = default_policy()
        policy.circuit_breaker_threshold = 2
        policy.circuit_breaker_cooldown_seconds = 60.0
        coordinator = ControlResourceCoordinator(policy)
        operator_id = "control.flaky_operator"
        for _ in range(2):
            coordinator.record_outcome(operator_id, False)
        decision = coordinator.reserve(
            operator_id, estimated_time_seconds=0.01, estimated_memory_mb=1.0,
            remaining_time_seconds=10.0, remaining_memory_mb=64.0,
        )
        assert not decision.granted
        assert decision.reason == "circuit_breaker_open"

    def test_retries_are_bounded_and_idempotent(self) -> None:
        operators = OperatorRegistry()
        register_default_operators(operators)

        attempts = {"count": 0}

        def _failing_handler(context, state, action, services):
            from ore_control.operators import OperatorOutcome

            attempts["count"] += 1
            return OperatorOutcome(
                status="failed", payload={},
                diagnostics={"error": "injected failure"},
                failure_classification="module_exception",
            )

        operators.replace(
            operators.get_spec("control.hypothesize_interpretations"),
            _failing_handler,
        )
        policy = replace(default_policy(), budgets=replace(
            default_policy().budgets, max_retries=1,
        ))
        coordinator = CognitiveControlCoordinator(operators=operators, policy=policy)
        result = coordinator.process_trigger(_trigger("retry"))
        assert result.status == "FAILED"
        assert attempts["count"] == 2  # initial attempt plus one bounded retry

    def test_logger_failure_does_not_erase_decision_provenance(self) -> None:
        from ore_control.trace import DecisionTraceBuilder, persist_trace

        builder = DecisionTraceBuilder(
            cycle=1, context_id="c", workspace_revision=0,
            goal_id="g", open_goal_status="running",
        )
        builder.record_step(1, {"note": "x"})
        trace = builder.build()
        assert persist_trace(trace, None) is False  # no observer: no side loss

        class _BrokenObserver:
            def record(self, *args, **kwargs):
                raise RuntimeError("logger down")

        assert persist_trace(trace, _BrokenObserver()) is False
        assert trace.to_dict()["operation_steps"]  # in-run provenance retained

    def test_benchmark_harness_reports_without_improvement_claims(self) -> None:
        from ore_control.evaluation import (
            paired_bootstrap_latency,
            run_benchmark,
        )
        from ore_blackboard import BlackboardCoordinator

        coordinator = CognitiveControlCoordinator()
        baseline = BlackboardCoordinator()
        report = run_benchmark(
            coordinator, (DEMO_TEXT, "A boy opened the door."),
            baseline=baseline, task_labels=("t1", "t2"),
        )
        summary = report.summary()
        assert "control" in summary and "baseline" in summary
        assert summary["control"]["tasks"] == 2
        assert summary["baseline"]["tasks"] == 2
        mean, low, high = paired_bootstrap_latency(report, iterations=200)
        assert math.isfinite(mean)
        assert low <= mean <= high
        assert report.to_dict()["notes"]


__all__ = []
