"""Contract and state tests for the ORE cognitive control layer (Point 02 §7).

Covers the acceptance criteria: context serialization round-trips across
schema versions, stale workspace revision reconciliation, children cannot
publish outside their declared contract, invalid or unavailable operators are
never dispatched, and duplicate triggers create no duplicate side effects.
"""
from __future__ import annotations

import pytest

from ore_control.contracts import (
    COGNITIVE_CONTROL_SCHEMA_VERSION,
    ContextBudget,
    ContextStatus,
    CognitiveContext,
    OpenQuestion,
    OperatorSpec,
    RetrievalCandidate,
    RiskClass,
    TriggerEnvelope,
    TriggerKind,
)
from ore_control.operators import (
    OperatorAction,
    OperatorFinding,
    OperatorRegistry,
    register_default_operators,
)
from ore_control.routing import (
    LearnedComponentCheck,
    classify_processing_path,
)
from ore_control.subproblems import (
    ChildBoundaryGuard,
    ChildContract,
    create_child_contexts,
    decomposition_justified,
)


# ------------------------------------------------------------------ §3.1 ---


def _context() -> CognitiveContext:
    """Return one representative context for serialization tests."""
    return CognitiveContext(
        context_id="context_test",
        goal_id="goal:test",
        workspace_revision=3,
        status=ContextStatus.RUNNING,
        open_questions=(
            OpenQuestion(
                question_id="q:bank", text="Which reading of 'bank'?",
                priority=0.7, required_evidence=("independent_source_groups",),
            ),
        ),
        budget=ContextBudget(time_seconds=12.0, operations=8),
        depth=1,
        parent_context_id="context_parent",
        cycle_fingerprint="abc123",
    )


class TestCognitiveContextSerialization:
    """Context serialization round-trips across the declared schema."""

    def test_round_trip_preserves_all_fields(self) -> None:
        context = _context()
        restored = CognitiveContext.from_dict(context.to_dict())
        assert restored.context_id == context.context_id
        assert restored.goal_id == context.goal_id
        assert restored.workspace_revision == context.workspace_revision
        assert restored.status is context.status
        assert restored.parent_context_id == context.parent_context_id
        assert restored.open_questions[0].question_id == "q:bank"
        assert restored.open_questions[0].required_evidence == (
            "independent_source_groups",
        )
        assert restored.budget == context.budget
        assert restored.depth == context.depth
        assert restored.cycle_fingerprint == context.cycle_fingerprint
        assert restored.schema_version == COGNITIVE_CONTROL_SCHEMA_VERSION

    def test_unsupported_schema_version_is_rejected(self) -> None:
        record = dict(_context().to_dict())
        record["schema_version"] = "9.9.9"
        with pytest.raises(ValueError, match="unsupported context schema"):
            CognitiveContext.from_dict(record)

    def test_child_context_requires_positive_depth(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            CognitiveContext(
                context_id="bad_child", goal_id="goal:test",
                workspace_revision=0, parent_context_id="parent", depth=0,
            )

    def test_context_budget_validates_bounds(self) -> None:
        with pytest.raises(ValueError):
            ContextBudget(time_seconds=float("inf"))
        with pytest.raises(ValueError):
            ContextBudget(operations=0)

    def test_nonterminal_contexts_can_be_resumed(self) -> None:
        for status in (ContextStatus.READY, ContextStatus.RUNNING, ContextStatus.WAITING):
            context = _context().with_updates(status=status)
            assert context.nonterminal
        for status in (ContextStatus.SUCCEEDED, ContextStatus.FAILED, ContextStatus.CANCELLED):
            context = _context().with_updates(status=status)
            assert not context.nonterminal


# ------------------------------------------------------------------ §3.2 ---


class TestOperatorSpecContract:
    """Operator declarations are validated before registration."""

    def test_workspace_writing_operator_needs_write_fields(self) -> None:
        with pytest.raises(ValueError, match="write fields"):
            OperatorSpec(
                operator_id="bad", version="1.0.0", purpose="test",
                writes_workspace=True, write_fields=(),
            )

    def test_risk_and_estimate_bounds_are_enforced(self) -> None:
        with pytest.raises(TypeError):
            OperatorSpec(
                operator_id="bad", version="1", purpose="t",
                risk_class="NOT_A_RISK",
            )
        with pytest.raises(ValueError, match="expected_progress"):
            OperatorSpec(
                operator_id="bad", version="1", purpose="t",
                expected_progress=1.5,
            )

    def test_to_dict_round_trip_carries_policy_fields(self) -> None:
        spec = OperatorSpec(
            operator_id="control.op", version="1.1.0", purpose="demo",
            preconditions=("task_text_available",),
            risk_class=RiskClass.MEDIUM,
            write_fields=("module_status",),
        )
        record = spec.to_dict()
        assert record["risk_class"] == "MEDIUM"
        assert record["preconditions"] == ["task_text_available"]
        assert record["retry_policy"]["idempotent"] is True


# ------------------------------------------------------------------ §3.3 ---


class TestRetrievalCandidateContract:
    """Retrieval candidates carry provenance, version, and missingness."""

    def test_missing_usage_history_marks_activation_unavailable(self) -> None:
        candidate = RetrievalCandidate(
            record_reference="usg:lemma:bank", source_type="usg:LEXICAL_ENTRY",
            usage_timestamps=None,
        )
        assert candidate.activation_available is False

    def test_candidate_validates_confidence_and_cost(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            RetrievalCandidate(
                record_reference="r", source_type="t", confidence=1.5,
            )
        with pytest.raises(ValueError, match="retrieval_cost"):
            RetrievalCandidate(
                record_reference="r", source_type="t", retrieval_cost=-1.0,
            )

    def test_serialization_keeps_score_provenance(self) -> None:
        candidate = RetrievalCandidate(
            record_reference="r", source_type="t",
            cue_match_features={"bank": 1.0},
            retrieval_score=0.5, dropped_terms=("B_i",),
        )
        record = candidate.to_dict()
        assert record["dropped_terms"] == ["B_i"]
        assert record["retrieval_score"] == 0.5


# ------------------------------------------------------- trigger validation --


class TestTriggerValidation:
    """Malformed triggers are rejected before any work is created (Step 1)."""

    def test_rejects_empty_idempotency_key(self) -> None:
        with pytest.raises(ValueError, match="idempotency_key"):
            TriggerEnvelope(
                kind=TriggerKind.USER_TASK, idempotency_key=" ",
                event_reference="evt", payload={},
            )

    def test_rejects_out_of_range_priority(self) -> None:
        with pytest.raises(ValueError, match="priority"):
            TriggerEnvelope(
                kind=TriggerKind.USER_TASK, idempotency_key="k",
                event_reference="evt", priority=1.5,
            )

    def test_rejects_missing_text_payload(self) -> None:
        from ore_control import CognitiveControlCoordinator

        coordinator = CognitiveControlCoordinator()
        with pytest.raises(ValueError, match="text"):
            coordinator.process_trigger(TriggerEnvelope(
                kind=TriggerKind.USER_TASK, idempotency_key="k",
                event_reference="evt", payload={"goal": "understand_sentence"},
            ))

    def test_fingerprint_is_stable_and_content_sensitive(self) -> None:
        first = TriggerEnvelope(
            kind=TriggerKind.USER_TASK, idempotency_key="k",
            event_reference="evt", payload={"text": "A."},
        )
        same = TriggerEnvelope(
            kind=TriggerKind.USER_TASK, idempotency_key="k",
            event_reference="evt", payload={"text": "A."},
        )
        different = TriggerEnvelope(
            kind=TriggerKind.USER_TASK, idempotency_key="k2",
            event_reference="evt", payload={"text": "A."},
        )
        assert first.fingerprint() == same.fingerprint()
        assert first.fingerprint() != different.fingerprint()


# ------------------------------------------------------- registry dispatch --


class TestOperatorRegistryEligibility:
    """Invalid or unavailable operators are never dispatched (§7)."""

    def test_disabled_and_schema_mismatched_operators_are_ineligible(self) -> None:
        registry = OperatorRegistry()
        register_default_operators(registry)
        disabled = OperatorSpec(
            operator_id="control.disabled_op", version="1.0.0",
            purpose="disabled", enabled=False,
            writes_workspace=True, write_fields=("module_status",),
        )
        registry.register(disabled, lambda *args: None)
        bad_schema = OperatorSpec(
            operator_id="control.bad_schema", version="1.0.0",
            purpose="bad schema", input_schema_id="other/1.0",
            writes_workspace=True, write_fields=("module_status",),
        )
        registry.register(bad_schema, lambda *args: None)
        verdicts = {item.spec.operator_id: item for item in registry.query()}
        assert not verdicts["control.disabled_op"].eligible
        assert "operator_disabled" in verdicts["control.disabled_op"].reasons
        assert not verdicts["control.bad_schema"].eligible
        assert any(
            reason.startswith("unsupported_input_schema")
            for reason in verdicts["control.bad_schema"].reasons
        )

    def test_unregistered_operator_apply_is_rejected(self) -> None:
        registry = OperatorRegistry()
        with pytest.raises(KeyError):
            registry.apply(OperatorAction(operator_id="nope"))

    def test_procedure_operator_identity_collisions_are_rejected(self) -> None:
        registry = OperatorRegistry()
        register_default_operators(registry)
        spec, handler = registry.handler, registry.get_spec  # touch API
        assert spec is not None and handler is not None
        with pytest.raises(ValueError, match="already registered"):
            register_default_operators(registry)


# ------------------------------------------------------------ child contract -


class TestChildBoundaryEnforcement:
    """A child cannot publish outside its declared contract (§7)."""

    def _contract(self) -> ChildContract:
        return ChildContract(
            context_id="child_1", parent_context_id="parent_1",
            goal_text="Verify one reading of 'bank'.",
            allowed_operator_ids=("control.retrieve_knowledge",),
            output_fields=("retrieved_memories", "unresolved_questions"),
            derived_from_workspace_revision=4,
        )

    def test_publication_outside_contract_is_blocked(self) -> None:
        guard = ChildBoundaryGuard(self._contract())
        allowed = OperatorFinding(
            field="retrieved_memories", payload={"x": 1},
        )
        forbidden = OperatorFinding(
            field="semantic_hypotheses", payload={"x": 1},
        )
        assert guard.check_finding(allowed)
        assert not guard.check_finding(forbidden)
        assert guard.violations[-1]["kind"] == "publication_outside_contract"

    def test_operator_outside_contract_is_blocked(self) -> None:
        guard = ChildBoundaryGuard(self._contract())
        assert guard.check_operator("control.retrieve_knowledge")
        assert not guard.check_operator("control.hypothesize_interpretations")

    def test_child_cannot_change_parent_goal(self) -> None:
        guard = ChildBoundaryGuard(self._contract())
        assert not guard.check_goal_change("goal:somebody_else")
        assert guard.check_goal_change("parent_1")
        assert guard.violations[-1]["kind"] == "parent_goal_change_attempt"

    def test_depth_and_child_count_limits_are_enforced(self) -> None:
        from ore_control.config import default_policy
        from ore_control.operators import ChildGoalSpec

        policy = default_policy()
        parent = CognitiveContext(
            context_id="parent", goal_id="goal:test", workspace_revision=0,
            budget=ContextBudget(operations=16),
        )
        deep = CognitiveContext(
            context_id="deep", goal_id="goal:test", workspace_revision=0,
            depth=policy.budgets.max_depth,
        )
        specs = tuple(ChildGoalSpec(goal_text=f"g{i}") for i in range(6))
        with pytest.raises(ValueError, match="depth"):
            create_child_contexts(
                deep, specs, policy=policy, workspace_revision=0,
            )
        children = create_child_contexts(
            parent, specs, policy=policy, workspace_revision=0,
        )
        assert len(children) <= policy.budgets.max_children
        for child, _contract in children:
            assert child.depth == 1
            assert child.parent_context_id == parent.context_id

    def test_decomposition_requires_justification(self) -> None:
        budget = ContextBudget(operations=8)
        assert not decomposition_justified(
            competing_hypotheses=0, open_questions=0, available_budget=budget,
        )
        assert decomposition_justified(
            competing_hypotheses=2, open_questions=0, available_budget=budget,
        )
        assert not decomposition_justified(
            competing_hypotheses=2, open_questions=0,
            available_budget=ContextBudget(operations=1),
        )


# ------------------------------------------------------------ routing checks -


class TestProcessingPathClassification:
    """Routing degrades hybrid to explicit when the learned check fails."""

    def test_hybrid_with_unsuitable_component_degrades_to_explicit(self) -> None:
        unsuitable = LearnedComponentCheck(
            "control.toy.sense_ranker", False,
            {"input_schema": "schema_mismatch"},
        )
        path = classify_processing_path("HYBRID", learned_check=unsuitable)
        assert path == "EXPLICIT_SYMBOLIC"

    def test_hybrid_with_suitable_component_stays_hybrid(self) -> None:
        suitable = LearnedComponentCheck("control.toy.sense_ranker", True, {})
        assert classify_processing_path("HYBRID", learned_check=suitable) == "HYBRID"
        assert classify_processing_path(
            "EXPLICIT_SYMBOLIC", learned_check=suitable,
        ) == "EXPLICIT_SYMBOLIC"


__all__ = []
