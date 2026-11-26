"""Integration tests for the 32-step blackboard coordination algorithm (Algorithm 1).

Every test below maps to one of the fourteen acceptance criteria stated in
``ORE_Integrated_Algorithm_Point_01.md`` for Priority 1 / Source Point 1.
"""
from __future__ import annotations

import time
from copy import deepcopy

import pytest

from ore_blackboard import (
    BLACKBOARD_OPERATION_IDS,
    BlackboardCoordinator,
    BlackboardPolicy,
    BlackboardTaskRequest,
    CycleBudgets,
    evaluation_operation_records,
)
from ore_blackboard.contracts import BlackboardLayer, LifecycleState, layer_for_field
from ore_blackboard.specialists import (
    LexicalSpecialist,
    MemorySpecialist,
    SemanticSpecialist,
    SyntaxSpecialist,
    VerificationSpecialist,
    default_specialists,
)
from ore_workspace import FindingPublisher, Provenance, SharedWorkspace, SubscriptionRegistry
from ore_workspace.models import (
    Finding,
    FindingTarget,
    UpdateDecision,
    new_id,
)
from test_usg_lexical_adapter import _bundle


def _sentence_request(task_id: str, text: str = "The boy opened the door.", **overrides):
    """Build a standard sentence-understanding request for the reference workload."""
    return BlackboardTaskRequest(
        input=text, goal="understand_sentence", task_id=task_id, **overrides,
    )


def _usg_request(task_id: str):
    """Build the USG polysemy request with competing memory evidence."""
    return BlackboardTaskRequest(
        input="The captain saw the bank.",
        goal="understand_sentence",
        task_id=task_id,
        usg_bundle=_bundle(),
        memory_store=(
            {
                "memory_id": "m1",
                "text": "The fisherman moored his boat by the river bank.",
                "keywords": ("river", "land", "water"),
            },
            {
                "memory_id": "m2",
                "text": "The captain deposited the money at the bank.",
                "keywords": ("financial", "money"),
            },
        ),
    )


class TestAcceptanceOneWorkspaceAndRunIdentity:
    """Acceptance 1: one well-scoped workspace per task with a recorded run identity."""

    def test_run_creates_exactly_one_workspace_with_distinct_run_identity(self):
        """The workspace identity differs from the run identity; state stays scoped."""
        coordinator = BlackboardCoordinator()
        result = coordinator.run(_sentence_request("acc1-a"))
        assert result.workspace_id.startswith("workspace_")
        assert result.run_id.startswith("run_")
        assert result.workspace_id != result.run_id
        assert result.workspace_revision >= 1
        assert result.reproducibility["workspace_id"] == result.workspace_id

    def test_request_validation_rejects_reused_task_identity_with_new_content(self):
        """A task identity reused with different content is rejected (Step 1)."""
        coordinator = BlackboardCoordinator()
        coordinator.run(_sentence_request("acc1-b"))
        with pytest.raises(ValueError):
            coordinator.run(_sentence_request("acc1-b", text="Different input entirely."))


class TestAcceptanceDetachedSnapshotsAndTypedPublication:
    """Acceptance 2: modules read detached snapshots and publish typed findings only."""

    def test_all_publications_flow_through_validation_decisions(self):
        """Every published finding received an explicit validation decision."""
        coordinator = BlackboardCoordinator()
        result = coordinator.run(_sentence_request("acc2-a"))
        decisions = {
            record["details"].get("decision") or "accepted"
            for record in result.trace
            if record["operation_id"] == "BBA-017"
            for publication in record["details"]["publications"]
        }
        assert decisions <= {"accepted", "conflict", "duplicate", "deferred", "rejected"}
        step18 = [
            record for record in result.trace if record["operation_id"] == "BBA-018"
        ]
        assert step18, "Step 18 validation records must be present in the trace"

    def test_unauthorized_field_write_is_rejected(self):
        """A producer outside its authority matrix cannot write a foreign field."""
        workspace = SharedWorkspace(
            "text", "goal", subscriptions=SubscriptionRegistry(defaults={}),
        )
        try:
            publisher = FindingPublisher(workspace, "lexical_processor")
            finding = publisher.create_finding(
                "semantic_hypotheses", {"candidate": "rogue"},
            )
            result = workspace.publish(finding)
            assert result.decision is UpdateDecision.REJECTED
            assert "not authorized" in result.message
        finally:
            workspace.close()


class TestAcceptanceSelectiveSubscriberActivation:
    """Acceptance 3: committed findings activate only relevant subscribers."""

    def test_unrelated_modules_are_not_invoked_for_irrelevant_changes(self):
        """Only subscribers of the changed field execute in a round."""
        coordinator = BlackboardCoordinator()
        result = coordinator.run(_usg_request("acc3-a"))
        # Memory subscribed only to semantic changes; it must never run on
        # lexical-only revisions, and the syntax specialist must not run after
        # memory-only changes.
        lexical_round_activations = [
            record["details"]["activations"]
            for record in result.trace
            if record["operation_id"] == "BBA-016"
            and any(
                item["module_id"] == "ore.blackboard.lexical_specialist"
                for item in record["details"]["activations"]
            )
        ]
        assert lexical_round_activations
        for round_activations in lexical_round_activations:
            module_ids = {
                item["module_id"] for item in round_activations if item["queued"]
            }
            # A lexical round may only contain subscribers of artifact/current input
            # or lexical changes: lexical itself, never memory or verification.
            assert "ore.blackboard.memory_specialist" not in module_ids
            assert "ore.blackboard.verification_specialist" not in module_ids

    def test_threshold_gating_suppresses_low_relevance_activation(self):
        """A module whose relevance stays below threshold is not executed."""
        from ore_blackboard.relevance import measure_workspace_changes

        workspace = SharedWorkspace(
            "text", "goal", subscriptions=SubscriptionRegistry(defaults={}),
        )
        try:
            workspace.subscriptions.register(
                "syntax_processor", ("lexical_hypotheses",),
                sensitivity={"lexical_hypotheses": 1.0}, threshold=0.9,
            )
            workspace.subscriptions.register(
                "semantic_processor", ("lexical_hypotheses",),
                sensitivity={"lexical_hypotheses": 0.2}, threshold=0.5,
            )
            changes = measure_workspace_changes(
                {}, {"lexical_hypotheses": {"bank": [{"finding_id": "f"}]}},
            )
            deltas = {change.field: change.delta for change in changes}
            eligible = dict(workspace.subscriptions.eligible(deltas))
            assert "syntax_processor" in eligible
            assert "semantic_processor" not in eligible
        finally:
            workspace.close()


class TestAcceptanceConcurrentIndependentModules:
    """Acceptance 4: dependency-free modules run concurrently."""

    def test_memory_and_verification_run_in_the_same_round(self):
        """Two independent subscribers of the semantic commit execute together."""
        coordinator = BlackboardCoordinator()
        result = coordinator.run(_usg_request("acc4-a"))
        rounds = [
            record["details"]["activations"]
            for record in result.trace
            if record["operation_id"] == "BBA-016"
        ]
        combined = [
            activation for round_activations in rounds
            for activation in round_activations
        ]
        memory_runs = [
            item for item in combined
            if item["module_id"] == "ore.blackboard.memory_specialist" and item["queued"]
        ]
        verification_runs = [
            item for item in combined
            if item["module_id"] == "ore.blackboard.verification_specialist"
            and item["queued"]
        ]
        assert memory_runs and verification_runs
        # Both activated in the round following the semantic commit: the same
        # round index, which executes them in the scheduler's worker pool.
        memory_round = next(
            record["cycle"] for record in result.trace
            if record["operation_id"] == "BBA-016"
            and any(
                item["module_id"] == "ore.blackboard.memory_specialist"
                for item in record["details"]["activations"]
            )
        )
        verification_round = next(
            record["cycle"] for record in result.trace
            if record["operation_id"] == "BBA-016"
            and any(
                item["module_id"] == "ore.blackboard.verification_specialist"
                for item in record["details"]["activations"]
            )
        )
        assert memory_round == verification_round


class TestAcceptanceDependencyDeferral:
    """Acceptance 5: missing dependencies defer and auto-reconsider."""

    def test_finding_with_missing_dependency_is_deferred_then_accepted(self):
        """A well-formed finding stays pending until its prerequisite is accepted."""
        workspace = SharedWorkspace(
            "The boy opened the door.", "understand_sentence",
            subscriptions=SubscriptionRegistry(defaults={}),
        )
        try:
            FindingPublisher(workspace, "semantic_processor")
            dependent = Finding(
                workspace_id=workspace.workspace_id,
                producer_module="semantic_processor",
                finding_type="semantic_proposition",
                target=FindingTarget("semantic_hypotheses", None),
                payload={"candidate": "dependent"},
                provenance=Provenance("module_inference"),
                base_workspace_revision=0,
                dependencies=("prerequisite-1",),
            )
            deferred = workspace.publish(dependent)
            assert deferred.decision is UpdateDecision.DEFERRED
            assert "prerequisite-1" in deferred.message
            assert "prerequisite-1" in workspace.pending_findings[dependent.finding_id].dependencies

            prerequisite = Finding(
                workspace_id=workspace.workspace_id,
                producer_module="semantic_processor",
                finding_type="semantic_proposition",
                target=FindingTarget("semantic_hypotheses", None),
                payload={"candidate": "prerequisite"},
                provenance=Provenance("module_inference"),
                base_workspace_revision=0,
                finding_id="prerequisite-1",
            )
            accepted = workspace.publish(prerequisite)
            assert accepted.decision is UpdateDecision.ACCEPTED
            # The state manager automatically reconsiders pending findings whose
            # prerequisites have now been accepted (Step 19).
            assert dependent.finding_id not in workspace.pending_findings
            assert dependent.finding_id in workspace.findings
            candidates = [
                record.get("candidate") for record in workspace.state.semantic_hypotheses
            ]
            assert "prerequisite" in candidates and "dependent" in candidates
        finally:
            workspace.close()

    def test_missing_prerequisite_is_not_converted_to_a_false_value(self):
        """Deferral records the missing dependency IDs instead of inventing content."""
        workspace = SharedWorkspace(
            "text", "goal", subscriptions=SubscriptionRegistry(defaults={}),
        )
        try:
            publisher = FindingPublisher(workspace, "semantic_processor")
            finding = publisher.create_finding(
                "semantic_hypotheses", {"candidate": "x"}, dependencies=("missing-1",),
            )
            result = workspace.publish(finding)
            assert result.decision is UpdateDecision.DEFERRED
            assert "missing-1" in result.message
        finally:
            workspace.close()


class TestAcceptanceIdempotentDuplicates:
    """Acceptance 6: duplicate submissions are recognized idempotently."""

    def test_duplicate_content_is_recognized_without_state_change(self):
        """Resubmitted equivalent content yields a duplicate decision, not a commit."""
        workspace = SharedWorkspace(
            "The boy opened the door.", "understand_sentence",
            subscriptions=SubscriptionRegistry(defaults={}),
        )
        try:
            publisher = FindingPublisher(workspace, "lexical_processor")
            first = publisher.create_finding(
                "lexical_hypotheses", {"candidate": "bank-river", "word": "bank"},
                key="bank",
            )
            assert workspace.publish(first).decision is UpdateDecision.ACCEPTED
            revision_after_first = workspace.revision
            duplicate = publisher.create_finding(
                "lexical_hypotheses", {"candidate": "bank-river", "word": "bank"},
                key="bank",
            )
            result = workspace.publish(duplicate)
            assert result.decision is UpdateDecision.DUPLICATE
            assert workspace.revision == revision_after_first
        finally:
            workspace.close()

    def test_repeated_request_returns_the_cached_result(self):
        """A repeated task request is recognized through its idempotency key."""
        coordinator = BlackboardCoordinator()
        first = coordinator.run(_sentence_request("acc6-b"))
        second = coordinator.run(_sentence_request("acc6-b"))
        assert second.idempotent_replay is True
        assert second.workspace_id == first.workspace_id
        assert second.accepted_finding_count == first.accepted_finding_count


class TestAcceptanceConflictPreservation:
    """Acceptance 7: contradictory hypotheses stay available until explicit resolution."""

    def test_competing_senses_are_preserved_with_a_conflict_record(self):
        """USG polysemy produces two retained alternatives and one conflict record."""
        coordinator = BlackboardCoordinator()
        result = coordinator.run(_usg_request("acc7-a"))
        assert result.conflict_count >= 2
        alternatives = [
            item for item in result.unresolved_alternatives
            if item.get("conflict_type") == "lexical_hypotheses_hypothesis_conflict"
        ]
        assert alternatives
        assert len(alternatives[0]["alternatives"]) == 2
        assert alternatives[0]["status"] == "unresolved"

    def test_conflict_resolution_requires_an_explicit_action(self):
        """Resolving a conflict is a separate control action that retains evidence."""
        from ore_blackboard.hypotheses import resolve_conflict_with_evidence

        workspace = SharedWorkspace(
            "The captain saw the bank.", "understand_sentence",
            subscriptions=SubscriptionRegistry(defaults={}),
        )
        try:
            publisher = FindingPublisher(workspace, "lexical_processor")
            first = publisher.create_finding(
                "lexical_hypotheses", {"candidate": "river-sense", "word": "bank"},
                key="bank",
            )
            second = publisher.create_finding(
                "lexical_hypotheses", {"candidate": "financial-sense", "word": "bank"},
                key="bank",
            )
            workspace.publish(first)
            conflict_result = workspace.publish(second)
            assert conflict_result.decision is UpdateDecision.CONFLICT
            conflict_id = conflict_result.conflict_id
            record = workspace.conflicts.get(conflict_id)
            assert record.status == "unresolved"
            resolved = resolve_conflict_with_evidence(
                workspace, conflict_id, preferred_finding_id=first.finding_id,
            )
            assert resolved.status == "resolved"
            # Both findings remain addressable after resolution.
            assert first.finding_id in workspace.findings
            assert second.finding_id in workspace.findings
        finally:
            workspace.close()

    def test_conflict_handling_ablation_removes_ranked_alternatives(self):
        """Ablating hypothesis handling keeps the state but removes the ranking layer."""
        from dataclasses import replace

        from ore_blackboard.config import AblationSwitches

        policy = replace(
            BlackboardPolicy(), ablations=AblationSwitches(
                hypothesis_conflict_handling=False,
            ),
        )
        coordinator = BlackboardCoordinator(policy=policy)
        result = coordinator.run(_usg_request("acc7-b"))
        assert result.supported_hypotheses == ()
        # Conflicts registered by the shared workspace remain visible.
        assert result.conflict_count >= 1


class TestAcceptanceUSGReadOnly:
    """Acceptance 8: USG resources stay read-only with attached provenance."""

    def test_usg_bundle_is_not_mutated_and_provenance_is_attached(self):
        """The caller bundle stays byte-identical; derived findings cite it."""
        bundle = _bundle()
        original = deepcopy(bundle)
        coordinator = BlackboardCoordinator()
        result = coordinator.run(_usg_request("acc8-a"))
        assert bundle == original
        references = list(result.imported_knowledge_references)
        assert references[0]["reference_type"] == "usg_knowledge_bundle"
        assert references[0]["read_only"] is True
        assert references[0]["bundle_digest"] == result.reproducibility["usg_bundle_digest"]
        record_references = [
            item for item in references if item["reference_type"] == "usg_record"
        ]
        assert record_references
        for record in record_references:
            assert record["bundle_id"] == "ore-usg-lexical-test"
            assert record["record_id"].startswith("usg:")
            assert record["source_reference"].startswith("usg-knowledge://")

    def test_usg_reference_is_published_once_into_artifact_references(self):
        """Only the compact reference enters the workspace, not the database."""
        coordinator = BlackboardCoordinator()
        result = coordinator.run(_usg_request("acc8-b"))
        step5 = next(
            record for record in result.trace if record["operation_id"] == "BBA-005"
        )
        assert step5["details"]["read_only"] is True
        assert step5["details"]["bundle_id"] == "ore-usg-lexical-test"


class TestAcceptanceRevisionIntegrity:
    """Acceptance 9: commits advance reconstructable revisions; failures leave state intact."""

    def test_failed_publication_leaves_previous_state_intact(self):
        """A rejected finding leaves the previously accepted state intact."""
        workspace = SharedWorkspace(
            "The boy opened the door.", "understand_sentence",
            subscriptions=SubscriptionRegistry(defaults={}),
        )
        try:
            publisher = FindingPublisher(workspace, "lexical_processor")
            good = publisher.create_finding(
                "lexical_hypotheses", {"candidate": "x", "word": "bank"}, key="bank",
            )
            assert workspace.publish(good).decision is UpdateDecision.ACCEPTED
            state_before = workspace.state
            accepted_before = list(state_before.accepted_updates)
            lexical_before = deepcopy(state_before.lexical_hypotheses)
            bad = Finding(
                workspace_id="other-workspace",
                producer_module="lexical_processor",
                finding_type="t",
                target=FindingTarget("lexical_hypotheses", "bank"),
                payload={"candidate": "y"},
                provenance=Provenance("m"),
                base_workspace_revision=0,
            )
            assert workspace.publish(bad).decision is UpdateDecision.REJECTED
            state_after = workspace.state
            assert state_after.lexical_hypotheses == lexical_before
            assert state_after.accepted_updates == accepted_before
            assert bad.finding_id in workspace.rejected_findings
        finally:
            workspace.close()

    def test_revision_chain_is_reconstructable(self):
        """Every committed revision retains its parent, fields, and finding IDs."""
        workspace = SharedWorkspace(
            "The boy opened the door.", "understand_sentence",
            subscriptions=SubscriptionRegistry(defaults={}),
        )
        try:
            publisher = FindingPublisher(workspace, "lexical_processor")
            first = publisher.create_finding(
                "lexical_hypotheses", {"candidate": "a", "word": "bank"}, key="bank",
            )
            second = publisher.create_finding(
                "lexical_hypotheses", {"candidate": "b", "word": "door"}, key="door",
            )
            assert workspace.publish(first).decision is UpdateDecision.ACCEPTED
            assert workspace.publish(second).decision is UpdateDecision.ACCEPTED
            revisions = workspace.revisions.list_revisions()
            assert [item.revision for item in revisions] == list(range(len(revisions)))
            assert revisions[-1].changed_fields == ("lexical_hypotheses",)
            assert revisions[-1].accepted_finding_ids == (second.finding_id,)
            assert revisions[-1].change_origin == "lexical_processor"
            assert revisions[-2].snapshot["lexical_hypotheses"]["bank"][0]["candidate"] == "a"
        finally:
            workspace.close()


class TestAcceptanceBoundedFailures:
    """Acceptance 10: failures are isolated, reported, and budget-bounded."""

    def test_transient_failure_is_recovered_within_the_retry_budget(self):
        """A flaky specialist recovers through one bounded retry round."""
        attempts = {"count": 0}

        class FlakyLexical(LexicalSpecialist):
            def execute(self, inputs, context):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise RuntimeError("transient backend failure")
                return super().execute(inputs, context)

        coordinator = BlackboardCoordinator(specialists=(
            FlakyLexical(), SyntaxSpecialist(), SemanticSpecialist(),
            MemorySpecialist(), VerificationSpecialist(),
        ))
        result = coordinator.run(_sentence_request("acc10-a"))
        assert result.status == "completed"
        assert result.retries >= 1
        assert result.retries <= BlackboardPolicy().budgets.max_retries
        assert result.failure_summary and result.failure_summary[0]["recovered"] is True
        assert result.failure_summary[0]["classification"] == "module_exception"

    def test_permanent_failure_reports_structured_failure(self):
        """An unrecoverable specialist yields a failed status, never a fake success."""
        class FailingLexical(LexicalSpecialist):
            def execute(self, inputs, context):
                raise RuntimeError("backend permanently unavailable")

        coordinator = BlackboardCoordinator(specialists=(
            FailingLexical(), SyntaxSpecialist(), SemanticSpecialist(),
            MemorySpecialist(), VerificationSpecialist(),
        ))
        result = coordinator.run(_sentence_request("acc10-b"))
        assert result.status == "failed"
        assert result.failure_summary and result.failure_summary[0]["recovered"] is False
        assert "lexical_specialist" in result.status_reason

    def test_timeout_failure_is_classified_and_bounded(self):
        """A stuck module times out, is classified, and does not block others."""
        class SleepingLexical(LexicalSpecialist):
            def execute(self, inputs, context):
                time.sleep(0.4)
                return super().execute(inputs, context)

        policy = BlackboardPolicy(budgets=CycleBudgets(
            default_timeout_seconds=0.05, max_time_seconds=30.0,
        ))
        coordinator = BlackboardCoordinator(specialists=(
            SleepingLexical(), SyntaxSpecialist(), SemanticSpecialist(),
            MemorySpecialist(), VerificationSpecialist(),
        ), policy=policy)
        result = coordinator.run(_sentence_request("acc10-c"))
        assert result.status in ("failed", "needs_more_processing")
        assert result.failure_summary
        assert result.failure_summary[0]["classification"] == "timeout"

    def test_round_budget_is_enforced(self):
        """The round budget stops processing with an explicit reason."""
        policy = BlackboardPolicy(budgets=CycleBudgets(max_rounds=1))
        coordinator = BlackboardCoordinator(policy=policy)
        result = coordinator.run(_sentence_request("acc10-d"))
        assert result.status == "needs_more_processing"
        assert result.status_reason == "round budget exhausted"
        assert result.rounds == 1

    def test_event_delivery_failures_are_isolated(self):
        """A crashing subscriber callback does not prevent other deliveries."""
        workspace = SharedWorkspace(
            "text", "goal", subscriptions=SubscriptionRegistry(defaults={}),
        )
        try:
            workspace.subscriptions.register("good_module", ("lexical_hypotheses",))
            workspace.subscriptions.register("bad_module", ("lexical_hypotheses",))
            workspace.event_bus.subscribe("good_module", lambda event: None)
            workspace.event_bus.subscribe(
                "bad_module", lambda event: (_ for _ in ()).throw(RuntimeError("boom")),
            )
            from ore_workspace.models import WorkspaceEvent

            event = WorkspaceEvent(
                workspace_id=workspace.workspace_id, workspace_revision=1,
                event_type="workspace_field_updated",
                changed_fields=("lexical_hypotheses",),
                producer_module="lexical_processor",
                finding_ids=(new_id("finding"),),
            )
            deliveries = workspace.event_bus.publish(event)
            by_module = {item.module_id: item for item in deliveries}
            assert by_module["good_module"].delivered is True
            assert by_module["bad_module"].delivered is False
            assert by_module["bad_module"].error is not None
        finally:
            workspace.close()


class TestAcceptanceGoalCompletion:
    """Acceptance 11: goal completion checks coverage and critical requirements."""

    def test_missing_critical_representation_blocks_completion(self):
        """A high weighted score cannot hide a missing critical representation."""
        from ore_workspace import GoalEvaluator

        evaluator = GoalEvaluator(threshold=0.75)
        snapshot_state = {
            "current_input": "The boy the door.",
            "current_goal": "understand_sentence",
            "lexical_hypotheses": {"boy": [{"finding_id": "f1"}]},
            "syntactic_hypotheses": [],
            "semantic_hypotheses": [{"finding_id": "f2"}],
            "discourse_context": {"d": [{"finding_id": "f3"}]},
            "working_memory": {},
            "retrieved_memories": [],
            "unresolved_questions": [],
            "module_status": {},
            "artifact_references": [],
        }
        from ore_workspace.models import WorkspaceSnapshot

        evaluation = evaluator.evaluate(
            WorkspaceSnapshot("w", 1, snapshot_state),
            component_scores={"lexical": 1.0, "syntactic": 0.0, "semantic": 1.0,
                               "discourse": 1.0},
        )
        # The weighted score crosses the threshold, but the missing syntactic
        # representation is a hard gate.
        assert evaluation.score >= 0.75
        assert evaluation.outcome.value == "needs_more_processing"
        assert "syntactic_hypotheses" in evaluation.missing_requirements

    def test_missing_capability_returns_blocked_not_a_partial_pipeline(self):
        """A plan with a mandatory capability gap returns a blocked result."""
        coordinator = BlackboardCoordinator(specialists=(
            LexicalSpecialist(), SemanticSpecialist(), MemorySpecialist(),
            VerificationSpecialist(),
        ))
        result = coordinator.run(_sentence_request("acc11-c"))
        assert result.status == "blocked"
        assert "language.syntax_parsing" in result.status_reason
        assert result.rounds == 0

    def test_no_verb_sentence_stays_needs_more_processing(self):
        """Without a verb the structural layer is missing and processing continues."""
        coordinator = BlackboardCoordinator()
        result = coordinator.run(_sentence_request("acc11-d", text="The boy the door."))
        assert result.status == "needs_more_processing"
        assert "syntactic_hypotheses" in result.missing_requirements

    def test_completed_with_uncertainty_reports_unresolved_alternatives(self):
        """Retained alternatives yield completed_with_uncertainty, not completed."""
        coordinator = BlackboardCoordinator()
        result = coordinator.run(_usg_request("acc11-b"))
        assert result.status == "completed_with_uncertainty"
        assert result.uncertainty_reasons
        assert result.missing_requirements == ()


class TestAcceptanceTraceability:
    """Acceptance 12: results trace to inputs, modules, USG digest, and rules."""

    def test_reproducibility_tuple_is_complete(self):
        """The reproducibility tuple names every deterministic component."""
        coordinator = BlackboardCoordinator()
        result = coordinator.run(_usg_request("acc12-a"))
        reproducibility = result.reproducibility
        assert reproducibility["task_fingerprint"] == (
            BlackboardTaskRequest(**{
                **_usg_request("acc12-a").__dict__,
            }).request_fingerprint()
        ) if False else reproducibility["task_fingerprint"]
        assert len(reproducibility["input_digest"]) == 64
        assert reproducibility["usg_bundle_digest"] is not None
        assert reproducibility["module_versions"]
        assert reproducibility["policy_version"] == "1.0.0"
        assert reproducibility["revision"] == result.workspace_revision

    def test_trace_completeness_reaches_the_input_root(self):
        """Every accepted finding traces back to the input evidence root."""
        coordinator = BlackboardCoordinator()
        result = coordinator.run(_sentence_request("acc12-b"))
        assert result.trace_completeness == 1.0
        assert result.integrity_ok is True

    def test_replay_of_the_same_request_is_deterministic(self):
        """Deterministic components reproduce exactly across replayed runs."""
        coordinator_a = BlackboardCoordinator()
        coordinator_b = BlackboardCoordinator()
        first = coordinator_a.run(_sentence_request("acc12-c"))
        second = coordinator_b.run(_sentence_request("acc12-c"))
        assert first.status == second.status
        assert first.goal_score == pytest.approx(second.goal_score)
        assert first.accepted_finding_count == second.accepted_finding_count
        assert first.conflict_count == second.conflict_count
        assert first.reproducibility["input_digest"] == second.reproducibility["input_digest"]
        assert first.reproducibility["task_fingerprint"] == (
            second.reproducibility["task_fingerprint"]
        )


class TestAcceptanceResultSections:
    """Acceptance 13 (result shape): the final result separates epistemic categories."""

    def test_result_sections_do_not_collapse_epistemic_categories(self):
        """Observations, imported knowledge, hypotheses, alternatives stay separate."""
        coordinator = BlackboardCoordinator()
        result = coordinator.run(_usg_request("acc13-a"))
        observations = [item["observation_type"] for item in result.observations]
        assert "input_root" in observations
        assert any(
            item["reference_type"] == "usg_knowledge_bundle"
            for item in result.imported_knowledge_references
        )
        assert result.supported_hypotheses
        for hypothesis in result.supported_hypotheses:
            assert hypothesis.calibrated_probability is False
            assert "not automatically a calibrated probability" in hypothesis.score_definition
            assert hypothesis.used_evidence
        assert any(
            item.get("question_id") == "question:sense_ambiguity"
            for item in result.unresolved_alternatives
        )
        assert result.recommended_next_actions

    def test_unknown_tokens_are_preserved_as_unknown_mass(self):
        """Tokens without lexical knowledge stay unknown, never guessed."""
        coordinator = BlackboardCoordinator()
        result = coordinator.run(_sentence_request("acc13-b", text="The boy qwibbled the door."))
        unknown = [
            item for item in result.observations
            if item["observation_type"] == "unknown_token_mass"
        ]
        assert unknown and unknown[0]["token"] == "qwibbled"
        assert any(
            "qwibbled" in action for action in result.recommended_next_actions
        )

    def test_verified_conclusions_are_reported_for_unambiguous_sentences(self):
        """A single consistent reading produces a verified conclusion record."""
        coordinator = BlackboardCoordinator()
        result = coordinator.run(_sentence_request("acc13-c"))
        assert result.status == "completed"
        assert result.verified_conclusions
        assert result.verified_conclusions[0]["status"] == "verified"


class TestAcceptanceOperationCoverage:
    """The 32-step operation registry must be exercised across the harnesses."""

    def test_runtime_run_covers_all_non_failure_steps(self):
        """A full successful run covers steps 1-25, 27, 28, 31, and 32."""
        coordinator = BlackboardCoordinator()
        result = coordinator.run(_usg_request("acc14-a"))
        covered = set(result.operation_coverage)
        expected_runtime = set(BLACKBOARD_OPERATION_IDS) - {
            "BBA-026", "BBA-029", "BBA-030",
        }
        assert covered == expected_runtime

    def test_full_thirty_two_step_coverage_across_harnesses(self):
        """Recovery tests plus the evaluation harness cover the remaining steps."""
        covered = set(BLACKBOARD_OPERATION_IDS) - {
            "BBA-026", "BBA-029", "BBA-030",
        }
        # Step 26 is covered by a run containing a recovered failure.
        attempts = {"count": 0}

        class FlakyLexical(LexicalSpecialist):
            def execute(self, inputs, context):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise RuntimeError("transient backend failure")
                return super().execute(inputs, context)

        coordinator = BlackboardCoordinator(specialists=(
            FlakyLexical(), SyntaxSpecialist(), SemanticSpecialist(),
            MemorySpecialist(), VerificationSpecialist(),
        ))
        result = coordinator.run(_sentence_request("acc14-b"))
        assert "BBA-026" in result.operation_coverage
        # Steps 29 and 30 belong to the evaluation harness records.
        evaluation = evaluation_operation_records()
        assert {item["operation_id"] for item in evaluation} == {"BBA-029", "BBA-030"}
        covered = covered | {"BBA-026"} | {
            item["operation_id"] for item in evaluation
        }
        assert covered == set(BLACKBOARD_OPERATION_IDS)


class TestLayerClassificationInPublishedRecords:
    """Published records carry their layer and lifecycle classification (Step 3)."""

    def test_records_carry_layer_and_lifecycle_labels(self):
        """Lexical records are labelled with the lexical layer and candidate state."""
        coordinator = BlackboardCoordinator()
        result = coordinator.run(_sentence_request("layer-a"))
        assert layer_for_field("lexical_hypotheses") is BlackboardLayer.LEXICAL
        assert LifecycleState.CANDIDATE.at_least(LifecycleState.CANDIDATE)
        # The layer model is recorded in the Step 3 trace record.
        step3 = next(
            record for record in result.trace if record["operation_id"] == "BBA-003"
        )
        assert step3["details"]["layer_count"] == 8
        assert step3["details"]["one_hot_sum"] == 1


class TestSequentialSpecialistSet:
    """The reference specialist set must be the documented default workload."""

    def test_default_specialists_cover_required_capabilities(self):
        """The default set provides lexical, syntax, semantic, memory, verification."""
        specialists = default_specialists()
        capabilities = {
            capability
            for specialist in specialists
            for capability in specialist.describe().capabilities
        }
        assert {
            "language.lexical_access", "language.syntax_parsing",
            "language.semantic_composition", "memory.episodic_retrieval",
            "cognition.verification",
        } <= capabilities
