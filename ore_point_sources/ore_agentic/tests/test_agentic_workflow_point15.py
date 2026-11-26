"""Acceptance tests for goal intake, graph compilation, state, recovery (Point 15).

Covers specification acceptance tests 1-7 and 17-23: valid goal intake,
ambiguous high-impact clarification, prohibited rejection, idempotent reuse,
typed task graphs, cycle rejection, state-machine validation, branch records,
join semantics, checkpoints, resume revalidation, cancellation, compensation
and duplicate-side-effect prevention.
"""
from __future__ import annotations

import sqlite3

import pytest

from ore_agentic import (
    AgenticDatabase,
    CheckpointIntegrityError,
    CycleError,
    GoalRequest,
    InvalidTransitionError,
    JoinPolicy,
    JoinSemantics,
    LeaseManager,
    OutcomeStatus,
    ReconciliationRequiredError,
    RequestRejection,
    ResourceVector,
    RetryPolicy,
    SideEffectClass,
    TaskGraph,
    TaskOutcomeStatus,
    TaskResult,
    TaskSpec,
    TaskState,
    WorkflowStateMachine,
    WorkflowStatus,
    WorkflowStore,
    evaluate_branch,
)
from ore_agentic.delegation.supervisor import Supervisor
from ore_agentic.delegation.workers import TransientTaskError
from ore_agentic.workflow.side_effects import (
    CompensationNotAuthorizedError,
    CompensationNotDeclaredError,
    EffectStatus,
    SideEffectLedger,
)

from agentic_point15_helpers import (
    CAP_A,
    CAP_B,
    make_engine,
    make_manifest,
    make_request,
    make_subgoal,
)


# ------------------------------------------------------------- intake (1-4)

def test_valid_goal_receives_unique_workflow_id_and_immutable_request():
    """Acceptance 1: valid goal receives unique workflow ID and immutable reference."""
    engine = make_engine()
    first = engine.run(make_request())
    second = engine.run(make_request())
    assert first.workflow_id != second.workflow_id
    assert first.request["objective"] == "Test objective: analyze the provided evidence."
    assert first.request["requester"] == "test-requester"
    assert first.status.value == "succeeded"
    assert first.outcome_status is OutcomeStatus.COMPLETED
    assert len(first.trace) >= 50


def test_ambiguous_high_impact_intent_requests_clarification():
    """Acceptance 2: ambiguous high-impact intent requests clarification."""
    request = make_request(
        flagged_ambiguities=("Which dataset should be analyzed?",),
        subgoals=(make_subgoal("alpha", side_effect_class=(
            SideEffectClass.EXTERNALLY_VISIBLE)),),
    )
    result = make_engine().run(request)
    assert result.status.value == "waiting_clarification"
    assert result.clarification_questions == ("Which dataset should be analyzed?",)
    assert result.task_results == {}  # nothing was dispatched


def test_prohibited_operation_is_rejected_before_dispatch():
    """Acceptance 3: prohibited operation is rejected before dispatch."""
    from ore_agentic import PolicyRule

    engine = make_engine(policy_rules=(
        PolicyRule(principal_id="*", action_id="agentic.workflow.execute",
                   effect="deny", priority=10,
                   description="test prohibition"),))
    result = engine.run(make_request())
    assert result.status.value == "rejected"
    assert result.task_results == {}
    assert any("prohibited" in entry["status"] or "prohibited" in entry["detail"]
               for entry in result.trace if entry["step"] == 3)


def test_duplicate_idempotency_key_reuses_workflow_without_duplicate_side_effect():
    """Acceptance 4: duplicate idempotency key causes no duplicate side effect."""
    effects: list[str] = []

    def handler(envelope):
        envelope.effects.execute(
            "publish", {"n": 1}, lambda: effects.append("published"))
        return {"value": "done", "evidence": ["e1"]}

    from ore_agentic import LocalWorker

    engine = make_engine()
    engine.register_worker(
        make_manifest("fx-worker", (CAP_A, CAP_B)),
        LocalWorker("fx-worker", "1.0.0", (CAP_A, CAP_B), handler))
    request = make_request(
        subgoals=(make_subgoal("alpha", inputs={
            "value": "v", "permitted_tools": ("publish",)}),),
        idempotency_key="idem-1")
    first = engine.run(request)
    assert first.status.value == "succeeded"
    assert effects == ["published"]
    second = engine.run(request)
    assert second.status.value == "reused"
    assert second.reused_from == first.workflow_id
    assert effects == ["published"]  # no duplicate side effect


def test_side_effect_executed_once_across_transient_retries():
    """Acceptance 4b: retries cannot duplicate externally visible side effects."""
    effects: list[str] = []
    attempts = {"n": 0}

    def handler(envelope):
        envelope.effects.execute(
            "publish", {"n": 1}, lambda: effects.append("published"))
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TransientTaskError("crash after effect")
        return {"value": "done", "evidence": ["e1"]}

    from ore_agentic import LocalWorker

    engine = make_engine()
    engine.register_worker(
        make_manifest("retry-fx-worker", (CAP_A, CAP_B)),
        LocalWorker("retry-fx-worker", "1.0.0", (CAP_A, CAP_B), handler))
    request = make_request(
        subgoals=(make_subgoal("alpha", inputs={
            "value": "v", "permitted_tools": ("publish",)}),))
    result = engine.run(request)
    assert result.status.value == "succeeded"
    assert result.task_results["task-alpha"].attempts == 2
    assert effects == ["published"]
    assert engine.ledger.deduplicated_hits == 1


# --------------------------------------------------- graph compilation (5-6)

def test_task_graph_has_valid_schemas_and_dependencies():
    """Acceptance 5: task graph has valid schemas and dependencies."""
    graph = TaskGraph()
    graph.add(TaskSpec(task_id="t1", name="first", capability=CAP_A,
                       output_schema={"type": "object",
                                      "properties": {"value": {"type": "string"}}}))
    graph.add(TaskSpec(task_id="t2", name="second", capability=CAP_B,
                       depends_on=("t1",)))
    order = graph.validate()
    assert order.index("t1") < order.index("t2")
    assert graph.frontier(()) == ("t1",)
    assert graph.frontier(("t1",)) == ("t2",)
    assert graph.critical_path_seconds() > 0.0
    assert len(graph.plan_hash()) == 16


def test_unbounded_dependency_cycle_is_rejected():
    """Acceptance 6: unbounded dependency cycle is rejected."""
    graph = TaskGraph()
    graph.add(TaskSpec(task_id="a", name="a", capability=CAP_A, depends_on=("b",)))
    graph.add(TaskSpec(task_id="b", name="b", capability=CAP_A, depends_on=("a",)))
    with pytest.raises(CycleError):
        graph.validate()
    request = make_request(subgoals=(
        make_subgoal("alpha", depends_on=("beta",)),
        make_subgoal("beta", depends_on=("alpha",)),
    ))
    result = make_engine().run(request)
    assert result.status.value == "rejected"
    assert any("cycle" in (entry.get("detail") or "")
               for entry in result.trace if entry["step"] == 5)


def test_malformed_or_empty_request_is_rejected():
    """Acceptance 1b: malformed requests are rejected at intake (Step 1)."""
    with pytest.raises(ValueError):
        GoalRequest(objective="   ", requester="test-requester")
    with pytest.raises(RequestRejection):
        engine = make_engine()
        engine.run("not-a-goal-request")  # type: ignore[arg-type]


# ------------------------------------------------------- state machine (17)

def test_invalid_state_transition_is_rejected():
    """Acceptance 17: invalid state transition is rejected."""
    machine = WorkflowStateMachine("wf-test")
    machine.announce("t1")
    with pytest.raises(InvalidTransitionError):
        machine.transition("t1", TaskState.SUCCEEDED)
    machine.transition("t1", TaskState.READY)
    machine.transition("t1", TaskState.LEASED)
    machine.transition("t1", TaskState.RUNNING)
    machine.transition("t1", TaskState.SUCCEEDED)
    with pytest.raises(InvalidTransitionError):
        machine.transition("t1", TaskState.RUNNING)
    with pytest.raises(InvalidTransitionError):
        machine.transition_workflow(WorkflowStatus.COMPILED)


# ------------------------------------------------------------- branch (18)

def test_branch_records_predicate_inputs_and_rule_version():
    """Acceptance 18: branch records predicate inputs and rule version."""
    taken = evaluate_branch("ready", {"ready": True})
    assert taken.outcome == "taken"
    assert taken.rule_version == "branch-v1"
    assert taken.inputs == {"ready": True}
    missing = evaluate_branch("ready", {"other": 1})
    assert missing.outcome == "clarification"
    unknown = evaluate_branch("ready", {"ready": None})
    assert unknown.outcome == "clarification"
    engine = make_engine()
    result = engine.run(make_request())
    step18 = [entry for entry in result.trace if entry["step"] == 18]
    assert step18 and "branch-v1" in step18[0]["detail"]


# --------------------------------------------------------------- joins (19)

def test_join_implements_all_any_quorum_semantics():
    """Acceptance 19: join implements all/any/quorum semantics."""
    from ore_agentic import AgentModuleRegistry

    supervisor = Supervisor(AgentModuleRegistry(), LeaseManager())
    ok = TaskResult(task_id="x", worker_id="w",
                    status=TaskOutcomeStatus.SUCCEEDED)
    bad = TaskResult(task_id="y", worker_id="w",
                     status=TaskOutcomeStatus.FAILED)

    join_all = TaskSpec(task_id="j", name="j", capability="reduce",
                        depends_on=("x", "y"),
                        join=JoinPolicy(JoinSemantics.ALL))
    result_all = supervisor.join(join_all, {"x": ok, "y": bad})
    assert result_all.status is TaskOutcomeStatus.FAILED
    assert result_all.payload["dissent"] == ["y"]  # dissent preserved

    join_any = TaskSpec(task_id="j", name="j", capability="reduce",
                        depends_on=("x", "y"),
                        join=JoinPolicy(JoinSemantics.ANY))
    assert supervisor.join(join_any, {"x": ok, "y": bad}).succeeded

    join_quorum = TaskSpec(task_id="j", name="j", capability="reduce",
                           depends_on=("x", "y"),
                           join=JoinPolicy(JoinSemantics.QUORUM, quorum=0.5))
    assert supervisor.join(join_quorum, {"x": ok, "y": bad}).succeeded

    join_weighted = TaskSpec(
        task_id="j", name="j", capability="reduce", depends_on=("x", "y"),
        join=JoinPolicy(JoinSemantics.WEIGHTED_QUORUM, quorum=0.8,
                        weights={"x": 0.9, "y": 0.1}))
    assert supervisor.join(join_weighted, {"x": ok, "y": bad}).succeeded

    join_missing = TaskSpec(task_id="j", name="j", capability="reduce",
                            depends_on=("x", "z"),
                            join=JoinPolicy(JoinSemantics.ALL))
    result_missing = supervisor.join(join_missing, {"x": ok})
    assert result_missing.payload["missing"] == ["z"]  # missing != negative


# -------------------------------------------------- checkpoints (20-21)

def test_checkpoint_restores_workflow_after_restart(tmp_path):
    """Acceptance 20: checkpoint restores workflow after restart."""
    from ore_agentic import CheckpointManager

    database = AgenticDatabase(tmp_path / "agentic.sqlite3")
    store = WorkflowStore(database)
    machine = WorkflowStateMachine("wf-ckpt")
    machine.announce("t1")
    machine.transition("t1", TaskState.READY)
    manager = CheckpointManager(store, max_retained=4)
    manager.capture(
        workflow_id="wf-ckpt", plan_hash="hash-1", plan_version="plan-1",
        workspace_revision=3, task_states=machine.snapshot(),
        completed_task_ids=("t1",), pending_task_ids=(),
        results={"t1": TaskResult(task_id="t1", worker_id="w",
                                  status=TaskOutcomeStatus.SUCCEEDED)},
        lease_epochs={"t1": 1})
    # Simulate a restart with a fresh manager over the same database.
    restarted = CheckpointManager(WorkflowStore(AgenticDatabase(
        tmp_path / "agentic.sqlite3")), max_retained=4)
    restored = restarted.latest("wf-ckpt")
    assert restored is not None
    assert restored.workspace_revision == 3
    assert restored.completed_task_ids == ("t1",)
    assert restored.plan_hash == "hash-1"
    machine2 = WorkflowStateMachine("wf-ckpt")
    machine2.restore(restored.task_states)
    assert machine2.task_state("t1") is TaskState.READY


def test_resume_revalidates_permissions_and_preconditions(tmp_path):
    """Acceptance 21: resume revalidates permissions and preconditions."""
    from ore_agentic import CheckpointManager

    database = AgenticDatabase(tmp_path / "agentic.sqlite3")
    store = WorkflowStore(database)

    manager = CheckpointManager(store)
    manager.capture(
        workflow_id="wf-resume", plan_hash="hash-a", plan_version="plan-1",
        workspace_revision=0, task_states={"t1": "pending"},
        completed_task_ids=(), pending_task_ids=("t1",),
        results={}, lease_epochs={})
    with pytest.raises(CheckpointIntegrityError):
        manager.restore(store.list_checkpoints("wf-resume")[-1],
                        expected_plan_hash="hash-b")
    with pytest.raises(CheckpointIntegrityError):
        manager.restore(store.list_checkpoints("wf-resume")[-1],
                        expected_workflow="wf-other")
    # Tamper with the stored snapshot: integrity verification must fail.
    with sqlite3.connect(tmp_path / "agentic.sqlite3") as connection:
        connection.execute(
            "UPDATE agentic_checkpoints SET snapshot_json = '{\"tampered\": true}' "
            "WHERE workflow_id = 'wf-resume'")
    with pytest.raises((CheckpointIntegrityError, ValueError)):
        manager.restore(store.list_checkpoints("wf-resume")[-1])


# ---------------------------------------------------- cancellation (22-23)

def test_cancellation_prevents_new_dispatch():
    """Acceptance 22: cancellation prevents new dispatch."""
    from ore_agentic import LocalWorker

    engine = make_engine()
    engine.register_worker(
        make_manifest("cancel-worker", (CAP_A, CAP_B)),
        LocalWorker("cancel-worker", "1.0.0", (CAP_A, CAP_B),
                    lambda envelope: (_cancel_from(envelope, engine),
                                      {"value": "cancelled", "evidence": ["e1"]})[1]))
    request = make_request(subgoals=(
        make_subgoal("alpha"),
        make_subgoal("beta", CAP_B, depends_on=("alpha",)),
        make_subgoal("gamma", depends_on=("beta",)),
    ))
    result = engine.run(request)
    assert result.status.value == "cancelled"
    assert list(result.task_results) == ["task-alpha"]
    assert "task-beta" in result.outcomes.cancelled_task_ids
    assert "task-gamma" in result.outcomes.cancelled_task_ids
    # No lease was ever issued for the undispatched tasks.
    assert set(engine.leases.epochs()) == {"task-alpha"}


def _cancel_from(envelope, engine) -> None:
    """Cancel the running workflow from inside a worker (Step 22)."""
    engine.cancel(envelope.lease.workflow_id, reason="worker-requested cancel")


def test_compensation_occurs_only_when_declared_and_authorized():
    """Acceptance 23: compensation occurs only when declared and authorized."""
    ledger = SideEffectLedger()
    compensated: list[str] = []
    ledger.execute_once("k1", "publish", lambda: "published")
    with pytest.raises(CompensationNotDeclaredError):
        ledger.compensate("k1", authorized=True)
    ledger.declare_compensation("publish", lambda receipt: compensated.append(
        receipt.idempotency_key))
    with pytest.raises(CompensationNotAuthorizedError):
        ledger.compensate("k1", authorized=False)
    assert compensated == []
    ledger.compensate("k1", authorized=True)
    assert compensated == ["k1"]


# ------------------------------------------------- side-effect safety (23-24)

def test_uncertain_side_effect_is_reconciled_before_retry():
    """Acceptance 24: uncertain side effect is reconciled before retry."""
    ledger = SideEffectLedger()
    executions: list[str] = []
    ledger.execute_once("k-uncertain", "send", lambda: executions.append("run"),
                        outcome_status=EffectStatus.UNCERTAIN)
    assert executions == ["run"]
    with pytest.raises(ReconciliationRequiredError):
        ledger.execute_once("k-uncertain", "send", lambda: executions.append("run2"))
    receipt = ledger.reconcile("k-uncertain", confirmed=False)
    assert receipt.status is EffectStatus.FAILED
    ledger.execute_once("k-uncertain", "send", lambda: executions.append("run3"))
    assert executions == ["run", "run3"]


def test_outbox_and_inbox_deduplication_records():
    """Acceptance 23b: outbox/inbox patterns record deduplication (Step 23)."""
    ledger = SideEffectLedger()
    entry = ledger.begin("wf:t1:publish", "publish", "wf", "t1")
    assert ledger.pending_outbox() == (entry,)
    receipt = ledger.commit("wf:t1:publish", "publish", {"ok": True})
    ledger.mark_dispatched(entry.entry_id, receipt)
    assert ledger.pending_outbox() == ()
    assert ledger.seen_message("msg-1") is False
    assert ledger.seen_message("msg-1") is True  # duplicate dropped


def test_resource_vector_budgets():
    """Resource vectors add component-wise and check budget fit (Step 7)."""
    budget = ResourceVector(cpu_units=2.0, memory_mb=100.0, io_units=1.0,
                            network_units=1.0, seconds=10.0)
    first = ResourceVector(cpu_units=1.0, memory_mb=60.0, io_units=0.5,
                           network_units=0.5, seconds=4.0)
    second = ResourceVector(cpu_units=1.0, memory_mb=50.0, io_units=0.5,
                            network_units=0.5, seconds=4.0)
    assert first.fits_within(budget)
    assert not first.add(second).fits_within(budget)


def test_lease_ttl_and_expiry_with_injected_clock():
    """Lease TTL expiry is enforced with an injectable clock (Step 15)."""
    from ore_agentic.contracts import TaskSpec as Spec

    clock = {"now": 100.0}
    manager = LeaseManager(default_ttl_seconds=5.0,
                           clock=lambda: clock["now"])
    task = Spec(task_id="t1", name="t1", capability=CAP_A)
    lease = manager.issue("wf", task, "w1")
    manager.validate(lease, enforce_ttl=True)  # within TTL
    clock["now"] = 200.0
    from ore_agentic import LeaseExpiredError
    with pytest.raises(LeaseExpiredError):
        manager.validate(lease, enforce_ttl=True)


def test_workflow_store_persists_and_transitions(tmp_path):
    """WorkflowStore persists plans, transitions and results transactionally."""
    database = AgenticDatabase(tmp_path / "agentic.sqlite3")
    store = WorkflowStore(database)
    spec = {
        "workflow_id": "wf-store-1",
        "request_id": "req-1",
        "request_digest": "digest-1",
        "objective": "objective",
        "plan_version": "plan-1",
        "plan_hash": "hash-1",
        "tasks": {"t1": {"capability": CAP_A, "side_effect_class": "read_only"}},
    }
    assert store.create(spec, idempotency_key="idem-store") == "wf-store-1"
    with pytest.raises(ValueError):
        store.create({**spec, "workflow_id": "wf-store-2"},
                     idempotency_key="idem-store")
    store.transition("wf-store-1", "dispatch", new_status="running",
                     task_id="t1", task_state="leased", worker_id="w1",
                     attempts=1)
    row = store.get("wf-store-1")
    assert row["status"] == "running"
    tasks = store.tasks("wf-store-1")
    assert tasks[0]["state"] == "leased"
    assert store.find_by_idempotency_key("idem-store")["workflow_id"] == "wf-store-1"


def test_retry_policy_bounds_are_validated():
    """Retry policies validate their bounds (Step 16)."""
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
    policy = RetryPolicy()
    assert policy.max_attempts >= 1
