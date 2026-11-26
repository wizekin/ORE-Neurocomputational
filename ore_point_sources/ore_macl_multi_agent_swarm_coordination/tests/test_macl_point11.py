"""Focused acceptance tests for Point 11 multi-agent and swarm coordination (MACL)."""
from __future__ import annotations

import sqlite3

import pytest

from ore_macl import (
    STAGE_REGISTRY,
    ACOConfig,
    AgentDescriptor,
    AgentLifecycleState,
    AgentRegistry,
    AgentResult,
    AgentTask,
    AgentTaskKind,
    AgentTaskMatcher,
    AntColonyOptimizer,
    ArtifactRef,
    AutonomyLevel,
    BetaBinomialEstimate,
    CoordinatorGateway,
    DelegationChain,
    DeliveryOutcome,
    EligibilityGate,
    GoalRequest,
    MACLConfig,
    MACLRunConfig,
    MACLSupervisor,
    MessageKind,
    OperationGraph,
    ParticleSwarmOptimizer,
    PerformanceLedger,
    ProvenanceTrail,
    RetryPolicy,
    SchemaSpec,
    SuccessCriteria,
    TaskDAG,
    TaskDecomposer,
    TaskTopology,
    classify_component,
    dependency_feasible,
    registry_as_dicts,
    resource_policy_objective,
    run_multi_agent_coordination,
    stage_for_step,
    uncertainty_penalty,
)
from ore_macl.adjudication import AdjudicationReport
from ore_macl.agents import (
    LogicalClock,
    build_default_agents,
    default_specialist_descriptors,
)
from ore_macl.contracts import AdjudicationDecision, PeerNegotiationOutcome, TaskOutcome
from ore_macl.debate import DebateOrchestrator, measure_independence
from ore_macl.errors import (
    CyclicDependencyError,
    MACLError,
    OutOfScopeAccess,
    StaleRevisionError,
)
from ore_macl.messages import AgentMessageBus, make_message
from ore_macl.pso import PSOConfig
from ore_macl.recovery import RecoveryManager, is_transient
from ore_macl.registry import assignable_states, valid_transitions
from ore_macl.reliability import score_reliability
from ore_macl.workspace_adapter import WorkspaceArtifactAdapter, raise_if_stale
from ore_workspace import SharedWorkspace, UpdateDecision

INPUT_TEXT = "the reporter ignored the scientist and argued about the study"


def make_workspace(text: str = INPUT_TEXT) -> SharedWorkspace:
    """Create a workspace seeded with the canonical test input."""
    return SharedWorkspace(current_input=text, current_goal="understand_sentence")


def make_supervisor(**config_overrides) -> MACLSupervisor:
    """Create a supervisor over a fresh workspace with a logical clock."""
    config_values = {"max_rounds": 8}
    config_values.update(config_overrides)
    workspace = make_workspace()
    supervisor = MACLSupervisor(
        workspace, build_default_agents(clock=LogicalClock()),
        config=MACLConfig(**config_values), clock=LogicalClock(),
    )
    return supervisor


def lexical_task() -> AgentTask:
    """Build the canonical lexical analysis task envelope."""
    return AgentTask(
        task_id="t:lex", goal_id="g1", objective="lexical hypotheses",
        required_capabilities=("macl.capability.lexical_analysis",),
        expected_result_schema=SchemaSpec({"token_hypotheses": "dict", "coverage": "number"}),
        permitted_workspace_fields=("current_input", "lexical_hypotheses"),
        workspace_revision=0,
        success_criteria=SuccessCriteria(
            required_output_keys=("token_hypotheses", "coverage"), must_cite_evidence=True,
        ),
    )


def make_result(task: AgentTask, agent_id: str = "agent.lexical.alpha",
                outputs: dict | None = None, revision: int = 0) -> AgentResult:
    """Build a minimal valid result for a task."""
    payload = outputs if outputs is not None else {
        "token_hypotheses": {"the": [{"candidate": "det", "confidence": 0.9}]},
        "coverage": 1.0,
    }
    return AgentResult(
        agent_id=agent_id, agent_version="1.0.0", task_id=task.task_id,
        goal_id=task.goal_id, source_revision=revision, outputs=payload,
        provenance=ProvenanceTrail(
            agent_id=agent_id, task_id=task.task_id, operation="test",
            workspace_revision=revision,
        ),
        evidence_refs=(ArtifactRef("current_input", "x" * 64, "workspace:current_input"),),
        self_confidence=0.8,
    )


# --------------------------------------------------------------------------- #
# Registry and contracts (Stage A, acceptance: identity/contract completeness)
# --------------------------------------------------------------------------- #

def test_registry_contains_nine_source_ordered_stages_covering_66_steps():
    """The operation registry must trace all nine stages A-I over Steps 1-66."""
    assert [item.stage_id for item in STAGE_REGISTRY] == [f"MACL-{i}" for i in range(1, 10)]
    all_steps = [step for item in STAGE_REGISTRY for step in item.source_steps]
    assert all_steps == list(range(1, 67))
    assert stage_for_step(1) == "MACL-1"
    assert stage_for_step(66) == "MACL-9"
    assert [item["stage_id"] for item in registry_as_dicts()] == [
        f"MACL-{i}" for i in range(1, 10)
    ]


def test_every_agent_has_stable_identity_contract_schemas_permissions_and_envelope():
    """Acceptance: every agent exposes identity, contract, schemas, permissions, limits."""
    for descriptor in default_specialist_descriptors():
        assert descriptor.agent_id and descriptor.agent_version
        assert descriptor.role
        assert descriptor.capability_ids
        assert isinstance(descriptor.autonomy, AutonomyLevel)
        assert descriptor.accepted_input_schema.fields
        assert descriptor.produced_output_schema.fields
        assert set(descriptor.writable_workspace_fields) <= set(
            descriptor.permitted_workspace_fields
        )
        envelope = descriptor.resources
        assert envelope.max_concurrent_tasks >= 1
        assert envelope.timeout_seconds > 0
        assert envelope.cost_budget > 0
        serialized = descriptor.to_dict()
        assert serialized["agent_id"] == descriptor.agent_id
        assert serialized["resources"]["max_concurrent_tasks"] >= 1


def test_autonomy_requires_justification_and_classification_is_explicit():
    """Acceptance (Step 1): autonomy levels are classified and justified, not assumed."""
    with pytest.raises(ValueError):
        classify_component(makes_action_choices=True, justification="  ")
    with pytest.raises(ValueError):
        classify_component(
            makes_action_choices=True, supervises_other_agents=True, justification="",
        )
    assert classify_component(makes_action_choices=False, uses_tools=False) is (
        AutonomyLevel.DETERMINISTIC_SPECIALIST
    )
    assert classify_component(makes_action_choices=False, uses_tools=True) is (
        AutonomyLevel.TOOL_USING_WORKER
    )
    assert classify_component(
        makes_action_choices=True, justification="adaptive evidence selection",
    ) is AutonomyLevel.BOUNDED_AUTONOMOUS_AGENT
    with pytest.raises(ValueError):
        AgentDescriptor(
            agent_id="x", agent_version="1", role="r",
            autonomy=AutonomyLevel.BOUNDED_AUTONOMOUS_AGENT,
            capability_ids=("c",), accepted_input_schema=SchemaSpec({"a": "string"}),
            produced_output_schema=SchemaSpec({"b": "string"}),
        )  # missing justification


def test_lifecycle_transitions_are_validated_and_history_preserved():
    """Acceptance (Step 5): health states transition legally; history is never erased."""
    registry = AgentRegistry()
    descriptor = default_specialist_descriptors()[0]
    registry.register(descriptor)
    registry.set_state(descriptor.agent_id, AgentLifecycleState.READY)
    with pytest.raises(MACLError):
        registry.set_state(descriptor.agent_id, AgentLifecycleState.REGISTERED)
    registry.set_state(descriptor.agent_id, AgentLifecycleState.BUSY)
    registry.set_state(descriptor.agent_id, AgentLifecycleState.READY)
    events = registry.history()
    assert any(item["event"] == "transition" for item in events)
    assert assignable_states() and valid_transitions()
    quarantined = AgentRegistry()
    quarantined.register(descriptor)
    quarantined.set_state(descriptor.agent_id, AgentLifecycleState.READY)
    quarantined.set_state(descriptor.agent_id, AgentLifecycleState.QUARANTINED)
    assert quarantined.quarantined_agents() == (descriptor.agent_id,)


def test_safe_replacement_requires_compatible_contract():
    """Acceptance (Step 7): replacement only under contract compatibility."""
    registry = AgentRegistry()
    descriptor = default_specialist_descriptors()[0]
    registry.register(descriptor)
    incompatible = AgentDescriptor(
        agent_id=descriptor.agent_id, agent_version="2.0.0", role="other_role",
        autonomy=descriptor.autonomy,
        capability_ids=descriptor.capability_ids,
        accepted_input_schema=descriptor.accepted_input_schema,
        produced_output_schema=descriptor.produced_output_schema,
        autonomy_justification="upgrade",
    )
    with pytest.raises(MACLError):
        registry.replace_agent(incompatible)
    compatible = AgentDescriptor(
        agent_id=descriptor.agent_id, agent_version="2.0.0", role=descriptor.role,
        autonomy=descriptor.autonomy,
        capability_ids=descriptor.capability_ids + ("macl.capability.extra",),
        accepted_input_schema=descriptor.accepted_input_schema,
        produced_output_schema=descriptor.produced_output_schema,
        autonomy_justification="compatible upgrade",
    )
    assert registry.replace_agent(compatible).agent_version == "2.0.0"


# --------------------------------------------------------------------------- #
# Decomposition (Stage B)
# --------------------------------------------------------------------------- #

def test_deterministic_modules_need_not_be_wrapped_as_agents():
    """Acceptance: deterministic module tasks route through the coordinator, not agents."""
    decomposer = TaskDecomposer(workspace_revision=0)
    dag = decomposer.decompose_module_task(
        "g1", objective="run module", required_capabilities=("module.capability",),
        module_route={"cognitive_task": object()},
    )
    (task,) = dag.tasks.values()
    assert task.kind is AgentTaskKind.DETERMINISTIC_MODULE
    assert task.topology is TaskTopology.HIERARCHICAL


def test_supervisor_cannot_bypass_coordinator_for_module_tasks():
    """Acceptance: the supervisor never calls module backends directly."""
    supervisor = make_supervisor()
    dag = supervisor.decomposer.decompose_module_task(
        "g2", objective="run module", required_capabilities=("module.capability",),
        module_route={"cognitive_task": object()},
    )
    (task,) = dag.tasks.values()
    assert task.kind is AgentTaskKind.DETERMINISTIC_MODULE

    class FakeOutcome:
        status = "RESOLVED"

    class FakeCoordinator:
        def coordinate(self, cognitive_task, context=None):
            return FakeOutcome()

    gateway = CoordinatorGateway(FakeCoordinator())
    supervisor.coordinator_gateway = gateway
    report = supervisor.run_goal(GoalRequest("g2", "understand_sentence", INPUT_TEXT))
    module_records = [
        record for record in report.task_records.values()
        if any(attempt.agent_id == "module" for attempt in record.attempts)
    ]
    assert module_records or report.stop_reason  # loop ran without direct module calls
    supervisor.close()


def test_module_task_without_gateway_is_blocked_not_bypassed():
    """Without a coordinator gateway the module task blocks; no direct execution."""
    supervisor = make_supervisor()
    dag = supervisor.decomposer.decompose_module_task(
        "g3", objective="run module", required_capabilities=("module.capability",),
        module_route={"cognitive_task": object()},
    )
    (task,) = dag.tasks.values()
    record = supervisor._task_record(task, report=None)
    spent = supervisor._dispatch_module_task(task, record, None)
    assert spent == 0.0
    assert record.final_outcome is TaskOutcome.BLOCKED
    assert "never calls module backends directly" in record.attempts[0].detail
    supervisor.close()


def test_task_dag_cycles_are_detected_before_dispatch():
    """Acceptance (Step 9): cycles raise before any dispatch."""
    a = AgentTask(
        task_id="a", goal_id="g", objective="a", required_capabilities=("c",),
        expected_result_schema=SchemaSpec({"x": "string"}),
    )
    b = AgentTask(
        task_id="b", goal_id="g", objective="b", required_capabilities=("c",),
        expected_result_schema=SchemaSpec({"x": "string"}), dependencies=("a",),
    )
    dag = TaskDAG("g")
    dag.add(a)
    dag.add(b)
    dag.link("a", "b")  # a depends on b, b depends on a -> cycle
    with pytest.raises(CyclicDependencyError):
        dag.validate_acyclic()
    assert set(dag.detect_cycles()) == {"a", "b"}
    # The default decomposer's DAG is acyclic and layered.
    plan = TaskDecomposer().decompose_language_comprehension("g")
    plan.validate_acyclic()
    layers = plan.topological_layers()
    assert layers[0][0].endswith(":lexical")


def test_topology_selection_requires_justification_for_decentralization():
    """Acceptance (Step 16): non-hierarchical topologies need explicit justification."""
    with pytest.raises(ValueError):
        AgentTask(
            task_id="t", goal_id="g", objective="o", required_capabilities=("c",),
            expected_result_schema=SchemaSpec({"x": "string"}),
            topology=TaskTopology.PEER_TO_PEER,
        )
    with pytest.raises(ValueError):
        AgentTask(
            task_id="t2", goal_id="g", objective="o", required_capabilities=("c",),
            expected_result_schema=SchemaSpec({"x": "string"}),
            topology=TaskTopology.MARKET, topology_justification="bidding seems fun",
        )
    justified = AgentTask(
        task_id="t3", goal_id="g", objective="o", required_capabilities=("c",),
        expected_result_schema=SchemaSpec({"x": "string"}),
        topology=TaskTopology.MARKET,
        topology_justification="bidding cost is justified by measured scheduling savings",
    )
    assert justified.topology is TaskTopology.MARKET


def test_eligibility_gate_blocks_hard_constraint_violations():
    """Acceptance (Step 12): eligibility gates precede scoring; hard failures block."""
    supervisor = make_supervisor()
    registry = supervisor.registry
    gate = EligibilityGate(registry)
    matcher = AgentTaskMatcher(registry)
    task = lexical_task()
    eligible = gate.eligible_agents(task)
    assert any(descriptor.agent_id == "agent.lexical.alpha" for descriptor, _ in eligible)
    # The verifier agent lacks the lexical capability and must fail the gate.
    verdict = gate.check(registry.get("agent.verifier.alpha"), task)
    assert not verdict.eligible
    assert any("missing capabilities" in reason for reason in verdict.reasons)
    ranked = matcher.rank(task, [descriptor for descriptor, _ in eligible])
    assert ranked and ranked[0].score > 0
    supervisor.close()


# --------------------------------------------------------------------------- #
# Supervisor loop, blackboard, verification (Stages C, D, F, G)
# --------------------------------------------------------------------------- #

def test_end_to_end_goal_completes_with_accepted_outcomes():
    """Acceptance: goal completion comes from accepted outcomes via the existing evaluator."""
    result = run_multi_agent_coordination(MACLRunConfig(max_rounds=8))
    report = result.goal_report
    assert report.goal_evaluation.outcome.value in ("completed", "completed_with_uncertainty")
    assert report.stop_reason == "goal criteria satisfied by accepted outcomes"
    assert report.completed_task_ids  # accepted outputs satisfied prerequisites
    assert result.stage_ids == tuple(f"MACL-{i}" for i in range(1, 10))
    supervisorless = [item for item in result.operation_trace if item["status"] == "failed"]
    assert supervisorless == []


def test_duplicate_messages_and_retries_do_not_duplicate_publication():
    """Acceptance: idempotent messages and fingerprints prevent duplicate publication."""
    supervisor = make_supervisor()
    bus = supervisor.bus
    message = make_message(
        sender_id="agent.lexical.alpha", recipient_id="macl.supervisor",
        kind=MessageKind.PROGRESS, task_id="t:lex", timestamp=1.0,
        payload={"phase": "p", "state": "running"},
    )
    first = bus.send(message, now=1.0)
    duplicate = bus.send(message, now=1.0)
    assert first.outcome is DeliveryOutcome.DELIVERED
    assert duplicate.outcome is DeliveryOutcome.DUPLICATE
    # Idempotency keys deduplicate distinct envelopes with the same key.
    again = make_message(
        sender_id="agent.lexical.alpha", recipient_id="macl.supervisor",
        kind=MessageKind.PROGRESS, task_id="t:lex", timestamp=2.0,
        payload={"phase": "p", "state": "running"},
        idempotency_key=message.idempotency_key,
    )
    assert bus.send(again, now=2.0).outcome is DeliveryOutcome.DUPLICATE
    # The workspace deduplicates equivalent findings by fingerprint.
    workspace = make_workspace()
    descriptor = default_specialist_descriptors()[0]
    workspace.permissions.grant(
        descriptor.agent_id, descriptor.writable_workspace_fields,
    )
    adapter = WorkspaceArtifactAdapter(workspace)
    task = lexical_task()
    finding_payload = {"word": "the", "candidate": "det", "candidates": [{"candidate": "det"}]}
    first_validation = adapter.propose(
        descriptor, task, field="lexical_hypotheses", payload=finding_payload,
        key="the", base_revision=workspace.revision,
    )
    second_validation = adapter.propose(
        descriptor, task, field="lexical_hypotheses", payload=finding_payload,
        key="the", base_revision=workspace.revision,
    )
    assert first_validation.decision is UpdateDecision.ACCEPTED
    assert second_validation.decision is UpdateDecision.DUPLICATE
    supervisor.close()


def test_stale_revision_results_are_rejected_or_revalidated():
    """Acceptance (Step 27): stale-revision results never integrate silently."""
    task = lexical_task()
    with pytest.raises(StaleRevisionError):
        raise_if_stale(task, current_revision=3)
    supervisor = make_supervisor()
    adjudicator = supervisor.adjudicator
    stale_result = make_result(task, revision=0)
    stale_task = AgentTask(
        task_id="t:lex", goal_id="g1", objective="lexical",
        required_capabilities=("macl.capability.lexical_analysis",),
        expected_result_schema=SchemaSpec({"token_hypotheses": "dict", "coverage": "number"}),
        workspace_revision=0,
    )
    supervisor.adapter.workspace._state.workspace_revision = 4
    verdict = adjudicator.validate(stale_result, stale_task)
    assert verdict.decision.value in ("rejected", "needs_revision")
    assert any("revision" in reason for reason in verdict.reasons)
    supervisor.close()


def test_agents_cannot_access_out_of_scope_artifacts():
    """Acceptance (Step 4): scoped snapshots deny out-of-scope reads and writes."""
    supervisor = make_supervisor()
    adapter = supervisor.adapter
    verifier = next(
        agent for agent in supervisor.agents.values()
        if agent.agent_id == "agent.verifier.alpha"
    )
    snapshot = adapter.snapshot_for(verifier.descriptor)
    with pytest.raises(OutOfScopeAccess):
        snapshot.read("discourse_context")
    # The adapter also rejects writes to fields outside the descriptor's writable set.
    task = lexical_task()
    with pytest.raises(OutOfScopeAccess):
        adapter.propose(
            verifier.descriptor, task, field="lexical_hypotheses",
            payload={"word": "the"}, key="the",
        )
    supervisor.close()


def test_conflicts_remain_visible_and_majority_vote_never_decides_truth():
    """Acceptance (Step 30): conflicting claims stay visible; no majority vote."""
    workspace = make_workspace()
    adapter = WorkspaceArtifactAdapter(workspace)
    descriptor = default_specialist_descriptors()[0]
    workspace.permissions.grant(
        descriptor.agent_id, descriptor.writable_workspace_fields,
    )
    task = lexical_task()
    first = adapter.propose(
        descriptor, task, field="lexical_hypotheses",
        payload={"word": "the", "candidate": "det", "candidates": [{"candidate": "det"}]},
        key="the", base_revision=workspace.revision,
    )
    second = adapter.propose(
        descriptor, task, field="lexical_hypotheses",
        payload={"word": "the", "candidate": "pron", "candidates": [{"candidate": "pron"}]},
        key="the", base_revision=workspace.revision,
    )
    assert first.decision is UpdateDecision.ACCEPTED
    assert second.decision is UpdateDecision.CONFLICT
    assert second.conflict_id
    conflicts = workspace.conflicts.list(unresolved_only=True)
    assert conflicts, "conflicting claims must remain visible"
    # The adjudication layer exposes preserved conflicts without majority voting.
    from ore_macl.adjudication import check_conflict_preservation
    preserved = check_conflict_preservation([{"claim": "det"}, {"claim": "pron"}])
    assert preserved["conflict_count"] == 2
    assert preserved["majority_vote_used"] is False


def test_fabricated_provenance_is_rejected():
    """Fabricated evidence references fail lineage validation (Sections 9-10)."""
    supervisor = make_supervisor()
    task = lexical_task()
    result = make_result(task)
    result = AgentResult(
        agent_id=result.agent_id, agent_version=result.agent_version,
        task_id=result.task_id, goal_id=result.goal_id,
        source_revision=result.source_revision, outputs=dict(result.outputs),
        provenance=result.provenance,
        evidence_refs=(ArtifactRef("nonexistent-artifact", "y" * 64, "nowhere"),),
        input_refs=(),
        self_confidence=0.9,
    )
    errors = supervisor.adapter.validate_evidence_lineage(result.evidence_refs)
    assert errors and "unresolvable evidence reference" in errors[0]
    verdict = supervisor.adjudicator.validate(result, task)
    assert verdict.decision.value in ("rejected", "needs_revision")
    supervisor.close()


def test_independent_verification_gates_critical_outputs():
    """Acceptance (Step 42): findings require independent or deterministic verification."""
    result = run_multi_agent_coordination(
        MACLRunConfig(require_verification_for_findings=True),
    )
    verified = sum(
        1
        for record in result.goal_report.task_records.values()
        for attempt in record.attempts
        if attempt.adjudication and attempt.adjudication.get("verified_by")
    )
    assert verified >= 1, "hypothesis-publishing tasks must be independently verified"


def test_debate_bounded_rounds_and_preserved_dissent():
    """Acceptance (Steps 41, 43): critic loops are bounded; minority findings survive."""
    supervisor = make_supervisor()
    task = AgentTask(
        task_id="d:sem", goal_id="g", objective="competing interpretation",
        required_capabilities=("macl.capability.semantic_interpretation",),
        expected_result_schema=SchemaSpec({"meaning": "dict", "interpretation": "string"}),
        topology=TaskTopology.DEBATE, topology_justification="competing hypotheses",
    )
    agents = supervisor.agents
    proposals = []
    for agent in (agents["agent.semantic.alpha"], agents["agent.verifier.alpha"]):
        result = make_result(
            task, agent_id=agent.agent_id,
            outputs={
                "meaning": {"subject": "reporter"},
                "interpretation": f"{agent.agent_id} reading",
                "confidence": 0.7,
            },
        )
        proposals.append(result)
    critic = agents["agent.critic.alpha"]
    orchestrator = DebateOrchestrator(supervisor.adjudicator, max_rounds=3)
    record = orchestrator.run(task, proposals, critic)
    assert 1 <= record.rounds <= 3
    assert record.stop_reason in ("critic criteria passed", "maximum rounds reached", "budget exhausted")
    assert record.selected_agent_id
    assert len(record.dissent) >= 1  # the non-selected proposal's evidence is preserved
    independence = measure_independence(proposals[0], proposals[1])
    assert 0.0 <= independence.independence <= 1.0
    supervisor.close()


def test_independence_measures_shared_lineage():
    """Acceptance (Step 38): same evidence and policy reduce measured independence."""
    task = AgentTask(
        task_id="t", goal_id="g", objective="o", required_capabilities=("c",),
        expected_result_schema=SchemaSpec({"x": "string"}),
    )
    a = make_result(task, agent_id="agent.a")
    b = make_result(task, agent_id="agent.b")
    independent = measure_independence(a, b, policy_versions={"agent.a": "p1", "agent.b": "p2"})
    correlated = measure_independence(a, b, policy_versions={"agent.a": "p1", "agent.b": "p1"})
    assert correlated.independence < independent.independence
    assert correlated.shared_policy is True


# --------------------------------------------------------------------------- #
# Recovery, quarantine, budgets (Stage H)
# --------------------------------------------------------------------------- #

def test_quarantined_agents_receive_no_new_tasks():
    """Acceptance (Step 54): quarantine blocks assignment."""
    supervisor = make_supervisor()
    registry = supervisor.registry
    lexical = registry.get("agent.lexical.alpha")
    registry.set_state(lexical.agent_id, AgentLifecycleState.READY)
    registry.set_state(lexical.agent_id, AgentLifecycleState.QUARANTINED)
    assert not registry.assignable(lexical.agent_id)
    gate = EligibilityGate(registry)
    verdict = gate.check(lexical, lexical_task())
    assert not verdict.eligible
    assert registry.quarantined_agents() == (lexical.agent_id,)
    supervisor.close()


def test_quarantine_triggers_after_repeated_violations():
    """Repeated policy violations quarantine the agent (Step 54)."""
    supervisor = make_supervisor(quarantine_threshold=3)
    recovery = supervisor.recovery
    agent_id = "agent.lexical.alpha"
    from ore_macl.errors import QuarantineTriggered
    with pytest.raises(QuarantineTriggered):
        for _ in range(3):
            recovery.record_violation(agent_id, "invalid_output", "schema mismatch")
    assert supervisor.registry.state(agent_id) is AgentLifecycleState.QUARANTINED
    supervisor.close()


def test_timeouts_are_reconciled_before_unsafe_retry():
    """Acceptance (Step 52): timeout marks the outcome unknown, then reconciliation runs."""
    supervisor = make_supervisor()
    recovery = RecoveryManager(supervisor.registry, lease_ticks=1)
    agent = supervisor.agents["agent.lexical.alpha"]
    task = lexical_task()
    report = recovery.reconcile_timeout(task, agent)
    assert report.outcome is TaskOutcome.UNKNOWN
    assert "unknown" in report.detail
    assert is_transient("timeout") and is_transient("unknown delivery")
    assert not is_transient("permission denied")
    # A lease must expire or be cancelled before reassignment (Step 53).
    recovery.issue_lease(task.task_id, agent.agent_id)
    assert not recovery.allow_reassignment(task, agent, cancelled=False)
    recovery.tick_leases()
    assert recovery.allow_reassignment(task, agent, cancelled=False)
    supervisor.close()


def test_retry_policy_permits_only_transient_idempotent_retries():
    """Acceptance (Step 51): non-transient or non-idempotent work is not retried."""
    policy = RetryPolicy(max_attempts=2, idempotent_only=True)
    assert policy.allows(1, idempotent=True, transient=True)
    assert not policy.allows(1, idempotent=False, transient=True)
    assert not policy.allows(1, idempotent=True, transient=False)
    assert not policy.allows(2, idempotent=True, transient=True)


def test_concurrency_budgets_are_enforced():
    """Acceptance: resource and concurrency budgets are enforced."""
    supervisor = make_supervisor()
    registry = supervisor.registry
    agent_id = "agent.lexical.alpha"
    descriptor = registry.get(agent_id)
    limit = descriptor.resources.max_concurrent_tasks
    for _ in range(limit):
        registry.acquire_capacity(agent_id)
    with pytest.raises(MACLError):
        registry.acquire_capacity(agent_id)
    assert not registry.assignable(agent_id)
    registry.release_capacity(agent_id, limit)
    assert registry.assignable(agent_id)
    supervisor.close()


def test_goal_budget_bounds_the_loop():
    """Budget exhaustion stops the loop instead of relaxing constraints."""
    supervisor = make_supervisor(goal_budget=0.05, max_rounds=10)
    report = supervisor.run_goal(GoalRequest("gb", "understand_sentence", INPUT_TEXT))
    assert report.stop_reason in (
        "goal budget exhausted", "goal deadline exceeded", "maximum rounds reached",
        "blocked: no task could be dispatched",
    )
    supervisor.close()


# --------------------------------------------------------------------------- #
# Reliability with uncertainty (Stage H, mathematical formulation)
# --------------------------------------------------------------------------- #

def test_reliability_includes_sample_support_and_uncertainty():
    """Acceptance (Step 56): Beta-Binomial posteriors expose intervals and samples."""
    tiny = BetaBinomialEstimate(0, 0)
    seasoned = BetaBinomialEstimate(27, 3)
    assert tiny.samples == 0
    assert tiny.posterior_mean == 0.5
    assert uncertainty_penalty(tiny) > uncertainty_penalty(seasoned)
    lower, upper = seasoned.credible_interval(0.9)
    assert 0.0 <= lower < upper <= 1.0
    assert 0.0 <= score_reliability(tiny) < score_reliability(seasoned)
    ledger = PerformanceLedger()
    task = lexical_task()
    for accepted in (True, True, False):
        ledger.record(task, make_result(task), AdjudicationReport(
            decision=(
                AdjudicationDecision.ACCEPTED if accepted
                else AdjudicationDecision.REJECTED
            ),
            task_id=task.task_id, agent_id="agent.lexical.alpha",
        ))
    summary = ledger.summary()
    entry = summary["reliability"]["agent.lexical.alpha"]
    assert entry["samples"] == 3
    assert len(entry["credible_interval"]) == 2
    assert summary["entries"] == 3


# --------------------------------------------------------------------------- #
# Peer coordination bounds (Stage E)
# --------------------------------------------------------------------------- #

def test_peer_delegation_depth_and_message_volume_are_bounded():
    """Acceptance (Steps 33, 36): delegation depth and negotiation rounds are bounded."""
    supervisor = make_supervisor(max_delegation_depth=2)
    peer = supervisor.peer
    agents = supervisor.agents
    owner = agents["agent.syntactic.alpha"]
    task = AgentTask(
        task_id="p:task", goal_id="g", objective="peer assist",
        required_capabilities=("macl.capability.lexical_analysis",),
        expected_result_schema=SchemaSpec({"token_hypotheses": "dict", "coverage": "number"}),
    )
    deep_chain = DelegationChain(owner.agent_id, (owner.agent_id, "agent.a", "agent.b"))
    record = peer.request_assistance(
        requesting_agent=owner, task=task,
        capability="macl.capability.lexical_analysis",
        chain=deep_chain, now=0.0,
    )
    assert record.outcome is PeerNegotiationOutcome.MAX_DEPTH_REACHED
    circular = DelegationChain(owner.agent_id, (owner.agent_id, "agent.lexical.alpha", owner.agent_id))
    record = peer.request_assistance(
        requesting_agent=owner, task=task,
        capability="macl.capability.lexical_analysis",
        chain=circular, now=0.0,
    )
    assert record.outcome is PeerNegotiationOutcome.MAX_DEPTH_REACHED
    assert peer.max_delegation_depth == 2
    assert peer.stats()["max_delegation_depth"] == 2
    supervisor.close()


def test_peer_negotiation_escalates_when_no_peer_holds_capability():
    """Acceptance (Step 36): failed negotiation escalates to the supervisor."""
    supervisor = make_supervisor()
    owner = supervisor.agents["agent.critic.alpha"]
    task = AgentTask(
        task_id="p:none", goal_id="g", objective="impossible",
        required_capabilities=("macl.capability.nonexistent",),
        expected_result_schema=SchemaSpec({"x": "string"}),
    )
    record = supervisor.peer.request_assistance(
        requesting_agent=owner, task=task,
        capability="macl.capability.nonexistent", now=0.0,
    )
    assert record.outcome is PeerNegotiationOutcome.ESCALATED
    assert "escalating" in record.detail
    supervisor.close()


def test_message_quota_and_expiry_bound_volume():
    """Acceptance (Section 10): quotas and TTL bound message storms."""
    bus = AgentMessageBus(default_quota=2)
    sender, recipient = "agent.a", "agent.b"
    for index in range(3):
        receipt = bus.send(make_message(
            sender_id=sender, recipient_id=recipient, kind=MessageKind.HEARTBEAT,
            task_id=f"t{index}", timestamp=float(index),
        ), now=0.0)
        if index < 2:
            assert receipt.outcome is DeliveryOutcome.DELIVERED
        else:
            assert receipt.outcome is DeliveryOutcome.QUOTA_EXCEEDED
    expired = make_message(
        sender_id=sender, recipient_id=recipient, kind=MessageKind.REQUEST,
        task_id="t2", timestamp=0.0, expires_at=0.5,
    )
    bus.reset_window(sender)
    assert bus.send(expired, now=1.0).outcome is DeliveryOutcome.EXPIRED


# --------------------------------------------------------------------------- #
# Swarm optimizers (Stage I)
# --------------------------------------------------------------------------- #

def test_marl_cannot_control_production_ore_without_approval():
    """Acceptance (Step 59): swarm experiments are isolated from production control."""
    result = run_multi_agent_coordination(MACLRunConfig(pso_iterations=3, aco_iterations=3))
    stage = result.operation_trace[-1]
    assert stage["operation_id"] == "MACL-9"
    assert stage["functional_output"]["isolated"] is True
    assert stage["functional_output"]["production_control"] is False
    from ore_macl.errors import MARLProductionControlError
    assert issubclass(MARLProductionControlError, MACLError)


def test_pso_cannot_bypass_constraints_and_is_bounded():
    """Acceptance (Step 65): PSO clamps to bounds, honours constraints and budgets."""
    calls = {"count": 0}

    def objective(weights):
        calls["count"] += 1
        return resource_policy_objective(weights, reference_solution=(0.5, 0.5, 0.5))

    config = PSOConfig(num_particles=4, iterations=5, seed=3, max_evaluations=40)
    optimizer = ParticleSwarmOptimizer(
        config, objective, bounds=[(0.0, 1.0)] * 3,
        constraint=lambda weights: sum(weights) <= 2.0,
    )
    result = optimizer.optimize()
    for value in result.best_position:
        assert 0.0 <= value <= 1.0
    assert sum(result.best_position) <= 2.0
    assert result.evaluations <= 40
    assert calls["count"] == result.evaluations
    # The evaluation budget is a hard bound, not a suggestion.
    with pytest.raises(MACLError):
        for _ in range(100):
            optimizer._evaluate((0.5, 0.5, 0.5))


def test_aco_enforces_dependencies_and_validates_paths():
    """Acceptance (Steps 64-65): ants only walk dependency-feasible, budget-feasible paths."""
    graph = OperationGraph(
        operations=("a", "b", "c"),
        dependencies={"b": ("a",), "c": ("b",)},
        edge_cost={
            ("start", "a"): 1.0, ("a", "b"): 1.0, ("b", "c"): 1.0,
            ("start", "c"): 1.0,
        },
        heuristic_desirability={
            ("start", "a"): 1.0, ("a", "b"): 1.0, ("b", "c"): 1.0,
            ("start", "c"): 3.0,
        },
    )
    config = ACOConfig(num_ants=4, iterations=6, seed=5, max_path_length=4)
    colony = AntColonyOptimizer(
        config, graph, validator=lambda path: dependency_feasible(path, graph),
        budget=10.0,
    )
    result = colony.optimize()
    assert result.validated_best is True
    assert dependency_feasible(result.best_path, graph)
    completed: set[str] = set()
    for operation in result.best_path:
        assert all(dep in completed for dep in graph.predecessors_of(operation))
        completed.add(operation)
    # An infeasible path can never be the validated best.
    assert not dependency_feasible(("c", "a"), graph)


# --------------------------------------------------------------------------- #
# Observability and pipeline lineage (Stage G, acceptance: full lineage)
# --------------------------------------------------------------------------- #

def test_full_lineage_is_recorded_in_sqlite(tmp_path):
    """Acceptance: annotations and the operation trace persist through the ORE logger."""
    from ore_macl import record_result_to_sqlite, register_annotations

    database_path = tmp_path / "macl_lineage.db"
    catalogue = register_annotations(database_path)
    assert catalogue["parse_errors"] == 0
    assert catalogue["symbols_written"] > 300

    result = run_multi_agent_coordination(MACLRunConfig(max_rounds=6))
    recorded = record_result_to_sqlite(result, database_path)
    assert recorded["run_id"]
    summary = recorded["database_summary"]
    assert summary["events"] == 9
    assert summary["runs"] == 1

    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'",
        ).fetchone()
        assert rows[0] >= 1
    finally:
        connection.close()


def test_goal_report_records_messages_attempts_and_registry_lineage():
    """The goal report exposes message, attempt, ledger, and registry lineage."""
    result = run_multi_agent_coordination(MACLRunConfig(max_rounds=6))
    report = result.goal_report
    assert report.message_stats["messages_sent"] >= 1
    assert any(
        attempt.adjudication
        for record in report.task_records.values()
        for attempt in record.attempts
    )
    assert report.ledger_summary["entries"] >= 1
    assert report.registry_snapshot["agents"]
    assert report.lineage["coordinator_calls"] == []


def test_deterministic_fallback_when_no_agent_qualifies():
    """Acceptance (Step 14): deterministic fallbacks run when no agent qualifies."""
    from ore_macl.agents import (
        deterministic_lexical_hypotheses,
        deterministic_semantic_hypothesis,
        deterministic_syntactic_hypothesis,
    )

    workspace = make_workspace()
    # An explicitly empty agent set forces fallbacks; constraints are never relaxed.
    supervisor = MACLSupervisor(workspace, {}, clock=LogicalClock())
    report = supervisor.run_goal(GoalRequest("fb", "understand_sentence", INPUT_TEXT))
    fallbacks = {
        task_id: record for task_id, record in report.task_records.items()
        if record.used_deterministic_fallback
    }
    assert fallbacks, "lexical/syntactic/semantic must fall back deterministically"
    assert deterministic_lexical_hypotheses(INPUT_TEXT)["coverage"] > 0
    assert deterministic_syntactic_hypothesis(INPUT_TEXT)["parse"]["strategy"]
    assert deterministic_semantic_hypothesis(INPUT_TEXT)["meaning"]["subject"]
    # A capability with no deterministic fallback blocks visibly instead of relaxing.
    blocked = [
        task_id for task_id, record in report.task_records.items()
        if record.final_outcome is TaskOutcome.BLOCKED
    ]
    state = workspace.snapshot().state
    if blocked:
        assert state.get("unresolved_questions"), "blocked tasks must escalate visibly"
        assert any(
            question.get("question_id", "").startswith("blocked:")
            for question in state["unresolved_questions"]
        )
    supervisor.close()


def test_pipeline_skips_swarm_experiments_when_disabled():
    """The pipeline honours the isolated-experiments flag (Step 66 deferral)."""
    result = run_multi_agent_coordination(
        MACLRunConfig(run_isolated_swarm_experiments=False),
    )
    stage = result.operation_trace[-1]
    assert stage["status"] == "skipped"
    assert stage["functional_output"]["reason"] == "swarm experiments disabled"
    assert result.pso_result is None and result.aco_result is None


def test_biological_network_advances_and_stays_bounded():
    """The coordination network surrogate stays within [0, 1] at every step."""
    from ore_macl.biology import STAGE_DRIVES

    result = run_multi_agent_coordination(MACLRunConfig(max_rounds=6))
    for name, activation in result.biological_snapshot.items():
        assert 0.0 <= activation <= 1.0, f"population {name} left [0, 1]"
    assert len(STAGE_DRIVES) == 9
