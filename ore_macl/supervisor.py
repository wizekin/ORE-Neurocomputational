"""MACL supervisor: the end-to-end hierarchical coordination loop (spec Section 3).

The supervisor owns the top-level objective, decomposition, assignments, policy
gates, and integration. It is not a second scheduler or second blackboard: agent
work is dispatched through the bounded executor, deterministic module work is
routed through the existing CognitiveTaskCoordinator gateway, and every accepted
contribution is published through the existing shared workspace (spec Section 1).
"""
from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ore_workspace import GoalEvaluator, SharedWorkspace, UpdateDecision

from .adjudication import AdjudicationReport, ResultAdjudicator
from .agents import (
    BaseBoundedAgent,
    ExecutionContext,
    TaskExecutor,
    WallClock,
    build_default_agents,
    deterministic_lexical_hypotheses,
    deterministic_semantic_hypothesis,
    deterministic_syntactic_hypothesis,
)
from .aggregation import PublicationRecord, ResultAggregator, evaluate_goal_progress
from .biology import STAGE_DRIVES, MultiAgentCoordinationNetwork
from .contracts import (
    AdjudicationDecision,
    AgentDescriptor,
    AgentLifecycleState,
    AgentResult,
    AgentTask,
    AgentTaskKind,
    ArtifactRef,
    AutonomyLevel,
    MessageKind,
    ProvenanceTrail,
    SchemaSpec,
    TaskOutcome,
    TaskTopology,
)
from .debate import DebateOrchestrator, verify_independently
from .decomposition import TaskDAG, TaskDecomposer
from .errors import (
    QuarantineTriggered,
)
from .ledger import PerformanceLedger
from .matching import AgentTaskMatcher, EligibilityGate, MatchingConfig, assign_owners
from .messages import AgentMessageBus, make_message
from .peer import PeerCoordinator
from .recovery import RecoveryManager, is_transient
from .registry import AgentRegistry
from .workspace_adapter import WorkspaceArtifactAdapter

SUPERVISOR_PRODUCER_ID = "macl.supervisor"


@dataclass
class MACLConfig:
    """Bound every supervisor loop with explicit limits and policies."""

    max_rounds: int = 12
    goal_budget: float = 200.0
    goal_deadline_seconds: float = 120.0
    assignment_strategy: str = "greedy"
    debate_max_rounds: int = 3
    revision_attempts: int = 2
    task_attempt_limit: int = 3
    require_verification_for_findings: bool = True
    enable_peer_topology: bool = False
    max_delegation_depth: int = 2
    quarantine_threshold: int = 3
    lease_ticks: int = 1
    use_threads: bool = False
    task_timeout_seconds: float | None = 15.0
    matching: MatchingConfig = field(default_factory=MatchingConfig)

    def __post_init__(self) -> None:
        """Validate configuration bounds."""
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least one")
        if self.goal_budget <= 0:
            raise ValueError("goal_budget must be positive")
        if self.goal_deadline_seconds <= 0:
            raise ValueError("goal_deadline_seconds must be positive")
        if self.assignment_strategy not in ("greedy", "max_weight"):
            raise ValueError("assignment_strategy must be 'greedy' or 'max_weight'")


@dataclass
class AttemptRecord:
    """Record one dispatch attempt from assignment to adjudication."""

    agent_id: str
    outcome: TaskOutcome
    detail: str = ""
    adjudication: dict[str, Any] | None = None
    retries: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a detached attempt record."""
        return {
            "agent_id": self.agent_id,
            "outcome": self.outcome.value,
            "detail": self.detail,
            "adjudication": self.adjudication,
            "retries": self.retries,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class TaskExecutionRecord:
    """Trace one task's attempts, owner, final outcome, and publication decision."""

    task_id: str
    objective: str
    owner_agent_id: str | None = None
    attempts: list[AttemptRecord] = field(default_factory=list)
    final_outcome: TaskOutcome = TaskOutcome.PENDING
    adjudication: dict[str, Any] | None = None
    publication: dict[str, Any] | None = None
    used_deterministic_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a detached task record."""
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "owner_agent_id": self.owner_agent_id,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "final_outcome": self.final_outcome.value,
            "adjudication": self.adjudication,
            "publication": self.publication,
            "used_deterministic_fallback": self.used_deterministic_fallback,
        }


@dataclass
class GoalRunReport:
    """Report one complete goal run with lineage, conflicts, dissent, and metrics."""

    goal_id: str
    goal: str
    workspace_id: str
    rounds: int = 0
    stop_reason: str = ""
    goal_evaluation: Any = None
    task_records: dict[str, TaskExecutionRecord] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    dissent: list[dict[str, Any]] = field(default_factory=list)
    debates: list[dict[str, Any]] = field(default_factory=list)
    ledger_summary: dict[str, Any] = field(default_factory=dict)
    biology_trace: list[dict[str, float]] = field(default_factory=list)
    message_stats: dict[str, Any] = field(default_factory=dict)
    registry_snapshot: dict[str, Any] = field(default_factory=dict)
    resource_usage_total: float = 0.0
    lineage: dict[str, Any] = field(default_factory=dict)

    @property
    def completed_task_ids(self) -> tuple[str, ...]:
        """Return task IDs whose outputs were integrated."""
        return tuple(
            task_id for task_id, record in sorted(self.task_records.items())
            if record.final_outcome is TaskOutcome.COMPLETED
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible report."""
        return {
            "goal_id": self.goal_id,
            "goal": self.goal,
            "workspace_id": self.workspace_id,
            "rounds": self.rounds,
            "stop_reason": self.stop_reason,
            "goal_evaluation": (
                dataclasses.asdict(self.goal_evaluation)
                if self.goal_evaluation is not None else None
            ),
            "task_records": {
                task_id: record.to_dict() for task_id, record in sorted(self.task_records.items())
            },
            "conflicts": list(self.conflicts),
            "dissent": list(self.dissent),
            "debates": list(self.debates),
            "ledger_summary": self.ledger_summary,
            "biology_trace": list(self.biology_trace),
            "message_stats": self.message_stats,
            "registry_snapshot": self.registry_snapshot,
            "resource_usage_total": self.resource_usage_total,
            "lineage": dict(self.lineage),
        }


@dataclass
class GoalRequest:
    """Validate one top-level goal before decomposition (Section 3, step 1)."""

    goal_id: str
    goal: str
    input_text: str
    component_scores: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        """Reject empty goals or inputs."""
        for name in ("goal_id", "goal", "input_text"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")


class CoordinatorGateway:
    """Route deterministic module tasks through an existing coordinator (Section 6)."""

    def __init__(self, coordinator: Any) -> None:
        """Bind the gateway to a CognitiveTaskCoordinator instance."""
        self.coordinator = coordinator
        self.calls: list[str] = []

    def coordinate(self, cognitive_task: Any, context: Any = None) -> Any:
        """Delegate one CognitiveTask and record that the coordinator was used."""
        self.calls.append(str(getattr(cognitive_task, "task_id", "task")))
        return self.coordinator.coordinate(cognitive_task, context)


class MACLSupervisor:
    """Own the top-level objective, decomposition, assignments, and integration."""

    def __init__(
        self,
        workspace: SharedWorkspace,
        agents: Mapping[str, BaseBoundedAgent] | None = None,
        *,
        coordinator_gateway: CoordinatorGateway | None = None,
        config: MACLConfig | None = None,
        clock: Any = None,
        goal_evaluator: GoalEvaluator | None = None,
        event_logger: Any = None,
        decomposer: TaskDecomposer | None = None,
    ) -> None:
        """Wire the supervisor above the existing workspace and coordinator."""
        if not isinstance(workspace, SharedWorkspace):
            raise TypeError("workspace must be a SharedWorkspace")
        self.workspace = workspace
        self.config = config or MACLConfig()
        self.clock = clock or WallClock()
        self.goal_evaluator = goal_evaluator or GoalEvaluator()
        self.coordinator_gateway = coordinator_gateway
        self.event_logger = event_logger
        self.adapter = WorkspaceArtifactAdapter(workspace)
        self.registry = AgentRegistry()
        self.bus = AgentMessageBus()
        self.ledger = PerformanceLedger()
        self.adjudicator = ResultAdjudicator(self.adapter)
        self.recovery = RecoveryManager(
            self.registry,
            quarantine_threshold=self.config.quarantine_threshold,
            lease_ticks=self.config.lease_ticks,
        )
        self.peer = PeerCoordinator(
            self.registry, self.bus,
            max_delegation_depth=self.config.max_delegation_depth,
        )
        self.matcher = AgentTaskMatcher(
            self.registry,
            config=self.config.matching,
            estimates={},
            conflict_rates={},
        )
        self.gate = EligibilityGate(self.registry, estimates={})
        self.debate = DebateOrchestrator(
            self.adjudicator, max_rounds=self.config.debate_max_rounds,
        )
        self.aggregator = ResultAggregator(self.adapter, ledger=self.ledger)
        self.executor = TaskExecutor(use_threads=self.config.use_threads)
        self.biology = MultiAgentCoordinationNetwork()
        self.decomposer = decomposer or TaskDecomposer(workspace_revision=workspace.revision)
        self.agents: dict[str, BaseBoundedAgent] = dict(
            agents if agents is not None else build_default_agents(clock=self.clock)
        )
        self._completed_keys: set[str] = set()
        self._integrated_task_ids: set[str] = set()
        self._register_agents()

    # ------------------------------------------------------------------ #
    # Registration and permissions
    # ------------------------------------------------------------------ #

    def _register_agents(self) -> None:
        """Register every agent contract and grant least-privilege permissions (Steps 2, 4)."""
        for agent in self.agents.values():
            self.registry.register(agent.descriptor)
            self.registry.set_state(agent.agent_id, AgentLifecycleState.READY)
            if agent.descriptor.writable_workspace_fields:
                self.workspace.permissions.grant(
                    agent.agent_id, agent.descriptor.writable_workspace_fields,
                )
        supervisor_fields = {
            "unresolved_questions", "module_status", "artifact_references",
            "lexical_hypotheses", "syntactic_hypotheses", "semantic_hypotheses",
        }
        self.workspace.permissions.grant(SUPERVISOR_PRODUCER_ID, supervisor_fields)
        self.peer.bind_agents(list(self.agents.values()))

    def _record(self, method_name: str, metadata: Mapping[str, Any]) -> None:
        """Persist one supervisor event when a logger is attached."""
        if self.event_logger is not None:
            self.event_logger.record(
                method_name, dict(metadata),
                class_name="MACLSupervisor", module_name="ore_macl.supervisor",
            )

    # ------------------------------------------------------------------ #
    # Main coordination loop (Section 3)
    # ------------------------------------------------------------------ #

    def run_goal(self, request: GoalRequest) -> GoalRunReport:
        """Plan, dispatch, adjudicate, integrate, and stop one bounded goal run."""
        if not isinstance(request, GoalRequest):
            raise TypeError("request must be a GoalRequest")
        started = self.clock.now()
        dag = self._build_dag(request)
        report = GoalRunReport(
            goal_id=request.goal_id, goal=request.goal,
            workspace_id=self.workspace.workspace_id,
        )
        remaining_budget = self.config.goal_budget
        stop_reason = ""
        self._record("run_goal", {
            "goal_id": request.goal_id, "goal": request.goal,
            "tasks": sorted(dag.tasks), "start_revision": self.adapter.revision,
        })
        for round_index in range(1, self.config.max_rounds + 1):
            report.rounds = round_index
            report.biology_trace.append(self.biology.advance(STAGE_DRIVES[min(round_index - 1, len(STAGE_DRIVES) - 1)]))
            self.recovery.tick_leases()
            ready = self._ready_tasks(dag, report)
            if not ready:
                stop_reason = "no eligible work remains" if self._integrated_task_ids else (
                    "blocked: no task could be dispatched"
                )
                break
            spent = self._run_round(request, dag, ready, report, remaining_budget=remaining_budget)
            remaining_budget = max(0.0, remaining_budget - spent)
            report.resource_usage_total += spent
            evaluation = evaluate_goal_progress(
                self.goal_evaluator, self.workspace,
                component_scores=request.component_scores,
            )
            report.goal_evaluation = evaluation
            if evaluation.outcome.value in ("completed", "completed_with_uncertainty"):
                stop_reason = "goal criteria satisfied by accepted outcomes"
                break
            if evaluation.outcome.value in ("failed", "blocked"):
                stop_reason = f"goal evaluator reported {evaluation.outcome.value}"
                break
            if remaining_budget <= 0:
                stop_reason = "goal budget exhausted"
                break
            if self.clock.now() - started > self.config.goal_deadline_seconds:
                stop_reason = "goal deadline exceeded"
                break
            if all(
                record.final_outcome is not TaskOutcome.PENDING
                for record in report.task_records.values()
            ) and not self._ready_tasks(dag, report):
                stop_reason = "all tasks reached terminal states"
                break
        else:
            stop_reason = "maximum rounds reached"
        if not stop_reason:
            stop_reason = "maximum rounds reached"
        evaluation = evaluate_goal_progress(
            self.goal_evaluator, self.workspace,
            component_scores=request.component_scores,
        )
        report.goal_evaluation = evaluation
        report.stop_reason = stop_reason
        report.conflicts = [
            {"conflict_id": record.conflict_id, "status": record.status,
             "findings": list(record.conflicting_findings), "type": record.conflict_type}
            for record in self.workspace.conflicts.list()
        ]
        report.ledger_summary = self.ledger.summary()
        report.message_stats = self.bus.stats()
        report.registry_snapshot = self.registry.snapshot()
        report.lineage = {
            "messages_sent": report.message_stats.get("messages_sent", 0),
            "findings_accepted": len(self.workspace.findings),
            "quarantined_agents": list(self.registry.quarantined_agents()),
            "negotiations": self.peer.stats(),
            "coordinator_calls": list(self.coordinator_gateway.calls) if self.coordinator_gateway else [],
        }
        self._record("run_goal_complete", {
            "goal_id": request.goal_id, "stop_reason": stop_reason,
            "rounds": report.rounds, "tasks_completed": len(report.completed_task_ids),
        })
        return report

    def _build_dag(self, request: GoalRequest) -> TaskDAG:
        """Decompose the goal and detect dependency cycles before dispatch (Steps 8-9)."""
        self.decomposer.workspace_revision = self.adapter.revision
        return self.decomposer.decompose_language_comprehension(request.goal_id)

    def _task_record(self, task: AgentTask, *, report: GoalRunReport | None) -> TaskExecutionRecord:
        """Return (and lazily create) the execution record for one task."""
        if report is not None:
            record = report.task_records.get(task.task_id)
            if record is None:
                record = TaskExecutionRecord(task_id=task.task_id, objective=task.objective)
                report.task_records[task.task_id] = record
            return record
        return TaskExecutionRecord(task_id=task.task_id, objective=task.objective)

    def _ready_tasks(self, dag: TaskDAG, report: GoalRunReport | None = None) -> tuple[AgentTask, ...]:
        """Return dispatchable tasks with fresh revision bindings (Step 27).

        Tasks that already reached a terminal failure, block, or cancellation are
        never re-dispatched: rework happens through bounded retries inside one
        attempt, not by silent duplicate execution (Step 21).
        """
        terminal: set[str] = set()
        if report is not None:
            terminal = {
                task_id for task_id, record in report.task_records.items()
                if record.final_outcome in (
                    TaskOutcome.FAILED, TaskOutcome.BLOCKED, TaskOutcome.CANCELLED,
                )
            }
        ready = dag.ready_tasks(self._integrated_task_ids)
        fresh = []
        for task in ready:
            if task.idempotency_key in self._completed_keys:
                continue
            if task.task_id in terminal:
                continue
            if task.workspace_revision != self.adapter.revision:
                task = dataclasses.replace(task, workspace_revision=self.adapter.revision)
            fresh.append(task)
        return tuple(fresh)

    def _run_round(
        self,
        request: GoalRequest,
        dag: TaskDAG,
        ready: Sequence[AgentTask],
        report: GoalRunReport,
        *,
        remaining_budget: float,
    ) -> float:
        """Dispatch, collect, adjudicate, verify, and integrate one round (Steps 5-15)."""
        spent = 0.0
        for task in ready:
            record = self._task_record(task, report=report)
            if len(record.attempts) >= self.config.task_attempt_limit:
                record.final_outcome = TaskOutcome.BLOCKED
                record.attempts.append(AttemptRecord(
                    record.owner_agent_id or "supervisor", TaskOutcome.BLOCKED,
                    detail=(
                        "task exceeded the bounded attempt limit "
                        f"({self.config.task_attempt_limit}) without a terminal decision"
                    ),
                ))
                self._publish_unresolved_question(task, report)
                continue
            if task.kind is AgentTaskKind.DETERMINISTIC_MODULE:
                spent += self._dispatch_module_task(task, record, report)
                continue
            if task.topology is TaskTopology.DEBATE:
                spent += self._run_debate_task(task, record, report, remaining_budget=remaining_budget - spent)
                continue
            if task.topology is TaskTopology.PEER_TO_PEER and self.config.enable_peer_topology:
                spent += self._dispatch_peer_task(task, record, report)
                continue
            spent += self._dispatch_autonomous_task(
                task, record, report, remaining_budget=remaining_budget - spent,
            )
        return spent

    # ------------------------------------------------------------------ #
    # Autonomous dispatch path
    # ------------------------------------------------------------------ #

    def _dispatch_autonomous_task(
        self,
        task: AgentTask,
        record: TaskExecutionRecord,
        report: GoalRunReport,
        *,
        remaining_budget: float,
    ) -> float:
        """Match, assign, dispatch, adjudicate, verify, and integrate one task."""
        assignments, trace = assign_owners(
            self.registry, self.matcher, self.gate,
            {task.task_id: task},
            strategy=self.config.assignment_strategy,
            remaining_goal_budget=remaining_budget,
        )
        self._record("assign", {"task_id": task.task_id, "assignments": assignments})
        owner_id = assignments.get(task.task_id)
        if owner_id is None:
            return self._apply_deterministic_fallback(task, record, report, trace)
        owner = self.agents[owner_id]
        self.peer.record_ownership(task.task_id, owner_id)
        record.owner_agent_id = owner_id
        result, spent = self._execute_with_recovery(task, owner, record, report)
        if result is None:
            return spent
        adjudication = self.adjudicator.validate(
            result, task, known_agent_versions={
                agent_id: agent.descriptor.agent_version for agent_id, agent in self.agents.items()
            },
        )
        if adjudication.decision is AdjudicationDecision.NEEDS_REVISION:
            result, adjudication = self._bounded_revision(task, owner, result, adjudication, record, report)
            if result is None:
                return spent
        if adjudication.decision is AdjudicationDecision.ACCEPTED and (
            task.requires_independent_verification
            or (self.config.require_verification_for_findings and self._writes_findings(task))
        ):
            adjudication = self._independent_verification(task, result, adjudication, owner_id, report)
        record.adjudication = adjudication.to_dict()
        attempt = record.attempts[-1] if record.attempts else AttemptRecord(owner_id, TaskOutcome.FAILED)
        attempt.adjudication = adjudication.to_dict()
        if adjudication.decision is not AdjudicationDecision.ACCEPTED:
            self._record_rejection(task, result, adjudication, record, attempt, owner_id)
            return spent
        publication = self.aggregator.publish(
            task, result, adjudication, self.registry.get(owner_id),
        )
        attempt = record.attempts[-1] if record.attempts else None
        self._finalize_publication(
            task, record, publication, report,
            result=result, adjudication=adjudication,
            retried=bool(attempt.retries) if attempt is not None else False,
        )
        return spent

    def _execute_with_recovery(
        self,
        task: AgentTask,
        owner: BaseBoundedAgent,
        record: TaskExecutionRecord,
        report: GoalRunReport,
    ) -> tuple[AgentResult | None, float]:
        """Dispatch with accept/reject, timeout reconciliation, and bounded retries."""
        snapshot = self.adapter.task_snapshot(task, owner.descriptor)
        decision = owner.accept_task(task, snapshot)
        attempt = AttemptRecord(owner.agent_id, TaskOutcome.RUNNING)
        record.attempts.append(attempt)
        if not decision.accepted:
            attempt.outcome = TaskOutcome.FAILED
            attempt.detail = f"agent rejected the envelope: {decision.reason}"
            self.bus.send(make_message(
                sender_id=owner.agent_id, recipient_id=SUPERVISOR_PRODUCER_ID,
                kind=MessageKind.REJECT, task_id=task.task_id, timestamp=self.clock.now(),
                payload={"reason": decision.reason},
                idempotency_key=f"reject:{task.task_id}:{owner.agent_id}",
            ), now=self.clock.now())
            self.registry.set_state(owner.agent_id, AgentLifecycleState.READY)
            return None, 0.0
        self.registry.heartbeat(owner.agent_id)
        self.registry.acquire_capacity(owner.agent_id)
        self.recovery.issue_lease(task.task_id, owner.agent_id)
        context = ExecutionContext(
            task=task, snapshot=snapshot, clock=self.clock,
            progress=self._progress_emitter(owner.agent_id, task),
            cancellation_requested=lambda task_id: task_id in owner._cancelled_tasks,
        )
        result, failure = self.executor.dispatch(
            owner, context, timeout_seconds=self.config.task_timeout_seconds,
        )
        spent = 0.0
        if result is not None:
            spent = float(result.resource_usage)
            attempt.outcome = TaskOutcome.COMPLETED
            attempt.duration_seconds = result.elapsed_seconds
        else:
            attempt.outcome = TaskOutcome.UNKNOWN if failure == "timeout" else TaskOutcome.FAILED
            attempt.detail = failure or ""
            reconciliation = self.recovery.reconcile_timeout(task, owner)
            if reconciliation.outcome is TaskOutcome.COMPLETED:
                recorded = owner.reconcile(task.task_id)
                if recorded is not None:
                    result = recorded
                    spent = float(result.resource_usage)
                    attempt.outcome = TaskOutcome.COMPLETED
                    attempt.detail = "result recovered by timeout reconciliation"
        if result is None and is_transient(failure):
            retries = 0
            while retries < owner.descriptor.retry.max_attempts:
                retries += 1
                context_retry = ExecutionContext(
                    task=task, snapshot=snapshot, clock=self.clock,
                    progress=self._progress_emitter(owner.agent_id, task),
                    cancellation_requested=lambda task_id: task_id in owner._cancelled_tasks,
                )
                result, failure = self.executor.dispatch(
                    owner, context_retry, timeout_seconds=self.config.task_timeout_seconds,
                )
                if result is not None:
                    spent += float(result.resource_usage)
                    attempt.outcome = TaskOutcome.COMPLETED
                    attempt.retries = retries
                    attempt.detail = f"recovered after {retries} transient retry(s)"
                    break
                attempt.retries = retries
        self.registry.release_capacity(owner.agent_id)
        self.registry.heartbeat(owner.agent_id)
        if result is None:
            self.registry.set_state(owner.agent_id, AgentLifecycleState.DEGRADED)
        return result, spent

    def _bounded_revision(
        self,
        task: AgentTask,
        owner: BaseBoundedAgent,
        result: AgentResult,
        adjudication: AdjudicationReport,
        record: TaskExecutionRecord,
        report: GoalRunReport,
    ) -> tuple[AgentResult | None, AdjudicationReport]:
        """Run bounded clarification and revision cycles (Section 3, step 12)."""
        for revision_round in range(1, self.config.revision_attempts + 1):
            self.bus.send(make_message(
                sender_id=SUPERVISOR_PRODUCER_ID, recipient_id=owner.agent_id,
                kind=MessageKind.CLARIFICATION, task_id=task.task_id,
                timestamp=self.clock.now(),
                payload={"revision_round": revision_round, "gaps": list(adjudication.criteria_gaps)},
                idempotency_key=f"clarify:{task.task_id}:{revision_round}",
            ), now=self.clock.now())
            snapshot = self.adapter.task_snapshot(task, owner.descriptor)
            context = ExecutionContext(
                task=task, snapshot=snapshot, clock=self.clock,
                progress=self._progress_emitter(owner.agent_id, task),
            )
            try:
                revised = owner.execute(context)
            except Exception:  # noqa: BLE001 - revision failure keeps the prior decision
                return result, adjudication
            revised = dataclasses.replace(
                revised, self_confidence=max(revised.self_confidence, 0.5),
                outputs={**dict(revised.outputs)},
            )
            revised_adjudication = self.adjudicator.validate(revised, task)
            if revised_adjudication.decision is AdjudicationDecision.ACCEPTED:
                return revised, revised_adjudication
            adjudication = revised_adjudication
        return result, adjudication

    def _independent_verification(
        self,
        task: AgentTask,
        result: AgentResult,
        adjudication: AdjudicationReport,
        owner_id: str,
        report: GoalRunReport,
    ) -> AdjudicationReport:
        """Require independent or deterministic verification for critical outputs (Step 42)."""
        verifiers = [
            agent for agent_id, agent in self.agents.items()
            if agent_id != owner_id and "macl.capability.evidence_verification" in agent.descriptor.capability_ids
            and self.registry.assignable(agent_id)
        ]
        if not verifiers:
            return AdjudicationReport(
                decision=AdjudicationDecision.REJECTED, task_id=task.task_id,
                agent_id=result.agent_id,
                reasons=tuple(adjudication.reasons) + ("no independent verifier was available",),
                evidence_errors=adjudication.evidence_errors,
                criteria_gaps=adjudication.criteria_gaps,
                fingerprint=adjudication.fingerprint,
                verified_by=adjudication.verified_by,
            )
        verifier = verifiers[0]
        snapshot = self.adapter.snapshot_for(verifier.descriptor)
        context_factory = lambda: ExecutionContext(
            task=task, snapshot=snapshot, clock=self.clock,
            progress=self._progress_emitter(verifier.agent_id, task),
        )
        verified = verify_independently(verifier, context_factory, self.adjudicator, adjudication)
        verifier_result = verifier.reconcile(task.task_id)
        if verifier_result is not None:
            passed = bool(verified.verified_by) and not verified.verified_by[-1].endswith(":failed")
            self.ledger.record(task, verifier_result, verified, verifier_disagreed=not passed)
        return verified

    def _record_rejection(
        self,
        task: AgentTask,
        result: AgentResult,
        adjudication: AdjudicationReport,
        record: TaskExecutionRecord,
        attempt: AttemptRecord,
        owner_id: str,
    ) -> None:
        """Apply rejection accounting, violations, and quarantine escalation."""
        record.final_outcome = TaskOutcome.FAILED
        attempt.outcome = TaskOutcome.FAILED
        attempt.adjudication = adjudication.to_dict()
        self.ledger.record(task, result, adjudication)
        schema_like = bool(adjudication.reasons or adjudication.criteria_gaps)
        if schema_like:
            try:
                self.recovery.record_violation(
                    owner_id, "invalid_output",
                    "; ".join(adjudication.reasons) or "; ".join(adjudication.criteria_gaps),
                )
            except QuarantineTriggered as exc:
                record.attempts.append(AttemptRecord(
                    owner_id, TaskOutcome.CANCELLED, detail=f"agent quarantined: {exc}",
                ))

    # ------------------------------------------------------------------ #
    # Deterministic module path (never bypasses the coordinator)
    # ------------------------------------------------------------------ #

    def _dispatch_module_task(
        self,
        task: AgentTask,
        record: TaskExecutionRecord,
        report: GoalRunReport,
    ) -> float:
        """Route deterministic module work through the existing coordinator (Section 6)."""
        if self.coordinator_gateway is None:
            record.final_outcome = TaskOutcome.BLOCKED
            record.attempts.append(AttemptRecord(
                "module", TaskOutcome.BLOCKED,
                detail="deterministic module routing requires an attached coordinator gateway; "
                       "the supervisor never calls module backends directly",
            ))
            return 0.0
        route = dict(task.module_route)
        cognitive_task = route.get("cognitive_task")
        if cognitive_task is None:
            record.final_outcome = TaskOutcome.BLOCKED
            record.attempts.append(AttemptRecord(
                "module", TaskOutcome.BLOCKED, detail="module route is missing a cognitive task",
            ))
            return 0.0
        outcome = self.coordinator_gateway.coordinate(cognitive_task)
        status = str(getattr(outcome, "status", "UNKNOWN"))
        attempt = AttemptRecord("module", TaskOutcome.RUNNING, detail=f"coordinator status: {status}")
        record.attempts.append(attempt)
        if status == "RESOLVED":
            attempt.outcome = TaskOutcome.COMPLETED
            record.final_outcome = TaskOutcome.COMPLETED
            self._integrated_task_ids.add(task.task_id)
            self._completed_keys.add(task.idempotency_key)
        else:
            attempt.outcome = TaskOutcome.FAILED
            record.final_outcome = TaskOutcome.FAILED
        return 2.0

    # ------------------------------------------------------------------ #
    # Debate and peer paths
    # ------------------------------------------------------------------ #

    def _run_debate_task(
        self,
        task: AgentTask,
        record: TaskExecutionRecord,
        report: GoalRunReport,
        *,
        remaining_budget: float,
    ) -> float:
        """Generate independent proposals, critique, and select by evidence (Steps 37-43)."""
        candidates = [
            agent for agent_id, agent in self.agents.items()
            if set(task.required_capabilities).issubset(set(agent.descriptor.capability_ids))
            and self.registry.assignable(agent_id)
        ]
        spent = 0.0
        proposals: list[AgentResult] = []
        for candidate in candidates:
            snapshot = self.adapter.task_snapshot(task, candidate.descriptor)
            context = ExecutionContext(
                task=task, snapshot=snapshot, clock=self.clock,
                progress=self._progress_emitter(candidate.agent_id, task),
            )
            result, failure = self.executor.dispatch(
                candidate, context, timeout_seconds=self.config.task_timeout_seconds,
            )
            if result is not None:
                proposals.append(result)
                spent += float(result.resource_usage)
            else:
                record.attempts.append(AttemptRecord(
                    candidate.agent_id, TaskOutcome.FAILED, detail=failure or "",
                ))
        critics = [
            agent for agent in self.agents.values()
            if "macl.capability.critic_review" in agent.descriptor.capability_ids
        ]
        debate_record = self.debate.run(
            task, proposals, critics[0] if critics else None,
        )
        report.debates.append(debate_record.to_dict())
        report.dissent.extend(debate_record.to_dict()["dissent"])
        selected = next(
            (proposal for proposal in proposals if proposal.agent_id == debate_record.selected_agent_id),
            None,
        )
        if selected is None:
            record.final_outcome = TaskOutcome.FAILED
            record.attempts.append(AttemptRecord(
                record.owner_agent_id or "debate", TaskOutcome.FAILED,
                detail=f"debate stopped: {debate_record.stop_reason} with no selected proposal",
            ))
            return spent
        record.owner_agent_id = selected.agent_id
        adjudication = self.adjudicator.validate(selected, task)
        attempt = AttemptRecord(selected.agent_id, TaskOutcome.COMPLETED)
        record.attempts.append(attempt)
        attempt.adjudication = adjudication.to_dict()
        record.adjudication = adjudication.to_dict()
        if adjudication.decision is not AdjudicationDecision.ACCEPTED:
            record.final_outcome = TaskOutcome.FAILED
            self.ledger.record(task, selected, adjudication)
            return spent
        if task.requires_independent_verification:
            adjudication = self._independent_verification(task, selected, adjudication, selected.agent_id, report)
            attempt.adjudication = adjudication.to_dict()
            if adjudication.decision is not AdjudicationDecision.ACCEPTED:
                record.final_outcome = TaskOutcome.FAILED
                self.ledger.record(task, selected, adjudication)
                return spent
        publication = self.aggregator.publish(
            task, selected, adjudication, self.registry.get(selected.agent_id),
        )
        self._finalize_publication(
            task, record, publication, report,
            result=selected, adjudication=adjudication,
        )
        return spent

    def _dispatch_peer_task(
        self,
        task: AgentTask,
        record: TaskExecutionRecord,
        report: GoalRunReport,
    ) -> float:
        """Let the recorded owner request bounded peer assistance (Steps 31-36)."""
        owner_id = record.owner_agent_id or self._select_owner(task)
        if owner_id is None:
            record.final_outcome = TaskOutcome.BLOCKED
            return 0.0
        owner = self.agents[owner_id]
        self.peer.record_ownership(task.task_id, owner_id)
        record.owner_agent_id = owner_id
        negotiation = self.peer.request_assistance(
            requesting_agent=owner,
            task=task,
            capability=task.required_capabilities[0],
            execute_with_peer=self._peer_execution,
            remaining_budget=task.cost_budget,
            now=self.clock.now(),
        )
        record.attempts.append(AttemptRecord(
            owner_id,
            TaskOutcome.COMPLETED if negotiation.outcome.value == "assisted" else TaskOutcome.FAILED,
            detail=negotiation.to_dict()["detail"],
        ))
        result = negotiation.result
        if result is None:
            record.final_outcome = TaskOutcome.BLOCKED
            return 0.0
        adjudication = self.adjudicator.validate(result, task)
        record.adjudication = adjudication.to_dict()
        if adjudication.decision is not AdjudicationDecision.ACCEPTED:
            record.final_outcome = TaskOutcome.FAILED
            self.ledger.record(task, result, adjudication)
            return float(result.resource_usage)
        publication = self.aggregator.publish(
            task, result, adjudication, self.registry.get(result.agent_id),
        )
        self._finalize_publication(
            task, record, publication, report,
            result=result, adjudication=adjudication,
        )
        return float(result.resource_usage)

    def _peer_execution(self, peer: BaseBoundedAgent, task: AgentTask) -> AgentResult | None:
        """Execute one task with an assisting peer under the supervisor's context."""
        snapshot = self.adapter.task_snapshot(task, peer.descriptor)
        decision = peer.accept_task(task, snapshot)
        if not decision.accepted:
            return None
        context = ExecutionContext(
            task=task, snapshot=snapshot, clock=self.clock,
            progress=self._progress_emitter(peer.agent_id, task),
        )
        result, _ = self.executor.dispatch(
            peer, context, timeout_seconds=self.config.task_timeout_seconds,
        )
        return result

    # ------------------------------------------------------------------ #
    # Fallbacks, helpers, and finalization
    # ------------------------------------------------------------------ #

    def _select_owner(self, task: AgentTask) -> str | None:
        """Return the best eligible agent for a task without assigning it yet."""
        assignments, _ = assign_owners(
            self.registry, self.matcher, self.gate, {task.task_id: task},
            strategy=self.config.assignment_strategy,
        )
        return assignments.get(task.task_id)

    def _apply_deterministic_fallback(
        self,
        task: AgentTask,
        record: TaskExecutionRecord,
        report: GoalRunReport,
        assignment_trace: Mapping[str, Any],
    ) -> float:
        """Keep deterministic specialists and explicit blocking as fallbacks (Step 14)."""
        capability = task.required_capabilities[0]
        text = self.adapter.current_input()
        outputs: dict[str, Any] | None = None
        if capability == "macl.capability.lexical_analysis":
            outputs = deterministic_lexical_hypotheses(text)
        elif capability == "macl.capability.syntactic_analysis":
            outputs = deterministic_syntactic_hypothesis(text)
        elif capability == "macl.capability.semantic_interpretation":
            outputs = deterministic_semantic_hypothesis(text)
        if outputs is None:
            record.final_outcome = TaskOutcome.BLOCKED
            ineligibility = [
                reason
                for item in assignment_trace.get("eligibility", {}).get(task.task_id, [])
                if not item["eligible"]
                for reason in item["reasons"]
            ]
            record.attempts.append(AttemptRecord(
                "supervisor", TaskOutcome.BLOCKED,
                detail="no agent qualified and no deterministic fallback exists; "
                       f"mandatory constraints were not relaxed: {ineligibility}",
            ))
            self._publish_unresolved_question(task, report)
            return 0.0
        record.used_deterministic_fallback = True
        record.attempts.append(AttemptRecord(
            "supervisor", TaskOutcome.COMPLETED,
            detail=f"deterministic fallback applied for {capability} because no agent "
                   "passed the eligibility gate",
        ))
        result = AgentResult(
            agent_id=SUPERVISOR_PRODUCER_ID, agent_version="1.0.0",
            task_id=task.task_id, goal_id=task.goal_id,
            source_revision=task.workspace_revision,
            outputs=outputs,
            provenance=ProvenanceTrail(
                agent_id=SUPERVISOR_PRODUCER_ID, task_id=task.task_id,
                operation="deterministic_fallback", workspace_revision=task.workspace_revision,
                transformation_history=("deterministic_fallback",),
            ),
            evidence_refs=(ArtifactRef(
                artifact_id="current_input",
                payload_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                location="workspace:current_input",
            ),),
            uncertainty=0.3,
            self_confidence=0.55,
            self_evaluation="deterministic fallback; no autonomous action selection",
            resource_usage=0.5,
        )
        adjudication = self.adjudicator.validate(result, task)
        record.adjudication = adjudication.to_dict()
        if adjudication.decision is not AdjudicationDecision.ACCEPTED:
            record.final_outcome = TaskOutcome.FAILED
            return 0.5
        publication = self._republish_as_supervisor(task, outputs)
        self._finalize_publication(
            task, record, publication, report,
            result=result, adjudication=adjudication,
        )
        return 0.5

    def _republish_as_supervisor(self, task: AgentTask, outputs: Mapping[str, Any]) -> PublicationRecord:
        """Publish a fallback finding under the supervisor's producer identity."""
        supervisor_descriptor = AgentDescriptor(
            agent_id=SUPERVISOR_PRODUCER_ID, agent_version="1.0.0",
            role="supervisor", autonomy=AutonomyLevel.SUPERVISOR,
            capability_ids=("macl.capability.supervision",),
            accepted_input_schema=SchemaSpec({"current_input": "string"}),
            produced_output_schema=SchemaSpec({"outputs": "any"}),
            permitted_workspace_fields=(
                "current_input", "lexical_hypotheses", "syntactic_hypotheses",
                "semantic_hypotheses", "unresolved_questions", "module_status",
                "artifact_references",
            ),
            writable_workspace_fields=(
                "lexical_hypotheses", "syntactic_hypotheses",
                "semantic_hypotheses", "unresolved_questions", "module_status",
                "artifact_references",
            ),
            autonomy_justification="the supervisor is the accountable coordinator for fallbacks",
        )
        capability = task.required_capabilities[0]
        field = {
            "macl.capability.lexical_analysis": "lexical_hypotheses",
            "macl.capability.syntactic_analysis": "syntactic_hypotheses",
            "macl.capability.semantic_interpretation": "semantic_hypotheses",
        }.get(capability, "semantic_hypotheses")
        evidence = ({"artifact_id": "current_input", "payload_hash": "", "location": "workspace:current_input"},)
        if field == "lexical_hypotheses" and isinstance(outputs.get("token_hypotheses"), Mapping):
            for token, candidates in outputs["token_hypotheses"].items():
                self.adapter.propose(
                    supervisor_descriptor, task, field=field,
                    payload={
                        "word": token,
                        "candidates": candidates if isinstance(candidates, list) else [candidates],
                        "candidate": str(
                            (candidates[0] or {}).get("candidate", "")
                            if isinstance(candidates, list) and candidates and isinstance(candidates[0], Mapping)
                            else candidates,
                        ),
                        "source_agent": SUPERVISOR_PRODUCER_ID,
                        "source_task": task.task_id,
                        "deterministic_fallback": True,
                    },
                    key=str(token),
                    evidence=evidence,
                    rules_applied=("macl:deterministic_fallback",),
                    base_revision=task.workspace_revision,
                )
            return PublicationRecord(
                task_id=task.task_id, agent_id=SUPERVISOR_PRODUCER_ID, field=field,
                decision=UpdateDecision.ACCEPTED.value,
                message="deterministic fallback published one keyed finding per token",
                workspace_revision=self.adapter.revision,
            )
        payload = {
            **dict(outputs),
            "candidate": str(outputs.get("interpretation", outputs.get("coverage", ""))),
            "source_agent": SUPERVISOR_PRODUCER_ID,
            "source_task": task.task_id,
            "deterministic_fallback": True,
        }
        key = next(iter(outputs.get("token_hypotheses", {})), None) if field == "lexical_hypotheses" else None
        validation = self.adapter.propose(
            supervisor_descriptor, task, field=field, payload=payload, key=key,
            evidence=evidence,
            rules_applied=("macl:deterministic_fallback",),
            base_revision=task.workspace_revision,
        )
        return PublicationRecord(
            task_id=task.task_id, agent_id=SUPERVISOR_PRODUCER_ID, field=field,
            decision=validation.decision.value, message=validation.message,
            workspace_revision=validation.workspace_revision,
            conflict_id=validation.conflict_id,
        )

    def _publish_unresolved_question(self, task: AgentTask, report: GoalRunReport) -> None:
        """Escalate a blocked task as an unresolved question (Step 14)."""
        supervisor_descriptor = AgentDescriptor(
            agent_id=SUPERVISOR_PRODUCER_ID, agent_version="1.0.0",
            role="supervisor", autonomy=AutonomyLevel.SUPERVISOR,
            capability_ids=("macl.capability.supervision",),
            accepted_input_schema=SchemaSpec({"current_input": "string"}),
            produced_output_schema=SchemaSpec({"question": "string"}),
            permitted_workspace_fields=("unresolved_questions", "module_status"),
            writable_workspace_fields=("unresolved_questions", "module_status"),
            autonomy_justification="the supervisor is the accountable coordinator",
        )
        self.adapter.propose(
            supervisor_descriptor, task, field="unresolved_questions",
            payload={
                "question": f"Task {task.task_id} is blocked: {task.objective}",
                "question_id": f"blocked:{task.task_id}", "resolved": False,
                "source_agent": SUPERVISOR_PRODUCER_ID,
            },
            key=f"blocked:{task.task_id}",
            finding_type="blocked_task_escalation",
            rules_applied=("macl:blocked_task",),
            base_revision=self.adapter.revision,
        )

    def _writes_findings(self, task: AgentTask) -> bool:
        """Return whether the task publishes a cognitive hypothesis field."""
        return bool(set(task.permitted_workspace_fields) & {
            "lexical_hypotheses", "syntactic_hypotheses", "semantic_hypotheses",
        })

    def _progress_emitter(self, agent_id: str, task: AgentTask) -> Callable[..., None]:
        """Build the bounded progress emitter for one agent and task (Step 20)."""
        def emit(*, phase: str, state: str, blockers: str = "", estimate: float | None = None) -> None:
            self.bus.send(make_message(
                sender_id=agent_id, recipient_id=SUPERVISOR_PRODUCER_ID,
                kind=MessageKind.PROGRESS, task_id=task.task_id,
                timestamp=self.clock.now(),
                payload={"phase": phase, "state": state, "blockers": blockers, "estimate": estimate},
                idempotency_key=f"progress:{task.task_id}:{agent_id}:{phase}:{state}",
            ), now=self.clock.now())
        return emit

    def _finalize_publication(
        self,
        task: AgentTask,
        record: TaskExecutionRecord,
        publication: Any,
        report: GoalRunReport,
        *,
        result: AgentResult | None = None,
        adjudication: AdjudicationReport | None = None,
        retried: bool = False,
    ) -> None:
        """Apply one publication decision and record its ledger outcome (Steps 48, 55)."""
        record.publication = publication.to_dict()
        decision = publication.decision
        integrated = decision in (
            UpdateDecision.ACCEPTED.value, UpdateDecision.MERGED.value,
            UpdateDecision.CONFLICT.value, UpdateDecision.DUPLICATE.value,
        )
        if integrated:
            record.final_outcome = TaskOutcome.COMPLETED
            self._integrated_task_ids.add(task.task_id)
            self._completed_keys.add(task.idempotency_key)
        elif decision == UpdateDecision.DEFERRED.value:
            record.final_outcome = TaskOutcome.PENDING
        else:
            record.final_outcome = TaskOutcome.FAILED
        if result is not None and adjudication is not None:
            effective = adjudication
            if not integrated and decision == UpdateDecision.REJECTED.value:
                effective = dataclasses.replace(
                    adjudication,
                    decision=AdjudicationDecision.REJECTED,
                    reasons=tuple(adjudication.reasons) + (
                        f"publication rejected: {publication.message}",
                    ),
                )
            self.ledger.record(task, result, effective, retried=retried)

    def close(self) -> None:
        """Release bounded execution resources."""
        self.executor.close()
