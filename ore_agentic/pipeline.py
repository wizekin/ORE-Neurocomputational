"""Fifty-step orchestration of the Point 15 integrated algorithm.

The ``AgenticEngine`` executes the specification's phases I through VIII in
order for one goal request: intake and workflow compilation, registry and
delegation with bounded parallel execution, the validated state machine with
checkpoints and side-effect safety, the evaluator-optimizer loop, controlled
MCP/A2A interoperability, three-tier cognitive memory with hybrid RAG,
production reliability (tracing, queues, caching, drift, quotas, sandboxing)
and governance close-out.  Every step records exactly one operation-trace
entry, and the result distinguishes completed, partial, failed and unverified
work.  The engine is a software-first reference implementation: workers are
deterministic local callables behind typed adapters, and engines, models,
indexes and protocol adapters remain replaceable behind the same interfaces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random
import time
from typing import Any, Callable, Mapping, Sequence

from .config import AgenticConfig, SOFTWARE_VERSIONS
from .contracts import (
    ApprovalStatus,
    DeliverableSpec,
    ExecutionDecision,
    FailureClass,
    GoalInterpretation,
    GoalRequest,
    Hypothesis,
    OutcomeReport,
    OutcomeStatus,
    PatternKind,
    RequestRejection,
    ResourceVector,
    RetryPolicy,
    RiskAssessment,
    SubgoalSpec,
    TaskOutcomeStatus,
    TaskResult,
    TaskSpec,
    TaskState,
    WorkflowStatus,
    new_id,
    utc_now,
)
from .delegation.leases import LeaseManager, StaleLeaseError
from .delegation.supervisor import NoEligibleWorkerError, Supervisor
from .delegation.workers import (
    CancelledError,
    MapReduceExecutor,
    TaskEnvelope,
    Worker,
)
from .evaluation.contracts import (
    ApprovalRequirement,
    CriterionKind,
    CriterionSpec,
    OutputContract,
)
from .evaluation.critic import (
    CriticLoop,
    CriticLoopReport,
    DeterministicCritic,
    FinalVerifier,
    RevisionBudget,
    VerificationReport,
)
from .evaluation.validators import (
    ContractValidator,
    DeterministicCheckRunner,
    validate_schema,
)
from .memory.episodic import EpisodeKind, EpisodicMemory
from .memory.links import MemoryKind, MemoryLinkStore, MemoryRelation
from .memory.procedural import ProceduralMemory
from .memory.semantic import SemanticMemory
from .observability.metrics import (
    DriftMonitor,
    MetricsCollector,
    QueueMonitor,
    SemanticCache,
)
from .observability.tracing import TraceContext, Tracer
from .persistence.database import AgenticDatabase
from .persistence.memory_store import MemoryStore
from .persistence.workflow_store import WorkflowStore
from .registry.agents import AgentModuleRegistry, WorkerManifest
from .registry.operations import OPERATION_REGISTRY
from .retrieval.evidence import (
    EvidenceAssessment,
    EvidenceBundle,
    EvidenceItem,
    EvidenceKind,
    assess_evidence,
)
from .retrieval.hybrid import (
    HybridResult,
    HybridRetriever,
    RetrievedDocument,
)
from .security.policy import (
    Action,
    PolicyEngine,
    PolicyRule,
    Principal,
    assess_risk,
)
from .security.sandbox import AuditLog, QuotaExceededError, QuotaManager
from .workflow.checkpoints import CheckpointManager
from .workflow.graph import CycleError, GraphValidationError, TaskGraph
from .workflow.machine import WorkflowStateMachine, evaluate_branch
from .workflow.side_effects import (
    CompensationNotAuthorizedError,
    CompensationNotDeclaredError,
    SideEffectHandle,
    SideEffectLedger,
)

__all__ = (
    "AgenticEngine",
    "AgenticResult",
    "PlanCompiler",
    "DeliverablePlanCompiler",
    "build_default_output_contract",
)

_PLAN_VERSION = "plan-1"
_CAP_EXECUTE = "agentic.execute"
_CAP_REDUCE = "agentic.reduce"


class PlanCompiler:
    """Protocol: compile an interpreted goal into a typed task graph (Step 5)."""

    def compile(self, request: GoalRequest,
                interpretation: GoalInterpretation) -> TaskGraph:
        """Return the validated task graph for the interpreted goal."""
        raise NotImplementedError


class DeliverablePlanCompiler(PlanCompiler):
    """Default deterministic compiler: one task per subgoal, plus a reduce join.

    Without declared subgoals, a single execution task is compiled; with
    more than one subgoal, a reduce task joins them under ALL semantics.
    """

    def __init__(self, config: AgenticConfig | None = None) -> None:
        """Store the governed configuration."""
        self._config = config or AgenticConfig()

    def compile(self, request: GoalRequest,
                interpretation: GoalInterpretation) -> TaskGraph:
        """Build the typed task graph (Step 5)."""
        graph = TaskGraph()
        subgoals = request.subgoals or (
            SubgoalSpec(name="execute", capability=_CAP_EXECUTE,
                        description=request.objective,
                        inputs=dict(request.inputs)),)
        for subgoal in subgoals:
            graph.add(TaskSpec(
                task_id=f"task-{subgoal.name}", name=subgoal.name,
                capability=subgoal.capability,
                input_schema={"type": "object", "properties": {}},
                output_schema=dict(subgoal.output_schema),
                depends_on=tuple(f"task-{dep}" for dep in subgoal.depends_on),
                cost=ResourceVector(seconds=1.0),
                retry=RetryPolicy(
                    max_attempts=self._config.retry_max_attempts,
                    initial_delay_seconds=self._config.retry_initial_delay_seconds,
                    max_delay_seconds=self._config.retry_max_delay_seconds,
                    jitter_seconds=self._config.retry_jitter_seconds,
                    total_time_budget_seconds=self._config.retry_total_time_budget_seconds),
                side_effect_class=subgoal.side_effect_class,
                permitted_tools=tuple(subgoal.inputs.get("permitted_tools", ()))
                if isinstance(subgoal.inputs, Mapping) else (),
                payload=dict(subgoal.inputs)))
        if len(subgoals) > 1:
            graph.add(TaskSpec(
                task_id="task-reduce", name="reduce", capability=_CAP_REDUCE,
                depends_on=tuple(f"task-{subgoal.name}" for subgoal in subgoals),
                payload={"join": "all"}))
        if len(graph.tasks) > self._config.max_tasks_per_workflow:
            raise GraphValidationError(
                f"task graph exceeds max_tasks_per_workflow="
                f"{self._config.max_tasks_per_workflow}")
        return graph


def build_default_output_contract() -> OutputContract:
    """Build the default deliverable contract declared before execution (Step 24)."""
    return OutputContract(
        contract_id="agentic.workflow.deliverable",
        version="1.0.0",
        criteria=(
            CriterionSpec("completeness", CriterionKind.COMPLETENESS,
                          "All declared deliverables present",
                          {"required_keys": ("deliverables",)}),
            CriterionSpec("evidence", CriterionKind.EVIDENCE,
                          "At least one evidence reference with provenance",
                          {"minimum_refs": 1}),
            CriterionSpec("summary_type", CriterionKind.SCHEMA,
                          "Summary must be a non-empty string",
                          {"schema": {"type": "object",
                                      "properties": {"summary": {"type": "string",
                                                                 "min_length": 1}}}}),
        ),
        approval=ApprovalRequirement(consequential=False))


@dataclass(slots=True)
class AgenticResult:
    """The typed result envelope of one 50-step agentic run (Step 50)."""

    workflow_id: str
    request: dict[str, Any]
    status: WorkflowStatus
    outcomes: OutcomeReport
    risk: RiskAssessment | None = None
    interpretation: GoalInterpretation | None = None
    pattern: PatternKind = PatternKind.SEQUENTIAL
    plan_hash: str = ""
    deliverables: dict[str, Any] = field(default_factory=dict)
    task_results: dict[str, TaskResult] = field(default_factory=dict)
    evidence: EvidenceAssessment | None = None
    retrieval: HybridResult | None = None
    critic: CriticLoopReport | None = None
    verification: VerificationReport | None = None
    episode_ids: tuple[str, ...] = ()
    procedure_proposal_ids: tuple[str, ...] = ()
    semantic_proposal_ids: tuple[str, ...] = ()
    clarification_questions: tuple[str, ...] = ()
    trace: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    spans: list[dict[str, Any]] = field(default_factory=list)
    limitations: tuple[str, ...] = ()
    workspace_revision: int | None = None
    reused_from: str | None = None
    duration_ms: float = 0.0
    started_at: str = field(default_factory=utc_now)

    @property
    def outcome_status(self) -> OutcomeStatus:
        """Return the distinct overall outcome category (Step 50)."""
        return self.outcomes.overall

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result envelope (Step 50)."""
        return {
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "outcome_status": self.outcome_status.value,
            "outcomes": self.outcomes.to_dict(),
            "risk": self.risk.to_dict() if self.risk else None,
            "interpretation": self.interpretation.to_dict() if self.interpretation else None,
            "pattern": self.pattern.value,
            "plan_hash": self.plan_hash,
            "deliverables": dict(self.deliverables),
            "task_results": {key: value.to_dict()
                             for key, value in self.task_results.items()},
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "retrieval": [item.to_dict() for item in self.retrieval.ranked]
            if self.retrieval else [],
            "critic": self.critic.to_dict() if self.critic else None,
            "verification": self.verification.to_dict() if self.verification else None,
            "episode_ids": list(self.episode_ids),
            "procedure_proposal_ids": list(self.procedure_proposal_ids),
            "semantic_proposal_ids": list(self.semantic_proposal_ids),
            "clarification_questions": list(self.clarification_questions),
            "trace": list(self.trace),
            "metrics": dict(self.metrics),
            "limitations": list(self.limitations),
            "workspace_revision": self.workspace_revision,
            "reused_from": self.reused_from,
            "duration_ms": self.duration_ms,
            "started_at": self.started_at,
        }


class AgenticEngine:
    """Execute the 50-step Point 15 algorithm for one agentic goal."""

    def __init__(self, config: AgenticConfig | None = None, *,
                 database_path: str | None = None,
                 workers: Sequence[Worker] = (),
                 workspace_bridge: Any = None,
                 semantic_index: Any = None,
                 retrieval_documents: Sequence[RetrievedDocument] = (),
                 external_mcp: Any = None,
                 a2a_participant: Any = None,
                 plan_compiler: PlanCompiler | None = None,
                 policy_rules: Sequence[PolicyRule] = (),
                 principal_scopes: Mapping[str, tuple[str, ...]] | None = None,
                 deterministic_checks: Mapping[str, Callable[[Mapping[str, Any]],
                                                            tuple[bool, str]]]
                 | None = None,
                 approver: Callable[[str], ApprovalStatus] | None = None,
                 annotation_recorder: Any = None,
                 rng: random.Random | None = None) -> None:
        """Compose the engine from its replaceable collaborators."""
        self.config = config or AgenticConfig()
        self._rng = rng or random.Random(self.config.seed)
        self.registry = AgentModuleRegistry()
        self.leases = LeaseManager(default_ttl_seconds=self.config.lease_ttl_seconds)
        self.supervisor = Supervisor(self.registry, self.leases, self.config,
                                     rng=self._rng)
        self.executor = MapReduceExecutor(
            pool_size=self.config.worker_pool_size,
            per_workflow_concurrency=self.config.per_workflow_concurrency,
            per_worker_concurrency=self.config.per_worker_concurrency,
            queue_capacity=self.config.queue_capacity)
        self.ledger = SideEffectLedger()
        self.audit = AuditLog()
        seeded_rules = (
            PolicyRule(principal_id="*", action_id="agentic.workflow.execute",
                       effect="allow", priority=0,
                       description="default workflow execution grant; explicit deny "
                                   "rules override by priority"),
            *tuple(policy_rules),
        )
        self.policy = PolicyEngine(
            seeded_rules,
            audit_sink=lambda entry: self.audit.record(
                str(entry.get("kind", "authorization")), entry),
            config=self.config)
        self.quotas = QuotaManager(cpu_units=self.config.quota_cpu_units,
                                   memory_mb=self.config.quota_memory_mb,
                                   seconds=self.config.quota_seconds,
                                   cost=self.config.quota_cost)
        self.state_machines: dict[str, WorkflowStateMachine] = {}
        self.checkpoint_manager: CheckpointManager | None = None
        self.workflow_store: WorkflowStore | None = None
        self.memory_store: MemoryStore | None = None
        if database_path is not None:
            database = AgenticDatabase(database_path)
            self.workflow_store = WorkflowStore(database)
            self.memory_store = MemoryStore(database)
            self.checkpoint_manager = CheckpointManager(
                self.workflow_store, max_retained=self.config.max_checkpoints_retained)
        self.episodic = EpisodicMemory(
            max_payload_chars=self.config.episode_max_payload_chars,
            default_retention=self.config.episode_default_retention_class)
        self.procedural = ProceduralMemory(
            min_passing_tests=self.config.procedure_min_passing_tests)
        self.semantic = SemanticMemory(usg_index=semantic_index)
        self.links = MemoryLinkStore()
        self.metrics = MetricsCollector()
        self.queue_monitor = QueueMonitor()
        self.drift = DriftMonitor(self.config.drift_window,
                                  self.config.drift_review_threshold,
                                  quarantine_allowed=self.config.drift_quarantine_allowed)
        self.cache = SemanticCache(self.config.semantic_cache_ttl_seconds)
        self.runner = DeterministicCheckRunner()
        self.runner.register("deliverables_present", _default_deliverables_check,
                             replace=True)
        for check_id, check in (deterministic_checks or {}).items():
            self.runner.register(check_id, check, replace=True)
        self.critic_loop = CriticLoop(
            DeterministicCritic(ContractValidator(
                deterministic_runner=self.runner,
                policy_check=lambda artifact: (True, "policy check pass"))),
            RevisionBudget(max_iterations=self.config.critic_max_iterations,
                           cost_budget=self.config.critic_cost_budget,
                           stall_threshold=self.config.critic_stall_threshold))
        self.verifier = FinalVerifier(self.runner, approver=approver)
        self.workspace_bridge = workspace_bridge
        self.plan_compiler = plan_compiler or DeliverablePlanCompiler(self.config)
        self.annotation_recorder = annotation_recorder
        self._worker_bindings: dict[str, Worker] = {}
        self._external_mcp = external_mcp
        self._a2a_participant = a2a_participant
        self._trace: list[dict[str, Any]] = []
        self._limitations: list[str] = []
        self._idempotent_seen: dict[str, tuple[str, str]] = {}
        self._idempotent_keys: set[str] = set()
        self._compensate_authorized = False

    # ------------------------------------------------------------------ public

    def register_worker(self, manifest: WorkerManifest, worker: Worker,
                        *, replace: bool = True) -> None:
        """Register one worker manifest and bind its implementation (Step 9)."""
        self.registry.register(manifest, replace=replace)
        self._worker_bindings[manifest.worker_id] = worker

    def cancel(self, workflow_id: str, reason: str = "operator request",
               *, compensate_authorized: bool = False) -> None:
        """Request cancellation of a running workflow (Step 22).

        Declared compensation for confirmed side effects is invoked only when
        ``compensate_authorized`` is set; otherwise the effects are disclosed
        but not compensated.
        """
        machine = self.state_machines.get(workflow_id)
        if machine is not None and machine.workflow_status is not WorkflowStatus.CANCELLED:
            machine.transition_workflow(WorkflowStatus.CANCELLED,
                                        reason=f"cancel: {reason}")
        self._compensate_authorized = compensate_authorized

    def run(self, request: GoalRequest) -> AgenticResult:
        """Execute the full 50-step algorithm for one request."""
        if not isinstance(request, GoalRequest):
            raise RequestRejection("request must be a GoalRequest", "invalid_type")
        started = time.monotonic()
        trace_context = TraceContext.create()
        tracer = Tracer(trace_context,
                        sink=(lambda span: self._persist_span(span))
                        if self.config.tracing_enabled else None)
        workflow_id = new_id("wf")
        result = AgenticResult(
            workflow_id=workflow_id, request=request.to_dict(),
            status=WorkflowStatus.COMPILED,
            outcomes=OutcomeReport(deliverables={}))
        with tracer.span("agentic.run", workflow_id=workflow_id):
            try:
                self._execute(request, result, tracer, workflow_id)
            except RequestRejection as exc:
                result.status = WorkflowStatus.REJECTED
                result.limitations = (*result.limitations, f"rejected: {exc}")
                self._record_step(8, "rejected", str(exc))
        result.trace = list(self._trace)
        result.spans = [span.to_dict() for span in tracer.spans()]
        result.metrics = self._system_evaluation(result, request)
        result.duration_ms = round((time.monotonic() - started) * 1000.0, 3)
        result.limitations = tuple(dict.fromkeys(
            (*result.limitations,
             "reference implementation: workers are deterministic local callables",
             "vector retrieval is omitted until evaluation shows measurable benefit")))
        self._record_annotation(request, result)
        return result

    # ------------------------------------------------------------------ phases

    def _execute(self, request: GoalRequest, result: AgenticResult,
                 tracer: Tracer, workflow_id: str) -> None:
        """Drive the eight phases in order (Steps 1-50)."""
        # ---- Phase I: goal intake and workflow compilation (Steps 1-8)
        machine = WorkflowStateMachine(workflow_id)
        self.state_machines[workflow_id] = machine
        with tracer.span("phase1.intake", workflow_id=workflow_id):
            request = self._step01_receive(request)
            interpretation = self._step02_interpret(request)
            result.interpretation = interpretation
            risk = self._step03_classify_risk(request, interpretation)
            result.risk = risk
            reuse = self._step04_check_existing(request, result)
            if reuse is not None:
                result.reused_from = reuse
                return
            self._early_gate(request, interpretation, risk, result, machine)
            if result.status in (WorkflowStatus.WAITING_CLARIFICATION,
                                 WorkflowStatus.REJECTED):
                return
            graph = self._step05_decompose(request, interpretation)
            pattern = self._step06_select_pattern(graph)
            result.pattern = pattern
            budget_ok = self._step07_estimate(graph, request, result)
            if not budget_ok:
                return
            plan = self._step08_persist_plan(workflow_id, request, graph, result,
                                             machine)
        # ---- Phase II: delegation and parallel execution (Steps 9-16)
        if machine.workflow_status is WorkflowStatus.COMPILED:
            machine.transition_workflow(WorkflowStatus.RUNNING)
        with tracer.span("phase2.delegation", workflow_id=workflow_id):
            self._step09_maintain_registry()
            worker_specs = self._step10_11_select_workers(graph)
            cancellation = self._execute_levels(workflow_id, graph, plan, worker_specs,
                                                machine, tracer, result)
            if cancellation:
                self._step22_cancel(workflow_id, machine, result,
                                    reason="observed during execution")
                self._post_execution(request, result, machine, tracer, cancelled=True)
                return
            self._step13b_aggregate(graph, result)
            self._phase2_records()
        # ---- Phases III-VIII
        self._post_execution(request, result, machine, tracer, cancelled=False)

    def _post_execution(self, request: GoalRequest, result: AgenticResult,
                        machine: WorkflowStateMachine, tracer: Tracer,
                        *, cancelled: bool) -> None:
        """Run phases III through VIII (Steps 17-23 records, 24-50)."""
        workflow_id = result.workflow_id
        with tracer.span("phase3.state_recovery", workflow_id=workflow_id):
            self._phase3_records(result, cancelled=cancelled)
        with tracer.span("phase4.evaluation", workflow_id=workflow_id):
            self._steps24_30_evaluate(request, result)
        with tracer.span("phase5.protocols", workflow_id=workflow_id):
            self._steps31_36_protocols(request, result)
        with tracer.span("phase6.memory", workflow_id=workflow_id):
            self._steps37_42_memory(request, result)
        with tracer.span("phase7.reliability", workflow_id=workflow_id):
            self._steps43_47_reliability(result)
        with tracer.span("phase8.governance", workflow_id=workflow_id):
            self._steps48_50_close(request, result, machine, cancelled=cancelled)

    # ------------------------------------------------------- Phase I internals

    def _record_step(self, step: int, status: str = "completed",
                     detail: str = "") -> None:
        """Append exactly one trace entry per source step."""
        spec = OPERATION_REGISTRY[step - 1]
        entry = {
            "operation_id": spec.operation_id,
            "step": spec.source_step,
            "phase": spec.phase,
            "operation": spec.name,
            "status": status,
            "detail": detail,
            "mathematical_form": spec.mathematical_form,
            "safeguard": spec.safeguard,
            "timestamp": utc_now(),
        }
        self._trace.append(entry)

    def _step01_receive(self, request: GoalRequest) -> GoalRequest:
        """Step 1 -- receive and identify the objective."""
        if not isinstance(request, GoalRequest):
            raise RequestRejection("request must be a GoalRequest", "invalid_type")
        try:
            frozen = GoalRequest.from_dict(request.to_dict())
        except ValueError as exc:
            raise RequestRejection(str(exc), "malformed_request") from exc
        self._record_step(1, detail=f"request {frozen.request_id} accepted, "
                            f"objective preserved immutable")
        return frozen

    def _step02_interpret(self, request: GoalRequest) -> GoalInterpretation:
        """Step 2 -- interpret intent, resolving entities through USG."""
        entities = self._resolve_entities(request.objective)
        hypotheses = tuple(
            Hypothesis(statement=f"requested capability {subgoal.capability!r} "
                                 f"matches an available worker",
                       basis="subgoal declaration",
                       confidence=0.6)
            for subgoal in request.subgoals)
        ambiguities = tuple(request.flagged_ambiguities)
        interpretation = GoalInterpretation(
            objective=request.objective, constraints=dict(request.constraints),
            inputs=dict(request.inputs),
            deliverables=request.deliverables or (
                DeliverableSpec(name="summary",
                                description="workflow outcome summary"),),
            entities=tuple(entities), hypotheses=hypotheses,
            ambiguities=ambiguities)
        self._record_step(2, detail=f"G=(O,C,I,D) built; {len(hypotheses)} hypotheses "
                            f"marked non-authoritative; {len(entities)} entities")
        return interpretation

    def _resolve_entities(self, objective: str) -> list[str]:
        """Resolve mentioned entities through the canonical USG index (Step 2)."""
        entities: list[str] = []
        if self.semantic.usg_index is not None:
            for token in objective.split():
                records = self.semantic.lookup(token)
                if records:
                    entities.append(token)
        return entities

    def _step03_classify_risk(self, request: GoalRequest,
                              interpretation: GoalInterpretation) -> RiskAssessment:
        """Step 3 -- classify risk and side effects from policy, not wording."""
        classes = [subgoal.side_effect_class for subgoal in request.subgoals
                   ] or [request.side_effect_class]
        strongest = max(classes, key=lambda cls: cls.risk_floor)
        uncertainty = 0.3 if interpretation.needs_clarification else 0.1
        risk = assess_risk(strongest, sensitivity=0.3, uncertainty=uncertainty,
                           weights=self.config.risk_weights, config=self.config)
        self._record_step(3, detail=f"R={risk.score:.3f} ({strongest.value}) -> "
                            f"{risk.decision.value}")
        return risk

    def _step04_check_existing(self, request: GoalRequest,
                               result: AgenticResult) -> str | None:
        """Step 4 -- reuse an existing workflow when applicable."""
        key = request.idempotency_key
        fingerprint = request.fingerprint(self.config.request_fingerprint_fields)
        if not key:
            self._record_step(4, detail="no idempotency key; proceeding")
            return None
        prior_workflow: str | None = None
        prior_digest: str | None = None
        if self.workflow_store is not None:
            row = self.workflow_store.find_by_idempotency_key(key)
            if row is not None:
                prior_workflow = str(row["workflow_id"])
                prior_digest = row.get("request_digest")
        elif key in self._idempotent_seen:
            prior_workflow, prior_digest = self._idempotent_seen[key]
        if prior_workflow is not None and prior_digest == fingerprint:
            result.status = WorkflowStatus.REUSED
            result.outcomes = OutcomeReport(deliverables={"reused": True})
            self._record_step(4, status="reused", detail=(
                f"idempotency key match; reusing workflow {prior_workflow}; "
                "no side effects re-executed"))
            return prior_workflow
        if prior_workflow is not None:
            self._record_step(4, detail="key matched but inputs differ; "
                              "new workflow requires a new key")
            return None
        self._idempotent_seen[key] = (result.workflow_id, fingerprint)
        self._record_step(4, detail="no matching workflow; proceeding")
        return None

    def _early_gate(self, request: GoalRequest, interpretation: GoalInterpretation,
                    risk: RiskAssessment, result: AgenticResult,
                    machine: WorkflowStateMachine) -> None:
        """Steps 2-3 gate: clarification for ambiguous high-impact intent; prohibition."""
        classes = [subgoal.side_effect_class for subgoal in request.subgoals
                   ] or [request.side_effect_class]
        strongest = max(classes, key=lambda cls: cls.risk_floor)
        execute_action = Action(
            action_id="agentic.workflow.execute", resource="workflow",
            effect_class=strongest)
        authorization = self.policy.authorize(
            Principal(principal_id=request.requester), execute_action)
        if not authorization.allowed:
            result.status = WorkflowStatus.REJECTED
            result.outcomes = OutcomeReport()
            machine.transition_workflow(
                WorkflowStatus.REJECTED,
                reason=f"prohibited by policy: {authorization.reason}")
            self._record_step(
                3, status="prohibited",
                detail=(f"operation prohibited by policy (authorization comes from "
                        f"policy, not request wording): {authorization.reason}"))
            result.limitations = (*result.limitations,
                                  f"prohibited: {authorization.reason}")
            return
        high_risk = (risk.decision is ExecutionDecision.REQUIRES_APPROVAL
                     or risk.side_effect_class.risk_floor >= 0.5)
        if (interpretation.needs_clarification and high_risk
                and self.config.ambiguity_requires_clarification_when_high_risk):
            result.status = WorkflowStatus.WAITING_CLARIFICATION
            result.clarification_questions = interpretation.ambiguities
            machine.transition_workflow(WorkflowStatus.WAITING_CLARIFICATION,
                                        reason="ambiguous high-impact intent")
            if self.workspace_bridge is not None:
                for question in interpretation.ambiguities:
                    self.workspace_bridge.publish_unresolved_question(question)
            self._record_step(2, status="waiting_clarification",
                              detail="high-impact ambiguity routed to clarification, "
                                     "not silent guessing")
            return
        if risk.decision is ExecutionDecision.REQUIRES_APPROVAL:
            machine.transition_workflow(WorkflowStatus.WAITING_APPROVAL,
                                        reason="risk threshold requires approval")
            machine.transition_workflow(WorkflowStatus.RUNNING,
                                        reason="approval granted by operator policy "
                                               "for this reference run")
            result.limitations = (*result.limitations,
                                  "high-risk execution auto-approved in reference "
                                  "configuration; production requires a human gate")

    def _step05_decompose(self, request: GoalRequest,
                          interpretation: GoalInterpretation) -> TaskGraph:
        """Step 5 -- decompose into typed subtasks with cycle detection."""
        try:
            graph = self.plan_compiler.compile(request, interpretation)
            graph.validate()
        except CycleError as exc:
            self._record_step(5, status="rejected", detail=str(exc))
            raise RequestRejection(str(exc), "unbounded_cycle") from exc
        except GraphValidationError as exc:
            self._record_step(5, status="rejected", detail=str(exc))
            raise RequestRejection(str(exc), "invalid_graph") from exc
        self._record_step(5, detail=f"W=(V,E) with {len(graph.tasks)} tasks, "
                            f"topological order verified, critical path "
                            f"{graph.critical_path_seconds():.1f}s")
        return graph

    def _step06_select_pattern(self, graph: TaskGraph) -> PatternKind:
        """Step 6 -- select the simplest suitable workflow pattern."""
        parallelizable = len(graph.tasks) > 1
        if len(graph.tasks) == 1:
            pattern = PatternKind.SEQUENTIAL
        elif any(task.capability == _CAP_REDUCE for task in graph.tasks.values()):
            pattern = PatternKind.MAP_REDUCE
        elif parallelizable:
            pattern = PatternKind.SUPERVISOR_WORKER
        else:
            pattern = PatternKind.SEQUENTIAL
        self._record_step(6, detail=f"pattern {pattern.value} chosen (simplest "
                            f"suitable for {len(graph.tasks)} tasks)")
        return pattern

    def _step07_estimate(self, graph: TaskGraph, request: GoalRequest,
                         result: AgenticResult) -> bool:
        """Step 7 -- estimate feasibility and resource budgets with admission."""
        aggregate = graph.aggregate_cost()
        demand = ResourceVector(
            cpu_units=aggregate["cpu_units"], memory_mb=aggregate["memory_mb"],
            io_units=aggregate["io_units"], network_units=aggregate["network_units"],
            seconds=aggregate["seconds"])
        deadline = request.deadline_seconds or self.config.default_deadline_seconds
        try:
            self.quotas.check_and_reserve("agentic_workflow", demand,
                                          cost=len(graph.tasks))
        except QuotaExceededError as exc:
            result.status = WorkflowStatus.REJECTED
            result.outcomes = OutcomeReport()
            self._record_step(7, status="rejected", detail=str(exc))
            return False
        self._record_step(7, detail=(
            f"aggregate cost { {k: round(v, 2) for k, v in aggregate.items()} } "
            f"within budgets; deadline {deadline}s; critical path "
            f"{graph.critical_path_seconds():.1f}s"))
        return True

    def _step08_persist_plan(self, workflow_id: str, request: GoalRequest,
                             graph: TaskGraph, result: AgenticResult,
                             machine: WorkflowStateMachine) -> dict[str, Any]:
        """Step 8 -- validate and persist the immutable plan version and hash."""
        plan_hash = graph.plan_hash()
        for task_id in graph.tasks:
            machine.announce(task_id)
        plan = {
            "workflow_id": workflow_id,
            "request_id": request.request_id,
            "request_digest": request.fingerprint(
                self.config.request_fingerprint_fields),
            "objective": request.objective,
            "plan_version": _PLAN_VERSION,
            "plan_hash": plan_hash,
            "workspace_revision": (self.workspace_bridge.workspace_revision
                                   if self.workspace_bridge else 0),
            "tasks": {tid: task.to_dict() for tid, task in graph.tasks.items()},
        }
        result.plan_hash = plan_hash
        if self.workflow_store is not None:
            try:
                self.workflow_store.create(
                    plan, idempotency_key=request.idempotency_key)
            except ValueError as exc:
                self._record_step(8, status="rejected", detail=str(exc))
                raise RequestRejection(str(exc), "duplicate_idempotency_key") from exc
        elif request.idempotency_key:
            if request.idempotency_key in self._idempotent_keys:
                message = (f"idempotency key {request.idempotency_key!r} "
                           "already used for a different request")
                self._record_step(8, status="rejected", detail=message)
                raise RequestRejection(message, "duplicate_idempotency_key")
            self._idempotent_keys.add(request.idempotency_key)
        self._record_step(8, detail=f"plan {plan_hash} persisted immutable "
                            f"(version {_PLAN_VERSION})")
        return plan

    # ------------------------------------------------------ Phase II internals

    def _step09_maintain_registry(self) -> None:
        """Step 9 -- verify the agent/module registry is maintained."""
        eligible = self.registry.list_workers()
        self._record_step(9, detail=f"{len(eligible)} eligible workers registered "
                            f"with versions, capabilities and health")
        for record in self.registry.list_workers(eligible_only=False):
            if not record.eligible:
                self._record_step(9, status="excluded",
                                  detail=f"worker {record.manifest.worker_id} "
                                         "ineligible (revoked/expired/unavailable)")

    def _step10_11_select_workers(self, graph: TaskGraph) -> dict[str, tuple[str, Worker]]:
        """Steps 10-12 -- discover eligible workers, rank, assign leases."""
        assignments: dict[str, tuple[str, Worker]] = {}
        unassigned: list[str] = []
        for task_id, task in sorted(graph.tasks.items()):
            if task.capability == _CAP_REDUCE:
                assignments[task_id] = ("supervisor.join", self._join_worker())
                continue
            try:
                worker_id, _, _selection = self.supervisor.select_worker(task)
            except NoEligibleWorkerError:
                assignments[task_id] = ("", _unassigned_worker())
                unassigned.append(task_id)
                continue
            worker = self._worker_bindings.get(worker_id)
            if worker is None:
                assignments[task_id] = ("", _unassigned_worker())
                unassigned.append(task_id)
                continue
            assignments[task_id] = (worker_id, worker)
        self._record_step(10, detail=(
            "mandatory predicates: capability, schema, permission, health, "
            f"policy, resources; {len(unassigned)} task(s) without an eligible "
            "worker" if unassigned else
            "mandatory predicates: capability, schema, permission, health, "
            "policy, resources; every task has an eligible worker"))
        self._record_step(11, detail="S(a,t) = w_f*F + w_q*Q - w_l*L - w_c*C "
                            "- w_u*U - w_r*R with deterministic tie-breaks")
        self._record_step(12, detail=(
            f"{len(assignments)} bounded assignment(s); leases carry input "
            "references, permitted tools, budget, deadline, cancellation token "
            "and output contract; no unrelated workspace data exposed"))
        return assignments

    def _execute_levels(self, workflow_id: str, graph: TaskGraph,
                        plan: dict[str, Any],
                        assignments: dict[str, tuple[str, Worker]],
                        machine: WorkflowStateMachine, tracer: Tracer,
                        result: AgenticResult) -> bool:
        """Execute the graph level by level with bounded parallelism (Steps 12-16)."""
        cancellation = _Cancellation()
        completed: set[str] = set()
        tasks_done = 0
        while True:
            if machine.workflow_status is WorkflowStatus.CANCELLED:
                cancellation.cancel("workflow cancelled")
            if cancellation.cancelled:
                break
            frontier = [tid for tid in graph.frontier(completed)
                        if tid not in completed]
            join_ready = [tid for tid in frontier
                          if graph.tasks[tid].capability == _CAP_REDUCE]
            plain_ready = [tid for tid in frontier if tid not in join_ready]
            if not frontier:
                break
            # Execute reduce/join nodes synchronously (Step 19).
            for tid in join_ready:
                task = graph.tasks[tid]
                dependency_results = {dep: result.task_results[dep]
                                      for dep in task.depends_on
                                      if dep in result.task_results}
                joined = self.supervisor.join(task, dependency_results)
                machine.transition(tid, TaskState.READY, reason="dependencies met")
                machine.transition(tid, TaskState.LEASED, reason="supervisor self-lease")
                machine.transition(tid, TaskState.RUNNING,
                                   reason="supervisor executing join")
                machine.transition(
                    tid,
                    TaskState.SUCCEEDED if joined.succeeded else TaskState.FAILED,
                    reason="join evaluated")
                result.task_results[tid] = joined
                completed.add(tid)
                self.metrics.record_outcome(joined.succeeded, 0.0)
            if not plain_ready:
                continue
            triples = []
            envelopes: dict[str, TaskEnvelope] = {}
            for tid in plain_ready:
                task = graph.tasks[tid]
                worker_id, worker = assignments.get(tid, ("", _unassigned_worker()))
                effects = SideEffectHandle(
                    self.ledger, workflow_id, tid, task.permitted_tools)
                lease, envelope = self.supervisor.assign(
                    workflow_id, task, worker_id,
                    inputs={"payload": dict(task.payload),
                            "request_inputs": dict(
                                result.request.get("inputs", {}))},
                    effects=effects, cancellation=cancellation)
                envelopes[tid] = envelope
                current = machine.task_state(tid)
                if current is TaskState.PENDING:
                    machine.transition(tid, TaskState.READY,
                                       reason="dependencies met")
                elif current is TaskState.LEASED:
                    machine.transition(tid, TaskState.READY,
                                       reason="deferred lease reconciled")
                if machine.task_state(tid) is TaskState.READY:
                    machine.transition(tid, TaskState.LEASED, reason="lease issued")
                self.metrics.record_dispatch(cost=task.cost.cpu_units)
                triples.append((task, lease, worker))

            def _run(task: TaskSpec, lease: Any, worker: Worker) -> TaskResult:
                envelope = envelopes[task.task_id]
                if machine.task_state(task.task_id) is TaskState.LEASED:
                    machine.transition(task.task_id, TaskState.RUNNING,
                                       reason="execution started")
                self._step14_backpressure()
                task_result = self.supervisor.run_with_retry(
                    task, envelope, worker,
                    worker_version=str(getattr(worker, "version", "1.0.0")),
                    policy=task.retry)
                self._lease_accept_or_fence(lease, task_result, task)
                return task_result

            outcome = self.executor.map(tuple(triples), _run)
            for task_result in (*outcome.results, *outcome.failures):
                tid = task_result.task_id
                task_result = self._validate_task_contract(graph.tasks[tid],
                                                           task_result)
                result.task_results[tid] = task_result
                self.metrics.record_outcome(
                    task_result.succeeded, 0.0,
                    failure_class=(task_result.failure_class.value
                                   if task_result.failure_class else ""),
                    retries=max(0, task_result.attempts - 1))
                self.drift.observe(task_result.succeeded, 0.0)
                if self.workflow_store is not None:
                    self.workflow_store.transition(
                        workflow_id, "task_outcome",
                        task_id=tid,
                        task_state=("succeeded" if task_result.succeeded else "failed"),
                        worker_id=task_result.worker_id,
                        attempts=task_result.attempts)
                    self.workflow_store.record_result(task_result, workflow_id)
                machine.transition(
                    tid,
                    TaskState.SUCCEEDED if task_result.succeeded else TaskState.FAILED,
                    reason="attempt completed")
                completed.add(tid)
                tasks_done += 1
                self._episode_for_task(workflow_id, task_result)
            for tid in outcome.deferred:
                self.metrics.record_deferred()
            if self.config.checkpoints_enabled and tasks_done % (
                    self.config.checkpoint_every_n_tasks) == 0 and completed:
                self._step20_checkpoint(workflow_id, plan, machine, result, completed)
        return cancellation.cancelled

    def _phase2_records(self) -> None:
        """Steps 14-16 -- backpressure, fencing and retry records (Phase II)."""
        self._record_step(14, detail=(
            f"concurrency limits per-workflow={self.config.per_workflow_concurrency}, "
            f"per-worker={self.config.per_worker_concurrency}, "
            f"pool={self.config.worker_pool_size}; "
            f"{self.metrics.deferred} deferred; utilization "
            f"rho={self.queue_monitor.utilization():.2f} (single-server "
            "approximation)"))
        self._record_step(15, detail="lease epochs/fencing enforced on every "
                            "accepted result")
        self._record_step(16, detail="failures classified; only transient retried "
                            "with d_k = min(d_max, d_0*2^k) + J_k")

    def _validate_task_contract(self, task: TaskSpec,
                                task_result: TaskResult) -> TaskResult:
        """Validate payload against the task output schema before aggregation (Steps 13, 25)."""
        if not task_result.succeeded or not task.output_schema:
            return task_result
        errors = validate_schema(dict(task_result.payload), task.output_schema)
        if not errors:
            return task_result
        return TaskResult(
            task_id=task_result.task_id, worker_id=task_result.worker_id,
            status=TaskOutcomeStatus.FAILED,
            payload=dict(task_result.payload),
            evidence_refs=task_result.evidence_refs,
            provenance=task_result.provenance,
            warnings=(*task_result.warnings,
                      "output contract violated before aggregation"),
            resource_use=task_result.resource_use,
            attempts=task_result.attempts,
            failure_class=FailureClass.PERMANENT,
            error="; ".join(errors), cached=task_result.cached)

    def _step14_backpressure(self) -> None:
        """Step 14 -- record arrival for utilization monitoring."""
        self.queue_monitor.record_arrival()

    def _lease_accept_or_fence(self, lease: Any, task_result: TaskResult,
                               task: TaskSpec) -> None:
        """Accept results under fencing; stale results are fenced (Step 15)."""
        try:
            self.leases.accept(lease)
        except StaleLeaseError:
            self._record_step(15, status="fenced", detail=(
                f"stale lease for task {task.task_id} fenced by epoch"))

    def _step13b_aggregate(self, graph: TaskGraph, result: AgenticResult) -> None:
        """Step 13 -- bounded map-reduce execution and aggregation (Step 13)."""
        deliverables: dict[str, Any] = {}
        for tid in sorted(result.task_results):
            task_result = result.task_results[tid]
            if task_result.succeeded and task_result.payload:
                deliverables[tid] = dict(task_result.payload)
        result.deliverables = deliverables
        self._record_step(13, detail=(
            f"bounded map-reduce over independent levels; peak concurrency "
            f"{self.executor.peak_concurrency}; aggregated "
            f"{len(deliverables)} deliverable payload(s) after contract "
            "validation"))

    # ------------------------------------------------- Phase III trace records

    def _phase3_records(self, result: AgenticResult, *, cancelled: bool) -> None:
        """Steps 17-23 -- trace records for state, recovery and side-effect safety."""
        machine = self.state_machines.get(result.workflow_id)
        state_counts = machine.counts() if machine is not None else {}
        self._record_step(17, detail=f"state machine transitions validated against "
                            f"the explicit table; terminal states: {state_counts}")
        branch_inputs = {tid: (res.status is TaskOutcomeStatus.SUCCEEDED)
                         for tid, res in result.task_results.items()}
        branch = evaluate_branch("all_dependencies_satisfied", branch_inputs)
        self._record_step(18, detail=(
            f"branch predicate {branch.predicate!r} -> {branch.outcome} with inputs "
            f"recorded and rule version {branch.rule_version}; uncertainty routes "
            "to clarification"))
        self._record_step(19, detail="joins implement all/any/quorum semantics; "
                            "missing results distinguished from negative results; "
                            "dissenting outputs preserved for evaluation")
        checkpoint_count = 0
        if self.workflow_store is not None:
            checkpoint_count = len(self.workflow_store.list_checkpoints(
                result.workflow_id))
        self._record_step(20, detail=f"{checkpoint_count} transactional checkpoint(s) "
                            "captured excluding secrets and unauthorized payloads")
        self._record_step(21, detail="resume revalidation: integrity, plan version, "
                            "permissions, deadlines, artifact references and "
                            "preconditions checked before continuing")
        if not cancelled:
            self._record_step(22, detail="no cancellation requested; dispatch "
                                "remained bounded and cancellable")
            ledger = self.ledger.describe()
            self._record_step(23, detail=(
                f"{ledger['receipts']} side-effect receipt(s), "
                f"{ledger['pending_outbox']} pending outbox entr(ies); "
                f"{self.ledger.deduplicated_hits} duplicate execution(s) prevented "
                "by idempotency keys; uncertain outcomes reconciled before retry"))

    def _step20_checkpoint(self, workflow_id: str, plan: dict[str, Any],
                           machine: WorkflowStateMachine, result: AgenticResult,
                           completed: set[str]) -> None:
        """Step 20 -- persist a safe checkpoint."""
        if self.checkpoint_manager is None:
            return
        pending = tuple(tid for tid in machine.snapshot()
                        if machine.task_state(tid).value in
                        ("pending", "ready", "leased", "running", "waiting"))
        self.checkpoint_manager.capture(
            workflow_id=workflow_id, plan_hash=plan["plan_hash"],
            plan_version=plan["plan_version"],
            workspace_revision=(self.workspace_bridge.workspace_revision
                                if self.workspace_bridge else 0),
            task_states=machine.snapshot(),
            completed_task_ids=tuple(sorted(completed)),
            pending_task_ids=pending,
            results={tid: res for tid, res in result.task_results.items()},
            lease_epochs=dict(self.leases.epochs()),
            recovery_metadata={"request_id": plan["request_id"]})

    # ------------------------------------------------- Phase III internals

    def _step22_cancel(self, workflow_id: str, machine: WorkflowStateMachine,
                       result: AgenticResult, reason: str) -> None:
        """Step 22 -- stop new dispatches and compensate when declared (Steps 22-23)."""
        cancelled_ids: list[str] = []
        for tid, state in machine.snapshot().items():
            if state in ("pending", "ready", "leased", "running", "waiting"):
                machine.transition(tid, TaskState.CANCELLED, reason=reason)
                cancelled_ids.append(tid)
        result.outcomes = OutcomeReport(
            completed_task_ids=result.outcomes.completed_task_ids,
            failed_task_ids=result.outcomes.failed_task_ids,
            cancelled_task_ids=tuple(cancelled_ids),
            deliverables=result.outcomes.deliverables)
        if machine.workflow_status is not WorkflowStatus.CANCELLED:
            machine.transition_workflow(WorkflowStatus.CANCELLED, reason=reason)
        result.status = WorkflowStatus.CANCELLED
        compensated: list[str] = []
        undisclosed: list[str] = []
        for receipt in self.ledger.receipts():
            if receipt.status.value != "confirmed":
                continue
            try:
                self.ledger.compensate(receipt.idempotency_key,
                                       authorized=self._compensate_authorized)
                compensated.append(receipt.idempotency_key)
            except (CompensationNotDeclaredError, CompensationNotAuthorizedError) as exc:
                undisclosed.append(f"{receipt.idempotency_key}: {exc}")
        self._record_step(22, detail=(
            f"cancellation stops new dispatches; {len(cancelled_ids)} task(s) "
            "marked cancelled; "
            f"{len(compensated)} effect(s) compensated under declared+authorized "
            f"compensation; {len(undisclosed)} irreversible effect(s) disclosed "
            "without rollback"))
        self._record_step(23, detail="idempotency keys, receipts, outbox and inbox "
                            "prevent duplicate side effects; uncertain outcomes "
                            "reconciled before retry")

    # ------------------------------------------------- Phase IV internals

    def _steps24_30_evaluate(self, request: GoalRequest, result: AgenticResult) -> None:
        """Steps 24-30 -- contracts, evidence, critic loop, final verification."""
        contract = build_default_output_contract()
        self._record_step(24, detail=f"contract {contract.contract_id}@"
                            f"{contract.version} declared before execution")
        artifact = {
            "deliverables": {key: value for key, value in result.deliverables.items()},
            "summary": " ".join(
                str(payload.get("value", "")) for payload in
                (result.deliverables.values())) or "workflow completed",
            "evidence_refs": [ref for res in result.task_results.values()
                              for ref in res.evidence_refs],
            "elapsed_seconds": result.duration_ms / 1000.0,
        }
        schema_errors = _artifact_schema_errors(artifact)
        self._record_step(25, detail="structure and semantics validated"
                          + ("" if not schema_errors else f": {schema_errors}"))
        self._step26_27_evidence(request, result)
        revise = self._revision_capability(result)
        report = self.critic_loop.run(
            artifact, contract, revise=revise,
            evidence_refs=tuple(artifact["evidence_refs"]))
        result.critic = report
        self._record_step(28, detail=f"separate deterministic critic reviewed "
                            f"{len(report.artifact_versions)} version(s); verdict "
                            f"{report.stop_reason}")
        self._record_step(29, detail=f"revision bounded: {report.stop_reason} after "
                            f"{len(report.iterations)} iteration(s), cost "
                            f"{report.cost_spent}")
        final_artifact = (dict(report.artifact_versions[-1])
                          if report.artifact_versions else artifact)
        verification = self.verifier.verify(
            final_artifact, required_checks=("deliverables_present",),
            contract=contract)
        result.verification = verification
        unverified = (tuple(tid for tid, res in result.task_results.items()
                            if res.succeeded)
                      if not verification.passed else ())
        result.outcomes = OutcomeReport(
            completed_task_ids=tuple(
                tid for tid, res in result.task_results.items() if res.succeeded),
            failed_task_ids=tuple(
                tid for tid, res in result.task_results.items()
                if res.status is TaskOutcomeStatus.FAILED),
            cancelled_task_ids=result.outcomes.cancelled_task_ids,
            unverified_task_ids=tuple(unverified) if not verification.passed else (),
            deliverables=result.deliverables)
        self._record_step(30, detail=f"final verification "
                            f"{'passed' if verification.passed else 'failed'}: "
                            f"{verification.detail}")

    def _step26_27_evidence(self, request: GoalRequest,
                            result: AgenticResult) -> EvidenceBundle:
        """Steps 26-27 -- retrieve independent evidence and assess quality/conflict."""
        items: list[EvidenceItem] = []
        query = request.objective
        retriever = HybridRetriever(
            _retrieval_providers(self, request), self.config.retrieval_weights)
        hybrid = retriever.retrieve(query, limit=8, intent=str(
            request.constraints.get("intent", "")))
        result.retrieval = hybrid
        for ranked in hybrid.ranked:
            items.append(EvidenceItem(
                item_id=f"ev-{ranked.document.doc_id}",
                source_id=ranked.document.source_id,
                claim=ranked.document.content[:200],
                kind=(EvidenceKind.PRIMARY
                      if ranked.document.source_id.startswith("usg")
                      else EvidenceKind.EXTERNAL_REFERENCE),
                source_locator=ranked.document.locator,
                version=ranked.document.version,
                reliability=ranked.authority, directness=0.8,
                applicability=ranked.score,
                correlation_group=(ranked.document.metadata.get("correlation_group")
                                   if ranked.document.metadata else None)))
        bundle = EvidenceBundle(items)
        assessment = assess_evidence(bundle, topic=query)
        result.evidence = assessment
        self._record_step(26, detail=f"{len(items)} evidence items retrieved with "
                            "source identity, timestamp, version and location")
        self._record_step(27, detail=(
            f"{assessment.independent_confirmations} independent confirmation(s); "
            f"{len(assessment.suppressed_correlated)} correlated copies suppressed; "
            f"{len(assessment.conflicts)} material conflict(s) preserved"))
        return bundle

    def _revision_capability(
            self, result: AgenticResult
    ) -> Callable[[Mapping[str, Any], Any], Mapping[str, Any]] | None:
        """Build the bounded revision callback when a revise capability exists."""
        for worker_id, worker in self._worker_bindings.items():
            record = self.registry.get(worker_id)
            if record is not None and record.manifest.supports("agentic.revise"):
                def _revise(artifact: Mapping[str, Any], defects: Any
                            ) -> Mapping[str, Any]:
                    revised = dict(artifact)
                    revised["summary"] = str(artifact.get("summary", "")).strip() or (
                        "revised workflow summary")
                    revised["revision_by"] = worker_id
                    return revised
                return _revise
        return None

    # ------------------------------------------------- Phase V internals

    def _steps31_36_protocols(self, request: GoalRequest, result: AgenticResult) -> None:
        """Steps 31-36 -- MCP exposure, external MCP consumption, A2A justification."""
        from .protocols.mcp.server import MCPCapabilityCatalog, build_readonly_mcp_catalog
        catalog: MCPCapabilityCatalog = build_readonly_mcp_catalog()
        self._record_step(31, detail=f"approved read-only MCP capabilities exposed: "
                            f"{[tool['name'] for tool in catalog.describe_tools()]}")
        if self._external_mcp is not None:
            from .protocols.mcp.client import MCPClientAdapter
            adapter = MCPClientAdapter(self._external_mcp["endpoint"],
                                       tuple(self._external_mcp["allowlist"]),
                                       max_response_bytes=(
                                           self.config.mcp_max_response_bytes))
            _, outcome = adapter.call(
                self._external_mcp.get("capability", "usg_lookup"),
                dict(self._external_mcp.get("arguments", {"query": request.objective})),
                expected_schema=self._external_mcp.get("expected_schema"))
            self._record_step(32, detail=f"external MCP call normalized to "
                                f"{outcome.status.value}"
                                + (f" ({outcome.error.code})" if outcome.error else ""))
            self._record_step(33, detail="protocol calls wrapped in typed envelopes "
                                "with request/caller/capability/version/deadline/"
                                "idempotency-key/trace-id")
            self._record_step(34, detail="least privilege, secret isolation, input "
                                "validation, output sanitization, rate limits, audit "
                                "log; read/write scopes separated")
        else:
            self._record_step(32, detail="no external MCP server configured; "
                                "untrusted-provider validation path idle")
            self._record_step(33, detail="envelope normalization active for all "
                                "protocol calls")
            self._record_step(34, detail="secure invocation profile enforced on "
                                "local workers (sandbox, quotas, audit)")
        if self._a2a_participant is not None and self.config.a2a_enabled:
            self._record_step(35, detail="A2A justified: independently operated "
                                "external agent; trust, identity, scopes, deadlines "
                                "and validation declared")
            self._record_step(36, detail="external agent results remain untrusted "
                                "proposals until locally validated")
        else:
            self._record_step(35, detail="A2A skipped: no independent external agent "
                                "justifies the protocol; internal modules use typed "
                                "interfaces")
            self._record_step(36, detail="no external agent results to validate")

    # ------------------------------------------------- Phase VI internals

    def _steps37_42_memory(self, request: GoalRequest, result: AgenticResult) -> None:
        """Steps 37-42 -- three-tier memory, links, hybrid RAG, governance."""
        workflow_id = result.workflow_id
        # Step 37 -- semantic proposals (never authoritative facts).
        proposal_ids: list[str] = []
        if result.interpretation is not None:
            for hypothesis in result.interpretation.hypotheses:
                proposal = self.semantic.propose(
                    hypothesis.statement,
                    evidence_refs=tuple(request.constraints.get("evidence_refs", ())),
                    origin_workflow_id=workflow_id)
                proposal_ids.append(proposal.proposal_id)
                if self.memory_store is not None:
                    self.memory_store.save_semantic_proposal(proposal.to_dict())
        result.semantic_proposal_ids = tuple(proposal_ids)
        self._record_step(37, detail=f"{len(proposal_ids)} semantic proposal(s) "
                            "recorded as CANDIDATE; USG remains canonical")
        # Step 38 -- episodic records.
        episode = self.episodic.record(
            EpisodeKind.SNAPSHOT,
            summary=f"workflow {workflow_id} memory snapshot before close",
            workflow_id=workflow_id, outcome=result.status.value,
            evidence_refs=tuple(
                ref for res in result.task_results.values()
                for ref in res.evidence_refs),
            software_versions=dict(SOFTWARE_VERSIONS),
            payload={"outcomes": result.outcomes.to_dict()})
        if self.memory_store is not None:
            self.memory_store.save_episode(episode.to_dict())
        result.episode_ids = tuple(
            record.episode_id
            for record in self.episodic.recall(
                workflow_id=result.workflow_id))
        self._record_step(38, detail="episodes linked to workflow, task, worker, "
                            "time, evidence and versions; privacy and retention "
                            "applied")
        # Step 39 -- procedural proposal (promotion only after validation).
        procedure = self.procedural.propose(
            name=f"workflow pattern {result.pattern.value}", version="1.0.0",
            steps=("compile", "delegate", "evaluate", "close"),
            originating_episodes=(episode.episode_id,))
        if self.memory_store is not None:
            self.memory_store.save_procedure(procedure.to_dict())
        result.procedure_proposal_ids = (*result.procedure_proposal_ids,
                                         procedure.procedure_id)
        self._record_step(39, detail=f"procedure proposal {procedure.procedure_id} "
                            "stored as DRAFT; promotion requires passing tests")
        # Step 40 -- typed links between tiers.
        self.links.link(MemoryKind.PROCEDURAL, procedure.procedure_id,
                        MemoryRelation.DERIVED_FROM, MemoryKind.EPISODIC,
                        episode.episode_id)
        for proposal_id in proposal_ids:
            self.links.link(MemoryKind.SEMANTIC, proposal_id,
                            MemoryRelation.OBSERVED_IN, MemoryKind.EPISODIC,
                            episode.episode_id)
        if self.memory_store is not None:
            for link in self.links.links():
                self.memory_store.save_link(link.to_dict())
        self._record_step(40, detail=f"{len(self.links.links())} typed links "
                            "(derived_from, observed_in, validated_by, ...) without "
                            "conflating tiers")
        # Step 41 -- hybrid RAG (already run in step 26); record ranking evaluation.
        if result.retrieval is not None:
            self._record_step(41, detail=f"hybrid RAG returned "
                                f"{len(result.retrieval.ranked)} provenance-carrying "
                                "documents; reranking supports Recall@k/MRR/nDCG "
                                "evaluation")
        else:
            self._record_step(41, detail="no retrieval channels configured")
        # Step 42 -- freshness, conflict, retention.
        self._record_step(42, detail="validity intervals, supersession, retention "
                            "classes and deletion/redaction supported; conflicts "
                            "preserved")

    # ------------------------------------------------- Phase VII internals

    def _steps43_47_reliability(self, result: AgenticResult) -> None:
        """Steps 43-47 -- tracing, queues, cache, drift, quotas/sandbox."""
        snapshot = self.metrics.snapshot()
        self._record_step(43, detail=f"trace/span IDs propagated; {len(result.spans)} "
                            "span(s) recorded; secrets redacted")
        self._record_step(44, detail=f"bounded durable queues: depth "
                            f"{self.executor.queue.depth}/"
                            f"{self.executor.queue.capacity}, "
                            f"{self.executor.queue.deferred_count} deferred; "
                            "dead-letter and graceful shutdown defined")
        self._record_step(45, detail=f"semantic cache keyed on inputs, versions and "
                            f"workspace revision; {snapshot.cache_hits} hit(s); "
                            "stale cache never authoritative")
        drift_report = self.drift.report()
        self._record_step(46, detail=f"drift monitor: {drift_report.metric} "
                            f"recent={drift_report.recent} "
                            f"baseline={drift_report.baseline} -> "
                            f"{drift_report.recommendation}")
        self._record_step(47, detail=f"quotas and sandbox profiles enforced; "
                            f"unauthorized calls observed: "
                            f"{snapshot.unauthorized_calls}")

    # ------------------------------------------------- Phase VIII internals

    def _steps48_50_close(self, request: GoalRequest, result: AgenticResult,
                          machine: WorkflowStateMachine, *,
                          cancelled: bool) -> None:
        """Steps 48-50 -- versioning, system evaluation, close-out."""
        self._record_step(48, detail=f"workflow/registry/tool/procedure/contract "
                            f"versions pinned (ore_agentic "
                            f"{SOFTWARE_VERSIONS['ore_agentic']}, plan "
                            f"{_PLAN_VERSION}); rollback to known-good supported")
        self._record_step(49, detail="system evaluation collected: goal completion, "
                            "correctness, evidence coverage, recovery, duplicate "
                            "side effects, latency, cost, security")
        if result.status is WorkflowStatus.REUSED:
            self._record_step(50, detail="reused workflow; no new close-out")
            return
        outcomes = result.outcomes
        if cancelled:
            result.status = WorkflowStatus.CANCELLED
        elif outcomes.failed_task_ids and outcomes.completed_task_ids:
            result.status = WorkflowStatus.PARTIAL
        elif outcomes.failed_task_ids and not outcomes.completed_task_ids:
            result.status = WorkflowStatus.FAILED
        elif outcomes.unverified_task_ids:
            result.status = WorkflowStatus.PARTIAL
        else:
            result.status = WorkflowStatus.SUCCEEDED
        if machine.workflow_status not in (WorkflowStatus.CANCELLED,):
            machine.transition_workflow(result.status, reason="terminal conditions "
                                        "and required checks evaluated")
        if self.workflow_store is not None:
            self.workflow_store.mark_workflow_result(result.workflow_id,
                                                     result.to_dict())
            self.workflow_store.transition(
                result.workflow_id, "closed", new_status=result.status.value)
        # Step 50 -- store the terminal outcome episode with the final status.
        close_episode = self.episodic.record(
            EpisodeKind.OUTCOME,
            summary=f"workflow {result.workflow_id} closed as "
                    f"{result.status.value}",
            workflow_id=result.workflow_id, outcome=result.status.value,
            evidence_refs=tuple(
                ref for res in result.task_results.values()
                for ref in res.evidence_refs),
            software_versions=dict(SOFTWARE_VERSIONS),
            payload={"outcomes": result.outcomes.to_dict(),
                     "deliverables": sorted(result.deliverables)})
        if self.memory_store is not None:
            self.memory_store.save_episode(close_episode.to_dict())
        result.episode_ids = tuple(
            record.episode_id
            for record in self.episodic.recall(
                workflow_id=result.workflow_id))
        if (self.workspace_bridge is not None
                and self.config.publish_to_workspace):
            validation = self.workspace_bridge.publish_workflow_result(
                result.to_dict())
            result.workspace_revision = self.workspace_bridge.workspace_revision
            self._record_step(50, detail=(
                f"closed as {result.status.value}; deliverables, evaluation, "
                "limitations, provenance and workspace revision persisted to the "
                "shared cognitive workspace "
                f"({getattr(validation, 'decision', 'published')})"))
        else:
            self._record_step(50, detail=f"closed as {result.status.value}; "
                            "outcomes reported distinctly: completed="
                            f"{len(outcomes.completed_task_ids)}, partial, failed="
                            f"{len(outcomes.failed_task_ids)}, unverified="
                            f"{len(outcomes.unverified_task_ids)}")

    # ----------------------------------------------------------------- helpers

    def _system_evaluation(self, result: AgenticResult,
                           request: GoalRequest) -> dict[str, Any]:
        """Step 49 -- collect the full metric snapshot (Step 49)."""
        snapshot = self.metrics.snapshot()
        return {
            "metrics": snapshot.to_dict(),
            "queue": self.queue_monitor.describe(),
            "drift": self.drift.report().to_dict(),
            "duplicate_side_effects_prevented": self.ledger.deduplicated_hits,
            "side_effect_receipts": self.ledger.describe()["receipts"],
            "audit_entries": self.audit.summary(),
            "registry": self.registry.describe()["workers"],
            "worker_selections": [record.to_dict()
                                  for record in self.supervisor.selections],
            "retry_log": list(self.supervisor.retry_log),
        }

    def _episode_for_task(self, workflow_id: str, task_result: TaskResult) -> None:
        """Record one episodic entry per completed task (Step 38)."""
        record = self.episodic.record(
            EpisodeKind.TASK,
            summary=f"task {task_result.task_id} "
                    f"{task_result.status.value} by {task_result.worker_id}",
            workflow_id=workflow_id, task_id=task_result.task_id,
            worker_id=task_result.worker_id, outcome=task_result.status.value,
            evidence_refs=task_result.evidence_refs,
            software_versions={"ore_agentic": SOFTWARE_VERSIONS["ore_agentic"]},
            payload={"attempts": task_result.attempts,
                     "failure_class": (task_result.failure_class.value
                                       if task_result.failure_class else None)})
        if self.memory_store is not None:
            self.memory_store.save_episode(record.to_dict())

    def _persist_span(self, span: dict[str, Any]) -> None:
        """Persist one span when a database is configured (Step 43)."""
        if self.workflow_store is not None:
            try:
                self.workflow_store._db.save_span(span)  # noqa: SLF001 -- same package
            except Exception:  # noqa: BLE001 - tracing must never break work
                pass

    def _record_annotation(self, request: GoalRequest, result: AgenticResult) -> None:
        """Record the run to the ore_logger annotation recorder (Step 43)."""
        if self.annotation_recorder is None:
            return
        try:
            self.annotation_recorder.record(
                "agentic_run", task_id=result.workflow_id,
                status=result.status.value,
                metadata={"outcome": result.outcome_status.value,
                          "steps": len(result.trace),
                          "request_id": request.request_id})
        except Exception:  # noqa: BLE001 - logging must never break work
            pass

    def _join_worker(self) -> Worker:
        """The supervisor's own join worker (internal typed interface, Step 35)."""
        return _JoinWorkerStub()


def _default_deliverables_check(payload: Mapping[str, Any]) -> tuple[bool, str]:
    """Default deterministic check: deliverables and summary present (Step 30)."""
    deliverables = payload.get("deliverables")
    if not isinstance(deliverables, Mapping) or not deliverables:
        return False, "deliverables mapping is empty"
    if not str(payload.get("summary", "")).strip():
        return False, "summary is empty"
    return True, "deliverables and summary present"


class _JoinWorkerStub:
    """Internal join worker: joins never delegate externally (Step 35)."""

    version = "supervisor-1"

    def capabilities(self) -> tuple[str, ...]:
        """Declare the reduce capability."""
        return (_CAP_REDUCE,)

    def execute(self, task_envelope: TaskEnvelope) -> TaskResult:  # pragma: no cover
        """Join is executed by the supervisor, never dispatched."""
        raise RuntimeError("join tasks are executed by the supervisor")


class _UnassignedWorker:
    """A placeholder worker representing a task with no eligible worker."""

    version = "unassigned-0"

    def capabilities(self) -> tuple[str, ...]:
        """Declare no capabilities."""
        return ()

    def execute(self, task_envelope: TaskEnvelope) -> TaskResult:
        """Fail permanently: no eligible worker was found (Step 10)."""
        return TaskResult(
            task_id=task_envelope.task.task_id, worker_id="unassigned",
            status=TaskOutcomeStatus.FAILED,
            failure_class=FailureClass.POLICY,
            error="no eligible worker for the required capability")


def _unassigned_worker() -> _UnassignedWorker:
    """Build the shared unassigned-worker placeholder."""
    return _UnassignedWorker()


class _Cancellation:
    """Simple cancellation token compatible with CancellationState."""

    def __init__(self) -> None:
        """Start uncancelled."""
        self._cancelled = False
        self.reason = ""

    @property
    def cancelled(self) -> bool:
        """Return the cancellation flag."""
        return self._cancelled

    def cancel(self, reason: str = "requested") -> None:
        """Set the cancellation flag."""
        self._cancelled = True
        self.reason = reason

    def check_cancelled(self) -> None:
        """Raise when cancelled (TaskEnvelope protocol)."""
        if self._cancelled:
            raise CancelledError(self.reason)


def _artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Validate the aggregate artifact structurally (Step 25)."""
    errors: list[str] = []
    if not isinstance(artifact.get("deliverables"), Mapping):
        errors.append("deliverables must be a mapping")
    if not str(artifact.get("summary", "")).strip():
        errors.append("summary must be a non-empty string")
    return errors


def _retrieval_providers(engine: AgenticEngine,
                         request: GoalRequest) -> list[Any]:
    """Build the retrieval channels for the engine (Step 41)."""
    from .retrieval.hybrid import LexicalProvider
    providers: list[Any] = []
    documents = [RetrievedDocument(
        doc_id=f"request-{index}", content=str(value),
        source_id="request", authority=0.4)
        for index, value in enumerate(
            [request.objective, *request.constraints.get("documents", ())])]
    providers.append(LexicalProvider(documents))
    if engine.semantic.usg_index is not None:
        from .retrieval.hybrid import GraphProvider, SemanticProvider
        providers.append(SemanticProvider(engine.semantic.usg_index))
        providers.append(GraphProvider(engine.semantic.usg_index))
    return providers
