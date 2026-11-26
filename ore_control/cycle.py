"""The 40-step cognitive control cycle (Point 02 Algorithm, Phases A-H).

``CognitiveControlCoordinator.process_trigger`` executes one Soar-inspired
decision cycle per iteration: trigger intake, authoritative snapshot, goal
resolution, context build/resume, state summary, operator generation and
filtering, activation-based retrieval, expected-utility selection, bounded
subproblems, explicit/learned routing with meta-cognitive monitoring,
resource-reserved dispatch with checkpoints, contract validation, atomic
workspace integration, goal-progress recomputation, stagnation detection,
and the continue/revise/wait/stop decision. After a verified success, the
conservative procedure-learning phase runs (Steps 37-40).

The layer is executive-only: execution and state integration are delegated to
the existing workspace, coordinators, and the Point 01 blackboard layer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ore_workspace import (
    FindingPublisher,
    GoalEvaluator,
    SharedWorkspace,
    SubscriptionRegistry,
    WorkspaceManager,
)
from ore_workspace.models import (
    GoalOutcome,
    Provenance,
    WORKSPACE_FIELDS,
)

from ore_blackboard.usg_integration import BlackboardUSGSnapshot

from .activation import (
    base_level_activation,
    choose_retrieval_depth,
    cue_association,
    rank_candidates,
    retrieval_score,
)
from .config import ControlPolicy, default_policy
from .contracts import (
    COGNITIVE_CONTROL_PRODUCER,
    COGNITIVE_CONTROL_SCHEMA_VERSION,
    CONTROL_PACKAGE_VERSION,
    CandidateRecord,
    ContextBudget,
    ContextStatus,
    CognitiveContext,
    ControlRunResult,
    DecisionTrace,
    OperatorSpec,
    RetrievalCandidate,
    RiskClass,
    SelectionRecord,
    TriggerEnvelope,
    TriggerKind,
    new_control_id,
    sha256_json,
)
from .metacognition import monitor as metacognitive_monitor
from .operators import (
    ControlServices,
    OperatorAction,
    OperatorFinding,
    OperatorOutcome,
    OperatorRegistry,
    _hypothesis_records,
    candidate_actions_for_state,
    deduplicate_actions,
    evaluate_preconditions,
    register_default_operators,
)
from .procedures import (
    ProcedureRecord,
    ProcedureRegistry,
    ProcedureSequenceStep,
    ReplayResult,
    VerifiedRunRecord,
    evaluate_generalization,
    evaluate_promotion_gates,
    extract_candidate_procedure,
    procedure_operator_spec,
    wrap_procedure,
)
from .resources import ControlResourceCoordinator, ReservationDecision
from .routing import (
    LearnedComponentCheck,
    LearnedComponentRegistry,
    classify_processing_path,
    default_learned_components,
)
from .subproblems import (
    ChildBoundaryGuard,
    ChildOutcome,
    create_child_contexts,
    decomposition_justified,
    reconcile_child_outcomes,
)
from .trace import DecisionTraceBuilder, persist_trace
from .utility import (
    apply_hard_constraints,
    compute_utilities,
    estimate_outcomes,
    select_action,
)

_TERMINAL_GOAL_OUTCOMES = {
    GoalOutcome.COMPLETED, GoalOutcome.COMPLETED_WITH_UNCERTAINTY,
    GoalOutcome.FAILED, GoalOutcome.BLOCKED,
}
_SUCCESS_OUTCOMES = {
    GoalOutcome.COMPLETED, GoalOutcome.COMPLETED_WITH_UNCERTAINTY,
}


@dataclass
class _CycleCounters:
    """Shared, bounded counters across the cycles of one run (§8)."""

    cycles: int = 0
    operator_applications: int = 0
    retrievals: int = 0
    retrieval_cost: float = 0.0
    retries: int = 0
    children: int = 0
    learned_proposals: int = 0
    escalations: list[str] = field(default_factory=list)


class CognitiveControlCoordinator:
    """Run the 40-step cognitive control algorithm over registered operators."""

    def __init__(
        self,
        *,
        operators: OperatorRegistry | None = None,
        policy: ControlPolicy | None = None,
        blackboard: Any | None = None,
        usg_bundle: Mapping[str, Any] | None = None,
        usg_expected_digest: str | None = None,
        learned_components: LearnedComponentRegistry | None = None,
        procedure_registry: ProcedureRegistry | None = None,
        workspace_manager: WorkspaceManager | None = None,
        clock: Callable[[], float] | None = None,
        repository_root: str | Path | None = None,
        logger_database_path: str | Path | None = None,
    ) -> None:
        """Bind registries, policy, knowledge resources, and the run clock."""
        self.policy = policy if policy is not None else default_policy()
        self.operators = operators if operators is not None else OperatorRegistry()
        if operators is None:
            register_default_operators(self.operators)
        self.blackboard = blackboard
        self.usg: BlackboardUSGSnapshot | None = None
        if usg_bundle is not None:
            self.usg = BlackboardUSGSnapshot(
                usg_bundle, expected_digest=usg_expected_digest,
            )
        self.learned_components = (
            learned_components if learned_components is not None
            else default_learned_components()
        )
        self.procedure_registry = (
            procedure_registry if procedure_registry is not None
            else ProcedureRegistry()
        )
        self.repository_root = (
            Path(repository_root).resolve() if repository_root
            else Path(__file__).resolve().parents[1]
        )
        self.logger_database_path = (
            Path(logger_database_path).resolve() if logger_database_path else None
        )
        self._injected_workspace_manager = workspace_manager
        self._clock = clock
        self._goal_evaluator = GoalEvaluator(
            threshold=self.policy.goal_threshold,
            critical_requirements={
                "understand_sentence": (
                    "lexical_hypotheses", "syntactic_hypotheses", "semantic_hypotheses",
                ),
            },
        )
        self.resource_coordinator = ControlResourceCoordinator(self.policy)
        self._runs: dict[str, ControlRunResult] = {}
        self._workspaces: dict[str, SharedWorkspace] = {}
        self._contexts: dict[str, CognitiveContext] = {}
        self._verified_runs: list[VerifiedRunRecord] = []
        self._usage_history: dict[str, list[float]] = {}
        self._fingerprint_history: dict[str, list[str]] = {}

    # ------------------------------------------------------------- services

    def _now(self, *, run_epoch: float | None = None, cycle: int = 0) -> float:
        """Return the activation clock reading (deterministic when injected)."""
        if self._clock is not None:
            return float(self._clock())
        if run_epoch is not None:
            return run_epoch + cycle
        return time.time()

    def _record_usage(self, terms: Sequence[str], at: float) -> None:
        """Append episodic usage timestamps for encountered terms (Step 13)."""
        for term in terms:
            self._usage_history.setdefault(term, []).append(at)

    def _services(self, task_text: str, run_id: str) -> ControlServices:
        """Build the read-only services bundle for one cycle (Step 31)."""
        return ControlServices(
            task_text=task_text,
            usg_index=self.usg.index if self.usg is not None else None,
            usg_reference=(
                self.usg.reference.to_dict() if self.usg is not None else None
            ),
            learned_components=self.learned_components,
            blackboard=self.blackboard,
            run_id=run_id,
            usage_history={
                key: tuple(value) for key, value in self._usage_history.items()
            },
        )

    def _state_summary(
        self, snapshot_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Derive the compact cognitive state summary (Step 5)."""
        open_questions = [
            item for item in snapshot_state.get("unresolved_questions", [])
            if isinstance(item, Mapping) and not item.get("resolved", False)
        ]
        hypothesis_records = _hypothesis_records(snapshot_state)
        hypotheses = len(hypothesis_records)
        competing = 0
        for record in hypothesis_records:
            alternates = record.get("competing_hypotheses", [])
            if isinstance(alternates, Sequence) and not isinstance(alternates, (str, bytes)):
                competing = max(competing, len(alternates))
        unresolved_conflicts = (
            sum(
                1 for item in snapshot_state.get("unresolved_questions", [])
                if isinstance(item, Mapping)
                and item.get("finding_type") == "conflict_preservation"
                and not item.get("resolved", False)
            )
        )
        satisfied = tuple(
            field_name for field_name in (
                "lexical_hypotheses", "syntactic_hypotheses", "semantic_hypotheses",
                "retrieved_memories", "unresolved_questions", "discourse_context",
            ) if snapshot_state.get(field_name)
        )
        evidence_ids = sorted({
            str(item.get("record_reference"))
            for field_name in ("retrieved_memories",)
            for item in snapshot_state.get(field_name, [])
            if isinstance(item, Mapping) and item.get("record_reference")
        })
        return {
            "satisfied_conditions": satisfied,
            "unsatisfied_conditions": tuple(
                name for name in WORKSPACE_FIELDS if name not in satisfied
            ),
            "open_questions": open_questions,
            "hypotheses": hypotheses,
            "competing_hypotheses": competing,
            "unresolved_conflicts": unresolved_conflicts,
            "evidence_ids": evidence_ids,
            "latest_evidence_revision": len(
                snapshot_state.get("retrieved_memories", []) or []
            ),
            "capabilities": tuple(
                sorted({
                    capability for spec in self.operators.specs()
                    for capability in spec.required_capabilities
                })
            ),
        }

    def _cycle_fingerprint(
        self, goal_id: str, summary: Mapping[str, Any], revision: int,
    ) -> str:
        """Compute the stable cycle fingerprint (Step 5, §5.7)."""
        return sha256_json({
            "goal_id": goal_id,
            "open_questions": sorted(
                str(item.get("question_id")) for item in summary["open_questions"]
            ),
            "hypotheses": summary["hypotheses"],
            "competing": summary["competing_hypotheses"],
            "unresolved_conflicts": summary["unresolved_conflicts"],
            "revision": revision,
        })

    # ------------------------------------------------------------ publishing

    def _publish_finding(
        self, workspace: SharedWorkspace, finding: OperatorFinding,
    ) -> Mapping[str, Any]:
        """Integrate one validated operator finding atomically (Step 33)."""
        publisher = FindingPublisher(workspace, COGNITIVE_CONTROL_PRODUCER)
        envelope = publisher.create_finding(
            finding.field,
            finding.payload,
            key=finding.key,
            finding_type=finding.finding_type,
            provenance=Provenance(
                source=finding.source,
                rules_applied=tuple(finding.rules_applied),
                evidence=tuple(finding.evidence),
                notes=finding.notes or "",
            ),
            dependencies=finding.dependencies,
            confidence=finding.confidence,
        )
        decision = workspace.publish(envelope)
        return {
            "finding_id": decision.finding_id,
            "field": finding.field,
            "decision": decision.decision.value
            if hasattr(decision.decision, "value") else str(decision.decision),
            "message": decision.message,
            "workspace_revision": decision.workspace_revision,
            "conflict_id": decision.conflict_id,
        }

    def _validate_outcome(
        self, outcome: OperatorOutcome, spec: OperatorSpec | None,
    ) -> tuple[bool, Mapping[str, Any]]:
        """Validate an operator outcome against its contract (Step 32)."""
        writes_workspace = spec.writes_workspace if spec is not None else True
        checks: dict[str, Any] = {
            "status_allowed": outcome.status in {"completed", "deferred", "waiting"}
            or (outcome.status == "failed" and outcome.failure_classification),
            "findings_typed": all(
                isinstance(item, OperatorFinding) for item in outcome.findings
            ),
            "fields_known": all(
                item.field in WORKSPACE_FIELDS for item in outcome.findings
            ),
            "provenance_present": all(
                bool(item.source) for item in outcome.findings
            ),
            "diagnostics_present": bool(outcome.diagnostics) or bool(outcome.payload),
            "postcondition_findings": (
                (not writes_workspace) or bool(outcome.findings)
                or outcome.status in {"deferred", "waiting"}
            ),
        }
        passed = all(bool(value) for value in checks.values())
        return passed, checks

    # ----------------------------------------------------------- retrieval

    def _retrieval_phase(
        self,
        trace: DecisionTraceBuilder,
        *,
        task_text: str,
        summary: Mapping[str, Any],
        context: CognitiveContext,
        run_epoch: float,
        cycle: int,
        counters: _CycleCounters,
    ) -> tuple[tuple[RetrievalCandidate, ...], Mapping[str, Any]]:
        """Execute the retrieval phase for one cycle (Steps 10-17)."""
        terms = _task_terms(task_text)
        open_targets = [
            str(item.get("target_term")) for item in summary["open_questions"]
            if item.get("target_term")
        ]
        cue_weights: dict[str, float] = {}
        for term in terms:
            cue_weights[term] = 0.3
        for term in open_targets:
            if term in cue_weights:
                cue_weights[term] = 1.0
            else:
                cue_weights[term] = 0.9
        ambiguous = [
            term for term in terms
            if self.usg is not None and self.usg.index.lookup(term).exact_match
        ]
        for term in ambiguous:
            cue_weights[term] = max(cue_weights.get(term, 0.0), 0.8)
        trace.record_step(10, {
            "cue_terms": sorted(cue_weights),
            "cue_count": len(cue_weights),
            "traceable_to": "goal/open_questions/hypotheses/operator_requirements",
        })

        cache_records = _cache_candidates(summary)
        domains = ["local_cache"]
        if self.usg is not None:
            domains.append("structured_index")
        trace.record_step(11, {"selected_domains": domains})

        candidates: list[RetrievalCandidate] = list(cache_records)
        now = self._now(run_epoch=run_epoch, cycle=cycle)

        depth = choose_retrieval_depth(
            remaining_budget=max(
                0.0, self.policy.budgets.retrieval_budget - counters.retrieval_cost,
            ),
            sufficient_evidence_met=bool(cache_records),
            depth_policy=self.policy.retrieval_depth,
        )
        trace.record_step(17, depth.to_dict())

        if "structured_index" in domains and depth.level in {
            "structured_index", "full_resource",
        }:
            for term in sorted(cue_weights):
                result = self.usg.index.lookup(term)
                counters.retrievals += 1
                counters.retrieval_cost += depth.expected_cost
                for item in result.items:
                    candidates.append(RetrievalCandidate(
                        record_reference=item.record_id,
                        source_type=f"usg:{item.kind}",
                        cue_match_features={term: 1.0},
                        usage_timestamps=tuple(
                            self._usage_history.get(term, ())
                        ) or None,
                        semantic_relevance=1.0 if term in open_targets else 0.5,
                        confidence=None,
                        provenance={
                            "bundle_id": result.bundle_id,
                            "bundle_digest": result.bundle_digest,
                            "source_reference": item.source_reference,
                            "retrieved_by": "exact_normalized_lookup",
                        },
                        retrieval_cost=depth.expected_cost / max(1, len(result.items)),
                        version_reference=(
                            f"usg:{result.bundle_id}:{result.bundle_digest[:12]}"
                        ),
                        inclusion_reasons=(f"exact_match_for_cue:{term}",),
                    ))
            trace.record_step(12, {
                "candidate_count": len(candidates),
                "cache_candidates": len(cache_records),
                "usg_candidates": len(candidates) - len(cache_records),
                "checks": ("access_permissions", "version_filters", "provenance"),
            })

        for index, candidate in enumerate(candidates):
            activation = base_level_activation(
                candidate.usage_timestamps or (),
                now=now,
                decay=self.policy.activation_decay,
                epsilon=self.policy.activation_epsilon,
            )
            associations = {
                cue: strength
                for cue, strength in candidate.cue_match_features.items()
            }
            association, contributions = cue_association(cue_weights, associations)
            candidates[index] = RetrievalCandidate(
                record_reference=candidate.record_reference,
                source_type=candidate.source_type,
                cue_match_features=candidate.cue_match_features,
                usage_timestamps=candidate.usage_timestamps,
                semantic_relevance=candidate.semantic_relevance,
                confidence=candidate.confidence,
                provenance=candidate.provenance,
                retrieval_cost=candidate.retrieval_cost,
                version_reference=candidate.version_reference,
                inclusion_reasons=candidate.inclusion_reasons,
                exclusion_reasons=candidate.exclusion_reasons,
                base_level_activation=activation,
                activation_available=activation is not None,
                cue_association=association if contributions else None,
                cue_contributions=contributions,
            )
        trace.record_step(13, {
            "formula": "B_i = ln(eps + sum t^-d)",
            "decay": self.policy.activation_decay,
            "missing_history_count": sum(
                1 for item in candidates if not item.activation_available
            ),
        })
        trace.record_step(14, {
            "formula": "S_i = sum_k W_k A_ik",
            "cue_count": len(cue_weights),
        })

        scored = [
            retrieval_score(
                item,
                coefficients=self.policy.retrieval,
                task_relevance=min(
                    1.0, item.semantic_relevance,
                ),
                provenance_support=(
                    1.0 if str(item.source_type).startswith("usg:") else 0.6
                ),
            )
            for item in candidates
        ]
        trace.record_step(15, {
            "formula": "R_i = alpha B_i + beta S_i + gamma Q_i + delta P_i - lambda C_i",
            "coefficients": {
                "alpha": self.policy.retrieval.alpha,
                "beta": self.policy.retrieval.beta,
                "gamma": self.policy.retrieval.gamma,
                "delta": self.policy.retrieval.delta,
                "lambda": self.policy.retrieval.lam,
            },
            "missing_feature_policy": "drop_term_and_flag",
        })
        shortlist = rank_candidates(scored, policy=self.policy.ambiguity)
        trace.record_step(16, {
            "shortlist": [item.record_reference for item in shortlist.shortlist],
            "winner": shortlist.winner.record_reference if shortlist.winner else None,
            "ambiguities": [dict(item) for item in shortlist.ambiguity_records],
        })
        trace.retrieval_candidates = [item.to_dict() for item in shortlist.ranked]
        trace.retrieval_shortlist_ids = [
            item.record_reference for item in shortlist.shortlist
        ]
        trace.retrieval_ambiguities = [
            dict(item) for item in shortlist.ambiguity_records
        ]
        return shortlist.ranked, {
            "shortlist": shortlist,
            "domains": domains,
            "depth": depth,
        }

    # ------------------------------------------------------------- children

    def _run_child(
        self,
        child: CognitiveContext,
        contract,
        guard: ChildBoundaryGuard,
        services: ControlServices,
        parent_state: Mapping[str, Any],
    ) -> ChildOutcome:
        """Run one bounded child context to its success predicate (Steps 22-25)."""
        child_state: dict[str, Any] = {
            key: (
                list(value) if isinstance(value, list)
                else dict(value) if isinstance(value, dict) else value
            )
            for key, value in parent_state.items()
            if isinstance(value, (list, dict))
        }
        for hypothesis_field in ("lexical_hypotheses", "semantic_hypotheses"):
            child_state[hypothesis_field] = [
                dict(record) for record in _hypothesis_records(parent_state)
            ]
        findings: list[OperatorFinding] = []
        applied = 0
        executed_operators: set[str] = set()
        preference = ("control.retrieve_knowledge", "control.verify_hypotheses",
                      "control.request_missing_evidence")
        hypothesis_id = next(iter(child.active_hypotheses), None)
        for _ in range(contract.budget.operations):
            chosen: tuple[OperatorSpec, OperatorAction] | None = None
            for operator_id in preference:
                if operator_id in executed_operators:
                    continue
                if not guard.check_operator(operator_id):
                    continue
                if operator_id not in contract.allowed_operator_ids:
                    continue
                spec = self.operators.get_spec(operator_id)
                if spec is None:
                    continue
                ok, _failures = evaluate_preconditions(spec, child_state, services)
                if ok:
                    actions = candidate_actions_for_state(
                        (spec,), state_summary=child_state, services=services,
                        context=child,
                    )
                    if actions:
                        chosen = (spec, actions[0])
                        break
            if chosen is None:
                break
            spec, action = chosen
            executed_operators.add(action.operator_id)
            handler = self.operators.handler(action.operator_id)
            if handler is None:
                break
            outcome: OperatorOutcome = handler(child, child_state, action, services)
            applied += 1
            allowed_findings = [
                finding for finding in outcome.findings
                if guard.check_finding(finding)
            ]
            findings.extend(allowed_findings)
            for finding in allowed_findings:
                bucket = child_state.setdefault(finding.field, [])
                if isinstance(bucket, list):
                    bucket.append(dict(finding.payload))
            resolved = any(
                isinstance(item, Mapping)
                and item.get("resolved", False)
                and (not hypothesis_id or item.get("hypothesis_id") == hypothesis_id)
                for item in child_state.get("unresolved_questions", [])
            )
            if resolved:
                return ChildOutcome(
                    context_id=child.context_id,
                    goal_text=contract.goal_text,
                    status="SUCCEEDED",
                    contract_satisfied=True,
                    findings=tuple(findings),
                    decision="",
                    elapsed_ms=0.0,
                )
            if outcome.status == "waiting":
                break
        return ChildOutcome(
            context_id=child.context_id,
            goal_text=contract.goal_text,
            status="INCOMPLETE",
            contract_satisfied=False,
            findings=tuple(findings),
            unresolved_issues=(
                "child budget exhausted before success predicate",
            ),
            decision="",
            elapsed_ms=0.0,
        )

    # ------------------------------------------------------------------ run

    def process_trigger(self, trigger: TriggerEnvelope) -> ControlRunResult:
        """Execute the 40-step cognitive control algorithm for one trigger."""
        if not isinstance(trigger, TriggerEnvelope):
            raise TypeError("trigger must be a TriggerEnvelope")
        started = time.perf_counter()
        run_epoch = time.time()

        # Step 1 - receive and validate the trigger (idempotent intake).
        fingerprint = trigger.fingerprint()
        cached = self._runs.get(fingerprint)
        if cached is not None:
            return cached
        task_text = str(trigger.payload.get("text", "")).strip()
        if not task_text:
            raise ValueError(
                "trigger payload must carry a non-empty 'text' for the reference "
                "cognitive task"
            )
        run_id = new_control_id("control_run")
        counters = _CycleCounters()
        counters.escalations = []
        procedure_events: list[Mapping[str, Any]] = []

        # Step 2 - read the authoritative workspace snapshot.
        manager = self._injected_workspace_manager or WorkspaceManager(
            repository_root=self.repository_root,
            logger_database_path=self.logger_database_path,
            enable_logging=False,
        )
        owns_manager = self._injected_workspace_manager is None
        resume_run = str(trigger.payload.get("resume_run_id", "") or "")
        workspace = None
        if trigger.kind in {
            TriggerKind.SCHEDULED_CONTINUATION, TriggerKind.SUBTASK_COMPLETED,
        } and resume_run:
            prior = self._runs.get(
                next(
                    (key for key, result in self._runs.items()
                     if result.run_id == resume_run), "",
                )
            )
            if prior is not None:
                workspace = self._workspaces.get(prior.workspace_id)
        if workspace is None:
            from ore_workspace.permissions import PermissionPolicy

            control_authority = frozenset({
                "lexical_hypotheses", "syntactic_hypotheses", "semantic_hypotheses",
                "retrieved_memories", "unresolved_questions", "module_status",
                "artifact_references", "discourse_context",
            })
            workspace = manager.create_workspace(
                task_text,
                str(trigger.payload.get("goal", "understand_sentence")),
                permissions=PermissionPolicy({
                    COGNITIVE_CONTROL_PRODUCER: control_authority,
                }),
                subscriptions=SubscriptionRegistry(defaults={}),
            )
            self._workspaces[run_id] = workspace
        snapshot = workspace.snapshot()
        workspace_revision = snapshot.revision

        # Step 3 - resolve the governing goal.
        goal_id = f"goal:{workspace.workspace_id}"
        goal_evaluator = self._goal_evaluator
        goal_evaluation = workspace.evaluate_goal(goal_evaluator)
        services = self._services(task_text, run_id)

        # Step 4 - build or resume the cognitive context.
        context = CognitiveContext(
            context_id=new_control_id("context"),
            goal_id=goal_id,
            workspace_revision=workspace_revision,
            status=ContextStatus.RUNNING,
            budget=ContextBudget(
                time_seconds=max(
                    5.0, self.policy.budgets.max_cycles * 2.0,
                ),
                operations=self.policy.budgets.max_operator_applications,
                memory_mb=256.0,
                concurrency=2,
            ),
        )
        self._contexts[run_id] = context

        traces: list[DecisionTrace] = []
        executed_fingerprints: set[str] = set()
        applied_sequence: list[tuple[str, Mapping[str, Any]]] = []
        stagnation_detected = False
        wait_reason: str | None = None

        try:
            for cycle in range(1, self.policy.budgets.max_cycles + 1):
                counters.cycles = cycle
                snapshot = workspace.snapshot()
                stale = context.workspace_revision != snapshot.revision
                if stale:
                    context = context.with_updates(
                        workspace_revision=snapshot.revision,
                    )
                summary = self._state_summary(snapshot.state)
                state_view: dict[str, Any] = dict(snapshot.state)
                state_view["unresolved_conflicts"] = summary["unresolved_conflicts"]
                state_view["competing_hypotheses"] = summary["competing_hypotheses"]
                cycle_fingerprint = self._cycle_fingerprint(
                    goal_id, summary, snapshot.revision,
                )
                context = context.with_updates(cycle_fingerprint=cycle_fingerprint)
                trace = DecisionTraceBuilder(
                    cycle=cycle,
                    context_id=context.context_id,
                    workspace_revision=snapshot.revision,
                    goal_id=goal_id,
                    open_goal_status=goal_evaluation.outcome.value,
                )
                trace.open_question_ids = [
                    str(item.get("question_id")) for item in summary["open_questions"]
                ]
                trace.record_step(1, {
                    "trigger_kind": trigger.kind.value,
                    "idempotency_key": trigger.idempotency_key,
                    "resumed": stale or workspace_revision != snapshot.revision,
                })
                trace.record_step(2, {
                    "workspace_id": workspace.workspace_id,
                    "revision": snapshot.revision,
                    "stale_reconciled": stale,
                })
                trace.record_step(3, {
                    "goal_id": goal_id,
                    "critical_requirements": [
                        "lexical_hypotheses", "syntactic_hypotheses",
                        "semantic_hypotheses",
                    ],
                })
                trace.record_step(4, {
                    "context_id": context.context_id,
                    "depth": context.depth,
                    "status": context.status.value,
                })
                trace.record_step(5, {
                    "cycle_fingerprint": cycle_fingerprint,
                    "satisfied": list(summary["satisfied_conditions"]),
                    "open_questions": len(summary["open_questions"]),
                    "competing": summary["competing_hypotheses"],
                })

                # Phase B - operator generation and filtering (Steps 6-9).
                query = self.operators.query(
                    task_type="sentence_understanding",
                    required_capabilities=(
                        "knowledge.retrieval", "language.semantic_composition",
                        "cognition.verification", "coordination.delegation",
                    ),
                )
                procedure_specs: list[OperatorSpec] = []
                for record in self.procedure_registry.records():
                    healthy, _reason = self.procedure_registry.check_health(
                        record.procedure_id, policy=self.policy,
                    )
                    if healthy:
                        procedure_specs.append(procedure_operator_spec(record))
                eligible_specs = tuple(
                    item.spec for item in query if item.eligible
                ) + tuple(procedure_specs)
                trace.record_step(6, {
                    "registered": len(self.operators.specs()),
                    "eligible": len(eligible_specs),
                    "rejected": [
                        {"operator_id": item.spec.operator_id, "reasons": list(item.reasons)}
                        for item in query if not item.eligible
                    ],
                    "procedure_operators": len(procedure_specs),
                })

                candidates: list[CandidateRecord] = []
                for spec in eligible_specs:
                    ok, failures = evaluate_preconditions(
                        spec, state_view, services,
                    )
                    candidates.append(CandidateRecord(
                        operator_id=spec.operator_id,
                        operator_version=spec.version,
                        normalized_arguments={},
                        arguments_fingerprint="",
                        eligibility_reasons=() if ok else (),
                        precondition_failures=failures,
                    ))
                actions = candidate_actions_for_state(
                    eligible_specs, state_summary=summary, services=services,
                    context=context,
                )
                action_by_operator = {
                    action.operator_id: action for action in actions
                }
                deduped = deduplicate_actions(
                    actions, executed_fingerprints=executed_fingerprints,
                )
                candidate_records: list[CandidateRecord] = []
                for action in deduped:
                    spec = self.operators.get_spec(action.operator_id) or next(
                        (item for item in procedure_specs
                         if item.operator_id == action.operator_id), None,
                    )
                    if spec is None:
                        continue
                    ok, failures = evaluate_preconditions(spec, state_view, services)
                    rejections: list[str] = []
                    if self._already_satisfied(spec, summary):
                        rejections.append("goal_condition_already_satisfied")
                    if spec.estimated_time_seconds > context.budget.time_seconds:
                        rejections.append("time_budget_exceeded")
                    candidate_records.append(CandidateRecord(
                        operator_id=action.operator_id,
                        operator_version=spec.version,
                        normalized_arguments=dict(action.arguments),
                        arguments_fingerprint=action.arguments_fingerprint,
                        eligibility_reasons=("registry_eligible",) if ok else (),
                        precondition_failures=failures,
                        constraint_rejections=tuple(rejections),
                    ))
                trace.record_step(7, {
                    "precondition_failures": {
                        item.operator_id: list(item.precondition_failures)
                        for item in candidate_records
                        if item.precondition_failures
                    },
                })
                trace.record_step(8, {"candidate_count": len(candidate_records)})
                trace.candidates = candidate_records
                trace.record_step(9, {
                    "generated": len(actions),
                    "deduplicated": len(deduped),
                    "removed_redundant": len(actions) - len(deduped),
                })

                # Phase C - retrieval (Steps 10-17).
                retrieved, retrieval_info = self._retrieval_phase(
                    trace,
                    task_text=task_text,
                    summary=summary,
                    context=context,
                    run_epoch=run_epoch,
                    cycle=cycle,
                    counters=counters,
                )
                services = ControlServices(
                    task_text=services.task_text,
                    usg_index=services.usg_index,
                    usg_reference=services.usg_reference,
                    retrieved_knowledge=tuple(
                        item.to_dict() for item in retrieved
                    ),
                    memory_records=services.memory_records,
                    learned_components=services.learned_components,
                    blackboard=services.blackboard,
                    run_id=services.run_id,
                    usage_history=services.usage_history,
                )

                # Phase D - utility, hard constraints, selection (Steps 18-21).
                goal_gaps = [
                    field_name for field_name in (
                        "lexical_hypotheses", "syntactic_hypotheses",
                        "semantic_hypotheses",
                    ) if not snapshot.state.get(field_name)
                ]
                runnable = [
                    item for item in candidate_records
                    if not item.precondition_failures
                    and not item.constraint_rejections
                ]
                specs_by_id = {
                    spec.operator_id: spec for spec in eligible_specs
                }
                estimates = estimate_outcomes(
                    specs_by_id.values(), goal_gaps=goal_gaps,
                )
                utilities = compute_utilities(
                    runnable, estimates, policy=self.policy,
                )
                verdicts = apply_hard_constraints(
                    runnable, specs_by_id,
                    remaining_time_seconds=context.budget.time_seconds,
                    remaining_operations=context.budget.operations
                    - counters.operator_applications,
                    remaining_memory_mb=context.budget.memory_mb,
                    critical_requirements_unmet=goal_evaluation.missing_requirements,
                )
                verdict_map = {item.operator_id: item for item in verdicts}
                scored_records: list[CandidateRecord] = []
                for record in candidate_records:
                    operator_utilities = utilities.get(record.operator_id)
                    scored_records.append(CandidateRecord(
                        operator_id=record.operator_id,
                        operator_version=record.operator_version,
                        normalized_arguments=record.normalized_arguments,
                        arguments_fingerprint=record.arguments_fingerprint,
                        eligibility_reasons=record.eligibility_reasons,
                        precondition_failures=record.precondition_failures,
                        constraint_rejections=record.constraint_rejections,
                        expected_utility=(
                            operator_utilities.expected_utility
                            if operator_utilities else None
                        ),
                        normalized_utility=(
                            operator_utilities.normalized_utility
                            if operator_utilities else None
                        ),
                    ))
                trace.candidates = scored_records
                trace.resource_estimates = {
                    item.operator_id: {
                        "probability": estimates[item.operator_id].probability_useful_progress,
                        "value": estimates[item.operator_id].value,
                        "time_seconds": estimates[item.operator_id].time_seconds,
                        "memory_mb": estimates[item.operator_id].memory_mb,
                        "provenance": estimates[item.operator_id].provenance,
                    }
                    for item in runnable if item.operator_id in estimates
                }
                trace.record_step(18, {
                    "estimate_provenance": {
                        item.operator_id: estimates[item.operator_id].provenance
                        for item in runnable if item.operator_id in estimates
                    },
                })
                trace.record_step(19, {
                    "formulas": [
                        "EU(o) = p*V/(eps + C)",
                        "U(o) = w_g*G + w_i*I + w_c*K - w_t*T - w_r*R - w_f*F",
                    ],
                    "utilities": {
                        key: value.normalized_utility
                        for key, value in utilities.items()
                    },
                })
                trace.record_step(20, {
                    "verdicts": [item.to_dict() for item in verdicts],
                })
                selection = select_action(
                    runnable, utilities=utilities, verdicts=verdict_map,
                    specs=specs_by_id,
                )
                trace.selection = SelectionRecord(
                    selected_operator_id=(
                        selection.selected.operator_id if selection.selected else ""
                    ),
                    selected_arguments_fingerprint=(
                        selection.selected.arguments_fingerprint
                        if selection.selected else ""
                    ),
                    rationale=selection.rationale,
                    parallel_set=tuple(
                        item.operator_id for item in selection.parallel_set
                    ),
                    rejected_alternatives=tuple(
                        item[0] for item in selection.rejected
                    ),
                    utility_score=(
                        utilities[selection.selected.operator_id].normalized_utility
                        if selection.selected
                        and selection.selected.operator_id in utilities else None
                    ),
                )
                trace.record_step(21, {
                    "selected": selection.selected.operator_id
                    if selection.selected else None,
                    "mode": selection.selection_mode,
                    "rationale": selection.rationale,
                    "rejected": [list(item) for item in selection.rejected],
                })

                # Phase F (pre-execution) - routing and meta-cognition (Steps 26-29).
                selected_spec = (
                    specs_by_id.get(selection.selected.operator_id)
                    if selection.selected else None
                )
                learned_check: LearnedComponentCheck | None = None
                if selected_spec is not None:
                    learned_check = self.learned_components.check(
                        "control.toy.sense_ranker",
                        task_type="sentence_understanding",
                        available_memory_mb=context.budget.memory_mb,
                    )
                    processing_path = classify_processing_path(
                        selected_spec.processing_path, learned_check=learned_check,
                    )
                    trace.record_step(26, {
                        "operator_processing_path": selected_spec.processing_path,
                        "classified_processing_path": processing_path,
                        "learned_component": (
                            learned_check.to_dict() if learned_check else None
                        ),
                    })
                    if learned_check is not None:
                        trace.record_step(27, learned_check.to_dict())
                    trace.record_step(28, {
                        "proposal_policy": LEARNED_PROPOSAL_BOUNDARY,
                    })
                report = metacognitive_monitor(
                    summary,
                    policy=self.policy,
                    goal_evaluation={
                        "outcome": goal_evaluation.outcome.value,
                        "score": goal_evaluation.score,
                    },
                    repeated_failures=counters.retries,
                    budget_consumption=(
                        counters.operator_applications / max(
                            1, context.budget.operations,
                        )
                    ),
                )
                trace.metacognitive_report = report.to_dict()
                for escalation in report.escalations:
                    if escalation not in counters.escalations:
                        counters.escalations.append(escalation)
                trace.record_step(29, report.to_dict())

                if selection.selected is None:
                    trace.next_cycle_reason = (
                        "no eligible operator; stopping with a structured reason"
                    )
                    trace.state_transition = "no_eligible_operator"
                    traces.append(trace.build())
                    wait_reason = (
                        "external evidence awaited"
                        if any(
                            item.get("awaiting_external_evidence")
                            for item in summary["open_questions"]
                        ) else None
                    )
                    context = context.with_updates(
                        status=(
                            ContextStatus.WAITING if wait_reason
                            else ContextStatus.FAILED
                        ),
                    )
                    break

                # Phase E - decomposition detection (Step 22).
                action = action_by_operator.get(
                    selection.selected.operator_id,
                ) or OperatorAction(
                    operator_id=selection.selected.operator_id,
                )
                decompose_now = decomposition_justified(
                    competing_hypotheses=summary["competing_hypotheses"],
                    open_questions=len(summary["open_questions"]),
                    available_budget=context.budget,
                )
                trace.record_step(22, {
                    "decomposition_justified": decompose_now,
                    "competing": summary["competing_hypotheses"],
                    "open_questions": len(summary["open_questions"]),
                })

                # Phase G - reservation, dispatch, validation (Steps 30-33).
                estimate = estimates[selection.selected.operator_id]
                reservation: ReservationDecision = self.resource_coordinator.reserve(
                    selection.selected.operator_id,
                    estimated_time_seconds=estimate.time_seconds,
                    estimated_memory_mb=estimate.memory_mb,
                    remaining_time_seconds=context.budget.time_seconds,
                    remaining_memory_mb=context.budget.memory_mb,
                )
                trace.record_step(30, reservation.to_dict())
                if not reservation.granted:
                    trace.next_cycle_reason = (
                        f"reservation denied ({reservation.reason}); deferred"
                    )
                    trace.state_transition = "reservation_denied"
                    traces.append(trace.build())
                    continue

                outcome = None
                attempts = 0
                max_attempts = (
                    selected_spec.retry_policy.max_retries + 1
                    if selected_spec else 1
                )
                execution_error: str | None = None
                while attempts < max_attempts:
                    attempts += 1
                    cycle_start = time.perf_counter()
                    try:
                        if selected_spec is not None and selected_spec.handler_reference.startswith(
                            "ore_control.procedures"
                        ):
                            handler = self._procedure_handler(
                                selection.selected.operator_id,
                            )
                        else:
                            handler = self.operators.handler(
                                selection.selected.operator_id,
                            )
                        if handler is None:
                            raise KeyError(
                                f"operator handler unavailable: "
                                f"{selection.selected.operator_id}"
                            )
                        outcome = handler(context, snapshot.state, action, services)
                    except Exception as error:  # noqa: BLE001 - classified failure
                        execution_error = f"{type(error).__name__}: {error}"
                        outcome = OperatorOutcome(
                            status="failed",
                            payload={},
                            diagnostics={"error": execution_error},
                            failure_classification="module_exception",
                        )
                    outcome = OperatorOutcome(
                        status=outcome.status,
                        payload=outcome.payload,
                        findings=outcome.findings,
                        diagnostics=outcome.diagnostics,
                        decompose=outcome.decompose,
                        child_goals=outcome.child_goals,
                        wait_reason=outcome.wait_reason,
                        checkpoint_events=(
                            {"event": "start", "operator": selection.selected.operator_id},
                            {"event": "completion", "operator": selection.selected.operator_id},
                            *outcome.checkpoint_events,
                        ),
                        elapsed_ms=(time.perf_counter() - cycle_start) * 1000,
                        failure_classification=outcome.failure_classification,
                    )
                    if outcome.status != "failed":
                        break
                    counters.retries += 1
                    if counters.retries > self.policy.budgets.max_retries:
                        break
                trace.execution_result = {
                    "operator_id": selection.selected.operator_id,
                    "status": outcome.status if outcome else "failed",
                    "attempts": attempts,
                    "execution_error": execution_error,
                    "elapsed_ms": outcome.elapsed_ms if outcome else 0.0,
                    "failure_classification": (
                        outcome.failure_classification if outcome else "module_exception"
                    ),
                }
                trace.record_step(31, {
                    "checkpoint_events": [
                        dict(item) for item in (outcome.checkpoint_events if outcome else ())
                    ],
                    "attempts": attempts,
                    "idempotent": True,
                })
                if outcome is None or outcome.status == "failed":
                    self.resource_coordinator.record_outcome(
                        selection.selected.operator_id, False,
                    )
                    trace.state_transition = "operator_failed"
                    trace.next_cycle_reason = "operator failure bounded by retry budget"
                    traces.append(trace.build())
                    context = context.with_updates(status=ContextStatus.FAILED)
                    break
                counters.operator_applications += attempts
                # Suppress only reusable successful outputs (Step 9/23): an
                # execution that produced no integrated findings is not a
                # reusable output and stays re-selectable so stagnation can
                # be detected instead of silently looping on nothing.
                if outcome.findings:
                    executed_fingerprints.add(action.arguments_fingerprint)
                applied_sequence.append((
                    action.operator_id, dict(action.arguments),
                ))
                self.resource_coordinator.record_measurement(
                    selection.selected.operator_id,
                    latency_ms=outcome.elapsed_ms,
                )
                self.resource_coordinator.record_outcome(
                    selection.selected.operator_id, True,
                )
                trace.cost_measurements = {
                    "latency_ms": outcome.elapsed_ms,
                }
                counters.learned_proposals += len(
                    outcome.diagnostics.get("proposals", [])
                )

                validation_passed, validation_checks = self._validate_outcome(
                    outcome, selected_spec,
                )
                trace.validation_result = {
                    "passed": validation_passed,
                    "checks": validation_checks,
                }
                trace.record_step(32, validation_checks)
                if not validation_passed:
                    trace.state_transition = "validation_rejected"
                    trace.next_cycle_reason = (
                        "output failed contract validation; not integrated"
                    )
                    traces.append(trace.build())
                    continue

                integration_decisions = [
                    self._publish_finding(workspace, finding)
                    for finding in outcome.findings
                ]
                trace.integration_decisions = integration_decisions
                trace.record_step(33, {
                    "decisions": integration_decisions,
                    "publication_path": "FindingPublisher->workspace.publish",
                })

                # Phase E execution - child contexts when justified (Steps 22-25).
                if (decompose_now or outcome.decompose) and outcome.child_goals:
                    post_state = workspace.snapshot().state
                    children = create_child_contexts(
                        context, outcome.child_goals,
                        policy=self.policy,
                        workspace_revision=workspace.revision,
                    )
                    trace.record_step(23, {
                        "children": [
                            {"context_id": child.context_id,
                             "goal_text": contract.goal_text}
                            for child, contract in children
                        ],
                    })
                    trace.record_step(24, {
                        "max_depth": self.policy.budgets.max_depth,
                        "max_children": self.policy.budgets.max_children,
                        "child_operation_budget": (
                            self.policy.budgets.child_operation_budget,
                        ),
                    })
                    counters.children += len(children)
                    child_outcomes = [
                        self._run_child(
                            child, contract, ChildBoundaryGuard(contract),
                            services, post_state,
                        )
                        for child, contract in children
                    ]
                    merged_findings, reconciliation_records = (
                        reconcile_child_outcomes(
                            child_outcomes, policy=self.policy,
                        )
                    )
                    child_integration = [
                        self._publish_finding(workspace, finding)
                        for finding in merged_findings
                    ]
                    trace.record_step(25, {
                        "child_outcomes": [
                            outcome.to_dict() for outcome in child_outcomes
                        ],
                        "reconciliation": [
                            dict(item) for item in reconciliation_records
                        ],
                        "integration": child_integration,
                    })
                    trace.integration_decisions = tuple(
                        [*trace.integration_decisions, *child_integration]
                    )

                # Step 34 - recompute goal progress.
                goal_evaluation = workspace.evaluate_goal(goal_evaluator)
                trace.state_transition = (
                    f"goal:{goal_evaluation.outcome.value}"
                )
                trace.record_step(34, {
                    "outcome": goal_evaluation.outcome.value,
                    "score": goal_evaluation.score,
                    "threshold": goal_evaluation.threshold,
                    "missing": list(goal_evaluation.missing_requirements),
                    "formula": "P = sum w_k s_k / sum w_k with critical hard gates",
                })

                # Step 35 - stagnation detection.
                history = self._fingerprint_history.setdefault(context.context_id, [])
                progressed = (
                    not history or history[-1] != cycle_fingerprint
                )
                history.append(cycle_fingerprint)
                repeats = history.count(cycle_fingerprint)
                stagnation = (
                    repeats > self.policy.budgets.stagnation_window
                    and not progressed
                )
                stagnation_detected = stagnation_detected or stagnation
                trace.record_step(35, {
                    "cycle_fingerprint": cycle_fingerprint,
                    "repeats": repeats,
                    "stagnation": stagnation,
                    "retry_budget": self.policy.budgets.max_retries,
                })

                # Step 36 - continue / revise / wait / stop.
                if goal_evaluation.outcome in _TERMINAL_GOAL_OUTCOMES:
                    if goal_evaluation.outcome in _SUCCESS_OUTCOMES:
                        context = context.with_updates(status=ContextStatus.SUCCEEDED)
                        trace.next_cycle_reason = (
                            "goal gates satisfied; stopping successfully"
                        )
                    else:
                        context = context.with_updates(status=ContextStatus.FAILED)
                        trace.next_cycle_reason = (
                            f"goal evaluator returned {goal_evaluation.outcome.value}"
                        )
                    traces.append(trace.build())
                    break
                if outcome.status == "waiting":
                    wait_reason = outcome.wait_reason or "external dependency required"
                    context = context.with_updates(status=ContextStatus.WAITING)
                    trace.next_cycle_reason = f"wait: {wait_reason}"
                    traces.append(trace.build())
                    break
                if stagnation:
                    trace.next_cycle_reason = (
                        "stagnation: repeated state fingerprint without progress"
                    )
                    traces.append(trace.build())
                    context = context.with_updates(status=ContextStatus.FAILED)
                    break
                if counters.operator_applications >= context.budget.operations:
                    trace.next_cycle_reason = "operation budget exhausted"
                    traces.append(trace.build())
                    context = context.with_updates(status=ContextStatus.FAILED)
                    break
                if (
                    time.perf_counter() - started > context.budget.time_seconds
                ):
                    trace.next_cycle_reason = "time budget exhausted"
                    traces.append(trace.build())
                    context = context.with_updates(status=ContextStatus.FAILED)
                    break
                trace.next_cycle_reason = (
                    "useful eligible action remains; continuing the cycle"
                )
                traces.append(trace.build())
            else:
                context = context.with_updates(status=ContextStatus.FAILED)
        finally:
            if owns_manager:
                try:
                    workspace.close()
                    manager.close()
                except Exception:  # pragma: no cover - defensive close
                    pass

        # Phase H - procedure learning after verified success (Steps 37-40).
        status_reason = {
            ContextStatus.SUCCEEDED: "goal gates satisfied",
            ContextStatus.FAILED: "structured failure or budget stop",
            ContextStatus.WAITING: "waiting for an external dependency",
        }.get(context.status, context.status.value)
        if context.status is ContextStatus.SUCCEEDED:
            self._record_usage(_task_terms(task_text), self._now(run_epoch=run_epoch))
            verified = VerifiedRunRecord(
                run_id=run_id,
                goal_fingerprint=sha256_json({
                    "goal": str(snapshot.state.get("current_goal", "understand_sentence")),
                    "task_family": "sentence_understanding",
                }),
                task_fingerprint=sha256_json({"text": task_text}),
                operator_sequence=tuple(
                    ProcedureSequenceStep(
                        operator_id=operator_id, normalized_arguments=arguments,
                    )
                    for operator_id, arguments in applied_sequence
                ),
                status="SUCCEEDED",
                mean_latency_ms=(time.perf_counter() - started) * 1000,
            )
            self._verified_runs.append(verified)
            procedure_events = self._procedure_learning_phase(
                task_text=task_text,
                run_epoch=run_epoch,
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        result = ControlRunResult(
            trigger_fingerprint=fingerprint,
            run_id=run_id,
            context_id=context.context_id,
            status=context.status.value,
            status_reason=(
                f"{status_reason}"
                + (f"; {wait_reason}" if wait_reason else "")
            ),
            cycles=counters.cycles,
            operator_applications=counters.operator_applications,
            retrievals=counters.retrievals,
            retrieval_cost=round(counters.retrieval_cost, 6),
            child_contexts=counters.children,
            conflicts_preserved=_unresolved_conflicts(workspace),
            learned_proposals_used=counters.learned_proposals,
            escalations=tuple(counters.escalations),
            goal_outcome=goal_evaluation.outcome.value,
            goal_score=round(goal_evaluation.score, 6),
            goal_threshold=goal_evaluation.threshold,
            missing_requirements=tuple(goal_evaluation.missing_requirements),
            uncertainty_reasons=tuple(goal_evaluation.uncertainty_reasons),
            decision_traces=tuple(trace.to_dict() for trace in traces),
            procedure_events=tuple(dict(item) for item in procedure_events),
            workspace_id=workspace.workspace_id,
            workspace_revision=workspace.revision,
            elapsed_ms=round(elapsed_ms, 3),
            reproducibility={
                "trigger_fingerprint": fingerprint,
                "workspace_id": workspace.workspace_id,
                "workspace_revision": workspace.revision,
                "operator_versions": {
                    spec.operator_id: spec.version
                    for spec in self.operators.specs()
                },
                "usg_bundle_digest": (
                    self.usg.reference.bundle_digest if self.usg else None
                ),
                "policy_fingerprint": self.policy_fingerprint(),
                "schema_version": COGNITIVE_CONTROL_SCHEMA_VERSION,
                "package_version": CONTROL_PACKAGE_VERSION,
                "stagnation_detected": bool(stagnation_detected),
            },
        )
        for trace_record in traces:
            persist_trace(trace_record, workspace.observer)
        self._runs[fingerprint] = result
        return result

    # -------------------------------------------------------- procedures

    def _procedure_handler(self, procedure_id: str) -> Callable | None:
        """Return the wrapped procedure handler for reuse (Step 40)."""
        record = self.procedure_registry.get(procedure_id)
        if record is None:
            return None
        _spec, handler = wrap_procedure(
            record, registry=self.operators,
            procedure_registry=self.procedure_registry,
        )
        return handler

    def _replay_procedure(
        self, sequence, task_text: str,
    ) -> tuple[bool, float, bool]:
        """Replay one candidate sequence on a held-out input (Step 38)."""
        import time as _time

        started = _time.perf_counter()
        services = self._services(task_text, "procedure_replay")
        state: dict[str, Any] = {"current_input": task_text}
        produced_verified = False
        for step in sequence:
            spec = self.operators.get_spec(step.operator_id)
            if spec is None:
                return False, 0.0, False
            handler = self.operators.handler(step.operator_id)
            if handler is None:
                return False, 0.0, False
            action = OperatorAction(
                operator_id=step.operator_id,
                arguments={
                    key: value for key, value in step.normalized_arguments.items()
                    if not key.startswith("_")
                },
            )
            try:
                outcome = handler(_replay_context(), state, action, services)
            except Exception:  # noqa: BLE001 - replay is a test
                return False, 0.0, False
            if outcome.status == "failed":
                return False, 0.0, False
            for finding in outcome.findings:
                bucket = state.setdefault(finding.field, [])
                if isinstance(bucket, list):
                    bucket.append(dict(finding.payload))
            produced_verified = produced_verified or any(
                isinstance(item, Mapping) and item.get("resolved", False)
                for item in state.get("unresolved_questions", [])
            )
        success = bool(
            state.get("lexical_hypotheses") and state.get("syntactic_hypotheses")
            and state.get("semantic_hypotheses")
        )
        return (
            success,
            (_time.perf_counter() - started) * 1000,
            produced_verified,
        )

    def _procedure_learning_phase(
        self, *, task_text: str, run_epoch: float,
    ) -> list[Mapping[str, Any]]:
        """Run the conservative procedure-learning phase (Steps 37-40)."""
        events: list[Mapping[str, Any]] = []
        candidate = extract_candidate_procedure(
            self._verified_runs, policy=self.policy,
        )
        if candidate is None:
            events.append({
                "event": "procedure_extraction",
                "result": "no_repeatable_sequence",
                "min_repeatable_runs": self.policy.procedures.min_repeatable_runs,
            })
            return events
        events.append({
            "event": "procedure_extraction",
            "result": "candidate_extracted",
            "candidate": candidate.to_dict(),
        })

        def replay(text: str):
            success, latency, verified = self._replay_procedure(
                candidate.operator_sequence, text,
            )
            return ReplayResult(success, latency, verified)

        report = evaluate_generalization(
            candidate,
            held_out_inputs=_held_out_inputs(),
            counterexample_inputs=_counterexample_inputs(),
            replay_executor=replay,
            policy=self.policy,
        )
        events.append({
            "event": "procedure_generalization",
            "report": report.to_dict(),
        })
        decision = evaluate_promotion_gates(
            candidate, report, policy=self.policy, risk_class=RiskClass.LOW,
        )
        events.append({
            "event": "procedure_promotion",
            "decision": decision.to_dict(),
        })
        if decision.promoted:
            existing = self.procedure_registry.get(candidate.procedure_id)
            if existing is not None:
                events.append({
                    "event": "procedure_already_registered",
                    "procedure_id": candidate.procedure_id,
                    "version": existing.version,
                })
                return events
            record = ProcedureRecord(
                procedure_id=candidate.procedure_id,
                version="1.0.0",
                operator_sequence=candidate.operator_sequence,
                goal_fingerprint=candidate.goal_fingerprint,
                preconditions=candidate.preconditions,
                parameters={
                    "sequence": [step.to_dict() for step in candidate.operator_sequence],
                },
                provenance={
                    **dict(candidate.extraction_provenance),
                    "non_authoritative": True,
                },
                test_results=report.to_dict(),
                limitations=(
                    "reference replay evaluation only; no benchmark claim",
                ),
                rollback_information={"prior_version": None, "disable": "registry.disable"},
                risk_class=RiskClass.LOW,
                approved=True,
            )
            self.procedure_registry.register(record)
            events.append({
                "event": "procedure_registered",
                "procedure_id": record.procedure_id,
                "version": record.version,
                "non_authoritative": True,
            })
        return events

    # ------------------------------------------------------------- helpers

    def _already_satisfied(
        self, spec: OperatorSpec, summary: Mapping[str, Any],
    ) -> bool:
        """Detect actions whose goal contributions are already satisfied (Step 9)."""
        if not spec.writes_workspace or not spec.write_fields:
            # A read-only operator never contributes goal state; it is never
            # "already satisfied" and stays selectable for monitoring work.
            return False
        satisfied = set(summary["satisfied_conditions"])
        if not set(spec.write_fields) <= satisfied:
            return False
        unresolved_work = (
            summary["competing_hypotheses"] > 0
            or summary["open_questions"]
            or summary["unresolved_conflicts"] > 0
        )
        return not unresolved_work

    def policy_fingerprint(self) -> str:
        """Return a stable fingerprint of the coordination policy."""
        return sha256_json({
            "goal_threshold": self.policy.goal_threshold,
            "max_cycles": self.policy.budgets.max_cycles,
            "max_operator_applications": (
                self.policy.budgets.max_operator_applications
            ),
            "retrieval": {
                "alpha": self.policy.retrieval.alpha,
                "beta": self.policy.retrieval.beta,
                "gamma": self.policy.retrieval.gamma,
                "delta": self.policy.retrieval.delta,
                "lam": self.policy.retrieval.lam,
            },
            "utility": {
                "w_g": self.policy.utility.w_g,
                "w_i": self.policy.utility.w_i,
                "w_c": self.policy.utility.w_c,
                "w_t": self.policy.utility.w_t,
                "w_r": self.policy.utility.w_r,
                "w_f": self.policy.utility.w_f,
            },
            "activation_decay": self.policy.activation_decay,
        })


LEARNED_PROPOSAL_BOUNDARY = (
    "learned outputs are proposals with provenance; schema validation, "
    "evidence checks, and deterministic verification precede publication, "
    "and authoritative USG knowledge is never mutated"
)


def _task_terms(text: str) -> tuple[str, ...]:
    """Tokenize task text deterministically (reference tokenization)."""
    return tuple(
        token for token in (
            part.strip(".,!?;:").casefold() for part in text.split()
        ) if token
    )


def _cache_candidates(summary: Mapping[str, Any]) -> list[RetrievalCandidate]:
    """Lift existing workspace knowledge into retrieval candidates (Step 11)."""
    candidates: list[RetrievalCandidate] = []
    for item in summary.get("evidence_ids", []):
        candidates.append(RetrievalCandidate(
            record_reference=str(item),
            source_type="workspace_cache",
            cue_match_features={},
            semantic_relevance=0.4,
            provenance={"source": "workspace_retrieved_memories"},
            retrieval_cost=0.1,
            version_reference="workspace:cache",
            inclusion_reasons=("cache_hit",),
        ))
    return candidates


def _unresolved_conflicts(workspace: SharedWorkspace) -> int:
    """Count unresolved conflicts preserved in the authoritative workspace."""
    records = getattr(workspace.conflicts, "records", None)
    if callable(records):
        try:
            return sum(
                1 for record in records()
                if not getattr(record, "resolved", False)
            )
        except Exception:  # pragma: no cover - defensive
            return 0
    return 0


def _held_out_inputs() -> tuple[str, ...]:
    """Return the deterministic held-out replay inputs (Step 38)."""
    return (
        "The captain saw the boat.",
        "A boy opened the door.",
        "The fisherman sat near the river.",
    )


def _counterexample_inputs() -> tuple[str, ...]:
    """Return counterexample inputs the procedure must reject (Step 38)."""
    return ("Zzz qqq wwww.", "??? !!! ###")


def _replay_context() -> CognitiveContext:
    """Return the detached context used during procedure replay (Step 38)."""
    return CognitiveContext(
        context_id="procedure_replay_context",
        goal_id="goal:replay",
        workspace_revision=0,
        status=ContextStatus.RUNNING,
        budget=ContextBudget(operations=8),
    )


__all__ = ["CognitiveControlCoordinator", "LEARNED_PROPOSAL_BOUNDARY"]
