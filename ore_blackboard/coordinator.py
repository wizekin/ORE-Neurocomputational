"""Blackboard-based cognitive coordination: the 32-step Algorithm 1 orchestrator.

This coordinator connects ORE's existing shared workspace, typed contracts,
module registry, bounded scheduler, adaptive resource coordination, USG adapters,
and SQLite logging into one bounded, inspectable blackboard loop. It does not
create a competing workspace and it never bypasses typed publication: every
mutation flows through ``WorkspaceStateManager`` validation, field merge
policies, atomic commits, and post-commit events.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from ore_contracts import ContractCatalog
from ore_contracts.module_registry import ModuleRegistry
from ore_workspace import (
    FindingPublisher,
    GoalEvaluator,
    ModuleScheduler,
    SharedWorkspace,
    SubscriptionRegistry,
    WorkspaceManager,
)
from ore_workspace.evidence import EvidenceContribution, HypothesisScorer
from ore_workspace.models import Provenance, new_id
from ore_usg import register_usg_contracts

from .bridge import (
    BLACKBOARD_USG_MODULE_KNOWLEDGE_CONTRACT,
    register_blackboard_contracts,
)
from .config import BlackboardPolicy
from .contracts import (
    BLACKBOARD_PACKAGE_VERSION,
    COGNITIVE_CONTROL_PRODUCER,
    BlackboardTaskRequest,
    BlackboardTaskResult,
    FailureClassification,
    InputRoot,
    RunIdentity,
    SupportedHypothesis,
    build_input_root,
)
from .execution import (
    ComputationRegistry,
    SpecialistInvocation,
    build_input_root_artifact,
    build_memory_artifact,
    build_rules_artifact,
    build_snapshot_artifact,
    make_run_context,
)
from .hypotheses import SCORE_DEFINITION, goal_progress, summarize_open_conflicts
from .planning import (
    build_dependency_plan,
    build_permission_policy,
    build_task_profile,
    register_specialist_modules,
)
from .relevance import measure_workspace_changes
from .registry import operation_by_step
from .resources import BlackboardResourceCoordinator, exponential_backoff
from .specialists import default_specialists
from .trace import ProvenanceTraceBuilder
from .usg_integration import BlackboardUSGSnapshot, publish_usg_reference

_STATUS_LABELS = {
    "completed": "completed",
    "completed_with_uncertainty": "completed_with_uncertainty",
    "needs_more_processing": "needs_more_processing",
    "blocked": "blocked",
    "failed": "failed",
}

_EXCEPTION_PREFIXES = (
    "ValueError", "TypeError", "KeyError", "PermissionError", "RuntimeError",
    "IndexError", "AttributeError", "ZeroDivisionError", "AssertionError",
    "OSError", "Exception", "StopIteration", "RecursionError", "ArithmeticError",
)


@dataclass(slots=True)
class _CycleState:
    """Mutable bookkeeping for one bounded blackboard run (Steps 24-26)."""

    rounds: int = 0
    activations: int = 0
    module_calls: int = 0
    retries: int = 0
    failures: dict[str, dict[str, Any]] = field(default_factory=dict)
    stagnation_rounds: int = 0
    stopped_reason: str = ""
    delivery_failures: int = 0


class BlackboardCoordinator:
    """Run the 32-step blackboard coordination algorithm over registered specialists."""

    def __init__(
        self,
        *,
        specialists: Iterable[Any] | None = None,
        policy: BlackboardPolicy | None = None,
        catalog: ContractCatalog | None = None,
        repository_root: str | Path | None = None,
        logger_database_path: str | Path | None = None,
    ) -> None:
        """Bind the shared catalog, registry, specialists, and coordination policy."""
        self.policy = policy or BlackboardPolicy()
        self.catalog = catalog if catalog is not None else ContractCatalog()
        register_blackboard_contracts(self.catalog)
        register_usg_contracts(self.catalog)
        self.registry = ModuleRegistry(self.catalog)
        self.specialists = tuple(
            specialists if specialists is not None else default_specialists()
        )
        self.repository_root = (
            Path(repository_root).resolve()
            if repository_root else Path(__file__).resolve().parents[1]
        )
        self.logger_database_path = (
            Path(logger_database_path).resolve() if logger_database_path else None
        )
        self._registration = register_specialist_modules(
            self.registry, self.catalog, self.specialists,
        )
        self._results: dict[str, BlackboardTaskResult] = {}
        # Reuse one immutable snapshot for repeated tasks against the same large
        # USG bundle. Replacing the digest replaces the cache (bounded to one).
        self._usg_snapshot_cache: BlackboardUSGSnapshot | None = None
        self._usg_snapshot_cache_digest: str | None = None

    # ------------------------------------------------------------------ helpers

    def _selected_usg_route_demands(
        self, request: BlackboardTaskRequest,
    ) -> tuple[dict[str, Any], ...]:
        """Find selected specialists that explicitly declare USG knowledge domains."""
        requested = set(request.required_capabilities) | set(request.optional_capabilities)
        selected: set[str] = set()
        manifests = {
            specialist.describe().identity.module_id: specialist.describe()
            for specialist in self.specialists
        }
        for module_id, manifest in manifests.items():
            if requested.intersection(manifest.capabilities):
                selected.add(module_id)
        pending = list(selected)
        while pending:
            module_id = pending.pop()
            manifest = manifests.get(module_id)
            if manifest is None:
                continue
            for dependency in manifest.dependencies:
                if dependency.required and dependency.module_id not in selected:
                    selected.add(dependency.module_id)
                    pending.append(dependency.module_id)
        demands: list[dict[str, Any]] = []
        allowed_domains = {
            "LEXICAL", "SEMANTIC", "RELATIONAL", "PREDICATE",
            "MORPHOLOGY", "CONTEXT", "EVIDENCE", "SOURCE_PROVENANCE",
        }
        for specialist in self.specialists:
            manifest = specialist.describe()
            module_id = manifest.identity.module_id
            domains = tuple(sorted({
                str(item).upper()
                for item in getattr(specialist, "usg_knowledge_domains", ())
            }))
            if module_id not in selected or not domains:
                continue
            unknown_domains = set(domains) - allowed_domains
            if unknown_domains:
                raise ValueError(
                    f"{module_id} declares unsupported USG domains: {sorted(unknown_domains)}"
                )
            knowledge_port = next(
                (item for item in manifest.input_ports if item.port_id == "usg_knowledge"),
                None,
            )
            if (
                knowledge_port is None
                or knowledge_port.contract_id != BLACKBOARD_USG_MODULE_KNOWLEDGE_CONTRACT
                or knowledge_port.accepted_versions != ("1.0.0",)
            ):
                raise ValueError(
                    f"{module_id} declares USG knowledge domains but lacks the "
                    "ORE.BLACKBOARD.USG_MODULE_KNOWLEDGE@1.0.0 input port"
                )
            mandatory = bool(set(request.required_capabilities).intersection(
                manifest.capabilities
            ))
            demands.append({
                "module_id": module_id,
                "domains": domains,
                "priority": float(getattr(specialist, "usg_knowledge_priority", 0.5)),
                "mandatory": mandatory,
            })
        return tuple(sorted(demands, key=lambda item: item["module_id"]))

    def _make_usg_artifact_provider(
        self,
        usg: BlackboardUSGSnapshot | None,
        specialist: Any,
        *,
        run_id: str,
        input_root: InputRoot,
        language: str | None = None,
        delivery_scope: str = "default",
    ):
        """Create a per-invocation provider for the specialist's bounded USG slice."""
        domains = tuple(getattr(specialist, "usg_knowledge_domains", ()))
        if usg is None or not domains:
            return None
        module_id = specialist.describe().identity.module_id
        initial_terms = tuple(item.token for item in input_root.tokens)

        def provide(snapshot: Any) -> Mapping[str, Any]:
            """Route only the current module's terms and domains from the cached snapshot."""
            state = snapshot.state
            terms = initial_terms
            predicates: tuple[str, ...] = ()
            if module_id == "ore.blackboard.semantic_specialist":
                lexical_state = state.get("lexical_hypotheses", {})
                if isinstance(lexical_state, Mapping):
                    terms = tuple(str(item) for item in lexical_state.keys())
                syntactic = state.get("syntactic_hypotheses", ())
                predicates = tuple(
                    str(item.get("verb"))
                    for item in syntactic
                    if isinstance(item, Mapping) and item.get("verb")
                )
            knowledge = usg.router.route(
                module_id,
                domains=domains,
                lexical_terms=terms,
                predicate_terms=predicates,
                language=language,
                workspace_revision=int(getattr(snapshot, "revision", 0)),
                delivery_scope=delivery_scope,
            )
            return {
                "usg_knowledge": usg.module_knowledge_artifact(run_id, knowledge),
            }

        return provide

    def _publish_usg_feedback(
        self, workspace: SharedWorkspace, usg: BlackboardUSGSnapshot | None,
        publications: Iterable[Any],
    ) -> tuple[Any, ...]:
        """Record specialist feedback and publish it as separate ORE unresolved questions."""
        if usg is None:
            return ()
        for result in publications:
            finding_id = getattr(result, "finding_id", None)
            finding = (
                workspace.findings.get(finding_id)
                or workspace.pending_findings.get(finding_id)
                or workspace.rejected_findings.get(finding_id)
            ) if finding_id else None
            if finding is not None:
                usg.router.record_finding_feedback(
                    source_module=finding.producer_module,
                    finding_id=finding.finding_id,
                    workspace_revision=workspace.revision,
                    payload=finding.payload,
                )
        feedback_publications = []
        for feedback in usg.router.unannounced_feedback(
            publication_scope=workspace.workspace_id,
        ):
            publisher = FindingPublisher(workspace, COGNITIVE_CONTROL_PRODUCER)
            finding = publisher.create_finding(
                "unresolved_questions",
                {
                    **feedback.to_dict(),
                    "question_id": feedback.feedback_id,
                    "question": feedback.reason,
                    "resolved": False,
                    "status": "unresolved",
                    "question_type": "usg_knowledge_feedback",
                },
                key=f"usg_feedback:{feedback.feedback_id}",
                finding_type="usg_knowledge_feedback",
                provenance=Provenance(
                    source="blackboard_usg_feedback_channel",
                    rules_applied=(
                        "feedback_is_separate_from_usg",
                        "digest_scoped_feedback",
                        "deduplicate_identical_requests",
                    ),
                    notes="This ORE feedback record does not modify the source USG bundle.",
                ),
            )
            result = workspace.publish(finding)
            feedback_publications.append(result)
            usg.router.mark_feedback_published(
                feedback.feedback_id,
                publication_scope=workspace.workspace_id,
            )
        return tuple(feedback_publications)

    def _subscription_spec(
        self, specialist: Any,
    ) -> tuple[tuple[str, ...], Mapping[str, float], float]:
        """Resolve one specialist's task-specific subscription declaration (Step 8)."""
        fields = tuple(specialist.subscription_fields)
        weights = getattr(specialist, "sensitivity_weights", None) or {
            field: 1.0 for field in fields
        }
        threshold = float(getattr(specialist, "activation_threshold", 0.0) or 0.0)
        return fields, weights, threshold

    @staticmethod
    def _record_step(
        records: list[dict[str, Any]], step: int, details: Mapping[str, Any],
        *, cycle: int | None = None,
    ) -> None:
        """Append one operation trace record for an executed algorithm step."""
        spec = operation_by_step(step)
        records.append({
            "operation_id": spec.operation_id,
            "step": spec.step,
            "title": spec.title,
            "cycle": cycle,
            "details": dict(details),
        })

    def _publish_input_root(self, workspace: SharedWorkspace, input_root: InputRoot) -> Any:
        """Publish the immutable input evidence root (Step 4)."""
        publisher = FindingPublisher(workspace, COGNITIVE_CONTROL_PRODUCER)
        finding = publisher.create_finding(
            "artifact_references",
            input_root.to_dict(),
            key=f"input_root:{input_root.input_digest[:16]}",
            finding_type="input_root_observation",
            provenance=Provenance(
                source="task_request",
                rules_applied=("sha256_input_digest", "source_group_preservation"),
                notes="Immutable evidence root; downstream findings link back to this record.",
            ),
        )
        return workspace.publish(finding)

    def _publish_module_status(self, workspace: SharedWorkspace, module_id: str, status: str) -> None:
        """Publish a module lifecycle status finding through the control producer (Step 26)."""
        publisher = FindingPublisher(workspace, COGNITIVE_CONTROL_PRODUCER)
        finding = publisher.create_finding(
            "module_status",
            {"status": status},
            key=module_id,
            finding_type="module_execution_status",
            provenance=Provenance(
                source="blackboard_coordinator",
                rules_applied=("failure_classification",),
            ),
        )
        workspace.publish(finding)

    def _make_bridge(self, workspace: SharedWorkspace, specialist: Any, input_root: InputRoot):
        """Create the artifact-to-finding bridge for one specialist invocation (Step 17)."""
        from .bridge import ArtifactFindingBridge

        manifest = specialist.describe()
        return ArtifactFindingBridge(
            workspace,
            manifest.identity.module_id,
            allowed_fields=frozenset(specialist.write_fields.values()),
            input_length=len(input_root.primary_text()),
            provenance_checks=self.policy.ablations.provenance_checks,
            dependency_deferral=self.policy.ablations.dependency_deferral,
        )

    def _build_resource_plan(
        self, request: BlackboardTaskRequest,
        resource_coordinator: BlackboardResourceCoordinator,
    ) -> Mapping[str, Any]:
        """Build the adaptive resource execution plan for the task profile (Steps 14-15)."""
        if not self.policy.ablations.resource_aware_choice:
            return {"enabled": False, "reason": "resource_aware_choice ablated"}
        from ore_contracts.resource_coordination import ExecutionPlanBuilder

        profile = build_task_profile(request, self.policy)
        builder = ExecutionPlanBuilder(self.registry, resource_coordinator.estimator)
        return builder.build(profile).to_dict()

    def _gate_module(self, workspace: SharedWorkspace, module_id: str) -> dict[str, Any]:
        """Raise a module's activation threshold while its circuit is open (Steps 14, 26)."""
        subscription = workspace.subscriptions.get(module_id)
        if subscription is None:
            return {}
        original = {
            "fields": sorted(subscription.fields),
            "sensitivity": dict(subscription.sensitivity),
            "threshold": subscription.threshold,
        }
        workspace.subscriptions.register(
            module_id, original["fields"],
            sensitivity=original["sensitivity"], threshold=1e9,
        )
        return original

    @staticmethod
    def _ungate_module(
        workspace: SharedWorkspace, module_id: str, original: Mapping[str, Any],
    ) -> None:
        """Restore a gated module's original subscription threshold."""
        if not original:
            return
        workspace.subscriptions.register(
            module_id, original["fields"],
            sensitivity=dict(original["sensitivity"]), threshold=original["threshold"],
        )

    # ------------------------------------------------------------------ run()

    def run(self, request: BlackboardTaskRequest) -> BlackboardTaskResult:
        """Execute the full 32-step blackboard coordination algorithm for one task."""
        if not isinstance(request, BlackboardTaskRequest):
            raise TypeError("request must be a BlackboardTaskRequest")
        step_records: list[dict[str, Any]] = []
        started = time.perf_counter()

        # Step 1 - validate the request and establish the run boundary.
        fingerprint = request.request_fingerprint()
        cached = self._results.get(request.task_id)
        if cached is not None:
            if cached.reproducibility.get("task_fingerprint") != fingerprint:
                raise ValueError(
                    "task identity was reused with different request content; "
                    "use a new task_id for a new task"
                )
            self._record_step(step_records, 1, {
                "task_id": request.task_id, "idempotent_replay": True,
            })
            from dataclasses import replace
            return replace(cached, idempotent_replay=True)
        run_id = new_id("run")
        run_identity = RunIdentity(
            task_id=request.task_id,
            run_id=run_id,
            task_fingerprint=fingerprint,
            prior_run_id=request.prior_run_id,
        )
        run_context = make_run_context(run_id, self.policy, request.execution_mode)
        self._record_step(step_records, 1, {
            "task_id": request.task_id, "run_id": run_id,
            "task_fingerprint": fingerprint, "idempotency_key": request.task_id,
        })

        # Step 2 - create one isolated shared workspace (or resume an explicit one).
        manager = WorkspaceManager(
            repository_root=self.repository_root,
            logger_database_path=self.logger_database_path,
            enable_logging=self.policy.enable_logging,
        )
        input_root = build_input_root(request.input)
        if request.resume_workspace_id is not None:
            workspace = manager.get_workspace(request.resume_workspace_id)
            run_identity = RunIdentity(
                task_id=request.task_id, run_id=run_id, task_fingerprint=fingerprint,
                prior_run_id=request.prior_run_id,
                resumed_workspace_id=request.resume_workspace_id,
            )
        else:
            workspace = manager.create_workspace(
                input_root.primary_text(),
                request.goal,
                permissions=build_permission_policy(self._registration.authority_matrix),
                subscriptions=SubscriptionRegistry(defaults={}),
            )
        baseline_state = dict(workspace.snapshot().state)
        self._record_step(step_records, 2, {
            "workspace_id": workspace.workspace_id, "run_id": run_id,
            "resumed": request.resume_workspace_id is not None,
        })

        # Step 3 - the logical blackboard layers over one controlled shared state.
        from .contracts import LAYER_FIELDS

        layer_model = {layer.name: sorted(fields) for layer, fields in sorted(
            LAYER_FIELDS.items(), key=lambda item: int(item[0]),
        )}
        self._record_step(step_records, 3, {
            "layer_count": len(layer_model), "one_hot_sum": 1,
            "layer_fields": layer_model,
            "lifecycle_order": ["externally_supplied", "candidate", "reviewed", "accepted"],
        })

        # Step 4 - preserve the immutable input evidence root.
        input_root_result = self._publish_input_root(workspace, input_root)
        input_root_finding_id = input_root_result.finding_id
        self._record_step(step_records, 4, {
            "input_digest": input_root.input_digest,
            "token_count": len(input_root.tokens),
            "source_count": len(input_root.sources),
            "finding_id": input_root_finding_id,
        })

        # Step 5 - identify actual USG demand before opening the source snapshot.
        usg_route_demands = (
            self._selected_usg_route_demands(request)
            if self.policy.ablations.usg_retrieval else ()
        )
        usg: BlackboardUSGSnapshot | None = None
        usg_artifacts: dict[str, Any] = {}
        snapshot_cache_reused = False
        if usg_route_demands and request.usg_bundle is not None:
            requested_bundle_digest = sha256_json(dict(request.usg_bundle))
            if (
                self._usg_snapshot_cache is not None
                and self._usg_snapshot_cache_digest == requested_bundle_digest
            ):
                usg = self._usg_snapshot_cache
                snapshot_cache_reused = True
                if request.usg_expected_digest is not None and (
                    usg.reference.bundle_digest != request.usg_expected_digest
                ):
                    raise ValueError("cached USG snapshot digest does not match usg_expected_digest")
            else:
                usg = BlackboardUSGSnapshot(
                    request.usg_bundle, expected_digest=request.usg_expected_digest,
                )
                self._usg_snapshot_cache = usg
                self._usg_snapshot_cache_digest = requested_bundle_digest
            publish_usg_reference(workspace, usg.reference)
            self._record_step(step_records, 5, {
                "attached": True,
                "bundle_id": usg.reference.bundle_id,
                "bundle_digest": usg.reference.bundle_digest,
                "namespace": usg.reference.namespace,
                "resource_categories": dict(usg.reference.resource_categories),
                "selected_modules": [item["module_id"] for item in usg_route_demands],
                "routed_domains": {
                    item["module_id"]: list(item["domains"])
                    for item in usg_route_demands
                },
                "snapshot_cache_reused": snapshot_cache_reused,
                "full_lexicon_attached_to_task": False,
                "read_only": True,
            })
        else:
            reason = (
                "no selected specialist declares a USG knowledge requirement"
                if not usg_route_demands else
                "no USG bundle supplied"
                if request.usg_bundle is None else
                "USG retrieval ablated"
            )
            self._record_step(step_records, 5, {
                "attached": False,
                "reason": reason,
                "selected_usg_modules": [item["module_id"] for item in usg_route_demands],
            })

        # Step 6 - register specialists and their authority boundaries.
        required = set(request.required_capabilities)
        capabilities: set[str] = set()
        for specialist in self.specialists:
            capabilities.update(specialist.describe().capabilities)
        missing_mandatory = tuple(sorted(required - capabilities))
        coverage = len(required & capabilities) / len(required) if required else 1.0
        self._record_step(step_records, 6, {
            "registered_modules": list(self._registration.registered_modules),
            "capability_coverage": coverage,
            "missing_mandatory_capabilities": list(missing_mandatory),
        })
        if missing_mandatory:
            return self._blocked_result(
                request, run_identity, workspace, manager, step_records, started,
                input_root, f"missing mandatory capabilities: {list(missing_mandatory)}",
            )

        # Step 7 - construct the dependency and compatibility graph.
        snapshot = workspace.snapshot()
        planning_artifacts: dict[str, Any] = {
            "input_root": build_input_root_artifact(input_root, run_id, request.task_id),
            "workspace_snapshot": build_snapshot_artifact(snapshot, run_id),
        }
        if request.memory_store:
            planning_artifacts["memory_store"] = build_memory_artifact(
                tuple(request.memory_store), run_id,
            )
        if request.verification_rules:
            planning_artifacts["verification_rules"] = build_rules_artifact(
                tuple(request.verification_rules), run_id,
            )
        plan = build_dependency_plan(self.registry, self.catalog, request, planning_artifacts)
        self._record_step(step_records, 7, {
            "selected_modules": list(plan.selected_modules),
            "layers": [list(item) for item in plan.layers],
            "edges": [list(item) for item in plan.edges],
            "executable": plan.executable,
            "missing_capabilities": list(plan.missing_capabilities),
            "missing_inputs": list(plan.missing_inputs),
            "cycle_nodes": list(plan.cycle_nodes),
            "capability_coverage": plan.capability_coverage,
        })
        if not plan.executable:
            return self._blocked_result(
                request, run_identity, workspace, manager, step_records, started,
                input_root,
                "non-executable plan: " + ", ".join(
                    list(plan.missing_capabilities) + list(plan.missing_dependencies)
                    + list(plan.missing_inputs) + list(plan.cycle_nodes)
                ),
            )

        # Step 8 - task-specific module subscriptions through the scheduler.
        scheduler = ModuleScheduler(
            max_workers=(
                self.policy.budgets.max_workers if self.policy.ablations.bounded_parallelism else 1
            ),
            max_rounds=self.policy.budgets.max_rounds,
            max_activations=self.policy.budgets.max_activations,
            default_timeout_seconds=self.policy.budgets.default_timeout_seconds,
        )
        computation_registry = ComputationRegistry()
        measurements: list[tuple[str, float]] = []
        invocations: dict[str, SpecialistInvocation] = {}
        adapter_artifacts = {
            key: value for key, value in planning_artifacts.items()
            if key != "workspace_snapshot"
        }
        artifact_provider_factory = lambda selected_specialist: self._make_usg_artifact_provider(
            usg,
            selected_specialist,
            run_id=run_id,
            input_root=input_root,
            language=(
                str(request.metadata["language"])
                if request.metadata.get("language") is not None else None
            ),
            delivery_scope=request.task_id,
        )
        for specialist in self.specialists:
            fields, weights, threshold = self._subscription_spec(specialist)
            if not self.policy.ablations.selective_subscriptions:
                fields = tuple(sorted(set(fields) | {
                    "current_input", "lexical_hypotheses", "syntactic_hypotheses",
                    "semantic_hypotheses", "retrieved_memories", "unresolved_questions",
                    "artifact_references",
                }))
                threshold = 0.0
            manifest = specialist.describe()
            invocation = SpecialistInvocation(
                specialist,
                run_context=run_context,
                catalog=self.catalog,
                task_artifacts=adapter_artifacts,
                bridge_factory=lambda snap, s=specialist: self._make_bridge(
                    workspace, s, input_root,
                ),
                computation_registry=computation_registry,
                task_id=request.task_id,
                subscribed_fields=fields,
                measurement_sink=lambda module_id, duration, store=measurements: store.append(
                    (module_id, duration),
                ),
                artifact_provider=artifact_provider_factory(specialist),
            )
            invocation.artifact_provider_factory = artifact_provider_factory
            invocations[manifest.identity.module_id] = invocation
            scheduler.register_module(
                workspace, manifest.identity.module_id, invocation, fields,
                sensitivity=dict(weights), threshold=threshold,
            )
        self._record_step(step_records, 8, {
            "subscriptions": workspace.subscriptions.as_dict(),
        })

        # Step 9 - goal requirements and the completion evaluator.
        evaluator = GoalEvaluator(
            weights=self.policy.goal_weights,
            threshold=self.policy.goal_threshold,
            critical_requirements=self.policy.critical_requirements,
        )
        resource_coordinator = BlackboardResourceCoordinator(
            self.policy, task_family=request.task_family,
        )
        resource_plan = dict(self._build_resource_plan(request, resource_coordinator))
        usg_route_priority_plan = (
            resource_coordinator.prioritize_knowledge_routes(list(usg_route_demands))
            if usg_route_demands and self.policy.ablations.resource_aware_choice
            else {
                "route_order": [item["module_id"] for item in usg_route_demands],
                "routes": [],
                "mandatory_routes_preserved": True,
                "heuristic": True,
            }
        )
        resource_plan["usg_knowledge_routes"] = usg_route_priority_plan
        resource_plan["usg_route_cache"] = usg.router.statistics() if usg is not None else {}
        critical = (self.policy.critical_requirements or {}).get(
            request.goal, ("lexical_hypotheses", "syntactic_hypotheses", "semantic_hypotheses"),
        )
        self._record_step(step_records, 9, {
            "goal": request.goal, "threshold": self.policy.goal_threshold,
            "critical_requirements": list(critical),
        })
        self._record_step(step_records, 14, {
            "heuristic_information_gain": True,
            "reliability_prior": [
                self.policy.reliability_prior_alpha, self.policy.reliability_prior_beta,
            ],
            "circuit_breaker_threshold": self.policy.circuit_breaker_threshold,
            "comparison_score": "U_m heuristic (ore_blackboard.resources)",
        })

        # Pre-seed caller-supplied retrieved memories with their own provenance.
        for memory in request.retrieved_memories or ():
            publisher = FindingPublisher(workspace, COGNITIVE_CONTROL_PRODUCER)
            workspace.publish(publisher.create_finding(
                "retrieved_memories",
                dict(memory),
                key=str(memory.get("memory_id", new_id("memory"))),
                finding_type="caller_supplied_memory",
                provenance=Provenance(
                    source="caller_request",
                    rules_applied=("caller_evidence_preservation",),
                ),
            ))

        # Steps 10-24 - the bounded blackboard cycle.
        cycle = _CycleState()
        trace_builder = ProvenanceTraceBuilder(input_root)
        if usg is not None:
            trace_builder.add_usg_bundle(
                usg.reference.bundle_id, usg.reference.bundle_digest,
            )
        goal_evaluations: list[Any] = []
        previous_goal = None
        stop_reason = "idle_no_pending_work"
        while True:
            current_snapshot = workspace.snapshot()
            changes = measure_workspace_changes(baseline_state, current_snapshot.state)
            deltas = {change.field: change.delta for change in changes}
            if not deltas:
                stop_reason = cycle.stopped_reason or "idle_no_pending_work"
                break
            budget_reason = self._budget_stop_reason(cycle, started)
            if budget_reason:
                stop_reason = budget_reason
                cycle.stopped_reason = budget_reason
                break
            self._record_step(step_records, 11, {
                "changes": [
                    {"field": change.field, "delta": round(change.delta, 4),
                     "metric": change.metric}
                    for change in changes
                ],
            }, cycle=cycle.rounds)
            eligible = workspace.subscriptions.eligible(deltas)
            self._record_step(step_records, 12, {
                "eligible": [
                    {"module_id": module_id, "relevance": round(score, 4)}
                    for module_id, score in sorted(eligible)
                ],
            }, cycle=cycle.rounds)

            # Step 14 - circuit-breaker gating before execution.
            gated: dict[str, dict[str, Any]] = {}
            if self.policy.ablations.resource_aware_choice:
                for module_id, _score in eligible:
                    if module_id in invocations and resource_coordinator.circuit_open(module_id):
                        gated[module_id] = self._gate_module(workspace, module_id)
                        resource_coordinator.record_outcome(module_id, False)
            try:
                # Steps 10, 15, 16 - bounded parallel execution on detached snapshots.
                round_started = time.perf_counter()
                self._record_step(step_records, 10, {
                    "snapshot_revision": current_snapshot.revision,
                }, cycle=cycle.rounds)
                self._record_step(step_records, 15, {
                    "max_workers": scheduler.max_workers,
                    "gated_modules": sorted(gated),
                }, cycle=cycle.rounds)
                activations, publications = scheduler.run_round(
                    workspace, tuple(deltas), deltas,
                )
                feedback_publications = self._publish_usg_feedback(
                    workspace, usg, publications,
                )
                if feedback_publications:
                    publications = tuple(publications) + tuple(feedback_publications)
                cycle.rounds += 1
                cycle.activations += sum(1 for item in activations if item.queued)
                cycle.module_calls += sum(
                    1 for item in activations
                    if item.queued and not _suppressed(invocations, item.module_id)
                )
                self._record_step(step_records, 16, {
                    "activations": [
                        {
                            "module_id": item.module_id,
                            "queued": item.queued,
                            "relevance": round(item.relevance_score, 4),
                            "reason": item.reason,
                        }
                        for item in activations
                    ],
                    "round_duration_ms": round(
                        (time.perf_counter() - round_started) * 1000.0, 3,
                    ),
                }, cycle=cycle.rounds - 1)
                self._record_step(step_records, 17, {
                    "publications": [
                        {
                            "decision": result.decision.value,
                            "finding_id": result.finding_id,
                            "changed_fields": list(result.changed_fields),
                        }
                        for result in publications
                    ],
                }, cycle=cycle.rounds - 1)
                self._record_step(step_records, 18, {
                    "validation_decisions": sorted({
                        result.decision.value for result in publications
                    }),
                }, cycle=cycle.rounds - 1)

                # Step 26 - bounded recovery for failed activations.
                failed = [
                    item for item in activations
                    if not item.queued and (
                        "timed out" in item.reason
                        or item.reason.startswith(_EXCEPTION_PREFIXES)
                    )
                ]
                if failed:
                    self._record_step(step_records, 26, {
                        "failed_modules": sorted({item.module_id for item in failed}),
                        "retry_budget_remaining": (
                            self.policy.budgets.max_retries - cycle.retries
                        ),
                    }, cycle=cycle.rounds - 1)
                for failure in failed:
                    cycle.failures.setdefault(failure.module_id, {
                        "module_id": failure.module_id,
                        "classification": _classify_activation(failure.reason),
                        "reason": failure.reason,
                        "attempts": 0,
                        "recovered": False,
                    })
                for failure in failed:
                    recovered = self._recover(
                        workspace, scheduler, invocations, deltas, cycle,
                        failure.module_id, input_root,
                    )
                    cycle.failures[failure.module_id]["recovered"] = recovered

                # Step 13 - committed events were dispatched by the state manager.
                deliveries = list(workspace.delivery_results)
                failed_deliveries = [item for item in deliveries if not item.delivered]
                cycle.delivery_failures = len(failed_deliveries)
                self._record_step(step_records, 13, {
                    "delivered": len(deliveries) - len(failed_deliveries),
                    "failed_deliveries": len(failed_deliveries),
                }, cycle=cycle.rounds - 1)

                # Steps 19-23 - deferral, merge, conflict, commit, routing records.
                self._record_step(step_records, 19, {
                    "pending_findings": sorted(workspace.pending_findings),
                }, cycle=cycle.rounds - 1)
                self._record_step(step_records, 20, {
                    "merge_policies": "field-specific merge policies applied by the "
                                      "workspace state manager",
                }, cycle=cycle.rounds - 1)
                self._record_step(step_records, 21, {
                    "conflicts": len(workspace.conflicts.list(workspace.workspace_id)),
                }, cycle=cycle.rounds - 1)
                self._record_step(step_records, 22, {
                    "workspace_revision": workspace.revision,
                }, cycle=cycle.rounds - 1)
                self._record_step(step_records, 23, {
                    "suppressed_computations": computation_registry.count_suppressed(),
                }, cycle=cycle.rounds - 1)
                self._record_step(step_records, 24, {
                    "rounds_used": cycle.rounds,
                    "max_rounds": self.policy.budgets.max_rounds,
                }, cycle=cycle.rounds - 1)

                # Step 25 - evaluate the goal after every meaningful revision.
                goal = workspace.evaluate_goal(evaluator)
                goal_evaluations.append(goal)
                trace_builder.add_goal_evaluation(
                    new_id("goal_eval"), goal.outcome.value, goal.score, cycle.rounds - 1,
                )
                self._record_step(step_records, 25, {
                    "outcome": goal.outcome.value,
                    "score": round(goal.score, 4),
                    "missing_requirements": list(goal.missing_requirements),
                }, cycle=cycle.rounds - 1)
                if goal.outcome.value in ("failed", "blocked"):
                    stop_reason = goal.outcome.value
                    break
                queued_any = any(item.queued for item in activations)
                revision_advanced = workspace.revision > current_snapshot.revision
                if not queued_any and not publications and not revision_advanced:
                    # No relevant subscriber remains for the last committed change:
                    # the blackboard is drained and the goal evaluation stands.
                    stop_reason = (
                        goal.outcome.value if goal.outcome.value != "needs_more_processing"
                        else cycle.stopped_reason or "idle_no_pending_work"
                    )
                    break
                progress = goal_progress(previous_goal, goal)
                if cycle.rounds > 1 and abs(progress) <= self.policy.stagnation.epsilon:
                    cycle.stagnation_rounds += 1
                else:
                    cycle.stagnation_rounds = 0
                previous_goal = goal
                if cycle.stagnation_rounds >= self.policy.stagnation.patience_rounds:
                    stop_reason = "stagnation_no_progress"
                    cycle.stopped_reason = stop_reason
                    break
                baseline_state = dict(current_snapshot.state)
            finally:
                for module_id, original in gated.items():
                    self._ungate_module(workspace, module_id, original)

        for module_id, failure in cycle.failures.items():
            if not failure["recovered"]:
                self._publish_module_status(workspace, module_id, "failed")
        for record in computation_registry.records():
            trace_builder.add_invocation(
                record.module_id, record.invocation_id, status=record.status,
                duration_ms=record.duration_ms, suppressed=record.suppressed,
                error=record.error,
            )

        # Step 27 - reconstruct the reasoning and provenance trace.
        trace_builder = self._populate_finding_trace(
            workspace, trace_builder, input_root_finding_id,
        )
        final_findings = list(workspace.findings)
        completeness, untraceable = trace_builder.completeness(final_findings)
        final_state = workspace.snapshot().state
        integrity_ok = trace_builder.verify_input_integrity(str(final_state.get("current_input", "")))
        self._record_step(step_records, 27, {
            "trace_completeness": round(completeness, 4),
            "untraceable_finding_ids": list(untraceable),
            "integrity_ok": integrity_ok,
            "node_count": len(trace_builder.nodes),
            "edge_count": len(trace_builder.edges),
        })

        # Step 28 - produce the task result with separate epistemic categories.
        final_goal = workspace.evaluate_goal(evaluator)
        self._record_step(step_records, 28, {
            "status": final_goal.outcome.value,
            "epistemic_sections": (
                "observations", "imported_knowledge_references", "supported_hypotheses",
                "unresolved_alternatives", "verified_conclusions", "recommended_next_actions",
            ),
        })
        result = self._assemble_result(
            request=request,
            run_identity=run_identity,
            workspace=workspace,
            final_goal=final_goal,
            stop_reason=stop_reason,
            cycle=cycle,
            step_records=step_records,
            trace_builder=trace_builder,
            completeness=completeness,
            integrity_ok=integrity_ok,
            usg=usg,
            plan=plan,
            resource_plan=resource_plan,
            started=started,
            invocations=computation_registry,
        )

        # Step 31 - learned updates stay outside ordinary inference.
        for module_id, duration_ms in measurements:
            resource_coordinator.record_measurement(module_id, latency_ms=duration_ms)
        for module_id, failure in cycle.failures.items():
            resource_coordinator.record_outcome(module_id, bool(failure["recovered"]))
        self._record_step(step_records, 31, {
            "measurements_recorded": len(measurements),
            "transaction": "post_run_measurement_only",
            "inference_answer_unchanged": True,
        })

        # Step 32 - close cleanly and retain reproducibility metadata.
        scheduler.close(wait_for_tasks=True)
        resource_coordinator.close()
        manager.close()
        self._record_step(step_records, 32, {
            "workspace_id": workspace.workspace_id,
            "final_revision": workspace.revision,
            "state_retained": True,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "reproducibility": dict(result.reproducibility),
        })
        from dataclasses import replace as _replace

        result = _replace(
            result,
            trace=tuple(step_records),
            operation_coverage=tuple(
                dict.fromkeys(record["operation_id"] for record in step_records)
            ),
        )

        self._results[request.task_id] = result
        return result

    # ------------------------------------------------------------------ pieces

    def _recover(
        self,
        workspace: SharedWorkspace,
        scheduler: ModuleScheduler,
        invocations: Mapping[str, SpecialistInvocation],
        deltas: Mapping[str, float],
        cycle: _CycleState,
        module_id: str,
        input_root: InputRoot,
    ) -> bool:
        """Retry a failed module within the shared retry budget (Step 26)."""
        failure = cycle.failures.get(module_id)
        if failure is None:
            return False
        if cycle.retries >= self.policy.budgets.max_retries:
            failure["reason"] += "; retry budget exhausted"
            return False
        delay = exponential_backoff(
            cycle.retries,
            initial_seconds=self.policy.retry_backoff_initial_seconds,
            cap_seconds=self.policy.retry_backoff_cap_seconds,
            jitter_seconds=0.0,
        )
        if delay > 0:
            time.sleep(delay)
        cycle.retries += 1
        failure["attempts"] += 1
        activations, _publications = scheduler.run_round(
            workspace, tuple(deltas), dict(deltas),
        )
        cycle.activations += sum(1 for item in activations if item.queued)
        cycle.module_calls += sum(
            1 for item in activations
            if item.queued and not _suppressed(invocations, item.module_id)
        )
        retried = [item for item in activations if item.module_id == module_id]
        if retried and retried[0].queued and "completed" in retried[0].reason:
            return True
        alternative = self._recover_with_alternative(
            workspace, scheduler, invocations, module_id, input_root, deltas, cycle,
        )
        if alternative is not None:
            failure["reason"] += f"; recovered via alternative module {alternative}"
            return True
        return False

    def _recover_with_alternative(
        self,
        workspace: SharedWorkspace,
        scheduler: ModuleScheduler,
        invocations: Mapping[str, SpecialistInvocation],
        failed_module_id: str,
        input_root: InputRoot,
        deltas: Mapping[str, float],
        cycle: _CycleState,
    ) -> str | None:
        """Bind one compatible alternative provider for a failed module (Step 26)."""
        manifest = self.registry.get_manifest(failed_module_id)
        if manifest is None:
            return None
        for capability in manifest.capabilities:
            for record in self.registry.find_by_capability(capability):
                candidate = record.module_id
                if candidate == failed_module_id or not record.runnable:
                    continue
                specialist = self.registry.get_backend(candidate)
                if specialist is None or candidate in invocations:
                    continue
                fields, weights, threshold = self._subscription_spec(specialist)
                failed_invocation = invocations[failed_module_id]
                invocation = SpecialistInvocation(
                    specialist,
                    run_context=failed_invocation.run_context,
                    catalog=self.catalog,
                    task_artifacts=failed_invocation.task_artifacts,
                    bridge_factory=lambda snap, s=specialist: self._make_bridge(
                        workspace, s, input_root,
                    ),
                    computation_registry=failed_invocation.computation_registry,
                    task_id=failed_invocation.task_id,
                    subscribed_fields=fields,
                    artifact_provider=(
                        failed_invocation.artifact_provider_factory(specialist)
                        if getattr(failed_invocation, "artifact_provider_factory", None)
                        else None
                    ),
                )
                if hasattr(failed_invocation, "artifact_provider_factory"):
                    invocation.artifact_provider_factory = failed_invocation.artifact_provider_factory
                invocations[candidate] = invocation
                scheduler.register_module(
                    workspace, candidate, invocation, fields,
                    sensitivity=dict(weights), threshold=threshold,
                )
                workspace.permissions.grant(
                    candidate, specialist.write_fields.values(),
                )
                # Activate the alternative immediately with the triggering deltas;
                # the failed module simply fails again and other modules are
                # suppressed by their reusable successful outputs.
                activations, _publications = scheduler.run_round(
                    workspace, tuple(deltas), dict(deltas),
                )
                cycle.activations += sum(1 for item in activations if item.queued)
                cycle.module_calls += sum(
                    1 for item in activations
                    if item.queued and not _suppressed(invocations, item.module_id)
                )
                activated = [
                    item for item in activations
                    if item.module_id == candidate and item.queued
                    and "completed" in item.reason
                ]
                if activated:
                    return candidate
                return None
        return None

    def _budget_stop_reason(self, cycle: _CycleState, started: float) -> str | None:
        """Return the budget-exhaustion reason, if any (Step 24)."""
        budgets = self.policy.budgets
        if cycle.rounds >= budgets.max_rounds:
            return "round budget exhausted"
        if cycle.activations >= budgets.max_activations:
            return "activation budget exhausted"
        if cycle.module_calls >= budgets.max_module_calls:
            return "module-call budget exhausted"
        if budgets.max_time_seconds is not None:
            if time.perf_counter() - started >= budgets.max_time_seconds:
                return "time budget exhausted"
        return None

    def _populate_finding_trace(
        self,
        workspace: SharedWorkspace,
        trace_builder: ProvenanceTraceBuilder,
        input_root_finding_id: str,
    ) -> ProvenanceTraceBuilder:
        """Add findings, events, and conflicts to the reasoning trace (Step 27)."""
        if input_root_finding_id:
            trace_builder.link_input_root(input_root_finding_id)
        for finding_id, finding in sorted(workspace.findings.items()):
            trace_builder.add_finding(
                finding_id,
                producer_module=finding.producer_module,
                decision="accepted",
                target_field=finding.target.field,
                payload=dict(finding.payload),
                dependencies=finding.dependencies,
                usg_references=finding.payload.get("usg_references", ()),
                span=finding.provenance.source_span,
            )
        for finding_id, finding in sorted(workspace.pending_findings.items()):
            trace_builder.add_finding(
                finding_id,
                producer_module=finding.producer_module,
                decision="deferred",
                target_field=finding.target.field,
                payload=dict(finding.payload),
                dependencies=finding.dependencies,
            )
        for finding_id, finding in sorted(workspace.rejected_findings.items()):
            trace_builder.add_finding(
                finding_id,
                producer_module=finding.producer_module,
                decision="rejected",
                target_field=finding.target.field,
                payload=dict(finding.payload),
                dependencies=finding.dependencies,
            )
        for event in workspace.events:
            trace_builder.add_event(
                event.event_id, event.workspace_revision, event.changed_fields,
            )
        for conflict in workspace.conflicts.list(workspace.workspace_id):
            trace_builder.add_conflict(conflict.conflict_id, conflict.conflicting_findings)
        return trace_builder

    def _assemble_result(
        self, *, request: BlackboardTaskRequest, run_identity: RunIdentity,
        workspace: SharedWorkspace, final_goal: Any, stop_reason: str,
        cycle: _CycleState, step_records: list[dict[str, Any]],
        trace_builder: ProvenanceTraceBuilder, completeness: float,
        integrity_ok: bool, usg: BlackboardUSGSnapshot | None, plan: Any,
        resource_plan: Mapping[str, Any], started: float,
        invocations: ComputationRegistry,
    ) -> BlackboardTaskResult:
        """Assemble the final result with epistemic categories separate (Step 28)."""
        state = workspace.snapshot().state
        resource_plan = dict(resource_plan)
        resource_plan["usg_route_cache"] = (
            usg.router.statistics() if usg is not None else {
                "cache_hits": 0, "cache_misses": 0, "route_calls": 0,
                "cached_module_routes": 0, "feedback_count": 0,
                "feedback_pending": 0,
            }
        )
        observations: list[dict[str, Any]] = [
            {
                "observation_type": "input_root",
                "input_digest": trace_builder.input_root.input_digest,
                "token_count": len(trace_builder.input_root.tokens),
                "sources": [
                    {"source_id": item.source_id, "source_group": item.source_group}
                    for item in trace_builder.input_root.sources
                ],
            }
        ]
        unknown_tokens = {
            record.get("word")
            for records in state.get("lexical_hypotheses", {}).values()
            for record in records if record.get("unknown")
        }
        for token in sorted(unknown_tokens):
            observations.append({
                "observation_type": "unknown_token_mass",
                "token": token,
                "note": "No lexical entry; preserved as unknown rather than guessed.",
            })
        imported_references: list[dict[str, Any]] = []
        if usg is not None:
            imported_references.append(usg.reference.to_dict())
            observations.append({
                "observation_type": "usg_route_statistics",
                **usg.router.statistics(),
                "snapshot_cache_available_for_reuse": (
                    self._usg_snapshot_cache is usg
                ),
                "source_database_mutation_performed": False,
            })
        for finding in workspace.findings.values():
            for reference in finding.payload.get("usg_references", ()):
                imported_references.append({
                    "reference_type": "usg_record",
                    "bundle_id": reference.get("bundle_id"),
                    "bundle_digest": reference.get("bundle_digest"),
                    "record_id": reference.get("record_id"),
                    "source_reference": reference.get("source_reference"),
                    "used_by_finding": finding.finding_id,
                })
        supported = self._rank_hypotheses(workspace)
        alternatives = list(summarize_open_conflicts(
            workspace, self.policy.ablations.hypothesis_conflict_handling,
        ))
        unresolved_questions = [
            record for record in state.get("unresolved_questions", [])
            if not record.get("resolved")
        ]
        for question in unresolved_questions:
            alternatives.append({
                "alternative_type": "unresolved_question",
                "question_id": question.get("question_id"),
                "question": question.get("question"),
            })
        verified = [
            {
                "question_id": record.get("question_id"),
                "proposition": record.get("proposition"),
                "status": record.get("status"),
            }
            for record in state.get("unresolved_questions", [])
            if record.get("resolved")
        ]
        next_actions: list[str] = []
        if unresolved_questions:
            next_actions.append(
                "Resolve the retained competing readings with additional context or an "
                "explicit evidence-backed resolution rule."
            )
        for token in sorted(unknown_tokens):
            next_actions.append(f"Provide lexical knowledge for the unknown token {token!r}.")
        if cycle.failures:
            next_actions.append(
                "Investigate failed specialist executions recorded in the failure summary."
            )
        if "budget exhausted" in stop_reason:
            next_actions.append(
                "Raise task budgets or supply missing inputs to continue processing."
            )
        module_versions = {
            specialist.describe().identity.module_id: specialist.describe().identity.module_version
            for specialist in self.specialists
        }
        reproducibility = {
            "task_fingerprint": run_identity.task_fingerprint,
            "workspace_id": workspace.workspace_id,
            "revision": workspace.revision,
            "input_digest": trace_builder.input_root.input_digest,
            "usg_bundle_digest": usg.reference.bundle_digest if usg else None,
            "usg_snapshot_resource_id": usg.resource.resource_id if usg else None,
            "usg_router_statistics": usg.router.statistics() if usg else {},
            "module_versions": dict(sorted(module_versions.items())),
            "policy_version": BLACKBOARD_PACKAGE_VERSION,
            "artifact_digests": {
                "input_root": trace_builder.input_root.input_digest,
                "usg_bundle": usg.reference.bundle_digest if usg else None,
            },
        }
        status = _STATUS_LABELS.get(final_goal.outcome.value, final_goal.outcome.value)
        if status == "completed_with_uncertainty" and not self.policy.allow_uncertain_completion:
            status = "needs_more_processing"
        status_reason = stop_reason
        if status in ("completed", "completed_with_uncertainty") and stop_reason in (
            "idle_no_pending_work", "completed", "completed_with_uncertainty", "",
            "stagnation_no_progress",
        ):
            status_reason = "goal satisfied; blackboard drained with no pending subscriber work"
        if status == "failed":
            status_reason = "a required module reported failure: " + ", ".join(
                sorted(cycle.failures)
            )
        return BlackboardTaskResult(
            task_id=request.task_id,
            run_id=run_identity.run_id,
            workspace_id=workspace.workspace_id,
            status=status,
            status_reason=status_reason,
            goal_score=final_goal.score,
            goal_threshold=final_goal.threshold,
            component_scores=dict(final_goal.component_scores),
            missing_requirements=tuple(final_goal.missing_requirements),
            uncertainty_reasons=tuple(final_goal.uncertainty_reasons),
            observations=tuple(observations),
            imported_knowledge_references=tuple(imported_references),
            supported_hypotheses=tuple(supported),
            unresolved_alternatives=tuple(alternatives),
            verified_conclusions=tuple(verified),
            recommended_next_actions=tuple(next_actions),
            workspace_revision=workspace.revision,
            accepted_finding_count=len(workspace.findings),
            pending_finding_count=len(workspace.pending_findings),
            rejected_finding_count=len(workspace.rejected_findings),
            conflict_count=len(workspace.conflicts.list(workspace.workspace_id)),
            rounds=cycle.rounds,
            activations=cycle.activations,
            module_calls=cycle.module_calls,
            retries=cycle.retries,
            suppressed_computations=invocations.count_suppressed(),
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            trace_completeness=round(completeness, 4),
            integrity_ok=integrity_ok,
            failure_summary=tuple(dict(item) for item in cycle.failures.values()),
            reproducibility=reproducibility,
            plan=plan.to_dict(),
            resource_plan=dict(resource_plan),
            module_results={
                record.module_id: {
                    "status": record.status,
                    "duration_ms": round(record.duration_ms, 3),
                    "suppressed": record.suppressed,
                    "error": record.error,
                }
                for record in invocations.records()
            },
            trace=tuple(step_records),
            operation_coverage=tuple(
                dict.fromkeys(record["operation_id"] for record in step_records)
            ),
        )

    def _rank_hypotheses(self, workspace: SharedWorkspace) -> list[SupportedHypothesis]:
        """Rank retained semantic hypotheses with the transparent evidence score (Steps 21, 28)."""
        if not self.policy.ablations.hypothesis_conflict_handling:
            return []
        import re as _re

        state = workspace.snapshot().state
        semantics = state.get("semantic_hypotheses", [])
        memories = state.get("retrieved_memories", [])
        lexical = state.get("lexical_hypotheses", {})
        scorer = HypothesisScorer()
        ranked: list[SupportedHypothesis] = []
        for structure in semantics:
            hypothesis_id = structure.get("finding_id")
            proposition = structure.get("proposition", "")
            contributions: list[EvidenceContribution] = []
            for word, sense in (
                (structure.get("agent"), structure.get("agent_sense")),
                (structure.get("patient"), structure.get("patient_sense")),
                (structure.get("predicate"), structure.get("predicate")),
            ):
                if not word:
                    continue
                records = lexical.get(str(word).casefold(), ())
                matching = tuple(
                    record for record in records
                    if sense and record.get("gloss") == sense
                ) or (records[:1] if records else ())
                for record in matching[:1]:
                    if record.get("unknown"):
                        continue
                    contributions.append(EvidenceContribution(
                        source_id=f"lexical:{word}",
                        contribution=float(record.get("confidence") or 0.5),
                        weight=1.0,
                        correlation_group=f"lexical:{word}",
                    ))
            for evidence in structure.get("usg_evidence", ()):
                if not isinstance(evidence, Mapping):
                    continue
                status = str(evidence.get("support_status", "UNASSESSED"))
                contribution = (
                    1.0 if status == "SUPPORTS"
                    else -1.0 if status == "CONTRADICTS"
                    else 0.0
                )
                if status in {"SUPPORTS", "CONTRADICTS"}:
                    contributions.append(EvidenceContribution(
                        source_id=f"usg-evidence:{evidence.get('evidence_id', evidence.get('source_id', 'unknown'))}",
                        contribution=contribution,
                        weight=1.0,
                        # Each evidence record remains independently comparable.
                        # Repeated references to the same evidence ID are deduplicated
                        # upstream by the semantic specialist's evidence ledger.
                        correlation_group=(
                            f"usg-evidence:{evidence.get('evidence_id', 'unknown')}"
                        ),
                        metadata={
                            "support_status": status,
                            "verified_fact": False,
                            "relationship_id": evidence.get("relationship_id"),
                        },
                    ))
            sense_tokens: set[str] = set()
            for key in ("agent_sense", "patient_sense"):
                value = structure.get(key)
                if value:
                    sense_tokens.update(
                        token.casefold() for token in _re.findall(r"\w+", str(value))
                    )
            for memory in memories:
                keywords = {
                    str(item).casefold()
                    for item in memory.get("matched_keywords", ())
                }
                if keywords & sense_tokens:
                    contributions.append(EvidenceContribution(
                        source_id=f"memory:{memory.get('memory_id')}",
                        contribution=float(memory.get("relevance_score") or 0.0),
                        weight=1.0,
                        correlation_group="memory",
                    ))
            score = scorer.score(contributions)
            ranked.append(SupportedHypothesis(
                hypothesis_id=hypothesis_id,
                label=proposition,
                score=round(score.score, 4),
                used_evidence=tuple(
                    {
                        "source_id": item.source_id,
                        "contribution": item.contribution,
                        "weight": item.weight,
                        "correlation_group": item.correlation_group,
                        "metadata": dict(item.metadata),
                    }
                    for item in score.used_evidence
                ),
                suppressed_sources=tuple(score.suppressed_correlated_sources),
                score_definition=(
                    SCORE_DEFINITION
                    + "; includes USG SUPPORTS (+1), CONTRADICTS (-1), and ORE memory/lexical contributions; "
                    "uncalibrated evidence score, not probability or fact verification"
                ),
                calibrated_probability=False,
            ))
        ranked.sort(key=lambda item: (-item.score, item.hypothesis_id or ""))
        return ranked

    def _blocked_result(
        self, request: BlackboardTaskRequest, run_identity: RunIdentity,
        workspace: SharedWorkspace, manager: WorkspaceManager,
        step_records: list[dict[str, Any]], started: float, input_root: InputRoot,
        reason: str,
    ) -> BlackboardTaskResult:
        """Return a blocked result without starting a partial pipeline (Step 7)."""
        self._record_step(step_records, 28, {"status": "blocked", "reason": reason})
        self._record_step(step_records, 31, {"measurements_recorded": 0})
        self._record_step(step_records, 32, {"state_retained": True, "blocked": True})
        reproducibility = {
            "task_fingerprint": run_identity.task_fingerprint,
            "workspace_id": workspace.workspace_id,
            "revision": workspace.revision,
            "input_digest": input_root.input_digest,
            "usg_bundle_digest": None,
            "module_versions": {},
            "policy_version": BLACKBOARD_PACKAGE_VERSION,
            "artifact_digests": {"input_root": input_root.input_digest},
        }
        result = BlackboardTaskResult(
            task_id=request.task_id,
            run_id=run_identity.run_id,
            workspace_id=workspace.workspace_id,
            status="blocked",
            status_reason=reason,
            goal_score=0.0,
            goal_threshold=self.policy.goal_threshold,
            component_scores={},
            missing_requirements=tuple(request.required_capabilities),
            uncertainty_reasons=(),
            observations=(
                {
                    "observation_type": "input_root",
                    "input_digest": input_root.input_digest,
                    "token_count": len(input_root.tokens),
                },
            ),
            imported_knowledge_references=(),
            supported_hypotheses=(),
            unresolved_alternatives=(),
            verified_conclusions=(),
            recommended_next_actions=(
                f"Resolve the blocking condition before rerunning: {reason}",
            ),
            workspace_revision=workspace.revision,
            accepted_finding_count=len(workspace.findings),
            pending_finding_count=len(workspace.pending_findings),
            rejected_finding_count=len(workspace.rejected_findings),
            conflict_count=0,
            rounds=0,
            activations=0,
            module_calls=0,
            retries=0,
            suppressed_computations=0,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            trace_completeness=1.0,
            integrity_ok=True,
            failure_summary=(),
            reproducibility=reproducibility,
            plan={},
            resource_plan={},
            module_results={},
            trace=tuple(step_records),
            operation_coverage=tuple(
                dict.fromkeys(record["operation_id"] for record in step_records)
            ),
        )
        manager.close()
        self._results[request.task_id] = result
        return result


def _suppressed(invocations: Mapping[str, SpecialistInvocation], module_id: str) -> bool:
    """Return whether the module's latest invocation was computation-suppressed."""
    invocation = invocations.get(module_id)
    return bool(invocation and invocation.last_record and invocation.last_record.suppressed)


def _classify_activation(reason: str) -> str:
    """Classify one failed activation reason (Step 26)."""
    if "timed out" in reason:
        return FailureClassification.TIMEOUT.value
    return FailureClassification.MODULE_EXCEPTION.value


__all__ = ["BlackboardCoordinator"]
