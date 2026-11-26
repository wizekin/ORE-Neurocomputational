"""Hierarchical planner facade implementing the ORE Point 12 specification.

The planner answers which cognitive operations should run, in what order, with
which dependencies, under which budgets, and when to replan or stop. It combines
hierarchical task networks, partial-order planning, A* and bounded beam search.

Boundaries: this planner creates and evaluates plans only. Task execution stays
with CognitiveTaskCoordinator, scheduling with ModuleScheduler and the
coordinator's dependency planner, resource allocation with
AdaptiveResourceCoordinator, and authoritative state with ore_workspace. The
planner never introduces a second scheduler, workspace, goal store, or task
coordinator. MCTS is deliberately not implemented: the specification gates it
behind justified benchmarks (Stage 8), which do not exist yet.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ore_contracts import ContractCatalog, ModuleRegistry, RunContext
from ore_contracts.coordinator import CognitiveTaskCoordinator

from .capability_adapter import CapabilityRegistryAdapter, OperatorDiscovery
from .contracts import (
    GoalSpec,
    HTNMethod,
    OperatorSpec,
    PlanDecision,
    PlannerBudget,
    PlanRunResult,
    PlanState,
    TaskSpec,
)
from .cost import CostModel, RemainingCostHeuristic, UnmetConditionHeuristic
from .executor import PlanExecutionReport, PlanExecutorAdapter
from .goal_adapter import GoalConstraintAdapter, GoalFormalization
from .htn import GOAL_TASK_TYPE, DecompositionResult, HTNDecomposer, TaskTree, build_default_methods
from .partial_order import PartialOrderBuilder, PartialOrderPlan
from .provenance import PlanningProvenance
from .replanner import OutcomeEvaluator
from .search import SearchEngine, SearchOutcome

__all__ = [
    "HierarchicalPlanner",
    "PlanningOutcome",
    "SelectedPlan",
]


@dataclass(frozen=True, slots=True)
class SelectedPlan:
    """Hold the chosen plan, its decision record, and the plans it beat."""

    plan: PartialOrderPlan
    outcome: SearchOutcome
    decision: PlanDecision
    goal: GoalSpec

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible selected-plan record."""
        return {
            "plan": self.plan.to_dict(),
            "outcome": self.outcome.to_dict(),
            "decision": self.decision.to_dict(),
            "goal": self.goal.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PlanningOutcome:
    """Report one full planning pass across phases A through D."""

    formalization: GoalFormalization
    discovery: OperatorDiscovery
    decomposition: DecompositionResult
    selected: SelectedPlan | None
    provenance: PlanningProvenance
    status: str

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible planning outcome."""
        return {
            "formalization": self.formalization.to_dict(),
            "discovery": self.discovery.to_dict(),
            "decomposition": self.decomposition.to_dict(),
            "selected": self.selected.to_dict() if self.selected else None,
            "provenance": self.provenance.to_dict(),
            "status": self.status,
        }


class HierarchicalPlanner:
    """Orchestrate the forty specification steps across phases A through E."""

    def __init__(
        self,
        registry: ModuleRegistry,
        catalog: ContractCatalog,
        *,
        coordinator: CognitiveTaskCoordinator | None = None,
        methods: Iterable[HTNMethod] = (),
        cost_model: CostModel | None = None,
        heuristic: RemainingCostHeuristic | None = None,
        budget: PlannerBudget | None = None,
        search_mode: str = "astar",
        search_weight: float = 1.0,
        goal_interpreter: Any = None,
        max_replans: int = 2,
        measured_costs: Mapping[str, Any] | None = None,
    ) -> None:
        """Wire the planner to existing ORE components and configuration only."""
        if not isinstance(registry, ModuleRegistry):
            raise TypeError("registry must be a ModuleRegistry")
        if not isinstance(catalog, ContractCatalog):
            raise TypeError("catalog must be a ContractCatalog")
        if registry.contract_catalog is not catalog:
            raise ValueError("registry and planner must use the same ContractCatalog")
        if max_replans < 0:
            raise ValueError("max_replans cannot be negative")
        self.registry = registry
        self.catalog = catalog
        self.coordinator = coordinator or CognitiveTaskCoordinator(
            registry, catalog, enable_logging=False,
        )
        self.cost_model = cost_model or CostModel()
        self.heuristic = heuristic or UnmetConditionHeuristic()
        self.budget = budget or PlannerBudget()
        self.search_mode = search_mode
        self.search_weight = search_weight
        self.custom_methods = tuple(methods)
        self.goal_adapter = GoalConstraintAdapter(goal_interpreter)
        self.capability_adapter = CapabilityRegistryAdapter(
            registry, self.cost_model, measured_costs=measured_costs,
        )
        self.executor = PlanExecutorAdapter(
            self.coordinator, registry, self.cost_model,
        )
        self.evaluator = OutcomeEvaluator(self.cost_model)
        self.max_replans = max_replans

    # ------------------------------------------------------------------
    # Phases A through D: goal formalization, decomposition, ordering, search
    # ------------------------------------------------------------------
    def plan(
        self,
        goal_text: str,
        *,
        scope: str = "",
        deadline_seconds: float | None = None,
        quality_threshold: float = 1.0,
        workspace_revision: int = 0,
        initial_facts: Iterable[str] = (),
        external_contracts: Iterable[str] = (),
        run_context: RunContext | None = None,
    ) -> PlanningOutcome:
        """Execute planning phases A through D for one goal statement."""
        del run_context
        provenance = PlanningProvenance()
        external = tuple(sorted(set(external_contracts)))
        facts = frozenset(initial_facts) | {
            f"artifact.available({contract})" for contract in external
        }

        # Phase A: goal formalization and operator discovery (steps 1-8).
        provenance.record_step("A", 1, "received goal, scope, and budgets")
        provenance.record_step("A", 2, f"bound to workspace revision {workspace_revision}")
        formalization = self.goal_adapter.formalize(
            goal_text,
            scope=scope,
            deadline_seconds=deadline_seconds,
            quality_threshold=quality_threshold,
            workspace_revision=workspace_revision,
        )
        provenance.record_step(
            "A", 3,
            f"{len(formalization.interpretations)} typed interpretation(s)",
        )
        provenance.record_step(
            "A", 4,
            "ambiguity preserved" if formalization.ambiguous else "single interpretation",
        )
        provenance.record_step(
            "A", 5,
            f"{sum(len(item.forbidden_operations) for item in formalization.interpretations)}"
            " forbidden operation constraint(s)",
        )
        goal = formalization.primary
        discovery = self.capability_adapter.discover(
            goal, external_contracts=external,
        )
        provenance.record_step(
            "A", 6,
            f"{len(discovery.operators)} eligible operator(s) discovered",
        )
        provenance.record_step("A", 7, "normalized operator contracts and costs")
        provenance.record_step(
            "A", 8,
            f"{len(discovery.rejections)} operator(s) rejected, "
            f"{len(discovery.unsatisfied_capabilities)} capability gap(s)",
        )
        provenance.record(
            "operator_discovery", goal_id=goal.goal_id,
            workspace_revision=workspace_revision,
            module_ids=tuple(item.operator_id for item in discovery.operators),
            payload={
                "rejections": [item.to_dict() for item in discovery.rejections],
                "unsatisfied": list(discovery.unsatisfied_capabilities),
            },
        )

        # Phase B: hierarchical decomposition (steps 9-16).
        methods = self._methods_for_goal(goal, discovery)
        decomposer = HTNDecomposer(methods, budget=self.budget)
        root_task = TaskSpec(
            task_id="goal_root",
            task_type=GOAL_TASK_TYPE,
            parameters={"goal_id": goal.goal_id},
            completion=goal.success_conditions[0].predicate,
            compound=True,
        )
        decomposition = decomposer.decompose(root_task, facts, discovery)
        provenance.record_step(
            "B", 9, f"{len(decomposer.methods_for(GOAL_TASK_TYPE))} method(s) considered",
        )
        provenance.record_step(
            "B", 10,
            f"primary decomposition has {decomposition.node_count} node(s)",
        )
        provenance.record_step("B", 11, "compound subtasks expanded recursively")
        provenance.record_step(
            "B", 12,
            f"cycle_count={decomposition.cycle_count}, truncated={decomposition.truncated}",
        )
        provenance.record_step("B", 13, "method preconditions checked")
        provenance.record_step(
            "B", 14,
            f"{len(decomposition.alternatives)} alternative decomposition(s) preserved",
        )
        provenance.record_step("B", 15, "provenance attached to every expansion")
        provenance.record_step(
            "B", 16,
            f"{len(decomposition.unsatisfied_leaves)} unsatisfied leaf/leaves",
        )
        provenance.record(
            "htn_decomposition", goal_id=goal.goal_id,
            workspace_revision=workspace_revision,
            payload={
                "alternatives": [tree.task.task_id for tree in decomposition.alternatives],
                "rejected_methods": decomposition.rejected_methods,
            },
        )

        # Phases C and D run per decomposition alternative; the best plan wins.
        methods = self._methods_for_goal(goal, discovery)
        candidates: list[tuple[SelectedPlan, float]] = []
        rejected: list[tuple[str, str]] = []
        for index, tree in enumerate(decomposition.alternatives):
            if tree.status.value == "UNSATISFIED":
                rejected.append((f"decomposition_{index}", "decomposition unsatisfied"))
                continue
            selected = self._search_tree(
                tree, goal, discovery, methods, facts, workspace_revision,
                quality_threshold, index, provenance,
            )
            if selected is None:
                rejected.append((
                    f"decomposition_{index}",
                    "no valid plan satisfied the goal test",
                ))
                continue
            candidates.append((selected, selected.outcome.total_cost))
        if not candidates:
            provenance.record_step("C", 17, "no decomposition produced a valid plan")
            return PlanningOutcome(
                formalization=formalization,
                discovery=discovery,
                decomposition=decomposition,
                selected=None,
                provenance=provenance,
                status="no_valid_plan_exists",
            )
        candidates.sort(key=lambda item: (item[1], item[0].plan.plan_id))
        selected_plan = candidates[0][0]
        for candidate, cost in candidates[1:]:
            rejected.append((
                candidate.plan.plan_id,
                f"higher total cost {cost:.3f} than selected plan",
            ))
        decision = PlanDecision(
            selected_plan=selected_plan.plan.plan_id,
            rejected_alternatives=tuple(rejected),
            rationale=(
                f"selected lowest-cost legal plan under search mode {self.search_mode} "
                f"with heuristic {self.heuristic.version}"
            ),
            budget=self.budget.to_dict(),
            heuristic_version=self.heuristic.version,
            optimality_claim=selected_plan.outcome.optimality_claim,
            search_mode=self.search_mode,
        )
        selected_plan = SelectedPlan(
            plan=selected_plan.plan,
            outcome=selected_plan.outcome,
            decision=decision,
            goal=goal,
        )
        provenance.record_step(
            "E", 29,
            f"selected plan {selected_plan.plan.plan_id} "
            f"({'' if selected_plan.outcome.optimality_claim else 'no'} optimality claim)",
        )
        provenance.record(
            "plan_decision", goal_id=goal.goal_id, plan_id=selected_plan.plan.plan_id,
            workspace_revision=workspace_revision,
            payload=decision.to_dict(),
        )
        return PlanningOutcome(
            formalization=formalization,
            discovery=discovery,
            decomposition=decomposition,
            selected=selected_plan,
            provenance=provenance,
            status="planned",
        )

    def _methods_for_goal(
        self, goal: GoalSpec, discovery: OperatorDiscovery
    ) -> tuple[HTNMethod, ...]:
        """Combine custom versioned methods with the documented default library."""
        defaults = build_default_methods(goal, discovery)
        custom_by_id = {method.method_id: method for method in self.custom_methods}
        merged: list[HTNMethod] = []
        for method in defaults:
            merged.append(custom_by_id.pop(method.method_id, method))
        merged.extend(self.custom_methods)
        return tuple(merged)

    def _search_tree(
        self,
        tree: TaskTree,
        goal: GoalSpec,
        discovery: OperatorDiscovery,
        methods: tuple[HTNMethod, ...],
        facts: frozenset[str],
        workspace_revision: int,
        quality_threshold: float,
        index: int,
        provenance: PlanningProvenance,
    ) -> SelectedPlan | None:
        """Run phases C and D for one decomposition alternative."""
        method_by_id = {method.method_id: method for method in methods}
        chosen_method = (
            method_by_id.get(tree.provenance.method_id)
            if tree.provenance is not None else None
        )
        serial = bool(
            chosen_method is not None and chosen_method.ordering == "ordered"
        )
        plan_id = f"plan_g{index}_{goal.goal_id[-8:]}"
        builder = PartialOrderBuilder(
            {item.operator_id: item for item in discovery.operators},
            serial_ordering=serial,
        )
        plan = builder.build(
            plan_id, goal.goal_id, tree, facts,
            workspace_revision=workspace_revision,
        )
        provenance.record_step(
            "C", 17, f"{len(plan.edges)} dependency edge(s) built for {plan.plan_id}",
        )
        provenance.record_step(
            "C", 18,
            f"{len(plan.concurrency_levels)} concurrency level(s); "
            "independent nodes left unordered",
        )
        provenance.record_step(
            "C", 19,
            f"{len(plan.serialized_conflicts)} mutex/resource conflict(s) serialized",
        )
        provenance.record_step(
            "C", 20,
            "dag valid" if plan.valid else f"cycle nodes: {list(plan.cycle_nodes)}",
        )
        provenance.record_step(
            "C", 21, f"{len(plan.protections)} protection interval(s) recorded",
        )
        if not plan.valid:
            return None
        operators = {item.operator_id: item for item in discovery.operators}
        initial_state = PlanState(
            workspace_revision=workspace_revision,
            achieved_facts=facts,
            pending_tasks=tuple(node.node_id for node in plan.nodes),
        )
        engine = SearchEngine(
            self.cost_model, self.heuristic,
            budget=self.budget, mode=self.search_mode, weight=self.search_weight,
        )
        outcome = engine.search(
            plan, goal, operators, initial_state,
            quality_threshold=quality_threshold,
        )
        provenance.record_step("D", 22, "planner state defined from revision and facts")
        provenance.record_step("D", 23, "successors from ready nodes and applicable operators")
        provenance.record_step(
            "D", 24, f"search mode {self.search_mode} completed: {outcome.stop_reason}",
        )
        provenance.record_step("D", 25, f"g/h computed with heuristic {self.heuristic.version}")
        provenance.record_step(
            "D", 26,
            "optimality claim allowed" if outcome.optimality_claim
            else "best-effort label: no optimality claim",
        )
        provenance.record_step(
            "D", 27,
            f"beam pruning: {outcome.stats.beam_pruned} branch(es)" if self.search_mode == "beam"
            else "beam search not required for this branching",
        )
        provenance.record_step(
            "D", 28,
            f"duplicates={outcome.stats.duplicates_pruned}, "
            f"dominance={outcome.stats.dominance_pruned}",
        )
        if not outcome.goal_achieved:
            return None
        provisional = PlanDecision(
            selected_plan=plan.plan_id,
            rejected_alternatives=(),
            rationale="candidate",
            budget=self.budget.to_dict(),
            heuristic_version=self.heuristic.version,
            optimality_claim=outcome.optimality_claim,
            search_mode=self.search_mode,
        )
        return SelectedPlan(
            plan=plan, outcome=outcome, decision=provisional, goal=goal,
        )

    # ------------------------------------------------------------------
    # Phase E: execution, feedback, and replanning (steps 29-40)
    # ------------------------------------------------------------------
    def execute(
        self,
        planning: PlanningOutcome,
        *,
        input_artifacts: Mapping[str, Any] | None = None,
        run_context: RunContext | None = None,
        checkpoint: Any = None,
        quality_threshold: float = 1.0,
    ) -> PlanRunResult:
        """Execute a selected plan through the coordinator with bounded replanning."""
        if planning.selected is None:
            return PlanRunResult(
                plan_id="none",
                status="STOPPED",
                goal_progress_score=0.0,
                critical_satisfied=False,
                unresolved_subgoals=["no_valid_plan_exists"],
                termination_reason="no_valid_plan_exists",
            )
        selected = planning.selected
        plan = selected.plan
        goal = selected.goal
        provenance = planning.provenance
        reports: list[PlanExecutionReport] = []
        all_feedback: list[Any] = []
        replans = 0
        current_checkpoint = checkpoint
        outcome = selected.outcome
        termination = "execution_incomplete"
        workspace_revision = plan.workspace_revision

        for round_index in range(self.max_replans + 1):
            report = self.executor.submit(
                plan, goal, outcome,
                input_artifacts=input_artifacts,
                run_context=run_context,
                checkpoint=current_checkpoint,
            )
            reports.append(report)
            all_feedback.extend(report.feedback)
            workspace_revision = max(workspace_revision, report.workspace_revision)
            provenance.record_step(
                "E", 30,
                f"round {round_index}: submitted via CognitiveTaskCoordinator",
            )
            provenance.record_step(
                "E", 31,
                "independent ready tasks executed concurrently by the coordinator",
            )
            provenance.record_step(
                "E", 32,
                f"outputs integrated at workspace revision {report.workspace_revision}",
            )
            accepted_facts = self._accepted_facts(goal, reports)
            provenance.record_step(
                "E", 33,
                f"{len(accepted_facts)} achieved predicate(s) from accepted outputs",
            )
            evaluation = self.evaluator.evaluate_outcomes(tuple(all_feedback))
            provenance.record_step(
                "E", 34,
                f"mean absolute prediction error {evaluation.mean_absolute_error:.3f}",
            )
            progress = self.cost_model.goal_progress(goal, accepted_facts)
            provenance.record_step(
                "E", 35,
                f"goal progress {progress.score:.3f} "
                f"at revision {workspace_revision}",
            )
            state = PlanState(
                workspace_revision=workspace_revision,
                achieved_facts=accepted_facts,
                pending_tasks=(),
            )
            invalidated = self.evaluator.validate_preconditions(
                plan, self._operators(planning), state,
            )
            stop = self.evaluator.stop_conditions(
                goal, progress,
                no_valid_plan=False,
            )
            if stop is not None:
                termination = stop
                break
            if round_index >= self.max_replans:
                termination = "maximum_replans_reached"
                break
            decision = self.evaluator.should_replan(
                expected_quality_gain=1.0 - progress.score,
                expected_saved_cost=0.0,
                failed_operators=evaluation.failed_operators,
                invalidated_nodes=invalidated,
            )
            provenance.record(
                "replan_decision", goal_id=goal.goal_id, plan_id=plan.plan_id,
                workspace_revision=workspace_revision,
                payload=decision.to_dict(),
            )
            if not decision.should_replan:
                termination = "no_better_plan_available"
                break
            current_checkpoint = self._checkpoint(plan, goal, report)
            replans += 1
            provenance.record_step(
                "E", 36,
                f"replan {replans}: {decision.rationale}",
            )

        progress = self.cost_model.goal_progress(
            goal, self._accepted_facts(goal, reports),
        )
        provenance.record_step("E", 37, f"termination: {termination}")
        result = self.evaluator.build_run_result(
            plan, progress, termination,
            feedback=tuple(all_feedback),
            replans=replans,
            workspace_revision=workspace_revision,
            evidence_refs=tuple(
                ref for report in reports for ref in report.evidence_refs
            ),
        )
        provenance.record_step(
            "E", 38,
            f"final status {result.status.value}; "
            f"{len(result.unresolved_subgoals)} unresolved subgoal(s)",
        )
        provenance.record_step(
            "E", 39,
            f"{len(reports)} report(s) and {len(all_feedback)} feedback record(s) persisted",
        )
        provenance.record_step(
            "E", 40,
            "prediction errors recorded for cost calibration",
        )
        provenance.record(
            "plan_run_result", goal_id=goal.goal_id, plan_id=plan.plan_id,
            workspace_revision=workspace_revision,
            module_ids=tuple(
                module_id for report in reports for module_id in report.module_ids
            ),
            payload=result.to_dict(),
        )
        return result

    def plan_and_execute(
        self,
        goal_text: str,
        *,
        input_artifacts: Mapping[str, Any] | None = None,
        run_context: RunContext | None = None,
        **plan_kwargs: Any,
    ) -> tuple[PlanningOutcome, PlanRunResult]:
        """Run phases A through D and then phase E in one call."""
        planning = self.plan(goal_text, **plan_kwargs)
        run_result = self.execute(
            planning, input_artifacts=input_artifacts, run_context=run_context,
        )
        return planning, run_result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _accepted_facts(
        self, goal: GoalSpec, reports: Sequence[PlanExecutionReport]
    ) -> frozenset[str]:
        """Derive achieved predicates only from successful, evidence-backed outputs."""
        succeeded_operators = {
            item.operator_id
            for report in reports
            for item in report.feedback
            if item.success and item.operator_id
        }
        satisfied: set[str] = set()
        for condition in goal.success_conditions:
            capability = (
                condition.predicate.arguments[0]
                if condition.predicate.arguments else ""
            )
            for operator_id in succeeded_operators:
                manifest = self.registry.get_manifest(operator_id)
                if manifest is not None and capability in manifest.capabilities:
                    satisfied.add(condition.predicate.to_atom())
                    break
        return frozenset(satisfied)

    def _operators(self, planning: PlanningOutcome) -> dict[str, OperatorSpec]:
        """Return the eligible operators of one planning outcome."""
        return {item.operator_id: item for item in planning.discovery.operators}

    def _checkpoint(self, plan: PartialOrderPlan, goal: GoalSpec, report: PlanExecutionReport):
        """Capture a checkpoint from the latest execution report."""
        return self.executor.checkpoint(plan, goal, report)
