"""Strategy evolution service: the end-to-end coordinator (section 4 contract).

``StrategyEvolutionCoordinator.evolve(objective, budget) -> EvolutionRunReport``
runs the bounded pipeline of section 11's final recommendation:

Bounded Evolutionary Optimization -> Sandboxed Evaluation -> MAP-Elites Archive ->
Independent Verification -> Versioned Validated Strategy Registry -> Contextual
Coordinator Selection -> Live Monitoring and Rollback.

Phase A (steps 1-10) narrows one objective, captures the immutable baseline rollback
target, versions the evaluation protocol and seals the task split. Phase B (steps
11-17) builds the sandboxed evaluator under an admitted budget. Phase C (steps 18-30)
runs the genetic search with hard-constraint gates. Phase D (steps 31-38) applies to
genetic-programming targets through the typed grammar and declarative policy IR.
Phase E (steps 39-44) fills the MAP-Elites archive and hands off only validated
candidates. Phase F verifies independently (static, semantic, held-out), promotes
through the strict lifecycle, registers the artifact and optionally runs a canary
rollout with rollback.

A successful experiment is evidence about a bounded benchmark, not proof of general
intelligence (section 11). Evolution failures never interrupt ordinary ORE tasks
(acceptance test 36): ``evolve`` always returns a report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from ..archive.map_elites import MapElitesArchive, MapElitesSearch, QDSearchConfig
from ..domain.behavior_descriptor import BehaviorDescriptor, default_calibration
from ..domain.evaluation_record import (
    HardConstraints, NormalizationScales, ScalarFitness,
)
from ..domain.genome import (
    ParameterGenome, default_search_space, content_digest,
)
from ..domain.strategy import LineageRecord, StrategyCandidate
from ..evaluation.benchmark_manifest import (
    BenchmarkManifest, ScoringConfig, build_benchmark,
)
from ..evaluation.evaluator import StrategyEvaluator
from ..evaluation.heldout_validation import (
    HeldoutValidator, ValidationEnvelope,
)
from ..evaluation.sandbox import SandboxedStrategyExecutor
from ..observability.events import EvolutionEventRecorder
from ..observability.lineage import LineageTracker
from ..registry.artifact_store import StrategyArtifact, StrategyManifest
from ..registry.strategy_registry import ValidatedStrategyRegistry
from ..registry.versioning import RollbackTarget
from ..search.genetic_optimizer import GeneticOptimizer, GeneticSearchConfig
from ..search.genetic_programming import (
    ActionNode, CombineActionNode, CompareNode, ConstantNode, FeatureNode,
    IfActionNode, PolicyIRGenome, TypedGrammar, WhenActionNode,
)
from ..verification.promotion_policy import ApprovalRecord
from ..verification.regression_gate import RegressionGate
from ..verification.semantic_rule_check import SemanticRuleCheck
from ..verification.static_policy_check import StaticPolicyCheck
from .coordinator_adapter import (
    MODULE_SELECTION_TARGET, CoordinatorAdapter,
    ControlledConfigurationInterface, CoordinatorParameterTarget,
    baseline_configuration, genome_from_configuration,
)
from .resource_budget import AdmissionController, ResourceBudget
from .rollout_controller import RolloutController, canary_plan_from_tasks


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True, slots=True)
class EvolutionObjective:
    """One narrow, measurable optimization objective (steps 1-2)."""

    name: str
    target: CoordinatorParameterTarget
    constraints: HardConstraints = field(default_factory=HardConstraints)
    fitness: ScalarFitness = field(default_factory=ScalarFitness)
    practical_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.target.genome_kind not in ("parameters", "policy_table", "policy_ir"):
            raise ValueError(
                f"unknown genome kind {self.target.genome_kind!r} for objective"
            )

    @classmethod
    def module_selection(cls) -> "EvolutionObjective":
        """Priority 1 objective: coordinator parameters (section 10)."""
        return cls(
            name="module_selection_weights",
            target=MODULE_SELECTION_TARGET,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "target": self.target.as_dict(),
            "constraints": self.constraints.as_dict(),
            "fitness": self.fitness.as_dict(),
            "practical_threshold": self.practical_threshold,
        }


@dataclass(frozen=True, slots=True)
class EvolutionBudgetSpec:
    """Declared run budget (step 12: population, generations, total evaluations)."""

    total_evaluations: int = 400
    generation_cap: int = 18
    population_cap: int = 16
    ga_generations: int = 16
    qd_generations: int = 2
    seed: int = 20261122

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_evaluations": self.total_evaluations,
            "generation_cap": self.generation_cap,
            "population_cap": self.population_cap,
            "ga_generations": self.ga_generations,
            "qd_generations": self.qd_generations,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class EvolutionRunReport:
    """Complete record of one evolution run (sections 7-8 evidence requirements)."""

    run_id: str
    objective: Mapping[str, Any]
    status: str  # "completed" | "failed"
    error: str = ""
    started_at_utc: str = ""
    finished_at_utc: str = ""
    benchmark: Mapping[str, Any] = field(default_factory=dict)
    baseline: Mapping[str, Any] = field(default_factory=dict)
    budget: Mapping[str, Any] = field(default_factory=dict)
    genetic_search: Mapping[str, Any] = field(default_factory=dict)
    quality_diversity: Mapping[str, Any] = field(default_factory=dict)
    best: Mapping[str, Any] = field(default_factory=dict)
    verification: Mapping[str, Any] = field(default_factory=dict)
    registry: Mapping[str, Any] = field(default_factory=dict)
    canary: Mapping[str, Any] = field(default_factory=dict)
    events_summary: Mapping[str, Any] = field(default_factory=dict)
    lineage_summary: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "objective": dict(self.objective),
            "status": self.status, "error": self.error,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "benchmark": dict(self.benchmark), "baseline": dict(self.baseline),
            "budget": dict(self.budget), "genetic_search": dict(self.genetic_search),
            "quality_diversity": dict(self.quality_diversity),
            "best": dict(self.best), "verification": dict(self.verification),
            "registry": dict(self.registry), "canary": dict(self.canary),
            "events_summary": dict(self.events_summary),
            "lineage_summary": dict(self.lineage_summary),
            "notes": list(self.notes),
        }


#: Incumbent per-family module pairs of the deterministic baseline (step 3) — shared
#: by the policy-table and policy-IR baselines.
_INCUMBENT_FAMILY_MODULES: tuple[tuple[str, str], ...] = (
    ("ore_lexical_access", "ore_vwfa"),
    ("ore_semantic", "ore_semantic_composition"),
    ("ore_syntax", "ore_working_memory"),
    ("ore_reading", "ore_vwfa"),
    ("ore_discourse", "ore_semantic_composition"),
    ("ore_hippocampal_encoding", "ore_sleep_consolidation"),
)


def _baseline_table() -> Any:
    """The deterministic incumbent routing policy (step 3: current deterministic policy).

    Each family activates its two incumbent modules; safety-critical cells escalate
    verification; resource-pressured cells shed the secondary module. This is prior
    knowledge of the *incumbent coordinator*, not a learned candidate — evolution must
    beat it under the same gates.
    """
    from ..domain.genome import PolicyTableGenome, table_context_key
    incumbent: dict[str, tuple[str, ...]] = {
        family: (primary, secondary)
        for family, (primary, secondary) in zip(
            ("lexical_lookup", "semantic_composition", "syntax_parse",
             "reading", "discourse", "consolidation"),
            _INCUMBENT_FAMILY_MODULES,
        )
    }
    rules: dict[tuple[str, ...], frozenset] = {}
    for family, modules in incumbent.items():
        for difficulty in ("low", "medium", "high"):
            for safety in ("no", "yes"):
                for resource in ("normal", "pressured"):
                    key = table_context_key(
                        family, difficulty, safety == "yes", resource,
                    )
                    chosen = modules if resource == "normal" else modules[:1]
                    actions = {
                        _action("activate_module", {"module": module})
                        for module in chosen
                    }
                    if safety == "yes":
                        actions.add(_action("escalate_verification", {}))
                    if difficulty == "high" and family in (
                        "semantic_composition", "reading", "discourse",
                    ):
                        actions.add(_action("request_retrieval", {"depth": 1}))
                    rules[key] = frozenset(actions)
    return PolicyTableGenome(rules)


def _baseline_genome_for(objective: EvolutionObjective, grammar: TypedGrammar | None) -> Any:
    """Deterministic baseline genome per target kind (step 3)."""
    kind = objective.target.genome_kind
    if kind == "parameters":
        space = default_search_space()
        return ParameterGenome(
            {definition.name: definition.default_value() for definition in space},
            space,
        )
    if kind == "policy_table":
        return _baseline_table()
    grammar = grammar or TypedGrammar()
    # Deterministic incumbent policy IR (step 3): the same per-family incumbent
    # modules as the policy-table baseline, with verification escalation on
    # safety-critical tasks.
    family_modules = _INCUMBENT_FAMILY_MODULES
    chain: Any = CombineActionNode(
        ActionNode(_action("activate_module", {"module": family_modules[5][0]})),
        ActionNode(_action("activate_module", {"module": family_modules[5][1]})),
    )
    for index in range(4, -1, -1):
        primary, secondary = family_modules[index]
        chain = IfActionNode(
            CompareNode(
                "lt",
                (FeatureNode("task_family_index"), ConstantNode(index + 0.5)),
            ),
            CombineActionNode(
                ActionNode(_action("activate_module", {"module": primary})),
                ActionNode(_action("activate_module", {"module": secondary})),
            ),
            chain,
        )
    safety = FeatureNode("safety_critical")
    escalate = ActionNode(_action("escalate_verification", {}))
    return PolicyIRGenome(
        CombineActionNode(
            WhenActionNode(CompareNode("gt", (safety, ConstantNode(0.5))), escalate),
            chain,
        ),
        grammar=grammar,
    )


def _action(name: str, params: Mapping[str, Any]):
    from ..domain.actions import ActionCall
    return ActionCall(name, params)


class StrategyEvolutionCoordinator:
    """The bounded evolution service (section 4 contract; section 11 pipeline)."""

    def __init__(
        self,
        *,
        benchmark: BenchmarkManifest | None = None,
        event_recorder: EvolutionEventRecorder | None = None,
        repository: Any | None = None,
        workspace_publisher: Any | None = None,
        registry: ValidatedStrategyRegistry | None = None,
        grammar: TypedGrammar | None = None,
        auto_approve: bool = True,
        run_canary: bool = True,
        trials_per_task: int = 3,
        calibrator: Any | None = None,
        regression_policy: Any | None = None,
    ) -> None:
        self.benchmark = benchmark
        self.event_recorder = event_recorder or EvolutionEventRecorder()
        self.repository = repository
        self.workspace_publisher = workspace_publisher
        self.registry = registry or ValidatedStrategyRegistry()
        self.grammar = grammar
        self.auto_approve = auto_approve
        self.run_canary = run_canary
        self.trials_per_task = trials_per_task
        self.calibrator = calibrator or default_calibration()
        self.static_check = StaticPolicyCheck()
        self.semantic_check = SemanticRuleCheck()
        self.regression_gate = RegressionGate(regression_policy) if regression_policy else RegressionGate()

    # --------------------------------------------------------------------- evolve
    def evolve(
        self,
        objective: EvolutionObjective,
        budget_spec: EvolutionBudgetSpec | None = None,
    ) -> EvolutionRunReport:
        """Run the bounded pipeline; always returns a report (acceptance test 36)."""
        budget_spec = budget_spec or EvolutionBudgetSpec()
        run_id = f"ev-{content_digest({'objective': objective.as_dict(), 'seed': budget_spec.seed, 'started': _utc_now()})[:16]}"
        started = _utc_now()
        if self.repository is not None:
            self.repository.save_run(run_id, objective.name, started_at_utc=started)
        self.event_recorder.emit(
            "run_started", run_id,
            {"objective": objective.name, "budget": budget_spec.as_dict()},
        )
        if self.workspace_publisher is not None:
            self.workspace_publisher.publish_experiment_request(
                run_id, {"objective": objective.as_dict(), "budget": budget_spec.as_dict()},
            )
        state: dict[str, Any] = {
            "run_id": run_id, "objective": objective.as_dict(),
            "status": "completed", "error": "",
        }
        try:
            self._evolve_phases(objective, budget_spec, run_id, state)
        except Exception as error:  # acceptance test 36: never interrupt ORE tasks
            state["status"] = "failed"
            state["error"] = f"{type(error).__name__}: {error}"
            self.event_recorder.emit("run_failed", run_id, {"error": state["error"]})
            if self.workspace_publisher is not None:
                self.workspace_publisher.publish_issue(
                    f"{run_id}:failure",
                    {"run_id": run_id, "error": state["error"]},
                )
        finished = _utc_now()
        if self.repository is not None:
            self.repository.finish_run(
                run_id, state["status"], finished_at_utc=finished,
            )
        if state["status"] == "completed":
            self.event_recorder.emit("run_completed", run_id, state.get("best", {}))
            self.event_recorder.finish("completed")
        else:
            self.event_recorder.finish("failed")
        return EvolutionRunReport(
            run_id=run_id,
            objective=objective.as_dict(),
            status=state["status"],
            error=state.get("error", ""),
            started_at_utc=started,
            finished_at_utc=finished,
            benchmark=state.get("benchmark", {}),
            baseline=state.get("baseline", {}),
            budget=state.get("budget", {}),
            genetic_search=state.get("genetic_search", {}),
            quality_diversity=state.get("quality_diversity", {}),
            best=state.get("best", {}),
            verification=state.get("verification", {}),
            registry=state.get("registry", {}),
            canary=state.get("canary", {}),
            events_summary={
                "kinds_emitted": list(self.event_recorder.kinds()),
                "events_for_run": len(self.event_recorder.by_run(run_id)),
                "logger_run_id": self.event_recorder.logger_run_id,
            },
            lineage_summary=state.get("lineage", {}),
            notes=tuple(state.get("notes", ())),
        )

    # ------------------------------------------------------------------- phases
    def _evolve_phases(
        self,
        objective: EvolutionObjective,
        budget_spec: EvolutionBudgetSpec,
        run_id: str,
        state: dict[str, Any],
    ) -> None:
        # ---- Phase A: bounded search and evaluation definition (steps 1-10) ----
        if objective.target.genome_kind == "policy_ir" and self.grammar is None:
            from ..search.genetic_programming import DEFAULT_GRAMMAR
            self.grammar = DEFAULT_GRAMMAR
        self.event_recorder.emit(
            "objective_narrowed", run_id,
            {"target": objective.target.as_dict(), "behavior": objective.target.measurable_behavior},
        )
        benchmark = self.benchmark or build_benchmark(
            benchmark_id=f"ore_{objective.name}",
            seed=budget_spec.seed,
            scoring=ScoringConfig(
                fitness_config=objective.fitness.as_dict(),
                constraints_config=objective.constraints.as_dict(),
            ),
        )
        self.benchmark = benchmark
        state["benchmark"] = {
            "benchmark_id": benchmark.benchmark_id,
            "version": benchmark.version,
            "protocol_digest": benchmark.protocol_digest(),
            "split": benchmark.split_manifest(),
            "task_counts": {
                "train": len(benchmark.training_tasks()),
                "validation": len(benchmark.validation_tasks()),
                "test_sealed": len(benchmark.test_task_digests()),
            },
        }
        self.event_recorder.emit(
            "benchmark_versioned", run_id,
            {"version": benchmark.version, "protocol_digest": benchmark.protocol_digest()},
        )
        self.event_recorder.emit(
            "split_declared", run_id, {"policy": benchmark.split_policy},
        )

        evaluator = StrategyEvaluator(
            benchmark,
            sandbox=SandboxedStrategyExecutor(),
            constraints=objective.constraints,
            fitness=objective.fitness,
            trials_per_task=self.trials_per_task,
        )

        # Immutable baseline with rollback target (step 3).
        baseline_genome = _baseline_genome_for(objective, self.grammar)
        baseline = StrategyCandidate.from_genome(
            baseline_genome,
            lineage=LineageRecord(operation="initial", generation=0, run_id=run_id),
            run_id=run_id,
        )
        baseline_record = evaluator.evaluate_candidate(
            baseline, split="validation",
            seeds=self._seed_set(budget_spec.seed, 101), budget=None,
        )
        rollback_target = RollbackTarget(
            strategy_id=None,
            content_hash=baseline.content_hash,
            configuration=baseline_configuration().as_dict(),
            benchmark_version=benchmark.version,
            benchmark_results=baseline_record.metrics.as_dict(),
            source_version="ore_evolution_intelligence-1.0.0",
        )
        self.registry.versioning.pin_rollback_target(rollback_target)
        state["baseline"] = {
            "candidate_id": baseline.candidate_id,
            "content_hash": baseline.content_hash,
            "metrics": baseline_record.metrics.as_dict(),
            "feasible": baseline_record.gate.feasible,
            "rollback_target_digest": rollback_target.digest(),
        }
        if self.repository is not None:
            # Immutable run metadata, lineage, raw metrics and decisions (section 5:
            # SQLite / ore_logger row of the integration map).
            self.repository.save_candidate(baseline, run_id=run_id)
            self.repository.save_evaluation(baseline_record, run_id=run_id)
        self.event_recorder.emit(
            "baseline_captured", run_id, state["baseline"],
        )
        if self.workspace_publisher is not None:
            self.workspace_publisher.publish_hypothesis(
                baseline.candidate_id,
                {"kind": "baseline", "metrics": baseline_record.metrics.as_dict()},
            )

        # ---- Phase B: sandboxed execution budget (steps 11-17) ----
        resource_budget = ResourceBudget(
            total_evaluations=budget_spec.total_evaluations,
            generation_cap=budget_spec.generation_cap,
            population_cap=budget_spec.population_cap,
        )
        admission = AdmissionController().decide(resource_budget)
        self.event_recorder.emit(
            "budget_granted", run_id,
            {"budget": resource_budget.as_dict(), "admission": admission.as_dict()},
        )
        state["budget"] = {
            "resource_budget": resource_budget.as_dict(),
            "admission": admission.as_dict(),
        }
        if not admission.allowed:
            state["status"] = "failed"
            state["error"] = "admission denied: production reserve protection"
            state["notes"] = ["CROS deferred the experiment to protect production workload"]
            return

        # ---- Phase C: evolutionary parameter optimization (steps 18-30) ----
        ga_config = GeneticSearchConfig(
            population_size=min(budget_spec.population_cap, 12),
            generation_limit=budget_spec.ga_generations,
            seed=budget_spec.seed,
        )
        ga = GeneticOptimizer(
            evaluator, config=ga_config,
            event_recorder=self.event_recorder, grammar=self.grammar,
        )
        ga_report = ga.search(
            baseline_genome, run_id=run_id, budget=resource_budget.evaluation_budget,
        )
        state["genetic_search"] = ga_report.as_dict()
        if self.repository is not None:
            for stats in ga_report.history:
                self.repository.save_event(
                    "generation_completed", run_id, stats.as_dict(),
                    occurred_at_utc=_utc_now(),
                )
        best_candidate = ga_report.best_candidate
        finalists: list[StrategyCandidate] = list(ga_report.finalists)
        if best_candidate is None:
            state["notes"] = ["no feasible candidate found; baseline retained"]
            state["best"] = {
                "candidate_id": baseline.candidate_id,
                "state": baseline.state, "is_baseline": True,
            }
            return

        # ---- Phase E: MAP-Elites archive (steps 39-44) ----
        archive = MapElitesArchive(self.calibrator)
        qd = MapElitesSearch(
            evaluator,
            config=QDSearchConfig(
                generation_limit=budget_spec.qd_generations,
                seed=budget_spec.seed + 1,
            ),
            grammar=self.grammar,
            event_recorder=self.event_recorder,
        )
        seed_candidates = [baseline, best_candidate, *finalists[:2]]
        qd_report = qd.run(
            archive, seed_candidates, run_id=run_id,
            budget=resource_budget.evaluation_budget,
        )
        state["quality_diversity"] = qd_report.as_dict()

        # Choose the candidate for verification. The GA best was selected on
        # *validation* evidence (independent of the training benchmark), so it is the
        # deployment finalist; the archive handoff provides the diverse validated
        # backups (step 44: only independently validated candidates go to approval).
        chosen = best_candidate
        if chosen is None:
            handoff = archive.handoff()
            chosen = handoff[0].candidate if handoff else None
        if chosen is None:
            state["notes"] = ["no feasible candidate found; baseline retained"]
            state["best"] = {
                "candidate_id": baseline.candidate_id,
                "state": baseline.state, "is_baseline": True,
            }
            return

        train_record = evaluator.evaluate_candidate(
            chosen, split="train", seeds=self._seed_set(budget_spec.seed, 7),
            budget=None,
        )
        chosen_validation = evaluator.evaluate_candidate(
            chosen, split="validation", seeds=self._seed_set(budget_spec.seed, 11),
            budget=None,
        )
        state["best"] = {
            "candidate_id": chosen.candidate_id,
            "content_hash": chosen.content_hash,
            "state": chosen.state,
            "version": chosen.version,
            "genome": chosen.genome.as_dict(),
            "train_metrics": train_record.metrics.as_dict(),
            "validation_metrics": chosen_validation.metrics.as_dict(),
            "feasible": chosen_validation.gate.feasible,
            "quality_delta_vs_baseline": (
                chosen_validation.metrics.quality - baseline_record.metrics.quality
            ),
            "lineage": chosen.lineage.as_dict(),
        }
        if self.repository is not None:
            self.repository.save_candidate(chosen, run_id=run_id)
            self.repository.save_evaluation(train_record, run_id=run_id)
            self.repository.save_evaluation(chosen_validation, run_id=run_id)

        # ---- Phase F: independent verification and registry (steps 29, 36-38) ----
        static_report = self.static_check.check(chosen.genome)
        semantic_report = self.semantic_check.check(chosen.genome)
        heldout_validator = HeldoutValidator(
            evaluator, practical_threshold=objective.practical_threshold,
        )
        heldout = heldout_validator.compare(
            chosen, baseline, seeds=self._seed_set(budget_spec.seed, 211),
        )
        state["verification"] = {
            "static_check": static_report.as_dict(),
            "semantic_check": semantic_report.as_dict(),
            "heldout": heldout.as_dict(),
        }
        self.event_recorder.emit(
            "verification_completed", run_id,
            {
                "static_ok": static_report.ok, "semantic_ok": semantic_report.ok,
                "heldout_improved": heldout.improved,
                "candidate_id": chosen.candidate_id,
            },
        )
        if self.workspace_publisher is not None:
            self.workspace_publisher.publish_verification(
                chosen.candidate_id,
                {
                    "static_ok": static_report.ok, "semantic_ok": semantic_report.ok,
                    "heldout_improved": heldout.improved,
                    "quality_delta": heldout.quality_delta,
                },
            )
        self._promote_and_register(
            objective, run_id, state, chosen, baseline, train_record,
            static_report, semantic_report, heldout, chosen_validation,
            evaluator, resource_budget, budget_spec,
        )

        # Lineage reconstructability (acceptance test 24).
        tracker = LineageTracker()
        tracker.record(baseline)
        tracker.record(chosen)
        for entry in archive.entries():
            tracker.record(entry.candidate)
        state["lineage"] = {
            **tracker.population_summary(),
            "chosen_ancestry_depth": tracker.ancestry_depth(chosen.candidate_id),
        }

    # ------------------------------------------------- promotion, registry, canary
    def _promote_and_register(
        self,
        objective: EvolutionObjective,
        run_id: str,
        state: dict[str, Any],
        chosen: StrategyCandidate,
        baseline: StrategyCandidate,
        train_record: Any,
        static_report: Any,
        semantic_report: Any,
        heldout: Any,
        chosen_validation: Any,
        evaluator: StrategyEvaluator,
        resource_budget: ResourceBudget,
        budget_spec: EvolutionBudgetSpec,
    ) -> None:
        # GENERATED -> SANDBOX_TESTED (evidence: completed sandboxed evaluation).
        sandboxed, decision1 = self.registry.promote(
            chosen, "SANDBOX_TESTED", {"evaluation": train_record},
        )
        self.event_recorder.emit(
            "promotion_decided", run_id,
            {"candidate_id": chosen.candidate_id, "decision": decision1.as_dict()},
        )
        # SANDBOX_TESTED -> VALIDATED (static + semantic + held-out improvement).
        validated, decision2 = self.registry.promote(
            sandboxed, "VALIDATED",
            {
                "static_check": static_report,
                "semantic_check": semantic_report,
                "heldout": heldout,
            },
        )
        state["verification"]["promotion_decisions"] = [
            decision1.as_dict(), decision2.as_dict(),
        ]
        if not decision2.granted:
            state["notes"] = list(state.get("notes", [])) + [
                "candidate did not reach VALIDATED: " + "; ".join(decision2.reasons),
            ]
            state["best"]["state"] = sandboxed.state
            return

        if not self.auto_approve:
            state["notes"] = list(state.get("notes", [])) + [
                "candidate is VALIDATED and awaits explicit coordinator approval",
            ]
            state["best"]["state"] = validated.state
            return

        # VALIDATED -> APPROVED: governance approval (Main Brain Coordinator).
        approval = ApprovalRecord(
            approver="main_brain_coordinator",
            justification=(
                f"held-out improvement {heldout.quality_delta:+.4f} over baseline on "
                f"benchmark {self.benchmark.benchmark_id} v{self.benchmark.version}"
            ),
        )
        approved, decision3 = self.registry.promote(
            validated, "APPROVED", {"approval": approval},
        )
        state["best"]["state"] = approved.state

        # Registry with manifest + verified hash (acceptance tests 23, 24, 28).
        manifest = StrategyManifest(
            benchmark_id=self.benchmark.benchmark_id,
            benchmark_version=self.benchmark.version,
            split_digest=content_digest(
                self.benchmark.split_manifest()
            ),
            evaluator_version=evaluator.benchmark.evaluator_version,
            metrics_summary=chosen_validation.metrics.as_dict(),
            run_id=run_id,
            created_at_utc=_utc_now(),
            source_version="ore_evolution_intelligence-1.0.0",
            fitness_config=objective.fitness.as_dict(),
        )
        artifact = StrategyArtifact.build(
            {
                "candidate": approved.as_dict(),
                "objective": objective.as_dict(),
            },
            manifest,
        )
        record, note = self.registry.register(
            artifact, approval, candidate=approved,
        )
        state["registry"] = {
            "record": record.as_dict(), "note": note,
            "production_eligible": [
                r.as_dict() for r in self.registry.production_eligible()
            ],
        }
        self.event_recorder.emit(
            "registry_registered", run_id,
            {"strategy_id": record.strategy_id, "note": note},
        )
        if self.workspace_publisher is not None:
            self.workspace_publisher.publish_result(
                run_id,
                {"record": record.as_dict(), "heldout": heldout.as_dict()},
            )

        if not self.run_canary:
            state["notes"] = list(state.get("notes", [])) + [
                "canary skipped by configuration; strategy remains APPROVED",
            ]
            return

        # APPROVED -> CANARY -> ACTIVE with rollback (section 11 pipeline).
        # Stratified canary subset: every other validation task, spread across
        # families, so canary metrics are not biased by family ordering.
        validation_pool = list(evaluator.benchmark.validation_tasks())
        canary_tasks = validation_pool[::2][: max(len(validation_pool) // 4, 2)]
        adapter = CoordinatorAdapter(
            ControlledConfigurationInterface(
                event_recorder=self.event_recorder, run_id=run_id,
            )
        )
        rollout = RolloutController(
            evaluator, registry=self.registry, adapter=adapter,
            regression_gate=self.regression_gate,
            event_recorder=self.event_recorder, run_id=run_id,
        )
        canary_candidate, _plan_decision = self.registry.promote(
            approved, "CANARY",
            {"canary_plan": canary_plan_from_tasks(canary_tasks)},
        )
        # The validation envelope is measured on the *same canary task slice* with the
        # validation seeds, so drift detection compares like with like (acceptance
        # tests 33-34) instead of mixing different task distributions.
        envelope_source = evaluator.evaluate_on_tasks(
            chosen, canary_tasks, split="canary",
            seeds=self._seed_set(budget_spec.seed, 11),
        )
        envelope = ValidationEnvelope.from_record(envelope_source)
        final_candidate, outcome = rollout.canary(
            canary_candidate, baseline, canary_tasks,
            envelope=envelope,
            seeds=self._seed_set(budget_spec.seed, 307),
            budget=None,
        )
        if outcome.promoted_to_active:
            active, _decision = self.registry.promote(
                final_candidate, "ACTIVE",
                {"canary_decision": _canary_decision_from(outcome)},
            )
            state["best"]["state"] = active.state
            state["canary"] = outcome.as_dict()
            state["canary"]["final_state"] = active.state
        else:
            state["best"]["state"] = final_candidate.state
            state["canary"] = outcome.as_dict()
            state["canary"]["final_state"] = final_candidate.state
            state["notes"] = list(state.get("notes", [])) + [
                "canary regression triggered rollback to the baseline configuration",
            ]

    @staticmethod
    def _seed_set(base_seed: int, offset: int) -> tuple[int, ...]:
        return tuple(base_seed + offset + i for i in range(3))


def _canary_decision_from(outcome: Any) -> Any:
    from .rollout_controller import CanaryDecision
    return CanaryDecision(
        approved=outcome.promoted_to_active,
        reasons=outcome.reasons,
        drift=outcome.drift_report,
        regression=outcome.regression_decision,
    )


__all__ = [
    "EvolutionBudgetSpec", "EvolutionObjective", "EvolutionRunReport",
    "StrategyEvolutionCoordinator",
]
