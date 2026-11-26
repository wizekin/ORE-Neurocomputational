"""MACL end-to-end pipeline: run the coordination loop and trace all nine stages.

The pipeline wires the supervisor, registry, matching, adjudication, aggregation,
ledger, peer coordination, and the isolated swarm optimizers into one reproducible
run (spec Sections 3 and 8). Every stage emits a structured operation record with
functional and biological outputs so the full lineage can be persisted through the
existing ORE SQLite observer.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ore_workspace import GoalEvaluator, SharedWorkspace

from .aco import ACOConfig, ACOResult, AntColonyOptimizer, OperationGraph, dependency_feasible
from .agents import LogicalClock, build_default_agents
from .pso import ParticleSwarmOptimizer, PSOConfig, PSOResult, resource_policy_objective
from .registry import registry_as_dicts
from .reliability import reliability_report
from .supervisor import GoalRequest, GoalRunReport, MACLConfig, MACLSupervisor


@dataclass
class MACLRunConfig:
    """Configure one reproducible MACL pipeline run."""

    goal_id: str = "macl-goal"
    goal: str = "understand_sentence"
    input_text: str = "the reporter ignored the scientist and argued about the study"
    max_rounds: int = 12
    goal_budget: float = 200.0
    require_verification_for_findings: bool = True
    enable_peer_topology: bool = False
    run_isolated_swarm_experiments: bool = True
    pso_iterations: int = 12
    aco_iterations: int = 10
    seed: int = 11
    component_scores: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        """Validate pipeline configuration bounds."""
        for name in ("goal_id", "goal", "input_text"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least one")
        if self.goal_budget <= 0:
            raise ValueError("goal_budget must be positive")
        if self.pso_iterations < 1 or self.aco_iterations < 1:
            raise ValueError("swarm experiment iterations must be at least one")


@dataclass
class MACLRunResult:
    """Capture the complete pipeline outcome with per-stage lineage."""

    config: MACLRunConfig
    goal_report: GoalRunReport | None = None
    operation_trace: list[dict[str, Any]] = field(default_factory=list)
    pso_result: PSOResult | None = None
    aco_result: ACOResult | None = None
    biological_snapshot: dict[str, float] = field(default_factory=dict)
    reliability_report: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    @property
    def stage_ids(self) -> tuple[str, ...]:
        """Return the executed stage identifiers in source order."""
        return tuple(item["operation_id"] for item in self.operation_trace)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible run record."""
        return {
            "goal_id": self.config.goal_id,
            "goal": self.config.goal,
            "input_text": self.config.input_text,
            "stages_executed": list(self.stage_ids),
            "operation_trace": list(self.operation_trace),
            "goal_report": self.goal_report.to_dict() if self.goal_report else None,
            "pso_result": self.pso_result.to_dict() if self.pso_result else None,
            "aco_result": self.aco_result.to_dict() if self.aco_result else None,
            "biological_snapshot": dict(self.biological_snapshot),
            "reliability_report": dict(self.reliability_report),
            "elapsed_seconds": self.elapsed_seconds,
        }


def _stage_record(
    stage: Mapping[str, Any],
    *,
    status: str,
    functional_output: Any,
    biological_output: Any,
    notes: str = "",
) -> dict[str, Any]:
    """Build one structured operation record for a MACL stage."""
    return {
        "operation_id": stage["stage_id"],
        "operation": stage["name"],
        "source_steps": list(stage["source_steps"]),
        "status": status,
        "mathematical_form": stage["mathematical_form"],
        "biological_reference": stage["biological_reference"],
        "evidence_level": stage["evidence_level"],
        "functional_output": functional_output,
        "biological_output": biological_output,
        "notes": notes,
    }


def run_multi_agent_coordination(
    config: MACLRunConfig | None = None,
    *,
    workspace: SharedWorkspace | None = None,
    supervisor: MACLSupervisor | None = None,
    clock: Any = None,
) -> MACLRunResult:
    """Run the full nine-stage MACL loop and return its traced result."""
    resolved = config or MACLRunConfig()
    started = time.perf_counter()
    result = MACLRunResult(config=resolved)
    stages = registry_as_dicts()

    # Stage MACL-1: agent lifecycle and registry (Steps 1-7).
    stage = stages[0]
    own_supervisor = supervisor is None
    if supervisor is None:
        effective_workspace = workspace or SharedWorkspace(
            current_input=resolved.input_text,
            current_goal=resolved.goal,
        )
        supervisor = MACLSupervisor(
            effective_workspace,
            build_default_agents(clock=clock or LogicalClock()),
            config=MACLConfig(
                max_rounds=resolved.max_rounds,
                goal_budget=resolved.goal_budget,
                require_verification_for_findings=resolved.require_verification_for_findings,
                enable_peer_topology=resolved.enable_peer_topology,
            ),
            clock=clock or LogicalClock(),
            goal_evaluator=GoalEvaluator(),
        )
    registry_snapshot = supervisor.registry.snapshot()
    result.operation_trace.append(_stage_record(
        stage,
        status="completed" if registry_snapshot["agents"] else "failed",
        functional_output={
            "registered_agents": sorted(registry_snapshot["agents"]),
            "quarantined_agents": registry_snapshot["quarantined"],
            "lifecycle_states": {
                agent_id: item["state"]
                for agent_id, item in registry_snapshot["agents"].items()
            },
        },
        biological_output=supervisor.biology.snapshot(),
        notes="contracts registered, autonomy justified, least privilege granted",
    ))

    # Stage MACL-2: decomposition and allocation (Steps 8-16) via the run itself.
    stage = stages[1]
    request = GoalRequest(
        goal_id=resolved.goal_id,
        goal=resolved.goal,
        input_text=resolved.input_text,
        component_scores=dict(resolved.component_scores or {}),
    )
    goal_report = supervisor.run_goal(request)
    result.goal_report = goal_report
    tasks_planned = {
        task_id: {
            "owner": record.owner_agent_id,
            "outcome": record.final_outcome.value,
            "deterministic_fallback": record.used_deterministic_fallback,
        }
        for task_id, record in sorted(goal_report.task_records.items())
    }
    result.operation_trace.append(_stage_record(
        stage,
        status="completed" if tasks_planned else "failed",
        functional_output={
            "tasks": tasks_planned,
            "rounds": goal_report.rounds,
            "stop_reason": goal_report.stop_reason,
        },
        biological_output=goal_report.biology_trace[-1] if goal_report.biology_trace else {},
        notes="typed task DAG built, cycle-checked, matched, and assigned",
    ))

    # Stage MACL-3: supervisor/worker protocol (Steps 17-23).
    stage = stages[2]
    attempt_count = sum(
        len(record.attempts) for record in goal_report.task_records.values()
    )
    result.operation_trace.append(_stage_record(
        stage,
        status="completed" if attempt_count else "failed",
        functional_output={
            "attempts": attempt_count,
            "messages": goal_report.message_stats,
            "task_outcomes": {
                task_id: record.final_outcome.value
                for task_id, record in sorted(goal_report.task_records.items())
            },
        },
        biological_output=supervisor.biology.snapshot(),
        notes="bounded envelopes, accept/reject, progress, typed results",
    ))

    # Stage MACL-4: blackboard and shared evidence (Steps 24-30).
    stage = stages[3]
    workspace_state = supervisor.adapter.workspace.snapshot().state
    result.operation_trace.append(_stage_record(
        stage,
        status="completed" if goal_report.lineage.get("findings_accepted") else "partial",
        functional_output={
            "findings_accepted": goal_report.lineage.get("findings_accepted", 0),
            "workspace_revision": supervisor.adapter.revision,
            "lexical_tokens": len(workspace_state.get("lexical_hypotheses", {})),
            "syntactic_records": len(workspace_state.get("syntactic_hypotheses", ())),
            "semantic_records": len(workspace_state.get("semantic_hypotheses", ())),
            "unresolved_questions": len(workspace_state.get("unresolved_questions", ())),
        },
        biological_output=supervisor.biology.snapshot(),
        notes="versioned snapshots, keyed proposals, provenance, conflict preservation",
    ))

    # Stage MACL-5: selective peer-to-peer cooperation (Steps 31-36).
    stage = stages[4]
    peer_stats = supervisor.peer.stats()
    result.operation_trace.append(_stage_record(
        stage,
        status="completed",
        functional_output={
            "negotiations": peer_stats["negotiations"],
            "by_outcome": peer_stats["by_outcome"],
            "ownership": peer_stats["ownership"],
            "max_delegation_depth": peer_stats["max_delegation_depth"],
        },
        biological_output=supervisor.biology.snapshot(),
        notes="capability discovery without broadcast; bounded delegation depth",
    ))

    # Stage MACL-6: debate, critic, and verification (Steps 37-43).
    stage = stages[5]
    verifier_entries = [
        agent_id for agent_id in goal_report.ledger_summary.get("reliability", {})
        if "verifier" in agent_id.lower()
    ]
    verification_count = sum(
        1
        for record in goal_report.task_records.values()
        for attempt in record.attempts
        if attempt.adjudication and attempt.adjudication.get("verified_by")
    )
    result.operation_trace.append(_stage_record(
        stage,
        status="completed",
        functional_output={
            "debates": len(goal_report.debates),
            "dissent_preserved": len(goal_report.dissent),
            "independent_verifications": verification_count,
            "verifier_tracked": sorted(verifier_entries),
        },
        biological_output=supervisor.biology.snapshot(),
        notes="evidence adjudication; minority findings preserved, never majority vote",
    ))

    # Stage MACL-7: aggregation and integration (Steps 44-50).
    stage = stages[6]
    evaluation = goal_report.goal_evaluation
    result.operation_trace.append(_stage_record(
        stage,
        status="completed" if evaluation is not None else "failed",
        functional_output={
            "goal_outcome": evaluation.outcome.value if evaluation else None,
            "goal_score": evaluation.score if evaluation else None,
            "missing_requirements": list(evaluation.missing_requirements) if evaluation else [],
            "completed_tasks": len(goal_report.completed_task_ids),
            "conflicts_visible": len(goal_report.conflicts),
        },
        biological_output=supervisor.biology.snapshot(),
        notes="goal progress from accepted outcomes via the existing evaluator",
    ))

    # Stage MACL-8: recovery and performance learning (Steps 51-58).
    stage = stages[7]
    result.reliability_report = reliability_report(supervisor.ledger.records)
    result.operation_trace.append(_stage_record(
        stage,
        status="completed",
        functional_output={
            "ledger_entries": goal_report.ledger_summary.get("entries", 0),
            "reliability": {
                agent_id: {
                    "posterior_mean": item["posterior_mean"],
                    "samples": item["samples"],
                    "credible_interval": item["credible_interval"],
                }
                for agent_id, item in result.reliability_report.items()
            },
            "learning_reward": goal_report.ledger_summary.get("learning_reward", {}),
        },
        biological_output=supervisor.biology.snapshot(),
        notes="Beta-Binomial reliability with credible intervals and sample support",
    ))

    # Stage MACL-9: isolated MARL and swarm experiments (Steps 59-66).
    stage = stages[8]
    if resolved.run_isolated_swarm_experiments:
        pso_result, aco_result = run_isolated_swarm_experiments(
            pso_iterations=resolved.pso_iterations,
            aco_iterations=resolved.aco_iterations,
            seed=resolved.seed,
        )
        result.pso_result = pso_result
        result.aco_result = aco_result
        result.operation_trace.append(_stage_record(
            stage,
            status="completed",
            functional_output={
                "pso": pso_result.to_dict(),
                "aco": aco_result.to_dict(),
                "isolated": True,
                "production_control": False,
            },
            biological_output=supervisor.biology.snapshot(),
            notes="offline replay objectives only; optimizers propose, policies authorize",
        ))
    else:
        result.operation_trace.append(_stage_record(
            stage,
            status="skipped",
            functional_output={"isolated": True, "reason": "swarm experiments disabled"},
            biological_output=supervisor.biology.snapshot(),
            notes="Step 66: physical robotics and production control remain deferred",
        ))

    result.biological_snapshot = supervisor.biology.snapshot()
    result.elapsed_seconds = round(time.perf_counter() - started, 4)
    if own_supervisor:
        supervisor.close()
    return result


#: Reference resource-policy coefficients validated on fixed replay traffic.
REFERENCE_RESOURCE_POLICY = (0.55, 0.30, 0.15)

#: Reference operation ordering validated on a fixed knowledge-graph replay.
REFERENCE_OPERATION_PATH = ("retrieve", "analyze", "verify", "integrate")


def run_isolated_swarm_experiments(
    *,
    pso_iterations: int = 12,
    aco_iterations: int = 10,
    seed: int = 11,
) -> tuple[PSOResult, ACOResult]:
    """Run bounded, reproducible PSO and ACO experiments on replay objectives.

    Both optimizers are fully isolated from production control (Step 59): they
    evaluate fixed replay objectives, propose parameters or paths only, and can
    never alter permissions or bypass ORE validation (Step 65).
    """
    pso = ParticleSwarmOptimizer(
        PSOConfig(
            iterations=pso_iterations,
            seed=seed,
            num_particles=8,
            max_evaluations=max(64, pso_iterations * 8),
        ),
        lambda weights: resource_policy_objective(
            weights, reference_solution=REFERENCE_RESOURCE_POLICY,
        ),
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        constraint=lambda weights: abs(sum(weights) - 1.0) <= 0.25,
    )
    pso_result = pso.optimize()

    graph = OperationGraph(
        operations=("retrieve", "analyze", "verify", "integrate", "summarize"),
        dependencies={
            "analyze": ("retrieve",),
            "verify": ("analyze",),
            "integrate": ("verify",),
            "summarize": ("integrate",),
        },
        edge_cost={
            ("start", "retrieve"): 1.0,
            ("retrieve", "analyze"): 1.5,
            ("analyze", "verify"): 1.0,
            ("verify", "integrate"): 1.5,
            ("integrate", "summarize"): 1.0,
            ("analyze", "summarize"): 2.0,
            ("retrieve", "verify"): 2.5,
        },
        heuristic_desirability={
            ("start", "retrieve"): 2.0,
            ("retrieve", "analyze"): 1.5,
            ("analyze", "verify"): 2.0,
            ("verify", "integrate"): 1.8,
            ("integrate", "summarize"): 1.2,
            ("analyze", "summarize"): 0.5,
            ("retrieve", "verify"): 0.4,
        },
        start="start",
        terminal=None,
    )
    aco = AntColonyOptimizer(
        ACOConfig(iterations=aco_iterations, seed=seed, num_ants=6, max_path_length=6),
        graph,
        validator=lambda path: dependency_feasible(path, graph),
        budget=8.0,
    )
    aco_result = aco.optimize()
    return pso_result, aco_result


def register_annotations(
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, int]:
    """Index every MACL package class and callable docstring in the ORE SQLite catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    return sync_repository(
        LoggerDatabase(database_path), root,
        include_paths={"ore_macl"},
    )


def record_result_to_sqlite(
    result: MACLRunResult,
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist the nine-stage MACL operation trace and index all package annotations."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    database = LoggerDatabase(database_path)
    catalogue = sync_repository(
        database, root, include_paths={"ore_macl"}
    )
    run_id = database.create_run(
        script_path="ore_macl/point11",
        module_name="ore_macl.pipeline",
        entry_kind="multi_agent_swarm_coordination",
        working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id,
                sequence_no=sequence,
                event_kind="operation",
                source_path="ore_macl/pipeline.py",
                elapsed_ms=0.0,
                metadata={
                    "operation_id": item.get("operation_id"),
                    "operation": item.get("operation"),
                    "status": item.get("status"),
                    "source_steps": item.get("source_steps"),
                    "mathematical_form": item.get("mathematical_form"),
                    "functional_output": item.get("functional_output"),
                    "biological_output": item.get("biological_output"),
                },
            )
        database.finish_run(
            run_id, "completed",
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
    except Exception as exc:
        database.finish_run(
            run_id, "failed",
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            error_summary=type(exc).__name__ + ": " + str(exc),
        )
        raise
    return {
        "run_id": run_id,
        "annotation_catalogue": catalogue,
        "database_summary": database.summary(),
    }
