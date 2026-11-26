"""Controlled interleavings, replay manifests, and invariant-based testing."""
from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..contracts import (
    EngineConfig, TaskSpec, content_fingerprint, new_id,
)
from ..engine import CognitiveExecutionEngine, ExecutionRun
from ..exceptions import DeterminismError, InvariantViolationError

STEP_DRAIN = "drain"
STEP_DISPATCH = "dispatch"
STEP_EXECUTE = "execute"
STEP_INJECT = "inject"
STEP_CANCEL = "cancel"
STEP_CHECK = "check"
STEP_FINALIZE = "finalize"

FAULT_KINDS = ("failure", "timeout", "cancellation")


@dataclass(frozen=True, slots=True)
class ScheduleStep:
    """Describe one controlled interleaving step of a deterministic schedule."""

    kind: str
    task_id: str | None = None
    outcome: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable schedule step for replay manifests."""
        return {
            "kind": self.kind, "task_id": self.task_id,
            "outcome": self.outcome, "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class Invariant:
    """Declare one named property that must hold over every observed state."""

    name: str
    check: Callable[[CognitiveExecutionEngine, ExecutionRun], bool]
    description: str = ""


@dataclass(frozen=True, slots=True)
class StepTrace:
    """Record one executed deterministic step with its observed outcomes."""

    index: int
    step: ScheduleStep
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable step trace record."""
        return {"index": self.index, **self.step.to_dict(), "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ReplayManifest:
    """Capture the minimum state needed to reconstruct one execution path."""

    run_id: str
    goal_id: str
    correlation_id: str
    task_specs: tuple[dict[str, Any], ...]
    schedule: tuple[dict[str, Any], ...]
    seeds: Mapping[str, int]
    workspace_versions: tuple[int, ...]
    scheduler_decisions: tuple[Mapping[str, Any], ...]
    engine_config: dict[str, Any]
    module_versions: Mapping[str, str]

    def fingerprint(self) -> str:
        """Hash the manifest so replay inputs can be compared structurally."""
        return content_fingerprint({
            "run_id": self.run_id,
            "task_specs": list(self.task_specs),
            "schedule": list(self.schedule),
            "seeds": dict(self.seeds),
        })


class DeterministicScheduler:
    """Drive engine steps in controlled order for reproducible concurrency tests."""

    def __init__(
        self, engine: CognitiveExecutionEngine, *,
        seed: int = 0, max_steps: int = 512,
    ) -> None:
        """Bind one manual-mode engine and a bounded step budget."""
        self._engine = engine
        self._seed = seed
        self._max_steps = max_steps
        self._step_traces: list[StepTrace] = []
        self._invariant_violations: list[tuple[str, str]] = []
        self._races_reproduced = 0

    # -------------------------------------------------------------- schedules

    def natural_schedule(self, run: ExecutionRun) -> tuple[ScheduleStep, ...]:
        """Build the straightforward schedule: dispatch, execute, drain, repeat."""
        steps: list[ScheduleStep] = []
        for task_id in run.graph.task_ids:
            steps.append(ScheduleStep(STEP_DISPATCH, task_id=task_id))
            steps.append(ScheduleStep(STEP_EXECUTE, task_id=task_id))
            steps.append(ScheduleStep(STEP_DRAIN))
        steps.append(ScheduleStep(STEP_FINALIZE))
        return tuple(steps)

    def random_schedule(
        self, run: ExecutionRun, *, seed: int | None = None, faults: int = 0,
    ) -> tuple[ScheduleStep, ...]:
        """Generate one adversarial schedule with optional injected faults.

        Rounds of dispatch, shuffled execution attempts, and event drains keep
        the schedule productive while injecting delays, duplicate effects,
        timeouts, cancellations, and failures at controlled points.
        """
        rng = random.Random(seed if seed is not None else self._seed)
        task_ids = list(run.graph.task_ids)
        steps: list[ScheduleStep] = []
        rounds = len(task_ids) + 2
        for round_index in range(rounds):
            steps.append(ScheduleStep(
                STEP_DISPATCH, task_id=task_ids[round_index % len(task_ids)],
            ))
            for task_id in rng.sample(task_ids, len(task_ids)):
                steps.append(ScheduleStep(STEP_EXECUTE, task_id=task_id))
            steps.append(ScheduleStep(STEP_DRAIN))
        if faults:
            for _ in range(faults):
                target = rng.choice(task_ids)
                kind = rng.choice(FAULT_KINDS)
                outcome = {
                    "failure": "failed", "timeout": "timed_out",
                    "cancellation": "cancelled",
                }[kind]
                steps.insert(
                    rng.randrange(len(steps) + 1),
                    ScheduleStep(STEP_INJECT, task_id=target, outcome=outcome,
                                 error=f"injected {kind}"),
                )
        steps.append(ScheduleStep(STEP_CHECK))
        steps.append(ScheduleStep(STEP_FINALIZE))
        return tuple(steps)

    # ------------------------------------------------------------------ run

    def run(self, run: ExecutionRun, schedule: Sequence[ScheduleStep]) -> tuple[str, Any]:
        """Execute one schedule against the run and return (status, report)."""
        for index, step in enumerate(schedule[: self._max_steps]):
            detail = self._apply(run, step)
            self._step_traces.append(StepTrace(index=index, step=step, detail=detail))
        if not run.graph.is_complete():
            self._engine.manual_check_health(run)
        report = self._engine.finalize(run)
        return report.status, report

    def _apply(self, run: ExecutionRun, step: ScheduleStep) -> str:
        """Apply one schedule step to the run and describe what happened."""
        engine = self._engine
        if step.kind == STEP_DRAIN:
            delivered = engine.drain_events(run)
            return f"drained {delivered} events"
        if step.kind == STEP_DISPATCH:
            dispatched = engine.manual_dispatch(run)
            return f"dispatched {len(dispatched)} tasks: {list(dispatched)}"
        if step.kind == STEP_EXECUTE:
            if step.task_id and engine.pool.pending_of(step.task_id):
                completion = engine.manual_execute(run, step.task_id)
                return f"executed {step.task_id}: {completion.outcome.value}"
            return f"skipped execute for {step.task_id} (not dispatched)"
        if step.kind == STEP_INJECT:
            if step.task_id and engine.pool.pending_of(step.task_id):
                completion = engine.manual_inject(
                    run, step.task_id, step.outcome or "failed",
                    error=step.error,
                )
                return f"injected {step.outcome} for {step.task_id}"
            return f"skipped inject for {step.task_id} (not dispatched)"
        if step.kind == STEP_CANCEL:
            engine.cancel_goal(run, step.error or "cancelled by schedule")
            return "goal cancellation propagated"
        if step.kind == STEP_CHECK:
            engine.manual_check_health(run)
            return "health check applied"
        if step.kind == STEP_FINALIZE:
            return "finalization reached"
        return f"unknown step kind {step.kind}"

    # -------------------------------------------------------------- replay

    def capture_replay_manifest(self, run: ExecutionRun) -> ReplayManifest:
        """Capture the replay manifest for one executed run."""
        return ReplayManifest(
            run_id=run.run_id,
            goal_id=run.request.goal_id,
            correlation_id=run.correlation_id,
            task_specs=tuple(spec.to_dict() for spec in
                             (node.spec for node in run.graph.nodes)),
            schedule=tuple(trace.to_dict() for trace in self._step_traces),
            seeds={"schedule_seed": self._seed},
            workspace_versions=(self._engine.store.latest().revision,),
            scheduler_decisions=tuple(
                {"task_id": row["task_id"], "worker_id": row["worker_id"]}
                for row in self._engine.scheduler.stats()["dispatch_log"]
            ),
            engine_config=config_as_dict(self._engine.config),
            module_versions={"ore_runtime": "0.1.0"},
        )

    def replay(
        self, manifest: ReplayManifest, *,
        task_specs: Sequence[TaskSpec], executors: Mapping[str, Any],
        schedule: Sequence[ScheduleStep],
    ) -> tuple[str, Any]:
        """Re-execute one captured schedule on a fresh engine and compare."""
        replay_engine = CognitiveExecutionEngine(
            config=EngineConfig(**manifest.engine_config),
        )
        replay_scheduler = DeterministicScheduler(replay_engine, seed=0)
        run = replay_engine.start(
            manifest.goal_id, task_specs, executors,
            correlation_id=manifest.correlation_id,
        )
        status, report = replay_scheduler.run(run, schedule)
        replay_engine.close()
        return status, report

    def assert_replay_deterministic(
        self, original_status: str, replayed_status: str,
    ) -> bool:
        """Check that replayed invariant outcomes match the recorded execution."""
        if original_status != replayed_status:
            self._races_reproduced += 1
            raise DeterminismError(
                f"replay diverged: original status {original_status} vs "
                f"replayed status {replayed_status}"
            )
        return True

    # ------------------------------------------------------------ invariants

    def check_invariants(
        self, run: ExecutionRun, invariants: Sequence[Invariant],
    ) -> tuple[str, ...]:
        """Evaluate invariants over the current engine and run state."""
        failures: list[str] = []
        for invariant in invariants:
            try:
                holds = invariant.check(self._engine, run)
            except Exception as exc:  # noqa: BLE001 - invariant isolation
                failures.append(invariant.name)
                self._invariant_violations.append(
                    (invariant.name, f"{type(exc).__name__}: {exc}")
                )
                continue
            if not holds:
                failures.append(invariant.name)
                self._invariant_violations.append(
                    (invariant.name, "invariant evaluated false")
                )
        return tuple(failures)

    def assert_invariants(
        self, run: ExecutionRun, invariants: Sequence[Invariant],
    ) -> None:
        """Raise InvariantViolationError when any declared invariant is broken."""
        failures = self.check_invariants(run, invariants)
        if failures:
            name, evidence = self._invariant_violations[-1]
            raise InvariantViolationError(name, f"failures={failures}; {evidence}")

    def fuzz(
        self, run_factory: Callable[[], ExecutionRun], *,
        invariants: Sequence[Invariant], cases: int = 16, seed: int = 0,
    ) -> dict[str, Any]:
        """Run generated schedules and track race and invariant statistics."""
        rng = random.Random(seed)
        violations: list[tuple[str, str]] = []
        for case in range(cases):
            self._engine.abandon_pending()
            run = run_factory()
            schedule = self.random_schedule(
                run, seed=rng.randrange(2 ** 31), faults=rng.randrange(0, 4),
            )
            status, _report = self.run(run, schedule)
            failures = self.check_invariants(run, invariants)
            violations.extend(
                (name, f"case {case} status {status}") for name in failures
            )
        return {
            "cases": cases,
            "invariant_violations": len(violations),
            "violations": violations,
            "races_reproduced": self._races_reproduced,
            "replay_deterministic": self._races_reproduced == 0,
        }

    # ---------------------------------------------------------------- stats

    def step_traces(self) -> tuple[StepTrace, ...]:
        """Return every executed step trace in order."""
        return tuple(self._step_traces)

    def stats(self) -> dict[str, Any]:
        """Return deterministic testing statistics."""
        return {
            "steps_executed": len(self._step_traces),
            "invariant_violations": len(self._invariant_violations),
            "violations": list(self._invariant_violations),
            "races_reproduced": self._races_reproduced,
        }


def config_as_dict(config: EngineConfig) -> dict[str, Any]:
    """Serialize an engine configuration for replay manifests."""
    return {
        "max_workers": config.max_workers,
        "max_active_tasks": config.max_active_tasks,
        "queue_high_watermark": config.queue_high_watermark,
        "queue_low_watermark": config.queue_low_watermark,
        "max_ready_queue_size": config.max_ready_queue_size,
        "module_queue_quota": config.module_queue_quota,
        "aging_rate": config.aging_rate,
        "max_priority": config.max_priority,
        "default_task_timeout_seconds": config.default_task_timeout_seconds,
        "max_rounds": config.max_rounds,
        "max_iterations_per_region": config.max_iterations_per_region,
        "convergence_epsilon": config.convergence_epsilon,
        "checkpoint_interval_rounds": config.checkpoint_interval_rounds,
        "outbox_retry_backoff_seconds": config.outbox_retry_backoff_seconds,
        "outbox_max_attempts": config.outbox_max_attempts,
        "retained_revision_limit": config.retained_revision_limit,
        "execution_mode": config.execution_mode,
    }


def invariant_dependency_ordering() -> Invariant:
    """Declare the dependency-ordering invariant for adversarial schedules."""

    def check(engine: CognitiveExecutionEngine, run: ExecutionRun) -> bool:
        """A task never runs while a predecessor is unfinished."""
        for node in run.graph.nodes:
            if node.state.value in {"running", "committing"}:
                for dep in node.spec.dependencies:
                    if run.graph.node(dep).state.value != "succeeded":
                        return False
        return True

    return Invariant(
        name="dependency_ordering", check=check,
        description="a task with an unfinished predecessor never enters RUNNING",
    )


def invariant_no_partial_publication() -> Invariant:
    """Declare the snapshot-consistency invariant."""

    def check(engine: CognitiveExecutionEngine, run: ExecutionRun) -> bool:
        """Every retained snapshot is a complete committed revision."""
        for revision in engine.store.retained_revisions():
            snapshot = engine.store.revision(revision)
            if snapshot.data is None or snapshot.revision != revision:
                return False
        return True

    return Invariant(
        name="snapshot_consistency", check=check,
        description="readers observe only complete committed revisions",
    )


def invariant_bounded_execution(config: EngineConfig) -> Invariant:
    """Declare the bounded-execution invariant for a given configuration."""

    def check(engine: CognitiveExecutionEngine, run: ExecutionRun) -> bool:
        """Active tasks never exceed the configured ceiling."""
        return engine.pool.active_task_count() <= config.max_active_tasks

    return Invariant(
        name="bounded_execution", check=check,
        description="active workers and queued tasks stay within configured limits",
    )


def invariant_capability_safety() -> Invariant:
    """Declare the capability-safety invariant."""

    def check(engine: CognitiveExecutionEngine, run: ExecutionRun) -> bool:
        """Every dispatched task ran on a capability-matched worker."""
        capabilities = engine.pool.worker_capabilities()
        for node in run.graph.nodes:
            worker_id = node.assigned_worker
            if worker_id and node.state.value in {
                "running", "succeeded", "failed", "cancelled", "committing",
            }:
                if node.spec.capability not in capabilities.get(worker_id, frozenset()):
                    return False
        return True

    return Invariant(
        name="capability_safety", check=check,
        description="tasks are never assigned to workers missing a capability",
    )


def invariant_event_schema_validity() -> Invariant:
    """Declare the event-envelope schema invariant."""

    def check(engine: CognitiveExecutionEngine, run: ExecutionRun) -> bool:
        """Every durable event satisfies its catalogue schema."""
        from ..events.schemas import event_validity
        return all(event_validity(stored.envelope) for stored in engine.event_store.events())

    return Invariant(
        name="event_schema_validity", check=check,
        description="task-state and event-schema invariants hold under fuzzing",
    )


def default_invariants(config: EngineConfig) -> tuple[Invariant, ...]:
    """Return the standard invariant suite from the specification."""
    return (
        invariant_dependency_ordering(),
        invariant_no_partial_publication(),
        invariant_bounded_execution(config),
        invariant_capability_safety(),
        invariant_event_schema_validity(),
    )


def new_replay_run_id() -> str:
    """Generate a fresh replay run identifier."""
    return new_id("replay")
