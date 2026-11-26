"""Sandboxed strategy execution and measurement (Phase B, steps 11-13, 16).

The sandbox (step 11) is the isolated worker model: candidate strategies are *data
interpreted under capability grants*, never executed source. Only a read-only feature
view of the task reaches the genome (step 14: candidates cannot see labels, scoring,
the clock, task selection or validation data). Enforced budgets (step 12): interpreter
steps, action count, output size, memory and latency. Invalid behavior detection
(step 16): malformed genomes, prohibited actions, resource abuse and nontermination
are rejected and recorded, never silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..domain.actions import PROHIBITED_ACTIONS, validate_action
from ..domain.evaluation_record import TaskOutcomeRecord
from ..domain.genome import canonical_json
from .benchmark_manifest import TaskCase
from .task_simulators import CoordinatorSimulator

#: Capability grants handed to candidate strategies (step 11: explicit grants).
GRANTED_CAPABILITIES: frozenset[str] = frozenset({
    "read_task_features", "emit_approved_actions", "bounded_evaluation_steps",
})


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    """Enforced per-task execution budgets (step 12)."""

    max_steps_per_task: int = 512
    max_actions_per_task: int = 24
    max_output_bytes: int = 8192
    max_memory_mb: float = 128.0
    max_latency_ms: float = 2000.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_steps_per_task": self.max_steps_per_task,
            "max_actions_per_task": self.max_actions_per_task,
            "max_output_bytes": self.max_output_bytes,
            "max_memory_mb": self.max_memory_mb,
            "max_latency_ms": self.max_latency_ms,
        }


@dataclass(frozen=True, slots=True)
class ExecutionTrace:
    """Structured execution trace for one task (step 13)."""

    task_id: str
    steps: int
    actions: tuple[str, ...]
    outputs: dict[str, Any]
    exceptions: tuple[str, ...]
    resource_measurements: dict[str, float]
    terminated: bool = False
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "steps": self.steps,
            "actions": list(self.actions), "outputs": dict(self.outputs),
            "exceptions": list(self.exceptions),
            "resource_measurements": dict(self.resource_measurements),
            "terminated": self.terminated, "notes": list(self.notes),
        }


class SandboxedStrategyExecutor:
    """Execute one candidate genome on one task inside the sandbox envelope."""

    def __init__(
        self,
        *,
        simulator: CoordinatorSimulator | None = None,
        limits: SandboxLimits | None = None,
    ) -> None:
        self.simulator = simulator or CoordinatorSimulator()
        self.limits = limits or SandboxLimits()

    def granted_capabilities(self) -> frozenset[str]:
        """Explicit capability grants for candidate strategies (step 11)."""
        return GRANTED_CAPABILITIES

    def execute(self, genome: Any, task: TaskCase, rng: Any) -> tuple[TaskOutcomeRecord, ExecutionTrace]:
        """Run one genome on one task and produce the raw observation + trace.

        The genome receives only ``task.features()`` (a plain dict copy); it never
        sees the evaluator, labels, the clock or task selection (step 14).
        """
        exceptions: list[str] = []
        invalid_reasons: list[str] = []
        notes: list[str] = []

        # Malformed output detection (step 16): genome must be schema-valid.
        genome_errors = genome.validate()
        if genome_errors:
            invalid_reasons.append(f"malformed genome: {genome_errors[0]}")
            result = _invalid_result(task, genome_errors[0], 0)
            trace = ExecutionTrace(
                task_id=task.task_id, steps=0, actions=(),
                outputs={"genome_kind": genome.kind},
                exceptions=tuple(genre_error for genre_error in genome_errors[:3]),
                resource_measurements={}, notes=tuple(notes),
            )
            return _to_record(task, 0, rng.seed if hasattr(rng, "seed") else 0, result, trace), trace

        # Interpretation with the capability-restricted view (step 14).
        result = self.simulator.simulate(genome, task, rng)
        steps = min(result.steps, self.limits.max_steps_per_task + 1)
        if result.terminated or result.steps > self.limits.max_steps_per_task:
            # Nontermination: terminated and recorded (step 16; acceptance test 9).
            notes.append("nonterminating candidate terminated at the step budget")
            invalid_reasons.append("step budget exceeded")
            result = _replace_invalid(result, "step budget exceeded")

        # Prohibited / unknown action screening (step 8, 16; acceptance test 2).
        for action_name in result.actions:
            if action_name in PROHIBITED_ACTIONS or not _is_approved(action_name):
                reason = f"prohibited action attempted: {action_name!r}"
                invalid_reasons.append(reason)
                exceptions.append(reason)
        if len(result.actions) > self.limits.max_actions_per_task:
            invalid_reasons.append(
                f"action budget exceeded ({len(result.actions)} > {self.limits.max_actions_per_task})"
            )

        # Output size and resource abuse checks (steps 12, 16).
        output_bytes = len(canonical_json(
            {"actions": list(result.actions), "modules": list(result.active_modules)}
        ).encode("utf-8"))
        if output_bytes > self.limits.max_output_bytes:
            invalid_reasons.append(f"output budget exceeded ({output_bytes} bytes)")
        if result.memory_mb > self.limits.max_memory_mb:
            invalid_reasons.append(
                f"memory budget exceeded ({result.memory_mb:.1f}MB > {self.limits.max_memory_mb:.1f}MB)"
            )
        if result.latency_ms > self.limits.max_latency_ms:
            invalid_reasons.append(
                f"latency hard cap exceeded ({result.latency_ms:.1f}ms > {self.limits.max_latency_ms:.1f}ms)"
            )
        if invalid_reasons:
            result = _replace_invalid(result, invalid_reasons[0])
        notes.extend(reason for reason in invalid_reasons if reason not in notes)

        trace = ExecutionTrace(
            task_id=task.task_id,
            steps=steps,
            actions=tuple(result.actions),
            outputs={
                "genome_kind": genome.kind,
                "active_modules": list(result.active_modules),
                "output_bytes": output_bytes,
                "outcome": result.outcome,
                "quality": result.quality,
            },
            exceptions=tuple(exceptions),
            resource_measurements={
                "latency_ms": result.latency_ms, "memory_mb": result.memory_mb,
                "cpu_ms": result.cpu_ms, "cost": result.cost,
            },
            terminated=result.terminated,
            notes=tuple(notes),
        )
        trial_seed = getattr(rng, "seed", 0)
        record = _to_record(task, trial_seed, trial_seed, result, trace)
        return record, trace


def _is_approved(action_name: str) -> bool:
    from ..domain.actions import is_approved_action
    return is_approved_action(action_name)


def _invalid_result(task: TaskCase, reason: str, steps: int) -> Any:
    from .task_simulators import SimulationResult
    return SimulationResult(
        task_id=task.task_id, actions=(), active_modules=(),
        quality=0.0, outcome="INVALID", latency_ms=0.0, memory_mb=0.0,
        cpu_ms=0.0, cost=0.0, safety_violations=0,
        steps=steps, terminated=True, invalid_reasons=(reason,),
        evidence={"active_modules": 0},
    )


def _replace_invalid(result: Any, reason: str) -> Any:
    from dataclasses import replace
    return replace(
        result, outcome="INVALID", quality=0.0,
        invalid_reasons=tuple(dict.fromkeys((*result.invalid_reasons, reason))),
    )


def _to_record(
    task: TaskCase,
    trial: int,
    seed: int,
    result: Any,
    trace: ExecutionTrace,
) -> TaskOutcomeRecord:
    evidence: dict[str, Any] = dict(result.evidence)
    evidence["trace"] = trace.as_dict()
    # The recorded quality is the *measured* verified quality (partial credit,
    # section 8), not the discrete outcome credit: only failed/invalid/abstaining
    # executions collapse to zero/small credit so raw evidence stays informative
    # (step 17: never store only a fitness scalar or a coarse credit).
    if result.outcome in ("INVALID", "TIMEOUT"):
        recorded_quality = 0.0
    elif result.outcome == "ABSTAIN":
        recorded_quality = result.outcome_credit()
    else:
        recorded_quality = float(min(max(result.quality, 0.0), 1.0))
    return TaskOutcomeRecord(
        task_id=task.task_id,
        trial=trial,
        seed=seed,
        outcome=result.outcome,
        quality=recorded_quality,
        latency_ms=result.latency_ms,
        memory_mb=result.memory_mb,
        cpu_ms=result.cpu_ms,
        cost=result.cost,
        safety_violations=result.safety_violations,
        actions=tuple(result.actions),
        steps=result.steps,
        terminated=result.terminated,
        evidence=evidence,
    )


__all__ = [
    "ExecutionTrace", "GRANTED_CAPABILITIES", "SandboxLimits", "SandboxedStrategyExecutor",
]
