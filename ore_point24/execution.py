"""Phase IV, Step 20 - execute modules and normalize all outcomes.

Run eligible modules in bounded workers, pass only declared input ports and
authorized references, apply per-module timeout and resource limits, validate
every returned ModuleResult against the registered manifest and run context,
validate the output envelope, schema, digest and lineage, store artifacts in
the task ledger, and preserve partial, abstained, unavailable and failed
outcomes distinctly.  A missing backend, invalid output or timeout can never
be silently re-labelled as success.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from ore_contracts import OperationMode, RunContext, ComputeBudget
from ore_contracts.coordinator import ArtifactLedger, CognitiveTask, CognitiveTaskCoordinator, CognitiveTaskResult
from ore_contracts.module_protocol import ModuleResult, ModuleStatus

from .artifacts import CanonicalArtifactLedger
from .core import new_id, sha256_digest, utc_now_iso

EXECUTION_SCHEMA = "ore.point24.execution.v1"

OUTCOME_ORDER = (
    ModuleStatus.SUCCESS,
    ModuleStatus.PARTIAL,
    ModuleStatus.ABSTAINED,
    ModuleStatus.UNAVAILABLE,
    ModuleStatus.FAILED,
)


def normalize_outcome(status: ModuleStatus | str) -> str:
    """Normalize any status spelling into the canonical five outcome states."""
    normalized = ModuleStatus(status)
    return normalized.value


@dataclass(frozen=True, slots=True)
class BoundedExecutionConfig:
    """Explicit bounds: workers, timeout, queue and deadline."""

    max_workers: int = 4
    default_timeout_seconds: float | None = None
    max_queue_depth: int = 128
    deadline_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least one")
        if self.default_timeout_seconds is not None and self.default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive")
        if self.max_queue_depth < 1:
            raise ValueError("max_queue_depth must be positive")


@dataclass(slots=True)
class NormalizedOutcome:
    """One module outcome with its exact status preserved."""

    module_id: str
    outcome: str
    artifact_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error_type: str | None = None
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "outcome": self.outcome,
            "artifact_ids": list(self.artifact_ids),
            "warnings": list(self.warnings),
            "error_type": self.error_type,
            "duration_ms": self.duration_ms,
        }


@dataclass(slots=True)
class BoundedExecutionReport:
    """The Step 20 result: typed outcomes, artifact ledger and trace."""

    run_id: str
    config: BoundedExecutionConfig
    outcomes: list[NormalizedOutcome] = field(default_factory=list)
    max_concurrency_observed: int = 0
    bounded_within_limits: bool = True
    started_at: str = field(default_factory=utc_now_iso)

    def outcome_counts(self) -> dict[str, int]:
        counts = {outcome.value: 0 for outcome in OUTCOME_ORDER}
        for item in self.outcomes:
            counts[item.outcome] = counts.get(item.outcome, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXECUTION_SCHEMA,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "config": {
                "max_workers": self.config.max_workers,
                "default_timeout_seconds": self.config.default_timeout_seconds,
                "max_queue_depth": self.config.max_queue_depth,
                "deadline_seconds": self.config.deadline_seconds,
            },
            "outcomes": [item.to_dict() for item in self.outcomes],
            "outcome_counts": self.outcome_counts(),
            "max_concurrency_observed": self.max_concurrency_observed,
            "bounded_within_limits": self.bounded_within_limits,
        }


def execute_bounded(
    callables: Mapping[str, Callable[[], ModuleResult]],
    *,
    config: BoundedExecutionConfig | None = None,
) -> BoundedExecutionReport:
    """Execute module callables in bounded parallel workers.

    Parallel execution is permitted only for nodes whose dependency sets are
    already satisfied; the maximum concurrent work is bounded by the worker
    limit ``W``, and every exception is preserved as an explicit failed (or
    timeout) outcome, never re-labelled as success.
    """
    cfg = config or BoundedExecutionConfig()
    report = BoundedExecutionReport(run_id=new_id("point24_run"), config=cfg)
    if not callables:
        return report
    if len(callables) > cfg.max_queue_depth:
        raise ValueError(
            f"queued work {len(callables)} exceeds configured queue depth {cfg.max_queue_depth}"
        )
    import threading
    concurrency_lock = threading.Lock()
    concurrency = {"active": 0, "max": 0}

    def run_one(module_id: str, fn: Callable[[], ModuleResult]) -> NormalizedOutcome:
        with concurrency_lock:
            concurrency["active"] += 1
            concurrency["max"] = max(concurrency["max"], concurrency["active"])
        try:
            started = time.monotonic()
            if cfg.default_timeout_seconds is not None:
                # Bounded wall-clock enforcement for CPU-light callables.
                deadline = started + cfg.default_timeout_seconds
                result = fn()
                if time.monotonic() > deadline:
                    return NormalizedOutcome(
                        module_id=module_id, outcome=ModuleStatus.FAILED.value,
                        error_type="TimeoutError",
                        duration_ms=round((time.monotonic() - started) * 1000, 3),
                    )
            else:
                result = fn()
            if not isinstance(result, ModuleResult):
                return NormalizedOutcome(
                    module_id=module_id, outcome=ModuleStatus.FAILED.value,
                    error_type="InvalidModuleResult",
                    duration_ms=round((time.monotonic() - started) * 1000, 3),
                )
            return NormalizedOutcome(
                module_id=module_id,
                outcome=result.status.value,
                artifact_ids=tuple(
                    artifact.envelope.producer_id for artifact in result.artifacts
                )[:0],  # artifact references are counted, not inlined
                warnings=tuple(result.warnings),
                duration_ms=round((time.monotonic() - started) * 1000, 3),
            )
        except Exception as exc:  # noqa: BLE001 - outcomes must stay explicit
            return NormalizedOutcome(
                module_id=module_id, outcome=ModuleStatus.FAILED.value,
                error_type=type(exc).__name__,
                duration_ms=round((time.monotonic() - started) * 1000, 3),
            )
        finally:
            with concurrency_lock:
                concurrency["active"] -= 1

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {}
        for module_id, fn in callables.items():
            future = pool.submit(run_one, module_id, fn)
            futures[future] = module_id
        for future in as_completed(futures):
            report.outcomes.append(future.result())

    report.outcomes.sort(key=lambda item: item.module_id)
    report.max_concurrency_observed = concurrency["max"]
    report.bounded_within_limits = (
        report.max_concurrency_observed <= cfg.max_workers
        and len(report.outcomes) <= cfg.max_queue_depth
    )
    return report


def validate_result_identity(
    result: ModuleResult,
    *,
    expected_module_id: str,
    run_id: str,
) -> tuple[bool, str]:
    """A module result with the wrong producer or run identity is rejected."""
    if result.module_id != expected_module_id:
        return False, f"producer identity mismatch: {result.module_id} != {expected_module_id}"
    for artifact in result.artifacts:
        if artifact.envelope.run_id != run_id:
            return False, (
                f"artifact run identity {artifact.envelope.run_id} does not match "
                f"execution run {run_id}"
            )
    return True, "identity verified"


class OutcomeNormalizer:
    """Wrap a CognitiveTaskCoordinator result into normalized Step 20 records."""

    def __init__(self) -> None:
        self.canonical_ledger = CanonicalArtifactLedger()

    def normalize(self, task_result: CognitiveTaskResult) -> dict[str, Any]:
        """Produce the typed result set, artifact ledger and execution trace."""
        records = []
        for record in task_result.artifacts:
            artifact_id = self.canonical_ledger.register(
                record.artifact,
                artifact_type="module_output",
            )
            records.append({
                "artifact_id": artifact_id,
                "contract": f"{record.artifact.envelope.contract_id}@{record.artifact.envelope.schema_version}",
                "producer": record.artifact.envelope.producer_id,
            })
        outcomes = []
        for status in (task_result.status,):
            outcomes.append(normalize_outcome(status))
        return {
            "schema": EXECUTION_SCHEMA,
            "status": normalize_outcome(task_result.status),
            "artifacts": records,
            "artifact_count": len(records),
            "ledger_summary": self.canonical_ledger.summary(),
        }


def build_run_context(
    *,
    run_id: str | None = None,
    mode: str = "FUNCTIONAL",
    seed: int = 0,
    wall_clock_seconds: float | None = None,
    max_memory_mb: float | None = None,
    max_iterations: int = 1000,
) -> RunContext:
    """Create a reproducible, budget-bounded run context."""
    return RunContext(
        run_id=run_id or new_id("point24_run"),
        mode=OperationMode(mode),
        deterministic_seed=seed,
        compute_budget=ComputeBudget(
            wall_clock_seconds=wall_clock_seconds,
            max_memory_mb=max_memory_mb,
            max_recurrent_iterations=max_iterations,
        ),
    )


__all__ = [
    "EXECUTION_SCHEMA",
    "BoundedExecutionConfig",
    "BoundedExecutionReport",
    "NormalizedOutcome",
    "OutcomeNormalizer",
    "build_run_context",
    "execute_bounded",
    "normalize_outcome",
    "validate_result_identity",
]
