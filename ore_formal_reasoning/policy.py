"""Resource budgets, isolated workers, failure recovery, capability flags (Steps 47-48, 52).

External solvers run in bounded workers with CPU, memory, wall-clock, and
output limits; binaries are allowlisted and workers receive structured input
only — no arbitrary file or network access. Failures are classified, contained,
and reported as explicit non-success statuses; retries happen only for
transient operational failures under a bounded backoff, and never change the
correctness of a computed answer. Capabilities are released incrementally
behind flags promoted only after their acceptance suites pass (Step 52).
"""
from __future__ import annotations

import multiprocessing
import os
try:
    import resource
except ImportError:  # Windows does not provide POSIX setrlimit support.
    resource = None  # type: ignore[assignment]
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .contracts import ResourceBudget as Budget
from .exceptions import (
    GovernanceViolationError,
)

WORKER_PROTOCOL_VERSION = "frce-worker-1.0.0"

_ALLOWED_BINARIES = frozenset({"lean", "z3"})


class BinaryAllowlist:
    """Only allowlisted solver binaries may be executed (Step 47)."""

    def __init__(self, allowed: Sequence[str] = tuple(_ALLOWED_BINARIES)) -> None:
        self._allowed = frozenset(allowed)

    def check(self, binary: str) -> str:
        """Return the resolved binary path or raise."""
        if binary not in self._allowed:
            raise GovernanceViolationError(
                f"binary {binary!r} is not on the solver allowlist"
            )
        resolved = shutil.which(binary)
        if resolved is None:
            raise GovernanceViolationError(f"binary {binary!r} is not installed")
        return resolved

    @property
    def allowed(self) -> tuple[str, ...]:
        """Return the allowlisted binary names."""
        return tuple(sorted(self._allowed))


class FileAccessPolicy:
    """Solver workers may touch only the pinned working directory (Step 47)."""

    def __init__(self, working_root: str | Path | None = None) -> None:
        self._working_root = (
            Path(working_root).resolve()
            if working_root is not None
            else Path(os.environ.get("TMPDIR", "/tmp")).resolve()
        )

    def check_path_access(self, path: str | Path, mode: str) -> None:
        """Raise when a path escapes the pinned working root."""
        candidate = Path(path).resolve()
        if not candidate.is_relative_to(self._working_root):
            raise GovernanceViolationError(
                f"worker file access outside the pinned working root is blocked: "
                f"{candidate}"
            )
        if mode not in {"r", "w", "rw"}:
            raise ValueError("mode must be r, w, or rw")

    @property
    def working_root(self) -> Path:
        """Return the pinned working root."""
        return self._working_root


class NetworkPolicy:
    """Solver workers get no network access (Step 47: no unnecessary network)."""

    def check_network_access(self, host: str) -> None:
        """Always refuses: the FRCE solver workers make no network calls."""
        raise GovernanceViolationError(
            f"solver workers are not permitted network access (requested {host!r})"
        )


# ---------------------------------------------------------------------------
# Bounded worker (Step 47)
# ---------------------------------------------------------------------------


def _child_entry(
    function: Callable[..., Any],
    args: Sequence[Any],
    budget: Budget,
    result_path: str,
) -> None:
    """Child-process entry: apply resource limits, run, report a typed outcome.

    The result is written to a pre-created file rather than a queue: file
    handoff cannot deadlock on forked feeder threads, which matters when the
    parent process (for example a test runner) is multi-threaded.
    """
    import json

    try:
        if resource is not None:
            memory_bytes = max(64, budget.memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            cpu_limit = max(1, budget.cpu_seconds)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    except (OSError, ValueError):
        pass  # limits are best-effort in constrained environments
    try:
        payload = function(*args)
        try:
            encoded = json.dumps({"kind": "ok", "payload": payload})
        except (TypeError, ValueError):
            encoded = json.dumps({
                "kind": "ok",
                "payload": repr(payload),
                "note": "payload was not JSON-serializable and is recorded via repr",
            })
    except BaseException as error:  # noqa: BLE001 - contained in the child
        encoded = json.dumps({
            "kind": "exception",
            "detail": f"{type(error).__name__}: {error}",
        })
    try:
        with open(result_path, "w", encoding="utf-8") as handle:
            handle.write(encoded)
    except OSError:
        pass  # the parent will classify a missing result file as a crash


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    """The typed outcome of one bounded worker run (Step 47)."""

    ok: bool
    payload: Any = None
    failure_class: str | None = None   # timeout | crash | malformed_output | output_limit
    detail: str = ""
    wall_clock_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.ok and self.failure_class is None:
            raise ValueError("failed worker outcomes must carry a failure class")


class BoundedWorker:
    """Run solver work in a bounded child process (Step 47).

    When process isolation is disabled the function runs in-process under a
    wall-clock measurement; hard CPU/memory limits then apply only to isolated
    runs, which the capability flags record explicitly.
    """

    def __init__(self, *, process_isolation: bool = False) -> None:
        self.process_isolation = process_isolation

    def run(
        self,
        function: Callable[..., Any],
        args: Sequence[Any],
        budget: Budget,
    ) -> WorkerOutcome:
        """Execute under the budget; classify timeout/crash/malformed outcomes."""
        start = time.perf_counter()
        if not self.process_isolation:
            try:
                payload = function(*args)
            except BaseException as error:  # noqa: BLE001
                return WorkerOutcome(
                    ok=False,
                    failure_class="crash",
                    detail=f"in-process failure contained: {type(error).__name__}: {error}",
                    wall_clock_ms=(time.perf_counter() - start) * 1000.0,
                )
            wall_ms = (time.perf_counter() - start) * 1000.0
            if wall_ms > budget.wall_clock_ms:
                return WorkerOutcome(
                    ok=False,
                    failure_class="timeout",
                    detail=(
                        f"wall-clock budget exceeded: {wall_ms:.0f} ms > "
                        f"{budget.wall_clock_ms} ms"
                    ),
                    wall_clock_ms=wall_ms,
                )
            return WorkerOutcome(ok=True, payload=payload, wall_clock_ms=wall_ms)

        # Process isolation requires enforceable OS resource limits. Windows
        # has no resource.setrlimit equivalent in the standard library; do not
        # silently run an unbounded child process or pretend the budget applies.
        if resource is None or os.name != "posix":
            return WorkerOutcome(
                ok=False,
                failure_class="unsupported_platform",
                detail=(
                    "process-isolated execution is unavailable on this platform: "
                    "the required CPU and memory limits cannot be enforced"
                ),
                wall_clock_ms=(time.perf_counter() - start) * 1000.0,
            )

        context = multiprocessing.get_context("fork")
        import json
        import tempfile

        work_dir = tempfile.mkdtemp(prefix="frce-worker-")
        result_path = os.path.join(work_dir, "result.json")
        with open(result_path, "w", encoding="utf-8") as handle:
            handle.write("")
        process = context.Process(
            target=_child_entry,
            args=(function, tuple(args), budget, result_path),
            daemon=True,
        )
        process.start()
        process.join(timeout=max(0.05, budget.wall_clock_ms / 1000.0))
        wall_ms = (time.perf_counter() - start) * 1000.0

        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
            shutil.rmtree(work_dir, ignore_errors=True)
            return WorkerOutcome(
                ok=False,
                failure_class="timeout",
                detail=(
                    f"isolated worker exceeded the wall-clock budget "
                    f"({budget.wall_clock_ms} ms) and was terminated"
                ),
                wall_clock_ms=wall_ms,
            )
        exit_code = process.exitcode
        message: dict[str, Any] | None = None
        try:
            with open(result_path, "r", encoding="utf-8") as handle:
                content = handle.read().strip()
            if content:
                decoded = json.loads(content)
                message = decoded if isinstance(decoded, dict) else None
        except (OSError, ValueError):
            message = None
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
        if exit_code not in (0, None) and message is None:
            return WorkerOutcome(
                ok=False,
                failure_class="crash",
                detail=f"worker crashed with exit code {exit_code}",
                wall_clock_ms=wall_ms,
            )
        if message is None:
            return WorkerOutcome(
                ok=False,
                failure_class="crash",
                detail="worker produced no result before exiting",
                wall_clock_ms=wall_ms,
            )
        if message.get("kind") == "exception":
            return WorkerOutcome(
                ok=False,
                failure_class="crash",
                detail=f"worker raised: {message.get('detail', '')}",
                wall_clock_ms=wall_ms,
            )
        payload = message.get("payload")
        try:
            serialized = json.dumps(payload, default=str)
        except (TypeError, ValueError):
            serialized = "unserializable"
        if len(serialized.encode("utf-8")) > budget.max_output_bytes:
            return WorkerOutcome(
                ok=False,
                failure_class="output_limit",
                detail=(
                    f"worker output exceeded {budget.max_output_bytes} bytes and "
                    "was rejected"
                ),
                wall_clock_ms=wall_ms,
            )
        return WorkerOutcome(ok=True, payload=payload, wall_clock_ms=wall_ms)


# ---------------------------------------------------------------------------
# Failure classification and recovery (Step 48)
# ---------------------------------------------------------------------------


class FailureClass:
    """Failure vocabulary for recovery decisions (Step 48)."""

    TIMEOUT = "timeout"
    CRASH = "crash"
    MALFORMED_OUTPUT = "malformed_output"
    UNSUPPORTED_THEORY = "unsupported_theory"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED_PLATFORM = "unsupported_platform"

    TRANSIENT = (TIMEOUT, CRASH, RESOURCE_EXHAUSTED)
    ALL = (
        TIMEOUT, CRASH, MALFORMED_OUTPUT, UNSUPPORTED_THEORY,
        RESOURCE_EXHAUSTED, INVALID_INPUT, UNSUPPORTED_PLATFORM,
    )


@dataclass(frozen=True, slots=True)
class RecoveryOutcome:
    """The outcome of a recovered execution attempt chain (Step 48)."""

    ok: bool
    payload: Any = None
    failure_class: str | None = None
    attempts: int = 0
    diagnostics: tuple[str, ...] = ()
    wall_clock_ms: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


def classify_failure(error: BaseException) -> str:
    """Map an exception into the failure taxonomy (Step 48)."""
    from .exceptions import (
        InvalidFormalizationError,
        MalformedSolverOutputError,
        SolverBudgetExceededError,
        SolverCrashError,
        UnsupportedTheoryError,
    )

    if isinstance(error, SolverBudgetExceededError):
        return FailureClass.RESOURCE_EXHAUSTED
    if isinstance(error, SolverCrashError):
        return FailureClass.CRASH
    if isinstance(error, MalformedSolverOutputError):
        return FailureClass.MALFORMED_OUTPUT
    if isinstance(error, UnsupportedTheoryError):
        return FailureClass.UNSUPPORTED_THEORY
    if isinstance(error, InvalidFormalizationError):
        return FailureClass.INVALID_INPUT
    if isinstance(error, TimeoutError):
        return FailureClass.TIMEOUT
    return FailureClass.CRASH


class RecoveryPolicy:
    """Bounded retry with backoff for transient failures only (Step 48)."""

    def __init__(self, *, retry_backoff_ms: int = 100, max_attempts_cap: int = 3) -> None:
        self.retry_backoff_ms = max(0, retry_backoff_ms)
        self.max_attempts_cap = max(1, max_attempts_cap)

    def execute(
        self,
        function: Callable[..., Any],
        args: Sequence[Any],
        budget: Budget,
        worker: BoundedWorker,
    ) -> RecoveryOutcome:
        """Run under the worker, retrying only transient operational failures.

        A retry never changes correctness: only timeout/crash/resource classes
        are retried, at most ``budget.max_retries`` times (capped), with bounded
        backoff between attempts.
        """
        diagnostics: list[str] = []
        attempts_allowed = 1 + min(int(budget.max_retries), self.max_attempts_cap - 1)
        start = time.perf_counter()
        for attempt in range(1, attempts_allowed + 1):
            outcome = worker.run(function, args, budget)
            if outcome.ok:
                return RecoveryOutcome(
                    ok=True,
                    payload=outcome.payload,
                    attempts=attempt,
                    diagnostics=tuple(diagnostics),
                    wall_clock_ms=(time.perf_counter() - start) * 1000.0,
                )
            diagnostics.append(
                f"attempt {attempt}: {outcome.failure_class}: {outcome.detail}"
            )
            if outcome.failure_class not in FailureClass.TRANSIENT:
                return RecoveryOutcome(
                    ok=False,
                    failure_class=outcome.failure_class,
                    attempts=attempt,
                    diagnostics=tuple(diagnostics),
                    wall_clock_ms=(time.perf_counter() - start) * 1000.0,
                )
            if attempt < attempts_allowed:
                time.sleep(self.retry_backoff_ms / 1000.0 * attempt)
        return RecoveryOutcome(
            ok=False,
            failure_class=outcome.failure_class,
            attempts=attempts_allowed,
            diagnostics=tuple(diagnostics),
            wall_clock_ms=(time.perf_counter() - start) * 1000.0,
        )


# ---------------------------------------------------------------------------
# Capability flags (Step 52)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CapabilityState:
    """One capability flag with its acceptance-suite gate (Step 52)."""

    name: str
    enabled: bool
    acceptance_suite: str
    promoted_at: str | None = None
    note: str = ""


class CapabilityFlags:
    """Incremental release behind capability flags (Step 52).

    ``datalog_engine`` and ``smt_z3``/``cpsat_optimization`` follow backend
    availability; ``lean_export`` requires a pinned Lean binary;
    ``process_isolation`` is opt-in. A flag can only be promoted when its
    named acceptance suite has passed — disabled flags keep their suites off.
    """

    KNOWN = ("datalog_engine", "smt_z3", "cpsat_optimization", "lean_export", "process_isolation")

    def __init__(self, *, lean_available: bool = False) -> None:
        from .smt import Z3_AVAILABLE

        self._flags: dict[str, CapabilityState] = {
            "datalog_engine": CapabilityState(
                "datalog_engine", True, "suite.datalog",
                note="pure-Python engine, always available",
            ),
            "smt_z3": CapabilityState(
                "smt_z3", Z3_AVAILABLE, "suite.smt_z3",
                note="enabled when z3-solver is importable",
            ),
            "cpsat_optimization": CapabilityState(
                "cpsat_optimization", True, "suite.cpsat",
                note="CP-SAT backend with exact reference fallback",
            ),
            "lean_export": CapabilityState(
                "lean_export", lean_available, "suite.lean",
                note="enabled only with a pinned Lean toolchain",
            ),
            "process_isolation": CapabilityState(
                "process_isolation", False, "suite.isolation",
                note="opt-in bounded-worker isolation",
            ),
        }

    def is_enabled(self, name: str) -> bool:
        """Return whether one capability flag is enabled."""
        if name not in self._flags:
            raise ValueError(f"unknown capability flag: {name!r}")
        return self._flags[name].enabled

    def set(self, name: str, enabled: bool) -> None:
        """Enable or disable one flag (release control path)."""
        if name not in self._flags:
            raise ValueError(f"unknown capability flag: {name!r}")
        previous = self._flags[name]
        self._flags[name] = CapabilityState(
            name=previous.name,
            enabled=enabled,
            acceptance_suite=previous.acceptance_suite,
            promoted_at=previous.promoted_at if enabled else None,
            note=previous.note,
        )

    def promote(self, name: str, suite_id: str, suite_passed: bool) -> CapabilityState:
        """Promote a capability only after its acceptance suite passes (Step 52)."""
        if name not in self._flags:
            raise ValueError(f"unknown capability flag: {name!r}")
        previous = self._flags[name]
        if suite_id != previous.acceptance_suite:
            raise ValueError(
                f"capability {name!r} is gated by {previous.acceptance_suite!r}, "
                f"not {suite_id!r}"
            )
        if not suite_passed:
            raise ValueError(
                f"capability {name!r} cannot be promoted: acceptance suite "
                f"{suite_id!r} has not passed"
            )
        from .version import FORMAL_REASONING_PACKAGE_VERSION

        promoted = CapabilityState(
            name=name, enabled=True, acceptance_suite=suite_id,
            promoted_at=FORMAL_REASONING_PACKAGE_VERSION,
            note=previous.note,
        )
        self._flags[name] = promoted
        return promoted

    def snapshot(self) -> tuple[CapabilityState, ...]:
        """Return all capability states in a stable order."""
        return tuple(self._flags[name] for name in self.KNOWN)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe capability snapshot for envelopes and audit."""
        return {
            state.name: {
                "enabled": state.enabled,
                "acceptance_suite": state.acceptance_suite,
                "promoted_at": state.promoted_at,
                "note": state.note,
            }
            for state in self.snapshot()
        }


__all__ = [
    "WORKER_PROTOCOL_VERSION",
    "BinaryAllowlist",
    "FileAccessPolicy",
    "NetworkPolicy",
    "WorkerOutcome",
    "BoundedWorker",
    "FailureClass",
    "RecoveryOutcome",
    "classify_failure",
    "RecoveryPolicy",
    "CapabilityState",
    "CapabilityFlags",
]
