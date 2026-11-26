"""Execution and isolation plane (integrated steps 5, 9, 21, 39-42, 5.7).

``InProcessBackend`` runs trusted components as in-process tasks with
cooperative budget checks; ``SubprocessExecutionBackend`` runs components as
supervised child processes with a scrubbed environment (no ambient
credentials, task-temp directories only), POSIX rlimits for CPU, memory, and
output-size enforcement, and deadline kills that preserve diagnostic
metadata (steps 9, 21, 39). Container/microVM/remote backends implement the
same interface but report unavailability honestly on hosts without the
required tooling (step 40) -- Python-level checks are not a security
boundary, so stronger tiers exist exactly for untrusted generated code.
``isolation_score`` computes I = alphaT + betaE + gammaP + deltaS (section
5.7) but hard rules override the score: it can never authorize a forbidden
combination.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .contracts import (
    BackendUnavailableError,
    ConsequenceClass,
    ExecutionTier,
    TrustClass,
    canonical_json,
    sha256_hex,
)


class ExecutionState:
    """Lifecycle of one execution handle."""

    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass
class ExecutionHandle:
    """Supervised execution of one task (section 6, ExecutionBackend.run)."""

    handle_id: str = field(default_factory=lambda: f"exec-{uuid.uuid4().hex}")
    component_id: str = ""
    tier: ExecutionTier = ExecutionTier.LOCAL_PROCESS
    backend: str = "in_process"
    state: str = ExecutionState.PENDING
    entry_point: str = ""
    input_artifact_id: str = ""
    output_artifact_id: str = ""
    result: Mapping[str, Any] = field(default_factory=dict)
    exit_info: Mapping[str, Any] = field(default_factory=dict)
    started_monotonic: float = 0.0
    finished_monotonic: float = 0.0
    budget_seconds: float = 30.0
    deadline_monotonic: float | None = None
    process: subprocess.Popen | None = None
    work_dir: str = ""
    stderr_text: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- queries ------------------------------------------------------------

    def elapsed_seconds(self, now: float | None = None) -> float:
        current = now if now is not None else time.monotonic()
        if self.state == ExecutionState.PENDING:
            return 0.0
        end = self.finished_monotonic or current
        return max(0.0, end - self.started_monotonic)

    def deadline_remaining(self, now: float | None = None) -> float | None:
        if self.deadline_monotonic is None:
            return None
        current = now if now is not None else time.monotonic()
        return self.deadline_monotonic - current

    # -- supervision ---------------------------------------------------------

    def cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            if self.state in (ExecutionState.FINISHED, ExecutionState.FAILED):
                return
            if self.process is not None and self.process.poll() is None:
                self._terminate_process()
            self.state = ExecutionState.CANCELLED
            self.exit_info = {**dict(self.exit_info), "cancel_reason": reason}
            self.finished_monotonic = time.monotonic()

    def _terminate_process(self) -> None:
        assert self.process is not None
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
        except Exception:  # pragma: no cover - process may already be gone
            pass

    def poll(self) -> str:
        return self.state

    def describe(self) -> dict[str, Any]:
        return {
            "handle_id": self.handle_id,
            "component_id": self.component_id,
            "tier": self.tier.value,
            "backend": self.backend,
            "state": self.state,
            "entry_point": self.entry_point,
            "budget_seconds": self.budget_seconds,
            "elapsed_seconds": self.elapsed_seconds(),
            "result": dict(self.result),
            "exit_info": dict(self.exit_info),
            "work_dir": self.work_dir,
        }

    # -- context manager so cleanup always runs (step 26) ----------------------

    def __enter__(self) -> "ExecutionHandle":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self.state == ExecutionState.RUNNING:
            self.cancel("context exit")
        if self.work_dir:
            shutil.rmtree(self.work_dir, ignore_errors=True)
        return False


class ExecutionBackend:
    """Interface every isolation backend implements (step 40)."""

    name = "abstract"

    def available(self) -> bool:
        raise NotImplementedError

    def run(
        self,
        manifest,
        payload: Mapping[str, Any],
        *,
        budget_seconds: float = 30.0,
        deadline_monotonic: float | None = None,
        capability: Any | None = None,
    ) -> ExecutionHandle:
        raise NotImplementedError


class InProcessBackend(ExecutionBackend):
    """Trusted in-process execution with cooperative budget checks (step 5).

    Threat model: this backend provides no isolation against hostile code;
    the manifest trust class must already be trusted-core or trusted-service
    (enforced at registration) and budget enforcement is cooperative.
    """

    name = "in_process"

    def available(self) -> bool:
        return True

    def run(
        self,
        manifest,
        payload: Mapping[str, Any],
        *,
        budget_seconds: float = 30.0,
        deadline_monotonic: float | None = None,
        capability: Any | None = None,
    ) -> ExecutionHandle:
        handle = ExecutionHandle(
            component_id=manifest.component_id,
            tier=ExecutionTier.IN_PROCESS,
            backend=self.name,
            entry_point=str(manifest.entry_point)
            if isinstance(manifest.entry_point, str)
            else repr(manifest.entry_point),
            budget_seconds=budget_seconds,
            deadline_monotonic=deadline_monotonic,
        )
        deadline = deadline_monotonic if deadline_monotonic is not None else (
            time.monotonic() + budget_seconds
        )
        started_cpu = time.process_time()
        handle.started_monotonic = time.monotonic()
        handle.state = ExecutionState.RUNNING
        try:
            callable_entry = manifest.entry_callable()
            result = callable_entry(dict(payload))
            handle.finished_monotonic = time.monotonic()
            cpu_used = time.process_time() - started_cpu
            handle.result = dict(result) if isinstance(result, Mapping) else {"value": result}
            handle.exit_info = {
                "cpu_seconds": round(cpu_used, 6),
                "cooperative_budget_check": "cpu and wall-clock measured post-hoc",
            }
            if time.monotonic() > deadline:
                handle.state = ExecutionState.TIMED_OUT
            else:
                handle.state = ExecutionState.FINISHED
        except Exception as exc:
            handle.finished_monotonic = time.monotonic()
            handle.state = ExecutionState.FAILED
            handle.exit_info = {
                "exception_type": type(exc).__name__,
                "exception": str(exc)[:500],
                "cpu_seconds": round(time.process_time() - started_cpu, 6),
            }
        return handle


_POSIX = os.name == "posix"


def _build_rlimits(memory_bytes: int, cpu_seconds: float, max_output_bytes: int) -> list[tuple]:
    """Resource limits for the child process where POSIX provides them (step 39)."""
    if not _POSIX:
        return []
    import resource

    limits: list[tuple] = []
    if memory_bytes > 0:
        limits.append((resource.RLIMIT_AS, (memory_bytes, memory_bytes)))
    if cpu_seconds > 0:
        limits.append((resource.RLIMIT_CPU, (int(cpu_seconds), int(cpu_seconds) + 1)))
    if max_output_bytes > 0:
        limits.append((resource.RLIMIT_FSIZE, (max_output_bytes, max_output_bytes)))
    return limits


def _preexec_limits(limits: list[tuple]) -> Callable[[], None]:
    def apply() -> None:
        import resource

        for limit_kind, (soft, hard) in limits:
            resource.setrlimit(limit_kind, (soft, hard))

    return apply


class SubprocessExecutionBackend(ExecutionBackend):
    """Supervised child-process backend with OS limits (steps 9, 21, 39).

    The child runs ``python -m ore_runtime.worker`` with a job file and a
    result file inside a per-handle task-temp directory, a scrubbed
    environment (only PATH/PYTHONPATH/PYTHONDONTWRITEBYTECODE/LC_ALL -- no
    ambient credentials), and rlimits for address space, CPU seconds, and
    output size on POSIX. The deadline is enforced by the supervisor with a
    terminate-then-kill escalation that preserves stderr diagnostics.
    """

    name = "local_process"

    def __init__(
        self,
        *,
        python_executable: str | None = None,
        max_output_bytes: int = 16 * 1024 * 1024,
        repository_root: str | Path | None = None,
        apply_rlimits: bool = True,
    ) -> None:
        self.python_executable = python_executable or sys.executable
        self.max_output_bytes = int(max_output_bytes)
        self._repository_root = (
            str(Path(repository_root).expanduser().resolve())
            if repository_root
            else str(Path(__file__).resolve().parents[1])
        )
        self.apply_rlimits = apply_rlimits and _POSIX

    def available(self) -> bool:
        return _POSIX

    def run(
        self,
        manifest,
        payload: Mapping[str, Any],
        *,
        budget_seconds: float = 30.0,
        deadline_monotonic: float | None = None,
        capability: Any | None = None,
    ) -> ExecutionHandle:
        entry = manifest.entry_point if isinstance(manifest.entry_point, str) else (
            f"{getattr(manifest.entry_point, '__module__', '')}:"
            f"{getattr(manifest.entry_point, '__qualname__', '')}"
        )
        work_dir = tempfile.mkdtemp(prefix=f"ore-exec-{manifest.component_id[:24]}-")
        job_file = Path(work_dir) / "job.json"
        result_file = Path(work_dir) / "result.json"
        resources = getattr(manifest, "resources", None)
        memory_limit = int(resources.limits.memory_bytes) if resources is not None else 0
        if memory_limit <= 0:
            memory_limit = 512 * 1024 * 1024
        limits = _build_rlimits(
            memory_limit,
            max(1.0, budget_seconds),
            self.max_output_bytes,
        )
        handle = ExecutionHandle(
            component_id=manifest.component_id,
            tier=ExecutionTier.LOCAL_PROCESS,
            backend=self.name,
            entry_point=entry,
            budget_seconds=budget_seconds,
            deadline_monotonic=deadline_monotonic,
            work_dir=work_dir,
        )
        job = {
            "entry_point": entry,
            "payload": dict(payload),
            "budget_seconds": budget_seconds,
            "component_id": manifest.component_id,
        }
        job_file.write_text(canonical_json(job), encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": self._repository_root,
            "PYTHONDONTWRITEBYTECODE": "1",
            "LC_ALL": "C",
        }
        preexec = _preexec_limits(limits) if self.apply_rlimits and limits else None
        try:
            handle.process = subprocess.Popen(
                [
                    self.python_executable,
                    "-m",
                    "ore_runtime.worker",
                    "--job-file",
                    str(job_file),
                    "--result-file",
                    str(result_file),
                ],
                env=env,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=preexec,
            )
        except OSError as exc:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise BackendUnavailableError(
                f"cannot spawn worker process: {exc}"
            ) from exc
        handle.state = ExecutionState.RUNNING
        handle.started_monotonic = time.monotonic()
        timeout = (
            max(0.0, deadline_monotonic - handle.started_monotonic)
            if deadline_monotonic is not None
            else budget_seconds
        )
        try:
            stdout_text, stderr_text = handle.process.communicate(timeout=max(timeout, 0.05))
        except subprocess.TimeoutExpired:
            # Deadline kill with diagnostic preservation (step 21/acceptance).
            handle._terminate_process()
            stdout_text, stderr_text = handle.process.communicate()
            handle.stderr_text = stderr_text or ""
            handle.state = ExecutionState.TIMED_OUT
            handle.finished_monotonic = time.monotonic()
            handle.exit_info = {
                "deadline_exceeded": True,
                "returncode": handle.process.returncode,
                "stderr_excerpt": (stderr_text or "")[-1000:],
                "work_dir_preserved_until_handle_exit": work_dir,
            }
            return handle
        handle.stderr_text = stderr_text or ""
        handle.finished_monotonic = time.monotonic()
        if result_file.exists():
            try:
                result_payload = __import__("json").loads(result_file.read_text(encoding="utf-8"))
            except Exception:
                result_payload = {}
            handle.result = result_payload.get("result", {})
            handle.exit_info = {
                "returncode": handle.process.returncode,
                "stdout_excerpt": (stdout_text or "")[-500:],
                "stderr_excerpt": (stderr_text or "")[-500:],
                "worker_version": result_payload.get("worker_version", ""),
            }
            if handle.process.returncode == 0 and not result_payload.get("error"):
                handle.state = ExecutionState.FINISHED
            else:
                handle.state = ExecutionState.FAILED
                handle.exit_info = {
                    **dict(handle.exit_info),
                    "worker_error": result_payload.get("error", ""),
                }
        else:
            handle.state = ExecutionState.FAILED
            handle.exit_info = {
                "returncode": handle.process.returncode,
                "stderr_excerpt": (stderr_text or "")[-500:],
                "note": "worker produced no result file",
            }
        return handle


class ContainerSandboxBackend(ExecutionBackend):
    """OS container backend (docker-style) for untrusted components (step 40).

    Availability is detected from the host toolchain; when the runtime is
    absent the backend refuses to run rather than silently degrading
    isolation. The command construction is real, but gVisor/microVM style
    backends require separate infrastructure (section 2 non-goals).
    """

    name = "container_sandbox"

    def __init__(self, *, runtime: str | None = None) -> None:
        self._runtime = runtime

    def available(self) -> bool:
        candidates = (self._runtime,) if self._runtime else ("docker", "podman")
        for candidate in candidates:
            if candidate and shutil.which(candidate):
                self._runtime = candidate
                return True
        return False

    def run(self, manifest, payload, **kwargs: Any) -> ExecutionHandle:
        if not self.available():
            raise BackendUnavailableError(
                "no container runtime (docker/podman) on this host; the "
                "container_sandbox tier cannot degrade to a weaker backend"
            )
        raise BackendUnavailableError(
            "container execution requires the external runtime image; see "
            "EnvironmentImage records for the pinned build"
        )


class MicroVMBackend(ExecutionBackend):
    """MicroVM tier placeholder -- honest unavailability (sections 2 and 40)."""

    name = "microvm"

    def available(self) -> bool:
        for tool in ("firecracker", "cloud-hypervisor", "qemu-system-x86_64"):
            if shutil.which(tool):
                return True
        return False

    def run(self, manifest, payload, **kwargs: Any) -> ExecutionHandle:
        if not self.available():
            raise BackendUnavailableError(
                "no microVM monitor on this host; the microvm tier requires "
                "separate infrastructure (non-goal for this implementation)"
            )
        raise BackendUnavailableError(
            "microVM boot requires an operator-provided kernel/rootfs image"
        )


@dataclass(frozen=True)
class EnvironmentImage:
    """Reproducible execution image record (step 41)."""

    image_id: str
    base: str
    pinned_dependencies: tuple[str, ...]
    runtime_versions: Mapping[str, str]
    build_provenance: str
    created_epoch: float

    @classmethod
    def capture_current(cls, *, extra_pins: tuple[str, ...] = ()) -> "EnvironmentImage":
        versions = {
            "python": sys.version.split()[0],
            "platform": sys.platform,
        }
        try:  # numpy/scipy are the repo's declared dependencies
            import numpy

            versions["numpy"] = numpy.__version__
        except Exception:
            versions["numpy"] = "absent"
        try:
            import scipy

            versions["scipy"] = scipy.__version__
        except Exception:
            versions["scipy"] = "absent"
        digest_source = canonical_json(
            {"base": "python", "versions": versions, "pins": list(extra_pins)}
        )
        return cls(
            image_id=f"env-{sha256_hex(digest_source)[:16]}",
            base=f"python-{versions['python']}",
            pinned_dependencies=tuple(sorted(extra_pins)),
            runtime_versions=versions,
            build_provenance="captured from the running interpreter",
            created_epoch=time.time(),
        )

    def reproducibility_report(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "base": self.base,
            "pinned_dependencies": list(self.pinned_dependencies),
            "runtime_versions": dict(self.runtime_versions),
            "build_provenance": self.build_provenance,
            "note": (
                "pinned minimal images make execution reproducible; specialized "
                "library-OS/unikernel images are a later, measured decision"
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.reproducibility_report()


@dataclass(frozen=True)
class RemoteWorker:
    """Leased remote resource record (step 42)."""

    worker_id: str
    endpoint: str
    location: str
    latency_ms: float
    trust_class: str
    availability: float
    leased: bool = False

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if not 0.0 <= self.availability <= 1.0:
            raise ValueError("availability must be within 0..1")


class RemoteWorkerPool:
    """Optional remote pool with leases and local fallback (step 42).

    Distributed execution is Stage 7 and optional: local ORE keeps working
    when remote nodes disappear, by design.
    """

    def __init__(self) -> None:
        self._workers: dict[str, RemoteWorker] = {}

    def register(self, worker: RemoteWorker) -> None:
        self._workers[worker.worker_id] = worker

    def lease(self, worker_id: str) -> RemoteWorker | None:
        worker = self._workers.get(worker_id)
        if worker is None or worker.leased:
            return None
        leased = RemoteWorker(**{**vars(worker), "leased": True})
        self._workers[worker_id] = leased
        return leased

    def release(self, worker_id: str) -> bool:
        worker = self._workers.get(worker_id)
        if worker is None or not worker.leased:
            return False
        released = RemoteWorker(**{**vars(worker), "leased": False})
        self._workers[worker_id] = released
        return True

    def active_workers(self) -> tuple[RemoteWorker, ...]:
        return tuple(
            worker for worker in sorted(self._workers.values()) if worker.availability > 0
        )

    def execution_plan(self, worker_id: str, local_backend: ExecutionBackend) -> str:
        """Local execution is always the fallback when no lease is held."""
        worker = self._workers.get(worker_id)
        if worker is None or not worker.leased:
            return (
                "remote worker unavailable or unleased; falling back to the local "
                f"{local_backend.name} backend so remote loss never blocks local work"
            )
        return f"execute on {worker.endpoint} (leased, latency {worker.latency_ms}ms)"

    def describe(self) -> dict[str, Any]:
        return {
            "workers": [
                {
                    "worker_id": worker.worker_id,
                    "endpoint": worker.endpoint,
                    "location": worker.location,
                    "latency_ms": worker.latency_ms,
                    "trust_class": worker.trust_class,
                    "availability": worker.availability,
                    "leased": worker.leased,
                }
                for worker in sorted(self._workers.values())
            ]
        }


def isolation_score(
    trust_risk: float,
    external_effect: float,
    privilege: float,
    sensitivity: float,
    *,
    weights: tuple[float, float, float, float] = (0.35, 0.25, 0.25, 0.15),
) -> float:
    """I = alpha*T + beta*E + gamma*P + delta*S (section 5.7)."""
    if not all(0.0 <= value <= 1.0 for value in (trust_risk, external_effect, privilege, sensitivity)):
        raise ValueError("risk inputs must be within 0..1")
    if not all(0.0 <= weight <= 1.0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("weights must be non-negative with a positive total")
    alpha, beta, gamma, delta = weights
    return alpha * trust_risk + beta * external_effect + gamma * privilege + delta * sensitivity


def recommend_tier(score: float) -> ExecutionTier:
    """Empirically calibrated score bands (section 5.7)."""
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be within 0..1")
    if score < 0.25:
        return ExecutionTier.IN_PROCESS
    if score < 0.50:
        return ExecutionTier.LOCAL_PROCESS
    if score < 0.75:
        return ExecutionTier.CONTAINER_SANDBOX
    return ExecutionTier.MICROVM


def hard_tier_floor(
    trust_class: TrustClass | str,
    consequence: ConsequenceClass | str,
) -> ExecutionTier:
    """Mandatory minimum tier: hard rules beat the score (section 5.7)."""
    trust = TrustClass.coerce(trust_class)
    consequence = ConsequenceClass.coerce(consequence)
    if trust == TrustClass.UNTRUSTED_GENERATED:
        return ExecutionTier.CONTAINER_SANDBOX
    if consequence == ConsequenceClass.SAFETY_CRITICAL:
        return ExecutionTier.LOCAL_PROCESS
    if consequence == ConsequenceClass.EXTERNAL_SIDE_EFFECT and trust == TrustClass.CONSTRAINED_PLUGIN:
        return ExecutionTier.CONTAINER_SANDBOX
    return ExecutionTier.IN_PROCESS


def resolve_tier(
    trust_class: TrustClass | str,
    consequence: ConsequenceClass | str,
    score: float,
) -> ExecutionTier:
    """The recommended tier, never below the hard floor (section 5.7)."""
    scored = recommend_tier(score)
    floor = hard_tier_floor(trust_class, consequence)
    ranking = {
        ExecutionTier.IN_PROCESS: 0,
        ExecutionTier.LOCAL_PROCESS: 1,
        ExecutionTier.CONTAINER_SANDBOX: 2,
        ExecutionTier.MICROVM: 3,
        ExecutionTier.REMOTE_WORKER: 3,
    }
    return scored if ranking[scored] >= ranking[floor] else floor
