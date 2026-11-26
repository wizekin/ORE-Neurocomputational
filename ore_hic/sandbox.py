"""Restricted sandboxed execution of generated code (spec Phase H, Step 51).

Generated code runs in a subprocess with CPU, memory, file-size and wall-clock
limits, a scrubbed environment containing no credentials, a controlled
temporary working directory, and a static policy pre-check that blocks
unauthorized filesystem, network, subprocess and credential access before
anything is executed. No external side effects are available by default; a
human approval is required before any deployment or consequential action.
"""
from __future__ import annotations

import json
import os
try:
    import resource
except ImportError:  # Windows does not provide the POSIX resource module.
    resource = None  # type: ignore[assignment]
import subprocess  # used only by the sandbox itself, never by generated code
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .codegen import CodeArtifact
from .contracts import ArtifactKind, CanonicalArtifact, Capability, GateViolation
from .session import CompilationSession
from .verification import StaticAnalyzer


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    """The declared execution policy for one sandbox run."""

    cpu_seconds: int
    memory_mb: int
    wall_seconds: float
    output_byte_limit: int
    granted_capabilities: tuple[Capability, ...]
    env_whitelist: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_seconds": self.cpu_seconds,
            "memory_mb": self.memory_mb,
            "wall_seconds": self.wall_seconds,
            "output_byte_limit": self.output_byte_limit,
            "granted_capabilities": [c.value for c in self.granted_capabilities],
            "env_whitelist": list(self.env_whitelist),
        }

    @classmethod
    def default(cls, *, cpu_seconds: int = 5, memory_mb: int = 256,
                wall_seconds: float = 10.0, output_byte_limit: int = 65536) -> SandboxPolicy:
        return cls(
            cpu_seconds=cpu_seconds,
            memory_mb=memory_mb,
            wall_seconds=wall_seconds,
            output_byte_limit=output_byte_limit,
            granted_capabilities=(Capability.COMPUTE, Capability.MEMORY),
            env_whitelist=("PATH", "PYTHONHASHSEED", "SYSTEMROOT", "LANG"),
        )


@dataclass(frozen=True, slots=True)
class SandboxOutcome:
    """One executed fixture and its result."""

    fixture_name: str
    status: str  # PASSED | FAILED | ERROR | BLOCKED
    detail: str
    output_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_name": self.fixture_name,
            "status": self.status,
            "detail": self.detail,
            "output_digest": self.output_digest,
        }


class ExecutionReport(CanonicalArtifact):
    """Canonical artifact: reproducible sandbox execution evidence."""

    kind = ArtifactKind.EXECUTION_REPORT

    def __init__(
        self,
        *,
        artifact_id: str,
        code_artifact_id: str,
        policy: SandboxPolicy,
        outcomes: Iterable[SandboxOutcome],
        exit_code: int | None,
        wall_seconds: float,
        max_rss_kb: int,
        stdout_excerpt: str,
        stderr_excerpt: str,
        blocked_violations: Iterable[str],
        notes: Iterable[str],
        version: int = 1,
    ) -> None:
        self.artifact_id = artifact_id
        self.code_artifact_id = code_artifact_id
        self.policy = policy
        self.outcomes = tuple(outcomes)
        self.exit_code = exit_code
        self.wall_seconds = wall_seconds
        self.max_rss_kb = max_rss_kb
        self.stdout_excerpt = stdout_excerpt
        self.stderr_excerpt = stderr_excerpt
        self.blocked_violations = tuple(blocked_violations)
        self.notes = tuple(notes)
        self.version = version

    @property
    def all_passed(self) -> bool:
        return bool(self.outcomes) and all(o.status == "PASSED" for o in self.outcomes)

    def content_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "code_artifact_id": self.code_artifact_id,
            "policy": self.policy.to_dict(),
            "outcomes": [o.to_dict() for o in self.outcomes],
            "exit_code": self.exit_code,
            "wall_seconds": self.wall_seconds,
            "max_rss_kb": self.max_rss_kb,
            "stdout_excerpt": self.stdout_excerpt,
            "stderr_excerpt": self.stderr_excerpt,
            "blocked_violations": list(self.blocked_violations),
            "notes": list(self.notes),
            "version": self.version,
        }

    @property
    def semantic_digest(self) -> str:
        """Digest over deterministic execution evidence; timings excluded.

        Wall-clock seconds and peak RSS are measurements, not canonical
        content: the same sandboxed run logic yields the same semantic digest.
        """
        from .contracts import digest_payload

        payload = self.content_payload()
        payload.pop("artifact_id", None)
        payload.pop("code_artifact_id", None)
        payload.pop("wall_seconds", None)
        payload.pop("max_rss_kb", None)
        return digest_payload(payload)


_RUNNER_TEMPLATE = """\
import json
import sys

sys.path.insert(0, {module_dir!r})
import {module_name} as generated

results = []
{fixture_calls}
print("HIC_SANDBOX_RESULT:" + json.dumps(results, default=lambda o: getattr(o, "__dict__", str(o))))
"""


class SandboxExecutor:
    """Execute generated code under strict resource and permission limits."""

    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        self.policy = policy or SandboxPolicy.default()
        self.analyzer = StaticAnalyzer()

    # -- policy enforcement ---------------------------------------------------

    def precheck(self, source: str) -> tuple[str, ...]:
        """Static policy pre-check: block unauthorized behavior before running."""
        findings = self.analyzer.analyze_source(source, "sandbox-precheck")
        return tuple(
            f"{f.code}: {f.message}" for f in findings if f.severity == "ERROR"
        )

    def _build_env(self) -> dict[str, str]:
        """Scrubbed environment; no credentials or secrets are passed through."""
        env: dict[str, str] = {}
        for key in self.policy.env_whitelist:
            if key in os.environ:
                env[key] = os.environ[key]
        env["PYTHONHASHSEED"] = "0"
        return env

    def _limits(self) -> None:
        """Preexec hook: OS-level resource limits for the child process."""
        if resource is None:
            raise GateViolation(
                "OS-level sandbox resource limits are unavailable on this platform; "
                "execution is blocked rather than run without the declared limits"
            )
        resource.setrlimit(resource.RLIMIT_CPU, (self.policy.cpu_seconds, self.policy.cpu_seconds))
        memory_bytes = self.policy.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (self.policy.output_byte_limit,) * 2)
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))

    # -- execution --------------------------------------------------------------

    def execute(
        self,
        session: CompilationSession,
        code: CodeArtifact,
        fixtures: Sequence[tuple[str, str] | tuple[str, str, bool]] = (),
        *,
        extra_source: str | None = None,
    ) -> ExecutionReport:
        """Run the generated module against fixtures inside the sandbox.

        A fixture is ``(name, expression)`` or ``(name, expression, expect_error)``;
        with ``expect_error`` the fixture passes only when the expression raises.
        """
        if code.language != "python":
            raise GateViolation("sandbox currently executes Python artifacts only")
        sources = code.source
        if extra_source is not None:
            sources += "\n\n" + extra_source
        blocked = self.precheck(sources)
        if resource is None:
            blocked += (
                "OS-level CPU, memory, file-size and process limits cannot be "
                "enforced on this platform; sandbox execution is fail-closed",
            )
        if blocked:
            report = ExecutionReport(
                artifact_id=session.next_artifact_id(ArtifactKind.EXECUTION_REPORT),
                code_artifact_id=code.artifact_id,
                policy=self.policy,
                outcomes=(SandboxOutcome("precheck", "BLOCKED", "; ".join(blocked), ""),),
                exit_code=None,
                wall_seconds=0.0,
                max_rss_kb=0,
                stdout_excerpt="",
                stderr_excerpt="",
                blocked_violations=blocked,
                notes=("execution blocked by static policy pre-check",),
            )
            session.register(report)
            session.ledger.record(
                stage="H", step_id="HIC51",
                description="Sandbox execution blocked by policy pre-check",
                inputs=[code], outputs=[report],
                rule_references={"HIC-SANDBOX-BLOCK"},
            )
            return report

        with tempfile.TemporaryDirectory(prefix="hic-sandbox-") as module_dir:
            module_path = Path(module_dir) / f"{code.module_name}.py"
            module_path.write_text(sources, encoding="utf-8")
            fixture_calls: list[str] = []
            for fixture in fixtures:
                name, expression = fixture[0], fixture[1]
                expect_error = bool(fixture[2]) if len(fixture) > 2 else False
                if expect_error:
                    fixture_calls.append(
                        f"try:\n"
                        f"    {expression}\n"
                        f"    results.append({{\"fixture_name\": {name!r}, \"status\": \"FAILED\", "
                        f"\"detail\": \"expected an error but none was raised\", \"result\": None}})\n"
                        f"except Exception as _exc:\n"
                        f"    results.append({{\"fixture_name\": {name!r}, \"status\": \"PASSED\", "
                        f"\"detail\": f\"rejected as expected: {{_exc}}\", \"result\": None}})"
                    )
                else:
                    fixture_calls.append(
                        f"try:\n"
                        f"    _result = {expression}\n"
                        f"    results.append({{\"fixture_name\": {name!r}, \"status\": \"PASSED\", "
                        f"\"detail\": str(getattr(_result, 'status', 'ok')), "
                        f"\"result\": _result.to_dict() if hasattr(_result, 'to_dict') else str(_result)}})\n"
                        f"except Exception as _exc:\n"
                        f"    results.append({{\"fixture_name\": {name!r}, \"status\": \"FAILED\", "
                        f"\"detail\": str(_exc), \"result\": None}})"
                    )
            runner = _RUNNER_TEMPLATE.format(
                module_dir=module_dir,
                module_name=code.module_name,
                fixture_calls="\n".join(fixture_calls),
            )
            runner_path = Path(module_dir) / "hic_runner.py"
            runner_path.write_text(runner, encoding="utf-8")

            env = self._build_env()
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    [sys_python(), str(runner_path)],
                    capture_output=True,
                    text=True,
                    check=False,  # exit codes are recorded evidence, not errors
                    timeout=self.policy.wall_seconds,
                    cwd=module_dir,
                    env=env,
                    preexec_fn=self._limits if os.name == "posix" else None,
                )
                exit_code: int | None = completed.returncode
                stdout = completed.stdout[: self.policy.output_byte_limit]
                stderr = completed.stderr[: self.policy.output_byte_limit]
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                exit_code = None
                stdout = (exc.stdout or b"").decode("utf-8", "replace")[: self.policy.output_byte_limit] \
                    if isinstance(exc.stdout, bytes) else str(exc.stdout or "")[: self.policy.output_byte_limit]
                stderr = "sandbox wall-clock timeout exceeded"
                timed_out = True
            wall = time.monotonic() - started

        if resource is not None:
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            max_rss = int(usage.ru_maxrss)
        else:  # defensive; Windows execution is blocked above.
            max_rss = 0
        outcomes, notes = self._parse_outcomes(stdout, timed_out, fixtures)
        report = ExecutionReport(
            artifact_id=session.next_artifact_id(ArtifactKind.EXECUTION_REPORT),
            code_artifact_id=code.artifact_id,
            policy=self.policy,
            outcomes=outcomes,
            exit_code=exit_code,
            wall_seconds=round(wall, 4),
            max_rss_kb=max_rss,
            stdout_excerpt=stdout[-2000:],
            stderr_excerpt=stderr[-2000:],
            blocked_violations=(),
            notes=notes + (
                "execution success is evidence of passing tests, not proof of semantic correctness",
            ),
        )
        session.register(report)
        session.ledger.record(
            stage="H", step_id="HIC51",
            description="Sandboxed execution with resource limits and scrubbed environment",
            inputs=[code], outputs=[report],
            rule_references={"HIC-SANDBOX-1"},
        )
        session.ledger.audit(
            "SANDBOX_EXECUTION",
            report.content_digest,
            f"exit_code={exit_code}",
            f"outcomes={[o.status for o in report.outcomes]}",
        )
        return report

    @staticmethod
    def _parse_outcomes(
        stdout: str, timed_out: bool, fixtures: Sequence[tuple[str, str]]
    ) -> tuple[tuple[SandboxOutcome, ...], tuple[str, ...]]:
        from hashlib import sha256

        notes: list[str] = []
        marker = "HIC_SANDBOX_RESULT:"
        payload = None
        for line in stdout.splitlines():
            if line.startswith(marker):
                payload = line[len(marker):]
                break
        if payload is None:
            notes.append("no structured result emitted by the sandbox run")
            if timed_out:
                notes.append("wall-clock timeout exceeded")
            status = "ERROR" if not timed_out else "FAILED"
            return (
                tuple(
                    SandboxOutcome(name, status, "no result", "")
                    for name, _ in fixtures
                )
                or (SandboxOutcome("run", status, "no result", ""),),
                tuple(notes),
            )
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError:
            return (SandboxOutcome("parse", "ERROR", "result payload is not valid JSON", ""),)
        outcomes = []
        for item in raw:
            outcomes.append(
                SandboxOutcome(
                    str(item.get("fixture_name", "?")),
                    str(item.get("status", "ERROR")),
                    str(item.get("detail", ""))[:500],
                    sha256(str(item.get("result", "")).encode("utf-8")).hexdigest(),
                )
            )
        return tuple(outcomes), tuple(notes)


def sys_python() -> str:
    """Return the interpreter executable path for the child process."""
    import sys

    return sys.executable or "python3"
