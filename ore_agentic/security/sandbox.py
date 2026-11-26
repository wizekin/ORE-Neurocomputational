"""Quotas, sandbox profiles and untrusted-execution limits (Step 47).

Every delegated execution runs under a sandbox profile with CPU, memory,
wall-clock, network, filesystem and output limits.  The in-process reference
enforces wall-clock time and output size honestly and documents that
language-level restrictions alone are insufficient: a real isolation boundary
(subprocess or container) is required for generated or untrusted code, and
:class:`SandboxProfile.allow_network` defaults to False to make that boundary
explicit.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..contracts import ResourceVector, utc_now
from .policy import Principal

__all__ = (
    "AuditLog",
    "OutputSizeViolation",
    "QuotaExceededError",
    "QuotaLease",
    "QuotaManager",
    "SandboxOutcome",
    "SandboxProfile",
    "SandboxViolation",
    "SandboxedExecutor",
    "TimeoutViolation",
)


class SandboxViolation(RuntimeError):
    """Base class for sandbox limit violations (Step 47)."""


class TimeoutViolation(SandboxViolation):
    """The sandboxed execution exceeded its wall-clock limit."""


class OutputSizeViolation(SandboxViolation):
    """The sandboxed execution produced an oversized output."""


class QuotaExceededError(SandboxViolation):
    """A principal exhausted a governed quota (Step 47)."""


@dataclass(frozen=True, slots=True)
class SandboxProfile:
    """Resource and I/O limits for one execution context (Step 47)."""

    wall_clock_seconds: float = 10.0
    cpu_units: float = 1.0
    memory_mb: float = 256.0
    max_output_bytes: int = 65_536
    allow_network: bool = False
    allow_filesystem: bool = False
    max_processes: int = 1

    def __post_init__(self) -> None:
        """Validate the profile bounds."""
        if self.wall_clock_seconds <= 0.0 or self.memory_mb <= 0.0:
            raise ValueError("wall-clock and memory limits must be positive")
        if self.max_output_bytes <= 0 or self.max_processes <= 0:
            raise ValueError("output and process limits must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the profile."""
        return {
            "wall_clock_seconds": self.wall_clock_seconds,
            "cpu_units": self.cpu_units,
            "memory_mb": self.memory_mb,
            "max_output_bytes": self.max_output_bytes,
            "allow_network": self.allow_network,
            "allow_filesystem": self.allow_filesystem,
            "max_processes": self.max_processes,
        }


@dataclass(frozen=True, slots=True)
class SandboxOutcome:
    """The result of one sandboxed execution with its measured usage."""

    value: Any
    duration_ms: float
    truncated: bool
    violations: tuple[str, ...]
    profile: SandboxProfile

    @property
    def ok(self) -> bool:
        """Return whether the execution completed without violations."""
        return not self.violations


class SandboxedExecutor:
    """Run callables under wall-clock and output-size limits (Step 47).

    The reference implementation enforces the wall-clock budget on a separate
    thread and caps serialized output size.  It is an engineering guard rail,
    not a security boundary: generated or untrusted code requires a real
    process/container boundary, which is why :meth:`run` records the profile
    on every outcome for audit.
    """

    def __init__(self, profile: SandboxProfile | None = None) -> None:
        """Use the given profile or the governed default."""
        self.profile = profile or SandboxProfile()

    def run(self, action: Callable[[], Any],
            profile: SandboxProfile | None = None) -> SandboxOutcome:
        """Execute one action under the sandbox profile."""
        limits = profile or self.profile
        import json as _json

        def _measured() -> Any:
            return action()

        started = utc_now()
        del started  # timestamps measured with perf_counter below
        import time as _time
        begin = _time.perf_counter()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ore-sandbox") as pool:
            future = pool.submit(_measured)
            try:
                raw = future.result(timeout=limits.wall_clock_seconds)
            except FutureTimeout as exc:
                future.cancel()
                raise TimeoutViolation(
                    f"execution exceeded {limits.wall_clock_seconds}s wall clock") from exc
        duration_ms = (_time.perf_counter() - begin) * 1000.0
        violations: list[str] = []
        truncated = False
        try:
            serialized = _json.dumps(raw, default=str)
        except (TypeError, ValueError):
            serialized = str(raw)
        if len(serialized.encode("utf-8")) > limits.max_output_bytes:
            violations.append(
                f"output exceeded {limits.max_output_bytes} bytes")
            truncated = True
            raw = serialized[: limits.max_output_bytes]
        return SandboxOutcome(value=raw, duration_ms=round(duration_ms, 3),
                              truncated=truncated, violations=tuple(violations),
                              profile=limits)


@dataclass(slots=True)
class QuotaLease:
    """A reserved slice of a principal's quota, released after use (Step 47)."""

    principal_id: str
    reserved: ResourceVector
    cost: float
    released: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the lease."""
        return {"principal_id": self.principal_id,
                "reserved": self.reserved.to_dict(), "cost": self.cost,
                "released": self.released}


@dataclass(slots=True)
class _QuotaAccount:
    """Mutable per-principal quota accounting."""

    used: ResourceVector = field(default_factory=lambda: ResourceVector(0, 0, 0, 0, 0))
    cost_used: float = 0.0


class QuotaManager:
    """Per-principal resource and cost quotas with reservation leases (Step 47)."""

    def __init__(self, cpu_units: float = 32.0, memory_mb: float = 4096.0,
                 seconds: float = 300.0, cost: float = 100.0) -> None:
        """Set the governed quota ceilings."""
        self._limits = ResourceVector(cpu_units=cpu_units, memory_mb=memory_mb,
                                      io_units=0.0, network_units=0.0, seconds=seconds)
        self._cost_limit = cost
        self._accounts: dict[str, _QuotaAccount] = {}

    def check_and_reserve(self, principal: Principal | str,
                          demand: ResourceVector, cost: float = 0.0) -> QuotaLease:
        """Reserve capacity or raise QuotaExceededError (Step 7 admission)."""
        principal_id = principal.principal_id if isinstance(principal, Principal) else principal
        account = self._accounts.setdefault(principal_id, _QuotaAccount())
        projected = account.used.add(demand)
        if not projected.fits_within(self._limits):
            raise QuotaExceededError(
                f"principal {principal_id} would exceed a resource quota: "
                f"{projected.to_dict()} vs {self._limits.to_dict()}")
        if account.cost_used + cost > self._cost_limit:
            raise QuotaExceededError(
                f"principal {principal_id} would exceed the cost quota")
        account.used = projected
        account.cost_used += cost
        return QuotaLease(principal_id=principal_id, reserved=demand, cost=cost)

    def release(self, lease: QuotaLease) -> None:
        """Return an unused reservation to the pool."""
        account = self._accounts.get(lease.principal_id)
        if account is None or lease.released:
            return
        account.used = ResourceVector(
            cpu_units=max(0.0, account.used.cpu_units - lease.reserved.cpu_units),
            memory_mb=max(0.0, account.used.memory_mb - lease.reserved.memory_mb),
            io_units=max(0.0, account.used.io_units - lease.reserved.io_units),
            network_units=max(0.0, account.used.network_units - lease.reserved.network_units),
            seconds=max(0.0, account.used.seconds - lease.reserved.seconds))
        account.cost_used = max(0.0, account.cost_used - lease.cost)
        lease.released = True

    def usage(self, principal_id: str) -> dict[str, Any]:
        """Return the current usage snapshot for a principal."""
        account = self._accounts.get(principal_id)
        if account is None:
            return {"resource": ResourceVector(0, 0, 0, 0, 0).to_dict(), "cost": 0.0}
        return {"resource": account.used.to_dict(), "cost": account.cost_used}


class AuditLog:
    """Append-only audit log of authorization and tool events (Steps 34, 43)."""

    def __init__(self) -> None:
        """Start with an empty log."""
        self._entries: list[dict[str, Any]] = []

    def record(self, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Append one audit entry with a timestamp."""
        entry = {"kind": kind, "payload": dict(payload), "timestamp": utc_now()}
        self._entries.append(entry)
        return entry

    def entries(self, kind: str | None = None) -> tuple[dict[str, Any], ...]:
        """Return audit entries, optionally filtered by kind."""
        if kind is None:
            return tuple(self._entries)
        return tuple(entry for entry in self._entries if entry["kind"] == kind)

    def summary(self) -> dict[str, Any]:
        """Return a serializable summary."""
        return {"entries": len(self._entries),
                "kinds": sorted({entry["kind"] for entry in self._entries})}
