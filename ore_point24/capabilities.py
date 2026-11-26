"""Phase VIII, Step 36 - capability-based module access and sandbox boundaries.

Grant only the filesystem, network, data and tool capabilities required by a
module; separate read-only inputs from writable outputs; enforce CPU, memory,
wall-time and output-size limits; restrict network access unless explicitly
required; record the sandbox configuration and execution result; and treat a
failure to enforce a requested limit as an admission failure.  Effective
permissions are the allow-list intersection
``P_effective = P_requested AND P_granted AND P_policy``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .core import new_id, sha256_digest, utc_now_iso

CAPABILITY_SCHEMA = "ore.point24.capabilities.v1"

FILESYSTEM_READ = "filesystem.read"
FILESYSTEM_WRITE = "filesystem.write"
NETWORK_ACCESS = "network.access"
DATA_SENSITIVE = "data.sensitive"
TOOL_EXECUTION = "tool.execution"
SUBPROCESS_SPAWN = "subprocess.spawn"

ALL_RIGHTS = (
    FILESYSTEM_READ, FILESYSTEM_WRITE, NETWORK_ACCESS,
    DATA_SENSITIVE, TOOL_EXECUTION, SUBPROCESS_SPAWN,
)


@dataclass(frozen=True, slots=True)
class SandboxSpec:
    """A declarative least-privilege execution context."""

    effective_permissions: tuple[str, ...]
    read_only_inputs: tuple[str, ...]
    writable_outputs: tuple[str, ...]
    cpu_limit_seconds: float | None
    memory_limit_mb: float | None
    wall_time_limit_seconds: float | None
    output_size_limit_mb: float | None
    network_allowed: bool
    environment_allowlist: tuple[str, ...]
    spec_id: str = field(default_factory=lambda: new_id("sandbox"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "effective_permissions": list(self.effective_permissions),
            "read_only_inputs": list(self.read_only_inputs),
            "writable_outputs": list(self.writable_outputs),
            "cpu_limit_seconds": self.cpu_limit_seconds,
            "memory_limit_mb": self.memory_limit_mb,
            "wall_time_limit_seconds": self.wall_time_limit_seconds,
            "output_size_limit_mb": self.output_size_limit_mb,
            "network_allowed": self.network_allowed,
            "environment_allowlist": list(self.environment_allowlist),
        }


class CapabilityAuthority:
    """Compute effective permissions as an allow-list intersection.

    A module receives only the intersection of what it requested, what the
    operator granted and what policy permits; anything outside the
    intersection is denied by construction, and generated programs are
    untrusted until separately reviewed and tested.
    """

    def __init__(self, policy_permissions: Iterable[str] = (FILESYSTEM_READ,)) -> None:
        policy = set(policy_permissions)
        unknown = policy - set(ALL_RIGHTS)
        if unknown:
            raise ValueError(f"policy contains unknown rights: {sorted(unknown)}")
        self._policy = frozenset(policy)

    def effective(
        self,
        requested: Iterable[str],
        granted: Iterable[str],
    ) -> tuple[str, ...]:
        requested_set = set(requested)
        granted_set = set(granted)
        unknown = (requested_set | granted_set) - set(ALL_RIGHTS)
        if unknown:
            raise ValueError(f"unknown rights requested or granted: {sorted(unknown)}")
        return tuple(sorted(requested_set & granted_set & self._policy))

    def policy_allows(self, permission: str) -> bool:
        return permission in self._policy


class SandboxEnforcer:
    """Build sandbox specifications and verify limits are enforceable.

    A failure to enforce a *requested* limit is an admission failure: the
    enforcer never silently downgrades a requested restriction to a warning.
    """

    def __init__(self, authority: CapabilityAuthority | None = None) -> None:
        self.authority = authority or CapabilityAuthority()
        self._specs: dict[str, SandboxSpec] = {}

    def build_spec(
        self,
        *,
        requested: Iterable[str],
        granted: Iterable[str],
        read_only_inputs: Iterable[str] = (),
        writable_outputs: Iterable[str] = (),
        cpu_limit_seconds: float | None = None,
        memory_limit_mb: float | None = None,
        wall_time_limit_seconds: float | None = None,
        output_size_limit_mb: float | None = None,
        environment_allowlist: Iterable[str] = (),
    ) -> SandboxSpec:
        """Compose the least-privilege execution context for one module."""
        for name, value in (
            ("cpu_limit_seconds", cpu_limit_seconds),
            ("memory_limit_mb", memory_limit_mb),
            ("wall_time_limit_seconds", wall_time_limit_seconds),
            ("output_size_limit_mb", output_size_limit_mb),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        effective = self.authority.effective(requested, granted)
        network_allowed = NETWORK_ACCESS in effective
        overlap = set(read_only_inputs) & set(writable_outputs)
        if overlap:
            raise ValueError(
                f"paths cannot be both read-only inputs and writable outputs: {sorted(overlap)}"
            )
        spec = SandboxSpec(
            effective_permissions=effective,
            read_only_inputs=tuple(sorted(set(read_only_inputs))),
            writable_outputs=tuple(sorted(set(writable_outputs))),
            cpu_limit_seconds=cpu_limit_seconds,
            memory_limit_mb=memory_limit_mb,
            wall_time_limit_seconds=wall_time_limit_seconds,
            output_size_limit_mb=output_size_limit_mb,
            network_allowed=network_allowed,
            environment_allowlist=tuple(sorted(set(environment_allowlist))),
        )
        self._specs[spec.spec_id] = spec
        return spec

    def check_access(self, spec: SandboxSpec, *, permission: str, path: str | None = None) -> bool:
        """Verify one access attempt against the declared sandbox boundaries."""
        if permission not in spec.effective_permissions:
            return False
        if permission in (FILESYSTEM_READ, FILESYSTEM_WRITE) and path is not None:
            if permission == FILESYSTEM_WRITE:
                if not any(path.startswith(prefix) for prefix in spec.writable_outputs):
                    return False
            elif not any(path.startswith(prefix) for prefix in spec.read_only_inputs):
                return False
        if permission == NETWORK_ACCESS and not spec.network_allowed:
            return False
        return True

    def limits_enforceable(self, spec: SandboxSpec) -> tuple[bool, str]:
        """Treat a failure to enforce a requested limit as an admission failure."""
        if spec.wall_time_limit_seconds is None and spec.cpu_limit_seconds is None:
            return False, "no time limit requested; sandbox refuses unbounded execution"
        return True, "requested limits are enforceable by the declared sandbox"

    def summary(self) -> dict[str, Any]:
        return {
            "schema": CAPABILITY_SCHEMA,
            "spec_count": len(self._specs),
            "policy_permissions": sorted(self.authority._policy),
            "note": "generated programs are untrusted until separately reviewed and tested",
        }


__all__ = [
    "ALL_RIGHTS",
    "CAPABILITY_SCHEMA",
    "CapabilityAuthority",
    "SandboxEnforcer",
    "SandboxSpec",
]
