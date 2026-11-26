"""Agent and module capability registry (Step 9).

The registry stores stable worker identity, software/model version,
capabilities, accepted schemas, permission scopes, resource profile, health
and measured reliability history.  Expired or revoked entries are ineligible
for delegation regardless of their score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

from ..contracts import ResourceVector, utc_now

__all__ = (
    "AgentModuleRegistry",
    "HealthState",
    "RegistrationError",
    "WorkerHealthRecord",
    "WorkerManifest",
)


class HealthState(str, Enum):
    """Worker health used as a mandatory eligibility predicate (Step 10)."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class RegistrationError(ValueError):
    """Raised for invalid or conflicting registry entries."""


@dataclass(frozen=True, slots=True)
class WorkerManifest:
    """Stable registration record for one agent/module worker (Step 9)."""

    worker_id: str
    version: str
    capabilities: tuple[str, ...]
    accepted_schemas: tuple[str, ...] = ("object",)
    permission_scopes: tuple[str, ...] = ()
    resource_profile: ResourceVector = ResourceVector()
    reliability: float = 0.9
    latency_seconds: float = 1.0
    cost_per_task: float = 1.0
    load: float = 0.0
    evaluation_evidence: tuple[str, ...] = ()
    biological_reference: str = ""
    registered_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate the manifest envelope."""
        if not self.worker_id or not self.worker_id.strip():
            raise RegistrationError("worker_id must be a non-empty string")
        if not self.version or not self.version.strip():
            raise RegistrationError("version must be a non-empty string")
        if not self.capabilities:
            raise RegistrationError("a worker must declare at least one capability")
        if not 0.0 <= self.reliability <= 1.0:
            raise RegistrationError("reliability must be in [0, 1]")
        if self.latency_seconds < 0.0 or self.cost_per_task < 0.0:
            raise RegistrationError("latency and cost must be non-negative")
        if not 0.0 <= self.load <= 1.0:
            raise RegistrationError("load must be in [0, 1]")

    def supports(self, capability: str) -> bool:
        """Return whether the worker declares exactly this capability."""
        return capability in self.capabilities

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest."""
        return {
            "worker_id": self.worker_id,
            "version": self.version,
            "capabilities": list(self.capabilities),
            "accepted_schemas": list(self.accepted_schemas),
            "permission_scopes": list(self.permission_scopes),
            "resource_profile": self.resource_profile.to_dict(),
            "reliability": self.reliability,
            "latency_seconds": self.latency_seconds,
            "cost_per_task": self.cost_per_task,
            "load": self.load,
            "evaluation_evidence": list(self.evaluation_evidence),
            "biological_reference": self.biological_reference,
            "registered_at": self.registered_at,
        }


@dataclass(slots=True)
class WorkerHealthRecord:
    """Mutable health and reliability history for one worker (Steps 9, 15)."""

    manifest: WorkerManifest
    health: HealthState = HealthState.HEALTHY
    last_heartbeat: str = ""
    successes: int = 0
    failures: int = 0
    revoked: bool = False
    expired: bool = False

    @property
    def eligible(self) -> bool:
        """Return whether the worker is currently eligible for delegation."""
        return not self.revoked and not self.expired and self.health is not HealthState.UNAVAILABLE

    def record_outcome(self, succeeded: bool) -> None:
        """Record one measured execution outcome (Step 11 quality feature)."""
        if succeeded:
            self.successes += 1
        else:
            self.failures += 1

    @property
    def measured_success_rate(self) -> float | None:
        """Return the measured success rate, or None before any execution."""
        total = self.successes + self.failures
        if total == 0:
            return None
        return self.successes / total

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record."""
        return {
            "manifest": self.manifest.to_dict(),
            "health": self.health.value,
            "last_heartbeat": self.last_heartbeat,
            "successes": self.successes,
            "failures": self.failures,
            "revoked": self.revoked,
            "expired": self.expired,
        }


class AgentModuleRegistry:
    """In-memory registry of worker manifests with health and versions (Step 9)."""

    def __init__(self) -> None:
        """Start with an empty registry."""
        self._records: dict[str, WorkerHealthRecord] = {}

    def register(self, manifest: WorkerManifest, *, replace: bool = False) -> None:
        """Register a worker manifest; conflicting IDs require replace."""
        if not isinstance(manifest, WorkerManifest):
            raise TypeError("manifest must be a WorkerManifest")
        existing = self._records.get(manifest.worker_id)
        if existing is not None and not replace:
            raise RegistrationError(
                f"worker {manifest.worker_id} already registered; pass replace=True")
        self._records[manifest.worker_id] = WorkerHealthRecord(manifest=manifest)

    def unregister(self, worker_id: str) -> bool:
        """Remove a registration; returns whether it existed."""
        return self._records.pop(worker_id, None) is not None

    def revoke(self, worker_id: str) -> None:
        """Revoke a worker; revoked entries are ineligible (Step 9)."""
        self._require(worker_id).revoked = True

    def reinstate(self, worker_id: str) -> None:
        """Reinstate a revoked worker after review."""
        self._require(worker_id).revoked = False

    def expire(self, worker_id: str) -> None:
        """Expire a registration (e.g. stale version pin)."""
        self._require(worker_id).expired = True

    def set_health(self, worker_id: str, health: HealthState) -> None:
        """Update the health state of a worker (Step 15)."""
        self._require(worker_id).health = health

    def heartbeat(self, worker_id: str) -> None:
        """Record a heartbeat from a live worker (Step 15)."""
        self._require(worker_id).last_heartbeat = utc_now()

    def record_outcome(self, worker_id: str, succeeded: bool) -> None:
        """Record a measured execution outcome for ranking features (Step 11).

        Unknown workers (e.g. the unassigned placeholder) are ignored: health
        tracking only applies to registered workers.
        """
        record = self._records.get(worker_id)
        if record is None:
            return
        record.record_outcome(succeeded)

    def get(self, worker_id: str) -> WorkerHealthRecord | None:
        """Return the health record for a worker, or None."""
        return self._records.get(worker_id)

    def list_workers(self, *, eligible_only: bool = True) -> tuple[WorkerHealthRecord, ...]:
        """List worker records, optionally restricted to eligible ones."""
        records = list(self._records.values())
        if eligible_only:
            records = [record for record in records if record.eligible]
        return tuple(records)

    def by_capability(self, capability: str) -> tuple[WorkerHealthRecord, ...]:
        """Return every worker declaring this capability (Step 10).

        Eligibility filtering happens in the supervisor so exclusions are
        recorded with reasons rather than silently dropped.
        """
        return tuple(record for record in self.list_workers(eligible_only=False)
                     if record.manifest.supports(capability))

    def describe(self) -> dict[str, Any]:
        """Return a serializable registry summary."""
        return {
            "workers": len(self._records),
            "eligible": len(self.list_workers()),
            "manifests": [record.to_dict() for record in self._records.values()],
        }

    def _require(self, worker_id: str) -> WorkerHealthRecord:
        """Return the record or raise."""
        record = self._records.get(worker_id)
        if record is None:
            raise RegistrationError(f"unknown worker: {worker_id}")
        return record


def manifest_from_dict(value: Mapping[str, Any]) -> WorkerManifest:
    """Rebuild a WorkerManifest from its serialized form."""
    profile = value.get("resource_profile") or {}
    return WorkerManifest(
        worker_id=value["worker_id"],
        version=value["version"],
        capabilities=tuple(value.get("capabilities", ())),
        accepted_schemas=tuple(value.get("accepted_schemas", ("object",))),
        permission_scopes=tuple(value.get("permission_scopes", ())),
        resource_profile=ResourceVector(
            cpu_units=float(profile.get("cpu_units", 1.0)),
            memory_mb=float(profile.get("memory_mb", 64.0)),
            io_units=float(profile.get("io_units", 0.0)),
            network_units=float(profile.get("network_units", 0.0)),
            seconds=float(profile.get("seconds", 1.0))),
        reliability=float(value.get("reliability", 0.9)),
        latency_seconds=float(value.get("latency_seconds", 1.0)),
        cost_per_task=float(value.get("cost_per_task", 1.0)),
        load=float(value.get("load", 0.0)),
        evaluation_evidence=tuple(value.get("evaluation_evidence", ())),
        biological_reference=value.get("biological_reference", ""),
    )


def asdict_manifest(manifest: WorkerManifest) -> dict[str, Any]:  # pragma: no cover
    """Serialize a manifest via asdict for callers preferring the stdlib."""
    return asdict(manifest)
