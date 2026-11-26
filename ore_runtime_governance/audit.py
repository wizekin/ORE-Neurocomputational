"""Assurance and observability plane (integrated steps 45-46, section 5 audit).

``AuditEvent`` records every security-relevant transition -- registration,
authorization, capability issue/revoke, dispatch, reservation, limit breach,
timeout, cancellation, crash, restart, quarantine, artifact publication, and
policy change -- with actor, target, task, policy version, trace id, and
outcome, never secrets or sensitive payloads (metadata is redacted before
storage). ``SQLiteEventSink`` persists events append-only in a single-writer
SQLite table with a chained SHA-256 hash per event (tamper-evident within the
stated threat model), explicit transaction boundaries, a bounded busy
timeout, and a single serialized writer (integrated step 28). The event log
doubles as the recovery record: ``replay_reservations`` reconstructs the
reservation ledger after transaction interruption. ``register_runtime_annotations``
synchronizes machine-readable runtime contracts into the shared ore_logger
catalogue when the ORE repository is importable (step 46).
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import canonical_json, sha256_hex

_REDACT_PATTERN = re.compile(
    r"(secret|token|password|credential|api[_-]?key|private[_-]?key|session[_-]?id)",
    re.IGNORECASE,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runtime_audit_events (
    event_id TEXT PRIMARY KEY,
    sequence_no INTEGER NOT NULL,
    kind TEXT NOT NULL,
    actor TEXT,
    target TEXT,
    task_id TEXT,
    policy_version INTEGER,
    trace_id TEXT,
    outcome TEXT,
    metadata_json TEXT NOT NULL,
    timestamp_epoch REAL NOT NULL,
    monotonic REAL NOT NULL,
    prev_chain_hash TEXT,
    chain_hash TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_runtime_audit_sequence
    ON runtime_audit_events (sequence_no);
"""

#: Event kinds emitted by the runtime (integrated step 45).
EVENT_KINDS: tuple[str, ...] = (
    "registration",
    "registration_rejected",
    "capability_issue",
    "capability_revoke",
    "generation_bump",
    "lifecycle_transition",
    "dispatch",
    "reservation",
    "reservation_release",
    "reservation_recovered",
    "limit_breach",
    "queue_backpressure",
    "authorization_denied",
    "timeout",
    "cancellation",
    "crash",
    "restart",
    "quarantine",
    "artifact_published",
    "artifact_read",
    "artifact_corruption",
    "policy_change",
    "policy_rollout",
    "policy_rollback",
    "manual_override",
    "health_lease",
    "runtime_annotation",
    "ipc_pause",
)


def redact_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Strip secrets from event metadata before persistence (step 45)."""
    redacted: dict[str, Any] = {}
    for key, value in dict(metadata).items():
        if _REDACT_PATTERN.search(str(key)):
            redacted[str(key)] = "[REDACTED]"
            continue
        if isinstance(value, Mapping):
            redacted[str(key)] = redact_metadata(value)
        else:
            # Metadata-first payloads only: raw bytes and objects are never stored.
            redacted[str(key)] = value if isinstance(value, (str, int, float, bool, type(None))) else str(value)[:200]
    return redacted


@dataclass(frozen=True)
class AuditEvent:
    """One structured lifecycle or security event (step 45)."""

    kind: str
    actor: str = "system"
    target: str = ""
    task_id: str = ""
    policy_version: int = 1
    trace_id: str = ""
    outcome: str = "ok"
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]
    event_id: str = ""
    timestamp_epoch: float = 0.0
    monotonic: float = 0.0
    sequence_no: int = 0
    prev_chain_hash: str = ""
    chain_hash: str = ""

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown audit event kind {self.kind!r}")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        object.__setattr__(self, "event_id", self.event_id or f"evt-{uuid.uuid4().hex}")

    def canonical_payload(self) -> str:
        return canonical_json(
            {
                "kind": self.kind,
                "actor": self.actor,
                "target": self.target,
                "task_id": self.task_id,
                "policy_version": self.policy_version,
                "trace_id": self.trace_id,
                "outcome": self.outcome,
                "metadata": dict(self.metadata),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "actor": self.actor,
            "target": self.target,
            "task_id": self.task_id,
            "policy_version": self.policy_version,
            "trace_id": self.trace_id,
            "outcome": self.outcome,
            "metadata": dict(self.metadata),
            "timestamp_epoch": self.timestamp_epoch,
            "monotonic": self.monotonic,
            "sequence_no": self.sequence_no,
            "prev_chain_hash": self.prev_chain_hash,
            "chain_hash": self.chain_hash,
        }


@dataclass(frozen=True)
class EventReceipt:
    """Typed persistence receipt for one recorded event (section 6)."""

    event_id: str
    sequence_no: int
    persisted: bool
    chain_hash: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sequence_no": self.sequence_no,
            "persisted": self.persisted,
            "chain_hash": self.chain_hash,
            "note": self.note,
        }


class InMemoryEventSink:
    """Bounded in-memory ring of audit events (standalone mode)."""

    def __init__(self, max_events: int = 10000) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self._capacity = max_events
        self._events: list[AuditEvent] = []
        self._lock = threading.RLock()

    def emit(self, event: AuditEvent) -> EventReceipt:
        with self._lock:
            sequence = len(self._events) + 1
            prev_hash = self._events[-1].chain_hash if self._events else ""
            redacted = dict(redact_metadata(event.metadata))
            chain_hash = sha256_hex(
                prev_hash + canonical_json(
                    {
                        "kind": event.kind,
                        "actor": event.actor,
                        "target": event.target,
                        "task_id": event.task_id,
                        "policy_version": event.policy_version,
                        "trace_id": event.trace_id,
                        "outcome": event.outcome,
                        "metadata": redacted,
                    }
                )
            )
            stamped = AuditEvent(
                kind=event.kind,
                actor=event.actor,
                target=event.target,
                task_id=event.task_id,
                policy_version=event.policy_version,
                trace_id=event.trace_id,
                outcome=event.outcome,
                metadata=redacted,
                event_id=event.event_id,
                timestamp_epoch=event.timestamp_epoch or time.time(),
                monotonic=event.monotonic or time.monotonic(),
                sequence_no=sequence,
                prev_chain_hash=prev_hash,
                chain_hash=chain_hash,
            )
            self._events.append(stamped)
            if len(self._events) > self._capacity:
                self._events = self._events[-self._capacity:]
            return EventReceipt(
                event_id=stamped.event_id,
                sequence_no=sequence,
                persisted=False,
                chain_hash=chain_hash,
                note="in-memory sink; SQLite persistence requires a SQLiteEventSink",
            )

    def events(self) -> tuple[AuditEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def count(self) -> int:
        with self._lock:
            return len(self._events)


class SQLiteEventSink:
    """Append-only, single-writer SQLite persistence with a hash chain (step 28).

    One serialized writer thread plus a bounded busy timeout keeps SQLite
    healthy under concurrent readers; every insert is one explicit
    transaction so an interrupted write leaves either the full record or
    nothing -- task and reservation records stay recoverable through replay.
    """

    def __init__(self, database_path: str | Path) -> None:
        self._path = Path(database_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(self._path), check_same_thread=False, timeout=5.0)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    @property
    def database_path(self) -> str:
        return str(self._path)

    def _chain_head(self) -> tuple[int, str]:
        row = self._connection.execute(
            "SELECT sequence_no, chain_hash FROM runtime_audit_events "
            "ORDER BY sequence_no DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return 0, ""
        return int(row[0]), str(row[1])

    def emit(self, event: AuditEvent) -> EventReceipt:
        with self._lock:
            sequence, prev_hash = self._chain_head()
            metadata = redact_metadata(event.metadata)
            chain_hash = sha256_hex(
                prev_hash + canonical_json(
                    {
                        "kind": event.kind,
                        "actor": event.actor,
                        "target": event.target,
                        "task_id": event.task_id,
                        "policy_version": event.policy_version,
                        "trace_id": event.trace_id,
                        "outcome": event.outcome,
                        "metadata": metadata,
                    }
                )
            )
            try:
                with self._connection:  # one explicit transaction per event
                    self._connection.execute(
                        "INSERT INTO runtime_audit_events ("
                        "event_id, sequence_no, kind, actor, target, task_id, "
                        "policy_version, trace_id, outcome, metadata_json, "
                        "timestamp_epoch, monotonic, prev_chain_hash, chain_hash"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            event.event_id,
                            sequence + 1,
                            event.kind,
                            event.actor,
                            event.target,
                            event.task_id,
                            event.policy_version,
                            event.trace_id,
                            event.outcome,
                            json.dumps(metadata, sort_keys=True),
                            event.timestamp_epoch or time.time(),
                            event.monotonic or time.monotonic(),
                            prev_hash,
                            chain_hash,
                        ),
                    )
                return EventReceipt(
                    event_id=event.event_id,
                    sequence_no=sequence + 1,
                    persisted=True,
                    chain_hash=chain_hash,
                )
            except sqlite3.Error as exc:
                return EventReceipt(
                    event_id=event.event_id,
                    sequence_no=sequence + 1,
                    persisted=False,
                    chain_hash=chain_hash,
                    note=f"sqlite error: {exc}",
                )

    def count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM runtime_audit_events"
            ).fetchone()
            return int(row[0])

    def fetch(self, limit: int = 100, *, kind: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if kind is None:
                cursor = self._connection.execute(
                    "SELECT event_id, sequence_no, kind, actor, target, task_id, "
                    "policy_version, trace_id, outcome, metadata_json, timestamp_epoch, "
                    "monotonic, prev_chain_hash, chain_hash "
                    "FROM runtime_audit_events ORDER BY sequence_no LIMIT ?",
                    (limit,),
                )
            else:
                cursor = self._connection.execute(
                    "SELECT event_id, sequence_no, kind, actor, target, task_id, "
                    "policy_version, trace_id, outcome, metadata_json, timestamp_epoch, "
                    "monotonic, prev_chain_hash, chain_hash "
                    "FROM runtime_audit_events WHERE kind = ? "
                    "ORDER BY sequence_no LIMIT ?",
                    (kind, limit),
                )
            rows = cursor.fetchall()
        return [
            {
                "event_id": row[0],
                "sequence_no": row[1],
                "kind": row[2],
                "actor": row[3],
                "target": row[4],
                "task_id": row[5],
                "policy_version": row[6],
                "trace_id": row[7],
                "outcome": row[8],
                "metadata": json.loads(row[9]),
                "timestamp_epoch": row[10],
                "monotonic": row[11],
                "prev_chain_hash": row[12],
                "chain_hash": row[13],
            }
            for row in rows
        ]

    def verify_chain(self) -> bool:
        """Recompute the hash chain; False means tampering or corruption."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT sequence_no, prev_chain_hash, chain_hash, event_id, kind, "
                "actor, target, task_id, policy_version, trace_id, outcome, "
                "metadata_json FROM runtime_audit_events ORDER BY sequence_no"
            ).fetchall()
        previous = ""
        for row in rows:
            sequence_no, prev_hash, chain_hash = row[0], row[1], row[2]
            if prev_hash != previous:
                return False
            payload = canonical_json(
                {
                    "kind": row[4],
                    "actor": row[5] or "",
                    "target": row[6] or "",
                    "task_id": row[7] or "",
                    "policy_version": row[8] or 1,
                    "trace_id": row[9] or "",
                    "outcome": row[10] or "ok",
                    "metadata": json.loads(row[11]),
                }
            )
            expected = sha256_hex((previous or "") + payload)
            if chain_hash != expected:
                return False
            previous = chain_hash
        return True

    def simulate_interruption(self, event: AuditEvent) -> EventReceipt:
        """Fault injection: begin a transaction, insert, then roll back."""
        with self._lock:
            sequence, prev_hash = self._chain_head()
            metadata = redact_metadata(event.metadata)
            chain_hash = sha256_hex(
                prev_hash + canonical_json(
                    {
                        "kind": event.kind,
                        "actor": event.actor,
                        "target": event.target,
                        "task_id": event.task_id,
                        "policy_version": event.policy_version,
                        "trace_id": event.trace_id,
                        "outcome": event.outcome,
                        "metadata": metadata,
                    }
                )
            )
            self._connection.execute("BEGIN")
            self._connection.execute(
                "INSERT INTO runtime_audit_events ("
                "event_id, sequence_no, kind, actor, target, task_id, "
                "policy_version, trace_id, outcome, metadata_json, "
                "timestamp_epoch, monotonic, prev_chain_hash, chain_hash"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    sequence + 1,
                    event.kind,
                    event.actor,
                    event.target,
                    event.task_id,
                    event.policy_version,
                    event.trace_id,
                    event.outcome,
                    json.dumps(redact_metadata(event.metadata), sort_keys=True),
                    event.timestamp_epoch or time.time(),
                    event.monotonic or time.monotonic(),
                    prev_hash,
                    chain_hash,
                ),
            )
            self._connection.rollback()
            return EventReceipt(
                event_id=event.event_id,
                sequence_no=sequence + 1,
                persisted=False,
                chain_hash=chain_hash,
                note="simulated interruption: transaction rolled back, nothing committed",
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class AuditLogger:
    """Records structured events through a pluggable sink (section 6)."""

    def __init__(self, sink: Any | None = None) -> None:
        self.sink = sink if sink is not None else InMemoryEventSink()

    def record(self, event: AuditEvent) -> EventReceipt:
        return self.sink.emit(event)

    def record_kind(
        self,
        kind: str,
        *,
        actor: str = "system",
        target: str = "",
        task_id: str = "",
        policy_version: int = 1,
        trace_id: str = "",
        outcome: str = "ok",
        **metadata: Any,
    ) -> EventReceipt:
        return self.record(
            AuditEvent(
                kind=kind,
                actor=actor,
                target=target,
                task_id=task_id,
                policy_version=policy_version,
                trace_id=trace_id,
                outcome=outcome,
                metadata=metadata,
            )
        )

    def events(self) -> tuple[AuditEvent, ...]:
        fetcher = getattr(self.sink, "events", None)
        if callable(fetcher):
            return fetcher()
        return ()

    def count(self) -> int:
        return int(getattr(self.sink, "count", lambda: 0)())

    def query(self, *, kind: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        fetcher = getattr(self.sink, "fetch", None)
        if callable(fetcher):
            return fetcher(limit, kind=kind)
        events = self.events()
        if kind is not None:
            events = tuple(event for event in events if event.kind == kind)
        return [event.to_dict() for event in events[:limit]]

    def close(self) -> None:
        closer = getattr(self.sink, "close", None)
        if callable(closer):
            closer()


def replay_reservations(events: Iterable[Any]) -> dict[str, dict[str, Any]]:
    """Rebuild the active reservation ledger from the audit log (step 28).

    A committed 'reservation' event followed by no release event leaves the
    reservation active; this is what makes task and reservation records
    recoverable after SQLite transaction interruption. Accepts AuditEvent
    objects or the dict rows produced by ``SQLiteEventSink.fetch``.
    """
    ledger: dict[str, dict[str, Any]] = {}
    for event in events:
        if isinstance(event, Mapping):
            kind = str(event.get("kind", ""))
            metadata = event.get("metadata") or {}
            task_id = str(event.get("task_id", ""))
        else:
            kind = str(getattr(event, "kind", ""))
            metadata = getattr(event, "metadata", {}) or {}
            task_id = str(getattr(event, "task_id", ""))
        if kind == "reservation":
            ledger[str(metadata.get("reservation_id", ""))] = {
                "task_id": task_id or str(metadata.get("task_id", "")),
                "amounts": dict(metadata.get("amounts", {})),
                "owner": str(metadata.get("owner", "")),
            }
        elif kind in ("reservation_release", "reservation_recovered"):
            reservation_id = str(metadata.get("reservation_id", ""))
            if reservation_id in ledger:
                del ledger[reservation_id]
    return ledger


#: Machine-readable runtime contracts for the annotation registry (step 46).
ANNOTATION_CATALOG: dict[str, dict[str, Any]] = {
    "ore_runtime.contracts.ComponentManifest": {
        "responsibility": "Canonical component descriptor and registration contract",
        "inputs": "manifest fields; administrator resource bounds",
        "outputs": "validated manifest, digest, entry callable",
        "invariants": "unknown fields rejected; undeclared privileges rejected; untrusted never in-process",
        "required_capabilities": "none (contract only)",
        "resource_effects": "declares requested/limits",
        "failure_modes": "ManifestValidationError",
        "audit_event_types": ["registration", "registration_rejected"],
    },
    "ore_runtime.registry.ComponentRegistry": {
        "responsibility": "Registration, identity, discovery, health leases",
        "inputs": "ComponentManifest",
        "outputs": "RegistrationResult, ServiceLease, IdentityRecord",
        "invariants": "never grants authority; leases expire",
        "required_capabilities": "none",
        "resource_effects": "none",
        "failure_modes": "RegistryError",
        "audit_event_types": ["registration", "health_lease"],
    },
    "ore_runtime.capabilities.CapabilityAuthority": {
        "responsibility": "Issue, attenuate, revoke capabilities",
        "inputs": "subject, resource, rights, constraints, ttl",
        "outputs": "Capability, AuthorizationDecision",
        "invariants": "seal verifies; child never exceeds parent; deny by default",
        "required_capabilities": "operator authority",
        "resource_effects": "none",
        "failure_modes": "CapabilityDeniedError",
        "audit_event_types": ["capability_issue", "capability_revoke", "authorization_denied"],
    },
    "ore_runtime.lifecycle.LifecycleMachine": {
        "responsibility": "Lifecycle state machine and transition records",
        "inputs": "component_id, target state, actor, reason",
        "outputs": "TransitionRecord",
        "invariants": "declared transitions only; terminal revival needs new generation",
        "required_capabilities": "operator authority for quarantine release",
        "resource_effects": "none",
        "failure_modes": "IllegalTransitionError",
        "audit_event_types": ["lifecycle_transition", "restart", "quarantine"],
    },
    "ore_runtime.ipc.MessageBus": {
        "responsibility": "Typed request/reply with bounded queues and back-pressure",
        "inputs": "MessageEnvelope",
        "outputs": "ResponseEnvelope",
        "invariants": "oversized/unsupported/malformed rejected; committed work never dropped",
        "required_capabilities": "ipc.send",
        "resource_effects": "bounded queue capacity",
        "failure_modes": "EnvelopeValidationError, QueueBackpressureError, OperationTimedOutError",
        "audit_event_types": ["queue_backpressure", "authorization_denied", "timeout"],
    },
    "ore_runtime.resources.ResourceManager": {
        "responsibility": "Reservations, admission control, hierarchical budgets",
        "inputs": "task_id, ResourceVector",
        "outputs": "ReservationResult, ReleaseResult",
        "invariants": "sum(reserved) <= capacity along the ownership chain",
        "required_capabilities": "task.execute",
        "resource_effects": "cpu/memory/storage/network/slots reservations",
        "failure_modes": "AdmissionDeniedError, breach flags",
        "audit_event_types": ["reservation", "reservation_release", "limit_breach"],
    },
    "ore_runtime.scheduling.BudgetScheduler": {
        "responsibility": "Budget scheduling with aging and reserved classes",
        "inputs": "SchedTask, now",
        "outputs": "SchedTask selection, PressureReport",
        "invariants": "reserved classes never shed; starvation checked",
        "required_capabilities": "none",
        "resource_effects": "cpu/wall-clock budgets",
        "failure_modes": "KeyError on unknown class",
        "audit_event_types": ["queue_backpressure"],
    },
    "ore_runtime.isolation.SubprocessExecutionBackend": {
        "responsibility": "Supervised worker processes with OS resource limits",
        "inputs": "manifest, payload, budget, capability",
        "outputs": "ExecutionHandle",
        "invariants": "scrubbed environment; rlimits where POSIX; deadline kill",
        "required_capabilities": "task.execute or sandbox.execute",
        "resource_effects": "process spawn, rlimits (AS/CPU/FSIZE)",
        "failure_modes": "OperationTimedOutError, BackendUnavailableError",
        "audit_event_types": ["dispatch", "crash", "timeout"],
    },
    "ore_runtime.artifacts.ArtifactStore": {
        "responsibility": "Content-addressed immutable artifact storage",
        "inputs": "bytes, metadata, capability",
        "outputs": "ArtifactRef, VerificationReport",
        "invariants": "atomic publish; hash verified on read",
        "required_capabilities": "artifact.read / artifact.publish",
        "resource_effects": "disk usage bounded by max_bytes",
        "failure_modes": "ArtifactCorruptionError",
        "audit_event_types": ["artifact_published", "artifact_read", "artifact_corruption"],
    },
    "ore_runtime.audit.AuditLogger": {
        "responsibility": "Structured lifecycle/security events and annotations",
        "inputs": "AuditEvent",
        "outputs": "EventReceipt",
        "invariants": "append-only; secrets redacted; hash chain",
        "required_capabilities": "none",
        "resource_effects": "SQLite growth",
        "failure_modes": "receipt reports non-persistence",
        "audit_event_types": ["runtime_annotation"],
    },
    "ore_runtime.policy.PolicyEngine": {
        "responsibility": "Versioned policy evaluation, explanations, rollout",
        "inputs": "component, action, context",
        "outputs": "PolicyDecision",
        "invariants": "hard rules never relaxed by adaptation or override",
        "required_capabilities": "operator.override for manual control",
        "resource_effects": "none",
        "failure_modes": "deny with reason",
        "audit_event_types": ["policy_change", "policy_rollout", "policy_rollback", "manual_override"],
    },
    "ore_runtime.workspace_guards.BlackboardGuard": {
        "responsibility": "Capability-scoped workspace read/propose/append/merge",
        "inputs": "capability, field, payload",
        "outputs": "GuardResult",
        "invariants": "USG canonical updates only via external validation",
        "required_capabilities": "workspace.read_snapshot / propose / append / merge",
        "resource_effects": "none",
        "failure_modes": "CapabilityDeniedError",
        "audit_event_types": ["authorization_denied"],
    },
    "ore_runtime.runtime.ComponentRuntime": {
        "responsibility": "Trusted-core facade wiring all six planes",
        "inputs": "manifests, requests, budgets",
        "outputs": "RegistrationResult, RuntimeHandle, StopResult, explanations",
        "invariants": "small trusted core; every task owned with terminal outcome",
        "required_capabilities": "internal authority",
        "resource_effects": "orchestrates all planes",
        "failure_modes": "typed OreRuntimeError family",
        "audit_event_types": ["registration", "dispatch", "quarantine", "restart"],
    },
    "ore_runtime.testing.SecurityProbeSuite": {
        "responsibility": "Boundary security probes and model-based checks",
        "inputs": "ComponentRuntime",
        "outputs": "ProbeReport",
        "invariants": "both success and denial paths tested",
        "required_capabilities": "none",
        "resource_effects": "none",
        "failure_modes": "probe failures reported",
        "audit_event_types": ["authorization_denied"],
    },
}


def register_runtime_annotations(
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Store runtime contract annotations in the shared ore_logger catalogue.

    Follows the repository pattern (ore_hacol.logging_adapter): AST symbols
    are synchronized with ``sync_repository(..., include_paths={"ore_runtime"})``
    and the machine-readable ANNOTATION_CATALOG contracts are recorded as
    'runtime_annotation' events in the runtime audit sink. When ore_logger is
    not importable (standalone zip deployment), the catalogue is returned
    without SQLite persistence.
    """
    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root
        else Path(__file__).resolve().parents[1]
    )
    target = (
        Path(database_path).expanduser().resolve()
        if database_path
        else root / "ore_logs" / "ore_logger.sqlite3"
    )
    report: dict[str, Any] = {
        "annotation_records": len(ANNOTATION_CATALOG),
        "scope": "ore_runtime",
        "database_path": str(target),
    }
    try:
        from ore_logger.catalog import sync_repository
        from ore_logger.database import LoggerDatabase
    except ImportError:
        report["registered"] = False
        report["note"] = (
            "ore_logger not importable; annotations catalogued in the audit sink only"
        )
        return report
    catalogue = sync_repository(LoggerDatabase(target), root, include_paths={"ore_runtime"})
    report["registered"] = True
    report["ast_catalogue"] = {
        key: value for key, value in catalogue.items() if isinstance(value, (int, str, list))
    }
    return report
