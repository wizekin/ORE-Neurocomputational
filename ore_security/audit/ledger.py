"""Audit ledger: hash-chained security events with signed, anchored checkpoints.

The ledger chains every accepted event (``h_i = H(h_{i-1} || canonical(e_i))``),
persists event metadata to SQLite (never secrets), and produces signed checkpoints
anchored in a separate trust domain. Backup exports carry events plus checkpoints so
restoration can prove history integrity end to end.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..exceptions import AuditError, IntegrityError
from ..integrity.chain import HashChain
from ..integrity.checkpoints import CheckpointService, ExternalAnchorVault
from ..integrity.merkle import MerkleTree
from .events import RedactionEngine, SecurityEvent

Clock = Callable[[], float]

AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS security_events (
    sequence INTEGER PRIMARY KEY,
    event_id TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    correlation_id TEXT NOT NULL,
    principal_id TEXT,
    action TEXT NOT NULL,
    resource_ref TEXT,
    policy_decision TEXT,
    decision_reason TEXT,
    credential_id TEXT,
    key_id TEXT,
    algorithm TEXT,
    result TEXT NOT NULL,
    error_category TEXT,
    duration_ms REAL,
    chain_digest TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_security_events_correlation
    ON security_events(correlation_id);
CREATE TABLE IF NOT EXISTS security_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    range_start INTEGER NOT NULL,
    range_end INTEGER NOT NULL,
    chain_head TEXT NOT NULL,
    merkle_root TEXT NOT NULL,
    tree_version TEXT NOT NULL,
    signer_key_id TEXT NOT NULL,
    signature TEXT NOT NULL,
    timestamp REAL NOT NULL,
    anchor_id TEXT
);
"""


class AuditLedger:
    """Tamper-evident store for security events with checkpointed batches."""

    def __init__(self, *, redaction_engine: RedactionEngine | None = None,
                 checkpoint_service: CheckpointService | None = None,
                 anchor_vault: ExternalAnchorVault | None = None,
                 sqlite_path: str | Path | None = None,
                 clock: Clock = time.time) -> None:
        self.clock = clock
        self.redaction_engine = redaction_engine or RedactionEngine()
        self.checkpoint_service = checkpoint_service
        self.anchor_vault = anchor_vault
        self.chain = HashChain()
        self._sqlite_path = Path(sqlite_path) if sqlite_path is not None else None
        self._lock = threading.RLock()
        self._checkpoints: list[dict[str, Any]] = []
        self._anchor_ids: dict[str, str] = {}
        if self._sqlite_path is not None:
            self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self._sqlite_path) as connection:
                connection.executescript(AUDIT_SCHEMA)

    def record(self, event: SecurityEvent) -> dict[str, Any]:
        """Redact, chain and persist one security event."""
        if not isinstance(event, SecurityEvent):
            raise TypeError("event must be a SecurityEvent")
        sanitized_details = self.redaction_engine.sanitize(event.details)
        with self._lock:
            record = self.chain.append({
                "action": event.action,
                "result": event.result,
                "correlation_id": event.correlation_id,
                "timestamp": event.timestamp,
                "principal_id": event.principal_id,
                "resource_ref": event.resource_ref,
                "policy_decision": event.policy_decision,
                "decision_reason": event.decision_reason,
                "credential_id": event.credential_id,
                "key_id": event.key_id,
                "algorithm": event.algorithm,
                "error_category": event.error_category,
                "duration_ms": event.duration_ms,
                "details": sanitized_details,
            })
            if self._sqlite_path is not None:
                self._persist(record)
            return {
                "sequence": record["sequence"],
                "chain_digest": record["digest"],
                "recorded_at": self.clock(),
            }

    def _persist(self, record: dict[str, Any]) -> None:
        event = record["event"]
        with sqlite3.connect(self._sqlite_path) as connection:
            connection.execute(
                """INSERT INTO security_events (
                       sequence, event_id, timestamp_utc, correlation_id, principal_id,
                       action, resource_ref, policy_decision, decision_reason,
                       credential_id, key_id, algorithm, result, error_category,
                       duration_ms, chain_digest, details_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["sequence"],
                    "sev-" + record["digest"][:16],
                    _iso(event["timestamp"]),
                    event["correlation_id"],
                    event.get("principal_id", ""),
                    event["action"],
                    event.get("resource_ref", ""),
                    event.get("policy_decision", ""),
                    event.get("decision_reason", ""),
                    event.get("credential_id", ""),
                    event.get("key_id", ""),
                    event.get("algorithm", ""),
                    event["result"],
                    event.get("error_category", ""),
                    event.get("duration_ms", 0.0),
                    record["digest"],
                    json.dumps(event.get("details", {}), sort_keys=True),
                ),
            )

    def create_checkpoint(self) -> dict[str, Any]:
        """Sign and anchor a checkpoint over the full current history."""
        if self.checkpoint_service is None:
            raise AuditError("no checkpoint service configured for this ledger")
        entries = list(self.chain.entries())
        if not entries:
            raise AuditError("cannot checkpoint an empty ledger")
        checkpoint = self.checkpoint_service.create_checkpoint(self.chain, entries)
        anchor_id = None
        if self.anchor_vault is not None:
            anchor_id = self.anchor_vault.anchor(checkpoint)
        info = {
            "checkpoint": checkpoint.to_dict(),
            "chain_head": checkpoint.chain_head,
            "merkle_root": checkpoint.merkle_root,
            "range": [checkpoint.range_start, checkpoint.range_end],
            "anchor_id": anchor_id,
        }
        self._checkpoints.append(info)
        return info

    def verify_history(self) -> dict[str, Any]:
        """Verify the chain and the most recent checkpoint head."""
        result = self.chain.verify()
        if self._checkpoints:
            latest = self._checkpoints[-1]
            self.chain.verify(expected_head=latest["chain_head"])
            result["checkpoint_verified"] = True
            result["checkpoint_range"] = latest["range"]
        return result

    def query_correlation(self, correlation_id: str) -> list[dict[str, Any]]:
        """Return all events sharing a correlation id (SQLite when configured)."""
        if self._sqlite_path is not None:
            with sqlite3.connect(self._sqlite_path) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(
                    "SELECT * FROM security_events WHERE correlation_id = ? ORDER BY sequence",
                    (correlation_id,),
                ).fetchall()
                return [dict(row) for row in rows]
        return [record for record in self.chain.entries()
                if record["event"].get("correlation_id") == correlation_id]

    def export_backup(self) -> dict[str, Any]:
        """Export chain records and checkpoints for restoration tests."""
        return {
            "schema_version": "1.0.0",
            "entries": [dict(record) for record in self.chain.entries()],
            "checkpoints": [dict(item) for item in self._checkpoints],
        }

    @classmethod
    def restore_backup(cls, backup: dict[str, Any], *, clock: Clock = time.time) -> "AuditLedger":
        """Restore a ledger from a backup and verify the restored history."""
        ledger = cls(clock=clock)
        ledger.chain = HashChain.from_entries(backup["entries"])
        ledger._checkpoints = [dict(item) for item in backup.get("checkpoints", [])]
        verification = ledger.verify_history()
        if not verification.get("valid"):
            raise IntegrityError("restored audit history failed verification")
        return ledger

    def event_count(self) -> int:
        return self.chain.length

    def checkpoint_count(self) -> int:
        return len(self._checkpoints)

    def sqlite_integrity_report(self) -> dict[str, Any]:
        """Report persisted-event counts and confirm no secret fields exist in storage."""
        if self._sqlite_path is None:
            return {"sqlite_configured": False}
        with sqlite3.connect(self._sqlite_path) as connection:
            count = connection.execute("SELECT COUNT(*) FROM security_events").fetchone()[0]
            columns = [row[1] for row in connection.execute("PRAGMA table_info(security_events)")]
        forbidden = [column for column in columns
                     if any(marker in column.lower() for marker in
                            ("private", "secret", "passphrase", "password", "material"))]
        return {
            "sqlite_configured": True,
            "path": str(self._sqlite_path),
            "persisted_events": count,
            "chain_events": self.chain.length,
            "forbidden_columns_present": forbidden,
            "secrets_in_storage": bool(forbidden),
        }


def _iso(timestamp: float) -> str:
    from ..contracts import utc_timestamp
    return utc_timestamp(timestamp)
