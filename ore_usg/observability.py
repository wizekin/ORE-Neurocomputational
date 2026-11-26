"""SQLite-backed execution events and immutable USG artifact/source lineage."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
import sqlite3
from typing import Any

from ore_logger.database import LoggerDatabase, utc_now

OBSERVABILITY_SCHEMA_VERSION = 1
_SENSITIVE_KEYS = frozenset({
    "payload", "raw_payload", "bundle", "bundle_json", "definition",
    "evidence_text", "source_text", "content", "full_text", "text",
})


def _safe_metadata(value: Any, *, depth: int = 0) -> Any:
    """Bound nested diagnostic values and omit fields likely to contain source text."""
    if depth >= 4:
        return "[depth-limited]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:240]
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        items = list(value.items())
        for key, item in items[:30]:
            name = str(key)[:80]
            normalized = name.casefold()
            if normalized in _SENSITIVE_KEYS or any(
                token in normalized for token in ("raw_payload", "evidence_text", "source_text")
            ):
                continue
            output[name] = _safe_metadata(item, depth=depth + 1)
        if len(items) > 30:
            output["_truncated_fields"] = len(items) - 30
        return output
    if isinstance(value, (list, tuple, set, frozenset)):
        rows = list(value)
        result = [_safe_metadata(item, depth=depth + 1) for item in rows[:20]]
        if len(rows) > 20:
            result.append(f"[{len(rows) - 20} additional item(s) omitted]")
        return result
    return repr(value)[:240]


class USGSQLiteObservability:
    """Persist compact operational metadata in the same SQLite file as ORE annotations."""

    def __init__(self, database: LoggerDatabase | str | Path) -> None:
        self.database = database if isinstance(database, LoggerDatabase) else LoggerDatabase(database)
        self._initialize()

    def _initialize(self) -> None:
        with self.database.connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS module_execution_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    module_id TEXT NOT NULL,
                    module_version TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    run_id TEXT,
                    started_at_utc TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    input_artifact_ids_json TEXT NOT NULL,
                    output_artifact_ids_json TEXT NOT NULL,
                    bundle_id TEXT,
                    bundle_digest TEXT,
                    metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_usg_module_events_run
                    ON module_execution_events(run_id, event_id);
                CREATE INDEX IF NOT EXISTS idx_usg_module_events_bundle
                    ON module_execution_events(bundle_digest, event_id);
                CREATE TABLE IF NOT EXISTS artifact_lineage (
                    artifact_id TEXT PRIMARY KEY,
                    payload_hash TEXT NOT NULL,
                    contract_id TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    producer_module TEXT NOT NULL,
                    producer_version TEXT NOT NULL,
                    run_id TEXT,
                    parent_artifact_ids_json TEXT NOT NULL,
                    source_namespace TEXT,
                    source_bundle_id TEXT,
                    source_bundle_digest TEXT,
                    source_record_ids_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_usg_lineage_bundle
                    ON artifact_lineage(source_bundle_digest);
                CREATE TABLE IF NOT EXISTS usg_migration_events (
                    migration_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    migration_id TEXT NOT NULL,
                    source_contract TEXT NOT NULL,
                    target_contract TEXT NOT NULL,
                    source_digest TEXT NOT NULL,
                    target_digest TEXT NOT NULL,
                    semantic_change_declared INTEGER NOT NULL,
                    recorded_at_utc TEXT NOT NULL
                );
            """)
            connection.execute(
                "INSERT INTO settings(setting_key, setting_value) VALUES (?, ?) "
                "ON CONFLICT(setting_key) DO UPDATE SET setting_value=excluded.setting_value",
                ("usg_observability_schema_version", str(OBSERVABILITY_SCHEMA_VERSION)),
            )

    def record_module_execution(
        self,
        *,
        module_id: str,
        module_version: str,
        operation: str,
        status: str,
        duration_ms: float,
        run_id: str | None = None,
        started_at_utc: str | None = None,
        input_artifact_ids: tuple[str, ...] | list[str] = (),
        output_artifact_ids: tuple[str, ...] | list[str] = (),
        bundle_id: str | None = None,
        bundle_digest: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Append one module execution event without copying source bundle payloads."""
        if not module_id.strip() or not module_version.strip() or not operation.strip():
            raise ValueError("module ID, version, and operation are required")
        if duration_ms < 0:
            raise ValueError("duration_ms cannot be negative")
        safe = _safe_metadata(dict(metadata or {}))
        with self.database.connection() as connection:
            cursor = connection.execute(
                """INSERT INTO module_execution_events
                (module_id,module_version,operation,status,run_id,started_at_utc,duration_ms,
                 input_artifact_ids_json,output_artifact_ids_json,bundle_id,bundle_digest,metadata_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    module_id, module_version, operation, status, run_id,
                    started_at_utc or utc_now(), float(duration_ms),
                    json.dumps(list(input_artifact_ids), ensure_ascii=False),
                    json.dumps(list(output_artifact_ids), ensure_ascii=False),
                    bundle_id, bundle_digest,
                    json.dumps(safe, ensure_ascii=False, sort_keys=True),
                ),
            )
            return int(cursor.lastrowid)

    def record_artifact_lineage(
        self,
        *,
        artifact_id: str,
        payload_hash: str,
        contract_id: str,
        schema_version: str,
        producer_module: str,
        producer_version: str,
        parent_artifact_ids: tuple[str, ...] | list[str] = (),
        run_id: str | None = None,
        source_namespace: str | None = None,
        source_bundle_id: str | None = None,
        source_bundle_digest: str | None = None,
        source_record_ids: tuple[str, ...] | list[str] = (),
    ) -> bool:
        """Record immutable artifact lineage; return False for an identical prior record."""
        required = (artifact_id, payload_hash, contract_id, schema_version, producer_module, producer_version)
        if any(not isinstance(value, str) or not value.strip() for value in required):
            raise ValueError("artifact identity, hash, contract, schema and producer are required")
        parents = tuple(parent_artifact_ids)
        records = tuple(sorted(set(str(item) for item in source_record_ids)))
        expected = (
            payload_hash, contract_id, schema_version, producer_module, producer_version,
            json.dumps(parents), source_namespace, source_bundle_id,
            source_bundle_digest, json.dumps(records),
        )
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM artifact_lineage WHERE artifact_id=?", (artifact_id,)
            ).fetchone()
            if row is not None:
                actual = (
                    row["payload_hash"], row["contract_id"], row["schema_version"],
                    row["producer_module"], row["producer_version"],
                    row["parent_artifact_ids_json"], row["source_namespace"],
                    row["source_bundle_id"], row["source_bundle_digest"],
                    row["source_record_ids_json"],
                )
                if actual != expected:
                    raise ValueError(f"conflicting lineage already exists for artifact {artifact_id}")
                return False
            connection.execute(
                """INSERT INTO artifact_lineage
                (artifact_id,payload_hash,contract_id,schema_version,producer_module,
                 producer_version,run_id,parent_artifact_ids_json,source_namespace,
                 source_bundle_id,source_bundle_digest,source_record_ids_json,created_at_utc)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    artifact_id, payload_hash, contract_id, schema_version, producer_module,
                    producer_version, run_id, json.dumps(parents), source_namespace,
                    source_bundle_id, source_bundle_digest, json.dumps(records), utc_now(),
                ),
            )
        return True

    def record_source_bundle(
        self,
        *,
        artifact_id: str,
        payload_hash: str,
        contract_id: str,
        schema_version: str,
        producer_module: str,
        producer_version: str,
        bundle: Mapping[str, Any],
        run_id: str | None = None,
        bundle_digest: str | None = None,
    ) -> bool:
        """Link a source artifact to all original USG record, relation and evidence IDs."""
        record_ids: set[str] = set()
        for key in ("records", "relations", "mappings", "evidence", "contexts", "sources"):
            rows = bundle.get(key, ())
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, Mapping) and row.get("id") is not None:
                        record_ids.add(str(row["id"]))
        return self.record_artifact_lineage(
            artifact_id=artifact_id,
            payload_hash=payload_hash,
            contract_id=contract_id,
            schema_version=schema_version,
            producer_module=producer_module,
            producer_version=producer_version,
            run_id=run_id,
            source_namespace=str(bundle.get("namespace") or "") or None,
            source_bundle_id=str(bundle.get("bundle_id") or "") or None,
            source_bundle_digest=str(bundle_digest or bundle.get("bundle_digest") or payload_hash) or None,
            source_record_ids=tuple(sorted(record_ids)),
        )

    def record_migration(self, migration_result: Any, migration: Any) -> int:
        """Append the digest transition and semantic-change declaration for one migration."""
        with self.database.connection() as connection:
            cursor = connection.execute(
                """INSERT INTO usg_migration_events
                (migration_id,source_contract,target_contract,source_digest,target_digest,
                 semantic_change_declared,recorded_at_utc)
                VALUES (?,?,?,?,?,?,?)""",
                (
                    migration.migration_id, migration.source_contract, migration.target_contract,
                    migration_result.source_digest, migration_result.target_digest,
                    int(migration.semantic_effect == "DECLARED_CHANGE"), utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def list_module_execution_events(self, *, run_id: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        """Return newest execution events with bounded metadata decoded."""
        with self.database.connection() as connection:
            if run_id:
                rows = connection.execute(
                    "SELECT * FROM module_execution_events WHERE run_id=? ORDER BY event_id DESC LIMIT ?",
                    (run_id, max(1, min(int(limit), 5000))),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM module_execution_events ORDER BY event_id DESC LIMIT ?",
                    (max(1, min(int(limit), 5000)),),
                ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            for field, target in (
                ("input_artifact_ids_json", "input_artifact_ids"),
                ("output_artifact_ids_json", "output_artifact_ids"),
                ("metadata_json", "metadata"),
            ):
                item[target] = json.loads(item.pop(field))
            output.append(item)
        return output

    def get_artifact_lineage(self, artifact_id: str) -> dict[str, Any] | None:
        """Return one immutable lineage record by artifact ID."""
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM artifact_lineage WHERE artifact_id=?", (artifact_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["parent_artifact_ids"] = json.loads(result.pop("parent_artifact_ids_json"))
        result["source_record_ids"] = json.loads(result.pop("source_record_ids_json"))
        return result

    def trace(self, artifact_id: str, *, max_depth: int = 32) -> dict[str, Any]:
        """Trace artifact parents to the source bundle while detecting cycles and depth limits."""
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative")

        def walk(current: str, depth: int, active_path: frozenset[str]) -> dict[str, Any]:
            if current in active_path:
                return {"artifact_id": current, "cycle": True, "parents": []}
            if depth > max_depth:
                return {"artifact_id": current, "truncated": True, "parents": []}
            record = self.get_artifact_lineage(current)
            if record is None:
                return {"artifact_id": current, "recorded": False, "parents": []}
            next_path = active_path | {current}
            return {
                "artifact": record,
                "parents": [
                    walk(parent, depth + 1, next_path)
                    for parent in record["parent_artifact_ids"]
                ],
            }
        return walk(artifact_id, 0, frozenset())


__all__ = ["OBSERVABILITY_SCHEMA_VERSION", "USGSQLiteObservability"]
