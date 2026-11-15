"""SQLite storage for ORE runs, execution events, and symbol annotations."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS annotations (
    symbol_id TEXT PRIMARY KEY,
    module_name TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    symbol_kind TEXT NOT NULL,
    source_path TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    signature TEXT,
    annotation TEXT NOT NULL,
    source_docstring TEXT,
    annotation_source TEXT NOT NULL DEFAULT 'source_analysis',
    source_hash TEXT,
    stale INTEGER NOT NULL DEFAULT 0,
    updated_at_utc TEXT NOT NULL,
    UNIQUE(module_name, qualified_name, symbol_kind, source_path)
);
CREATE INDEX IF NOT EXISTS idx_annotations_module_qualname
    ON annotations(module_name, qualified_name, symbol_kind);
CREATE INDEX IF NOT EXISTS idx_annotations_source
    ON annotations(source_path, line_start);
CREATE TABLE IF NOT EXISTS source_files (
    source_path TEXT PRIMARY KEY,
    mtime_ns INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    indexed_at_utc TEXT NOT NULL,
    parse_error TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    display_label TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    started_at_local TEXT NOT NULL,
    ended_at_utc TEXT,
    duration_ms REAL,
    script_path TEXT NOT NULL,
    module_name TEXT,
    entry_kind TEXT NOT NULL,
    working_directory TEXT,
    process_id INTEGER,
    python_version TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    event_count INTEGER NOT NULL DEFAULT 0,
    error_summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at_utc DESC);
CREATE INDEX IF NOT EXISTS idx_runs_script ON runs(script_path, started_at_utc DESC);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    sequence_no INTEGER NOT NULL,
    timestamp_utc TEXT NOT NULL,
    elapsed_ms REAL NOT NULL,
    event_kind TEXT NOT NULL,
    symbol_id TEXT REFERENCES annotations(symbol_id),
    class_symbol_id TEXT REFERENCES annotations(symbol_id),
    source_path TEXT,
    line_number INTEGER,
    duration_ms REAL,
    thread_id INTEGER,
    thread_name TEXT,
    exception_type TEXT,
    exception_message TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(run_id, sequence_no)
);
CREATE INDEX IF NOT EXISTS idx_events_run_sequence ON events(run_id, sequence_no);
CREATE INDEX IF NOT EXISTS idx_events_symbol ON events(symbol_id);
CREATE TABLE IF NOT EXISTS settings (
    setting_key TEXT PRIMARY KEY,
    setting_value TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_run_identity() -> tuple[str, str, str, str]:
    now = datetime.now().astimezone()
    utc = now.astimezone(timezone.utc)
    run_id = uuid.uuid4().hex
    local_label = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    display_label = f"{local_label} · {run_id[:8]}"
    return run_id, utc.isoformat(timespec="milliseconds"), now.isoformat(timespec="milliseconds"), display_label


class LoggerDatabase:
    """Small SQLite repository. Explanations are stored once in annotations."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript(SCHEMA)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=20.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=20000")
        return connection

    @contextmanager
    def connection(self):
        connection = self.connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create_run(
        self,
        script_path: str,
        module_name: str | None = None,
        entry_kind: str = "script",
        working_directory: str | None = None,
        process_id: int | None = None,
        python_version: str | None = None,
    ) -> str:
        run_id, started_utc, started_local, label = new_run_identity()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO runs
                   (run_id, display_label, started_at_utc, started_at_local,
                    script_path, module_name, entry_kind, working_directory,
                    process_id, python_version, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')""",
                (run_id, label, started_utc, started_local, script_path,
                 module_name, entry_kind, working_directory, process_id, python_version),
            )
        return run_id

    def register_annotation(
        self,
        symbol_id: str,
        module_name: str,
        qualified_name: str,
        symbol_kind: str,
        source_path: str,
        annotation: str,
        line_start: int | None = None,
        line_end: int | None = None,
        signature: str | None = None,
        source_docstring: str | None = None,
        annotation_source: str = "source_analysis",
        source_hash: str | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO annotations
                   (symbol_id, module_name, qualified_name, symbol_kind, source_path,
                    line_start, line_end, signature, annotation, source_docstring,
                    annotation_source, source_hash, stale, updated_at_utc)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                   ON CONFLICT(symbol_id) DO UPDATE SET
                     module_name=excluded.module_name,
                     qualified_name=excluded.qualified_name,
                     symbol_kind=excluded.symbol_kind,
                     source_path=excluded.source_path,
                     line_start=excluded.line_start,
                     line_end=excluded.line_end,
                     signature=excluded.signature,
                     annotation=CASE
                       WHEN annotations.annotation_source='manual'
                       THEN annotations.annotation ELSE excluded.annotation END,
                     source_docstring=excluded.source_docstring,
                     source_hash=excluded.source_hash,
                     stale=CASE
                       WHEN annotations.annotation_source='manual'
                       AND annotations.source_hash IS NOT excluded.source_hash
                       THEN 1 ELSE 0 END,
                     updated_at_utc=excluded.updated_at_utc""",
                (symbol_id, module_name, qualified_name, symbol_kind, source_path,
                 line_start, line_end, signature, annotation, source_docstring,
                 annotation_source, source_hash, utc_now()),
            )

    def get_annotation(
        self, module_name: str, qualified_name: str,
        symbol_kind: str = "function", source_path: str | None = None,
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            if source_path:
                row = connection.execute(
                    """SELECT * FROM annotations
                       WHERE module_name=? AND qualified_name=? AND symbol_kind=?
                         AND source_path=?""",
                    (module_name, qualified_name, symbol_kind, source_path),
                ).fetchone()
            else:
                row = connection.execute(
                    """SELECT * FROM annotations
                       WHERE module_name=? AND qualified_name=? AND symbol_kind=?
                       ORDER BY stale ASC, source_path LIMIT 1""",
                    (module_name, qualified_name, symbol_kind),
                ).fetchone()
        return dict(row) if row else None

    def get_annotation_by_id(self, symbol_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM annotations WHERE symbol_id=?", (symbol_id,)
            ).fetchone()
        return dict(row) if row else None

    def save_manual_annotation(self, symbol_id: str, annotation: str) -> None:
        if not annotation.strip():
            raise ValueError("Annotation cannot be empty.")
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE annotations SET annotation=?, annotation_source='manual',
                   stale=0, updated_at_utc=? WHERE symbol_id=?""",
                (annotation.strip(), utc_now(), symbol_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown annotation: {symbol_id}")

    def record_event(
        self,
        run_id: str,
        sequence_no: int,
        event_kind: str,
        symbol_id: str | None = None,
        class_symbol_id: str | None = None,
        source_path: str | None = None,
        line_number: int | None = None,
        elapsed_ms: float = 0.0,
        duration_ms: float | None = None,
        thread_id: int | None = None,
        thread_name: str | None = None,
        exception_type: str | None = None,
        exception_message: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        timestamp_utc: str | None = None,
    ) -> dict[str, Any]:
        timestamp = timestamp_utc or utc_now()
        metadata_json = json.dumps(dict(metadata or {}), ensure_ascii=False, default=str)
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO events
                   (run_id, sequence_no, timestamp_utc, elapsed_ms, event_kind,
                    symbol_id, class_symbol_id, source_path, line_number, duration_ms,
                    thread_id, thread_name, exception_type, exception_message, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, sequence_no, timestamp, elapsed_ms, event_kind, symbol_id,
                 class_symbol_id, source_path, line_number, duration_ms, thread_id,
                 thread_name, exception_type, exception_message, metadata_json),
            )
            connection.execute(
                "UPDATE runs SET event_count=event_count+1 WHERE run_id=?", (run_id,)
            )
            symbol = None
            class_symbol = None
            if symbol_id:
                row = connection.execute(
                    "SELECT * FROM annotations WHERE symbol_id=?", (symbol_id,)
                ).fetchone()
                symbol = dict(row) if row else None
            if class_symbol_id:
                row = connection.execute(
                    "SELECT * FROM annotations WHERE symbol_id=?", (class_symbol_id,)
                ).fetchone()
                class_symbol = dict(row) if row else None
        return {
            "run_id": run_id, "sequence": sequence_no, "timestamp_utc": timestamp,
            "elapsed_ms": elapsed_ms, "kind": event_kind,
            "symbol": symbol, "class_symbol": class_symbol,
            "source_path": source_path, "line": line_number,
            "duration_ms": duration_ms, "thread_name": thread_name,
            "exception_type": exception_type, "exception_message": exception_message,
            "details": dict(metadata or {}),
        }

    def finish_run(
        self, run_id: str, status: str, duration_ms: float | None = None,
        error_summary: str | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """UPDATE runs SET ended_at_utc=?, duration_ms=?, status=?,
                   error_summary=? WHERE run_id=?""",
                (utc_now(), duration_ms, status, error_summary, run_id),
            )

    def list_runs(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY started_at_utc DESC LIMIT ?",
                (max(1, min(int(limit), 5000)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_events(self, run_id: str, limit: int = 50000) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT e.*, a.module_name, a.qualified_name, a.symbol_kind,
                          a.annotation, a.annotation_source, a.stale,
                          c.qualified_name AS class_qualified_name,
                          c.annotation AS class_annotation
                   FROM events e
                   LEFT JOIN annotations a ON a.symbol_id=e.symbol_id
                   LEFT JOIN annotations c ON c.symbol_id=e.class_symbol_id
                   WHERE e.run_id=? ORDER BY e.sequence_no LIMIT ?""",
                (run_id, max(1, min(int(limit), 250000))),
            ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            try:
                item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            except json.JSONDecodeError:
                item["metadata"] = {}
            output.append(item)
        return output

    def search_annotations(self, query: str = "", limit: int = 1000) -> list[dict[str, Any]]:
        with self.connect() as connection:
            if query.strip():
                pattern = f"%{query.strip()}%"
                rows = connection.execute(
                    """SELECT * FROM annotations
                       WHERE module_name LIKE ? OR qualified_name LIKE ?
                          OR source_path LIKE ? OR annotation LIKE ?
                       ORDER BY source_path, line_start LIMIT ?""",
                    (pattern, pattern, pattern, pattern, max(1, min(limit, 10000))),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM annotations ORDER BY source_path, line_start LIMIT ?",
                    (max(1, min(limit, 10000)),),
                ).fetchall()
        return [dict(row) for row in rows]

    def summary(self) -> dict[str, int]:
        with self.connect() as connection:
            runs = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            events = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            symbols = connection.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
            classes = connection.execute(
                "SELECT COUNT(*) FROM annotations WHERE symbol_kind='class'"
            ).fetchone()[0]
            functions = connection.execute(
                "SELECT COUNT(*) FROM annotations WHERE symbol_kind IN ('function','async_function')"
            ).fetchone()[0]
            size = self.path.stat().st_size if self.path.exists() else 0
        return {
            "runs": int(runs), "events": int(events), "symbols": int(symbols),
            "classes": int(classes), "functions": int(functions), "database_bytes": int(size),
        }

    def export_run(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        return {"run": run, "events": self.list_events(run_id)}
