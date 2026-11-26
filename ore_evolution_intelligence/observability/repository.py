"""SQLite persistence for evolution runs, candidates, evaluations and events.

Immutable run metadata, lineage, raw metrics, decisions and deployment events are
recorded in an append-only schema (section 5: SQLite / ore_logger row of the
integration map; section 9: broken-lineage safeguard). The repository is intentionally
separate from ore_logger's run/event schema so full evaluation payloads (raw component
metrics, step 17) survive independent of the logger's retention policy.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evolution_runs (
    run_id TEXT PRIMARY KEY,
    objective TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    started_at_utc TEXT NOT NULL,
    finished_at_utc TEXT
);
CREATE TABLE IF NOT EXISTS evolution_candidates (
    candidate_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    state TEXT NOT NULL,
    version INTEGER NOT NULL,
    lineage_json TEXT NOT NULL,
    genome_json TEXT NOT NULL,
    PRIMARY KEY (run_id, candidate_id)
);
CREATE TABLE IF NOT EXISTS evolution_evaluations (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    split TEXT NOT NULL,
    benchmark_version TEXT NOT NULL,
    feasible INTEGER NOT NULL,
    fitness REAL,
    metrics_json TEXT NOT NULL,
    recorded_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evolution_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    occurred_at_utc TEXT NOT NULL
);
"""


class SqliteEvolutionRepository:
    """Append-only SQLite repository for evolution experiments."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    # --------------------------------------------------------------------- runs
    def save_run(self, run_id: str, objective: str, *, started_at_utc: str) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO evolution_runs "
            "(run_id, objective, status, started_at_utc) VALUES (?, ?, 'running', ?)",
            (run_id, objective, started_at_utc),
        )
        self._connection.commit()

    def finish_run(self, run_id: str, status: str, *, finished_at_utc: str) -> None:
        self._connection.execute(
            "UPDATE evolution_runs SET status = ?, finished_at_utc = ? WHERE run_id = ?",
            (status, finished_at_utc, run_id),
        )
        self._connection.commit()

    def run(self, run_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT * FROM evolution_runs WHERE run_id = ?", (run_id,),
        ).fetchone()
        return dict(row) if row else None

    # --------------------------------------------------------------- candidates
    def save_candidate(
        self,
        candidate: Any,
        *,
        run_id: str,
    ) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO evolution_candidates "
            "(candidate_id, run_id, content_hash, state, version, lineage_json, genome_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                candidate.candidate_id, run_id, candidate.content_hash,
                candidate.state, candidate.version,
                json.dumps(candidate.lineage.as_dict(), sort_keys=True),
                json.dumps(candidate.genome.as_dict(), sort_keys=True),
            ),
        )
        self._connection.commit()

    def candidates_for_run(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT * FROM evolution_candidates WHERE run_id = ? ORDER BY candidate_id",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def candidate_lineage(self, candidate_id: str) -> list[dict[str, Any]]:
        """Reconstruct lineage edges involving one candidate across all runs."""
        rows = self._connection.execute(
            "SELECT * FROM evolution_candidates WHERE candidate_id = ? ORDER BY run_id",
            (candidate_id,),
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            lineage = json.loads(row["lineage_json"])
            result.append({
                "run_id": row["run_id"], "candidate_id": row["candidate_id"],
                "content_hash": row["content_hash"], "state": row["state"],
                "version": row["version"], "parents": lineage.get("parents", []),
                "operation": lineage.get("operation", ""),
            })
        return result

    # -------------------------------------------------------------- evaluations
    def save_evaluation(
        self,
        evaluation: Any,
        *,
        run_id: str,
    ) -> None:
        self._connection.execute(
            "INSERT INTO evolution_evaluations "
            "(run_id, candidate_id, split, benchmark_version, feasible, fitness, "
            "metrics_json, recorded_at_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, evaluation.candidate_id, evaluation.split,
                evaluation.benchmark_version,
                1 if evaluation.gate.feasible else 0, evaluation.fitness,
                json.dumps(evaluation.metrics.as_dict(), sort_keys=True),
                evaluation.evaluated_at_utc,
            ),
        )
        self._connection.commit()

    def evaluations_for_run(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT * FROM evolution_evaluations WHERE run_id = ? ORDER BY evaluation_id",
            (run_id,),
        ).fetchall()
        return [
            {
                **dict(row),
                "metrics": json.loads(row["metrics_json"]),
            }
            for row in rows
        ]

    # -------------------------------------------------------------------- events
    def save_event(
        self,
        kind: str,
        run_id: str,
        payload: Mapping[str, Any],
        *,
        occurred_at_utc: str,
    ) -> None:
        self._connection.execute(
            "INSERT INTO evolution_events (run_id, kind, payload_json, occurred_at_utc) "
            "VALUES (?, ?, ?, ?)",
            (run_id, kind, json.dumps(dict(payload), sort_keys=True), occurred_at_utc),
        )
        self._connection.commit()

    def events_for_run(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT * FROM evolution_events WHERE run_id = ? ORDER BY event_id",
            (run_id,),
        ).fetchall()
        return [
            {
                "kind": row["kind"], "run_id": row["run_id"],
                "payload": json.loads(row["payload_json"]),
                "occurred_at_utc": row["occurred_at_utc"],
            }
            for row in rows
        ]

    # --------------------------------------------------------------------- utils
    def table_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for table in ("evolution_runs", "evolution_candidates",
                      "evolution_evaluations", "evolution_events"):
            row = self._connection.execute(
                f"SELECT COUNT(*) AS n FROM {table}",
            ).fetchone()
            counts[table] = int(row["n"])
        return counts

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SqliteEvolutionRepository":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = ["SqliteEvolutionRepository"]
