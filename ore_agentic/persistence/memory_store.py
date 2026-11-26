"""MemoryStore: episodic, procedural and semantic-proposal persistence.

Episodes, procedures with their tests, typed memory links and semantic
proposals persist transactionally in the agentic database, mirroring the
in-memory stores so recovery after restart preserves the three memory tiers
without conflating them.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..contracts import utc_now
from .database import AgenticDatabase

__all__ = ("MemoryStore",)


class MemoryStore:
    """SQLite persistence for the three memory tiers (Steps 38-40)."""

    def __init__(self, database: AgenticDatabase) -> None:
        """Bind the store to the agentic database."""
        self._db = database

    # ---------------------------------------------------------------- episodes

    def save_episode(self, episode: Mapping[str, Any]) -> None:
        """Persist one episodic record (Step 38)."""
        self._db.execute(
            "INSERT OR REPLACE INTO agentic_episodes (episode_id, workflow_id, "
            "task_id, kind, summary, outcome, retention_class, payload_json, "
            "recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (episode["episode_id"], episode.get("workflow_id", ""),
             episode.get("task_id", ""), episode["kind"], episode["summary"],
             episode.get("outcome", ""), episode.get("retention_class", "standard"),
             self._db.dumps(dict(episode.get("payload", {}))),
             episode.get("recorded_at", utc_now())))

    def episodes_for(self, *, workflow_id: str | None = None,
                     task_id: str | None = None) -> list[dict[str, Any]]:
        """Recall episodes by workflow or task (Step 38)."""
        if workflow_id is not None and task_id is not None:
            return self._db.query(
                "SELECT * FROM agentic_episodes WHERE workflow_id = ? AND "
                "task_id = ? ORDER BY recorded_at", (workflow_id, task_id))
        if workflow_id is not None:
            return self._db.query(
                "SELECT * FROM agentic_episodes WHERE workflow_id = ? "
                "ORDER BY recorded_at", (workflow_id,))
        if task_id is not None:
            return self._db.query(
                "SELECT * FROM agentic_episodes WHERE task_id = ? "
                "ORDER BY recorded_at", (task_id,))
        return self._db.query("SELECT * FROM agentic_episodes ORDER BY recorded_at")

    # -------------------------------------------------------------- procedures

    def save_procedure(self, procedure: Mapping[str, Any]) -> None:
        """Persist one procedure record (Step 39)."""
        self._db.execute(
            "INSERT OR REPLACE INTO agentic_procedures (procedure_id, name, version, "
            "status, body_json, created_at, promoted_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (procedure["procedure_id"], procedure["name"], procedure["version"],
             procedure["status"], self._db.dumps(dict(procedure)),
             procedure.get("created_at", utc_now()), procedure.get("promoted_at")))

    def procedures(self, status: str | None = None) -> list[dict[str, Any]]:
        """List procedures, optionally filtered by status (Step 39)."""
        if status is None:
            return self._db.query(
                "SELECT * FROM agentic_procedures ORDER BY created_at")
        return self._db.query(
            "SELECT * FROM agentic_procedures WHERE status = ? ORDER BY created_at",
            (status,))

    def procedure(self, procedure_id: str) -> dict[str, Any] | None:
        """Return one procedure."""
        rows = self._db.query(
            "SELECT * FROM agentic_procedures WHERE procedure_id = ?",
            (procedure_id,))
        return rows[0] if rows else None

    def save_procedure_test(self, test: Mapping[str, Any]) -> None:
        """Persist one procedure test result (Step 39)."""
        self._db.execute(
            "INSERT OR REPLACE INTO agentic_procedure_tests (test_id, procedure_id, "
            "passed, detail, executed_at) VALUES (?, ?, ?, ?, ?)",
            (test["test_id"], test["procedure_id"], int(test["passed"]),
             test.get("detail", ""), test.get("executed_at", utc_now())))

    def procedure_tests(self, procedure_id: str) -> list[dict[str, Any]]:
        """Return the tests of one procedure (Step 39)."""
        return self._db.query(
            "SELECT * FROM agentic_procedure_tests WHERE procedure_id = ? "
            "ORDER BY executed_at", (procedure_id,))

    # -------------------------------------------------------------------- links

    def save_link(self, link: Mapping[str, Any]) -> None:
        """Persist one typed memory link (Step 40)."""
        self._db.execute(
            "INSERT OR REPLACE INTO agentic_memory_links (link_id, source_kind, "
            "source_id, relation, target_kind, target_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (link["link_id"], link["source_kind"], link["source_id"],
             link["relation"], link["target_kind"], link["target_id"],
             link.get("created_at", utc_now())))

    def links_for(self, record_id: str) -> list[dict[str, Any]]:
        """Return links touching a record in either direction (Step 40)."""
        return self._db.query(
            "SELECT * FROM agentic_memory_links WHERE source_id = ? OR target_id = ?",
            (record_id, record_id))

    def links(self) -> list[dict[str, Any]]:
        """Return all links."""
        return self._db.query("SELECT * FROM agentic_memory_links ORDER BY created_at")

    # --------------------------------------------------------------- proposals

    def save_semantic_proposal(self, proposal: Mapping[str, Any]) -> None:
        """Persist one semantic proposal with its lifecycle (Step 37)."""
        self._db.execute(
            "INSERT OR REPLACE INTO agentic_semantic_proposals (proposal_id, "
            "statement, lifecycle, evidence_json, origin_workflow_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (proposal["proposal_id"], proposal["statement"],
             proposal.get("lifecycle", "CANDIDATE"),
             self._db.dumps(list(proposal.get("evidence_refs", ()))),
             proposal.get("origin_workflow_id", ""),
             proposal.get("created_at", utc_now())))

    def semantic_proposals(self, lifecycle: str | None = None
                           ) -> list[dict[str, Any]]:
        """List semantic proposals, optionally by lifecycle (Step 37)."""
        if lifecycle is None:
            return self._db.query(
                "SELECT * FROM agentic_semantic_proposals ORDER BY created_at")
        return self._db.query(
            "SELECT * FROM agentic_semantic_proposals WHERE lifecycle = ? "
            "ORDER BY created_at", (lifecycle,))

    # ------------------------------------------------------------------- a2a

    def save_a2a_record(self, record: Mapping[str, Any]) -> None:
        """Persist one external-agent result record (Step 36)."""
        self._db.execute(
            "INSERT OR REPLACE INTO agentic_a2a_records (message_id, agent_id, "
            "delegation_id, validated, payload_json, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (record["message_id"], record["agent_id"], record["delegation_id"],
             int(record.get("validated", False)),
             self._db.dumps(dict(record.get("payload", {}))), utc_now()))

    def a2a_records(self, delegation_id: str | None = None
                    ) -> list[dict[str, Any]]:
        """List external-agent records (Step 36)."""
        if delegation_id is None:
            return self._db.query("SELECT * FROM agentic_a2a_records "
                                  "ORDER BY recorded_at")
        return self._db.query(
            "SELECT * FROM agentic_a2a_records WHERE delegation_id = ? "
            "ORDER BY recorded_at", (delegation_id,))
