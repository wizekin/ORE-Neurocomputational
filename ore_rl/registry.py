"""Policy registry: versioning, promotion, rollback, and retirement (Steps 5, 51-52).

A policy is registered with full metadata (Step 5), promoted only after a
predeclared-threshold evaluation and an approval record (Step 51), and
monitored in production: reward drift, task-mix drift, constraint-rejection
rates, out-of-distribution states, and lost provenance trigger freezing or
retirement with the previous validated version preserved (Step 52). Every
state change is appended to an audit table.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .contracts import stable_hash
from .evaluation import EvaluationReport, PromotionThresholds

REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS rl_policy_registry (
    policy_key TEXT PRIMARY KEY,
    policy_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    algorithm_family TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    status TEXT NOT NULL,
    evaluation_fingerprint TEXT,
    promotion_json TEXT,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rl_promotion_audit (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_key TEXT NOT NULL,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT NOT NULL,
    details_json TEXT NOT NULL,
    recorded_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rl_monitoring (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_key TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    mean_reward REAL,
    success_rate REAL,
    constraint_rejection_rate REAL,
    out_of_distribution_rate REAL,
    task_mix_json TEXT,
    notes TEXT
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class PolicyStatus(str, Enum):
    """Lifecycle states of a registered policy (Step 51-52)."""

    CANDIDATE = "CANDIDATE"
    SHADOW = "SHADOW"
    ACTIVE = "ACTIVE"
    FROZEN = "FROZEN"
    RETIRED = "RETIRED"


@dataclass(frozen=True, slots=True)
class PromotionApproval:
    """Human approval record required for promotion (Step 51)."""

    approver: str
    approved_at: str
    justification: str
    evaluation_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "approver": self.approver,
            "approved_at": self.approved_at,
            "justification": self.justification,
            "evaluation_fingerprint": self.evaluation_fingerprint,
        }


@dataclass(slots=True)
class PolicyRecord:
    """One registered policy version and its lifecycle state."""

    policy_key: str
    policy_id: str
    policy_version: str
    algorithm_family: str
    metadata_json: str
    status: PolicyStatus = PolicyStatus.CANDIDATE
    evaluation_fingerprint: str | None = None
    promotion: PromotionApproval | None = None
    created_at: str = ""
    updated_at: str = ""

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_json)


class PolicyRegistry:
    """SQLite-backed registry with promotion gating and rollback (Steps 51-52)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(REGISTRY_SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> PolicyRegistry:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- registration ------------------------------------------------------
    def register(
        self,
        *,
        policy_id: str,
        policy_version: str,
        algorithm_family: str,
        metadata: Mapping[str, Any],
        initial_status: PolicyStatus = PolicyStatus.CANDIDATE,
    ) -> PolicyRecord:
        """Register a candidate; training success alone never promotes it (Step 5)."""
        policy_key = f"{policy_id}@{policy_version}"
        metadata_json = json.dumps(dict(metadata), sort_keys=True)
        now = _utc_now()
        with self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO rl_policy_registry (policy_key, policy_id, "
                "policy_version, algorithm_family, metadata_json, status, "
                "evaluation_fingerprint, promotion_json, created_at_utc, updated_at_utc) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)",
                (
                    policy_key,
                    policy_id,
                    policy_version,
                    algorithm_family,
                    metadata_json,
                    initial_status.value,
                    now,
                    now,
                ),
            )
        self._audit(policy_key, "REGISTER", "system", "initial registration", {"status": initial_status.value})
        return self.get(policy_key)

    def get(self, policy_key: str) -> PolicyRecord:
        row = self._connection.execute(
            "SELECT * FROM rl_policy_registry WHERE policy_key = ?", (policy_key,)
        ).fetchone()
        if row is None:
            raise KeyError(f"policy not registered: {policy_key}")
        promotion = None
        if row["promotion_json"]:
            promotion = PromotionApproval(**json.loads(row["promotion_json"]))
        return PolicyRecord(
            policy_key=policy_key,
            policy_id=row["policy_id"],
            policy_version=row["policy_version"],
            algorithm_family=row["algorithm_family"],
            metadata_json=row["metadata_json"],
            status=PolicyStatus(row["status"]),
            evaluation_fingerprint=row["evaluation_fingerprint"],
            promotion=promotion,
            created_at=row["created_at_utc"],
            updated_at=row["updated_at_utc"],
        )

    def list_records(self, *, status: PolicyStatus | None = None) -> tuple[PolicyRecord, ...]:
        query = "SELECT policy_key FROM rl_policy_registry"
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status.value,)
        keys = [row["policy_key"] for row in self._connection.execute(query, params)]
        return tuple(self.get(key) for key in keys)

    def active_policy(self, policy_id: str) -> PolicyRecord | None:
        rows = self._connection.execute(
            "SELECT policy_key FROM rl_policy_registry WHERE policy_id = ? AND status = 'ACTIVE'",
            (policy_id,),
        ).fetchall()
        return self.get(rows[0]["policy_key"]) if rows else None

    # -- promotion and rollback -----------------------------------------------
    def promote(
        self,
        policy_key: str,
        evaluation: EvaluationReport,
        approval: PromotionApproval,
        *,
        thresholds: PromotionThresholds | None = None,
    ) -> PolicyRecord:
        """Promote only when predeclared thresholds are met (8.3, Step 51)."""
        thresholds = thresholds or evaluation.thresholds or PromotionThresholds()
        if evaluation.meets_promotion_thresholds is None:
            raise ValueError(
                "evaluation report does not include a promotion verdict; "
                "evaluate with explicit PromotionThresholds"
            )
        if not evaluation.meets_promotion_thresholds:
            self._audit(
                policy_key,
                "PROMOTION_BLOCKED",
                approval.approver,
                "evaluation did not meet predeclared thresholds",
                {
                    "improvement": evaluation.improvement_vs_baseline,
                    "ci95": list(evaluation.improvement_ci95 or ()),
                    "constraint_violations": evaluation.constraint_violations,
                },
            )
            raise PermissionError(
                f"promotion blocked for {policy_key}: predeclared thresholds not met"
            )
        if approval.evaluation_fingerprint != stable_hash(evaluation.to_dict()):
            self._audit(
                policy_key,
                "PROMOTION_BLOCKED",
                approval.approver,
                "approval references a different evaluation",
                {},
            )
            raise PermissionError(
                f"promotion blocked for {policy_key}: approval fingerprint mismatch"
            )
        previous = self.active_policy(self.get(policy_key).policy_id)
        now = _utc_now()
        with self._connection:
            if previous is not None and previous.policy_key != policy_key:
                self._connection.execute(
                    "UPDATE rl_policy_registry SET status = 'RETIRED', updated_at_utc = ? "
                    "WHERE policy_key = ?",
                    (now, previous.policy_key),
                )
            self._connection.execute(
                "UPDATE rl_policy_registry SET status = 'ACTIVE', evaluation_fingerprint = ?, "
                "promotion_json = ?, updated_at_utc = ? WHERE policy_key = ?",
                (
                    stable_hash(evaluation.to_dict()),
                    json.dumps(approval.to_dict(), sort_keys=True),
                    now,
                    policy_key,
                ),
            )
        self._audit(
            policy_key,
            "PROMOTE",
            approval.approver,
            approval.justification,
            {"superseded": previous.policy_key if previous else None},
        )
        return self.get(policy_key)

    def rollback(self, policy_id: str, *, actor: str = "system") -> PolicyRecord:
        """Restore the previous validated version (acceptance test 8.1)."""
        records = self.list_records()
        history = [
            record
            for record in records
            if record.policy_id == policy_id
            and record.status in (PolicyStatus.RETIRED, PolicyStatus.FROZEN)
        ]
        active = self.active_policy(policy_id)
        if active is None:
            raise ValueError(f"no active policy to roll back for {policy_id}")
        if not history:
            raise ValueError(f"no previous validated version to restore for {policy_id}")
        restore = history[-1]
        now = _utc_now()
        with self._connection:
            self._connection.execute(
                "UPDATE rl_policy_registry SET status = 'RETIRED', updated_at_utc = ? "
                "WHERE policy_key = ?",
                (now, active.policy_key),
            )
            self._connection.execute(
                "UPDATE rl_policy_registry SET status = 'ACTIVE', updated_at_utc = ? "
                "WHERE policy_key = ?",
                (now, restore.policy_key),
            )
        self._audit(
            restore.policy_key,
            "ROLLBACK",
            actor,
            f"restored after regression of {active.policy_key}",
            {"retired": active.policy_key},
        )
        return self.get(restore.policy_key)

    def freeze(self, policy_key: str, *, actor: str, reason: str) -> PolicyRecord:
        return self._transition(policy_key, PolicyStatus.FROZEN, actor, reason)

    def retire(self, policy_key: str, *, actor: str, reason: str) -> PolicyRecord:
        return self._transition(policy_key, PolicyStatus.RETIRED, actor, reason)

    def to_shadow(self, policy_key: str, *, actor: str = "system", reason: str = "") -> PolicyRecord:
        return self._transition(policy_key, PolicyStatus.SHADOW, actor, reason or "returned to shadow evaluation")

    def _transition(
        self, policy_key: str, status: PolicyStatus, actor: str, reason: str
    ) -> PolicyRecord:
        now = _utc_now()
        with self._connection:
            self._connection.execute(
                "UPDATE rl_policy_registry SET status = ?, updated_at_utc = ? "
                "WHERE policy_key = ?",
                (status.value, now, policy_key),
            )
        self._audit(policy_key, status.value, actor, reason, {})
        return self.get(policy_key)

    # -- monitoring -----------------------------------------------------------
    def record_monitoring_sample(
        self,
        policy_key: str,
        *,
        mean_reward: float,
        success_rate: float,
        constraint_rejection_rate: float,
        out_of_distribution_rate: float = 0.0,
        task_mix: Mapping[str, float] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """Record a production monitoring sample (Step 52)."""
        with self._connection:
            self._connection.execute(
                "INSERT INTO rl_monitoring (policy_key, timestamp_utc, mean_reward, "
                "success_rate, constraint_rejection_rate, out_of_distribution_rate, "
                "task_mix_json, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    policy_key,
                    _utc_now(),
                    float(mean_reward),
                    float(success_rate),
                    float(constraint_rejection_rate),
                    float(out_of_distribution_rate),
                    json.dumps(dict(task_mix or {}), sort_keys=True),
                    notes,
                ),
            )
        return self.monitoring_summary(policy_key)

    def monitoring_samples(self, policy_key: str) -> tuple[dict[str, Any], ...]:
        rows = self._connection.execute(
            "SELECT * FROM rl_monitoring WHERE policy_key = ? ORDER BY sample_id",
            (policy_key,),
        ).fetchall()
        return tuple(dict(row) for row in rows)

    def monitoring_summary(self, policy_key: str) -> dict[str, Any]:
        samples = self.monitoring_samples(policy_key)
        if not samples:
            return {"policy_key": policy_key, "samples": 0}
        rewards = [sample["mean_reward"] for sample in samples if sample["mean_reward"] is not None]
        drift = rewards[-1] - rewards[0] if len(rewards) >= 2 else 0.0
        latest = samples[-1]
        triggers: list[str] = []
        if drift < -0.25:
            triggers.append("reward_drift")
        if latest["constraint_rejection_rate"] > 0.2:
            triggers.append("constraint_rejections")
        if latest["out_of_distribution_rate"] > 0.3:
            triggers.append("out_of_distribution")
        return {
            "policy_key": policy_key,
            "samples": len(samples),
            "reward_drift": drift,
            "latest_success_rate": latest["success_rate"],
            "latest_constraint_rejection_rate": latest["constraint_rejection_rate"],
            "latest_out_of_distribution_rate": latest["out_of_distribution_rate"],
            "retirement_triggers": triggers,
        }

    def audit_log(self, policy_key: str | None = None) -> tuple[dict[str, Any], ...]:
        query = "SELECT * FROM rl_promotion_audit"
        params: tuple[Any, ...] = ()
        if policy_key:
            query += " WHERE policy_key = ?"
            params = (policy_key,)
        rows = self._connection.execute(query + " ORDER BY audit_id", params).fetchall()
        return tuple(dict(row) for row in rows)

    def _audit(
        self, policy_key: str, action: str, actor: str, reason: str, details: Mapping[str, Any]
    ) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO rl_promotion_audit (policy_key, action, actor, reason, "
                "details_json, recorded_at_utc) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    policy_key,
                    action,
                    actor,
                    reason,
                    json.dumps(dict(details), sort_keys=True),
                    _utc_now(),
                ),
            )

    def counts(self) -> dict[str, int]:
        def one(query: str) -> int:
            return int(self._connection.execute(query).fetchone()[0])

        return {
            "policies": one("SELECT COUNT(*) FROM rl_policy_registry"),
            "audit": one("SELECT COUNT(*) FROM rl_promotion_audit"),
            "monitoring": one("SELECT COUNT(*) FROM rl_monitoring"),
        }
