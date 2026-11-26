"""SQLite trajectory persistence, validation, quarantine, and datasets.

Point 10, Phase E (Steps 29-33) and Step 49. Stores state-action-outcome
trajectories with policy proposals, coordinator decisions, reward components,
evaluation evidence, and provenance in the shared ore_logger database
convention (a separate database file owned by ore_rl). Records are append-only
with a per-episode integrity hash chain. Trajectories failing completeness
checks are quarantined, never silently repaired. Dataset manifests carry
checksums and task-family/source-level splits to prevent leakage. Simulated
world-model transitions are tagged and kept out of the trusted loaders.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .contracts import (
    ActionOutcome,
    OREAction,
    OREState,
    OREReward,
    stable_hash,
)

if TYPE_CHECKING:  # pragma: no cover
    from .reward import RewardWeights

SCHEMA = """
CREATE TABLE IF NOT EXISTS rl_episodes (
    episode_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    task_family TEXT NOT NULL,
    source_case TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    started_at_utc TEXT NOT NULL,
    ended_at_utc TEXT,
    status TEXT NOT NULL,
    environment_version TEXT NOT NULL,
    run_seed INTEGER NOT NULL DEFAULT 0,
    policy_id TEXT,
    policy_version TEXT,
    provenance_json TEXT NOT NULL DEFAULT '{}',
    integrity_head TEXT
);
CREATE INDEX IF NOT EXISTS idx_rl_episodes_task ON rl_episodes(task_id);
CREATE INDEX IF NOT EXISTS idx_rl_episodes_family ON rl_episodes(task_family);
CREATE TABLE IF NOT EXISTS rl_state_refs (
    state_ref TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    task_family TEXT NOT NULL,
    workspace_id TEXT NOT NULL DEFAULT '',
    workspace_revision INTEGER NOT NULL DEFAULT 0,
    encoder_version TEXT NOT NULL,
    state_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rl_steps (
    step_id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL REFERENCES rl_episodes(episode_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    state_ref TEXT NOT NULL REFERENCES rl_state_refs(state_ref),
    next_state_ref TEXT REFERENCES rl_state_refs(state_ref),
    action_json TEXT NOT NULL,
    action_type TEXT NOT NULL,
    action_id TEXT NOT NULL,
    eligible_action_ids_json TEXT NOT NULL,
    outcome_json TEXT NOT NULL,
    outcome_status TEXT NOT NULL,
    measured_latency_ms REAL,
    measured_cpu_cost REAL,
    measured_memory_mb REAL,
    terminal INTEGER NOT NULL DEFAULT 0,
    origin TEXT NOT NULL DEFAULT 'OBSERVED',
    integrity_hash TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    UNIQUE(episode_id, step_index)
);
CREATE INDEX IF NOT EXISTS idx_rl_steps_episode ON rl_steps(episode_id, step_index);
CREATE TABLE IF NOT EXISTS rl_rewards (
    step_id INTEGER PRIMARY KEY REFERENCES rl_steps(step_id) ON DELETE CASCADE,
    quality_delta REAL NOT NULL DEFAULT 0.0,
    evidence_delta REAL NOT NULL DEFAULT 0.0,
    resource_cost REAL NOT NULL DEFAULT 0.0,
    latency_cost REAL NOT NULL DEFAULT 0.0,
    failure_cost REAL NOT NULL DEFAULT 0.0,
    total_reward REAL NOT NULL,
    weights_version TEXT NOT NULL,
    reward_spec_fingerprint TEXT NOT NULL DEFAULT '',
    evaluator_id TEXT NOT NULL DEFAULT '',
    evaluator_version TEXT NOT NULL DEFAULT '',
    attribution_method TEXT NOT NULL DEFAULT 'STEP',
    attribution_uncertainty REAL NOT NULL DEFAULT 0.0,
    terminal INTEGER NOT NULL DEFAULT 0,
    verified INTEGER NOT NULL DEFAULT 0,
    evidence_refs_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS rl_policy_decisions (
    step_id INTEGER PRIMARY KEY REFERENCES rl_steps(step_id) ON DELETE CASCADE,
    policy_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    proposed_action TEXT NOT NULL,
    proposal_rank_json TEXT NOT NULL DEFAULT '[]',
    propensity REAL,
    uncertainty REAL,
    coordinator_action TEXT,
    adopted INTEGER NOT NULL DEFAULT 0,
    authority TEXT NOT NULL DEFAULT 'PROPOSAL',
    fallback_used INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS rl_evaluations (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    step_id INTEGER REFERENCES rl_steps(step_id) ON DELETE CASCADE,
    episode_id TEXT NOT NULL,
    evaluator_id TEXT NOT NULL,
    evaluator_version TEXT NOT NULL,
    score_json TEXT NOT NULL,
    evidence_refs_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_rl_evaluations_episode ON rl_evaluations(episode_id);
CREATE TABLE IF NOT EXISTS rl_dataset_manifests (
    dataset_id TEXT PRIMARY KEY,
    split TEXT NOT NULL,
    query_fingerprint TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    checksum TEXT NOT NULL,
    definition_json TEXT NOT NULL,
    episode_ids_json TEXT NOT NULL,
    step_count INTEGER NOT NULL,
    created_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rl_quarantine (
    quarantine_id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT,
    step_id INTEGER,
    reason TEXT NOT NULL,
    details_json TEXT NOT NULL,
    quarantined_at_utc TEXT NOT NULL
);
"""


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class SplitRule:
    """Leakage-safe split definitions (Step 32)."""

    BY_TASK_FAMILY = "task_family"
    BY_SOURCE_CASE = "source_case"
    BY_SESSION = "session"
    BY_TIME_PERIOD = "time_period"

    def __init__(self, kind: str, groups: Sequence[str], *, label: str) -> None:
        self.kind = kind
        self.groups = tuple(groups)
        self.label = label

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "groups": list(self.groups), "label": self.label}

    def fingerprint(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class SplitDefinition:
    """One named dataset split with its grouping rule (Step 32)."""

    name: str
    rule: Mapping[str, Any]

    def fingerprint(self) -> str:
        return stable_hash({"name": self.name, "rule": dict(self.rule)})


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Immutable dataset description with checksum (acceptance test 8.2)."""

    dataset_id: str
    split: str
    query_fingerprint: str
    schema_version: str
    checksum: str
    definition: Mapping[str, Any]
    episode_ids: tuple[str, ...]
    step_count: int
    created_at: str


@dataclass(frozen=True, slots=True)
class TransitionTuple:
    """Validated (s, a, r, s', done) replay tuple with eligibility mask (Step 33)."""

    state_features: Mapping[str, float]
    action_id: str
    action_type: str
    reward: float
    next_state_features: Mapping[str, float]
    done: bool
    eligible_action_ids: tuple[str, ...]
    terminal: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_features": dict(self.state_features),
            "action_id": self.action_id,
            "action_type": self.action_type,
            "reward": self.reward,
            "next_state_features": dict(self.next_state_features),
            "done": self.done,
            "eligible_action_ids": list(self.eligible_action_ids),
            "terminal": self.terminal,
        }


@dataclass(frozen=True, slots=True)
class BanditTuple:
    """Context, chosen action, propensity, and reward for bandit training (Step 33)."""

    context_features: Mapping[str, float]
    action_id: str
    propensity: float | None
    reward: float
    task_family: str


class TrajectoryStore:
    """Append-only SQLite trajectory store with validation and dataset tooling."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> TrajectoryStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- episodes ------------------------------------------------------------
    def begin_episode(
        self,
        *,
        episode_id: str,
        task_id: str,
        task_family: str,
        environment_version: str,
        run_seed: int = 0,
        source_case: str = "",
        session_id: str = "",
        policy_id: str | None = None,
        policy_version: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> str:
        with self._connection:
            self._connection.execute(
                "INSERT INTO rl_episodes (episode_id, task_id, task_family, source_case, "
                "session_id, started_at_utc, status, environment_version, run_seed, "
                "policy_id, policy_version, provenance_json, integrity_head) "
                "VALUES (?, ?, ?, ?, ?, ?, 'RUNNING', ?, ?, ?, ?, ?, ?)",
                (
                    episode_id,
                    task_id,
                    task_family,
                    source_case or task_id,
                    session_id,
                    _utc_now(),
                    environment_version,
                    int(run_seed),
                    policy_id,
                    policy_version,
                    json.dumps(dict(provenance or {}), sort_keys=True),
                    "",
                ),
            )
        return episode_id

    def finish_episode(self, episode_id: str, *, status: str) -> None:
        with self._connection:
            self._connection.execute(
                "UPDATE rl_episodes SET ended_at_utc = ?, status = ? WHERE episode_id = ?",
                (_utc_now(), status, episode_id),
            )

    def _integrity_chain(self, episode_id: str) -> str:
        row = self._connection.execute(
            "SELECT integrity_head FROM rl_episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        return row["integrity_head"] if row and row["integrity_head"] else ""

    # -- steps ---------------------------------------------------------------
    def record_step(
        self,
        *,
        episode_id: str,
        step_index: int,
        state: OREState,
        action: OREAction,
        outcome: ActionOutcome,
        next_state: OREState | None,
        terminal: bool,
        reward: OREReward,
        eligible_action_ids: Sequence[str],
        policy_decision: Mapping[str, Any] | None = None,
        coordinator_action: str | None = None,
        adopted: bool = False,
        origin: str = "OBSERVED",
        evaluations: Sequence[Mapping[str, Any]] = (),
        provenance: Mapping[str, Any] | None = None,
    ) -> int:
        """Append one transition atomically; unverified rewards go to quarantine."""
        state_ref = self._store_state(state)
        next_ref = self._store_state(next_state) if next_state is not None else None
        if not reward.verified:
            self.quarantine(
                episode_id,
                None,
                "unverified_reward",
                {"step_index": step_index, "action_id": action.action_id},
            )
            raise ValueError(
                "unverified reward evidence cannot enter the trusted training set "
                f"(episode {episode_id}, step {step_index})"
            )
        prev_hash = self._integrity_chain(episode_id)
        payload = {
            "episode_id": episode_id,
            "step_index": step_index,
            "state_ref": state_ref,
            "action": action.to_dict(),
            "outcome": outcome.to_dict(),
            "next_state_ref": next_ref,
            "terminal": terminal,
            "eligible": sorted(eligible_action_ids),
            "origin": origin,
        }
        integrity_hash = stable_hash({"prev": prev_hash, "step": payload})
        decision_json = dict(policy_decision or {})
        with self._connection:
            cursor = self._connection.execute(
                "INSERT INTO rl_steps (episode_id, step_index, state_ref, next_state_ref, "
                "action_json, action_type, action_id, eligible_action_ids_json, outcome_json, "
                "outcome_status, measured_latency_ms, measured_cpu_cost, measured_memory_mb, "
                "terminal, origin, integrity_hash, prev_hash, created_at_utc) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    episode_id,
                    int(step_index),
                    state_ref,
                    next_ref,
                    json.dumps(action.to_dict(), sort_keys=True),
                    action.action_type.value,
                    action.action_id,
                    json.dumps(sorted(eligible_action_ids)),
                    json.dumps(outcome.to_dict(), sort_keys=True),
                    outcome.status.value,
                    float(outcome.measured_latency_ms),
                    float(outcome.measured_cpu_cost),
                    float(outcome.measured_memory_mb),
                    int(terminal),
                    origin,
                    integrity_hash,
                    prev_hash,
                    _utc_now(),
                ),
            )
            step_id = int(cursor.lastrowid)
            self._connection.execute(
                "INSERT INTO rl_rewards (step_id, quality_delta, evidence_delta, resource_cost, "
                "latency_cost, failure_cost, total_reward, weights_version, "
                "reward_spec_fingerprint, evaluator_id, evaluator_version, attribution_method, "
                "attribution_uncertainty, terminal, verified, evidence_refs_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
                (
                    step_id,
                    reward.quality_delta,
                    reward.evidence_delta,
                    reward.resource_cost,
                    reward.latency_cost,
                    reward.failure_cost,
                    reward.total,
                    reward.weights_version,
                    reward.reward_spec_fingerprint,
                    reward.evaluator_id,
                    reward.evaluator_version,
                    reward.attribution_method,
                    reward.attribution_uncertainty,
                    int(reward.terminal),
                    json.dumps(list(reward.evidence_refs)),
                ),
            )
            if decision_json:
                self._connection.execute(
                    "INSERT INTO rl_policy_decisions (step_id, policy_id, policy_version, "
                    "proposed_action, proposal_rank_json, propensity, uncertainty, "
                    "coordinator_action, adopted, authority, fallback_used) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        step_id,
                        str(decision_json.get("policy_id", "unknown")),
                        str(decision_json.get("policy_version", "unknown")),
                        str(decision_json.get("chosen_action_id", "")),
                        json.dumps(decision_json.get("ranked", []), sort_keys=True),
                        decision_json.get("propensity"),
                        decision_json.get("uncertainty"),
                        coordinator_action,
                        int(adopted),
                        str(decision_json.get("authority", "PROPOSAL")),
                        int(decision_json.get("fallback_used", False)),
                    ),
                )
            for evaluation in evaluations:
                self._connection.execute(
                    "INSERT INTO rl_evaluations (step_id, episode_id, evaluator_id, "
                    "evaluator_version, score_json, evidence_refs_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        step_id,
                        episode_id,
                        str(evaluation.get("evaluator_id", "")),
                        str(evaluation.get("evaluator_version", "")),
                        json.dumps(evaluation.get("scores", {}), sort_keys=True),
                        json.dumps(list(evaluation.get("evidence_refs", ()))),
                    ),
                )
            self._connection.execute(
                "UPDATE rl_episodes SET integrity_head = ? WHERE episode_id = ?",
                (integrity_hash, episode_id),
            )
        return step_id

    def _store_state(self, state: OREState) -> str:
        payload = json.dumps(state.to_dict(), sort_keys=True)
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO rl_state_refs (state_ref, task_id, task_family, "
                "workspace_id, workspace_revision, encoder_version, state_json, "
                "created_at_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    state.state_id,
                    state.task_id,
                    state.task_family,
                    state.workspace_id,
                    state.workspace_revision,
                    state.encoder_version,
                    payload,
                    _utc_now(),
                ),
            )
        return state.state_id

    def state_by_ref(self, state_ref: str) -> OREState | None:
        row = self._connection.execute(
            "SELECT state_json FROM rl_state_refs WHERE state_ref = ?", (state_ref,)
        ).fetchone()
        if row is None:
            return None
        return OREState.from_dict(json.loads(row["state_json"]))

    # -- quarantine -----------------------------------------------------------
    def quarantine(
        self,
        episode_id: str | None,
        step_id: int | None,
        reason: str,
        details: Mapping[str, Any] | None = None,
    ) -> int:
        with self._connection:
            cursor = self._connection.execute(
                "INSERT INTO rl_quarantine (episode_id, step_id, reason, details_json, "
                "quarantined_at_utc) VALUES (?, ?, ?, ?, ?)",
                (
                    episode_id,
                    step_id,
                    reason,
                    json.dumps(dict(details or {}), sort_keys=True),
                    _utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def list_quarantine(self) -> tuple[dict[str, Any], ...]:
        rows = self._connection.execute(
            "SELECT * FROM rl_quarantine ORDER BY quarantine_id"
        ).fetchall()
        return tuple(dict(row) for row in rows)

    # -- validation -------------------------------------------------------------
    def validate_episode(
        self,
        episode_id: str,
        *,
        weights_by_version: Mapping[str, "RewardWeights"] | None = None,
        tolerance: float = 1e-6,
    ) -> dict[str, Any]:
        """Completeness and integrity checks; failures are quarantined (Step 31).

        Non-terminal rewards are recomputed from stored components with the
        versioned weights (acceptance test 8.2). Terminal rewards use the
        documented terminal rule and are only checked for presence.
        """
        problems: list[str] = []
        episode = self._connection.execute(
            "SELECT * FROM rl_episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if episode is None:
            return {"episode_id": episode_id, "valid": False, "problems": ["missing_episode"]}
        steps = self._connection.execute(
            "SELECT * FROM rl_steps WHERE episode_id = ? ORDER BY step_index", (episode_id,)
        ).fetchall()
        expected_index = 0
        prev_hash = ""
        for row in steps:
            if row["step_index"] != expected_index:
                problems.append(f"step_index gap at {row['step_index']}")
            expected_index = row["step_index"] + 1
            if row["prev_hash"] != prev_hash:
                problems.append(f"integrity chain break at step {row['step_index']}")
            recomputed = stable_hash(
                {
                    "prev": row["prev_hash"],
                    "step": {
                        "episode_id": episode_id,
                        "step_index": row["step_index"],
                        "state_ref": row["state_ref"],
                        "action": json.loads(row["action_json"]),
                        "outcome": json.loads(row["outcome_json"]),
                        "next_state_ref": row["next_state_ref"],
                        "terminal": bool(row["terminal"]),
                        "eligible": json.loads(row["eligible_action_ids_json"]),
                        "origin": row["origin"],
                    },
                }
            )
            if recomputed != row["integrity_hash"]:
                problems.append(f"integrity hash mismatch at step {row['step_index']}")
            if row["next_state_ref"] is None and not row["terminal"]:
                problems.append(f"missing next state on non-terminal step {row['step_index']}")
            stored_outcome = json.loads(row["outcome_json"])
            if (
                abs(float(row["measured_latency_ms"] or 0.0)
                    - float(stored_outcome.get("measured_latency_ms", 0.0))) > 1e-6
                or abs(float(row["measured_cpu_cost"] or 0.0)
                       - float(stored_outcome.get("measured_cpu_cost", 0.0))) > 1e-6
                or abs(float(row["measured_memory_mb"] or 0.0)
                       - float(stored_outcome.get("measured_memory_mb", 0.0))) > 1e-6
            ):
                problems.append(
                    f"measured cost columns diverge from the stored outcome at step {row['step_index']}"
                )
            reward = self._connection.execute(
                "SELECT * FROM rl_rewards WHERE step_id = ?", (row["step_id"],)
            ).fetchone()
            if reward is None:
                problems.append(f"missing reward at step {row['step_index']}")
            else:
                if not reward["verified"]:
                    problems.append(f"unverified reward at step {row['step_index']}")
                if not reward["terminal"]:
                    weights = (weights_by_version or {}).get(reward["weights_version"])
                    if weights is not None:
                        recomputed_total = (
                            weights.alpha_quality * reward["quality_delta"]
                            + weights.beta_evidence * reward["evidence_delta"]
                            - weights.gamma_resource * reward["resource_cost"]
                            - weights.delta_latency * reward["latency_cost"]
                            - weights.eta_failure * reward["failure_cost"]
                        )
                        if abs(recomputed_total - reward["total_reward"]) > tolerance:
                            problems.append(
                                f"reward recomputation mismatch at step {row['step_index']}"
                            )
            action = json.loads(row["action_json"])
            eligible = set(json.loads(row["eligible_action_ids_json"]))
            if action.get("action_id") not in eligible:
                problems.append(
                    f"recorded action not in eligible mask at step {row['step_index']}"
                )
            prev_hash = row["integrity_hash"]
        if problems:
            for problem in problems:
                self.quarantine(episode_id, None, "validation", {"problem": problem})
        return {"episode_id": episode_id, "valid": not problems, "problems": problems}

    def verify_integrity(self, episode_id: str) -> bool:
        result = self.validate_episode(episode_id)
        return bool(result["valid"])

    # -- datasets ------------------------------------------------------------
    def list_episodes(self, *, status: str | None = None) -> tuple[dict[str, Any], ...]:
        query = "SELECT * FROM rl_episodes"
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)
        rows = self._connection.execute(query + " ORDER BY started_at_utc", params).fetchall()
        return tuple(dict(row) for row in rows)

    def _episode_rows(self, origin: str = "OBSERVED") -> dict[str, list[sqlite3.Row]]:
        if origin == "ANY":
            query = (
                "SELECT s.* FROM rl_steps s JOIN rl_episodes e ON s.episode_id = e.episode_id "
                "WHERE e.status != 'RUNNING' ORDER BY s.episode_id, s.step_index"
            )
            rows = self._connection.execute(query).fetchall()
        else:
            query = (
                "SELECT s.* FROM rl_steps s JOIN rl_episodes e ON s.episode_id = e.episode_id "
                "WHERE s.origin = ? AND e.status != 'RUNNING' ORDER BY s.episode_id, s.step_index"
            )
            rows = self._connection.execute(query, (origin,)).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(row["episode_id"], []).append(row)
        return grouped

    def build_dataset(
        self,
        *,
        split_name: str,
        rule: SplitRule,
        include_groups: Sequence[str],
    ) -> DatasetManifest:
        """Create a manifest for the episodes in the given groups (Step 32).

        Related episodes stay together inside a group; splits are selected at
        task-family / source-case / session / time-period granularity, never at
        individual episode granularity.
        """
        group_column = {
            SplitRule.BY_TASK_FAMILY: "task_family",
            SplitRule.BY_SOURCE_CASE: "source_case",
            SplitRule.BY_SESSION: "session_id",
            SplitRule.BY_TIME_PERIOD: "date(started_at_utc)",
        }[rule.kind]
        placeholders = ", ".join("?" for _ in include_groups)
        rows = self._connection.execute(
            f"SELECT DISTINCT episode_id, {group_column} AS grp FROM rl_episodes "
            f"WHERE {group_column} IN ({placeholders}) ORDER BY episode_id",
            tuple(include_groups),
        ).fetchall()
        episode_ids = tuple(row["episode_id"] for row in rows)
        step_count = 0
        for episode_id in episode_ids:
            count = self._connection.execute(
                "SELECT COUNT(*) AS n FROM rl_steps WHERE episode_id = ?", (episode_id,)
            ).fetchone()["n"]
            step_count += int(count)
        definition = {
            "split_name": split_name,
            "rule": rule.to_dict(),
            "include_groups": list(include_groups),
            "schema_version": "1.0.0",
        }
        checksum = stable_hash(
            {
                "episodes": list(episode_ids),
                "definition": definition,
            }
        )
        dataset_id = f"dataset_{checksum[:16]}"
        manifest = DatasetManifest(
            dataset_id=dataset_id,
            split=split_name,
            query_fingerprint=rule.fingerprint(),
            schema_version="1.0.0",
            checksum=checksum,
            definition=definition,
            episode_ids=episode_ids,
            step_count=step_count,
            created_at=_utc_now(),
        )
        with self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO rl_dataset_manifests (dataset_id, split, "
                "query_fingerprint, schema_version, checksum, definition_json, "
                "episode_ids_json, step_count, created_at_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    manifest.dataset_id,
                    manifest.split,
                    manifest.query_fingerprint,
                    manifest.schema_version,
                    manifest.checksum,
                    json.dumps(definition, sort_keys=True),
                    json.dumps(list(manifest.episode_ids)),
                    manifest.step_count,
                    manifest.created_at,
                ),
            )
        return manifest

    def get_manifest(self, dataset_id: str) -> DatasetManifest | None:
        row = self._connection.execute(
            "SELECT * FROM rl_dataset_manifests WHERE dataset_id = ?", (dataset_id,)
        ).fetchone()
        if row is None:
            return None
        return DatasetManifest(
            dataset_id=row["dataset_id"],
            split=row["split"],
            query_fingerprint=row["query_fingerprint"],
            schema_version=row["schema_version"],
            checksum=row["checksum"],
            definition=json.loads(row["definition_json"]),
            episode_ids=tuple(json.loads(row["episode_ids_json"])),
            step_count=int(row["step_count"]),
            created_at=row["created_at_utc"],
        )

    def verify_manifest(self, dataset_id: str) -> bool:
        """Reproduce the exact split from the manifest checksum (acceptance 8.2)."""
        manifest = self.get_manifest(dataset_id)
        if manifest is None:
            return False
        recomputed = stable_hash(
            {"episodes": list(manifest.episode_ids), "definition": dict(manifest.definition)}
        )
        return recomputed == manifest.checksum

    def load_replay_dataset(
        self, dataset_id: str, *, include_simulated: bool = False
    ) -> tuple[TransitionTuple, ...]:
        """Validated replay tuples for off-policy training (Step 33, Step 49).

        Simulated (world-model) transitions are excluded unless explicitly
        requested; the default keeps predicted and observed data separate.
        """
        manifest = self.get_manifest(dataset_id)
        if manifest is None:
            raise KeyError(f"unknown dataset: {dataset_id}")
        episodes = self._episode_rows("ANY" if include_simulated else "OBSERVED")
        tuples: list[TransitionTuple] = []
        for episode_id in manifest.episode_ids:
            rows = episodes.get(episode_id, [])
            for row in rows:
                if not include_simulated and row["origin"] != "OBSERVED":
                    continue
                reward_row = self._connection.execute(
                    "SELECT total_reward FROM rl_rewards WHERE step_id = ?", (row["step_id"],)
                ).fetchone()
                state = self.state_by_ref(row["state_ref"])
                next_state = (
                    self.state_by_ref(row["next_state_ref"])
                    if row["next_state_ref"]
                    else None
                )
                if state is None or (row["next_state_ref"] and next_state is None):
                    continue
                tuples.append(
                    TransitionTuple(
                        state_features=dict(state.features),
                        action_id=row["action_id"],
                        action_type=row["action_type"],
                        reward=float(reward_row["total_reward"]) if reward_row else 0.0,
                        next_state_features=dict(next_state.features) if next_state else {},
                        done=bool(row["terminal"]),
                        eligible_action_ids=tuple(
                            json.loads(row["eligible_action_ids_json"])
                        ),
                        terminal=bool(row["terminal"]),
                    )
                )
        return tuple(tuples)

    def load_bandit_dataset(
        self, dataset_id: str, *, include_simulated: bool = False
    ) -> tuple[BanditTuple, ...]:
        """Context/action/propensity/reward records for bandit training (Step 33)."""
        manifest = self.get_manifest(dataset_id)
        if manifest is None:
            raise KeyError(f"unknown dataset: {dataset_id}")
        episodes = self._episode_rows("ANY" if include_simulated else "OBSERVED")
        tuples: list[BanditTuple] = []
        for episode_id in manifest.episode_ids:
            episode = self._connection.execute(
                "SELECT task_family FROM rl_episodes WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            family = episode["task_family"] if episode else ""
            for row in episodes.get(episode_id, []):
                if not include_simulated and row["origin"] != "OBSERVED":
                    continue
                decision = self._connection.execute(
                    "SELECT propensity FROM rl_policy_decisions WHERE step_id = ?",
                    (row["step_id"],),
                ).fetchone()
                reward_row = self._connection.execute(
                    "SELECT total_reward FROM rl_rewards WHERE step_id = ?", (row["step_id"],)
                ).fetchone()
                state = self.state_by_ref(row["state_ref"])
                if state is None:
                    continue
                tuples.append(
                    BanditTuple(
                        context_features=dict(state.features),
                        action_id=row["action_id"],
                        propensity=(
                            float(decision["propensity"]) if decision and decision["propensity"]
                            else None
                        ),
                        reward=float(reward_row["total_reward"]) if reward_row else 0.0,
                        task_family=family,
                    )
                )
        return tuple(tuples)

    def counts(self) -> dict[str, int]:
        def one(table: str) -> int:
            return int(
                self._connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            )

        return {
            "episodes": one("rl_episodes"),
            "steps": one("rl_steps"),
            "rewards": one("rl_rewards"),
            "decisions": one("rl_policy_decisions"),
            "evaluations": one("rl_evaluations"),
            "state_refs": one("rl_state_refs"),
            "quarantine": one("rl_quarantine"),
            "manifests": one("rl_dataset_manifests"),
        }
