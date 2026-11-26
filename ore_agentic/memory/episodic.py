"""Episodic memory: linked, policy-governed experience records (Step 38).

Each episode records a task, observation, decision, transition, outcome,
failure or snapshot, linked to workflow, task, worker, time, evidence and
software versions.  Privacy redaction and retention policy are applied at
record time, and replay/recall never rewrites history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from ..contracts import new_id, utc_now

__all__ = (
    "EpisodeKind",
    "EpisodeRecord",
    "EpisodicMemory",
    "REDACTED",
    "redact_payload",
)

REDACTED = "***REDACTED***"
_FORBIDDEN_KEYS = frozenset({
    "secret", "secrets", "password", "token", "api_key", "credentials",
    "authorization", "cookie", "session_key", "private_key",
})


def redact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Redact secret-bearing keys from a payload before storage (Step 38)."""
    redacted: dict[str, Any] = {}
    for key, value in payload.items():
        if str(key).lower() in _FORBIDDEN_KEYS:
            redacted[key] = REDACTED
        elif isinstance(value, Mapping):
            redacted[key] = redact_payload(value)
        else:
            redacted[key] = value
    return redacted


class EpisodeKind(str, Enum):
    """The episode categories of the specification (Step 38)."""

    TASK = "task"
    OBSERVATION = "observation"
    DECISION = "decision"
    TRANSITION = "transition"
    OUTCOME = "outcome"
    FAILURE = "failure"
    SNAPSHOT = "snapshot"


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """One episodic record with full linkage (Step 38)."""

    episode_id: str
    kind: EpisodeKind
    summary: str
    workflow_id: str = ""
    task_id: str = ""
    worker_id: str = ""
    outcome: str = ""
    evidence_refs: tuple[str, ...] = ()
    software_versions: Mapping[str, str] = field(default_factory=dict)
    payload: Mapping[str, Any] = field(default_factory=dict)
    retention_class: str = "standard"
    redacted: bool = False
    recorded_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate the record envelope."""
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record."""
        return {
            "episode_id": self.episode_id,
            "kind": self.kind.value,
            "summary": self.summary,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "outcome": self.outcome,
            "evidence_refs": list(self.evidence_refs),
            "software_versions": dict(self.software_versions),
            "payload": dict(self.payload),
            "retention_class": self.retention_class,
            "redacted": self.redacted,
            "recorded_at": self.recorded_at,
        }


_RETENTION_ORDER = {"ephemeral": 0, "standard": 1, "audit": 2}


class EpisodicMemory:
    """Store and recall linked episodic records (Steps 38, 42)."""

    def __init__(self, max_payload_chars: int = 2000,
                 default_retention: str = "standard") -> None:
        """Configure payload limits and the default retention class."""
        self._episodes: dict[str, EpisodeRecord] = {}
        self._max_payload_chars = max_payload_chars
        self._default_retention = default_retention

    def record(self, kind: EpisodeKind, summary: str, *,
               workflow_id: str = "", task_id: str = "", worker_id: str = "",
               outcome: str = "", evidence_refs: Sequence[str] = (),
               software_versions: Mapping[str, str] | None = None,
               payload: Mapping[str, Any] | None = None,
               retention_class: str | None = None) -> EpisodeRecord:
        """Record one episode with redaction and retention applied (Step 38)."""
        safe_payload = redact_payload(payload or {})
        rendered = str(safe_payload)
        if len(rendered) > self._max_payload_chars:
            safe_payload = {"truncated": True,
                            "preview": rendered[: self._max_payload_chars]}
        record = EpisodeRecord(
            episode_id=new_id("episode"), kind=kind, summary=summary,
            workflow_id=workflow_id, task_id=task_id, worker_id=worker_id,
            outcome=outcome, evidence_refs=tuple(evidence_refs),
            software_versions=dict(software_versions or {}),
            payload=safe_payload,
            retention_class=retention_class or self._default_retention,
            redacted=any(value == REDACTED for value in safe_payload.values()))
        self._episodes[record.episode_id] = record
        return record

    def recall(self, *, workflow_id: str | None = None, task_id: str | None = None,
               kind: EpisodeKind | None = None) -> tuple[EpisodeRecord, ...]:
        """Recall episodes by workflow, task or kind (Step 38)."""
        results = []
        for record in self._episodes.values():
            if workflow_id is not None and record.workflow_id != workflow_id:
                continue
            if task_id is not None and record.task_id != task_id:
                continue
            if kind is not None and record.kind is not kind:
                continue
            results.append(record)
        return tuple(results)

    def episode(self, episode_id: str) -> EpisodeRecord | None:
        """Return one episode by ID."""
        return self._episodes.get(episode_id)

    def apply_retention(self, keep_classes: Sequence[str] = ("standard", "audit")
                        ) -> tuple[str, ...]:
        """Delete episodes outside the retained classes (Step 42).

        Returns the IDs of removed episodes; deletion is explicit and logged
        by the caller, never silent.
        """
        keep = set(keep_classes)
        removed = [episode_id for episode_id, record in self._episodes.items()
                   if record.retention_class not in keep]
        for episode_id in removed:
            del self._episodes[episode_id]
        return tuple(removed)

    def episodes(self) -> tuple[EpisodeRecord, ...]:
        """Return all episodes in insertion order."""
        return tuple(self._episodes.values())

    def describe(self) -> dict[str, Any]:
        """Return a serializable summary."""
        kinds: dict[str, int] = {}
        for record in self._episodes.values():
            kinds[record.kind.value] = kinds.get(record.kind.value, 0) + 1
        return {"episodes": len(self._episodes), "kinds": kinds}
