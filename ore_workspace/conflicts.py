"""Conflict registry that preserves alternative evidence and records explicit preferences."""
from __future__ import annotations

from dataclasses import replace
from .models import utc_now
from threading import RLock
from typing import Iterable

from .models import ConflictRecord, Finding


class ConflictRegistry:
    """Keep contradictory findings accessible until an explicit resolution is recorded."""

    def __init__(self) -> None:
        """Initialize an independent, thread-safe collection of conflict records."""
        self._lock = RLock()
        self._records: dict[str, ConflictRecord] = {}

    def register(
        self,
        workspace_id: str,
        existing: Finding,
        incoming: Finding,
        *,
        conflict_type: str = "competing_hypotheses",
    ) -> ConflictRecord:
        """Create a conflict record for two alternatives without deleting either finding."""
        if existing.finding_id == incoming.finding_id:
            raise ValueError("a finding cannot conflict with itself")
        record = ConflictRecord(
            workspace_id=workspace_id,
            conflicting_findings=(existing.finding_id, incoming.finding_id),
            conflict_type=conflict_type,
            affected_fields=(incoming.target.field,),
        )
        with self._lock:
            for current in self._records.values():
                if (current.workspace_id == workspace_id
                    and set(current.conflicting_findings) == set(record.conflicting_findings)
                    and current.status == "unresolved"):
                    return current
            self._records[record.conflict_id] = record
        return record

    def get(self, conflict_id: str) -> ConflictRecord:
        """Retrieve a conflict by ID or raise a key error when it is unknown."""
        with self._lock:
            return self._records[conflict_id]

    def list(self, workspace_id: str | None = None, *, unresolved_only: bool = False) -> tuple[ConflictRecord, ...]:
        """List conflicts optionally scoped to one workspace and unresolved status."""
        with self._lock:
            values = tuple(self._records.values())
        return tuple(
            item for item in values
            if (workspace_id is None or item.workspace_id == workspace_id)
            and (not unresolved_only or item.status == "unresolved")
        )

    def resolve(
        self,
        conflict_id: str,
        *,
        preferred_finding_id: str | None = None,
        resolution_candidates: Iterable[str] = (),
    ) -> ConflictRecord:
        """Record a preferred interpretation while retaining all conflicting evidence IDs."""
        with self._lock:
            current = self._records[conflict_id]
            if preferred_finding_id is not None and preferred_finding_id not in current.conflicting_findings:
                raise ValueError("preferred finding must belong to this conflict")
            candidates = tuple(dict.fromkeys(resolution_candidates))
            updated = replace(
                current, status="resolved", preferred_finding_id=preferred_finding_id,
                resolution_candidates=candidates, resolved_at=utc_now(),
            )
            self._records[conflict_id] = updated
            return updated