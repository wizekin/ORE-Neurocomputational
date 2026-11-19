"""Immutable revision history and snapshot lookup for reversible workspace reasoning."""
from __future__ import annotations

from threading import RLock

from .exceptions import RevisionNotFoundError
from .models import WorkspaceRevision, WorkspaceState


class RevisionStore:
    """Retain reconstructable state snapshots and the findings that produced each revision."""

    def __init__(self, initial_state: WorkspaceState) -> None:
        """Create revision zero from the initialized workspace state."""
        self._lock = RLock()
        self._revisions: dict[int, WorkspaceRevision] = {
            initial_state.workspace_revision: WorkspaceRevision(
                revision=initial_state.workspace_revision,
                snapshot=initial_state.to_dict(),
                changed_fields=(), accepted_finding_ids=(),
                change_origin="workspace_initialization", dependencies=(),
                validation_result="initialized",
            )
        }

    def add(self, revision: WorkspaceRevision) -> None:
        """Store a new immutable revision without allowing an existing revision to be replaced."""
        with self._lock:
            if revision.revision in self._revisions:
                raise ValueError(f"revision {revision.revision} already exists")
            latest = max(self._revisions)
            if revision.revision != latest + 1:
                raise ValueError(f"expected revision {latest + 1}, received {revision.revision}")
            self._revisions[revision.revision] = revision

    def get(self, revision: int) -> WorkspaceRevision:
        """Return a historical revision or raise a typed error when it is unavailable."""
        with self._lock:
            record = self._revisions.get(revision)
        if record is None:
            raise RevisionNotFoundError(f"workspace revision {revision} is not retained")
        return record

    def latest(self) -> WorkspaceRevision:
        """Return the most recent stored revision."""
        with self._lock:
            return self._revisions[max(self._revisions)]

    def list_revisions(self) -> tuple[WorkspaceRevision, ...]:
        """Return revision history ordered from the initial state to the latest state."""
        with self._lock:
            return tuple(self._revisions[key] for key in sorted(self._revisions))

    def __len__(self) -> int:
        """Return the number of retained workspace revisions."""
        with self._lock:
            return len(self._revisions)