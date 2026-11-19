"""Shared cognitive workspace instances and their lifecycle manager."""
from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable
from threading import RLock
from typing import Callable

from .conflicts import ConflictRegistry
from .event_bus import WorkspaceEventBus
from .exceptions import UnknownWorkspaceError
from .goals import GoalEvaluator
from .models import Finding, GoalEvaluation, WorkspaceSnapshot, WorkspaceState, new_id
from .observability import SQLiteWorkspaceObserver
from .permissions import PermissionPolicy
from .revisions import RevisionStore
from .state_manager import WorkspaceStateManager
from .subscriptions import SubscriptionRegistry


class SharedWorkspace:
    """Own one versioned shared cognitive state and route all changes through its state manager."""

    def __init__(
        self,
        current_input: str,
        current_goal: str,
        *,
        workspace_id: str | None = None,
        permissions: PermissionPolicy | None = None,
        subscriptions: SubscriptionRegistry | None = None,
        observer: SQLiteWorkspaceObserver | None = None,
    ) -> None:
        """Initialize state, revision zero, registries, event delivery and controlled publication."""
        if not isinstance(current_input, str):
            raise TypeError("current_input must be a string")
        if not isinstance(current_goal, str) or not current_goal.strip():
            raise ValueError("current_goal must be a non-empty string")
        self.lock = RLock()
        self._state = WorkspaceState(
            workspace_id=workspace_id or new_id("workspace"),
            current_input=current_input,
            current_goal=current_goal,
        )
        self.revisions = RevisionStore(self._state)
        self.permissions = permissions or PermissionPolicy()
        self.subscriptions = subscriptions or SubscriptionRegistry()
        self.conflicts = ConflictRegistry()
        self.event_bus = WorkspaceEventBus(self.subscriptions)
        self.observer = observer
        self.findings: dict[str, Finding] = {}
        self.finding_fingerprints: dict[str, str] = {}
        self.pending_findings: dict[str, Finding] = {}
        self.rejected_findings: dict[str, Finding] = {}
        self.events = []
        self.delivery_results = []
        self.state_manager = WorkspaceStateManager(self, observer=observer)
        self._closed = False

    @property
    def state(self) -> WorkspaceState:
        """Return a detached state copy so callers cannot bypass controlled publication."""
        with self.lock:
            return self._state.clone()

    @property
    def workspace_id(self) -> str:
        """Return this workspace's stable identifier."""
        return self._state.workspace_id

    @property
    def revision(self) -> int:
        """Return the current authoritative workspace revision number."""
        with self.lock:
            return self._state.workspace_revision

    def snapshot(self, revision: int | None = None) -> WorkspaceSnapshot:
        """Return a detached current or historical snapshot for an independent module read."""
        if revision is None:
            with self.lock:
                state = self._state.clone()
            return WorkspaceSnapshot(state.workspace_id, state.workspace_revision, state.to_dict())
        historical = self.revisions.get(revision)
        return WorkspaceSnapshot(self.workspace_id, revision, historical.snapshot)

    def publish(self, finding: Finding):
        """Submit a module finding to the validator and controlled state commit path."""
        return self.state_manager.submit(finding)

    def register_consumer(
        self,
        module_id: str,
        callback: Callable,
        fields: Iterable[str] | None = None,
        *,
        threshold: float = 0.0,
        sensitivity: dict[str, float] | None = None,
    ) -> None:
        """Register a notification callback and optionally configure its field subscriptions."""
        if fields is not None:
            self.subscriptions.register(module_id, fields, threshold=threshold, sensitivity=sensitivity)
        elif self.subscriptions.get(module_id) is None:
            raise ValueError("consumer must have a field subscription before it can receive events")
        self.event_bus.subscribe(module_id, callback)

    def evaluate_goal(
        self,
        evaluator: GoalEvaluator | None = None,
        component_scores: dict[str, float] | None = None,
    ) -> GoalEvaluation:
        """Evaluate the current goal using an explicit weighted-and-constrained evaluator."""
        return (evaluator or GoalEvaluator()).evaluate(self.snapshot(), component_scores)

    def close(self) -> None:
        """Release bounded event-dispatch resources without deleting stored workspace state."""
        if not self._closed:
            self.event_bus.close(wait=True)
            self._closed = True


class WorkspaceManager:
    """Create and locate isolated workspaces while sharing ORE's SQLite observation run."""

    def __init__(
        self,
        *,
        repository_root: str | Path | None = None,
        logger_database_path: str | Path | None = None,
        enable_logging: bool = True,
    ) -> None:
        """Configure the repository logger and prepare a thread-safe workspace directory."""
        self.repository_root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
        self.logger_database_path = Path(logger_database_path).expanduser().resolve() if logger_database_path else None
        self.observer: SQLiteWorkspaceObserver | None = None
        if enable_logging:
            self.observer = SQLiteWorkspaceObserver(self.logger_database_path, self.repository_root).open()
        self._lock = RLock()
        self._workspaces: dict[str, SharedWorkspace] = {}

    def create_workspace(
        self,
        current_input: str,
        current_goal: str = "understand_sentence",
        *,
        workspace_id: str | None = None,
        permissions: PermissionPolicy | None = None,
        subscriptions: SubscriptionRegistry | None = None,
    ) -> SharedWorkspace:
        """Create and register a workspace with independent state, access and subscription rules."""
        workspace = SharedWorkspace(
            current_input, current_goal, workspace_id=workspace_id,
            permissions=permissions, subscriptions=subscriptions, observer=self.observer,
        )
        with self._lock:
            if workspace.workspace_id in self._workspaces:
                raise ValueError(f"workspace ID already exists: {workspace.workspace_id}")
            self._workspaces[workspace.workspace_id] = workspace
        if self.observer:
            self.observer.record(
                "create_workspace", {"workspace_id": workspace.workspace_id, "goal": current_goal},
                class_name="WorkspaceManager", module_name="ore_workspace.workspace",
            )
        return workspace

    def get_workspace(self, workspace_id: str) -> SharedWorkspace:
        """Retrieve a registered workspace by ID or raise UnknownWorkspaceError."""
        with self._lock:
            workspace = self._workspaces.get(workspace_id)
        if workspace is None:
            raise UnknownWorkspaceError(f"unknown workspace: {workspace_id}")
        return workspace

    def list_workspaces(self) -> tuple[str, ...]:
        """Return registered workspace IDs in deterministic sorted order."""
        with self._lock:
            return tuple(sorted(self._workspaces))

    def close_workspace(self, workspace_id: str) -> bool:
        """Close one workspace's event bus while retaining the workspace's state and history."""
        workspace = self.get_workspace(workspace_id)
        workspace.close()
        return True

    def close(self) -> None:
        """Close every workspace and finish the single SQLite observation run."""
        with self._lock:
            workspaces = tuple(self._workspaces.values())
        for workspace in workspaces:
            workspace.close()
        if self.observer:
            self.observer.close()
            self.observer = None

    def __enter__(self) -> "WorkspaceManager":
        """Return this manager for context-managed lifecycle handling."""
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> bool:
        """Close managed workspaces and the logger run without suppressing body exceptions."""
        with self._lock:
            workspaces = tuple(self._workspaces.values())
        for workspace in workspaces:
            workspace.close()
        if self.observer:
            self.observer.close(error=exc)
            self.observer = None
        return False