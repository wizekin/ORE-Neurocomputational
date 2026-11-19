"""Versioned, permission-controlled shared cognitive workspace for ORE modules."""
from .conflicts import ConflictRegistry
from .event_bus import WorkspaceEventBus
from .evidence import EvidenceContribution, EvidenceScore, HypothesisScorer
from .goals import GoalEvaluator
from .merge_policies import MergePolicyRegistry
from .models import (
    APPEND_ONLY_FIELDS, WORKSPACE_FIELDS, ConflictRecord, DeliveryResult,
    Finding, FindingTarget, GoalEvaluation, GoalOutcome, ModuleActivationResult,
    Provenance, UpdateDecision, ValidationResult, WorkspaceEvent,
    WorkspaceRevision, WorkspaceSnapshot, WorkspaceState,
)
from .observability import SQLiteWorkspaceObserver, register_workspace_annotations
from .permissions import DEFAULT_WRITE_PERMISSIONS, PermissionPolicy
from .publication import FindingPublisher
from .revisions import RevisionStore
from .scheduler import ModuleScheduler, SchedulerReport
from .state_manager import WorkspaceStateManager
from .subscriptions import DEFAULT_SUBSCRIPTIONS, Subscription, SubscriptionRegistry
from .working_memory import RetentionWeights, WorkingMemoryPolicy
from .workspace import SharedWorkspace, WorkspaceManager

__all__ = [
    "APPEND_ONLY_FIELDS", "WORKSPACE_FIELDS", "ConflictRecord", "ConflictRegistry",
    "DeliveryResult", "DEFAULT_SUBSCRIPTIONS", "DEFAULT_WRITE_PERMISSIONS",
    "EvidenceContribution", "EvidenceScore", "HypothesisScorer",
    "Finding", "FindingPublisher", "FindingTarget", "GoalEvaluation", "GoalEvaluator",
    "GoalOutcome", "MergePolicyRegistry", "ModuleActivationResult", "ModuleScheduler",
    "PermissionPolicy", "Provenance", "RetentionWeights", "RevisionStore",
    "SQLiteWorkspaceObserver", "SchedulerReport", "SharedWorkspace", "Subscription",
    "SubscriptionRegistry", "UpdateDecision", "ValidationResult", "WorkingMemoryPolicy",
    "WorkspaceEvent", "WorkspaceEventBus", "WorkspaceManager", "WorkspaceRevision",
    "WorkspaceSnapshot", "WorkspaceState", "WorkspaceStateManager",
    "register_workspace_annotations",
]