"""Supervisor, worker contracts, leases and bounded parallel execution (Steps 10-16)."""
from __future__ import annotations

from .leases import LeaseExpiredError, LeaseManager, StaleLeaseError, TaskLease
from .supervisor import (
    EligibilityReport,
    NoEligibleWorkerError,
    SelectionRecord,
    Supervisor,
    classify_exception,
)
from .workers import (
    BackpressureError,
    BoundedWorkQueue,
    CancelledError,
    LocalWorker,
    MapOutcome,
    MapReduceExecutor,
    PermanentTaskError,
    PolicyTaskError,
    ResourceTaskError,
    TaskEnvelope,
    TransientTaskError,
    Worker,
)

__all__ = (
    "BackpressureError",
    "BoundedWorkQueue",
    "CancelledError",
    "EligibilityReport",
    "LeaseExpiredError",
    "LeaseManager",
    "LocalWorker",
    "MapOutcome",
    "MapReduceExecutor",
    "NoEligibleWorkerError",
    "PermanentTaskError",
    "PolicyTaskError",
    "ResourceTaskError",
    "SelectionRecord",
    "StaleLeaseError",
    "Supervisor",
    "TaskEnvelope",
    "TaskLease",
    "TransientTaskError",
    "Worker",
    "classify_exception",
)
