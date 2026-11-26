"""Cognitive integration with the shared workspace, USG, and coordinators (Steps 33-39)."""
from __future__ import annotations

from .publication import (
    WorkspaceBridge,
    build_coordinator_notification,
    build_downstream_task_hints,
)
from .usg_mapping import USGConceptMapper, UNKNOWN_USG_ID

__all__ = (
    "UNKNOWN_USG_ID",
    "USGConceptMapper",
    "WorkspaceBridge",
    "build_coordinator_notification",
    "build_downstream_task_hints",
)
