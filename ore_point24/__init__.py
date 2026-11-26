"""ORE Point 24 - Complete Integration, Residual Coverage and the Unified ORE Architecture.

This package implements the 50-step, 10-phase integration algorithm of the
Point 24 specification on top of the existing ORE infrastructure:

- ``ore_contracts`` - versioned contracts, module manifests, registry,
  task planning, DAG execution and resource coordination.
- ``ore_workspace`` - versioned shared cognitive state, controlled findings,
  conflicts and bounded event scheduling.
- ``ore_usg`` - versioned, read-only USG knowledge import.
- ``ore_logger`` - shared SQLite annotation catalogue, run records and events.

The 23 external point packages ("23 Extra ORE Algorithms") are integrated
through three honest tiers (native registration, manifest-only, status
adapters) so that registration, backend binding, validation and integration
testing remain separately evidenced facts.

Quick use::

    from ore_point24 import UnifiedOREIntegrator
    integrator = UnifiedOREIntegrator(repository_root=".")
    audit = integrator.run()
    assert audit["executed_step_count"] == 50
"""
from __future__ import annotations

from .core import (
    POINT24_VERSION,
    EvidenceState,
    ModuleStateVector,
    derive_evidence_state,
)
from .spec import (
    LAYERS,
    POINT_FAMILIES,
    RESIDUAL_CATEGORIES,
    RESIDUAL_TECHNOLOGIES,
    ROADMAP_PHASES,
    point_layer_incidence,
)
from .compatibility import register_point24_contracts
from .adapters import (
    MANIFEST_ONLY_TIER_PACKAGES,
    NATIVE_TIER_PACKAGES,
    Point24IntegrationModule,
    PointIntegrationManager,
    PointIntegrationReport,
    PointModuleAdapter,
)
from .orchestrator import StepOutcome, UnifiedOREIntegrator

__version__ = POINT24_VERSION

__all__ = [
    "POINT24_VERSION",
    "EvidenceState",
    "ModuleStateVector",
    "derive_evidence_state",
    "LAYERS",
    "POINT_FAMILIES",
    "RESIDUAL_CATEGORIES",
    "RESIDUAL_TECHNOLOGIES",
    "ROADMAP_PHASES",
    "point_layer_incidence",
    "register_point24_contracts",
    "MANIFEST_ONLY_TIER_PACKAGES",
    "NATIVE_TIER_PACKAGES",
    "Point24IntegrationModule",
    "PointIntegrationManager",
    "PointIntegrationReport",
    "PointModuleAdapter",
    "StepOutcome",
    "UnifiedOREIntegrator",
    "__version__",
]
