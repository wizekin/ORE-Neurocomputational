"""ORE Point 03 hippocampal replay and sleep-dependent systems consolidation."""
from .annotations import register_annotations
from .contracts import (
    HippocampalTrace,
    ReplayOpportunity,
    SleepConsolidationConfig,
    SleepConsolidationResult,
)
from .pipeline import SleepDependentSystemsConsolidationProcessor
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "HippocampalTrace",
    "ReplayOpportunity",
    "SleepConsolidationConfig",
    "SleepConsolidationResult",
    "SleepDependentSystemsConsolidationProcessor",
    "OPERATION_REGISTRY",
    "OperationSpec",
    "register_annotations",
    "registry_as_dicts",
]
