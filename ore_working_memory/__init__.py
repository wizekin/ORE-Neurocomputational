"""ORE working-memory and cognitive-control subsystem: five stages and 30 operations."""

from .biology import POPULATIONS, WorkingMemoryLIFSimulator
from .contracts import WorkingMemoryConfig, WorkingMemoryRequest, WorkingMemoryResult
from .pipeline import WorkingMemoryProcessor
from .registry import OPERATION_REGISTRY, STAGE_REGISTRY, OperationSpec, StageSpec, registry_as_dicts

__all__ = [
    "OPERATION_REGISTRY", "POPULATIONS", "STAGE_REGISTRY", "OperationSpec", "StageSpec",
    "WorkingMemoryConfig", "WorkingMemoryLIFSimulator", "WorkingMemoryProcessor",
    "WorkingMemoryRequest", "WorkingMemoryResult", "registry_as_dicts",
]
