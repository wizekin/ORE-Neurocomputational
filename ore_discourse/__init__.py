"""ORE eight-stage discourse, pragmatic and situation-model subsystem."""

from .biology import POPULATIONS, DiscourseLIFSimulator
from .contracts import DiscourseConfig, DiscourseRequest, DiscourseResult
from .pipeline import DiscourseN7Adapter, DiscourseProcessor
from .registry import OPERATION_REGISTRY, STAGE_REGISTRY, OperationSpec, StageSpec, registry_as_dicts

__all__ = [
    "DiscourseConfig",
    "DiscourseLIFSimulator",
    "DiscourseN7Adapter",
    "DiscourseProcessor",
    "DiscourseRequest",
    "DiscourseResult",
    "OPERATION_REGISTRY",
    "POPULATIONS",
    "STAGE_REGISTRY",
    "OperationSpec",
    "StageSpec",
    "registry_as_dicts",
]
