"""ORE nine-stage semantic-composition subsystem: 54 operations and two distinct tracks.

Import the processor, typed contracts, and operation registry from this package.
The package builds on, but does not replace, ore_semantic or ore_syntax.
"""
from .adapters import composition_request_from_results
from .biology import POPULATIONS, SemanticCompositionLIFSimulator
from .contracts import (
    PredicateFrame,
    SemanticCompositionConfig,
    SemanticCompositionRequest,
    SemanticCompositionResult,
)
from .pipeline import SemanticCompositionProcessor
from .registry import (
    OPERATION_REGISTRY,
    STAGE_REGISTRY,
    OperationSpec,
    StageSpec,
    registry_as_dicts,
    stages_as_dicts,
)

__all__ = [
    "OPERATION_REGISTRY",
    "POPULATIONS",
    "PredicateFrame",
    "SemanticCompositionConfig",
    "SemanticCompositionLIFSimulator",
    "SemanticCompositionProcessor",
    "SemanticCompositionRequest",
    "SemanticCompositionResult",
    "STAGE_REGISTRY",
    "OperationSpec",
    "StageSpec",
    "composition_request_from_results",
    "registry_as_dicts",
    "stages_as_dicts",
]
