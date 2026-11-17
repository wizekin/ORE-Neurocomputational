"""ORE 22-stage grounded semantic integration and predictive-language research subsystem.

The package combines executable semantic functions, an explicit mathematical-operation registry,
an optional leaky-integrate-and-fire reference model, and SQLite-backed symbol/stage annotations.
All neural mappings are computational analogies, not asserted biological ground truth.
"""
from .annotations import record_grounded_result, register_grounded_annotations
from .biology import GroundedNeuralReference, POPULATION_NAMES
from .contracts import (
    MODALITIES,
    GroundedSemanticConfig,
    GroundedSemanticRequest,
    GroundedSemanticResult,
    canonical_modality,
)
from .pipeline import GroundedSemanticIntegrationProcessor, run_grounded_semantic_pipeline
from .registry import (
    OPERATION_REGISTRY,
    STAGE_REGISTRY,
    OperationSpec,
    StageSpec,
    registry_as_dicts,
    registry_summary,
)

__version__ = "0.1.0"

__all__ = [
    "MODALITIES",
    "GroundedSemanticConfig",
    "GroundedSemanticRequest",
    "GroundedSemanticResult",
    "canonical_modality",
    "GroundedSemanticIntegrationProcessor",
    "run_grounded_semantic_pipeline",
    "GroundedNeuralReference",
    "POPULATION_NAMES",
    "OperationSpec",
    "StageSpec",
    "STAGE_REGISTRY",
    "OPERATION_REGISTRY",
    "registry_as_dicts",
    "registry_summary",
    "register_grounded_annotations",
    "record_grounded_result",
]
