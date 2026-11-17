"""ORE 36-stage Distributed Neural Semantic Composition and Integration package."""
from .annotations import (
    index_distributed_semantic_annotations,
    record_distributed_semantic_result,
)
from .biology import POPULATION_NAMES, STAGE_POPULATIONS, SemanticLIFSimulator
from .contracts import (
    MODALITIES,
    DistributedSemanticConfig,
    DistributedSemanticRequest,
    DistributedSemanticResult,
    canonical_modality,
    json_safe,
    normalize_modalities,
)
from .pipeline import (
    DistributedSemanticProcessor,
    operation_trace_as_dicts,
    request_from_text,
)
from .registry import (
    ALL_OPERATIONS,
    STAGE_REGISTRY,
    OperationSpec,
    StageSpec,
    registry_as_dicts,
)

__all__ = [
    "ALL_OPERATIONS",
    "MODALITIES",
    "POPULATION_NAMES",
    "STAGE_POPULATIONS",
    "STAGE_REGISTRY",
    "DistributedSemanticConfig",
    "DistributedSemanticProcessor",
    "DistributedSemanticRequest",
    "DistributedSemanticResult",
    "OperationSpec",
    "SemanticLIFSimulator",
    "StageSpec",
    "canonical_modality",
    "index_distributed_semantic_annotations",
    "json_safe",
    "normalize_modalities",
    "operation_trace_as_dicts",
    "record_distributed_semantic_result",
    "registry_as_dicts",
    "request_from_text",
]
