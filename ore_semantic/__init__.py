"""ORE semantic-cognition subsystem: 44 operations, functional and LIF reference tracks."""
from .adapters import SemanticN8Adapter, semantic_request_from_reading_result
from .biology import POPULATIONS, SemanticLIFSimulator
from .contracts import (
    SemanticConcept,
    SemanticConfig,
    SemanticRequest,
    SemanticResult,
    WordFormEpisode,
)
from .pipeline import SemanticCognitionProcessor
from .memory import MEMORY_OPERATION_REGISTRY, SemanticMemoryStore, SemanticPrototypeEvidence
from .source_mapping import SOURCE_COMPONENTS, SOURCE_OPERATION_MAP, source_mapping_as_dicts
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "OPERATION_REGISTRY",
    "OperationSpec",
    "POPULATIONS",
    "SemanticConcept",
    "SemanticConfig",
    "SemanticCognitionProcessor",
    "SemanticLIFSimulator",
    "SemanticMemoryStore",
    "MEMORY_OPERATION_REGISTRY",
    "SemanticPrototypeEvidence",
    "SOURCE_COMPONENTS",
    "SOURCE_OPERATION_MAP",
    "source_mapping_as_dicts",
    "SemanticN8Adapter",
    "SemanticRequest",
    "SemanticResult",
    "WordFormEpisode",
    "registry_as_dicts",
    "semantic_request_from_reading_result",
]
