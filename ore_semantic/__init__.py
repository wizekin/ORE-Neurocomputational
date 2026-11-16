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
from .memory import MEMORY_OPERATION_REGISTRY, SemanticMemoryStore
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
    "SemanticN8Adapter",
    "SemanticRequest",
    "SemanticResult",
    "WordFormEpisode",
    "registry_as_dicts",
    "semantic_request_from_reading_result",
]
