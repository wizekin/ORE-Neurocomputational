"""Connect ORE's episodic and semantic stores to its shared cognitive workspace."""
from .adapters import (
    AnalogicalMemoryAdapter,
    ReconstructiveMemoryAdapter,
    SchemaGuidedInferenceAdapter,
    SchemaMemoryAdapter,
    SemanticMemoryAdapter,
    WorkingMemoryAdapter,
)
from .annotations import register_annotations
from .contracts import (
    MemoryBatch,
    MemoryEvidence,
    MemoryHandoff,
    MemoryIntegrationConfig,
    MemoryIntegrationResult,
    MemoryQuery,
)
from .coordinator import (
    CognitiveMemoryRouter,
    MemoryEventHandler,
    MemoryIntegrationCoordinator,
)
from .policies import (
    EvidenceValidationPolicy,
    MemoryEncodingPolicy,
    MemoryRelevancePolicy,
    SourceIndependencePolicy,
)
from .registry import MEMORY_INTEGRATION_OPERATIONS, operation_registry_as_dicts
from .retrieval import (
    EpisodicRetrievalAdapter,
    MemoryRankingEngine,
    MemoryRetrievalCoordinator,
    SemanticRetrievalAdapter,
)

__all__ = [
    "AnalogicalMemoryAdapter",
    "CognitiveMemoryRouter",
    "EpisodicRetrievalAdapter",
    "EvidenceValidationPolicy",
    "MEMORY_INTEGRATION_OPERATIONS",
    "MemoryBatch",
    "MemoryEncodingPolicy",
    "MemoryEvidence",
    "MemoryEventHandler",
    "MemoryHandoff",
    "MemoryIntegrationConfig",
    "MemoryIntegrationCoordinator",
    "MemoryIntegrationResult",
    "MemoryQuery",
    "MemoryRankingEngine",
    "MemoryRelevancePolicy",
    "MemoryRetrievalCoordinator",
    "ReconstructiveMemoryAdapter",
    "SchemaGuidedInferenceAdapter",
    "SchemaMemoryAdapter",
    "SemanticMemoryAdapter",
    "SemanticRetrievalAdapter",
    "SourceIndependencePolicy",
    "WorkingMemoryAdapter",
    "operation_registry_as_dicts",
    "register_annotations",
]
