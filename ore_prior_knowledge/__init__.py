"""ORE Point 02: connect new information to prior knowledge and schemas."""
from .annotations import register_annotations
from .biology import (
    BiologicalPriorKnowledgeNetwork,
    DistributedCorticalAssociationNetwork,
    FrontoparietalRetrievalControlNetwork,
    HippocampalPatternCompletionNetwork,
    MedialPrefrontalSchemaNetwork,
    NeuralPopulation,
    NeuromodulatoryLearningGate,
)
from .contracts import (
    KnowledgeNode,
    KnowledgeRelation,
    KnowledgeSchema,
    LearningUnit,
    PriorKnowledgeConfig,
    PriorKnowledgeRequest,
    PriorKnowledgeResult,
    PriorKnowledgeState,
    RelationInput,
)
from .functional import (
    align_analogies,
    build_integrated_representation,
    build_retrieval_routes,
    choose_integration_strategy,
    clamp,
    compare_knowledge,
    cosine_similarity,
    evaluate_schema_fit,
    extract_new_representation,
    generate_retrieval_cues,
    rank_candidates,
    retrieve_candidates,
    shannon_entropy,
    softmax,
    stable_id,
    test_retrieval_routes,
    tokenize,
    vectorize_tokens,
)
from .pipeline import PriorKnowledgeProcessor, record_result_to_sqlite
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "RelationInput", "LearningUnit", "KnowledgeNode", "KnowledgeRelation", "KnowledgeSchema",
    "PriorKnowledgeConfig", "PriorKnowledgeState", "PriorKnowledgeRequest", "PriorKnowledgeResult",
    "NeuralPopulation", "HippocampalPatternCompletionNetwork",
    "DistributedCorticalAssociationNetwork", "MedialPrefrontalSchemaNetwork",
    "FrontoparietalRetrievalControlNetwork", "NeuromodulatoryLearningGate",
    "BiologicalPriorKnowledgeNetwork", "OperationSpec", "OPERATION_REGISTRY",
    "registry_as_dicts", "PriorKnowledgeProcessor", "register_annotations",
    "record_result_to_sqlite", "clamp", "tokenize", "softmax", "shannon_entropy",
    "cosine_similarity", "vectorize_tokens", "extract_new_representation",
    "generate_retrieval_cues", "retrieve_candidates", "rank_candidates",
    "compare_knowledge", "align_analogies", "evaluate_schema_fit",
    "choose_integration_strategy", "build_integrated_representation",
    "build_retrieval_routes", "test_retrieval_routes", "stable_id",
]
