"""ORE Point 11 cross-domain relational analogical transfer package."""

from .annotations import register_annotations
from .biology import (
    BiologicalAnalogicalNetwork,
    DorsolateralPrefrontalWorkingMemoryNetwork,
    HippocampalRelationalRetrievalNetwork,
    PosteriorParietalRelationalRepresentationNetwork,
    RostrolateralPrefrontalIntegrationNetwork,
    TemporalSemanticRelationNetwork,
    VentrolateralPrefrontalInterferenceControlNetwork,
)
from .contracts import (
    AnalogicalConfig,
    AnalogicalEntity,
    AnalogicalRelation,
    AnalogicalTransferResult,
    AnalogySchema,
    Disanalogy,
    EntityCorrespondence,
    HigherOrderRelation,
    MappingCandidate,
    ParameterBridge,
    SituationGraph,
    SourceCandidate,
    TransferConstraint,
    TransferHypothesis,
    TransferValidation,
)
from .functional import (
    abstract_relation_templates,
    build_transfer_hypotheses,
    candidate_entity_correspondences,
    canonical_relation_name,
    generate_partial_mappings,
    prioritize_working_memory,
    rank_mappings,
    recurrent_relation_activation,
    relation_similarity,
    retrieve_sources,
    score_mapping,
    translate_parameters,
    validate_transfer,
)
from .pipeline import AnalogicalTransferProcessor, record_result_to_sqlite
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "AnalogicalEntity", "AnalogicalRelation", "HigherOrderRelation",
    "TransferConstraint", "SituationGraph", "ParameterBridge",
    "EntityCorrespondence", "MappingCandidate", "TransferHypothesis",
    "Disanalogy", "AnalogySchema", "AnalogicalConfig", "SourceCandidate",
    "TransferValidation", "AnalogicalTransferResult",
    "HippocampalRelationalRetrievalNetwork",
    "PosteriorParietalRelationalRepresentationNetwork",
    "RostrolateralPrefrontalIntegrationNetwork",
    "DorsolateralPrefrontalWorkingMemoryNetwork",
    "VentrolateralPrefrontalInterferenceControlNetwork",
    "TemporalSemanticRelationNetwork", "BiologicalAnalogicalNetwork",
    "AnalogicalTransferProcessor", "OPERATION_REGISTRY", "OperationSpec",
    "registry_as_dicts", "register_annotations", "record_result_to_sqlite",
    "canonical_relation_name", "relation_similarity",
    "candidate_entity_correspondences", "generate_partial_mappings",
    "score_mapping", "rank_mappings", "retrieve_sources",
    "prioritize_working_memory", "recurrent_relation_activation",
    "build_transfer_hypotheses", "translate_parameters",
    "abstract_relation_templates", "validate_transfer",
]
