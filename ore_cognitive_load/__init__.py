"""ORE Point 06: minimize cognitive load using functional and biological-reference models."""
from .annotations import register_annotations
from .biology import DistributedCognitiveLoadNetwork, RatePopulation
from .contracts import CognitiveUnit, Point06Config, Point06Request, Point06Result, Point06State
from .functional import (
    adapt_learning_step, build_external_representation, build_guided_scaffold,
    clip01, estimate_element_interactivity, feature_similarity,
    form_meaningful_chunks, integrate_chunks, monitor_overload,
    order_by_prerequisites, reduce_presentation_demands, retrieve_prerequisites,
    segment_material, select_principal_subgoal, select_working_set, softmax,
    update_retrieval_and_support, verify_comprehension,
)
from .pipeline import Point06CognitiveLoadProcessor, record_result_to_sqlite
from .registry import OPERATION_REGISTRY, Point06OperationSpec, registry_as_dicts

__all__ = [
    "CognitiveUnit", "Point06Config", "Point06Request", "Point06Result", "Point06State",
    "RatePopulation", "DistributedCognitiveLoadNetwork", "Point06OperationSpec",
    "OPERATION_REGISTRY", "registry_as_dicts", "Point06CognitiveLoadProcessor",
    "register_annotations", "record_result_to_sqlite", "clip01", "softmax",
    "feature_similarity", "segment_material", "estimate_element_interactivity",
    "retrieve_prerequisites", "form_meaningful_chunks", "order_by_prerequisites",
    "select_principal_subgoal", "reduce_presentation_demands", "build_external_representation",
    "select_working_set", "monitor_overload", "adapt_learning_step", "build_guided_scaffold",
    "integrate_chunks", "update_retrieval_and_support", "verify_comprehension",
]