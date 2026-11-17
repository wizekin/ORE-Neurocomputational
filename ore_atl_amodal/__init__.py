"""ORE 19-stage amodal semantic processing through bilateral anterior temporal lobes."""

from .annotations import index_atl_amodal_annotations, record_atl_amodal_result
from .biology import ATLBiologySimulator, MODALITY_POPULATIONS, POPULATION_NAMES
from .contracts import (
    MODALITIES,
    AmodalATLConfig,
    AmodalATLRequest,
    AmodalATLResult,
    AmodalConcept,
    canonical_modality,
    normalize_modalities,
)
from .pipeline import AmodalATLProcessor
from .registry import ALL_OPERATIONS, STAGE_REGISTRY, OperationSpec, StageSpec, registry_as_dicts

__all__ = [
    "ALL_OPERATIONS",
    "MODALITIES",
    "MODALITY_POPULATIONS",
    "POPULATION_NAMES",
    "ATLBiologySimulator",
    "AmodalATLConfig",
    "AmodalATLProcessor",
    "AmodalATLRequest",
    "AmodalATLResult",
    "AmodalConcept",
    "OperationSpec",
    "STAGE_REGISTRY",
    "StageSpec",
    "canonical_modality",
    "index_atl_amodal_annotations",
    "normalize_modalities",
    "record_atl_amodal_result",
    "registry_as_dicts",
]
