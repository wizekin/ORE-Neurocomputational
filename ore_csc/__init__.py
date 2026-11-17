"""ORE Controlled Semantic Cognition: 18 stages and a separate LIF reference track."""

from .annotations import index_csc_annotations, record_csc_result
from .biology import POPULATION_NAMES, CSCLIFSimulator
from .contracts import CSCConfig, CSCRequest, CSCResult
from .pipeline import CSCProcessor
from .registry import ALL_OPERATIONS, STAGE_REGISTRY, OperationSpec, StageSpec, registry_as_dicts

__all__ = [
    "ALL_OPERATIONS",
    "POPULATION_NAMES",
    "CSCConfig",
    "CSCProcessor",
    "CSCRequest",
    "CSCResult",
    "CSCLIFSimulator",
    "OperationSpec",
    "STAGE_REGISTRY",
    "StageSpec",
    "index_csc_annotations",
    "record_csc_result",
    "registry_as_dicts",
]
