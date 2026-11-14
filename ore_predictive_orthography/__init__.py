"""ORE seven-stage master predictive orthography algorithm."""
from .biology import BIOLOGICAL_STAGE_NAMES, LIFConfig, simulate_lif_reference
from .contracts import PredictiveOrthographyResult
from .pipeline import PredictiveOrthographyProcessor
from .registry import (MASTER_OPERATION_REGISTRY, MASTER_STAGE_REGISTRY,
                       OperationSpec, StageSpec, registry_as_dict)

__all__ = [
    "BIOLOGICAL_STAGE_NAMES", "LIFConfig", "simulate_lif_reference",
    "PredictiveOrthographyResult", "PredictiveOrthographyProcessor",
    "MASTER_OPERATION_REGISTRY", "MASTER_STAGE_REGISTRY",
    "OperationSpec", "StageSpec", "registry_as_dict",
]
__version__ = "0.1.0"
