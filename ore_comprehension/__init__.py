"""ORE top-down predictive and incremental sentence-comprehension subsystem."""
from .biology import POPULATIONS, ComprehensionLIFSimulator, LIFConfig, population_drives
from .contracts import ComprehensionConfig, ComprehensionRequest, ComprehensionResult, tokenize_text
from .pipeline import ComprehensionProcessor
from .registry import OPERATION_REGISTRY, STAGE_REGISTRY, OperationSpec, StageSpec, registry_as_dicts

__all__ = [
    "ComprehensionConfig",
    "ComprehensionLIFSimulator",
    "ComprehensionProcessor",
    "ComprehensionRequest",
    "ComprehensionResult",
    "LIFConfig",
    "OPERATION_REGISTRY",
    "POPULATIONS",
    "STAGE_REGISTRY",
    "OperationSpec",
    "StageSpec",
    "population_drives",
    "registry_as_dicts",
    "tokenize_text",
]
__version__ = "0.1.0"
