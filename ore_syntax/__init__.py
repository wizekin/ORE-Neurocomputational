"""ORE nine-stage syntax-processing subsystem."""
from .biology import POPULATIONS, SyntaxLIFSimulator
from .contracts import SyntaxConfig, SyntaxRequest, SyntaxResult
from .pipeline import OREParser, SyntaxProcessor
from .registry import OPERATION_REGISTRY, STAGE_REGISTRY, OperationSpec, StageSpec, registry_as_dicts

__all__ = [
    "OREParser", "OPERATION_REGISTRY", "POPULATIONS", "STAGE_REGISTRY",
    "OperationSpec", "StageSpec", "SyntaxConfig", "SyntaxLIFSimulator",
    "SyntaxProcessor", "SyntaxRequest", "SyntaxResult", "registry_as_dicts",
]
