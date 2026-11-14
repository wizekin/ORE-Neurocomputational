"""ORE seven-stage lexical access and morphology architecture."""
from .biology import POPULATIONS, LexicalAccessLIFSimulator
from .contracts import LexicalAccessConfig, LexicalAccessRequest, LexicalAccessResult
from .pipeline import LexicalAccessProcessor
from .registry import OPERATION_REGISTRY, STAGE_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "LexicalAccessConfig", "LexicalAccessLIFSimulator", "LexicalAccessProcessor",
    "LexicalAccessRequest", "LexicalAccessResult", "OPERATION_REGISTRY",
    "OperationSpec", "POPULATIONS", "STAGE_REGISTRY", "registry_as_dicts",
]
