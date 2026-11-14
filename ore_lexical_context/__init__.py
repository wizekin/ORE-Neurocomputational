"""ORE lexical activation, morphology, competition and prediction-error subsystem."""
from .adapters import lexical_context_request_from_reading_result
from .biology import POPULATIONS, LexicalContextLIFSimulator
from .contracts import (G2PRule, LexicalContextConfig, LexicalContextRequest,
    LexicalContextResult, LexicalEntry, LexicalMeaning, MorphemeEntry)
from .pipeline import LexicalContextProcessor
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = ["G2PRule","LexicalContextConfig","LexicalContextLIFSimulator",
    "LexicalContextProcessor","LexicalContextRequest","LexicalContextResult",
    "LexicalEntry","LexicalMeaning","MorphemeEntry","OPERATION_REGISTRY",
    "OperationSpec","POPULATIONS","lexical_context_request_from_reading_result",
    "registry_as_dicts"]
