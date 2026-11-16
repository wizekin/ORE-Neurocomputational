"""Integrated Reading Algorithm public API."""
from .adapters import reading_request_from_n6_result
from .biology import BiologicalReadingSimulator, LIFConfig, LIFPopulation
from .contracts import G2PRule, ReadingConfig, ReadingLexiconEntry, ReadingRequest, ReadingResult
from .pipeline import IntegratedReadingProcessor, default_g2p_rules
from .modalities import (
    AuditoryConfig, AuditoryInputAdapter, AuditoryInputResult, AttentionPlan,
    ReadingAttentionController, SOURCE_OPERATION_IMPLEMENTATIONS,
    SpeechOutputAdapter, SpeechOutputPlan, SpeechSynthesisBackend,
)
from .registry import OPERATION_BY_ID, OPERATION_REGISTRY, OperationSpec, get_operation, registry_as_dicts

__all__ = [
    "AuditoryConfig", "AuditoryInputAdapter", "AuditoryInputResult", "AttentionPlan",
    "BiologicalReadingSimulator", "G2PRule", "IntegratedReadingProcessor", "LIFConfig",
    "ReadingAttentionController", "SOURCE_OPERATION_IMPLEMENTATIONS",
    "SpeechOutputAdapter", "SpeechOutputPlan", "SpeechSynthesisBackend",
    "LIFPopulation", "OPERATION_BY_ID", "OPERATION_REGISTRY", "OperationSpec",
    "ReadingConfig", "ReadingLexiconEntry", "ReadingRequest", "ReadingResult",
    "default_g2p_rules", "get_operation", "registry_as_dicts",
    "reading_request_from_n6_result",
]
