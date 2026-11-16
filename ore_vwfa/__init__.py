"""ORE six-stage Visual Word Form Area (VWFA) functional and biological-reference implementation."""
from .annotations import register_vwfa_annotations
from .biology import POPULATIONS, VWFABiologicalSimulator
from .contracts import (
    GraphemePrototype, LexicalEntry, MorphemeEntry, VWFAConfig, VWFAExample,
    VWFARequest, VWFAResult,
)
from .pipeline import VWFAProcessor
from .registry import OPERATION_REGISTRY, STAGE_STEP_COUNTS, OperationSpec, registry_as_dicts

__all__ = [
    "GraphemePrototype", "LexicalEntry", "MorphemeEntry", "VWFAConfig", "VWFAExample",
    "VWFARequest", "VWFAResult", "VWFAProcessor", "VWFABiologicalSimulator", "POPULATIONS",
    "OperationSpec", "OPERATION_REGISTRY", "STAGE_STEP_COUNTS", "registry_as_dicts",
    "register_vwfa_annotations",
]
