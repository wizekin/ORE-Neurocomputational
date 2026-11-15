"""ORE's Brain Networks and Biological Evidence Algorithm (BNBEA) package.

The API exposes functional reading representations, a separate LIF biological-reference track, and
experimental measurement/validation models without treating the three as interchangeable evidence.
"""
from .annotations import register_bnbea_annotations
from .contracts import BNBEAConfig, BNBEARequest, BNBEAResult, GraphemePrototype, LexicalEntry
from .pipeline import BNBEAProcessor
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "BNBEAConfig", "BNBEARequest", "BNBEAResult", "BNBEAProcessor",
    "GraphemePrototype", "LexicalEntry", "OperationSpec", "OPERATION_REGISTRY",
    "registry_as_dicts", "register_bnbea_annotations",
]
