"""ORE's visual acquisition and orthographic-representation processing package."""
from .annotations import register_visual_orthography_annotations
from .biology import BIOLOGICAL_POPULATIONS, BiologicalReferenceSimulator
from .contracts import (
    LetterPrototype,
    LexicalCandidate,
    VisualOrthographyConfig,
    VisualOrthographyRequest,
    VisualOrthographyResult,
)
from .math_ops import compare_orthographic_strings, open_bigram_activations
from .pipeline import VisualOrthographyProcessor
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "BIOLOGICAL_POPULATIONS",
    "BiologicalReferenceSimulator",
    "LetterPrototype",
    "LexicalCandidate",
    "OPERATION_REGISTRY",
    "OperationSpec",
    "VisualOrthographyConfig",
    "VisualOrthographyProcessor",
    "VisualOrthographyRequest",
    "VisualOrthographyResult",
    "compare_orthographic_strings",
    "open_bigram_activations",
    "register_visual_orthography_annotations",
    "registry_as_dicts",
]
