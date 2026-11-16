"""ORE N9 integrated neurocomputational controller."""
from .contracts import N9Adapters, N9Config, N9Input, N9Patch, N9Request, N9Result
from .pipeline import N9_ALGORITHM_STEPS, N9Processor
from .adapters import (
    LexicalAccessN6Adapter,
    SemanticCompositionN8Adapter,
    SyntaxN9Adapter,
    build_n9_language_adapters,
)

__all__ = [
    "N9Adapters",
    "N9Config",
    "N9Input",
    "N9Patch",
    "N9Request",
    "N9Result",
    "N9_ALGORITHM_STEPS",
    "N9Processor",
    "LexicalAccessN6Adapter",
    "SemanticCompositionN8Adapter",
    "SyntaxN9Adapter",
    "build_n9_language_adapters",
]
__version__ = "0.1.0"
