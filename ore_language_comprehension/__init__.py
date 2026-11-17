"""Source-ordered language-comprehension algorithms with functional and biological-reference tracks.

Point 1 wraps ORE's established low-level sensory processing implementation in the 12-step
perceptual-to-word-form algorithm from the Language Comprehension Brain Algorithms source.
"""
from .annotations import register_point01_annotations, register_point02_annotations, register_point03_annotations, register_point04_annotations, register_point05_annotations, register_point06_annotations, register_point07_annotations
from .point02 import (
    POINT2_STEP_SPECS,
    Point2LexicalAccessProcessor,
    Point2Request,
    Point2Result,
)
from .point03 import POINT3_STEP_SPECS, Point3Request, Point3Result, Point3SyntacticParser
from .point04 import POINT4_STEP_SPECS, Point4Request, Point4Result, Point4SemanticComposer
from .point05 import POINT5_STEP_SPECS, Point5ContextualPragmaticIntegrator, Point5Request, Point5Result
from .point06 import POINT6_STEP_SPECS, Point6PropositionalWorkingMemory, Point6Request, Point6Result
from .point07 import (\n    POINT7_STEP_SPECS,\n    Point7LIFReference,\n    Point7MemoryDecisionProcessor,\n    Point7MemoryStore,\n    Point7Processor,\n    Point7Request,\n    Point7Result,\n)\nfrom .point01 import (
    POINT1_STEP_SPECS,
    Point1PerceptualProcessor,
    Point1Result,
)

__all__ = [
    "POINT1_STEP_SPECS",
    "Point1PerceptualProcessor",
    "Point1Result",
    "register_point01_annotations",
    "POINT2_STEP_SPECS",
    "Point2LexicalAccessProcessor",
    "Point2Request",
    "Point2Result",
    "register_point02_annotations",
    "POINT3_STEP_SPECS",
    "Point3Request",
    "Point3Result",
    "Point3SyntacticParser",
    "register_point03_annotations",
    "POINT4_STEP_SPECS",
    "Point4Request",
    "Point4Result",
    "Point4SemanticComposer",
    "register_point04_annotations",
    "POINT5_STEP_SPECS",
    "Point5Request",
    "Point5Result",
    "Point5ContextualPragmaticIntegrator",
    "register_point05_annotations",
    "POINT6_STEP_SPECS",
    "Point6Request",
    "Point6Result",
    "Point6PropositionalWorkingMemory",
    "register_point06_annotations",
]
