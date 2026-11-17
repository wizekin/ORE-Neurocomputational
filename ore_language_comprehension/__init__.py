"""Source-ordered language-comprehension algorithms with functional and biological-reference tracks.

Point 1 wraps ORE's established low-level sensory processing implementation in the 12-step
perceptual-to-word-form algorithm from the Language Comprehension Brain Algorithms source.
"""
from .annotations import register_point01_annotations, register_point02_annotations, register_point03_annotations
from .point02 import (
    POINT2_STEP_SPECS,
    Point2LexicalAccessProcessor,
    Point2Request,
    Point2Result,
)
from .point03 import POINT3_STEP_SPECS, Point3Request, Point3Result, Point3SyntacticParser
from .point01 import (
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
]
