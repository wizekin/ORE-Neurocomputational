"""Source-ordered language-comprehension algorithms with functional and biological-reference tracks.

Point 1 wraps ORE's established low-level sensory processing implementation in the 12-step
perceptual-to-word-form algorithm from the Language Comprehension Brain Algorithms source.
"""
from .annotations import register_point01_annotations
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
]
