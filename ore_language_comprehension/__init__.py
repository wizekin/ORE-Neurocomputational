"""Source-ordered language-comprehension algorithms with functional and biological-reference tracks."""

from .annotations import (
    register_point01_annotations,
    register_point02_annotations,
    register_point03_annotations,
    register_point04_annotations,
    register_point05_annotations,
    register_point06_annotations,
    register_point07_annotations,
    register_point08_annotations,
)
from .point01 import POINT1_STEP_SPECS, Point1PerceptualProcessor, Point1Result
from .point02 import POINT2_STEP_SPECS, Point2LexicalAccessProcessor, Point2Request, Point2Result
from .point03 import POINT3_STEP_SPECS, Point3Request, Point3Result, Point3SyntacticParser
from .point04 import POINT4_STEP_SPECS, Point4Request, Point4Result, Point4SemanticComposer
from .point05 import (
    POINT5_STEP_SPECS,
    Point5ContextualPragmaticIntegrator,
    Point5Request,
    Point5Result,
)
from .point06 import (
    POINT6_STEP_SPECS,
    Point6PropositionalWorkingMemory,
    Point6Request,
    Point6Result,
)
from .point07 import (
    POINT7_STEP_SPECS,
    Point7LIFReference,
    Point7MemoryDecisionProcessor,
    Point7MemoryStore,
    Point7Processor,
    Point7Request,
    Point7Result,
)

__all__ = [
    "POINT1_STEP_SPECS", "Point1PerceptualProcessor", "Point1Result",
    "register_point01_annotations",
    "POINT2_STEP_SPECS", "Point2LexicalAccessProcessor", "Point2Request", "Point2Result",
    "register_point02_annotations",
    "POINT3_STEP_SPECS", "Point3Request", "Point3Result", "Point3SyntacticParser",
    "register_point03_annotations",
    "POINT4_STEP_SPECS", "Point4Request", "Point4Result", "Point4SemanticComposer",
    "register_point04_annotations",
    "POINT5_STEP_SPECS", "Point5Request", "Point5Result",
    "Point5ContextualPragmaticIntegrator", "register_point05_annotations",
    "POINT6_STEP_SPECS", "Point6Request", "Point6Result",
    "Point6PropositionalWorkingMemory", "register_point06_annotations",
    "POINT7_STEP_SPECS", "Point7Request", "Point7Result", "Point7MemoryStore",
    "Point7LIFReference", "Point7MemoryDecisionProcessor", "Point7Processor",
    "register_point07_annotations",
]

from .point08 import (
    POINT8_STAGE_NAMES,
    POINT8_STEP_SPECS,
    Point8CrossStageIntegrator,
    Point8LIFReference,
    Point8Processor,
    Point8Request,
    Point8Result,
    Point8State,
)

__all__ += [
    "POINT8_STAGE_NAMES", "POINT8_STEP_SPECS",
    "Point8Request", "Point8State", "Point8Result",
    "Point8LIFReference", "Point8CrossStageIntegrator", "Point8Processor",
    "register_point08_annotations",
]


from .point09 import (
    POINT9_DEFAULT_CONNECTIONS,
    POINT9_STEP_SPECS,
    POINT9_SYSTEM_NAMES,
    Point9LIFReference,
    Point9NeuralArchitectureCoordinator,
    Point9Request,
    Point9Result,
    Point9State,
)
from .annotations import register_point09_annotations

__all__ += [
    "POINT9_DEFAULT_CONNECTIONS", "POINT9_STEP_SPECS", "POINT9_SYSTEM_NAMES",
    "Point9Request", "Point9State", "Point9Result",
    "Point9LIFReference", "Point9NeuralArchitectureCoordinator",
    "register_point09_annotations",
]
