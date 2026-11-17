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


from .point10 import POINT10_METHODS, POINT10_STEP_SPECS, Point10EvidenceValidator, Point10Request, Point10Result
from .annotations import register_point10_annotations

__all__ += [
    "POINT10_METHODS", "POINT10_STEP_SPECS", "Point10Request", "Point10Result",
    "Point10EvidenceValidator", "register_point10_annotations",
]


from .point11 import (
    POINT11_DEFAULT_CONNECTIONS,
    POINT11_DEMAND_KEYS,
    POINT11_STEP_SPECS,
    POINT11_SYSTEMS,
    Point11FunctionalSeparationCoordinator,
    Point11LIFReference,
    Point11Request,
    Point11Result,
    Point11State,
)
from .annotations import register_point11_annotations

__all__ += [
    "POINT11_DEFAULT_CONNECTIONS", "POINT11_DEMAND_KEYS", "POINT11_STEP_SPECS",
    "POINT11_SYSTEMS", "Point11Request", "Point11State", "Point11Result",
    "Point11LIFReference", "Point11FunctionalSeparationCoordinator",
    "register_point11_annotations",
]


from .point12 import POINT12_METHODS, POINT12_STEP_SPECS, Point12ModelEvaluator, Point12Request, Point12Result
from .annotations import register_point12_annotations

__all__ += [
    "POINT12_METHODS", "POINT12_STEP_SPECS", "Point12Request", "Point12Result",
    "Point12ModelEvaluator", "register_point12_annotations",
]


from .point13 import (
    POINT13_COMPONENTS,
    POINT13_STEP_NAMES,
    POINT13_STEP_SPECS,
    Point13IndividualizedProcessor,
    Point13Request,
    Point13Result,
    register_point13_annotations,
)

__all__ += [
    "POINT13_COMPONENTS", "POINT13_STEP_NAMES", "POINT13_STEP_SPECS",
    "Point13Request", "Point13Result", "Point13IndividualizedProcessor",
    "register_point13_annotations",
]


from .point14 import (
    POINT14_COMPONENTS,
    POINT14_PROFILE_FAMILIES,
    POINT14_STEP_NAMES,
    POINT14_STEP_SPECS,
    Point14ClinicalProfileProcessor,
    Point14Request,
    Point14Result,
    register_point14_annotations,
)

__all__ += [
    "POINT14_COMPONENTS", "POINT14_PROFILE_FAMILIES", "POINT14_STEP_NAMES",
    "POINT14_STEP_SPECS", "Point14Request", "Point14Result",
    "Point14ClinicalProfileProcessor", "register_point14_annotations",
]
