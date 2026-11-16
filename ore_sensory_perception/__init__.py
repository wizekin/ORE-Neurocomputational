"""ORE integrated six-stage visual and auditory sensory-perception implementation.

The package combines a functional 39-operation algorithm with a separate configurable LIF reference
network and registers all class/function docstrings in the existing ORE SQLite annotation catalogue.
"""
from .annotations import register_sensory_annotations
from .biology import POPULATIONS, LIFPopulation, SensoryBiologicalSimulator
from .contracts import PhonemeSpec, SensoryConfig, SensoryRequest, SensoryResult
from .pipeline import SensoryPerceptionProcessor
from .registry import OPERATION_REGISTRY, STAGE_NAMES, STAGE_STEP_COUNTS, OperationSpec, registry_as_dicts

__all__ = [
    "PhonemeSpec", "SensoryConfig", "SensoryRequest", "SensoryResult",
    "SensoryPerceptionProcessor", "SensoryBiologicalSimulator", "LIFPopulation", "POPULATIONS",
    "OperationSpec", "OPERATION_REGISTRY", "STAGE_NAMES", "STAGE_STEP_COUNTS",
    "registry_as_dicts", "register_sensory_annotations",
]
