"""Biologically motivated Master Neurocomputational ORE (BNBEA)."""

from .biology import AssociativeMemory, LIFNeuron, NeuralPopulation, NeuralSimulationEngine, SynapticProjection, phase_coherence
from .config import BNBEAConfig
from .contracts import BNBEARequest, BNBEAResult, BrainValidationInput
from .operation_registry import OPERATION_SPECS, STAGES, STAGE_OPERATION_COUNTS
from .pipeline import BNBEAProcessor

__all__ = [
    "BNBEAConfig", "BNBEARequest", "BNBEAResult", "BrainValidationInput",
    "BNBEAProcessor", "LIFNeuron", "NeuralPopulation", "SynapticProjection",
    "NeuralSimulationEngine", "AssociativeMemory", "phase_coherence",
    "STAGES", "OPERATION_SPECS", "STAGE_OPERATION_COUNTS",
]
__version__ = "0.1.0"
