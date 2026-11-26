"""Isolated Neural Algorithmic Reasoning research track (ORE Point 8 Phase E, steps 41-48).

Narrow tasks, deterministic traces, an algorithmic GNN with stepwise/final supervision,
OOD generalization evaluation, and a production gate. The track is isolated from
production USG writes: every learned output is advisory and deterministically checked.
"""
from .algorithmic_gnn import (
    AlgorithmicProposal,
    NARModelConfig,
    NARPrediction,
    NARTrainer,
    NARTrainingReport,
    NeuralAlgorithmicService,
    TrainedNARModel,
)
from .generalization_eval import (
    NAREvaluationReport,
    NAREvaluator,
    NARGateDecision,
    NARProductionGate,
    NARUsePolicy,
    TaskEvaluation,
)
from .task_generators import (
    NAR_REFERENCE_VERSION,
    NAR_TASK_TYPES,
    TOPOLOGIES,
    TRAINING_TOPOLOGIES,
    NarrowTask,
    ReferenceOutput,
    TaskGenerator,
    reference_solve,
)
from .trace_dataset import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    AlgorithmTrace,
    TraceDataset,
    build_trace,
)

__all__ = [
    "AlgorithmicProposal", "AlgorithmTrace", "EDGE_FEATURE_DIM", "NODE_FEATURE_DIM",
    "NAR_REFERENCE_VERSION", "NAR_TASK_TYPES", "NAREvaluationReport", "NAREvaluator",
    "NARGateDecision", "NARModelConfig", "NARPrediction", "NARProductionGate",
    "NARTrainer", "NARTrainingReport", "NARUsePolicy", "NarrowTask",
    "NeuralAlgorithmicService", "ReferenceOutput", "TOPOLOGIES", "TRAINING_TOPOLOGIES",
    "TaskEvaluation", "TaskGenerator", "TraceDataset", "TrainedNARModel",
    "build_trace", "reference_solve",
]
