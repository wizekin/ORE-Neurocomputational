"""Model-family adapters (speculation Phase C, Steps 17-25)."""
from __future__ import annotations

from .base import (
    ModelFamilyError,
    ModelRunResult,
    ModelStepOutput,
    TemporalModel,
    run_model_over_window,
)
from .baseline import (
    BaselineKind,
    ExponentialSmoothingBaseline,
    MovingAverageBaseline,
    SimpleKalmanBaseline,
    ThresholdDetectorBaseline,
    build_baseline,
)
from .cfc import ClosedFormContinuousNetwork
from .lnn import LiquidNeuralNetwork
from .neural_ode import NeuralODEModel
from .snn import SpikingNeuralNetwork
from .ssm import SelectiveStateSpaceModel
from .state import ModelStateRecord, ModelStateStore, StateCompatibilityError
from .factory import (build_model, family_available, family_descriptions,
                      list_available_families)

__all__ = (
    "BaselineKind",
    "ClosedFormContinuousNetwork",
    "ExponentialSmoothingBaseline",
    "LiquidNeuralNetwork",
    "ModelFamilyError",
    "ModelRunResult",
    "ModelStateRecord",
    "ModelStateStore",
    "ModelStepOutput",
    "MovingAverageBaseline",
    "NeuralODEModel",
    "SelectiveStateSpaceModel",
    "SimpleKalmanBaseline",
    "SpikingNeuralNetwork",
    "StateCompatibilityError",
    "TemporalModel",
    "ThresholdDetectorBaseline",
    "build_baseline",
    "build_model",
    "family_available",
    "family_descriptions",
    "list_available_families",
    "run_model_over_window",
)
