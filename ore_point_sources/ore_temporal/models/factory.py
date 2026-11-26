"""Model-family construction behind feature flags (specification Steps 5 and 24)."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..config import TemporalConfig
from ..contracts import ModelFamily
from .base import ModelFamilyError, TemporalModel
from .baseline import BaselineKind, build_baseline
from .cfc import ClosedFormContinuousNetwork
from .lnn import LiquidNeuralNetwork
from .neural_ode import NeuralODEModel
from .snn import SpikingNeuralNetwork
from .ssm import SelectiveStateSpaceModel

__all__ = ("build_model", "family_available", "list_available_families")

_NEURAL_FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily.LNN, ModelFamily.CFC, ModelFamily.NEURAL_ODE, ModelFamily.SSM, ModelFamily.SNN,
)


def family_available(family: ModelFamily, config: TemporalConfig) -> bool:
    """Report whether a family is enabled under the current feature flags."""
    if family is ModelFamily.BASELINE:
        return True
    if config.baseline_only or not config.enable_neural_families:
        return False
    if family is ModelFamily.SNN and config.snn_encoding not in ("rate", "ttfs"):
        return False
    return family in _NEURAL_FAMILIES


def list_available_families(config: TemporalConfig) -> tuple[ModelFamily, ...]:
    """List families selectable under the current configuration."""
    families = [ModelFamily.BASELINE]
    if not config.baseline_only and config.enable_neural_families:
        families.extend(_NEURAL_FAMILIES)
    return tuple(families)


def build_model(family: ModelFamily, n_channels: int, config: TemporalConfig,
                seed: int | None = None, baseline_kind: str = BaselineKind.KALMAN
                ) -> TemporalModel:
    """Construct a model of the requested family, honouring feature flags."""
    model_seed = config.seed if seed is None else int(seed)
    if not family_available(family, config):
        raise ModelFamilyError(
            f"model family {family.value!r} is unavailable under the current configuration; "
            "baseline-only mode or feature flags restrict neural families")
    if family is ModelFamily.BASELINE:
        return build_baseline(baseline_kind, n_channels, seed=model_seed)
    if family is ModelFamily.LNN:
        return LiquidNeuralNetwork(
            n_channels=n_channels, tau=config.lnn_time_constant,
            solver_method=config.solver_method, max_step=config.solver_max_step,
            max_evaluations=config.max_solver_evaluations,
            state_norm_bound=config.state_norm_bound, seed=model_seed)
    if family is ModelFamily.CFC:
        return ClosedFormContinuousNetwork(
            n_channels=n_channels, tau=config.cfc_time_constant, seed=model_seed)
    if family is ModelFamily.NEURAL_ODE:
        return NeuralODEModel(
            n_channels=n_channels, hidden_units=config.ode_hidden_units,
            solver_method="rk4", max_step=config.solver_max_step,
            max_evaluations=max(config.max_solver_evaluations // 4, 1),
            state_norm_bound=config.state_norm_bound, seed=model_seed)
    if family is ModelFamily.SSM:
        return SelectiveStateSpaceModel(
            n_channels=n_channels, n_state=config.ssm_state_dim,
            state_norm_bound=config.state_norm_bound,
            max_sequence_length=config.max_sequence_length, seed=model_seed)
    if family is ModelFamily.SNN:
        return SpikingNeuralNetwork(
            n_channels=n_channels, tau_m=config.snn_membrane_tau,
            v_rest=config.snn_rest_voltage, v_threshold=config.snn_threshold,
            v_reset=config.snn_reset_voltage, resistance=config.snn_membrane_resistance,
            refractory_period=config.snn_refractory_seconds, encoding=config.snn_encoding,
            rate_window=config.snn_rate_window,
            max_spikes_per_run=config.max_spikes_per_run, seed=model_seed)
    raise ModelFamilyError(f"unknown model family {family!r}")


def ensure_matrix(values: np.ndarray, n_channels: int) -> np.ndarray:
    """Validate and normalize a value matrix for model execution."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[1] != n_channels:
        raise ValueError("values must be a (n_samples, n_channels) matrix")
    return array


def family_descriptions() -> dict[str, Any]:
    """Return a short description of each constructible family."""
    return {
        "baseline": "deterministic control condition (Kalman, smoothing, moving average)",
        "lnn": "continuous-time latent dynamics integrated between observations",
        "cfc": "closed-form exponential updates across irregular intervals",
        "neural_ode": "learned continuous dynamics with a bounded solver",
        "ssm": "selective state-space recurrent memory for long sequences",
        "snn": "leaky integrate-and-fire event-driven layer",
    }
