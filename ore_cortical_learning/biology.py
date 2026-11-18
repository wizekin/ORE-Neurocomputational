"""Biologically motivated neocortical population and plasticity reference models.

The model uses rate-like cortical populations, recurrent excitation, a shared inhibitory term,
local eligibility traces, gradual synaptic updates, and homeostatic gain regulation. It is a
transparent software analogy, not an anatomically fitted cortical network or physiological twin.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .contracts import CorticalLearningConfig
from .functional import sigmoid


class CorticalPopulationModel:
    """Recurrent rate-population model with local plasticity and homeostatic regulation.

    Input and prediction pathways are distributed across model units. Recurrent weights stand for
    effective connectivity, an inhibitory population summary limits runaway activity, and slow
    reference weights plus importance estimates approximate protection of well-established traces.
    """

    def __init__(self, config: CorticalLearningConfig) -> None:
        """Initialize deterministic population weights, activity, eligibility, and stability state."""
        self.config = config
        rng = np.random.default_rng(config.random_seed)
        input_scale = 0.35 / np.sqrt(config.input_size)
        self.input_weights = rng.normal(0.0, input_scale,
                                       size=(config.cortical_units, config.input_size))
        recurrent_scale = 0.18 / np.sqrt(config.cortical_units)
        recurrent = rng.normal(0.0, recurrent_scale,
                               size=(config.cortical_units, config.cortical_units))
        recurrent *= (rng.random(recurrent.shape) < 0.25)
        np.fill_diagonal(recurrent, 0.0)
        self.recurrent_weights = recurrent
        self.prediction_weights = rng.normal(
            0.0, 0.25 / np.sqrt(config.cortical_units),
            size=(config.output_units, config.cortical_units),
        )
        self.reference_recurrent_weights = self.recurrent_weights.copy()
        self.reference_prediction_weights = self.prediction_weights.copy()
        self.recurrent_importance = np.zeros_like(self.recurrent_weights)
        self.prediction_importance = np.zeros_like(self.prediction_weights)
        self.activity_mean = np.full(config.cortical_units, config.target_activity, dtype=np.float64)
        self.homeostatic_gain = np.ones(config.cortical_units, dtype=np.float64)
        self.homeostatic_bias = np.zeros(config.cortical_units, dtype=np.float64)
        self.previous_activity = np.zeros(config.cortical_units, dtype=np.float64)
        self.eligibility_trace = np.zeros_like(self.recurrent_weights)
        self.last_activity = np.zeros(config.cortical_units, dtype=np.float64)
        self.last_prediction = np.zeros(config.output_units, dtype=np.float64)
        self.last_drive = np.zeros(config.cortical_units, dtype=np.float64)
        self.update_count = 0

    def forward(self, encounter_vector: Sequence[float]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Compute recurrent cortical activity and a downstream prediction for one encounter."""
        vector = np.asarray(encounter_vector, dtype=np.float64).reshape(-1)
        if vector.size != self.config.input_size or not np.all(np.isfinite(vector)):
            raise ValueError("encounter vector must be finite and match configured input_size")
        feedforward = self.input_weights @ vector
        recurrent = self.config.recurrent_gain * (self.recurrent_weights @ self.previous_activity)
        pre_inhibition = feedforward + recurrent + self.homeostatic_bias
        inhibitory_pool = float(np.mean(np.maximum(pre_inhibition, 0.0)))
        inhibition = self.config.inhibitory_gain * inhibitory_pool
        drive = (pre_inhibition - inhibition) * self.homeostatic_gain
        activity = sigmoid(drive)
        prediction = sigmoid(self.prediction_weights @ activity)
        self.last_drive = drive
        self.last_activity = activity
        self.last_prediction = prediction
        report = {
            "mean_feedforward_drive": float(np.mean(np.abs(feedforward))),
            "mean_recurrent_drive": float(np.mean(np.abs(recurrent))),
            "inhibitory_pool_activity": inhibitory_pool,
            "global_inhibitory_signal": inhibition,
            "mean_cortical_activity": float(np.mean(activity)),
            "active_population_fraction": float(np.mean(activity >= 0.5)),
            "prediction_available": True,
            "model": "recurrent_rate_population_with_shared_inhibitory_pool",
        }
        return activity.copy(), prediction.copy(), report

    def compute_eligibility(self, interval_ms: float) -> tuple[np.ndarray, dict[str, Any]]:
        """Advance a decaying local co-activity eligibility trace with current and prior activity."""
        if not np.isfinite(interval_ms):
            raise ValueError("interval_ms must be finite")
        decay = float(np.exp(-abs(float(interval_ms)) / self.config.eligibility_tau_ms))
        previous = self.previous_activity if np.any(self.previous_activity) else self.last_activity
        temporal_coactivity = np.outer(previous, self.last_activity)
        instantaneous = np.outer(self.last_activity, self.last_activity)
        self.eligibility_trace = decay * self.eligibility_trace + (1.0 - decay) * temporal_coactivity
        self.eligibility_trace += 0.15 * instantaneous
        report = {
            "decay_factor": decay,
            "trace_mean_absolute": float(np.mean(np.abs(self.eligibility_trace))),
            "trace_max_absolute": float(np.max(np.abs(self.eligibility_trace))),
            "temporal_coactivity_magnitude": float(np.mean(np.abs(temporal_coactivity))),
            "instantaneous_coactivity_magnitude": float(np.mean(np.abs(instantaneous))),
        }
        return self.eligibility_trace.copy(), report

    def apply_plasticity(
        self,
        activity: Sequence[float],
        prediction_error: Sequence[float] | None,
        gate: float,
        reliability: float,
        timing_factor: float,
    ) -> dict[str, Any]:
        """Apply small covariance/Hebbian and optional predictive corrections with stability protection."""
        current = np.asarray(activity, dtype=np.float64).reshape(-1)
        if current.size != self.config.cortical_units or not np.all(np.isfinite(current)):
            raise ValueError("cortical activity must match population size and be finite")
        if not all(np.isfinite(value) for value in (gate, reliability, timing_factor)):
            raise ValueError("plasticity gate, reliability, and timing factor must be finite")
        if not 0.0 <= gate <= 1.0 or not 0.0 <= reliability <= 1.0 or not -1.0 <= timing_factor <= 1.0:
            raise ValueError("plasticity controls are outside their normalized ranges")
        previous = self.previous_activity if np.any(self.previous_activity) else current
        pre_centered = previous - float(np.mean(previous))
        post_centered = current - float(np.mean(current))
        covariance = np.outer(pre_centered, post_centered)
        eligibility_component = self.eligibility_trace
        local_rule = 0.75 * covariance + 0.25 * eligibility_component
        scale = self.config.learning_rate * gate * reliability * timing_factor
        raw_recurrent_delta = scale * local_rule
        self.recurrent_importance = (
            (1.0 - self.config.importance_decay) * self.recurrent_importance
            + self.config.importance_decay * np.abs(local_rule)
        )
        protection = self.config.interference_strength * self.recurrent_importance * (
            self.recurrent_weights - self.reference_recurrent_weights
        )
        protected_recurrent_delta = raw_recurrent_delta - self.config.learning_rate * protection
        maxed_recurrent_delta = np.clip(
            protected_recurrent_delta, -self.config.max_weight_step, self.config.max_weight_step
        )
        recurrent_before = self.recurrent_weights.copy()
        self.recurrent_weights = np.clip(
            (1.0 - self.config.weight_decay) * self.recurrent_weights + maxed_recurrent_delta,
            -self.config.max_abs_weight, self.config.max_abs_weight,
        )
        output_delta = np.zeros_like(self.prediction_weights)
        if prediction_error is not None:
            error = np.asarray(prediction_error, dtype=np.float64).reshape(-1)
            if error.size != self.config.output_units or not np.all(np.isfinite(error)):
                raise ValueError("prediction_error must match output_units and be finite")
            derivative = self.last_prediction * (1.0 - self.last_prediction)
            output_delta = self.config.learning_rate * gate * reliability * np.outer(
                error * derivative, current
            )
            self.prediction_importance = (
                (1.0 - self.config.importance_decay) * self.prediction_importance
                + self.config.importance_decay * np.abs(output_delta)
            )
            output_protection = self.config.interference_strength * self.prediction_importance * (
                self.prediction_weights - self.reference_prediction_weights
            )
            output_delta -= self.config.learning_rate * output_protection
            output_delta = np.clip(output_delta, -self.config.max_weight_step,
                                   self.config.max_weight_step)
            self.prediction_weights = np.clip(
                (1.0 - self.config.weight_decay) * self.prediction_weights + output_delta,
                -self.config.max_abs_weight, self.config.max_abs_weight,
            )
        reference_rate = self.config.reference_update_rate
        self.reference_recurrent_weights = (
            (1.0 - reference_rate) * self.reference_recurrent_weights
            + reference_rate * self.recurrent_weights
        )
        self.reference_prediction_weights = (
            (1.0 - reference_rate) * self.reference_prediction_weights
            + reference_rate * self.prediction_weights
        )
        self.update_count += 1
        return {
            "rule": "modulated_covariance_hebbian_with_optional_prediction_error",
            "timing_factor": float(timing_factor),
            "modulatory_gate": float(gate),
            "relationship_reliability": float(reliability),
            "mean_absolute_raw_recurrent_delta": float(np.mean(np.abs(raw_recurrent_delta))),
            "mean_absolute_applied_recurrent_delta": float(np.mean(np.abs(self.recurrent_weights - recurrent_before))),
            "recurrent_clipped_fraction": float(np.mean(np.abs(protected_recurrent_delta) > self.config.max_weight_step)),
            "prediction_update_applied": prediction_error is not None,
            "mean_absolute_prediction_delta": float(np.mean(np.abs(output_delta))),
            "update_count": self.update_count,
            "protection_rule": "importance_weighted_reference_constraint_analogy",
        }

    def regulate_homeostasis(self, activity: Sequence[float]) -> dict[str, Any]:
        """Adjust slow population gains and biases toward target activity while retaining responsiveness."""
        current = np.asarray(activity, dtype=np.float64).reshape(-1)
        if current.size != self.config.cortical_units or not np.all(np.isfinite(current)):
            raise ValueError("homeostatic activity must match population size and be finite")
        rate = self.config.homeostatic_rate
        self.activity_mean = (1.0 - rate) * self.activity_mean + rate * current
        error = self.config.target_activity - self.activity_mean
        self.homeostatic_bias = np.clip(self.homeostatic_bias + rate * error, -2.0, 2.0)
        self.homeostatic_gain = np.clip(
            self.homeostatic_gain * np.exp(rate * error),
            self.config.homeostatic_gain_min,
            self.config.homeostatic_gain_max,
        )
        recurrent_norms = np.linalg.norm(self.recurrent_weights, axis=1)
        return {
            "target_activity": self.config.target_activity,
            "mean_recent_activity": float(np.mean(self.activity_mean)),
            "mean_homeostatic_error": float(np.mean(error)),
            "mean_gain": float(np.mean(self.homeostatic_gain)),
            "mean_absolute_bias": float(np.mean(np.abs(self.homeostatic_bias))),
            "max_recurrent_row_norm": float(np.max(recurrent_norms)),
            "mechanism": "slow_intrinsic_gain_and_bias_homeostasis_surrogate",
        }

    def finish_encounter(self) -> None:
        """Store the current population response as the preceding state for the next encounter."""
        self.previous_activity = self.last_activity.copy()

    def snapshot(self) -> dict[str, Any]:
        """Expose population state, weight magnitudes, and explicit limitations of the circuit model."""
        return {
            "model": "distributed_neocortical_rate_population_surrogate",
            "population_size": self.config.cortical_units,
            "input_size": self.config.input_size,
            "output_size": self.config.output_units,
            "mean_activity": float(np.mean(self.last_activity)),
            "mean_activity_history": self.activity_mean.tolist(),
            "mean_recurrent_weight": float(np.mean(self.recurrent_weights)),
            "mean_absolute_recurrent_weight": float(np.mean(np.abs(self.recurrent_weights))),
            "mean_absolute_prediction_weight": float(np.mean(np.abs(self.prediction_weights))),
            "eligibility_trace_mean_absolute": float(np.mean(np.abs(self.eligibility_trace))),
            "update_count": self.update_count,
            "scope": "rate-based computational analogy; not a cell-level or fitted cortical model",
        }
