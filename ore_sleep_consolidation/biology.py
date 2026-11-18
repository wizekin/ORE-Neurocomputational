"""Biologically motivated hippocampal and cortical population dynamics for Point 03.

This is a rate-population reference model with recurrent completion, sequence replay, effective
hippocampal-cortical projections, cortical eligibility traces, bounded plasticity, and homeostasis.
It is not an anatomically fitted network or a digital twin of a human brain.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .contracts import HippocampalTrace, SleepConsolidationConfig
from .functional import cosine_similarity, fit_vector, normalize_vector, sigmoid, stdp_timing_kernel


class HippocampalCorticalNetwork:
    """Maintain coupled hippocampal and cortical population surrogates across replay events.

    The hippocampal recurrent matrix approximates associative completion. Effective hippocampal-
    cortical weights drive cortical population activity, while cortical recurrent weights receive
    bounded timing-dependent updates. A prototype store maintains distinct trace-specific cortical
    patterns so shared features can be integrated without collapsing all episodes into one vector.
    """

    def __init__(self, config: SleepConsolidationConfig) -> None:
        """Initialize deterministic connectivity, population states, eligibility, and prototypes."""
        self.config = config
        rng = np.random.default_rng(config.random_seed)
        nh, nc, ni = config.hippocampal_units, config.cortical_units, config.input_size
        self.input_to_hippocampus = rng.normal(0.0, 0.30 / np.sqrt(ni), (nh, ni))
        recurrent = rng.normal(0.0, 0.12 / np.sqrt(nh), (nh, nh))
        recurrent *= rng.random(recurrent.shape) < 0.30
        self.hippocampal_recurrent = 0.5 * (recurrent + recurrent.T)
        np.fill_diagonal(self.hippocampal_recurrent, 0.0)
        self.hippocampus_to_cortex = rng.normal(0.0, 0.35 / np.sqrt(nh), (nc, nh))
        cortical = rng.normal(0.0, 0.10 / np.sqrt(nc), (nc, nc))
        cortical *= rng.random(cortical.shape) < 0.22
        np.fill_diagonal(cortical, 0.0)
        self.cortical_weights = cortical
        self.reference_cortical_weights = cortical.copy()
        self.eligibility_trace = np.zeros((nc, nc), dtype=np.float64)
        self.homeostatic_gain = np.ones(nc, dtype=np.float64)
        self.homeostatic_activity = np.full(nc, config.target_activity, dtype=np.float64)
        self.previous_cortical_activity = np.zeros(nc, dtype=np.float64)
        self.cortical_activity = np.zeros(nc, dtype=np.float64)
        self.trace_patterns: dict[str, np.ndarray] = {}
        self.trace_sequences: dict[str, list[np.ndarray]] = {}
        self.cortical_memories: dict[str, np.ndarray] = {}
        self.update_count = 0
        self.plasticity_event_count = 0

    def _encode_feature(self, feature: Sequence[float]) -> np.ndarray:
        """Project an event feature vector into the hippocampal population."""
        vector = fit_vector(feature, self.config.input_size)
        return sigmoid(self.input_to_hippocampus @ vector)

    def register_trace(self, trace: HippocampalTrace) -> dict[str, object]:
        """Create and retain hippocampal and initial cortical codes for one episodic trace."""
        if len(trace.features) != self.config.input_size:
            raise ValueError("trace feature dimension must match config.input_size")
        encoded = self._encode_feature(trace.features)
        sequence = trace.sequence or (trace.features,)
        self.trace_patterns[trace.trace_id] = encoded.copy()
        self.trace_sequences[trace.trace_id] = [self._encode_feature(item) for item in sequence]
        initial_cortex = sigmoid(self.hippocampus_to_cortex @ encoded)
        self.cortical_memories[trace.trace_id] = normalize_vector(initial_cortex)
        return {"trace_id": trace.trace_id, "hippocampal_population_size": int(encoded.size),
                "sequence_element_count": len(sequence),
                "initial_cortical_code_norm": float(np.linalg.norm(initial_cortex))}

    def pattern_complete(
        self, trace_id: str, partial_cue: Sequence[float] | None = None
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Complete a stored hippocampal assembly from a full or partial cue by recurrent settling."""
        if trace_id not in self.trace_patterns:
            raise KeyError(f"unknown trace_id: {trace_id}")
        stored = self.trace_patterns[trace_id]
        cue = stored if partial_cue is None else self._encode_feature(partial_cue)
        cue = fit_vector(cue, stored.size)
        activity = 0.62 * stored + 0.38 * cue
        initial_similarity = cosine_similarity(activity, stored)
        for _ in range(self.config.pattern_iterations):
            recurrent_drive = self.hippocampal_recurrent @ activity
            inhibition = 0.18 * float(np.mean(np.maximum(recurrent_drive, 0.0)))
            candidate = sigmoid(1.25 * activity + recurrent_drive - inhibition)
            activity = 0.55 * activity + 0.45 * candidate
        similarity = cosine_similarity(activity, stored)
        return activity, {"initial_similarity": initial_similarity,
                          "completed_similarity": similarity,
                          "settling_iterations": self.config.pattern_iterations,
                          "population_sparsity": float(np.mean(activity < 0.25))}

    def replay_sequence(
        self, trace_id: str, direction: str = "forward"
    ) -> tuple[list[np.ndarray], dict[str, object]]:
        """Return the stored ordered neural sequence in forward or reverse replay order."""
        if trace_id not in self.trace_sequences:
            raise KeyError(f"unknown trace_id: {trace_id}")
        sequence = [item.copy() for item in self.trace_sequences[trace_id]]
        if direction == "reverse":
            sequence.reverse()
        return sequence, {"direction": direction, "element_count": len(sequence),
                          "sequence_order_preserved": True}

    def propagate_to_cortex(
        self, hippocampal_sequence: Sequence[np.ndarray], alignment_gain: float
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Propagate replayed hippocampal states into a recurrent cortical population model."""
        if not hippocampal_sequence:
            return self.cortical_activity.copy(), {"status": "no_replay_sequence"}
        previous = self.previous_cortical_activity.copy()
        states: list[np.ndarray] = []
        for hippocampal_state in hippocampal_sequence:
            hip_drive = self.hippocampus_to_cortex @ hippocampal_state
            recurrent_drive = self.cortical_weights @ previous
            drive = (1.0 + float(alignment_gain)) * hip_drive + recurrent_drive
            activity = sigmoid(1.35 * drive) * self.homeostatic_gain
            activity = np.clip(activity, 0.0, 1.0)
            states.append(activity)
            previous = activity
        self.previous_cortical_activity = self.cortical_activity.copy()
        self.cortical_activity = np.mean(np.vstack(states), axis=0)
        return self.cortical_activity.copy(), {
            "sequence_steps": len(states), "mean_cortical_activity": float(np.mean(self.cortical_activity)),
            "active_fraction": float(np.mean(self.cortical_activity > 0.5)),
            "hippocampal_to_cortical_drive_norm": float(np.linalg.norm(
                self.hippocampus_to_cortex @ hippocampal_sequence[-1])),
        }

    def update_eligibility(self, pre_activity: Sequence[float], post_activity: Sequence[float],
                           interval_ms: float) -> dict[str, float]:
        """Accumulate decaying cortical co-activity with a signed pre/post timing term."""
        pre = fit_vector(pre_activity, self.config.cortical_units)
        post = fit_vector(post_activity, self.config.cortical_units)
        decay = float(np.exp(-abs(interval_ms) / self.config.eligibility_tau_ms))
        timing = stdp_timing_kernel(interval_ms, self.config.stdp_tau_ms)
        coactivity = np.outer(pre, post)
        self.eligibility_trace = decay * self.eligibility_trace + timing * coactivity
        return {"decay_factor": decay, "timing_factor": timing,
                "eligibility_norm": float(np.linalg.norm(self.eligibility_trace)),
                "eligible_connection_count": int(np.count_nonzero(
                    np.abs(self.eligibility_trace) > 1e-8))}

    def apply_plasticity(self, alignment: float, state_confidence: float) -> dict[str, object]:
        """Apply bounded alignment-gated updates and protect the reference cortical weight state."""
        gate = float(np.clip(alignment * state_confidence, 0.0, 1.0))
        raw_delta = self.config.learning_rate * gate * self.eligibility_trace
        delta = np.clip(raw_delta, -self.config.max_weight_step, self.config.max_weight_step)
        candidate = self.cortical_weights + delta
        protection = 0.002 * (candidate - self.reference_cortical_weights)
        self.cortical_weights = np.clip(candidate - protection,
                                         -self.config.max_abs_weight, self.config.max_abs_weight)
        actual_delta = self.cortical_weights - candidate + delta
        self.update_count += 1
        self.plasticity_event_count += 1
        self.eligibility_trace *= 0.92
        return {"gate": gate, "weight_delta_norm": float(np.linalg.norm(actual_delta)),
                "max_abs_delta": float(np.max(np.abs(actual_delta))),
                "weight_norm": float(np.linalg.norm(self.cortical_weights)),
                "weight_bound_respected": bool(np.max(np.abs(self.cortical_weights))
                                               <= self.config.max_abs_weight),
                "model_update_count": self.update_count}

    def integrate_cortical_memory(
        self, trace_id: str, activity: Sequence[float]
    ) -> dict[str, object]:
        """Update one trace-specific cortical prototype while repelling excessive cross-trace overlap."""
        current = normalize_vector(activity)
        if trace_id not in self.cortical_memories:
            self.cortical_memories[trace_id] = current.copy()
            return {"trace_id": trace_id, "similarity_before": None,
                    "similarity_after": 1.0, "other_trace_count": len(self.cortical_memories) - 1,
                    "separation_adjustment": 0.0}
        previous = self.cortical_memories[trace_id]
        similarity_before = cosine_similarity(previous, current)
        integrated = normalize_vector((1.0 - self.config.integration_rate) * previous
                                      + self.config.integration_rate * current)
        repulsion = np.zeros_like(integrated)
        for other_id, other in self.cortical_memories.items():
            if other_id == trace_id:
                continue
            overlap = max(0.0, cosine_similarity(integrated, other) - 0.72)
            repulsion += overlap * other
        repulsion_norm = float(np.linalg.norm(repulsion))
        if repulsion_norm > 0.0:
            integrated = normalize_vector(integrated - self.config.separation_strength * repulsion)
        self.cortical_memories[trace_id] = integrated
        return {"trace_id": trace_id, "similarity_before": similarity_before,
                "similarity_after": cosine_similarity(previous, integrated),
                "other_trace_count": len(self.cortical_memories) - 1,
                "separation_adjustment": repulsion_norm}

    def stabilize_homeostasis(self) -> dict[str, object]:
        """Adjust cell-population gains toward a target mean activity and bound recurrent weights."""
        error = self.config.target_activity - self.cortical_activity
        multiplier = np.exp(self.config.homeostatic_rate * error)
        self.homeostatic_gain = np.clip(
            self.homeostatic_gain * multiplier,
            self.config.homeostatic_gain_min, self.config.homeostatic_gain_max,
        )
        self.homeostatic_activity = 0.95 * self.homeostatic_activity + 0.05 * self.cortical_activity
        self.cortical_weights = np.clip(self.cortical_weights,
                                        -self.config.max_abs_weight, self.config.max_abs_weight)
        return {"target_activity": self.config.target_activity,
                "observed_mean_activity": float(np.mean(self.cortical_activity)),
                "mean_homeostatic_gain": float(np.mean(self.homeostatic_gain)),
                "gain_bounds_respected": bool(
                    np.min(self.homeostatic_gain) >= self.config.homeostatic_gain_min
                    and np.max(self.homeostatic_gain) <= self.config.homeostatic_gain_max),
                "weight_bounds_respected": bool(
                    np.max(np.abs(self.cortical_weights)) <= self.config.max_abs_weight)}

    def retrieval_probe(
        self, trace_id: str, partial_cue: Sequence[float] | None = None
    ) -> dict[str, object]:
        """Compare cue-driven cortical reinstatement with and without hippocampal input."""
        if trace_id not in self.trace_patterns or trace_id not in self.cortical_memories:
            return {"status": "trace_not_available", "cortical_support": None,
                    "reduced_hippocampal_support": None, "dependence_estimate": None}
        stored = self.cortical_memories[trace_id]
        cue = self.trace_patterns[trace_id] if partial_cue is None else self._encode_feature(partial_cue)
        hip_drive = self.hippocampus_to_cortex @ cue
        recurrent = self.cortical_weights @ self.cortical_activity
        with_hippocampus = normalize_vector(sigmoid(hip_drive + recurrent))
        without_hippocampus = normalize_vector(sigmoid(recurrent))
        full_score = cosine_similarity(with_hippocampus, stored)
        reduced_score = cosine_similarity(without_hippocampus, stored)
        return {"status": "retrieval_probe_completed", "cortical_support": full_score,
                "reduced_hippocampal_support": reduced_score,
                "dependence_estimate": float(max(0.0, full_score - reduced_score)),
                "independence_claimed": False}

    def snapshot(self) -> dict[str, object]:
        """Return population, connectivity, eligibility, and replay-update diagnostics."""
        return {"trace_count": len(self.trace_patterns),
                "cortical_memory_count": len(self.cortical_memories),
                "mean_cortical_activity": float(np.mean(self.cortical_activity)),
                "cortical_weight_norm": float(np.linalg.norm(self.cortical_weights)),
                "eligibility_norm": float(np.linalg.norm(self.eligibility_trace)),
                "mean_homeostatic_gain": float(np.mean(self.homeostatic_gain)),
                "plasticity_event_count": self.plasticity_event_count,
                "update_count": self.update_count}
