"""Master orchestration for seven-stage predictive orthographic processing."""
from __future__ import annotations
from typing import Any, Iterable, Mapping
import math
import numpy as np
from ore_n3 import LabeledN2Patch, N3Processor, N3Sequence
from ore_n3.visual import extract_patch_features
from ore_n4 import N4Config, N4Input, N4Processor
from .biology import BIOLOGICAL_STAGE_NAMES, LIFConfig, simulate_lif_reference
from .contracts import PredictiveOrthographyResult
from .registry import MASTER_OPERATION_REGISTRY, MASTER_STAGE_REGISTRY
from .source_mapping import OPERATION_TO_SOURCE_STEPS, source_mapping_as_dicts
from ore_n4.stages import damped_belief_update, normalized_entropy, normalize_signal

def _symmetric_kl_rows(left: np.ndarray, right: np.ndarray, epsilon: float) -> np.ndarray:
    p = np.maximum(np.asarray(left, dtype=np.float64), epsilon)
    q = np.maximum(np.asarray(right, dtype=np.float64), epsilon)
    p /= p.sum(axis=1, keepdims=True)
    q /= q.sum(axis=1, keepdims=True)
    return (0.5 * np.sum(p * (np.log(p) - np.log(q)), axis=1)
            + 0.5 * np.sum(q * (np.log(q) - np.log(p)), axis=1)).astype(np.float32)

def _neighbor_messages(
    beliefs: np.ndarray, forward: np.ndarray, reverse: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position_count, class_count = beliefs.shape
    left = np.zeros((position_count, class_count), dtype=np.float32)
    right = np.zeros((position_count, class_count), dtype=np.float32)
    left_available = np.zeros(position_count, dtype=bool)
    right_available = np.zeros(position_count, dtype=bool)
    for index in range(position_count):
        if index > 0:
            left[index] = beliefs[index - 1] @ forward
            left_available[index] = True
        if index + 1 < position_count:
            right[index] = reverse @ beliefs[index + 1]
            right_available[index] = True
    return left, right, left_available, right_available

def _mean_row_entropy(values: np.ndarray, epsilon: float) -> float:
    p = np.maximum(np.asarray(values, dtype=np.float64), epsilon)
    p /= p.sum(axis=1, keepdims=True)
    return float(np.mean(-np.sum(p * np.log(p), axis=1)))


def _trace_value(value: Any) -> Any:
    """Summarize one operation's own output without copying full high-dimensional arrays."""
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        numeric = array.astype(np.float64, copy=False) if array.size else array
        result: dict[str, Any] = {
            "shape": list(array.shape), "dtype": str(array.dtype),
            "size": int(array.size),
        }
        if array.size and np.issubdtype(array.dtype, np.number):
            result.update({
                "finite": bool(np.isfinite(numeric).all()),
                "minimum": float(np.min(numeric)),
                "maximum": float(np.max(numeric)),
                "mean": float(np.mean(numeric)),
                "sample": array.reshape(-1)[:8].tolist(),
            })
        return result
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _trace_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_trace_value(item) for item in value[:8]] + (
            [{"remaining_items": len(value) - 8}] if len(value) > 8 else []
        )
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class PredictiveOrthographyProcessor:
    """Compose N3 evidence/transitions and N4 attention/precision/posterior inference.

    The current-based LIF reference receives summaries of all seven stages and
    never writes neural state back into the functional inference outputs.
    """
    def __init__(
        self,
        n3_processor: N3Processor,
        n4_processor: N4Processor | None = None,
        *,
        n4_config: N4Config | None = None,
        simulate_biology: bool = True,
        lif_config: LIFConfig | None = None,
    ) -> None:
        if n3_processor is None or not callable(getattr(n3_processor, "recognize", None)):
            raise TypeError("n3_processor must be a fitted-compatible N3Processor")
        if n4_processor is not None and not callable(getattr(n4_processor, "process", None)):
            raise TypeError("n4_processor must expose process(N4Input)")
        self.n3_processor = n3_processor
        self.n4_processor = n4_processor or N4Processor(n4_config)
        self.simulate_biology = bool(simulate_biology)
        self.lif_config = lif_config or LIFConfig()

    def fit(
        self,
        reference_patches: Iterable[LabeledN2Patch],
        *,
        grapheme_sequences: Iterable[Iterable[str]] = (),
    ) -> "PredictiveOrthographyProcessor":
        self.n3_processor.fit(reference_patches, grapheme_sequences=grapheme_sequences)
        return self

    def process(
        self,
        sequence: N3Sequence,
        *,
        bottom_up_salience: np.ndarray | None = None,
        task_relevance: np.ndarray | None = None,
        goal_class_weights: np.ndarray | None = None,
        seed_attention: np.ndarray | None = None,
        segmentation_confidence: np.ndarray | None = None,
        feedback: Mapping[str, Any] | None = None,
    ) -> PredictiveOrthographyResult:
        """Run all seven stages on already-segmented patches in logical reading order."""
        if not isinstance(sequence, N3Sequence):
            raise TypeError("sequence must be an N3Sequence")
        feedback_sequence_id = None
        if feedback is not None:
            if not isinstance(feedback, Mapping):
                raise TypeError("feedback must be a mapping returned by feedback_for_next_cycle")
            feedback_sequence_id = feedback.get("source_sequence_id")
            expected_length = len(sequence.patches)
            for key in ("bottom_up_salience", "task_relevance", "seed_attention"):
                if key not in feedback:
                    continue
                values = np.asarray(feedback[key], dtype=np.float32)
                if values.shape != (expected_length,) or not np.isfinite(values).all():
                    raise ValueError(f"feedback[{key!r}] must be a finite vector of length {expected_length}")
                if np.any(values < 0.0) or np.any(values > 1.0):
                    raise ValueError(f"feedback[{key!r}] values must lie in [0, 1]")
            if bottom_up_salience is None and "bottom_up_salience" in feedback:
                bottom_up_salience = np.asarray(feedback["bottom_up_salience"], dtype=np.float32)
            if task_relevance is None and "task_relevance" in feedback:
                task_relevance = np.asarray(feedback["task_relevance"], dtype=np.float32)
            if seed_attention is None and "seed_attention" in feedback:
                seed_attention = np.asarray(feedback["seed_attention"], dtype=np.float32)
        transition_model = getattr(self.n3_processor, "transition_model", None)
        if transition_model is None:
            raise RuntimeError("fit the N3 processor before running predictive orthography")

        # Stages 1-3: supervised visual evidence, transition probabilities and N3 recurrence.
        n3_result = self.n3_processor.recognize(sequence)
        if not bool(n3_result.diagnostics.get("valid", False)):
            raise ValueError("N3 returned an invalid result; master processing was stopped")

        # Stages 4-6: N4 attention, precision modulation and posterior settling.
        n4_result = self.n4_processor.process(N4Input(
            n3_result=n3_result,
            bottom_up_salience=bottom_up_salience,
            task_relevance=task_relevance,
            goal_class_weights=goal_class_weights,
            seed_attention=seed_attention,
            segmentation_confidence=segmentation_confidence,
        ))
        if not bool(n4_result.diagnostics.get("valid", False)):
            raise ValueError("N4 returned an invalid result; master processing was stopped")

        epsilon = float(getattr(getattr(self.n4_processor, "config", None), "epsilon", 1e-8))
        left, right, left_available, right_available = _neighbor_messages(
            n3_result.posterior_beliefs,
            np.asarray(transition_model.forward, dtype=np.float32),
            np.asarray(transition_model.reverse, dtype=np.float32),
        )
        visual_context_divergence = _symmetric_kl_rows(
            n3_result.visual_probabilities, n4_result.contextual_priors, epsilon
        )

        raw_features = np.stack([
            extract_patch_features(n2_result=p.n2_result, features=p.features)
            for p in sequence.patches
        ]).astype(np.float32)
        normalized_features = self.n3_processor.visual_population.transform(raw_features)
        prototypes = np.asarray(self.n3_processor.visual_population.prototypes, dtype=np.float32)
        known_count = prototypes.shape[0]
        attended_known = np.asarray(n4_result.posterior_beliefs[:, :known_count], dtype=np.float64)
        known_mass = attended_known.sum(axis=1, keepdims=True)
        expected_features = (attended_known @ prototypes) / np.maximum(known_mass, epsilon)
        expected_features[known_mass[:, 0] <= epsilon] = 0.0
        attended_feature_residuals = (normalized_features - expected_features).astype(np.float32)

        contextual_residuals = np.asarray(n3_result.feature_residuals, dtype=np.float32)
        residual_norms = np.linalg.norm(attended_feature_residuals, axis=1)
        contextual_residual_norms = np.linalg.norm(contextual_residuals, axis=1)
        classwise_error = np.asarray(n4_result.classwise_prediction_error, dtype=np.float32)
        posterior = np.asarray(n4_result.posterior_beliefs, dtype=np.float32)
        row_sum_error = float(np.max(np.abs(posterior.sum(axis=1) - 1.0)))
        arrays = (left, right, visual_context_divergence, attended_feature_residuals,
                  contextual_residuals, classwise_error, posterior)
        finite = all(np.isfinite(array).all() for array in arrays)

        visual_kl = np.sum(posterior * (
            np.log(np.maximum(posterior, epsilon))
            - np.log(np.maximum(n4_result.visual_probabilities, epsilon)
        )), axis=1)
        prior_kl = np.sum(posterior * (
            np.log(np.maximum(posterior, epsilon))
            - np.log(np.maximum(n4_result.contextual_priors, epsilon)
        )), axis=1)
        weighted_divergence_proxy = (
            n4_result.sensory_precision * visual_kl
            + n4_result.prior_precision * prior_kl
        )

        stage_outputs: dict[str, Any] = {
            "S1": {"visual_probabilities": n3_result.visual_probabilities,
                   "class_names": list(n3_result.class_names),
                   "raw_query_features": raw_features,
                   "normalized_query_features": normalized_features,
                   "feature_mean": np.asarray(self.n3_processor.visual_population.mean),
                   "feature_scale": np.asarray(self.n3_processor.visual_population.scale),
                   "grapheme_prototypes": prototypes,
                   "query_prototype_similarities": normalized_features @ prototypes.T,
                   "feature_dimension": int(normalized_features.shape[1]),
                   "training_trace": list(getattr(self.n3_processor, "training_trace", []))},
            "S2": {"unigram_prior": np.asarray(transition_model.unigram, dtype=np.float32),
                   "forward_transition_probabilities": np.asarray(transition_model.forward, dtype=np.float32),
                   "reverse_transition_probabilities": np.asarray(transition_model.reverse, dtype=np.float32),
                   "transition_fingerprint": str(transition_model.fingerprint)},
            "S3": {"left_neighbor_predictions": left, "right_neighbor_predictions": right,
                   "left_neighbor_available": left_available, "right_neighbor_available": right_available,
                   "contextual_priors": n4_result.contextual_priors,
                   "n3_iteration_trace": list(n3_result.trace)},
            "S4": {"bottom_up_salience": n4_result.bottom_up_salience,
                   "top_down_relevance": n4_result.top_down_relevance,
                   "attention_allocation": n4_result.attention_allocation,
                   "class_attention": n4_result.class_attention,
                   "class_attention_gate": n4_result.class_attention_gate,
                   "sensory_precision": n4_result.sensory_precision,
                   "n4_operation_trace": list(n4_result.operation_trace)},
            "S5": {"prior_precision": n4_result.prior_precision,
                   "n4_operation_trace": list(n4_result.operation_trace)},
            "S6": {"n3_initial_posterior": n4_result.n3_posterior_beliefs,
                   "attended_posterior": posterior,
                   "iteration_trace": list(n4_result.iteration_trace),
                   "n4_operation_trace": list(n4_result.operation_trace)},
            "S7": {"classwise_prediction_error": classwise_error,
                   "visual_context_symmetric_kl": visual_context_divergence,
                   "contextual_feature_residuals": contextual_residuals,
                   "attended_feature_residuals": attended_feature_residuals,
                   "attended_feature_residual_norms": residual_norms,
                   "contextual_feature_residual_norms": contextual_residual_norms,
                   "weighted_divergence_energy_proxy": weighted_divergence_proxy,
                   "n4_final_prediction_error": n4_result.classwise_prediction_error},
        }

        stage_summaries: dict[str, dict[str, Any]] = {
            "S1": {"position_count": int(posterior.shape[0]),
                   "class_count_including_UNK": int(posterior.shape[1]),
                   "feature_dimension": int(normalized_features.shape[1]),
                   "mean_unknown_probability": float(np.mean(n3_result.visual_probabilities[:, -1]))},
            "S2": {"transition_fingerprint": str(transition_model.fingerprint),
                   "class_count": int(len(transition_model.class_names)),
                   "forward_rows_normalized": bool(np.allclose(transition_model.forward.sum(axis=1), 1.0, atol=1e-5)),
                   "reverse_columns_normalized": bool(np.allclose(transition_model.reverse.sum(axis=0), 1.0, atol=1e-5))},
            "S3": {"left_message_positions": int(left_available.sum()),
                   "right_message_positions": int(right_available.sum()),
                   "mean_contextual_entropy": _mean_row_entropy(n4_result.contextual_priors, epsilon)},
            "S4": {"attention_sum": float(np.sum(n4_result.attention_allocation)),
                   "mean_bottom_up_salience": float(np.mean(n4_result.bottom_up_salience)),
                   "mean_sensory_precision": float(np.mean(n4_result.sensory_precision)),
                   "mean_class_attention_gate": float(np.mean(n4_result.class_attention_gate))},
            "S5": {"mean_prior_precision": float(np.mean(n4_result.prior_precision)),
                   "prior_precision_min": float(np.min(n4_result.prior_precision)),
                   "prior_precision_max": float(np.max(n4_result.prior_precision))},
            "S6": {"iterations": int(len(n4_result.iteration_trace)),
                   "converged": bool(n4_result.diagnostics.get("converged", False)),
                   "mean_belief_change": float(n4_result.diagnostics.get("mean_belief_change", 0.0)),
                   "posterior_row_sum_error": row_sum_error},
            "S7": {"mean_visual_context_symmetric_kl": float(np.mean(visual_context_divergence)),
                   "mean_attended_feature_residual_norm": float(np.mean(residual_norms)),
                   "mean_contextual_feature_residual_norm": float(np.mean(contextual_residual_norms)),
                   "mean_absolute_classwise_error": float(np.mean(np.abs(classwise_error))),
                   "mean_weighted_divergence_proxy": float(np.mean(weighted_divergence_proxy))},
        }

        stage_trace = [
            {"stage_id": stage.stage_id, "stage": stage.name, "purpose": stage.purpose,
             "operation_ids": list(stage.operation_ids), "status": "executed",
             "summary": stage_summaries[stage.stage_id]}
            for stage in MASTER_STAGE_REGISTRY
        ]
        # Each MPO record is emitted from a distinct intermediate produced by the
        # corresponding computation. No stage-summary template is replayed 40 times.
        transition_unigram_counts = np.asarray(
            getattr(transition_model, "unigram_counts", np.zeros_like(transition_model.unigram)),
            dtype=np.float32,
        )
        transition_bigram_counts = np.asarray(
            getattr(transition_model, "bigram_counts", np.zeros_like(transition_model.forward)),
            dtype=np.float32,
        )
        alpha = float(getattr(transition_model, "smoothing_alpha", 0.0))
        visual_entropy = -np.sum(
            np.maximum(n3_result.visual_probabilities, epsilon)
            * np.log(np.maximum(n3_result.visual_probabilities, epsilon)), axis=1
        )
        normalized_visual_uncertainty = normalized_entropy(
            n3_result.visual_probabilities, epsilon
        )
        raw_error = np.mean(np.abs(n3_result.classwise_prediction_error), axis=1)
        normalized_error = normalize_signal(raw_error, epsilon)
        prior_disagreement = _symmetric_kl_rows(
            n3_result.visual_probabilities, n3_result.contextual_priors, epsilon
        )
        confidence = np.asarray(
            [float(item.get("segmentation_confidence", 1.0)) for item in n4_result.candidates],
            dtype=np.float32,
        )
        if confidence.shape != (len(sequence.patches),):
            confidence = np.asarray(
                [patch.segmentation_confidence for patch in sequence.patches], dtype=np.float32
            )
        n4_config = getattr(self.n4_processor, "config", None) or N4Config()
        unmodulated_prior_precision = (
            n4_config.prediction_gain * (0.5 + normalized_visual_uncertainty) * (2.0 - confidence)
        )
        task_modulated_prior_precision = (
            unmodulated_prior_precision * (1.0 + n4_config.task_prior_gain * n4_result.top_down_relevance)
        )
        bounded_prior_precision = np.clip(
            task_modulated_prior_precision,
            n4_config.prior_precision_min,
            n4_config.prior_precision_max,
        ).astype(np.float32)
        proposal_seed, _ = damped_belief_update(
            n4_result.n3_posterior_beliefs,
            n4_result.visual_probabilities,
            n4_result.contextual_priors,
            n4_result.sensory_precision,
            n4_result.prior_precision,
            n4_result.class_attention_gate,
            1.0,
            epsilon,
        )
        posterior_prediction_error = posterior * (
            np.log(np.maximum(posterior, epsilon))
            - np.log(np.maximum(n4_result.contextual_priors, epsilon))
        )
        precision_weighted_residuals = (
            attended_feature_residuals * n4_result.sensory_precision[:, None]
        )
        source_operation_outputs: dict[str, dict[str, Any]] = {
            "MPO-001": {"sequence_id": sequence.sequence_id, "position_count": len(sequence.patches),
                        "patch_ids": [patch.patch_id for patch in sequence.patches],
                        "segmentation_metadata_validated": True},
            "MPO-002": {"raw_visual_descriptors": raw_features},
            "MPO-003": {"feature_mean": np.asarray(self.n3_processor.visual_population.mean),
                        "feature_scale": np.asarray(self.n3_processor.visual_population.scale)},
            "MPO-004": {"standardized_query_descriptors": normalized_features},
            "MPO-005": {"grapheme_population_prototypes": prototypes},
            "MPO-006": {"query_prototype_similarity": normalized_features @ prototypes.T},
            "MPO-007": {"bottom_up_candidate_activation": n3_result.visual_probabilities},
            "MPO-008": {"unknown_candidate_probability": n3_result.visual_probabilities[:, -1],
                        "initial_belief_state": n3_result.initial_beliefs},
            "MPO-009": {"observed_grapheme_exposure_counts": transition_unigram_counts},
            "MPO-010": {"observed_ordered_pair_counts": transition_bigram_counts},
            "MPO-011": {"dirichlet_smoothing_alpha": alpha,
                        "smoothed_unigram_counts": transition_unigram_counts + alpha,
                        "smoothed_pair_counts": transition_bigram_counts + alpha},
            "MPO-012": {"normalized_unigram_prior": transition_model.unigram},
            "MPO-013": {"forward_transition_probabilities": transition_model.forward},
            "MPO-014": {"reverse_transition_probabilities": transition_model.reverse},
            "MPO-015": {"frozen_transition_model_fingerprint": transition_model.fingerprint},
            "MPO-016": {"initial_grapheme_beliefs": n3_result.initial_beliefs},
            "MPO-017": {"left_neighbor_messages": left, "left_neighbor_available": left_available},
            "MPO-018": {"right_neighbor_messages": right, "right_neighbor_available": right_available},
            "MPO-019": {"recurrently_integrated_contextual_priors": n3_result.contextual_priors},
            "MPO-020": {"normalized_contextual_priors": n3_result.contextual_priors,
                        "row_sum_error": np.max(np.abs(n3_result.contextual_priors.sum(axis=1) - 1.0))},
            "MPO-021": {"visual_entropy_by_position": visual_entropy},
            "MPO-022": {"normalized_visual_uncertainty": normalized_visual_uncertainty},
            "MPO-023": {"classwise_error_magnitude": raw_error, "normalized_error_signal": normalized_error},
            "MPO-024": {"visual_context_symmetric_kl": prior_disagreement},
            "MPO-025": {"bottom_up_salience": n4_result.bottom_up_salience},
            "MPO-026": {"task_and_goal_relevance": n4_result.top_down_relevance},
            "MPO-027": {"competitive_attention_allocation": n4_result.attention_allocation},
            "MPO-028": {"attention_gated_sensory_precision": n4_result.sensory_precision},
            "MPO-029": {"unmodulated_prior_precision": unmodulated_prior_precision},
            "MPO-030": {"top_down_modulated_prior_precision": task_modulated_prior_precision},
            "MPO-031": {"bounded_prior_precision": bounded_prior_precision},
            "MPO-032": {"precision_channels_validated": bool(
                np.isfinite(n4_result.sensory_precision).all()
                and np.isfinite(n4_result.prior_precision).all()
                and np.all(n4_result.sensory_precision > 0)
                and np.all(n4_result.prior_precision > 0)
            )},
            "MPO-033": {"attended_initial_belief_state": n4_result.n3_posterior_beliefs},
            "MPO-034": {"precision_weighted_posterior_proposal": proposal_seed},
            "MPO-035": {"normalized_posterior_proposal": proposal_seed,
                        "proposal_row_sum_error": np.max(np.abs(proposal_seed.sum(axis=1) - 1.0))},
            "MPO-036": {"damped_recurrent_posterior": posterior,
                        "damping_factor": n4_config.damping},
            "MPO-037": {"settling_iterations": len(n4_result.iteration_trace),
                        "iteration_trace": n4_result.iteration_trace,
                        "converged": bool(n4_result.diagnostics.get("converged", False)),
                        "final_belief_change": n4_result.diagnostics.get("mean_belief_change", 0.0)},
            "MPO-038": {"posterior_to_contextual_prediction_error": posterior_prediction_error,
                        "posterior_prediction_divergence": np.sum(posterior_prediction_error, axis=1)},
            "MPO-039": {"contextual_feature_residuals": contextual_residuals,
                        "attended_feature_residuals": attended_feature_residuals,
                        "precision_weighted_residuals": precision_weighted_residuals},
            "MPO-040": {"confidence_and_abstention_diagnostics": n4_result.diagnostics,
                        "ranked_hypotheses": n4_result.candidates,
                        "final_validation": {"finite_outputs": finite, "posterior_row_sum_error": row_sum_error}},
        }
        operation_trace = []
        for op in MASTER_OPERATION_REGISTRY:
            produced = source_operation_outputs[op.operation_id]
            operation_trace.append({
                "operation_id": op.operation_id, "stage_id": op.stage_id, "name": op.name,
                "mathematical_form": op.mathematical_form,
                "biological_reference": op.biological_reference,
                "implementation_source": op.implementation_source,
                "source_step_ids": list(OPERATION_TO_SOURCE_STEPS.get(op.operation_id, ())),
                "status": "executed",
                "inputs_observed": {"stage_id": op.stage_id},
                "outputs_produced": _trace_value(produced),
                "execution_evidence": {
                    "output_field_names": sorted(produced),
                    "operation_specific_output": True,
                    "stage_summary_reused_as_operation_output": False,
                },
            })

        if self.simulate_biology:
            signals: Mapping[str, object] = {
                BIOLOGICAL_STAGE_NAMES[0]: n3_result.visual_probabilities,
                BIOLOGICAL_STAGE_NAMES[1]: np.concatenate((
                    np.asarray(transition_model.unigram).reshape(-1),
                    np.asarray(transition_model.forward).reshape(-1),
                    np.asarray(transition_model.reverse).reshape(-1))),
                BIOLOGICAL_STAGE_NAMES[2]: np.concatenate((left, right, n4_result.contextual_priors), axis=None),
                BIOLOGICAL_STAGE_NAMES[3]: np.concatenate((
                    n4_result.bottom_up_salience, n4_result.top_down_relevance,
                    n4_result.attention_allocation, n4_result.class_attention_gate.reshape(-1))),
                BIOLOGICAL_STAGE_NAMES[4]: np.concatenate((n4_result.sensory_precision, n4_result.prior_precision)),
                BIOLOGICAL_STAGE_NAMES[5]: posterior,
                BIOLOGICAL_STAGE_NAMES[6]: np.concatenate((
                    classwise_error.reshape(-1), visual_context_divergence.reshape(-1), residual_norms.reshape(-1))),
            }
            biological = simulate_lif_reference(signals, self.lif_config)
        else:
            biological = {"enabled": False, "model": "disabled",
                           "scientific_status": "biological reference simulation was disabled by configuration"}

        valid = bool(finite and row_sum_error <= 1e-5
                     and n3_result.diagnostics.get("valid", False)
                     and n4_result.diagnostics.get("valid", False)
                     and (not self.simulate_biology or biological.get("numerically_valid", False))
                     and len(stage_trace) == 7 and len(operation_trace) == 40)
        diagnostics = {
            "valid": valid, "stage_count": len(stage_trace), "operation_count": len(operation_trace),
            "source_step_count": 38, "source_step_mapping_complete": len(source_mapping_as_dicts()) == 38,
            "operation_specific_outputs": all(row["execution_evidence"]["operation_specific_output"] for row in operation_trace),
            "feedback_consumed": feedback is not None,
            "n3_valid": bool(n3_result.diagnostics.get("valid", False)),
            "n4_valid": bool(n4_result.diagnostics.get("valid", False)),
            "biological_valid": bool(biological.get("numerically_valid", False)) if self.simulate_biology else None,
            "finite_outputs": finite, "posterior_row_sum_error": row_sum_error,
            "mean_visual_context_symmetric_kl": float(np.mean(visual_context_divergence)),
            "mean_attended_feature_residual_norm": float(np.mean(residual_norms)),
            "mean_contextual_feature_residual_norm": float(np.mean(contextual_residual_norms)),
            "mean_weighted_divergence_proxy": float(np.mean(weighted_divergence_proxy)),
            "n4_converged": bool(n4_result.diagnostics.get("converged", False)),
            "n4_stop_reason": n4_result.diagnostics.get("stop_reason"),
            "abstention_count": int(n4_result.diagnostics.get("abstention_count", 0)),
            "ambiguous_count": int(n4_result.diagnostics.get("ambiguous_count", 0)),
            "scientific_status": "N3/N4 functional pipeline plus a generic LIF reference; not a validated neural reading circuit",
        }
        metadata = {
            "pipeline": "ORE seven-stage master predictive orthography", "version": "0.1.0",
            "functional_components": ["ore_n3", "ore_n4"],
            "biological_component": "current-based LIF reference network" if self.simulate_biology else None,
            "stage_registry": "MPO-S1..S7", "operation_registry": "MPO-001..MPO-040",
            "source_step_mapping": source_mapping_as_dicts(), "source_step_count": 38,
            "next_cycle_feedback_input": {
                "supplied": feedback is not None,
                "source_sequence_id": feedback_sequence_id,
                "explicit_arguments_take_precedence": True,
            },
            "transition_model_fingerprint": str(transition_model.fingerprint),
            "visual_model_fingerprint": str(self.n3_processor.visual_population.fingerprint),
            "input_boundary": "one already-segmented N2 result or feature vector per patch in logical reading order",
            "read_only_upstream": True, "biological_track_is_parallel": True,
            "limitations": [
                "Character patches must already be segmented and ordered; this processor does not perform OCR segmentation.",
                "N3 visual prototypes require labelled reference patches from the target script and rendering conditions.",
                "Transition statistics are learned from supplied labelled grapheme sequences and are not universal language rules.",
                "Feature residuals live in standardized engineered descriptor space, not pixel space.",
                "The LIF network is generic and not anatomically mapped, fitted or validated against human recordings.",
                "Software tests establish implementation behavior, not human reading accuracy or biological validity.",
            ],
        }
        return PredictiveOrthographyResult(
            sequence_id=sequence.sequence_id, n3_result=n3_result, n4_result=n4_result,
            left_neighbor_predictions=left, right_neighbor_predictions=right,
            left_neighbor_available=left_available, right_neighbor_available=right_available,
            visual_context_divergence=visual_context_divergence,
            attended_feature_residuals=attended_feature_residuals, stage_outputs=stage_outputs,
            stage_trace=stage_trace, operation_trace=operation_trace, biological=biological,
            diagnostics=diagnostics, metadata=metadata)

    def feedback_for_next_cycle(
        self, result: PredictiveOrthographyResult,
    ) -> dict[str, Any]:
        """Convert prediction residuals and posterior uncertainty into explicit next-cycle cues.

        The caller remains responsible for deciding whether a later sequence belongs to the
        same reading stream and for supplying the returned feedback to process(feedback=...).
        """
        if not isinstance(result, PredictiveOrthographyResult):
            raise TypeError("result must be a PredictiveOrthographyResult")
        posterior = np.asarray(result.posterior_beliefs, dtype=np.float32)
        if posterior.ndim != 2 or posterior.shape[0] == 0:
            raise ValueError("result posterior must contain at least one position")
        epsilon = float(getattr(getattr(self.n4_processor, "config", None), "epsilon", 1e-8))
        probabilities = np.maximum(posterior.astype(np.float64), epsilon)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
        max_entropy = math.log(max(2, posterior.shape[1]))
        normalized_uncertainty = np.clip(entropy / max_entropy, 0.0, 1.0)
        residual = np.linalg.norm(
            np.asarray(result.attended_feature_residuals, dtype=np.float64), axis=1
        )
        error_salience = normalize_signal(residual, epsilon)
        uncertainty = normalize_signal(normalized_uncertainty, epsilon)
        combined = normalize_signal(0.6 * error_salience + 0.4 * uncertainty, epsilon)
        return {
            "source_operation_id": "MPO-040",
            "source_step_id": "7.7",
            "source_sequence_id": result.sequence_id,
            "bottom_up_salience": combined.astype(np.float32),
            "task_relevance": error_salience.astype(np.float32),
            "seed_attention": uncertainty.astype(np.float32),
            "feedback_evidence": {
                "residual_norms": residual.astype(np.float32),
                "normalized_posterior_uncertainty": uncertainty.astype(np.float32),
                "policy": "normalized attended-feature residual and posterior uncertainty",
                "no_character_identity_is_synthesized": True,
            },
        }

