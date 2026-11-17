"""Implement source Point 1: sensory perception to uncertainty-aware auditory/visual word forms.

The low-level signal processing is delegated to the established ore_sensory_perception package.
This module adds the source-ordered 12-step algorithm, preserves modality separation and maps the
low-level LIF reference updates to source-level steps. It deliberately stops before lexical meaning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ore_sensory_perception import SensoryConfig, SensoryPerceptionProcessor, SensoryRequest
from ore_sensory_perception.contracts import to_jsonable


POINT1_STEP_SPECS: tuple[dict[str, Any], ...] = (
    {
        "operation_id": "LC-P01-01",
        "name": "Capture the sensory stream and maintain a running representation",
        "objective": "Validate available sound/image samples, retain their timing or spatial indexing, and summarize recent sensory evidence without discarding order.",
        "mathematical_operations": ("sampling and time/space indexing", "leaky evidence integration m[t]=lambda*m[t-1]+(1-lambda)*x[t] as a model abstraction", "signal/noise and finite-value summaries"),
        "biological_reference": "Cochlear and retinal transduction, early sensory pathways, persistent population activity and attention-dependent gain; the code stores samples rather than claiming to simulate receptors.",
    },
    {
        "operation_id": "LC-P01-02",
        "name": "Route evidence through auditory and/or visual pathways",
        "objective": "Select the pathway or parallel pathways supported by the actual request and retain their identities.",
        "mathematical_operations": ("availability gates g_A and g_V", "parallel pathway bookkeeping", "optional gain-weighted readout z=g_A*z_A+g_V*z_V without forcing common representations"),
        "biological_reference": "Auditory nerve/brainstem/medial geniculate/auditory cortex and retina/LGN/V1/extrastriate/ventral occipitotemporal pathways, with recurrent and parallel projections.",
    },
    {
        "operation_id": "LC-P01-03",
        "name": "Extract elementary modality-specific features",
        "objective": "Obtain frequency, temporal-envelope and acoustic-phonetic evidence from sound, or contrast, contour, orientation and glyph evidence from images.",
        "mathematical_operations": ("filter-bank convolution", "rectification and logarithmic compression", "STFT and spectral power", "Gabor filtering and local contrast", "feature-vector construction"),
        "biological_reference": "Basilar-membrane tonotopy and hair-cell/auditory-nerve coding; retinal/LGN center-surround processing and V1-like orientation/spatial-frequency populations.",
    },
    {
        "operation_id": "LC-P01-04",
        "name": "Normalize nuisance variation while preserving useful distinctions",
        "objective": "Reduce sensitivity to speaker, intensity, channel, font, scale and position changes without erasing phonetic or orthographic contrasts.",
        "mathematical_operations": ("speaker-relative z-scores or robust median/MAD normalization", "contrast normalization", "scale/translation normalization", "bounded nonlinear compression"),
        "biological_reference": "Adaptive gain and normalization across auditory populations and learned invariance along ventral visual pathways; parameters are model settings, not universal neural constants.",
    },
    {
        "operation_id": "LC-P01-05",
        "name": "Encode features in distributed neural-population representations",
        "objective": "Keep acoustic, phonetic, visual and orthographic evidence as graded population patterns rather than one premature symbolic decision.",
        "mathematical_operations": ("population feature vectors", "nonlinear activation and rectification", "divisive normalization", "recurrent population-state updates"),
        "biological_reference": "Distributed superior temporal gyrus populations for speech features and ventral occipitotemporal populations for letter/glyph features; simulated LIF populations are a qualified reference model.",
    },
    {
        "operation_id": "LC-P01-06",
        "name": "Integrate evidence over time or across visual locations",
        "objective": "Accumulate speech evidence across adjacent frames and preserve spatial relations among visible characters and regions.",
        "mathematical_operations": ("temporal smoothing and leaky integration", "cross-channel coherence", "local pooling and weighted spatial aggregation", "time-indexed feature matrices"),
        "biological_reference": "Auditory cortical temporal integration and phase-sensitive responses; recurrent visual grouping and spatially distributed receptive fields.",
    },
    {
        "operation_id": "LC-P01-07",
        "name": "Detect candidate sub-word units and preserve their order",
        "objective": "Build time-ordered phoneme/phonetic-segment candidates and position-ordered visual glyph/letter candidates without assigning full word meaning.",
        "mathematical_operations": ("temporal segmentation and onset/offset estimates", "ordinal position codes", "sequence assembly", "local transition scoring"),
        "biological_reference": "Dynamic phonetic categorization in superior temporal cortex and the graded retinotopic-to-ordinal transformation in the ventral visual pathway, including VWFA-related regions.",
    },
    {
        "operation_id": "LC-P01-08",
        "name": "Combine multiple cues according to their reliability",
        "objective": "Summarize the confidence contributed by separate acoustic and visual cues while preserving the source and modality of each estimate.",
        "mathematical_operations": ("normalized posterior probabilities", "cue-weighted evidence sums", "confidence and margin summaries", "Bayesian log-evidence when caller-supplied priors exist"),
        "biological_reference": "Distributed nonlinear cue integration and recurrent modulation; posterior values are model outputs and not direct neural firing probabilities.",
    },
    {
        "operation_id": "LC-P01-09",
        "name": "Maintain competing phonetic and orthographic hypotheses",
        "objective": "Retain the strongest alternatives and their uncertainty rather than applying an unconditional winner-take-all decision.",
        "mathematical_operations": ("softmax or normalized competition", "top-k candidate retention", "entropy H=-sum(p*log(p))", "posterior margin"),
        "biological_reference": "Competition and recurrent interactions across auditory and ventral visual populations; the exact neural implementation is unresolved.",
    },
    {
        "operation_id": "LC-P01-10",
        "name": "Estimate likely boundaries and preserve segmentation uncertainty",
        "objective": "Retain candidate phonetic onset/offset times and visual glyph/text-region positions with explicit boundary qualifications.",
        "mathematical_operations": ("change-point/onset detection", "frame-to-time conversion", "bounding-box and ordinal-position geometry", "boundary confidence and entropy"),
        "biological_reference": "Temporal event coding and perceptual grouping; eye movements and spatial attention influence the sampling of visual boundaries.",
    },
    {
        "operation_id": "LC-P01-11",
        "name": "Apply attention- and prediction-related modulation",
        "objective": "Expose the effects of supplied attention weights and explicit orthographic/phonetic priors without inventing missing context.",
        "mathematical_operations": ("attention logits and softmax", "multiplicative gain", "prior-weighted log evidence", "uncertainty-sensitive resampling signals"),
        "biological_reference": "Auditory and visual attentional gain plus recurrent feedback from language-related systems; early top-down modulation may overlap sensory analysis.",
    },
    {
        "operation_id": "LC-P01-12",
        "name": "Package modality-specific form hypotheses for lexical access",
        "objective": "Return ordered, time/position-indexed candidates, confidence, alternatives and uncertainty while explicitly withholding lexical meaning and sentence interpretation.",
        "mathematical_operations": ("candidate ranking", "modality-local normalization", "uncertainty propagation", "optional caller-supplied cross-modal alignment"),
        "biological_reference": "Handoff from auditory/visual form-processing populations to distributed lexical systems; no single anatomical handoff point is assumed.",
    },
)


_SOURCE_BIOLOGICAL_GROUPS: dict[int, dict[str, tuple[int, ...]]] = {
    1: {"visual": (1,), "auditory": (13,)},
    2: {"visual": (6,), "auditory": (18,)},
    3: {"visual": (1, 2), "auditory": (14, 15, 19)},
    4: {"visual": (11,), "auditory": (15, 29)},
    5: {"visual": (8, 11), "auditory": (15, 25, 32)},
    6: {"visual": (10, 11), "auditory": (16, 20, 37)},
    7: {"visual": (9, 10, 12), "auditory": (35, 39)},
    8: {"visual": (12,), "auditory": (24, 30, 33)},
    9: {"visual": (8, 12), "auditory": (31, 34, 35)},
    10: {"visual": (3, 9, 10), "auditory": (16, 39)},
    11: {"visual": (6, 12), "auditory": (18, 36, 38)},
    12: {"visual": (12,), "auditory": (39,)},
}


@dataclass(frozen=True, slots=True)
class Point1Result:
    """Hold Point 1's source-level functional and biological traces plus the underlying sensory traces.

    The two tracks are deliberately separate. Biological fields describe a simulated LIF reference
    network, never a participant's measured neural activity.
    """

    sample_id: str
    modality_route: Mapping[str, Any]
    form_hypotheses: Mapping[str, Any]
    functional_trace: Sequence[Mapping[str, Any]]
    biological_trace: Sequence[Mapping[str, Any]]
    sensory_operation_trace: Sequence[Mapping[str, Any]]
    sensory_biological_trace: Sequence[Mapping[str, Any]]
    annotation_report: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return bounded JSON-safe output, summarizing large arrays instead of duplicating raw samples."""
        return to_jsonable({
            "sample_id": self.sample_id,
            "modality_route": self.modality_route,
            "form_hypotheses": self.form_hypotheses,
            "functional_trace": self.functional_trace,
            "biological_trace": self.biological_trace,
            "sensory_operation_trace": self.sensory_operation_trace,
            "sensory_biological_trace": self.sensory_biological_trace,
            "annotation_report": self.annotation_report,
            "scientific_scope": {
                "semantic_meaning_retrieved": False,
                "sentence_proposition_built": False,
                "biological_trace_is_measured_data": False,
            },
        })


class Point1PerceptualProcessor:
    """Execute the 12-step source algorithm using ORE's functional sensory and LIF reference engines.

    The processor preserves parallel auditory/visual hypotheses, source-level operations and low-level
    sensory traces. It does not perform lexical access, semantic composition or pragmatic inference.
    """

    def __init__(
        self,
        config: SensoryConfig | None = None,
        *,
        top_k: int = 3,
        register_annotations: bool = True,
        repository_root: str | None = None,
        database_path: str | None = None,
    ) -> None:
        """Initialize the sensory engine, candidate-retention limit and shared SQLite annotation index."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.config = config or SensoryConfig()
        self.top_k = int(top_k)
        self.sensory_processor = SensoryPerceptionProcessor(
            config=self.config,
            register_annotations=False,
        )
        if register_annotations:
            from .annotations import register_point01_annotations

            self.annotation_report = register_point01_annotations(repository_root, database_path)
        else:
            self.annotation_report = {"status": "annotation_registration_disabled"}

    @staticmethod
    def _mean_finite(values: Any) -> float | None:
        """Return the finite mean of a numeric sequence, or None when the sequence has no evidence."""
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        return float(np.mean(finite)) if finite.size else None

    @staticmethod
    def _shape_summary(values: Any) -> dict[str, Any]:
        """Describe a feature array by shape and finite-value statistics without copying its full contents."""
        if values is None:
            return {"available": False}
        array = np.asarray(values)
        if array.size == 0:
            return {"available": True, "shape": list(array.shape), "finite_count": 0}
        numeric = np.asarray(array, dtype=np.float64)
        finite = numeric[np.isfinite(numeric)]
        return {
            "available": True,
            "shape": list(array.shape),
            "finite_count": int(finite.size),
            "mean": float(np.mean(finite)) if finite.size else None,
            "std": float(np.std(finite)) if finite.size else None,
            "minimum": float(np.min(finite)) if finite.size else None,
            "maximum": float(np.max(finite)) if finite.size else None,
        }

    def _auditory_hypotheses(self, result: Any) -> list[dict[str, Any]]:
        """Create time-indexed phonetic-form alternatives from the low-level phoneme posterior sequence."""
        output = result.phoneme_output
        symbols = [str(item) for item in output.get("phoneme_symbols", [])]
        posterior = np.asarray(output.get("candidate_probabilities", []), dtype=np.float64)
        segments = output.get("phoneme_sequence", [])
        hypotheses: list[dict[str, Any]] = []
        for segment in segments:
            start, end = segment.get("frame_span", [0, 0])
            start, end = int(start), int(end)
            alternatives: list[dict[str, Any]] = []
            if posterior.ndim == 2 and posterior.shape[0] and posterior.shape[1] and end > start:
                local = np.mean(posterior[max(0, start):min(end, posterior.shape[0])], axis=0)
                order = np.argsort(local)[::-1][:self.top_k]
                alternatives = [
                    {"symbol": symbols[int(index)] if int(index) < len(symbols) else str(index),
                     "probability": float(local[int(index)])}
                    for index in order
                ]
            hypotheses.append({
                "selected_symbol": segment.get("symbol"),
                "onset_seconds": segment.get("onset_seconds"),
                "offset_seconds": segment.get("offset_seconds"),
                "duration_seconds": segment.get("duration_seconds"),
                "confidence": segment.get("confidence"),
                "uncertainty_nats": segment.get("uncertainty_nats"),
                "frame_span": [start, end],
                "alternatives": alternatives,
                "boundary_status": "estimated_from_acoustic_frames_not_articulatory_ground_truth",
            })
        return hypotheses

    def _visual_hypotheses(self, result: Any) -> list[dict[str, Any]]:
        """Create position-indexed glyph alternatives from explicitly supplied grapheme prototypes and visual evidence."""
        output = result.visual_output
        posterior_items = output.get("letter_posteriors", [])
        ordered_glyphs = output.get("ordered_glyphs", [])
        glyph_by_id = {item.get("glyph_id"): item for item in ordered_glyphs}
        hypotheses: list[dict[str, Any]] = []
        for item in posterior_items:
            glyph = glyph_by_id.get(item.get("glyph_id"), {})
            candidates = sorted(
                item.get("candidates", []),
                key=lambda candidate: float(candidate.get("probability", 0.0)),
                reverse=True,
            )[:self.top_k]
            hypotheses.append({
                "glyph_id": item.get("glyph_id"),
                "region_id": glyph.get("region_id"),
                "bounding_box_y0_x0_y1_x1": glyph.get("bbox"),
                "position_code": glyph.get("position_code"),
                "candidates": [
                    {"grapheme": candidate.get("label"),
                     "probability": float(candidate.get("probability", 0.0))}
                    for candidate in candidates
                ],
                "confidence": float(candidates[0].get("probability", 0.0)) if candidates else None,
                "entropy_nats": item.get("entropy"),
                "status": item.get("status", "unlabelled_visual_shape"),
            })
        return hypotheses

    def _step_01_capture(self, request: SensoryRequest) -> dict[str, Any]:
        """Capture the sampled input contract and preserve its temporal or spatial coordinate system."""
        visual_shape = list(np.asarray(request.visual_image).shape) if request.visual_image is not None else None
        audio = np.asarray(request.auditory_waveform) if request.auditory_waveform is not None else None
        return {
            "visual_shape": visual_shape,
            "auditory_sample_count": int(audio.size) if audio is not None else 0,
            "sample_rate_hz": float(request.sample_rate_hz) if request.sample_rate_hz is not None else None,
            "duration_seconds": float(audio.size / request.sample_rate_hz) if audio is not None else None,
            "language": request.language,
            "coordinate_preservation": "auditory_sample_time_and_visual_pixel_positions",
            "raw_samples_persisted_to_logger": False,
        }

    def _step_02_route(self, result: Any) -> dict[str, Any]:
        """Route only the sensory modalities that were actually supplied by the caller."""
        return {
            "auditory_pathway_active": bool(result.integration.get("auditory_available")),
            "visual_pathway_active": bool(result.integration.get("visual_available")),
            "processing_mode": "parallel" if result.integration.get("auditory_available") and result.integration.get("visual_available") else "single_modality",
            "missing_modalities_fabricated": False,
        }

    def _step_03_extract_features(self, result: Any) -> dict[str, Any]:
        """Expose the modality-specific low-level acoustic and visual feature representations."""
        visual = result.visual_output
        auditory = result.auditory_output
        phonetic = result.phonetic_output
        return {
            "visual": {
                "text_region_count": len(visual.get("text_regions", [])),
                "character_part_count": len(visual.get("character_parts", [])),
                "glyph_hypothesis_count": len(visual.get("glyph_hypotheses", [])),
            },
            "auditory": {
                "cochlear_channel_count": len(auditory.get("cochlear_centers_hz", [])),
                "acoustic_feature_matrix": self._shape_summary(auditory.get("acoustic_feature_matrix")),
                "phonetic_population_code": self._shape_summary(phonetic.get("phonetic_population_code")),
            },
            "feature_sources": "ore_sensory_perception_functional_pipeline",
        }

    def _step_04_normalize(self, result: Any) -> dict[str, Any]:
        """Report the speaker and visual invariance normalization already performed by the sensory pipeline."""
        visual_glyphs = result.visual_output.get("ordered_glyphs", [])
        speaker = result.phonetic_output.get("speaker_statistics", {})
        return {
            "visual_invariant_glyph_count": sum("invariant_features" in glyph for glyph in visual_glyphs),
            "visual_invariance_scope": "scale_and_translation_normalization_while_retaining_order",
            "speaker_statistics_available": bool(speaker),
            "speaker_normalization": to_jsonable(speaker),
            "nuisance_variation_policy": "normalize_when_supported_and_preserve_original_modality_features",
        }

    def _step_05_distributed_encoding(self, result: Any) -> dict[str, Any]:
        """Summarize distributed visual and phonetic population codes without converting them into meanings."""
        return {
            "visual_glyph_population_count": len(result.visual_output.get("glyph_hypotheses", [])),
            "visual_letter_posterior_count": len(result.visual_output.get("letter_posteriors", [])),
            "auditory_phonetic_code": self._shape_summary(result.phonetic_output.get("phonetic_population_code")),
            "auditory_phoneme_posterior": self._shape_summary(result.phoneme_output.get("candidate_probabilities")),
            "representational_policy": "graded_population_evidence_and_alternatives",
        }

    def _step_06_integrate_evidence(self, result: Any) -> dict[str, Any]:
        """Describe temporal accumulation for speech and spatial/ordinal accumulation for visible text."""
        times = result.auditory_output.get("stft_times_seconds", [])
        return {
            "auditory_time_bin_count": len(times),
            "auditory_time_span_seconds": [float(times[0]), float(times[-1])] if len(times) else None,
            "visual_ordered_glyph_count": len(result.visual_output.get("ordered_glyphs", [])),
            "visual_text_region_count": len(result.visual_output.get("text_regions", [])),
            "integration_dimensions": ["auditory_time", "visual_position"],
            "modalities_fused_into_one_code": False,
        }

    def _step_07_subword_candidates(self, result: Any, state: dict[str, Any]) -> dict[str, Any]:
        """Construct ordered phonetic-segment and orthographic-glyph candidates while retaining their modalities."""
        auditory = self._auditory_hypotheses(result)
        visual = self._visual_hypotheses(result)
        state["auditory_form_hypotheses"] = auditory
        state["visual_form_hypotheses"] = visual
        return {
            "auditory_segment_count": len(auditory),
            "visual_glyph_candidate_count": len(visual),
            "auditory_forms": auditory,
            "visual_forms": visual,
            "ordering": "time_ordered_audio_and_position_ordered_visual",
        }

    def _step_08_combine_cues(self, state: dict[str, Any]) -> dict[str, Any]:
        """Calculate modality-local confidence summaries while avoiding unsupported cross-modal fusion."""
        auditory = state.get("auditory_form_hypotheses", [])
        visual = state.get("visual_form_hypotheses", [])
        auditory_confidence = self._mean_finite([item.get("confidence") for item in auditory if item.get("confidence") is not None])
        visual_confidence = self._mean_finite([item.get("confidence") for item in visual if item.get("confidence") is not None])
        return {
            "auditory_mean_candidate_confidence": auditory_confidence,
            "visual_mean_candidate_confidence": visual_confidence,
            "confidence_interpretation": "model_posterior_summary_not_empirical_neural_reliability",
            "cross_modal_fusion": "not_performed_without_explicit_alignment_and_calibration",
        }

    def _step_09_competition(self, state: dict[str, Any]) -> dict[str, Any]:
        """Retain top-k alternatives for every phonetic segment and visual glyph candidate."""
        auditory_alternatives = [
            {"frame_span": item.get("frame_span"), "alternatives": item.get("alternatives", [])}
            for item in state.get("auditory_form_hypotheses", [])
        ]
        visual_alternatives = [
            {"glyph_id": item.get("glyph_id"), "alternatives": item.get("candidates", [])}
            for item in state.get("visual_form_hypotheses", [])
        ]
        return {
            "top_k": self.top_k,
            "auditory_competition": auditory_alternatives,
            "visual_competition": visual_alternatives,
            "winner_take_all_forced": False,
        }

    def _step_10_boundaries(self, result: Any, state: dict[str, Any]) -> dict[str, Any]:
        """Package estimated acoustic segment times and visual region/glyph positions with explicit uncertainty."""
        auditory = [
            {"onset_seconds": item.get("onset_seconds"), "offset_seconds": item.get("offset_seconds"),
             "frame_span": item.get("frame_span"), "status": item.get("boundary_status")}
            for item in state.get("auditory_form_hypotheses", [])
        ]
        visual = [
            {"glyph_id": item.get("glyph_id"), "bbox": item.get("bounding_box_y0_x0_y1_x1"),
             "position_code": item.get("position_code")}
            for item in state.get("visual_form_hypotheses", [])
        ]
        regions = [
            {"region_id": item.get("region_id"), "bbox": item.get("bbox"), "confidence": item.get("confidence")}
            for item in result.visual_output.get("text_regions", [])
        ]
        return {"auditory_segment_boundaries": auditory, "visual_glyph_positions": visual,
                "visual_text_region_boundaries": regions,
                "boundary_interpretation": "candidate_estimates_not_exact_articulatory_or_eye_fixation_boundaries"}

    def _step_11_attention(self, request: SensoryRequest, result: Any) -> dict[str, Any]:
        """Expose caller-supplied attention and prior effects while keeping early prediction separate from meaning."""
        attended_streams = result.auditory_output.get("attended_auditory_streams", [])
        auditory_weights = [
            item.get("attention_weight")
            for item in attended_streams
            if item.get("attention_weight") is not None
        ]
        return {
            "visual_attention_weights": to_jsonable(result.visual_output.get("visual_attention_weights", [])),
            "auditory_attention_weights": to_jsonable(auditory_weights),
            "visual_attention_input_supplied": bool(request.visual_attention),
            "auditory_attention_input_supplied": bool(request.auditory_attention),
            "orthographic_prior_supplied": bool(request.orthographic_prior),
            "phonetic_context_supplied": bool(request.phonetic_context),
            "prediction_scope": "sensory_or_form_level_modulation_only",
        }

    def _step_12_package(self, request: SensoryRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Create the downstream form-hypothesis contract and explicitly withhold lexical and sentence meaning."""
        explicit_alignment = request.metadata.get("orthography_phonology_alignment")
        return {
            "auditory": {
                "form_hypotheses": state.get("auditory_form_hypotheses", []),
                "representation": "ordered_phonetic_segment_candidates",
            },
            "visual": {
                "form_hypotheses": state.get("visual_form_hypotheses", []),
                "representation": "position_ordered_grapheme_candidates",
            },
            "cross_modal_alignment": to_jsonable(explicit_alignment) if explicit_alignment is not None else None,
            "cross_modal_alignment_source": "caller_supplied" if explicit_alignment is not None else "not_inferred",
            "lexical_meaning_retrieved": False,
            "syntactic_structure_built": False,
            "sentence_proposition_composed": False,
            "pragmatic_inference_performed": False,
            "handoff_target": "Point 2 lexical access",
        }

    def _build_biological_trace(self, result: Any) -> list[dict[str, Any]]:
        """Aggregate real low-level LIF reference updates into the 12 source-level algorithm steps."""
        low_level = list(result.biological_trace)
        visual_available = bool(result.integration.get("visual_available"))
        auditory_available = bool(result.integration.get("auditory_available"))
        trace: list[dict[str, Any]] = []
        for step_number, groups in _SOURCE_BIOLOGICAL_GROUPS.items():
            indices: list[int] = []
            if visual_available:
                indices.extend(groups.get("visual", ()))
            if auditory_available:
                indices.extend(groups.get("auditory", ()))
            selected = [low_level[index - 1] for index in indices if 0 < index <= len(low_level)]
            if not low_level:
                trace.append({"operation_id": POINT1_STEP_SPECS[step_number - 1]["operation_id"],
                              "status": "biological_simulation_disabled"})
                continue
            if not selected:
                trace.append({"operation_id": POINT1_STEP_SPECS[step_number - 1]["operation_id"],
                              "status": "skipped_no_matching_sensory_input"})
                continue
            target_names = sorted({str(item.get("target_population")) for item in selected})
            population_rows: dict[str, list[Mapping[str, Any]]] = {}
            for item in selected:
                for population, values in item.get("population_states", {}).items():
                    population_rows.setdefault(str(population), []).append(values)
            population_summary = {
                population: {
                    "mean_voltage_mv": self._mean_finite([row.get("mean_voltage_mv") for row in rows]),
                    "mean_firing_rate_hz": self._mean_finite([row.get("mean_firing_rate_hz") for row in rows]),
                    "spikes": int(sum(int(row.get("spikes_this_update", 0)) for row in rows)),
                }
                for population, rows in population_rows.items()
            }
            trace.append({
                "operation_id": POINT1_STEP_SPECS[step_number - 1]["operation_id"],
                "status": "simulated_reference_not_measured_brain_data",
                "low_level_operation_indices": indices,
                "target_populations": target_names,
                "simulation_time_ms": [selected[0].get("time_ms"), selected[-1].get("time_ms")],
                "network_mean_voltage_mv": self._mean_finite([item.get("network_mean_voltage_mv") for item in selected]),
                "network_mean_rate_hz": self._mean_finite([item.get("network_mean_rate_hz") for item in selected]),
                "population_summary": population_summary,
                "biological_status": "simulated_reference_not_measured_brain_data",
            })
        return trace

    def process(self, request: SensoryRequest) -> Point1Result:
        """Run all 12 source-level operations and return functional, biological and uncertainty traces."""
        if not isinstance(request, SensoryRequest):
            raise TypeError("request must be an ore_sensory_perception.SensoryRequest")
        sensory_result = self.sensory_processor.process(request)
        state: dict[str, Any] = {}
        functional_steps = (
            self._step_01_capture(request),
            self._step_02_route(sensory_result),
            self._step_03_extract_features(sensory_result),
            self._step_04_normalize(sensory_result),
            self._step_05_distributed_encoding(sensory_result),
            self._step_06_integrate_evidence(sensory_result),
            self._step_07_subword_candidates(sensory_result, state),
            self._step_08_combine_cues(state),
            self._step_09_competition(state),
            self._step_10_boundaries(sensory_result, state),
            self._step_11_attention(request, sensory_result),
            self._step_12_package(request, state),
        )
        functional_trace: list[dict[str, Any]] = []
        for specification, functional_result in zip(POINT1_STEP_SPECS, functional_steps, strict=True):
            functional_trace.append({
                **specification,
                "status": "completed",
                "functional_result": to_jsonable(functional_result),
            })
        final_hypotheses = functional_steps[-1]
        modality_route = functional_steps[1]
        return Point1Result(
            sample_id=request.sample_id,
            modality_route=modality_route,
            form_hypotheses=final_hypotheses,
            functional_trace=functional_trace,
            biological_trace=self._build_biological_trace(sensory_result),
            sensory_operation_trace=sensory_result.operation_trace,
            sensory_biological_trace=sensory_result.biological_trace,
            annotation_report=self.annotation_report,
        )
