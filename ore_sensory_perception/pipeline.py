"""Coordinate the six-stage visual/auditory sensory-perception algorithm and its LIF reference track."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .annotations import register_sensory_annotations
from .auditory import AcousticFeatureExtractor, AuditorySignalProcessor
from .biology import SensoryBiologicalSimulator
from .contracts import SensoryConfig, SensoryRequest, SensoryResult, to_jsonable
from .phonemes import PhonemeIdentifier
from .phonetics import PhoneticFeatureDetector
from .registry import OPERATION_REGISTRY
from .visual import CharacterSegmentationProcessor, TextRegionDetector


class SensoryPerceptionProcessor:
    """Run the six registered sensory stages in order while preserving visual/auditory separation.

    The functional track produces candidate text regions and character hypotheses from images, plus
    acoustic, phonetic and phoneme evidence from audio. The independent biological track advances
    configurable recurrent LIF populations. Neither path claims to reproduce a complete human brain.
    """

    def __init__(
        self,
        config: SensoryConfig | None = None,
        *,
        register_annotations: bool = True,
        repository_root: str | None = None,
        database_path: str | None = None,
    ) -> None:
        """Create all six stage processors and optionally index every repository definition in SQLite.

        The default logger database is ``ore_logs/ore_logger.sqlite3``. Registration uses the existing
        ORE annotation catalogue; no separate package-specific database or logger schema is created.
        """
        self.config = config or SensoryConfig()
        self.annotation_report = (
            register_sensory_annotations(repository_root, database_path)
            if register_annotations else {"status": "annotation_registration_disabled"}
        )
        self.text_detector = TextRegionDetector(self.config)
        self.character_processor = CharacterSegmentationProcessor(self.config)
        self.auditory_processor = AuditorySignalProcessor(self.config)
        self.acoustic_extractor = AcousticFeatureExtractor(self.config)
        self.phonetic_detector = PhoneticFeatureDetector(self.config)
        self.phoneme_identifier = PhonemeIdentifier(self.config)
        self.biological_simulator = (
            SensoryBiologicalSimulator(self.config) if self.config.simulate_biology else None
        )

    @staticmethod
    def _activity_strength(value: Any) -> float:
        """Compress a stage report into a bounded non-negative activity proxy for the LIF input current.

        Only semantically meaningful numeric summaries contribute. Shape dimensions, raw counts and
        timestamps are ignored so a large image or long recording cannot create artificial neural gain.
        """
        selected: list[float] = []
        preferred = ("mean", "confidence", "probability", "posterior", "energy", "salience",
                     "score", "rate", "entropy", "margin", "voiced_candidate_fraction")

        def collect(item: Any, key: str = "") -> None:
            """Recursively collect bounded numeric evidence from a small operation summary dictionary."""
            if isinstance(item, dict):
                for child_key, child_value in item.items():
                    if child_key in {"shape", "bbox", "frame_span", "operation_index", "sample_count"}:
                        continue
                    collect(child_value, str(child_key).lower())
            elif isinstance(item, (tuple, list)):
                for child in item[:64]:
                    collect(child, key)
            elif isinstance(item, (int, float, np.number)) and np.isfinite(float(item)):
                if any(token in key for token in preferred):
                    selected.append(float(item))

        collect(value)
        if not selected:
            return 0.04
        scale = float(np.mean(np.clip(np.abs(selected), 0.0, 1.0)))
        return float(np.clip(np.tanh(scale), 0.0, 1.0))

    @staticmethod
    def _collect_output(state: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
        """Copy selected named state values into an independently serializable modality output."""
        return {name: to_jsonable(state[name]) for name in names if name in state}

    def process(self, request: SensoryRequest) -> SensoryResult:
        """Execute the complete 39-operation registry and return modality outputs, traces and neural states.

        Missing modalities are marked as skipped instead of being replaced by synthetic input. Each
        functional operation is logged with its registry specification, and each simulated neural
        update is stored separately to avoid conflating modeled activity with empirical measurements.
        """
        if not isinstance(request, SensoryRequest):
            raise TypeError("request must be a SensoryRequest")
        if self.biological_simulator is not None:
            self.biological_simulator.reset()
        state: dict[str, Any] = {"request": request}
        steps: list[tuple[Callable[[dict[str, Any]], dict[str, Any]], int]] = [
            (self.text_detector.step_01, 1), (self.text_detector.step_02, 2),
            (self.text_detector.step_03, 3), (self.text_detector.step_04, 4),
            (self.text_detector.step_05, 5), (self.text_detector.step_06, 6),
            (self.character_processor.step_07, 7), (self.character_processor.step_08, 8),
            (self.character_processor.step_09, 9), (self.character_processor.step_10, 10),
            (self.character_processor.step_11, 11), (self.character_processor.step_12, 12),
            (self.auditory_processor.step_13, 13), (self.auditory_processor.step_14, 14),
            (self.auditory_processor.step_15, 15), (self.auditory_processor.step_16, 16),
            (self.auditory_processor.step_17, 17), (self.auditory_processor.step_18, 18),
            (self.acoustic_extractor.step_19, 19), (self.acoustic_extractor.step_20, 20),
            (self.acoustic_extractor.step_21, 21), (self.acoustic_extractor.step_22, 22),
            (self.acoustic_extractor.step_23, 23), (self.acoustic_extractor.step_24, 24),
            (self.phonetic_detector.step_25, 25), (self.phonetic_detector.step_26, 26),
            (self.phonetic_detector.step_27, 27), (self.phonetic_detector.step_28, 28),
            (self.phonetic_detector.step_29, 29), (self.phonetic_detector.step_30, 30),
            (self.phonetic_detector.step_31, 31), (self.phoneme_identifier.step_32, 32),
            (self.phoneme_identifier.step_33, 33), (self.phoneme_identifier.step_34, 34),
            (self.phoneme_identifier.step_35, 35), (self.phoneme_identifier.step_36, 36),
            (self.phoneme_identifier.step_37, 37), (self.phoneme_identifier.step_38, 38),
            (self.phoneme_identifier.step_39, 39),
        ]
        operation_trace: list[dict[str, Any]] = []
        biological_trace: list[dict[str, Any]] = []
        for index, (function, expected_index) in enumerate(steps, start=1):
            if index != expected_index or index > len(OPERATION_REGISTRY):
                raise RuntimeError("runtime step sequence does not match the canonical 39-operation registry")
            result = function(state)
            specification = OPERATION_REGISTRY[index - 1]
            operation_trace.append({
                **specification.to_dict(),
                "status": str(result.get("status", "completed")),
                "functional_result": to_jsonable(result),
            })
            if self.biological_simulator is not None:
                biological_trace.append(self.biological_simulator.advance_operation(
                    index, specification.operation_id, self._activity_strength(result)
                ))
        visual_output = self._collect_output(state, (
            "text_regions", "attended_regions", "visual_attention_weights", "character_parts",
            "glyph_hypotheses", "segmentation_alternatives", "ordered_glyphs", "letter_posteriors",
        ))
        auditory_output = self._collect_output(state, (
            "cochlear_centers_hz", "auditory_nerve_rates", "temporal_envelopes", "onset_times_seconds",
            "auditory_stream_candidates", "attended_auditory_streams", "stft_frequencies_hz",
            "stft_times_seconds", "spectral_centroid_hz", "frame_rms", "f0_hz", "formants_hz",
            "vot_seconds", "acoustic_feature_matrix",
        ))
        phonetic_output = self._collect_output(state, (
            "voicing_probability", "manner_labels", "manner_posteriors", "place_labels",
            "place_posteriors", "vowel_feature_estimates", "speaker_statistics",
            "coarticulatory_trajectory", "phonetic_population_code",
        ))
        phoneme_output = self._collect_output(state, (
            "phoneme_symbols", "category_likelihoods", "temporally_integrated_posteriors",
            "categorical_voicing_posteriors", "contextual_phoneme_posteriors",
            "recurrent_phoneme_state", "selected_phoneme_frames", "candidate_probabilities",
            "phoneme_frame_confidence", "phoneme_entropy", "phoneme_sequence", "sequence_confidence",
        ))
        integration = {
            "visual_available": bool(state.get("visual_available", False)),
            "auditory_available": bool(state.get("auditory_available", False)),
            "text_region_count": len(state.get("text_regions", [])),
            "character_hypothesis_count": len(state.get("glyph_hypotheses", [])),
            "phoneme_segment_count": len(state.get("phoneme_sequence", [])),
            "shared_language_interface": {
                "orthographic_candidates": visual_output.get("letter_posteriors", []),
                "phonological_candidates": phoneme_output.get("candidate_probabilities", []),
                "cross_modal_alignment": to_jsonable(request.metadata.get("orthography_phonology_alignment")),
                "alignment_source": "caller_supplied_only" if request.metadata.get("orthography_phonology_alignment") is not None else "not_inferred",
            },
            "biological_scope": "parallel_and_recurrent_reference_model_not_a_literal_brain_execution_trace",
        }
        return SensoryResult(
            sample_id=request.sample_id,
            visual_output=visual_output,
            auditory_output=auditory_output,
            phonetic_output=phonetic_output,
            phoneme_output=phoneme_output,
            operation_trace=operation_trace,
            biological_trace=biological_trace,
            integration=integration,
            annotation_report=self.annotation_report,
        )
