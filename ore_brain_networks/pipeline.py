"""Integrated twelve-operation functional and biological-reference BNBEA pipeline."""
from __future__ import annotations

from typing import Any

import numpy as np

from .annotations import register_bnbea_annotations
from .auditory_streams import AuditoryStreamProcessor
from .biology import POPULATIONS, NeuralDynamicsSimulator, normalized_drive
from .contracts import BNBEAConfig, BNBEARequest, BNBEAResult
from .eeg_meg import EEGMEGModel
from .fmri import FMRIMeasurementModel
from .grapheme_populations import GraphemePopulationDecoder
from .human_validation import HumanReadingValidation
from .language_network import LanguageNetworkProcessor, REGIONS
from .lesion_validation import LesionValidationModel
from .registry import OPERATION_REGISTRY
from .tms_validation import TMSValidationModel
from .visual_gabor import GaborVisualProcessor
from .visual_hierarchy import VisualHierarchyProcessor
from .vwfa_hierarchy import VWFAHierarchyProcessor


_STAGE_DRIVES = {
    1: ("retina_lgn",),
    2: ("v1",),
    3: ("v2",),
    4: ("v4", "parietal_dorsal"),
    5: ("vwfa_posterior", "vwfa_anterior"),
    6: ("mtg", "ifg", "atl"),
    7: ("stg_ventral", "stg_dorsal"),
}


class BNBEAProcessor:
    """Coordinate BNBEA's twelve operations while retaining separate computational evidence tracks.

    The processor does not perform OCR or speech recognition from labels. Grapheme prototypes and
    an optional orthographic sequence are supplied upstream; auditory output is a phonological feature
    representation rather than a fabricated transcript. EEG/MEG, fMRI, lesion, TMS and human outcomes
    are forward models or analyses, not claims of measured participant biology.
    """

    def __init__(
        self,
        config: BNBEAConfig | None = None,
        *,
        register_annotations: bool = True,
        repository_root: str | None = None,
        database_path: str | None = None,
    ) -> None:
        """Initialize subsystem processors and optionally synchronize their docstrings to ORE SQLite."""
        self.config = config or BNBEAConfig()
        self.annotation_report = (
            register_bnbea_annotations(repository_root, database_path)
            if register_annotations else {"status": "annotation_registration_disabled"}
        )
        self.visual_processor = GaborVisualProcessor(self.config)
        self.grapheme_decoder = GraphemePopulationDecoder(self.config)
        self.hierarchy_processor = VisualHierarchyProcessor(self.config)
        self.vwfa_processor = VWFAHierarchyProcessor()
        self.language_processor = LanguageNetworkProcessor(self.config)
        self.auditory_processor = AuditoryStreamProcessor()
        self.neural_simulator = NeuralDynamicsSimulator(self.config) if self.config.simulate_biology else None
        self.eeg_meg_model = EEGMEGModel(self.config)
        self.fmri_model = FMRIMeasurementModel(
            sample_interval_seconds=self.config.dt_ms / 1000.0,
            duration_seconds=self.config.hrf_duration_seconds,
        )
        self.lesion_model = LesionValidationModel()
        self.tms_model = TMSValidationModel()
        self.human_validator = HumanReadingValidation()

    def _trace(
        self,
        operation_index: int,
        status: str,
        functional_output: str,
        biological_output: str,
        experimental_output: str,
    ) -> dict[str, Any]:
        """Build one stable operation trace row with the three evidence channels kept distinct."""
        spec = OPERATION_REGISTRY[operation_index - 1]
        return {
            "operation_id": spec.operation_id,
            "operation_name": spec.operation_name,
            "status": status,
            "functional_output": functional_output,
            "biological_output": biological_output,
            "experimental_output": experimental_output,
            "mathematical_operations": list(spec.mathematical_operations),
            "biological_reference": spec.biological_reference,
        }

    def _advance_biology(self, operation_index: int, strength: float) -> dict[str, Any] | None:
        """Advance the LIF populations using an operation-specific normalized sensory/network drive."""
        if self.neural_simulator is None or operation_index not in _STAGE_DRIVES:
            return None
        drive = normalized_drive(strength)
        for population in _STAGE_DRIVES[operation_index]:
            self.neural_simulator.advance(
                {population: drive}, steps=self.config.lif_steps_per_stage
            )
        snapshot = self.neural_simulator.snapshot()
        return {
            "operation_index": operation_index,
            "driven_populations": list(_STAGE_DRIVES[operation_index]),
            "normalized_drive": drive,
            "time_ms": snapshot["time_ms"],
            "population_activity": {
                name: snapshot["populations"][name]["mean_activity"]
                for name in POPULATIONS
            },
        }

    @staticmethod
    def _source_summary(source_activity: np.ndarray, names: tuple[str, ...]) -> dict[str, np.ndarray]:
        """Convert the LIF time-by-population history into region-keyed traces for the fMRI model."""
        return {
            name: source_activity[:, index]
            for index, name in enumerate(names)
            if source_activity.shape[0] > 0
        }

    def process(self, request: BNBEARequest) -> BNBEAResult:
        """Execute all twelve operation contracts and return functional, biological and empirical tracks."""
        if not isinstance(request, BNBEARequest):
            raise TypeError("request must be a BNBEARequest")
        functional: dict[str, Any] = {}
        biological_steps: list[dict[str, Any]] = []
        experimental: dict[str, Any] = {}
        trace: list[dict[str, Any]] = []
        visual_features: np.ndarray | None = None
        auditory_output: dict[str, Any] | None = None
        hierarchy_output: dict[str, Any] | None = None
        grapheme_output: dict[str, Any] | None = None
        language_output: dict[str, Any] | None = None
        orthographic_sequence = list(request.orthographic_sequence)

        # Operation 1 — early visual filters and energy maps.
        if request.visual_image is not None:
            visual_output = self.visual_processor.process(request.visual_image)
            visual_features = np.asarray(visual_output["feature_stack"], dtype=np.float64)
            functional["early_visual"] = visual_output
            strength = float(np.mean(np.abs(visual_features)) * 5.0)
            if self.neural_simulator:
                biological_steps.append(self._advance_biology(1, strength))
            trace.append(self._trace(
                1, "completed", "Gabor maps and local orientation energy computed",
                "Retina/LGN input population advanced" if self.neural_simulator else "disabled",
                "not_applicable",
            ))
        else:
            trace.append(self._trace(
                1, "skipped_no_visual_input", "no visual input", "not_applicable", "not_applicable"
            ))

        # Operation 2 — distributed grapheme tuning and ordinal sequence.
        if visual_features is not None:
            grapheme_output = self.grapheme_decoder.process(
                visual_features, request.grapheme_prototypes
            )
            functional["grapheme_populations"] = grapheme_output
            if not orthographic_sequence:
                orthographic_sequence = [
                    str(item["grapheme"])
                    for item in grapheme_output["position_preserving_sequence"]
                ]
            confidence = float(np.mean([
                item["confidence"] for item in grapheme_output["candidate_regions"]
            ])) if grapheme_output["candidate_regions"] else 0.0
            if self.neural_simulator:
                biological_steps.append(self._advance_biology(2, confidence))
            trace.append(self._trace(
                2, "completed", grapheme_output["recognition_status"],
                "V1-like population drive advanced" if self.neural_simulator else "disabled",
                "not_applicable",
            ))
        else:
            trace.append(self._trace(
                2, "skipped_no_visual_features", "no visual feature maps",
                "not_applicable", "not_applicable",
            ))

        # Operation 3 — recurrent LIF network gets its initial sensory drive from operations 1–2.
        if self.neural_simulator is not None:
            if not any(item and item["operation_index"] == 1 for item in biological_steps):
                biological_steps.append(self._advance_biology(3, 0.0))
            else:
                drive_strength = max(
                    float(np.mean(np.abs(visual_features))) if visual_features is not None else 0.0,
                    0.15,
                )
                biological_steps.append(self._advance_biology(3, drive_strength))
            trace.append(self._trace(
                3, "completed", "functional representations retained independently",
                "LIF voltage, current, refractory, spike and recurrence state advanced",
                "not_applicable",
            ))
        else:
            trace.append(self._trace(
                3, "disabled_by_config", "functional track retained", "LIF disabled", "not_applicable"
            ))

        # Operation 4 — anatomical visual hierarchy and top-down spatial attention.
        if visual_features is not None:
            hierarchy_output = self.hierarchy_processor.process(visual_features)
            functional["visual_hierarchy"] = hierarchy_output
            attention_strength = float(np.mean(hierarchy_output["attention_weighted_features"]))
            if self.neural_simulator:
                biological_steps.append(self._advance_biology(4, abs(attention_strength)))
            trace.append(self._trace(
                4, "completed", "Retina/LGN/V1/V2/V4 and ventral/dorsal maps computed",
                "visual and dorsal-attention populations advanced" if self.neural_simulator else "disabled",
                "not_applicable",
            ))
        else:
            trace.append(self._trace(
                4, "skipped_no_visual_features", "no visual hierarchy input",
                "not_applicable", "not_applicable",
            ))

        # Operation 5 — VWFA levels, using only upstream orthography or decoded prototype sequence.
        if orthographic_sequence:
            visual_summary = (
                np.asarray(hierarchy_output["layers"]["ventral_temporal"])
                if hierarchy_output else None
            )
            vwfa_output = self.vwfa_processor.process(
                orthographic_sequence, visual_summary, request.lexicon
            )
            functional["vwfa_hierarchy"] = vwfa_output
            if self.neural_simulator:
                biological_steps.append(
                    self._advance_biology(5, float(vwfa_output["lexical_confidence"]))
                )
            trace.append(self._trace(
                5, "completed", "ordered posterior, n-gram and lexical representations computed",
                "posterior/anterior VWFA populations advanced" if self.neural_simulator else "disabled",
                "not_applicable",
            ))
        else:
            vwfa_output = {"status": "awaiting_orthographic_sequence_or_prototypes"}
            functional["vwfa_hierarchy"] = vwfa_output
            trace.append(self._trace(
                5, "awaiting_orthographic_sequence", "no sequence fabricated",
                "not_applicable", "not_applicable",
            ))

        # Operation 6 — distributed IFG/MTG/ATL network with available visual/context evidence.
        lexical_candidate = (
            vwfa_output.get("best_lexical_candidate")
            if isinstance(vwfa_output, dict) else None
        )
        language_output = self.language_processor.process(
            lexical_candidate, request.semantic_features, request.context_features
        )
        functional["language_network"] = language_output
        if self.neural_simulator:
            language_strength = float(np.mean(
                list(language_output["regional_activity_summary"].values())
            ))
            biological_steps.append(self._advance_biology(6, language_strength))
        trace.append(self._trace(
            6, "completed", "IFG/MTG/ATL recurrent integration computed",
            "MTG/IFG/ATL populations advanced" if self.neural_simulator else "disabled",
            "functional connectivity estimate; not empirical connectivity",
        ))

        # Operation 7 — speech sensory analysis and dorsal/ventral pathways; then cross-modal integration.
        if request.auditory_waveform is not None:
            auditory_output = self.auditory_processor.process(
                request.auditory_waveform, request.audio_sample_rate_hz
            )
            functional["auditory_streams"] = auditory_output
            if self.neural_simulator:
                auditory_drive = float(np.mean(
                    auditory_output["phonological_feature_vector"] > 0
                ))
                biological_steps.append(self._advance_biology(7, auditory_drive))
            language_output = self.language_processor.process(
                lexical_candidate, request.semantic_features, request.context_features,
                auditory_vector=np.asarray(auditory_output["phonological_feature_vector"]),
            )
            functional["language_network"] = language_output
            functional["cross_modal_integration"] = {
                "visual_orthography_available": bool(orthographic_sequence),
                "auditory_phonological_features_available": True,
                "shared_concept_vector": language_output["integrated_concept_vector"],
                "modality_evidence_kept_separate": True,
            }
            trace.append(self._trace(
                7, "completed", "STFT/cochlear-inspired ventral and dorsal vectors computed",
                "STG ventral/dorsal populations advanced" if self.neural_simulator else "disabled",
                "no transcript or human speech labels inferred",
            ))
        else:
            trace.append(self._trace(
                7, "skipped_no_auditory_input", "no acoustic waveform supplied",
                "not_applicable", "not_applicable",
            ))

        # Operation 8 — EEG/MEG forward model; generic lead fields are explicitly marked as toy models.
        source_activity = (
            self.neural_simulator.source_time_series()
            if self.neural_simulator is not None
            else np.zeros((0, len(POPULATIONS)), dtype=np.float64)
        )
        population_activity = (
            self.neural_simulator.activity_time_series()
            if self.neural_simulator is not None
            else np.zeros((0, len(POPULATIONS)), dtype=np.float64)
        )
        if source_activity.shape[0] > 0:
            eeg_meg_output = self.eeg_meg_model.process(
                source_activity, POPULATIONS, request.experiment_data
            )
            experimental["eeg_meg"] = eeg_meg_output
            trace.append(self._trace(
                8, "completed", "functional model states were not converted to biological facts",
                "source activity projected through EEG/MEG lead-field models",
                str(eeg_meg_output["status"]),
            ))
        else:
            eeg_meg_output = {"status": "disabled_or_no_neural_time_series"}
            experimental["eeg_meg"] = eeg_meg_output
            trace.append(self._trace(
                8, "unavailable_no_neural_sources", "not_applicable",
                "no neural sources", "not_applicable",
            ))

        # Operation 9 — hemodynamic proxy and fMRI model/observed RDM comparison.
        if population_activity.shape[0] > 0:
            regional_activity = self._source_summary(population_activity, POPULATIONS)
            fmri_output = self.fmri_model.process(regional_activity, request.experiment_data)
            experimental["fmri"] = fmri_output
            trace.append(self._trace(
                9, "completed", "regional population patterns retained",
                "neural activity convolved with canonical double-gamma HRF",
                str(fmri_output["status"]),
            ))
        else:
            fmri_output = {"status": "disabled_or_no_neural_time_series"}
            experimental["fmri"] = fmri_output
            trace.append(self._trace(
                9, "unavailable_no_neural_sources", "not_applicable",
                "no neural sources", "not_applicable",
            ))

        # Operation 10 — virtual lesions and optional patient lesion-symptom analysis.
        lesion_regions = (
            {
                name: language_output["regional_activity"].get(name, np.zeros(1))
                for name in REGIONS
            } if language_output else {}
        )
        lesion_output = self.lesion_model.process(
            lesion_regions, request.experiment_data,
            edge_weights=language_output.get("connection_weights", {}) if language_output else None,
        )
        experimental["lesion_validation"] = lesion_output
        trace.append(self._trace(
            10,
            "completed" if lesion_output.get("status") == "virtual_lesion_simulated"
            else "awaiting_intervention_or_patient_data",
            "functional candidate representations unchanged by validation",
            "virtual regional/edge intervention only when supplied",
            str(lesion_output.get("status")),
        ))

        # Operation 11 — virtual TMS contrasts, never human stimulation instructions.
        if population_activity.shape[0] > 0:
            tms_data = dict(request.experiment_data)
            if tms_data.get("tms_conditions"):
                tms_conditions = [dict(item) for item in tms_data["tms_conditions"]]
                for item in tms_conditions:
                    item.setdefault("sample_interval_ms", self.config.dt_ms)
                tms_data["tms_conditions"] = tms_conditions
            tms_output = self.tms_model.process(population_activity, POPULATIONS, tms_data)
        else:
            tms_output = {"status": "unavailable_no_neural_sources"}
        experimental["tms_validation"] = tms_output
        trace.append(self._trace(
            11,
            "completed" if tms_output.get("status") == "virtual_perturbations_simulated"
            else "awaiting_virtual_conditions_or_data",
            "functional candidate representations unchanged by validation",
            "unitless virtual perturbation, no physical stimulation values",
            str(tms_output.get("status")),
        ))

        # Operation 12 — independent-data-only validation, using caller-provided paired predictions.
        human_data = dict(request.experiment_data.get("human_validation", {}))
        if "human_eeg" in human_data and "eeg_sensor_prediction" in eeg_meg_output:
            human_data.setdefault("model_eeg", eeg_meg_output["eeg_sensor_prediction"])
        if "human_meg" in human_data and "meg_sensor_prediction" in eeg_meg_output:
            human_data.setdefault("model_meg", eeg_meg_output["meg_sensor_prediction"])
        if "human_rdm" in human_data and "model_rdm" not in human_data and "model_rdm" in fmri_output:
            human_data["model_rdm"] = fmri_output["model_rdm"]
        validation_output = self.human_validator.process(human_data)
        experimental["human_reading_validation"] = validation_output
        trace.append(self._trace(
            12,
            "completed" if validation_output.get("status") == "evaluated_supplied_independent_data"
            else "awaiting_independent_human_data",
            "model outputs available for comparison; no human scores fabricated",
            "simulated signals are candidate predictions only",
            str(validation_output.get("status")),
        ))

        biological = {
            "enabled": self.neural_simulator is not None,
            "operation_advances": [item for item in biological_steps if item is not None],
            "lif": self.neural_simulator.snapshot() if self.neural_simulator else None,
            "source_time_series": source_activity,
            "population_activity_time_series": population_activity,
            "population_names": list(POPULATIONS),
            "scope": (
                "Explicit population-level current-based LIF computational hypothesis. It is not an "
                "anatomically fitted, empirically validated or literal simulation of the human brain."
            ),
        }
        diagnostics = {
            "operation_count": len(trace),
            "registered_operation_count": len(OPERATION_REGISTRY),
            "operation_ids_complete": (
                [item["operation_id"] for item in trace]
                == [item.operation_id for item in OPERATION_REGISTRY]
            ),
            "functional_operations": sum(item["status"] == "completed" for item in trace),
            "awaiting_or_skipped_operations": sum(item["status"] != "completed" for item in trace),
            "separate_evidence_tracks": True,
            "annotation_registration": self.annotation_report,
        }
        metadata = {
            "algorithm": "BNBEA",
            "version": "1.0",
            "language": request.language,
            "config": self.config.to_dict(),
            "mathematical_scope": "Numerical approximations to candidate neural computations; not literal neural algorithms.",
            "scientific_limitations": [
                "Gabor filters, feature hashing, graph weights and LIF populations are engineering approximations.",
                "No OCR, automatic speech recognition, or unprovided lexical inventory is fabricated.",
                "Synthetic EEG/MEG lead fields are not participant head models.",
                "Predicted BOLD signals are not scanner measurements.",
                "Virtual lesions and virtual TMS are not clinical or participant interventions.",
                "Biological validity requires independently collected human behavior and neural recordings.",
            ],
        }
        return BNBEAResult(
            sample_id=request.sample_id,
            functional=functional,
            biological=biological,
            experimental=experimental,
            operation_trace=trace,
            diagnostics=diagnostics,
            metadata=metadata,
        )
