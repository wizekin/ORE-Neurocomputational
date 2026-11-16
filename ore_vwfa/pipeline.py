"""End-to-end orchestrator for the six-stage, 24-operation VWFA implementation."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .annotations import register_vwfa_annotations
from .biology import VWFABiologicalSimulator
from .contracts import VWFAConfig, VWFARequest, VWFAResult
from .registry import OPERATION_REGISTRY, STAGE_STEP_COUNTS
from .stages import (
    StageAProcessor, StageBProcessor, StageCProcessor, StageDProcessor,
    StageEProcessor, StageFProcessor,
)


class VWFAProcessor:
    """Coordinate functional A–F processing, a separate LIF reference circuit, and authorized literacy learning.

    The functional path estimates visual/orthographic and lexical evidence. The biological path advances
    population-level LIF states. Learning is opt-in and uses only caller-supplied labelled examples. These
    tracks are computational references and must not be described as measured human neural activity.
    """

    def __init__(
        self,
        config: VWFAConfig | None = None,
        *,
        register_annotations: bool = True,
        repository_root: str | Path | None = None,
        database_path: str | Path | None = None,
    ) -> None:
        """Create all six stage processors, the neural reference network, and an isolated learning store."""
        self.config = config or VWFAConfig()
        self.annotation_report = (
            register_vwfa_annotations(repository_root, database_path)
            if register_annotations else {"status": "annotation_registration_disabled"}
        )
        self.stage_processors = {
            "A": StageAProcessor(self.config),
            "B": StageBProcessor(self.config),
            "C": StageCProcessor(self.config),
            "D": StageDProcessor(self.config),
            "E": StageEProcessor(self.config),
            "F": StageFProcessor(self.config),
        }
        self.biological_simulator = VWFABiologicalSimulator(self.config) if self.config.simulate_biology else None
        self.learning_state: dict[str, Any] = {
            "visual_phonological_weights": None,
            "word_selectivity_vector": np.zeros(64, dtype=np.float64),
            "fast_prototypes": {},
            "slow_prototypes": {},
        }

    def reset_learning_state(self) -> None:
        """Clear only this processor's learned associations and memories; retain configuration and logger."""
        self.learning_state = {
            "visual_phonological_weights": None,
            "word_selectivity_vector": np.zeros(64, dtype=np.float64),
            "fast_prototypes": {},
            "slow_prototypes": {},
        }

    @staticmethod
    def _numeric_strength(value: Any) -> float:
        """Extract a bounded scalar activation proxy from a step report for LIF input drive."""
        samples: list[float] = []

        def collect(item: Any) -> None:
            """Visit a small subset of numeric leaves without serializing or traversing large image arrays."""
            if len(samples) >= 64:
                return
            if isinstance(item, (int, float, np.number)) and math.isfinite(float(item)):
                samples.append(abs(float(item)))
            elif isinstance(item, Mapping):
                for child in item.values():
                    collect(child)
            elif isinstance(item, (tuple, list)):
                for child in item[:32]:
                    collect(child)

        from collections.abc import Mapping
        collect(value)
        return float(np.clip(np.mean(samples) if samples else 0.05, 0.0, 1.0))

    @staticmethod
    def _clean_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove binary segmentation masks from serializable region summaries while preserving geometry."""
        return [{key: value for key, value in item.items() if key != "mask"} for item in regions]

    def process(self, request: VWFARequest) -> VWFAResult:
        """Run the exact 24-step registry in A–F order and return separate functional and neural-reference outputs."""
        if not isinstance(request, VWFARequest):
            raise TypeError("request must be a VWFARequest")
        if self.biological_simulator is not None:
            self.biological_simulator.reset_activity()
        state: dict[str, Any] = {
            "request": request,
            "config": self.config,
            "functional_steps": {},
        }
        biological_trace: list[dict[str, Any]] = []
        operation_trace: list[dict[str, Any]] = []
        step_methods = [
            ("A", "step_01"), ("A", "step_02"), ("A", "step_03"), ("A", "step_04"),
            ("B", "step_05"), ("B", "step_06"), ("B", "step_07"), ("B", "step_08"),
            ("C", "step_09"), ("C", "step_10"), ("C", "step_11"), ("C", "step_12"),
            ("D", "step_13"), ("D", "step_14"), ("D", "step_15"), ("D", "step_16"),
            ("D", "step_17"), ("E", "step_18"), ("E", "step_19"), ("E", "step_20"),
            ("E", "step_21"), ("F", "step_22"), ("F", "step_23"), ("F", "step_24"),
        ]
        stage_status: dict[str, str] = {}
        for operation_index, (stage, method_name) in enumerate(step_methods, start=1):
            method = getattr(self.stage_processors[stage], method_name)
            if stage == "F":
                result = method(state, self.learning_state)
            else:
                result = method(state)
            operation_id = f"VWFA-{operation_index:03d}"
            state["functional_steps"][operation_id] = result
            biological_result = None
            if self.biological_simulator is not None:
                biological_result = self.biological_simulator.advance_operation(
                    operation_index, self._numeric_strength(result)
                )
                if request.learning_authorized and operation_index == 22:
                    plasticity = self.biological_simulator.apply_hebbian_update(
                        {"vwfa_posterior": 0.5}, {"phonological_association": 0.5}
                    )
                    biological_result["authorized_plasticity"] = plasticity
                if request.learning_authorized and operation_index == 23:
                    plasticity = self.biological_simulator.apply_hebbian_update(
                        {"vwfa_intermediate": 0.5}, {"vwfa_anterior": 0.5, "orthographic_memory": 0.5}
                    )
                    biological_result["authorized_plasticity"] = plasticity
                if request.learning_authorized and operation_index == 24:
                    biological_result["consolidation"] = self.biological_simulator.consolidate_weights()
                biological_trace.append(biological_result)
            spec = OPERATION_REGISTRY[operation_index - 1]
            status = str(result.get("status", "completed")) if isinstance(result, dict) else "completed"
            stage_status[stage] = "completed"
            operation_trace.append({
                "operation_id": spec.operation_id,
                "stage": spec.stage,
                "stage_name": spec.stage_name,
                "step": spec.step,
                "operation_name": spec.operation_name,
                "status": status,
                "functional_output": result,
                "biological_output": biological_result if biological_result is not None else "disabled",
                "mathematical_operations": list(spec.mathematical_operations),
                "biological_reference": spec.biological_reference,
            })

        functional = {
            "stage_outputs": state["functional_steps"],
            "input_source": "visual_image" if request.visual_image is not None else "caller_supplied_orthographic_sequence",
            "visual_available": bool(state.get("visual_available", False)),
            "grapheme_recognition_status": state.get("grapheme_recognition_status", "not_run"),
            "candidate_regions": self._clean_regions(state.get("regions", [])),
            "position_codes": state.get("position_codes", []),
            "orthographic_sequence": state.get("orthographic_sequence", []),
            "grapheme_posteriors": state.get("grapheme_posteriors", []),
            "sublexical_ngrams": state.get("sublexical_ngrams", []),
            "whole_word_candidates": state.get("stabilized_candidates", state.get("whole_word_candidates", [])),
            "morphology_candidates": state.get("morphology_candidates", []),
            "parallel_morphology_activity": state.get("parallel_morphology_activity", {}),
            "composed_language_features": state.get("composed_language_features", {}),
            "selected_candidate": state.get("selected_candidate"),
            "network_regions": state.get("regional_activity", {}),
            "attention_weights": np.asarray(state.get("attention_weights", []), dtype=np.float64).tolist(),
            "invariance_score": float(state.get("invariance_score", 0.0)),
            "learning_state": {
                "fast_memory_count": len(self.learning_state.get("fast_prototypes", {})),
                "slow_memory_count": len(self.learning_state.get("slow_prototypes", {})),
                "visual_phonological_weight_shape": (
                    list(self.learning_state["visual_phonological_weights"].shape)
                    if self.learning_state.get("visual_phonological_weights") is not None else None
                ),
                "learning_authorized_for_this_request": bool(request.learning_authorized),
            },
        }
        biological = {
            "enabled": self.biological_simulator is not None,
            "model_scope": (
                "Separate population-level current-based LIF reference; not an anatomically fitted or "
                "empirically validated simulation of an individual human brain."
            ),
            "operation_trace": biological_trace,
            "final_snapshot": self.biological_simulator.snapshot() if self.biological_simulator else None,
            "activity_time_series": self.biological_simulator.activity_series().tolist()
            if self.biological_simulator else [],
        }
        finite_checks = []
        for key in ("normalized_image", "gabor_stack", "conjunction_map", "posterior_vwfa_activity",
                    "anterior_vwfa_activity", "refined_orthographic_state", "attended_representation"):
            value = state.get(key)
            if value is not None:
                finite_checks.append(bool(np.all(np.isfinite(np.asarray(value, dtype=np.float64)))))
        diagnostics = {
            "valid": all(finite_checks) if finite_checks else True,
            "finite_checks": len(finite_checks),
            "stage_count": len(STAGE_STEP_COUNTS),
            "operation_count": len(operation_trace),
            "operation_ids_complete": [item["operation_id"] for item in operation_trace]
            == [item.operation_id for item in OPERATION_REGISTRY],
            "stage_status": stage_status,
            "learning_applied": bool(request.learning_authorized and request.learning_examples),
            "annotation_registration": self.annotation_report,
        }
        metadata = {
            "schema": "ore.vwfa.six_stage.1.0",
            "language": request.language,
            "subject_id": request.subject_id,
            "config": self.config.to_dict(),
            "operation_registry": [item.to_dict() for item in OPERATION_REGISTRY],
            "biological_scope": "computational reference; independent human validation not performed",
            "input_contract": "no OCR or character segmentation is inferred from the supplied image",
        }
        return VWFAResult(request.sample_id, functional, biological, operation_trace, diagnostics, metadata)
