"""End-to-end 17-stage developmental language and executive-cognition processor."""
from __future__ import annotations

from pathlib import Path
import math
import time
from typing import Any, Mapping

from .biology import DevelopmentalBiologicalNetwork
from .cognition import (
    calibrate_developmental_support, integrate_hippocampal_prefrontal_control,
    maintain_working_memory, refine_developing_circuits, refine_multi_step_plan,
    update_executive_control,
)
from .contracts import (
    DevelopmentalConfig, DevelopmentalExperience, DevelopmentalLearnerState,
    DevelopmentalProcessingResult,
)
from .functional import (
    accumulate_statistical_regularities, build_provisional_phonetic_categories,
    clamp, establish_broad_sensitivity, extract_acoustic_features, infer_word_boundaries,
    integrate_multimodal_referents, tune_environment_specific_perception,
)
from .grounded_learning import (
    accelerate_vocabulary_learning, build_grounded_concepts, map_word_forms_to_meanings,
    stabilize_developmental_memory,
)
from .registry import OPERATION_REGISTRY


def _record(
    trace: list[dict[str, Any]],
    index: int,
    functional_output: Mapping[str, Any],
    biological_output: Mapping[str, Any],
) -> None:
    """Append one ordered operation with its equation, purpose, and biological trace."""
    spec = OPERATION_REGISTRY[index]
    trace.append({
        "step": index + 1, "operation_id": spec.operation_id, "operation": spec.name,
        "status": "completed", "purpose": spec.purpose, "evidence_level": spec.evidence_level,
        "mathematical_form": spec.mathematical_form,
        "functional_output": dict(functional_output),
        "biological_output": dict(biological_output),
    })


class DevelopmentalLanguageCognitionProcessor:
    """Coordinate the full developmental path while preserving separate domain-specific state."""

    def __init__(
        self,
        config: DevelopmentalConfig | None = None,
        state: DevelopmentalLearnerState | None = None,
        biological: DevelopmentalBiologicalNetwork | None = None,
    ) -> None:
        """Initialize developmental dynamics and seed WM capacity from the selected configuration."""
        self.config = config or DevelopmentalConfig()
        self.state = state if state is not None else DevelopmentalLearnerState(
            working_memory_capacity=self.config.memory_capacity
        )
        self.biological = biological or DevelopmentalBiologicalNetwork()
        self.process_count = 0

    def process(
        self,
        event: DevelopmentalExperience,
        state: DevelopmentalLearnerState | None = None,
    ) -> DevelopmentalProcessingResult:
        """Execute all 17 developmental stages and return their separate intermediate results."""
        active_state = state if state is not None else self.state
        config = self.config
        biological = self.biological
        trace: list[dict[str, Any]] = []
        self.process_count += 1

        # 1. Establish broad early sensitivity; no adult phoneme inventory is assumed.
        features = extract_acoustic_features(event)
        broad = establish_broad_sensitivity(features, active_state)
        bio = biological.engage(
            "auditory_perception", "maintain_broad_auditory_sensitivity",
            broad["broad_sensitivity"] * float(features["frame_count"] > 0),
        )
        _record(trace, 0, broad, bio)

        # 2. Extract time-varying features while preserving frame order and uncertainty.
        acoustic = {
            **features,
            "sample_rate": event.sample_rate,
            "time_window_scope": "overlapping_acoustic_frames",
            "raw_audio_decoding_limit": "feature extraction only; not full speech recognition",
        }
        bio = biological.engage(
            "auditory_feature_analysis", "analyze_speech_over_time",
            float(features["frame_count"] > 0) * (1.0 - float(features["uncertainty"])),
        )
        _record(trace, 1, acoustic, bio)

        # 3. Build provisional categories from recurring sound examples.
        phonetic = build_provisional_phonetic_categories(features, event, active_state, config)
        bio = biological.engage(
            "superior_temporal_speech", "update_provisional_sound_categories",
            phonetic["category_confidence"],
        )
        _record(trace, 2, phonetic, bio)

        # 4. Accumulate syllable frequencies and transition statistics.
        statistics = accumulate_statistical_regularities(event, active_state, config)
        bio = biological.engage(
            "statistical_pattern_learning", "accumulate_environmental_regularities",
            clamp(float(statistics["observations"]) / 20.0),
        )
        _record(trace, 3, statistics, bio)

        # 5. Infer candidate word boundaries from transitions and multiple cues.
        segmentation = infer_word_boundaries(event, statistics, active_state, config)
        bio = biological.engage(
            "superior_temporal_speech", "segment_candidate_word_forms",
            segmentation["boundary_confidence"],
        )
        _record(trace, 4, segmentation, bio)

        # 6. Tune language-environment contrast weights without deleting unfamiliar distinctions.
        tuning = tune_environment_specific_perception(event, active_state, config)
        tuning["broad_sensitivity_after_tuning"] = active_state.broad_sensitivity
        bio = biological.engage(
            "auditory_perception", "tune_language_environment_contrasts",
            float(tuning["exposure_count"]) / (float(tuning["exposure_count"]) + 5.0),
        )
        _record(trace, 5, tuning, bio)

        # 7. Integrate auditory, visual, gaze, gesture, and contingent social evidence.
        multimodal = integrate_multimodal_referents(event, config)
        bio = biological.engage(
            "multisensory_social_learning", "integrate_social_multimodal_cues",
            multimodal["confidence"],
        )
        _record(trace, 6, multimodal, bio)

        # 8. Map candidate word forms onto uncertain scene/concept meanings.
        mapping = map_word_forms_to_meanings(
            event, segmentation, multimodal, active_state, config
        )
        bio = biological.engage(
            "temporal_semantic_learning", "map_word_forms_to_candidate_meanings",
            mapping["meaning_confidence"],
        )
        _record(trace, 7, mapping, bio)

        # 9. Use prior vocabulary, concepts, syntax, and support to accelerate later word learning.
        vocabulary = accelerate_vocabulary_learning(event, mapping, active_state, config)
        bio = biological.engage(
            "temporal_semantic_learning", "update_developing_vocabulary",
            vocabulary.get("learning_efficiency", 0.0),
        )
        _record(trace, 8, vocabulary, bio)

        # 10. Build grounded concepts, feature regularities, hierarchies, and relations.
        concepts = build_grounded_concepts(event, mapping, active_state, config)
        bio = biological.engage(
            "distributed_conceptual_semantic", "form_grounded_hierarchical_concepts",
            clamp(float(concepts["concept_count"]) / 20.0),
        )
        _record(trace, 9, concepts, bio)

        # 11. Bind the specific event and consolidate repeated word-meaning associations.
        memory = stabilize_developmental_memory(
            event, mapping, concepts, active_state, config
        )
        bio = biological.engage(
            "hippocampal_episodic_memory", "encode_retrieve_and_consolidate_experience",
            clamp(float(memory["episodic_count"]) / config.maximum_episodes),
        )
        _record(trace, 10, memory, bio)

        # 12. Maintain active information with decay, distraction, capacity, and chunking.
        working_memory = maintain_working_memory(event, active_state, config)
        bio = biological.engage(
            "frontoparietal_working_memory", "maintain_update_and_select_active_items",
            clamp(float(working_memory["load"]) / max(1.0, working_memory["effective_capacity"])),
        )
        _record(trace, 11, working_memory, bio)

        # 13. Maintain goals, inhibit conflict, flexibly shift rules, and use error feedback.
        executive = update_executive_control(event, active_state, working_memory, config)
        bio = biological.engage(
            "prefrontal_executive_control", "maintain_goals_inhibit_and_shift_rules",
            executive["control_demand"],
        )
        _record(trace, 12, executive, bio)

        # 14. Refine coactive connections and model myelination-related timing changes.
        circuit = refine_developing_circuits(event, active_state, config)
        bio_plasticity = biological.engage(
            "activity_dependent_circuit_plasticity", "refine_activity_dependent_connections",
            clamp(len(circuit["connection_updates"]) / 10.0),
        )
        bio_myelin = biological.engage(
            "white_matter_myelination", "update_pathway_timing_proxy",
            clamp(len(circuit["myelination_updates"]) / 10.0),
        )
        _record(trace, 13, circuit, {
            "synaptic_plasticity_network": bio_plasticity,
            "white_matter_network": bio_myelin,
        })

        # 15. Coordinate retrieved episodic knowledge with prefrontal goal control.
        integration = integrate_hippocampal_prefrontal_control(
            event, active_state, executive, config
        )
        bio = biological.engage(
            "hippocampal_prefrontal_integration", "integrate_memory_and_control",
            integration["integrated_value"],
        )
        _record(trace, 14, integration, bio)

        # 16. Maintain a dependency graph of completed and pending multi-step operations.
        plan = refine_multi_step_plan(event, active_state)
        plan_load = len(plan["completed_steps"]) / max(
            1, len(plan["completed_steps"]) + len(plan["pending_steps"])
        )
        bio = biological.engage(
            "frontoparietal_multistep_planning", "maintain_and_update_multistep_plan",
            plan_load,
        )
        _record(trace, 15, plan, bio)

        # 17. Calibrate separate developmental domains from observed success and error.
        calibration = calibrate_developmental_support(event, active_state, config)
        bio = biological.engage(
            "developmental_calibration", "adapt_future_learning_support",
            clamp(len(calibration["domain_updates"]) / 8.0),
        )
        _record(trace, 16, calibration, bio)

        active_state.experience_count += 1
        return DevelopmentalProcessingResult(
            experience_id=event.experience_id,
            broad_perception=broad, acoustic_features=acoustic,
            phonetic_categories=phonetic, statistical_learning=statistics,
            word_segmentation=segmentation, perceptual_tuning=tuning,
            multimodal_grounding=multimodal, word_meaning_mapping=mapping,
            vocabulary_growth=vocabulary, concept_formation=concepts,
            memory_stabilization=memory, working_memory=working_memory,
            executive_control=executive, circuit_refinement=circuit,
            memory_control_integration=integration, multi_step_reasoning=plan,
            developmental_calibration=calibration, operation_trace=tuple(trace),
            biological=biological.snapshot(), final_state=active_state.to_dict(),
        )


def register_annotations(
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index all Point 16 classes and callable annotations in the ORE SQLite catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root else Path(__file__).resolve().parents[1]
    )
    return sync_repository(
        LoggerDatabase(database_path), root,
        include_paths={"ore_developmental_language_cognition"},
    )


def record_result_to_sqlite(
    result: DevelopmentalProcessingResult,
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist all 17 developmental operation records and sync package annotations in SQLite."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root else Path(__file__).resolve().parents[1]
    )
    database = LoggerDatabase(database_path)
    catalogue = sync_repository(
        database, root, include_paths={"ore_developmental_language_cognition"}
    )
    run_id = database.create_run(
        script_path="ore_developmental_language_cognition/point16",
        module_name="ore_developmental_language_cognition.pipeline",
        entry_kind="developmental_language_cognition", working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id, sequence_no=sequence, event_kind="operation",
                source_path="ore_developmental_language_cognition/pipeline.py", elapsed_ms=0.0,
                metadata={
                    "operation_id": item.get("operation_id"),
                    "operation": item.get("operation"),
                    "status": item.get("status"),
                    "evidence_level": item.get("evidence_level"),
                    "mathematical_form": item.get("mathematical_form"),
                    "functional_output": item.get("functional_output"),
                    "biological_output": item.get("biological_output"),
                },
            )
        database.finish_run(
            run_id, "completed",
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
    except Exception as exc:
        database.finish_run(
            run_id, "failed",
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            error_summary=type(exc).__name__ + ": " + str(exc),
        )
        raise
    return {
        "run_id": run_id, "annotation_catalogue": catalogue,
        "database_summary": database.summary(),
    }
