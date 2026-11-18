"""End-to-end 15-stage Point 15 language/domain-general reasoning processor."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

from .biology import BiologicalLanguageReasoningNetwork
from .contracts import (
    LanguageEvent, LanguageLearnerState, LanguageProcessingConfig, LanguageProcessingResult,
)
from .functional import (
    activate_language_representations, apply_deliberate_reasoning,
    build_incremental_structure, compose_sentence_meaning, convert_to_language_units,
    detect_mismatch_and_revise, encode_sensory_signal, estimate_processing_demand,
    infer_social_pragmatics, integrate_interacting_systems, recruit_domain_general_control,
    retrieve_conceptual_knowledge, retrieve_lexical_knowledge, select_task_output,
    update_learned_representations,
)
from .registry import OPERATION_REGISTRY


def _trace(trace: list[dict[str, Any]], index: int, functional: Mapping[str, Any],
           biological: Mapping[str, Any] | None = None) -> None:
    """Append one source-ordered operation with formula and biological interpretation."""
    spec = OPERATION_REGISTRY[index]
    trace.append({
        "step": index + 1, "operation_id": spec.operation_id, "operation": spec.name,
        "status": "completed", "purpose": spec.purpose, "evidence_level": spec.evidence_level,
        "mathematical_form": spec.mathematical_form, "functional_output": dict(functional),
        "biological_output": dict(biological or {}),
    })


class LanguageDomainReasoningProcessor:
    """Coordinate language-selective, conceptual, control, reasoning, pragmatic, and learning stages."""

    def __init__(self, config: LanguageProcessingConfig | None = None,
                 state: LanguageLearnerState | None = None,
                 biological: BiologicalLanguageReasoningNetwork | None = None) -> None:
        """Initialize a learner's persistent language state and network surrogates."""
        self.config = config or LanguageProcessingConfig()
        self.state = state or LanguageLearnerState()
        self.biological = biological or BiologicalLanguageReasoningNetwork()
        self.process_count = 0

    def process(self, event: LanguageEvent,
                state: LanguageLearnerState | None = None) -> LanguageProcessingResult:
        """Execute all 15 operations while keeping explicit and inferred information separate."""
        active_state = state if state is not None else self.state
        self.process_count += 1
        trace: list[dict[str, Any]] = []
        bio, cfg = self.biological, self.config

        # 1. Preserve the time-ordered input and its modality-specific evidence.
        features = encode_sensory_signal(event)
        bio.engage("auditory_visual_perceptual", "preserve_signal",
                   {"load": 1.0 - features["signal_quality"]})
        _trace(trace, 0, features, bio.perceptual.snapshot())

        # 2. Convert perceptual evidence into competing language-relevant candidate units.
        units = convert_to_language_units(event, features, cfg)
        bio.engage("auditory_visual_perceptual", "extract_language_relevant_units",
                   {"load": units["mean_candidate_entropy"]})
        _trace(trace, 1, units, bio.perceptual.snapshot())

        # 3. Activate the distributed language-selective representation.
        active_language = activate_language_representations(
            features, units, {"activation": active_state.language_activation}
        )
        bio.engage("frontotemporal_language_selective", "activate_recurrent_language_state",
                   {"load": 1.0 - active_language["linguistic_form_confidence"]})
        _trace(trace, 2, active_language, bio.language_selective.snapshot())

        # 4. Retrieve lexical candidates and their stored linguistic properties.
        lexical = retrieve_lexical_knowledge(event, units, active_state, cfg)
        bio.engage("frontotemporal_language_selective", "retrieve_lexical_knowledge",
                   {"load": lexical["lexical_uncertainty_nats"] / 3.0})
        _trace(trace, 3, lexical, bio.language_selective.snapshot())

        # 5. Build an incremental structural hypothesis and keep unresolved dependencies.
        structure = build_incremental_structure(event, lexical, cfg, active_state)
        bio.engage("frontotemporal_language_selective", "construct_incremental_structure",
                   {"load": len(structure["unresolved_dependencies"]) / 5.0})
        _trace(trace, 4, structure, bio.language_selective.snapshot())

        # 6. Compose sentence meaning while retaining a direct-content record.
        composition = compose_sentence_meaning(lexical, structure, event)
        linguistic = {
            "language_state": active_language, "lexical": lexical,
            "linguistic_structure": structure,
            "sentence_meaning": composition["sentence_meaning"],
            "said_or_encoded": composition["said_or_encoded"],
            "ambiguity_preserved": composition["ambiguity_preserved"],
        }
        bio.engage("frontotemporal_language_selective", "compose_sentence_meaning",
                   {"load": 1.0 - composition["sentence_meaning"]["compositional_confidence"]})
        _trace(trace, 5, composition, bio.language_selective.snapshot())

        # 7. Retrieve relevant conceptual knowledge as a distinct information source.
        concepts = retrieve_conceptual_knowledge(event, linguistic, cfg)
        bio.engage("distributed_conceptual_semantic", "retrieve_relevant_concepts",
                   {"load": concepts["conceptual_confidence"]})
        _trace(trace, 6, concepts, bio.conceptual_semantic.snapshot())

        # 8. Estimate general cognitive demand independently from sentence content.
        demand = estimate_processing_demand(linguistic, concepts, event)
        bio.engage("multiple_demand_control", "estimate_processing_demand",
                   {"demand_score": demand["demand_score"]})
        _trace(trace, 7, demand, bio.multiple_demand_control.snapshot())

        # 9. Recruit additional attention and working-memory support when required.
        control = recruit_domain_general_control(demand, linguistic, event, cfg)
        bio.engage("multiple_demand_control", "allocate_control_and_working_memory",
                   {"control_allocation": control["control_allocation"]})
        _trace(trace, 8, control, bio.multiple_demand_control.snapshot())

        # 10. Apply explicit reasoning rules without rewriting the linguistic representation.
        reasoning = apply_deliberate_reasoning(event, linguistic, concepts, control)
        bio.engage("domain_general_reasoning", "apply_deliberate_inference",
                   {"load": float(reasoning["reasoning_invoked"])})
        _trace(trace, 9, reasoning, bio.domain_general_reasoning.snapshot())

        # 11. Infer speaker intention only when context or task makes it relevant.
        pragmatic = infer_social_pragmatics(event, linguistic, cfg)
        if pragmatic["recruited"]:
            bio.engage("social_pragmatic_cognition", "infer_speaker_intention",
                       {"load": 1.0 - pragmatic["confidence"]})
        _trace(trace, 10, pragmatic, bio.social_pragmatic.snapshot())

        # 12. Integrate outputs while preserving provenance and subsystem boundaries.
        integrated = integrate_interacting_systems(
            linguistic, concepts, demand, control, reasoning, pragmatic, event
        )
        bio.engage("cross_network_integration_monitoring", "integrate_separate_representations",
                   {"load": demand["demand_score"]})
        _trace(trace, 11, integrated, bio.integration_monitoring.snapshot())

        # 13. Detect prediction mismatch and target only the affected representation layer.
        mismatch = detect_mismatch_and_revise(event, integrated, cfg)
        if mismatch["mismatch_detected"]:
            bio.engage("cross_network_integration_monitoring", "monitor_and_revise",
                       {"load": mismatch["updated_uncertainty"]})
        _trace(trace, 12, mismatch, bio.integration_monitoring.snapshot())

        # 14. Select a task output or retain a stable comprehension representation.
        task_result = select_task_output(event, integrated, mismatch)
        bio.engage("language_production_task_output", "select_response_or_comprehension",
                   {"load": demand["demand_score"]})
        _trace(trace, 13, task_result, bio.language_production.snapshot())

        # 15. Update learned representations only from reliable repeated evidence.
        learning = update_learned_representations(event, active_state, linguistic, concepts, cfg)
        bio.engage("hippocampal_cortical_learning", "update_from_experience",
                   {"load": float(learning["eligible"])})
        _trace(trace, 14, learning, bio.hippocampal_cortical_learning.snapshot())

        active_state.episode_history[-1]["control_recruited"] = control["control_recruited"]
        active_state.episode_history[-1]["pragmatics_recruited"] = pragmatic["recruited"]
        active_state.episode_history[-1]["mismatch_detected"] = mismatch["mismatch_detected"]
        return LanguageProcessingResult(
            event_id=event.event_id, linguistic_representation=linguistic,
            conceptual_interpretation=concepts, control_request={**demand, **control},
            reasoning_result=reasoning, pragmatic_interpretation=pragmatic,
            integrated_interpretation=integrated, mismatch_and_revision=mismatch,
            task_result=task_result, learning_update=learning, operation_trace=tuple(trace),
            biological=bio.snapshot(), final_state=active_state.to_dict(),
        )


def register_annotations(database_path: str | Path,
                          repository_root: str | Path | None = None) -> dict[str, Any]:
    """Index every Point 15 class and callable annotation in the ORE SQLite catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    return sync_repository(LoggerDatabase(database_path), root,
                           include_paths={"ore_language_domain_reasoning"})


def record_result_to_sqlite(result: LanguageProcessingResult, database_path: str | Path,
                            repository_root: str | Path | None = None) -> dict[str, Any]:
    """Persist all 15 operation rows and sync package annotations to the ORE SQLite logger."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    database = LoggerDatabase(database_path)
    catalogue = sync_repository(database, root, include_paths={"ore_language_domain_reasoning"})
    run_id = database.create_run(
        script_path="ore_language_domain_reasoning/point15",
        module_name="ore_language_domain_reasoning.pipeline",
        entry_kind="language_domain_reasoning", working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id, sequence_no=sequence, event_kind="operation",
                source_path="ore_language_domain_reasoning/pipeline.py", elapsed_ms=0.0,
                metadata={
                    "operation_id": item.get("operation_id"), "operation": item.get("operation"),
                    "status": item.get("status"), "evidence_level": item.get("evidence_level"),
                    "mathematical_form": item.get("mathematical_form"),
                    "functional_output": item.get("functional_output"),
                    "biological_output": item.get("biological_output"),
                },
            )
        database.finish_run(run_id, "completed",
                            duration_ms=round((time.perf_counter() - started) * 1000.0, 3))
    except Exception as exc:
        database.finish_run(run_id, "failed",
                            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
                            error_summary=type(exc).__name__ + ": " + str(exc))
        raise
    return {"run_id": run_id, "annotation_catalogue": catalogue,
            "database_summary": database.summary()}
