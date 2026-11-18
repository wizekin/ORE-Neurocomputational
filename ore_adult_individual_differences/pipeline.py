"""End-to-end 15-stage adult individual-differences comprehension processor."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

from .biology import AdultComprehensionBiologicalNetwork
from .cognition import (
    adapt_processing_support, calibrate_decision_and_confidence,
    estimate_interactivity_and_load, estimate_processing_schedule,
    monitor_interference_and_hypotheses, retrieve_gist_detail_source_memory,
)
from .contracts import (
    AdultComprehensionExperience, AdultComprehensionResult,
    AdultDifferencesConfig, AdultLearnerState,
)
from .functional import (
    establish_task_goal, estimate_current_resources, segment_input,
)
from .knowledge import (
    convert_familiar_details_to_chunks, incorporate_cultural_and_lived_experience,
    integrate_knowledge_with_provenance, retrieve_schema_candidates,
    update_knowledge_without_false_assimilation, weight_language_cues,
)
from .registry import OPERATION_REGISTRY


def _record(
    trace: list[dict[str, Any]],
    index: int,
    functional_output: Mapping[str, Any],
    biological_output: Mapping[str, Any],
) -> None:
    """Append one ordered operation with mathematical form and functional/biological outputs."""
    spec = OPERATION_REGISTRY[index]
    trace.append({
        "step": index + 1,
        "operation_id": spec.operation_id,
        "operation": spec.name,
        "status": "completed",
        "purpose": spec.purpose,
        "evidence_level": spec.evidence_level,
        "mathematical_form": spec.mathematical_form,
        "functional_output": dict(functional_output),
        "biological_output": dict(biological_output),
    })


class AdultIndividualDifferencesProcessor:
    """Coordinate the 15 stages while keeping learner differences domain- and task-specific."""

    def __init__(
        self,
        config: AdultDifferencesConfig | None = None,
        state: AdultLearnerState | None = None,
        biological: AdultComprehensionBiologicalNetwork | None = None,
    ) -> None:
        """Initialize processing parameters, a persistent adult learner profile, and network surrogates."""
        self.config = config or AdultDifferencesConfig()
        self.state = state or AdultLearnerState()
        self.biological = biological or AdultComprehensionBiologicalNetwork()
        self.process_count = 0

    def process(
        self,
        event: AdultComprehensionExperience,
        state: AdultLearnerState | None = None,
    ) -> AdultComprehensionResult:
        """Execute all 15 stages and return detailed trace, separate evidence, and updated learner state."""
        learner = state if state is not None else self.state
        config = self.config
        bio = self.biological
        trace: list[dict[str, Any]] = []
        self.process_count += 1

        # 1. Detect the task and establish explicit goal priorities.
        goal = establish_task_goal(event)
        activity = bio.engage(
            "goal_attention_control", "maintain_task_goal",
            max(goal["objective_vector"].values()),
        )
        _record(trace, 0, goal, activity)

        # 2. Estimate current attention and maintenance support from task conditions.
        resources = estimate_current_resources(event, learner, config)
        attention_activity = bio.engage(
            "goal_attention_control", "allocate_attention_to_current_task",
            resources["attention_control"],
        )
        wm_activity = bio.engage(
            "frontoparietal_working_memory", "estimate_available_maintenance_resources",
            resources["working_memory_availability"],
        )
        _record(trace, 1, resources, {
            "attention_network": attention_activity,
            "working_memory_network": wm_activity,
        })

        # 3. Segment the stream into ordered units while preserving source references.
        segmented = segment_input(event, config)
        activity = bio.engage(
            "temporal_processing_coordination", "segment_ordered_input",
            min(1.0, segmented["unit_count"] / 12.0),
        )
        _record(trace, 2, segmented, activity)

        # 4. Retrieve domain-relevant semantic schemas and prior knowledge.
        retrieved = retrieve_schema_candidates(event, segmented, goal, learner, config)
        activity = bio.engage(
            "cortical_semantic_knowledge", "retrieve_relevant_prior_knowledge",
            float(retrieved["selected_schema"]["probability"])
            if retrieved["selected_schema"] else 0.0,
        )
        _record(trace, 3, retrieved, activity)

        # 5. Convert familiar details into chunks only when schema fit and cost warrant it.
        chunking = convert_familiar_details_to_chunks(segmented, retrieved, config)
        activity = bio.engage(
            "expertise_schema_chunking", "group_familiar_schema_elements",
            min(1.0, chunking["cost_reduction"] / max(1.0, chunking["separate_representation_cost"])),
        )
        _record(trace, 4, chunking, activity)

        # 6. Estimate interactivity and dynamic working-memory load.
        load = estimate_interactivity_and_load(segmented, chunking, event, resources, config)
        activity = bio.engage(
            "frontoparietal_working_memory", "estimate_element_interactivity_and_load",
            min(1.0, load["load_pressure"] / 5.0),
        )
        _record(trace, 5, load, activity)

        # 7. Estimate processing durations for each component and temporal availability.
        schedule = estimate_processing_schedule(event, learner, resources, load, config)
        activity = bio.engage(
            "temporal_processing_coordination", "coordinate_component_processing_times",
            min(1.0, schedule["temporal_pressure"]),
        )
        _record(trace, 6, schedule, activity)

        # 8. Weight language cues through the active language's experience profile.
        language = weight_language_cues(event, learner)
        activity = bio.engage(
            "distributed_language_network", "weight_language_cues_by_proficiency_and_use",
            language["proficiency_estimate"],
        )
        _record(trace, 7, language, activity)

        # 9. Incorporate contextual conventions only from relevant individual experience.
        culture = incorporate_cultural_and_lived_experience(event, learner)
        activity = bio.engage(
            "contextual_social_semantics", "activate_individual_contextual_knowledge",
            min(1.0, len(culture["context_hypotheses"]) / 5.0),
        )
        _record(trace, 8, culture, activity)

        # 10. Integrate current content with schemas while retaining provenance boundaries.
        integrated = integrate_knowledge_with_provenance(
            segmented, retrieved, language, culture
        )
        binding_activity = bio.engage(
            "hippocampal_memory_binding", "bind_new_input_to_active_knowledge",
            event.provenance_completeness,
        )
        semantic_activity = bio.engage(
            "cortical_semantic_knowledge", "integrate_observed_and_retrieved_content",
            float(bool(retrieved["ranked_schemas"])),
        )
        _record(trace, 9, integrated, {
            "memory_binding_network": binding_activity,
            "semantic_network": semantic_activity,
        })

        # 11. Monitor interference and keep competing hypotheses when the error cost is high.
        monitor = monitor_interference_and_hypotheses(
            integrated, event, resources, load, config
        )
        activity = bio.engage(
            "hypothesis_conflict_monitor", "monitor_interference_and_candidate_competition",
            max(monitor["conflict"], monitor["interference"]),
        )
        wm_activity = bio.engage(
            "frontoparietal_working_memory", "maintain_live_interpretation_candidates",
            min(1.0, len(monitor["live_hypotheses"]) / config.max_live_hypotheses),
        )
        _record(trace, 10, monitor, {
            "conflict_monitor_network": activity,
            "working_memory_network": wm_activity,
        })

        # 12. Adapt support to domain expertise, current load, and recent performance.
        support = adapt_processing_support(
            event, learner, resources, load, retrieved, monitor, config
        )
        activity = bio.engage(
            "adaptive_support_control", "select_person_task_matched_processing_support",
            support["support_level"],
        )
        _record(trace, 11, support, activity)

        # 13. Estimate gist, episodic detail, and source retrieval independently.
        aging = retrieve_gist_detail_source_memory(event)
        memory_activity = bio.engage(
            "gist_detail_source_memory", "estimate_gist_detail_and_source_retrieval",
            aging["task_weighted_memory_value"],
        )
        hippocampal_activity = bio.engage(
            "hippocampal_memory_binding", "bind_distinctive_episode_and_source_cues",
            aging["source_context_probability"],
        )
        _record(trace, 12, aging, {
            "gist_detail_network": memory_activity,
            "episodic_source_network": hippocampal_activity,
        })

        # 14. Update learned representations without assimilating contradictions silently.
        update = update_knowledge_without_false_assimilation(event, learner, config)
        update_activity = bio.engage(
            "prediction_error_knowledge_update", "update_knowledge_from_reliable_evidence",
            min(1.0, len(update["updates"]) / 5.0),
        )
        semantic_activity = bio.engage(
            "cortical_semantic_knowledge", "revise_semantic_estimates_and_exceptions",
            min(1.0, len(update["exceptions"]) + len(update["contradictions"])) / 5.0,
        )
        _record(trace, 13, update, {
            "prediction_error_network": update_activity,
            "semantic_update_network": semantic_activity,
        })

        # 15. Calibrate confidence and choose the next action.
        decision = calibrate_decision_and_confidence(
            event, integrated, monitor, retrieved, load, support, aging, config, learner
        )
        activity = bio.engage(
            "confidence_decision_monitor", "calibrate_confidence_and_select_next_action",
            decision["calibrated_confidence"],
        )
        _record(trace, 14, decision, activity)

        learner.experience_count += 1
        return AdultComprehensionResult(
            experience_id=event.experience_id,
            task_goal=goal,
            current_resources=resources,
            segmented_units=segmented,
            retrieved_knowledge=retrieved,
            chunking=chunking,
            working_memory_load=load,
            processing_schedule=schedule,
            language_interpretation=language,
            cultural_interpretation=culture,
            integrated_representation=integrated,
            interference_monitor=monitor,
            adaptive_support=support,
            aging_memory=aging,
            knowledge_update=update,
            calibrated_decision=decision,
            operation_trace=tuple(trace),
            biological=bio.snapshot(),
            final_state=learner.to_dict(),
        )


def register_annotations(
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index all Point 17 classes and callable annotations in the ORE SQLite catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root else Path(__file__).resolve().parents[1]
    )
    return sync_repository(
        LoggerDatabase(database_path), root,
        include_paths={"ore_adult_individual_differences"},
    )


def record_result_to_sqlite(
    result: AdultComprehensionResult,
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist all 15 Point 17 operation records and synchronize package annotations."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root else Path(__file__).resolve().parents[1]
    )
    database = LoggerDatabase(database_path)
    catalogue = sync_repository(
        database, root, include_paths={"ore_adult_individual_differences"}
    )
    run_id = database.create_run(
        script_path="ore_adult_individual_differences/point17",
        module_name="ore_adult_individual_differences.pipeline",
        entry_kind="adult_individual_differences", working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id,
                sequence_no=sequence,
                event_kind="operation",
                source_path="ore_adult_individual_differences/pipeline.py",
                elapsed_ms=0.0,
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
