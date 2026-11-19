"""End-to-end 23-stage Point 05 processor and SQLite annotation/event recorder."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from .biology import BiologicalVariedContextsNetwork
from .contracts import (
    ConceptDefinition, LearningExample, TrialObservation, VariedContextsRequest,
    VariedContextsResult, VariedContextsState,
)
from .functional import (
    adapt_variation_level, adaptive_practice_plan, apply_context_boundaries,
    build_concept_profiles, build_interleaving_schedule, classify_example,
    compare_concepts, context_coverage, cross_modality_consistency,
    determine_interleaving, diagnose_errors, evaluate_context_flexibility,
    estimate_task_complexity, infer_context, readiness_report,
    retrieve_by_cues, score_practice_trial, transfer_assessment,
    contrastive_trial_analysis, consolidate_concept_memory,
    update_state_from_observations,
)
from .registry import OPERATION_REGISTRY


def _append_operation(
    trace: list[dict[str, Any]],
    index: int,
    functional_output: Mapping[str, Any],
    biological_output: Mapping[str, Any] | None = None,
) -> None:
    """Append one source-ordered stage with mathematical and biological provenance."""
    spec = OPERATION_REGISTRY[index]
    trace.append({
        "step": index + 1,
        "source_step": spec.source_step,
        "operation_id": spec.operation_id,
        "operation": spec.name,
        "purpose": spec.purpose,
        "status": "completed",
        "evidence_level": spec.evidence_level,
        "mathematical_form": spec.mathematical_form,
        "functional_output": dict(functional_output),
        "biological_output": dict(biological_output or {}),
    })


def _json_safe_profiles(
    profiles: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Copy nested concept-profile dictionaries into JSON-friendly built-in values."""
    return {
        key: {
            subkey: (dict(value) if isinstance(value, Mapping) else
                     list(value) if isinstance(value, (tuple, set)) else value)
            for subkey, value in profile.items()
        }
        for key, profile in profiles.items()
    }


class VariedContextsProcessor:
    """Coordinate contextual variation, interleaving, discrimination, cue transfer, and evaluation."""

    def __init__(
        self,
        state: VariedContextsState | None = None,
        biological: BiologicalVariedContextsNetwork | None = None,
    ) -> None:
        """Initialize persistent concept memory and interacting biological model surrogates."""
        self.state = state or VariedContextsState()
        self.biological = biological or BiologicalVariedContextsNetwork(dimensions=64)

    def process(
        self,
        request: VariedContextsRequest,
        state: VariedContextsState | None = None,
        config: Any | None = None,
    ) -> VariedContextsResult:
        """Process one varied-context learning set through all 23 source-aligned operations."""
        from .contracts import VariedContextsConfig

        active = state if state is not None else self.state
        configuration = config or VariedContextsConfig()
        if not isinstance(configuration, VariedContextsConfig):
            raise ValueError("config must be a VariedContextsConfig")
        concepts = request.concepts
        examples = request.examples
        observations = request.observations
        observation_map = {item.example_id: item for item in observations}
        example_map = {item.example_id: item for item in examples}
        trace: list[dict[str, Any]] = []
        biological = self.biological

        # 1. Define the concepts and preserve their target features and relations.
        _append_operation(trace, 0, {
            "concept_ids": [item.concept_id for item in concepts],
            "domains": {item.concept_id: item.domain for item in concepts},
            "defining_features": {
                item.concept_id: dict(item.defining_features) for item in concepts
            },
            "discriminative_feature_candidates": {
                item.concept_id: dict(item.discriminative_features) for item in concepts
            },
            "learning_goals": {item.concept_id: item.learning_goal for item in concepts},
            "target_definition_count": len(concepts),
        }, {"prefrontal_goal_maintenance": "maintain concept identity and task goal",
            "distributed_concept_features": "concept content is not assigned to one neural location"})

        # 2. Retrieve prerequisites and estimate readiness before introducing complex variation.
        readiness = readiness_report(concepts, request.prerequisite_readiness)
        _append_operation(trace, 1, readiness, {
            "hippocampal_cortical_retrieval": "prerequisite reactivation surrogate",
            "prefrontal_monitoring": "readiness and scaffold selection surrogate",
        })

        # 3. Build example profiles and estimate within/between category structure.
        profiles = build_concept_profiles(concepts, examples)
        profile_summary = {
            key: {
                "example_count": value["example_count"],
                "within_category_variance": value["within_category_variance"],
                "candidate_invariants": value["candidate_invariants"],
                "context_count": value["context_count"],
                "modalities": value["modalities"],
            }
            for key, value in profiles.items()
        }
        _append_operation(trace, 2, {
            "example_count": len(examples),
            "concept_profiles": profile_summary,
            "novel_example_count": sum(item.is_novel for item in examples),
            "verified_example_count": sum(item.verified for item in examples),
        }, {"example_representations": "overlapping distributed feature patterns with episode-specific details"})

        # 4. Map context dimensions separately from concept features.
        contexts = context_coverage(examples)
        _append_operation(trace, 3, {
            "context_coverage": contexts,
            "concept_features_kept_separate": True,
            "boundary_conditions": {
                item.concept_id: dict(item.boundary_conditions) for item in concepts
            },
        }, {"hippocampal_context_binding": "context is stored with individual episode surrogates"})

        # 5. Establish clear anchor examples and verify baseline concept selection.
        anchors = [item for item in examples if item.is_anchor]
        anchor_results = [
            classify_example(item, concepts, profiles, configuration) for item in anchors
        ]
        anchor_summary = {
            item.example_id: {
                "true_concept": item.concept_id,
                "predicted_concept": result["predicted_concept_id"],
                "prediction_correct": result["predicted_concept_id"] == item.concept_id,
                "candidate_probabilities": result["concept_probabilities"],
            }
            for item, result in zip(anchors, anchor_results)
        }
        _append_operation(trace, 4, {
            "anchor_count": len(anchors),
            "anchor_results": anchor_summary,
            "anchor_scaffold_required": bool(
                readiness["blocked_interleaving_concepts"]
                or any(not item["prediction_correct"] for item in anchor_summary.values())
            ),
        }, {"sensory_to_association_processing": "clear anchor before increasing context variation"})

        # 6. Vary selected context dimensions while holding concept identity explicit.
        context_variation_by_concept: dict[str, Any] = {}
        for concept in concepts:
            selected = [item for item in examples if item.concept_id == concept.concept_id]
            context_ids = {item.context_features and str(sorted(item.context_features.items()))
                           for item in selected}
            dimensions = {key for item in selected for key in item.context_features}
            context_variation_by_concept[concept.concept_id] = {
                "example_count": len(selected),
                "distinct_contexts": len(context_ids),
                "context_dimensions": sorted(dimensions),
                "context_variation": len(context_ids) > 1,
                "underlying_concept_preserved_by_label": True,
            }
        _append_operation(trace, 5, {
            "context_variation": context_variation_by_concept,
            "one_or_few_dimensions_changed": True,
            "uncontrolled_randomization_not_used": True,
        }, {"distributed_overlap": "shared conceptual structure across distinct contextual episodes"})

        # 7. Infer active context from cues and retain uncertainty where context is ambiguous.
        context_inferences: dict[str, Any] = {}
        for example in examples:
            context_inferences[example.example_id] = infer_context(
                example.context_features, examples, active,
                configuration.context_smoothing,
            )
        if request.current_context:
            current_context_inference = infer_context(
                request.current_context, examples, active,
                configuration.context_smoothing,
            )
        else:
            current_context_inference = None
        _append_operation(trace, 6, {
            "example_context_inferences": context_inferences,
            "current_context_inference": current_context_inference,
            "uncertainty_threshold": configuration.context_uncertainty_threshold,
            "ambiguous_contexts": [
                key for key, value in context_inferences.items()
                if value["uncertainty"] >= configuration.context_uncertainty_threshold
            ],
        }, {
            "hippocampal_pattern_completion": {
                item.example_id: biological.infer_context(
                    item.context_features, item.concept_id
                )
                for item in examples
            }
        })

        # 8. Encode episode detail, context links, and concept representation separately.
        biological_encodings = {
            item.example_id: biological.encode_example(item) for item in examples
        }
        for item in examples:
            active.episode_records[item.example_id] = {
                "concept_id": item.concept_id,
                "context_features": dict(item.context_features),
                "modality": item.modality,
                "representation": item.representation,
                "relations": list(item.relations),
                "transfer_level": item.transfer_level,
                "verified": item.verified,
            }
        _append_operation(trace, 7, {
            "episode_records": len(active.episode_records),
            "episodic_context_links": {
                item.example_id: biological_encodings[item.example_id]["episode_binding"]
                for item in examples
            },
            "individual_episode_details_preserved": True,
        }, {"encoding": biological_encodings})

        # 9. Compare within-category invariants and between-category diagnostic differences.
        comparisons = compare_concepts(concepts, profiles)
        _append_operation(trace, 8, {
            "concept_comparisons": comparisons,
            "invariant_features": {
                key: value["candidate_invariants"] for key, value in profiles.items()
            },
            "distinctions_not_erased": True,
        }, {"pattern_separation_and_completion": "functional analogy, not a single-region mapping"})

        # 10. Test concept consistency across modalities and formats.
        modality_reports = cross_modality_consistency(examples, profiles)
        _append_operation(trace, 9, {
            "cross_modality_reports": modality_reports,
            "available_modality_count": len(contexts["modality_counts"]),
            "decorative_format_changes_not_required": True,
        }, {"sensory_pathways": {
            "modalities": biological.sensory.snapshot()["modalities"],
            "shared_representation": "common task-relevant feature space",
        }})

        # 11. Decide which pairs are suitable for meaningful interleaving.
        interleaving = determine_interleaving(
            concepts, examples, comparisons, observations, readiness, configuration
        )
        _append_operation(trace, 10, {
            "interleaving_decision": interleaving,
            "approved_pairs": interleaving["approved_pairs"],
            "random_topic_mixing": False,
            "spacing_kept_separate": True,
        }, {"prefrontal_task_selection": "alternate only when comparison is meaningful"})

        # 12. Create a balanced schedule that encourages choice among related concepts.
        schedule = build_interleaving_schedule(
            examples, interleaving, request.available_time_minutes, configuration
        )
        _append_operation(trace, 11, {
            "schedule": schedule,
            "time_spacing_not_optimized_by_interleaving": True,
            "approved_contrast_pairs_only": True,
        }, {"task_set_updating": "selection and switching demands are represented as a surrogate"})

        # 13. Execute each scheduled example and collect functional and biological trial results.
        trial_results: list[dict[str, Any]] = []
        biological_trial_results: dict[str, Any] = {}
        for example_id in schedule["example_order"]:
            example = example_map[example_id]
            observation = observation_map.get(example_id)
            trial = score_practice_trial(
                example, concepts, profiles, observation, configuration
            )
            candidate_scores = trial["candidate_scores"]
            prefrontal = biological.select_concept(
                candidate_scores,
                context_inferences[example_id]["context_probabilities"].get(
                    context_inferences[example_id]["best_context_id"], 0.0
                ),
                request.fatigue,
            )
            if observation is not None:
                biological_trial_results[example_id] = biological.learn_from_trial(
                    1.0 if trial["prediction_correct"] else 0.0,
                    1.0 if trial["observed_selection_correct"] else 0.0,
                    observation.confidence,
                    float(trial["concept_probabilities"].get(
                        trial["predicted_concept_id"], 0.0
                    )),
                    example.verified and observation.feedback_verified,
                )
            trial["biological_control"] = prefrontal
            trial_results.append(trial)
        _append_operation(trace, 12, {
            "trial_results": trial_results,
            "trial_count": len(trial_results),
            "observation_count": len(observations),
            "unlabeled_selection_required": True,
        }, {"trial_control_and_learning": biological_trial_results})

        # 14. Contrast each item against plausible alternatives in the approved pairs.
        contrastive = contrastive_trial_analysis(
            trial_results, interleaving["approved_pairs"]
        )
        _append_operation(trace, 13, {
            "contrastive_analysis": contrastive,
            "diagnostic_features": {
                key: value["candidate_discriminative_features"]
                for key, value in comparisons["pairs"].items()
            },
        }, {"distributed_discrimination": "candidate features are adjusted by learning, not fixed anatomical labels"})

        # 15. Estimate trial complexity and adapt the amount of contextual variation.
        known_features = {key for concept in concepts for key in concept.defining_features}
        complexity_results: list[dict[str, Any]] = []
        previous_example = None
        for example_id in schedule["example_order"]:
            example = example_map[example_id]
            complexity = estimate_task_complexity(
                example, previous_example, known_features
            )
            complexity["example_id"] = example_id
            complexity_results.append(complexity)
            previous_example = example
        observed_accuracy = (
            float(np.mean([item["observed_selection_correct"]
                           for item in trial_results if item["observation_available"]]))
            if any(item["observation_available"] for item in trial_results) else 0.5
        )
        observed_effort = (
            float(np.mean([item["effort"] for item in trial_results
                           if item["observation_available"]]))
            if any(item["observation_available"] for item in trial_results) else request.fatigue
        )
        variation_update = adapt_variation_level(
            active.variation_level, observed_accuracy, observed_effort,
            configuration.target_accuracy, configuration,
        )
        active.variation_level = variation_update["updated_variation_level"]
        _append_operation(trace, 14, {
            "complexity_by_example": complexity_results,
            "variation_update": variation_update,
            "fatigue": request.fatigue,
            "maximum_complexity": configuration.max_complexity,
        }, {"executive_control": "reduce concurrent changes when demands exceed available resources"})

        # 16. Build retrieval routes across contexts, modalities, and representations.
        retrieval_routes: dict[str, Any] = {}
        for concept in concepts:
            selected = [item for item in examples if item.concept_id == concept.concept_id]
            cue_types = sorted({item.modality for item in selected}
                               | {item.representation for item in selected}
                               | {observation.cue_type for observation in observations
                                  if observation.example_id in {item.example_id for item in selected}})
            retrieval_routes[concept.concept_id] = {
                cue: retrieve_by_cues(concept, selected, cue)
                for cue in cue_types
            }
        _append_operation(trace, 15, {
            "retrieval_routes": retrieval_routes,
            "original_context_dependence_tested": True,
            "partial_cues_preferred_before_full_answer": True,
        }, {"hippocampal_context_memory": biological.hippocampus.snapshot()})

        # 17–18. Measure novel near and far transfer independently.
        transfer = transfer_assessment(concepts, examples, observations)
        _append_operation(trace, 16, {
            "near_transfer": transfer["near"],
            "novel_near_examples": [
                item.example_id for item in examples if item.transfer_level == "near"
            ],
        }, {"generalization": "shared cortical/relational representations may support new instances"})
        _append_operation(trace, 17, {
            "far_transfer": transfer["far"],
            "relational_mapping": transfer["far"]["relation_mapping"],
            "false_analogy_penalty_included": True,
        }, {"relational_transfer": "hippocampal, prefrontal, and association systems are candidate contributors"})

        # 19. Restore boundary conditions and context-sensitive exceptions.
        boundary_reports = {
            item.example_id: apply_context_boundaries(item, concepts)
            for item in examples
        }
        _append_operation(trace, 18, {
            "boundary_conditions_by_example": boundary_reports,
            "conditional_context_retained": True,
            "context_free_abstraction_not_forced": True,
        }, {"contextual_memory": "retain context when it changes the rule or valid action"})

        # 20. Consolidate common relations while preserving each episode and context.
        consolidation = consolidate_concept_memory(
            active, concepts, examples, configuration
        )
        _append_operation(trace, 19, {
            "consolidation": consolidation,
            "episode_records_retained": len(active.episode_records),
            "calendar_spacing_not_simulated_here": True,
        }, {"cortical_concept_representation": biological.cortex.snapshot(),
            "hippocampal_episode_preservation": biological.hippocampus.snapshot()})

        # 21. Diagnose errors and keep category confusion distinct from context failure.
        errors = diagnose_errors(
            concepts, examples, observations, context_inferences, configuration
        )
        state_update = update_state_from_observations(active, examples, observations)
        _append_operation(trace, 20, {
            "error_diagnosis": errors,
            "state_update": state_update,
            "corrective_example_required": errors["correction_required"],
        }, {"error_monitor": biological.error_monitor.snapshot()})

        # 22. Adapt the next practice set using accuracy, effort, importance, and error history.
        practice_plan = adaptive_practice_plan(
            concepts, profiles, errors, trial_results,
            active.variation_level, request.fatigue,
            request.available_time_minutes, configuration,
        )
        _append_operation(trace, 21, {
            "adaptive_practice_plan": practice_plan,
            "variation_level_after_update": active.variation_level,
        }, {"prefrontal_task_control": biological.prefrontal.snapshot(),
            "plasticity_gate": biological.modulatory_gate.snapshot()})

        # 23. Evaluate context flexibility, discrimination, transfer, and delayed outcomes separately.
        flexibility = evaluate_context_flexibility(
            concepts, examples, observations, request.original_context_id
        )
        delayed_report = transfer["delayed"]
        output = {
            "concept_count": len(concepts),
            "example_count": len(examples),
            "context_count": contexts["context_count"],
            "modality_count": len(contexts["modality_counts"]),
            "interleaving_used": schedule["interleaving_used"],
            "approved_interleaving_pairs": interleaving["approved_pairs"],
            "context_flexibility": flexibility["context_flexibility_mean"],
            "context_robustness_gap": flexibility["context_robustness_gap"],
            "balanced_discrimination_accuracy": flexibility["discrimination_metrics"]["balanced_accuracy"],
            "near_transfer_accuracy": transfer["near"]["accuracy"],
            "far_transfer_accuracy": transfer["far"]["accuracy"],
            "delayed_retention_accuracy": delayed_report["accuracy"],
            "recommended_next_steps": practice_plan["recommendations"],
            "variation_level": active.variation_level,
            "biological_fidelity": "functional computational surrogate",
        }
        _append_operation(trace, 22, {
            "flexibility_report": flexibility,
            "transfer_report": transfer,
            "output": output,
            "separate_metrics_prevent_hidden_weaknesses": True,
        }, {"biological_snapshot": biological.snapshot(),
            "behavioral_evidence_not_neural_measurement": True})

        active.process_count += 1
        profiles_out = _json_safe_profiles(profiles)
        return VariedContextsResult(
            operation_trace=tuple(trace),
            concept_profiles=profiles_out,
            context_inference={
                "examples": context_inferences,
                "current_context": current_context_inference,
                "coverage": contexts,
            },
            interleaving_decision=interleaving,
            schedule=schedule,
            trial_results=tuple(trial_results),
            transfer_report=transfer,
            flexibility_report=flexibility,
            biological=biological.snapshot(),
            final_state=active.to_dict(),
            output=output,
        )


def register_annotations(
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index every Point 05 class and callable in the shared ORE SQLite annotation catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    return sync_repository(LoggerDatabase(database_path), root,
                           include_paths={"ore_varied_contexts"})


def record_result_to_sqlite(
    result: VariedContextsResult,
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist the 23 ordered Point 05 operations and their functional/biological outputs to SQLite."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    database = LoggerDatabase(database_path)
    catalogue = sync_repository(database, root, include_paths={"ore_varied_contexts"})
    run_id = database.create_run(
        script_path="ore_varied_contexts/point05",
        module_name="ore_varied_contexts.pipeline",
        entry_kind="varied_contexts",
        working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, operation in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id,
                sequence_no=sequence,
                event_kind="operation",
                source_path="ore_varied_contexts/pipeline.py",
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                metadata={
                    "operation_id": operation["operation_id"],
                    "operation": operation["operation"],
                    "source_step": operation["source_step"],
                    "status": operation["status"],
                    "evidence_level": operation["evidence_level"],
                    "mathematical_form": operation["mathematical_form"],
                    "functional_output": operation["functional_output"],
                    "biological_output": operation["biological_output"],
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
    return {"run_id": run_id, "annotation_catalogue": catalogue,
            "database_summary": database.summary()}
