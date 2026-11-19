"""End-to-end 17-stage processor for Point 01 active engagement."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .biology import BiologicalActiveEngagementNetwork
from .contracts import (
    ActiveEngagementConfig,
    ActiveEngagementRequest,
    ActiveEngagementResult,
    ActiveEngagementState,
)
from .functional import (
    allocate_attention,
    bind_episode,
    construct_imagery,
    decide_closure,
    elaborate_explanation,
    evaluate_explanation,
    extract_propositions,
    generate_candidate_answers,
    generate_questions,
    monitor_comprehension,
    prepare_teaching_output,
    reconstruct_explanation,
    repair_explanation,
    segment_material,
    select_target,
    tokenize,
    vectorize_texts,
)
from .registry import OPERATION_REGISTRY


def _append_operation(
    trace: list[dict[str, Any]],
    index: int,
    functional_output: Mapping[str, Any],
    biological_output: Mapping[str, Any] | None = None,
) -> None:
    """Append one numbered source operation with its mathematical and biological metadata."""
    spec = OPERATION_REGISTRY[index]
    trace.append({
        "step": index + 1,
        "source_step": spec.source_step,
        "operation_id": spec.operation_id,
        "operation": spec.name,
        "status": "completed",
        "explanation": spec.purpose,
        "evidence_level": spec.evidence_level,
        "mathematical_form": spec.mathematical_form,
        "functional_output": dict(functional_output),
        "biological_output": dict(biological_output or {}),
    })


class ActiveEngagementProcessor:
    """Coordinate functional comprehension operations and distributed biological surrogates."""

    def __init__(
        self,
        learner_id: str,
        config: ActiveEngagementConfig | None = None,
        state: ActiveEngagementState | None = None,
        biological: BiologicalActiveEngagementNetwork | None = None,
    ) -> None:
        """Initialize one learner's state, operation configuration, and biological network."""
        if not learner_id or not learner_id.strip():
            raise ValueError("learner_id must be non-empty")
        self.learner_id = learner_id.strip()
        self.config = config or ActiveEngagementConfig()
        self.state = state or ActiveEngagementState(learner_id=self.learner_id)
        if self.state.learner_id != self.learner_id:
            raise ValueError("processor learner_id must match ActiveEngagementState")
        self.biological = biological or BiologicalActiveEngagementNetwork(
            self.config.embedding_dimensions
        )
        if self.biological.dimensions != self.config.embedding_dimensions:
            raise ValueError("biological network dimensions must match configuration")
        self.process_count = 0

    def process(
        self,
        request: ActiveEngagementRequest,
        state: ActiveEngagementState | None = None,
    ) -> ActiveEngagementResult:
        """Run all 17 source-aligned operations and return functional and biological diagnostics."""
        if request.learner_id != self.learner_id:
            raise ValueError("request learner_id must match this processor")
        active_state = state if state is not None else self.state
        if active_state.learner_id != self.learner_id:
            raise ValueError("request learner_id must match the supplied state")
        self.process_count += 1
        trace: list[dict[str, Any]] = []
        bio = self.biological
        config = self.config
        segments = tuple(request.material)

        # 1. Establish the active episode and maintain the learner's current goal.
        goal_state = bio.prefrontal_control.maintain_goal(request.goal)
        episode_start = {
            "learner_id": self.learner_id,
            "goal": request.goal,
            "task_type": request.task_type,
            "segment_count": len(segments),
            "context": dict(request.context),
            "stopping_policy": "quality threshold plus explicit critical-gap review",
        }
        _append_operation(trace, 0, episode_start, {
            "prefrontal_goal_maintenance": goal_state,
            "network_components": list(bio.snapshot()),
        })

        # 2. Build online population representations while preserving segment order.
        encoded_segments: list[dict[str, Any]] = []
        last_source_activity = np.zeros(config.embedding_dimensions, dtype=float)
        for segment in segments:
            encoded = bio.temporal_language.encode(segment.text)
            last_source_activity = np.asarray(
                encoded["population_activity"], dtype=float
            )
            encoded_segments.append({
                "segment_id": segment.segment_id,
                "text": segment.text,
                "modality": segment.modality,
                "reliability": segment.reliability,
                "embedding": encoded["embedding"],
                "population_activity": encoded["population_activity"],
            })
        stream_representation = {
            "segments": encoded_segments,
            "sequence_order_preserved": True,
            "stream_embedding": vectorize_texts(
                [item.text for item in segments], config.embedding_dimensions
            ).tolist(),
        }
        _append_operation(trace, 1, stream_representation, {
            "temporal_language": bio.temporal_language.snapshot(),
            "online_recurrent_encoding": True,
        })

        # 3. Segment the stream and extract propositions and explicit relation markers.
        source_units = segment_material(segments)
        propositions = extract_propositions(source_units)
        segmentation = {
            "units": source_units,
            "propositions": propositions,
            "proposition_count": len(propositions),
            "uncertain_boundaries": [
                item["proposition_source_id"]
                for item in source_units
                if float(item["boundary_confidence"]) < 0.7
            ],
        }
        _append_operation(trace, 2, segmentation, {
            "temporal_language": bio.temporal_language.snapshot(),
            "temporal_grouping_surrogate": True,
        })

        # 4. Select the main target by goal relevance, local centrality, and uncertainty.
        target_selection = select_target(propositions, request.goal, config)
        target = target_selection["target"]
        _append_operation(trace, 3, target_selection, {
            "prefrontal_control": bio.prefrontal_control.snapshot(),
            "goal_bias_applied": True,
        })

        # 5. Generate prioritized questions and provisional predictions.
        questions = generate_questions(
            target, propositions, request.goal, request.max_questions, config
        )
        question_output = {
            "questions": questions,
            "selected_question": questions[0],
            "question_count": len(questions),
            "selection_method": "estimated_information_gain_times_importance_over_cost",
        }
        _append_operation(trace, 4, question_output, {
            "prefrontal_control": bio.prefrontal_control.snapshot(),
            "uncertainty_monitoring": "model-level entropy/information-value proxy",
        })

        # 6. Allocate attention to relevant source evidence under graded competition.
        attention = allocate_attention(questions[0], source_units, config)
        evidence_vectors = [
            item["embedding"] for item in encoded_segments
        ]
        source_scores = [
            float(item["relevance_score"])
            for item in attention["selected_evidence"]
        ]
        selected_ids = {
            str(item["segment_id"]) for item in attention["selected_evidence"]
        }
        selected_vectors = [
            vector for segment, vector in zip(segments, evidence_vectors)
            if segment.segment_id in selected_ids
        ]
        attention_bio = bio.frontoparietal_attention.allocate(
            selected_vectors, source_scores
        ) if selected_vectors else bio.frontoparietal_attention.allocate([], [])
        attention["biological_gain_modulation"] = attention_bio["gain_modulation"]
        _append_operation(trace, 5, attention, {
            "frontoparietal_attention": attention_bio,
            "top_down_gain_control": True,
        })

        # 7. Generate competing source-grounded answer candidates.
        candidates = generate_candidate_answers(questions[0], attention)
        _append_operation(trace, 6, {
            "candidate_answers": candidates,
            "candidate_count": len(candidates),
            "unsupported_inferences_labeled": True,
        }, {
            "temporal_language": bio.temporal_language.snapshot(),
            "hippocampal_partial_cue_retrieval": bio.hippocampal_episode.reinstate(
                np.asarray(encoded_segments[0]["embedding"], dtype=float)
            ),
        })

        # 8. Reconstruct meaning, distinguishing learner-generated text from a scaffold.
        explanation = reconstruct_explanation(
            candidates, request.learner_explanation
        )
        explanation_vector = vectorize_texts(
            [explanation["text"]], config.embedding_dimensions
        )
        _append_operation(trace, 7, explanation, {
            "distributed_language_generation": {
                "explanation_representation_energy": float(
                    np.mean(explanation_vector ** 2)
                ),
                "learner_generation_observed": bool(
                    explanation["is_learner_generated"]
                ),
            },
        })

        # 9. Elaborate source-supported relations and preserve unsupported details as gaps.
        elaboration = elaborate_explanation(explanation, propositions)
        _append_operation(trace, 8, elaboration, {
            "temporal_language": bio.temporal_language.snapshot(),
            "distributed_association_processing": True,
        })

        # 10. Evaluate fidelity, proposition/term coverage, coherence, and evidence.
        quality = evaluate_explanation(
            explanation, propositions, target, config, request.required_concepts
        )
        _append_operation(trace, 9, quality, {
            "prefrontal_monitoring": bio.prefrontal_control.snapshot(),
            "error_signal": 1.0 - float(quality["quality_score"]),
        })

        # 11. Generate localized repair prompts without fabricating absent evidence.
        repair = repair_explanation(
            explanation, quality, propositions, config.maximum_repair_passes
        )
        _append_operation(trace, 10, repair, {
            "recurrent_reprocessing_requested": bool(repair["repair_questions"]),
            "unsupported_facts_added": False,
        })

        # 12. Prepare a teaching outline adapted to the selected listener level.
        teaching = prepare_teaching_output(
            explanation, propositions, request.listener_level,
            request.enable_teaching,
        )
        _append_operation(trace, 11, teaching, {
            "language_and_audience_control": "distributed language/control surrogate",
            "listener_model_is_explicitly_approximate": True,
        })

        # 13. Optionally build a symbolic spatial/process representation.
        imagery = construct_imagery(
            propositions, elaboration, explanation,
            request.enable_imagery, config.embedding_dimensions,
        )
        _append_operation(trace, 12, imagery, {
            "imagery_policy": "optional",
            "perceptual_reactivation": "symbolic scene/process surrogate",
        })

        # 14. Bind source, questions, explanation, action context, and imagery as an episode.
        episode_memory = bind_episode(
            request, propositions, questions, explanation, imagery,
            config.embedding_dimensions,
        )
        feature_vectors = {
            "source": vectorize_texts(
                [item["text"] for item in propositions], config.embedding_dimensions
            ),
            "question": vectorize_texts(
                [str(item["text"]) for item in questions], config.embedding_dimensions
            ),
            "explanation": explanation_vector,
            "goal": vectorize_texts([request.goal], config.embedding_dimensions),
        }
        if imagery.get("enabled"):
            feature_vectors["imagery"] = vectorize_texts(
                [str(item["label"]) for item in imagery.get("nodes", [])],
                config.embedding_dimensions,
            )
        hippocampal_binding = bio.hippocampal_episode.bind(
            feature_vectors,
            {
                "episode_id": episode_memory["episode_id"],
                "source_ids": episode_memory["source_proposition_ids"],
                "context": dict(request.context),
            },
        )
        episode_memory["hippocampal_binding"] = hippocampal_binding
        _append_operation(trace, 13, episode_memory, {
            "hippocampal_episode_binding": hippocampal_binding,
            "cortical_content_and_hippocampal_binding_interaction": True,
        })

        # 15. Update bounded associative weights using a learning gate and timing proxy.
        attention_quality = clamp(float(np.mean([
            item["attention_weight"] * item["reliability"]
            for item in attention["selected_evidence"]
        ]))) if attention["selected_evidence"] else 0.0
        gate = bio.prefrontal_control.learning_gate(
            attention_quality, float(quality["uncertainty"])
        )
        explanation_activity = bio.temporal_language.encode(
            str(explanation["text"])
        )["population_activity"]
        plasticity = bio.synaptic_plasticity.update(
            last_source_activity,
            explanation_activity,
            learning_rate=config.learning_rate,
            gate=min(config.plasticity_gate, gate["learning_gate"]),
            delta_t_seconds=float(request.metadata.get("pre_post_delta_t_seconds", 0.01)),
            state=active_state,
        )
        plasticity["learning_gate_diagnostics"] = gate
        plasticity["plasticity_interpretation"] = (
            "simplified activity-dependent model; not a universal biological learning rule"
        )
        _append_operation(trace, 14, plasticity, {
            "synaptic_plasticity": bio.synaptic_plasticity.snapshot(),
            "prefrontal_control": bio.prefrontal_control.snapshot(),
        })

        # 16. Estimate comprehension and confidence; calibrate only with external labels.
        comprehension = monitor_comprehension(
            quality, explanation, request.metadata
        )
        active_state.comprehension_history.append(
            float(comprehension["comprehension_estimate"])
        )
        _append_operation(trace, 15, comprehension, {
            "metacognitive_monitor": "confidence and error-monitoring surrogate",
            "independent_accuracy_labels_available": bool(
                comprehension["independent_calibration_available"]
            ),
        })

        # 17. Close only if supported; scaffolds alone cannot certify learner mastery.
        closure = decide_closure(
            quality, comprehension, config.minimum_quality
        )
        if not explanation["is_learner_generated"]:
            closure["episode_status"] = "incomplete_learner_explanation_required"
            closure["stop_condition_met"] = False
            closure["next_action"] = (
                "ask_learner_to_explain_in_own_words_then_reassess"
            )
            closure["reason"] = (
                "An extractive scaffold is not evidence that the learner independently "
                "generated or understands the explanation."
            )
        _append_operation(trace, 16, closure, {
            "prefrontal_goal_control": bio.prefrontal_control.snapshot(),
            "encoding_continues_beyond_immediate_task": True,
        })

        episode_memory["status"] = closure["episode_status"]
        episode_memory["quality_score"] = quality["quality_score"]
        episode_memory["comprehension_estimate"] = comprehension["comprehension_estimate"]
        active_state.episode_memory.append(episode_memory)
        active_state.process_count += 1

        output = {
            "episode_id": episode_memory["episode_id"],
            "episode_status": closure["episode_status"],
            "target_proposition": target["text"],
            "question_count": len(questions),
            "explanation_origin": explanation["origin"],
            "learner_generated_explanation": explanation["is_learner_generated"],
            "quality_score": quality["quality_score"],
            "comprehension_estimate": comprehension["comprehension_estimate"],
            "unresolved_gap_count": len(quality["unresolved_gaps"]),
            "teaching_enabled": teaching["enabled"],
            "imagery_enabled": imagery["enabled"],
            "retrieval_cues": episode_memory["retrieval_cues"],
            "spaced_repetition_created": False,
            "formal_testing_program_created": False,
            "biological_fidelity": "computational surrogate",
        }
        return ActiveEngagementResult(
            learner_id=self.learner_id,
            operation_trace=tuple(trace),
            questions=tuple(questions),
            candidate_answers=tuple(candidates),
            explanation=explanation,
            elaboration=elaboration,
            quality=quality,
            repair=repair,
            teaching=teaching,
            imagery=imagery,
            episode_memory=episode_memory,
            plasticity=plasticity,
            comprehension=comprehension,
            closure=closure,
            biological=bio.snapshot(),
            final_state=active_state.to_dict(),
            output=output,
        )


def register_annotations(
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index every Point 01 Python module, class, and callable in the ORE SQLite catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root else Path(__file__).resolve().parents[1]
    )
    return sync_repository(
        LoggerDatabase(database_path),
        root,
        include_paths={"ore_active_engagement"},
    )


def record_result_to_sqlite(
    result: ActiveEngagementResult,
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist all 17 operation records and index every Point 01 class/function annotation."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root else Path(__file__).resolve().parents[1]
    )
    database = LoggerDatabase(database_path)
    catalogue = sync_repository(
        database, root, include_paths={"ore_active_engagement"}
    )
    run_id = database.create_run(
        script_path="ore_active_engagement/point01",
        module_name="ore_active_engagement.pipeline",
        entry_kind="active_engagement_learning",
        working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id,
                sequence_no=sequence,
                event_kind="operation",
                source_path="ore_active_engagement/pipeline.py",
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                metadata={
                    "operation_id": item["operation_id"],
                    "operation": item["operation"],
                    "status": item["status"],
                    "source_step": item["source_step"],
                    "evidence_level": item["evidence_level"],
                    "mathematical_form": item["mathematical_form"],
                    "functional_output": item["functional_output"],
                    "biological_output": item["biological_output"],
                },
            )
        database.finish_run(
            run_id,
            "completed",
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
    except Exception as exc:
        database.finish_run(
            run_id,
            "failed",
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            error_summary=type(exc).__name__ + ": " + str(exc),
        )
        raise
    return {
        "run_id": run_id,
        "annotation_catalogue": catalogue,
        "database_summary": database.summary(),
    }
