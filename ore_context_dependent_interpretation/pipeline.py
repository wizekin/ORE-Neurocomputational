"""End-to-end 16-stage Point 12 context-dependent interpretation processor and SQLite logger."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .biology import BiologicalContextNetwork
from .contracts import (
    CandidateHypothesis, ContextEvidence, ContextInterpretationConfig,
    ContextInterpretationResult, ContextState, InputEvent, ScoredCandidate,
)
from .functional import (
    adapt_context_weights, bayesian_update, build_evidence_ledger,
    detect_conflicts, estimate_context_priors, generate_candidates,
    resolve_competition, retrieve_context, revise_candidates,
    score_input_evidence, select_goal, update_situation_model,
)
from .registry import OPERATION_REGISTRY


def _trace_operation(
    trace: list[dict[str, Any]],
    index: int,
    functional: Mapping[str, Any],
    biological: Mapping[str, Any] | None = None,
    status: str = "completed",
) -> None:
    """Append a canonical source-ordered operation record with math and biology notes."""
    spec = OPERATION_REGISTRY[index]
    trace.append({
        "step": index + 1,
        "operation_id": spec.operation_id,
        "operation": spec.name,
        "status": status,
        "explanation": spec.purpose,
        "evidence_level": spec.evidence_level,
        "mathematical_form": spec.mathematical_form,
        "functional_output": dict(functional),
        "biological_output": dict(biological or {}),
    })


def _scored_candidates(
    candidates: Sequence[CandidateHypothesis],
    priors: Mapping[str, float],
    prior_scores: Mapping[str, float],
    support: Mapping[str, Sequence[str]],
    likelihoods: Mapping[str, float],
    input_scores: Mapping[str, float],
    posteriors: Mapping[str, float],
    penalties: Mapping[str, float] | None = None,
) -> tuple[ScoredCandidate, ...]:
    """Join candidate evidence channels into validated posterior records."""
    penalties = penalties or {}
    return tuple(ScoredCandidate(
        candidate=item,
        context_prior=float(priors[item.candidate_id]),
        input_likelihood=float(likelihoods[item.candidate_id]),
        posterior=float(posteriors[item.candidate_id]),
        context_score=float(prior_scores[item.candidate_id]),
        input_score=float(input_scores[item.candidate_id]),
        evidence_ids=tuple(support.get(item.candidate_id, ())),
        conflict_penalty=float(penalties.get(item.candidate_id, 0.0)),
    ) for item in candidates)


class ContextDependentInterpretationProcessor:
    """Coordinate context retrieval, candidate competition, situation updating, and revision."""

    def __init__(
        self,
        config: ContextInterpretationConfig | None = None,
        state: ContextState | None = None,
    ) -> None:
        """Initialize a reusable Point 12 processor and its distributed biological surrogates."""
        self.config = config or ContextInterpretationConfig()
        self.state = state or ContextState()
        self.biological = BiologicalContextNetwork()
        self.process_count = 0

    def process(
        self,
        event: InputEvent,
        context: ContextState | None = None,
    ) -> ContextInterpretationResult:
        """Run all 16 context-dependent interpretation operations for one incoming event."""
        self.process_count += 1
        state = context if context is not None else self.state
        trace: list[dict[str, Any]] = []
        bio = self.biological
        encoded = bio.language.encode(event)
        _trace_operation(trace, 0, encoded, bio.language.snapshot())

        retrieved = retrieve_context(event, state, self.config)
        discourse_rows = retrieved["discourse"]
        reactivated = bio.hippocampal_discourse.retrieve(discourse_rows)
        _trace_operation(trace, 1, {
            "retrieved_count": len(discourse_rows), "reactivated_ids": reactivated,
            "topic_id": state.current_topic_id,
        }, bio.hippocampal_discourse.snapshot())

        knowledge_rows = retrieved["knowledge"]
        semantic_activation = bio.temporal_semantic.activate(knowledge_rows)
        _trace_operation(trace, 2, {
            "retrieved_count": len(knowledge_rows),
            "evidence_ids": [row["evidence_id"] for row in knowledge_rows],
        }, {"semantic_activation": semantic_activation})

        goal, goal_source = select_goal(event, state)
        state.current_goal = goal
        bio.medial_prefrontal.select(goal)
        # Re-score retrieval after selecting the active goal so goal-conditioned relevance
        # affects all contextual sources, not just the long-term knowledge lookup.
        retrieved = retrieve_context(event, state, self.config)
        _trace_operation(trace, 3, {
            "goal": goal, "goal_source": goal_source,
            "goal_source_preferences": dict(state.goal_source_preferences),
        }, bio.medial_prefrontal.snapshot())

        scene_rows = retrieved["scene"]
        scene_cues = bio.perceptual_scene.encode(event, scene_rows)
        _trace_operation(trace, 4, {
            "scene_cues": scene_cues,
            "retrieved_scene_ids": [row["evidence_id"] for row in scene_rows],
            "explicit_scene_cues": event.scene_cues,
        }, bio.perceptual_scene.snapshot())

        convention_rows = retrieved["convention"]
        speaker_rows = retrieved["speaker"]
        social_count = len(convention_rows) + len(speaker_rows)
        _trace_operation(trace, 5, {
            "conventions": [row["evidence_id"] for row in convention_rows],
            "speaker_context": [row["evidence_id"] for row in speaker_rows],
            "applicability_is_probabilistic": True,
        }, {"social_context_records": social_count})

        retrieved_flat = [row for rows in retrieved.values() for row in rows]
        provisional_model = bio.posterior_midline.integrate(state)
        _trace_operation(trace, 6, {
            "existing_situation": state.situation.to_dict(),
            "retrieved_context_count": len(retrieved_flat),
            "provisional_summary": provisional_model,
        }, bio.posterior_midline.snapshot())

        candidates = generate_candidates(event, retrieved, self.config)
        _trace_operation(trace, 7, {
            "candidate_ids": [item.candidate_id for item in candidates],
            "candidate_count": len(candidates),
            "open_set_alternative_present": any(
                item.candidate_id == self.config.unknown_candidate_id for item in candidates
            ),
        })

        priors, prior_scores, support = estimate_context_priors(
            candidates, retrieved, goal, self.config.temperature
        )
        _trace_operation(trace, 8, {
            "context_priors": priors, "context_fit_scores": prior_scores,
            "prior_support": {key: list(value) for key, value in support.items()},
        }, {"semantic_activation": bio.temporal_semantic.snapshot()})

        likelihoods, input_scores = score_input_evidence(event, candidates)
        _trace_operation(trace, 9, {
            "input_likelihoods": likelihoods, "direct_input_fit": input_scores,
            "context_and_input_scored_separately": True,
        }, bio.language.snapshot())

        posteriors = bayesian_update(priors, likelihoods, self.config.minimum_probability)
        scored = _scored_candidates(
            candidates, priors, prior_scores, support, likelihoods, input_scores, posteriors
        )
        status, selected, uncertainty, normalized_entropy = resolve_competition(
            scored, self.config
        )
        top = max((item.posterior for item in scored), default=0.0)
        second = sorted((item.posterior for item in scored), reverse=True)[1:2]
        margin = max(0.0, top - (second[0] if second else 0.0))
        selection_state = bio.prefrontal_control.select(status, len(scored))
        _trace_operation(trace, 10, {
            "status": status, "posterior_candidates": {
                item.candidate.candidate_id: item.posterior for item in scored
            }, "uncertainty_nats": uncertainty,
            "normalized_entropy": normalized_entropy, "posterior_margin": margin,
        }, selection_state)

        # Detect against the pre-update model, then integrate without erasing a
        # previously supported conflicting value. The conflict operation is recorded next.
        conflicts = detect_conflicts(state.situation, event)
        update_situation_model(
            state.situation, event, status, selected, uncertainty, conflicts, scored
        )
        _trace_operation(trace, 11, {
            "situation_event_count": len(state.situation.events),
            "explicit_fact_count": len(event.explicit_facts),
            "selected_candidate": selected.candidate.candidate_id if selected else None,
            "inferred_facts_remain_provenance_marked": True,
        }, bio.posterior_midline.integrate(state))

        _trace_operation(trace, 12, {
            "conflicts": [dict(item) for item in conflicts],
            "conflict_count": len(conflicts),
            "surprising_input_is_not_automatically_rejected": True,
        }, {"coherence_monitoring": "explicit_fact_mismatch_check"})

        revised = revise_candidates(scored, conflicts, self.config)
        revised_status, revised_selected, revised_uncertainty, revised_entropy = (
            resolve_competition(revised, self.config)
        )
        revised_margin = 0.0
        ranked_revised = sorted(revised, key=lambda item: item.posterior, reverse=True)
        if ranked_revised:
            revised_margin = ranked_revised[0].posterior - (
                ranked_revised[1].posterior if len(ranked_revised) > 1 else 0.0
            )
        previous_id = selected.candidate.candidate_id if selected else None
        new_id = revised_selected.candidate.candidate_id if revised_selected else None
        was_revised = bool(conflicts) and (
            revised_status != status or new_id != previous_id
            or any(item.conflict_penalty > 0 for item in revised)
        )
        if was_revised:
            status = "revised" if revised_selected is not None else "ambiguous"
            selected = revised_selected
            scored = revised
            uncertainty = revised_uncertainty
            normalized_entropy = revised_entropy
            margin = revised_margin
            state.situation.revision_count += 1
            # Remove only inferred facts introduced by this event, never explicit evidence.
            for fact_key in list(state.situation.fact_provenance):
                provenance = state.situation.fact_provenance.get(fact_key, {})
                if (provenance.get("event_id") == event.event_id
                        and provenance.get("status") == "inferred"):
                    state.situation.fact_provenance.pop(fact_key, None)
                    state.situation.facts.pop(fact_key, None)
            if selected is not None:
                for fact_key, fact_value in selected.candidate.facts.items():
                    if fact_key not in state.situation.facts:
                        state.situation.facts[str(fact_key)] = fact_value
                        state.situation.fact_provenance[str(fact_key)] = {
                            "status": "inferred", "provenance": "candidate_hypothesis",
                            "event_id": event.event_id,
                            "candidate_id": selected.candidate.candidate_id,
                            "confidence": selected.posterior,
                        }
                state.situation.confidence = selected.posterior
                state.situation.unresolved_items = [
                    item for item in state.situation.unresolved_items
                    if item.get("event_id") != event.event_id
                ]
            elif not any(
                item.get("event_id") == event.event_id
                for item in state.situation.unresolved_items
            ):
                state.situation.unresolved_items.append({
                    "event_id": event.event_id,
                    "candidate_ids": [item.candidate.candidate_id for item in scored],
                    "candidate_posteriors": {
                        item.candidate.candidate_id: item.posterior for item in scored
                    },
                    "reason": "new evidence left the interpretation unresolved",
                })
            bio.prefrontal_control.select(status, len(scored))
            if state.situation.events and state.situation.events[-1]["event_id"] == event.event_id:
                state.situation.events[-1].update({
                    "status": status,
                    "selected_meaning": selected.candidate.meaning if selected else None,
                    "confidence": selected.posterior if selected else None,
                    "revision_applied": True,
                })
        _trace_operation(trace, 13, {
            "revision_applied": was_revised,
            "previous_candidate_id": previous_id, "revised_candidate_id": new_id,
            "posterior_candidates": {
                item.candidate.candidate_id: item.posterior for item in scored
            },
            "revision_count": state.situation.revision_count,
        }, bio.prefrontal_control.snapshot())

        state.discourse_history.append(ContextEvidence(
            evidence_id="turn:" + event.event_id, source_type="discourse",
            content=event.content, confidence=event.reliability,
            reliability=event.reliability, concepts=tuple(encoded["tokens"]),
            facts=event.explicit_facts, goal_tags=(goal,),
            topic_id=str(event.metadata.get("topic_id", state.current_topic_id)),
            speaker_id=event.speaker_id, timestamp=str(event.metadata.get("timestamp", "")),
            provenance="current_event_history",
        ))
        if event.scene_cues:
            state.scene_evidence.append(ContextEvidence(
                evidence_id="scene:" + event.event_id, source_type="scene",
                content=" ".join(event.scene_cues), concepts=event.scene_cues,
                reliability=event.reliability, confidence=event.reliability,
                provenance="explicit_scene_cues",
            ))
        next_topic = event.metadata.get("topic_id")
        if isinstance(next_topic, str) and next_topic.strip():
            state.current_topic_id = next_topic.strip()
        updated_weights = adapt_context_weights(
            state, selected, retrieved, self.config.adaptation_rate
        )
        _trace_operation(trace, 14, {
            "context_weights": updated_weights,
            "adaptation_rate": self.config.adaptation_rate,
            "weight_update_only_after_confident_selection": selected is not None,
        }, {"goal_state": bio.medial_prefrontal.snapshot()})

        if selected is not None:
            output = {
                "output_type": "task_conditioned_interpretation",
                "goal": goal, "interpretation": selected.candidate.meaning,
                "confidence": selected.posterior, "clarification_needed": False,
                "assumptions": list(selected.candidate.assumptions),
            }
        elif status == "unresolved":
            output = {
                "output_type": "unresolved",
                "goal": goal, "interpretation": None,
                "clarification_needed": True,
                "clarification_request": "Please provide more context for this input.",
            }
        else:
            alternatives = [
                item.candidate.meaning for item in sorted(
                    scored, key=lambda item: item.posterior, reverse=True
                )[:3]
            ]
            output = {
                "output_type": "ambiguous_interpretation",
                "goal": goal, "interpretation": None,
                "candidate_meanings": alternatives, "clarification_needed": True,
                "clarification_request": (
                    "Which interpretation is intended: " + " or ".join(alternatives) + "?"
                    if alternatives else "Could you provide more context?"
                ),
            }
        ledger = build_evidence_ledger(event, retrieved, scored)
        biology_snapshot = bio.snapshot()
        _trace_operation(trace, 15, output, {
            "task_goal": goal, "situation_model": bio.posterior_midline.snapshot(),
            "selection_control": bio.prefrontal_control.snapshot(),
        })
        return ContextInterpretationResult(
            event_id=event.event_id, status=status, selected=selected,
            candidates=tuple(scored), uncertainty=uncertainty,
            normalized_entropy=normalized_entropy, posterior_margin=margin,
            conflicts=tuple(conflicts), evidence_ledger=ledger,
            operation_trace=tuple(trace), biological=biology_snapshot,
            output=output, situation=state.situation.to_dict(),
            diagnostics={
                "goal_source": goal_source,
                "retrieved_counts": {key: len(value) for key, value in retrieved.items()},
                "input_and_context_evidence_separated": True,
                "open_set_candidate_retained": any(
                    item.candidate.candidate_id == self.config.unknown_candidate_id
                    for item in scored
                ),
                "revision_applied": was_revised,
                "source_provenance_preserved": True,
            },
        )


def register_annotations(
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index every Point 12 class and callable docstring in the ORE SQLite annotation catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    return sync_repository(
        LoggerDatabase(database_path), root,
        include_paths={"ore_context_dependent_interpretation"},
    )


def record_result_to_sqlite(
    result: ContextInterpretationResult,
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist the 16-stage trace and index Point 12 annotations in the ORE SQLite database."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    database = LoggerDatabase(database_path)
    catalog = sync_repository(
        database, root, include_paths={"ore_context_dependent_interpretation"}
    )
    run_id = database.create_run(
        script_path="ore_context_dependent_interpretation/point12",
        module_name="ore_context_dependent_interpretation.pipeline",
        entry_kind="context_dependent_interpretation",
        working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id, sequence_no=sequence, event_kind="operation",
                source_path="ore_context_dependent_interpretation/pipeline.py",
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
        "run_id": run_id, "annotation_catalog": catalog,
        "database_summary": database.summary(),
    }
