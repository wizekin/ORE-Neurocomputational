"""Seventeen-stage Point 23 parsing pipeline and shared ORE SQLite integration."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

from .biology import GardenPathNetwork
from .contracts import (
    ParseHypothesis,
    Point23Config,
    Point23Result,
    Point23State,
    SentenceRequest,
)
from .functional import (
    activate_lexical_entries,
    decide_reanalysis_depth,
    detect_disambiguating_conflict,
    expected_behavioral_signatures,
    estimate_structural_expectations,
    generate_syntactic_continuations,
    integrate_semantic_constraints,
    integrate_surface_cues,
    locate_revision_points,
    maintain_dependencies,
    normalize_prior,
    reconcile_interpretations,
    reconstruct_alternative_structure,
    reassign_thematic_roles,
    segment_input,
    select_provisional_parse,
    softmax,
    test_parse_viability,
    trigger_reanalysis,
    update_expectations_from_experience,
    validate_point23,
)
from .registry import OPERATION_REGISTRY


_POPULATION_DRIVES: tuple[Mapping[str, float], ...] = (
    {"auditory_visual_input": 0.90, "attention": 0.70},
    {"lexical_temporal": 0.90, "posterior_temporal_syntax": 0.45},
    {"posterior_temporal_syntax": 0.85, "left_ifg_unification": 0.55},
    {"posterior_temporal_syntax": 0.80, "lexical_temporal": 0.60},
    {"anterior_temporal_semantics": 0.85, "semantic_role_binding": 0.65},
    {"left_ifg_unification": 0.70, "frontoparietal_control": 0.55},
    {"working_memory": 0.90, "attention": 0.75},
    {"conflict_monitoring": 0.85, "attention": 0.65},
    {"posterior_temporal_syntax": 0.80, "left_ifg_unification": 0.70},
    {"left_ifg_unification": 0.90, "frontoparietal_control": 0.80},
    {"working_memory": 0.85, "lexical_temporal": 0.65},
    {"left_ifg_unification": 0.90, "posterior_temporal_syntax": 0.85},
    {"semantic_role_binding": 0.90, "anterior_temporal_semantics": 0.85},
    {"metacognitive_confidence": 0.80, "working_memory": 0.70},
    {"frontoparietal_control": 0.75, "attention": 0.80},
    {"prosody_punctuation": 0.85, "anterior_temporal_semantics": 0.65},
    {"distributed_learning": 0.90, "lexical_temporal": 0.70},
)


def _append_operation(
    trace: list[dict[str, Any]],
    index: int,
    functional_output: Mapping[str, Any],
    biological_output: Mapping[str, Any],
) -> None:
    """Append a completed trace row aligned with the source-order registry."""
    spec = OPERATION_REGISTRY[index]
    trace.append({
        "step": index + 1,
        "operation_id": spec.operation_id,
        "operation": spec.name,
        "purpose": spec.purpose,
        "mathematical_form": spec.mathematical_form,
        "biological_reference": spec.biological_reference,
        "evidence_level": spec.evidence_level,
        "status": "completed",
        "functional_output": dict(functional_output),
        "biological_output": dict(biological_output),
    })


def _initial_parse_from_first_choice(
    parses: tuple[ParseHypothesis, ...] | list[ParseHypothesis],
    structural: Mapping[str, Any],
) -> tuple[str, int]:
    """Choose the prefix-leading parse at the first supplied structural choice point."""
    positions = [
        point.token_index for parse in parses for point in parse.choice_points
    ]
    index = min(positions) if positions else 0
    history = structural["posterior_history"]
    if not history:
        priors = normalize_prior(parses)
        return max(priors, key=priors.get), index
    index = min(index, len(history) - 1)
    probabilities = history[index]
    return max(probabilities, key=probabilities.get), index


class Point23GardenPathProcessor:
    """Coordinate lexical access, competing parses, conflict detection, revision, and adaptation.

    The processor scores grammar-generated parse hypotheses supplied by the caller. It does not
    claim to be a comprehensive natural-language grammar or an end-to-end parser for all languages.
    """

    def __init__(
        self,
        config: Point23Config | None = None,
        state: Point23State | None = None,
        biological: GardenPathNetwork | None = None,
    ) -> None:
        """Initialize the functional configuration and optional population-rate reference."""
        self.config = config or Point23Config()
        self.state = state or Point23State()
        self.biological = biological or GardenPathNetwork(enabled=self.config.simulate_biology)
        self.process_count = 0

    def process(
        self,
        request: SentenceRequest,
        state: Point23State | None = None,
    ) -> Point23Result:
        """Execute the complete seventeen-operation sentence-comprehension algorithm."""
        if not isinstance(request, SentenceRequest):
            raise TypeError("request must be a SentenceRequest")
        current = state if state is not None else self.state
        config = self.config
        network = self.biological if config.simulate_biology else None
        current.run_count += 1
        self.process_count += 1
        trace: list[dict[str, Any]] = []

        def biological(
            index: int,
            *,
            conflict: float = 0.0,
            memory_load: float = request.working_memory_load,
            cue_strength: float = 0.0,
        ) -> dict[str, Any]:
            """Advance the rate-coded reference or explicitly note that it is disabled."""
            if network is None:
                return {
                    "model": "disabled",
                    "operation_id": OPERATION_REGISTRY[index].operation_id,
                    "scope": "Biological-reference track disabled by configuration.",
                }
            return network.advance(
                OPERATION_REGISTRY[index].operation_id,
                _POPULATION_DRIVES[index],
                conflict=conflict,
                memory_load=memory_load,
                cue_strength=cue_strength,
            )

        # 1. Segment the incoming stream and preserve punctuation/time-order information.
        segmentation = segment_input(request)
        _append_operation(trace, 0, segmentation, biological(0))

        # 2. Retrieve lexical forms, senses, and grammatical-category candidates.
        lexical = activate_lexical_entries(request, segmentation)
        _append_operation(
            trace, 1, lexical,
            biological(1, cue_strength=min(1.0, lexical["ambiguous_word_count"] / 3.0)),
        )

        # 3. Extend a supplied grammar candidate set; a surface-only fallback is explicitly marked.
        generated = generate_syntactic_continuations(request, lexical)
        parses = tuple(generated["parse_hypotheses"])
        _append_operation(trace, 2, generated, biological(2))

        # 4. Score structural expectations incrementally and record surprisal by token.
        structural = estimate_structural_expectations(parses, lexical, config)
        _append_operation(
            trace, 3, structural,
            biological(3, conflict=min(1.0, structural["maximum_surprisal"] / 5.0)),
        )

        # 5. Combine syntax and semantic plausibility, but do not let plausibility veto grammar.
        integration = integrate_semantic_constraints(parses, structural, config)
        _append_operation(trace, 4, integration, biological(4))

        # 6. Select a provisional parse from the earliest choice-point prefix and retain a beam.
        selected = select_provisional_parse(parses, integration, config)
        initial_parse_id, initial_choice_index = _initial_parse_from_first_choice(parses, structural)
        selected["initial_parse_id"] = initial_parse_id
        selected["initial_choice_index"] = initial_choice_index
        selected["initial_parse_probability"] = structural["posterior_history"][
            min(initial_choice_index, len(structural["posterior_history"]) - 1)
        ].get(initial_parse_id, 0.0) if structural["posterior_history"] else 0.0
        current_best = max(integration["parse_probabilities"], key=integration["parse_probabilities"].get)
        selected["current_best_parse_id"] = current_best
        _append_operation(trace, 5, selected, biological(5, conflict=0.15 if selected["premature_commitment_risk"] else 0.0))

        # 7. Maintain dependencies, open roles, and accessibility under working-memory limits.
        memory = maintain_dependencies(parses, selected, lexical, request, config)
        _append_operation(
            trace, 6, memory,
            biological(6, memory_load=memory["estimated_load"]),
        )

        # 8. Detect disambiguation through surprisal, parse switching, and posterior conflict.
        conflict = detect_disambiguating_conflict(structural, selected, request)
        _append_operation(
            trace, 7, conflict,
            biological(7, conflict=conflict["conflict_score"]),
        )

        # 9. Test structural/lexical viability while retaining rejected candidates for audit.
        viability = test_parse_viability(parses, integration, config)
        _append_operation(trace, 8, viability, biological(8, conflict=conflict["conflict_score"]))

        # 10. Trigger revision only when candidate competition or incompatibility justifies it.
        trigger = trigger_reanalysis(selected, conflict, viability, config)
        _append_operation(
            trace, 9, trigger,
            biological(9, conflict=1.0 if trigger["revision_requested"] else 0.0),
        )

        # 11. Locate the earliest choice point separating the initial and alternative structures.
        revision_points = locate_revision_points(parses, trigger)
        _append_operation(trace, 10, revision_points, biological(10, memory_load=memory["estimated_load"]))

        # 12. Reconstruct from alternative grammar hypotheses without inventing unsupported syntax.
        revision = reconstruct_alternative_structure(parses, viability, trigger, revision_points)
        _append_operation(
            trace, 11, revision,
            biological(11, conflict=0.85 if revision["revision_attempted"] else 0.0),
        )

        # 13. Recompute thematic roles and dependencies from the selected/revised structure.
        pre_cue_parse_id = revision["revised_parse_id"]
        roles = reassign_thematic_roles(
            parses, pre_cue_parse_id, lexical, integration["parse_probabilities"],
        )
        _append_operation(
            trace, 12, roles,
            biological(12, conflict=conflict["conflict_score"]),
        )

        # 14. Reconcile old and new interpretations without assuming that the first parse disappears.
        reconciliation = reconcile_interpretations(
            selected["initial_parse_id"], pre_cue_parse_id,
            integration["parse_probabilities"], parses,
        )
        _append_operation(
            trace, 13, reconciliation,
            biological(13, conflict=1.0 if reconciliation["interpretation_changed"] else 0.0),
        )

        # 15. Choose full reanalysis, a good-enough representation, or explicit uncertainty.
        depth = decide_reanalysis_depth(
            parses, pre_cue_parse_id, integration["parse_probabilities"], memory, request, config,
        )
        _append_operation(trace, 14, depth, biological(14, memory_load=memory["estimated_load"]))

        # 16. Integrate prosody, punctuation, and discourse and make the final parse/role readout.
        surface = integrate_surface_cues(
            request, parses, integration["combined_log_scores"], lexical, config,
        )
        final_parse_id = surface["final_parse_id"]
        final_roles = reassign_thematic_roles(
            parses, final_parse_id, lexical, surface["parse_probabilities"],
        )
        final_reconciliation = reconcile_interpretations(
            selected["initial_parse_id"], final_parse_id, surface["parse_probabilities"], parses,
        )
        final_depth = decide_reanalysis_depth(
            parses, final_parse_id, surface["parse_probabilities"], memory, request, config,
        )
        if generated["fallback_used"]:
            final_depth["interpretation_status"] = "unresolved"
            final_depth["unresolved_ambiguities"] = list(final_depth["unresolved_ambiguities"]) + [
                "no_grammar_parse_hypotheses_supplied"
            ]
            final_depth["stopping_reason"] = "surface-only fallback cannot establish a syntactic parse"
        surface["final_thematic_roles"] = final_roles
        surface["reconciliation"] = final_reconciliation
        surface["interpretation_decision"] = final_depth
        _append_operation(
            trace, 15, surface,
            biological(
                15,
                conflict=conflict["conflict_score"],
                cue_strength=min(1.0, len(surface["cue_sources_used"]) / 4.0),
            ),
        )

        # 17. Store the parse outcome and adapt only from explicit correctness feedback.
        learning = update_expectations_from_experience(
            request, parses, final_parse_id, current, config,
        )
        _append_operation(trace, 16, learning, biological(16))

        validation = validate_point23(trace, surface["parse_probabilities"], parses)
        trace[-1]["functional_output"]["validation"] = validation
        current.interpretation_history = current.interpretation_history[-500:]
        final_parse = next(item for item in parses if item.parse_id == final_parse_id)
        p_final = float(surface["parse_probabilities"][final_parse_id])
        confidence = {
            "syntactic_structure": p_final,
            "semantic_interpretation": p_final * final_parse.semantic_plausibility,
            "thematic_roles": p_final if final_roles["thematic_roles"] else 0.0,
            "overall": (
                0.40 * p_final
                + 0.35 * p_final * final_parse.semantic_plausibility
                + 0.25 * (p_final if final_roles["thematic_roles"] else 0.0)
            ),
        }
        behavioral = expected_behavioral_signatures(
            parses, structural["token_trace"], (revision,), surface["parse_probabilities"], final_roles,
        )
        unresolved = tuple(final_depth["unresolved_ambiguities"])
        biological_snapshot = network.snapshot() if network else {
            "model": "disabled",
            "operation_count": 0,
            "scientific_limitations": ["Biological-reference track disabled by configuration."],
        }
        return Point23Result(
            sentence_id=request.sentence_id,
            tokens=tuple(row["text"] for row in segmentation["tokens"]),
            lexical_state=tuple(lexical["lexical_tokens"]),
            initial_parse_id=selected["initial_parse_id"],
            final_parse_id=final_parse_id,
            parse_probabilities=dict(surface["parse_probabilities"]),
            thematic_roles=dict(final_roles["thematic_roles"]),
            confidence=confidence,
            interpretation_status=final_depth["interpretation_status"],
            unresolved_ambiguities=unresolved,
            revisions=(revision,),
            diagnostics={
                "segmentation": segmentation,
                "lexical": lexical,
                "candidate_generation": generated,
                "structural_expectations": structural,
                "semantic_integration": integration,
                "provisional_selection": selected,
                "working_memory": memory,
                "conflict": conflict,
                "viability": viability,
                "revision_trigger": trigger,
                "revision_points": revision_points,
                "role_reassignment": final_roles,
                "reconciliation": final_reconciliation,
                "reanalysis_depth": final_depth,
                "surface_cue_integration": surface,
                "learning": learning,
                "validation": validation,
            },
            behavioral_predictions=behavioral,
            biological=biological_snapshot,
            operation_trace=tuple(trace),
            final_state=current.to_dict(),
            metadata={
                "model": "ORE Point 23 garden-path and syntactic ambiguity processor",
                "source_sequence_steps": 17,
                "token_count": len(segmentation["tokens"]),
                "parse_candidate_count": len(parses),
                "revision_attempted": revision["revision_attempted"],
                "revision_succeeded": revision["revision_succeeded"],
                "biological_track_enabled": network is not None,
                "grammar_candidates_supplied": not generated["fallback_used"],
                "rereading_not_equated_with_reanalysis": True,
                "initial_interpretation_persistence_tracked": True,
                "feedback_adaptation_requires_explicit_parse_id": True,
                "scientific_status": "configurable computational synthesis; not a validated neural or general-purpose parser",
            },
        )


def register_annotations(
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Register Point 23 package classes, functions, and methods in shared SQLite annotations."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    catalogue = sync_repository(LoggerDatabase(target), root, include_paths={"ore_garden_path"})
    return {
        **catalogue,
        "database_path": str(target),
        "scope": "ore_garden_path",
        "annotation_table": "annotations",
    }


def record_result_to_sqlite(
    result: Point23Result,
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist each of the seventeen functional/biological outputs through the shared ORE logger."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    if not isinstance(result, Point23Result):
        raise TypeError("result must be a Point23Result")
    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    database = LoggerDatabase(target)
    catalogue = sync_repository(database, root, include_paths={"ore_garden_path"})
    run_id = database.create_run(
        script_path="ore_garden_path/point23",
        module_name="ore_garden_path.pipeline",
        entry_kind="garden_path_point23",
        working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id,
                sequence_no=sequence,
                event_kind="operation",
                source_path="ore_garden_path/pipeline.py",
                elapsed_ms=0.0,
                metadata={
                    "operation_id": item["operation_id"],
                    "operation": item["operation"],
                    "status": item["status"],
                    "mathematical_form": item["mathematical_form"],
                    "biological_reference": item["biological_reference"],
                    "evidence_level": item["evidence_level"],
                    "functional_output": item["functional_output"],
                    "biological_output": item["biological_output"],
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
            error_summary=f"{type(exc).__name__}: {exc}",
        )
        raise
    return {
        "run_id": run_id,
        "annotation_catalogue": catalogue,
        "database_path": str(target),
        "database_summary": database.summary(),
    }
