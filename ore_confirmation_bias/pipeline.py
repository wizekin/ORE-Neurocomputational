"""Fourteen-operation Point 21 pipeline and shared ORE SQLite integration."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

from .biology import ConfirmationBiasNetwork
from .contracts import Point21Config, Point21Request, Point21Result, Point21State
from .functional import (
    allocate_attention,
    apply_corrective_control,
    assess_evidence_quality,
    build_evidence_audit,
    build_expectation_template,
    compare_evidence_symmetrically,
    detect_confirmation_feedback,
    encode_evidence_with_provenance,
    initialize_belief_state,
    interpret_ambiguous_evidence,
    readout_evidence_for_judgment,
    retrieve_belief_related_memories,
    select_information_sources,
    update_belief_confidence,
    validate_point21_stages,
)
from .registry import OPERATION_REGISTRY


_POPULATION_DRIVES: tuple[Mapping[str, float], ...] = (
    {"prefrontal_prior_expectation": 0.75, "hippocampal_context_binding": 0.45,
     "prefrontal_control_monitoring": 0.35},
    {"prefrontal_prior_expectation": 0.85, "temporal_semantic_associations": 0.55,
     "frontoparietal_attention": 0.50},
    {"prefrontal_prior_expectation": 0.55, "prefrontal_control_monitoring": 0.75,
     "salience_valuation": 0.35},
    {"frontoparietal_attention": 0.85, "prefrontal_control_monitoring": 0.65,
     "salience_valuation": 0.50},
    {"hippocampal_context_binding": 0.85, "temporal_semantic_associations": 0.65,
     "frontoparietal_attention": 0.40},
    {"temporal_semantic_associations": 0.80, "prefrontal_prior_expectation": 0.60,
     "prefrontal_control_monitoring": 0.55},
    {"prefrontal_control_monitoring": 0.70, "hippocampal_context_binding": 0.55,
     "metacognitive_conflict_monitoring": 0.50},
    {"parietal_evidence_readout": 0.85, "prefrontal_control_monitoring": 0.75,
     "metacognitive_conflict_monitoring": 0.45},
    {"parietal_evidence_readout": 0.90, "prefrontal_prior_expectation": 0.55,
     "prefrontal_control_monitoring": 0.65},
    {"prefrontal_prior_expectation": 0.55, "prefrontal_control_monitoring": 0.80,
     "metacognitive_conflict_monitoring": 0.75},
    {"hippocampal_context_binding": 0.90, "temporal_semantic_associations": 0.70,
     "prefrontal_prior_expectation": 0.55},
    {"metacognitive_conflict_monitoring": 0.90, "prefrontal_control_monitoring": 0.80,
     "salience_valuation": 0.50},
    {"prefrontal_control_monitoring": 0.95, "frontoparietal_attention": 0.75,
     "metacognitive_conflict_monitoring": 0.80},
    {"metacognitive_conflict_monitoring": 0.90, "parietal_evidence_readout": 0.75,
     "prefrontal_control_monitoring": 0.85},
)


def _append_operation(
    trace: list[dict[str, Any]],
    index: int,
    functional_output: Mapping[str, Any],
    biological_output: Mapping[str, Any],
) -> None:
    """Append an operation trace aligned with the source-ordered registry."""
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


class Point21ConfirmationBiasProcessor:
    """Coordinate belief state, information selection, evidence evaluation, and corrective control.

    Confirmation bias is modeled as a set of separable process parameters. The processor can
    run in an accuracy-first mode that removes the modelled congruence preferences while retaining
    prior beliefs and source-reliability weighting.
    """

    def __init__(
        self,
        config: Point21Config | None = None,
        state: Point21State | None = None,
        biological: ConfirmationBiasNetwork | None = None,
    ) -> None:
        """Initialize configuration, persistent state, and the optional biological reference."""
        self.config = config or Point21Config()
        self.state = state or Point21State()
        self.biological = biological or ConfirmationBiasNetwork(self.config)
        self.process_count = 0

    def process(
        self,
        request: Point21Request,
        state: Point21State | None = None,
    ) -> Point21Result:
        """Execute the complete fourteen-operation confirmation-bias algorithm."""
        if not isinstance(request, Point21Request):
            raise TypeError("request must be a Point21Request")
        current = state if state is not None else self.state
        config = self.config
        network = self.biological if config.simulate_biology else None
        trace: list[dict[str, Any]] = []
        self.process_count += 1
        current.run_count += 1

        def biological(index: int, salience: float = 0.0, conflict: float = 0.0) -> dict[str, Any]:
            """Advance the reference network or document that its track is disabled."""
            if network is None:
                return {
                    "model": "disabled",
                    "operation_id": OPERATION_REGISTRY[index].operation_id,
                    "scope": "Biological-reference simulation disabled by configuration.",
                }
            return network.advance(
                OPERATION_REGISTRY[index].operation_id,
                _POPULATION_DRIVES[index],
                salience=salience,
                conflict=conflict,
            )

        # 1. Establish the current belief state and retain its provenance.
        belief_state = initialize_belief_state(request, current)
        prior = dict(belief_state["beliefs"])
        _append_operation(trace, 0, belief_state, biological(0))

        # 2. Build predictions for the focal and alternative hypotheses.
        expectations = build_expectation_template(request, prior)
        _append_operation(trace, 1, expectations, biological(1))

        # 3. Select sources. In accuracy-first mode, congruence and challenge-cost preferences
        # are disabled; diagnostic value and reliability remain active.
        selection = select_information_sources(request, prior, config)
        _append_operation(trace, 2, selection, biological(2))

        # 4. Allocate attention among evidence from selected sources.
        attention = allocate_attention(
            request, selection["selected_source_ids"], prior, config,
        )
        attended_strength = max(
            (row["attention_strength"] for row in attention["attended_evidence"]),
            default=0.0,
        )
        _append_operation(trace, 3, attention, biological(3, salience=attended_strength))

        # 5. Encode content without losing the source, context, or evidence direction.
        encoding = encode_evidence_with_provenance(request, attention, current, prior)
        _append_operation(trace, 4, encoding, biological(4))

        # 6. Generate interpretations for each item; prior influence stays explicit.
        interpretations = interpret_ambiguous_evidence(
            request, encoding, prior, config,
        )
        _append_operation(trace, 5, interpretations, biological(5))

        # 7. Assess evidence quality and dependence independently of belief congruence.
        quality = assess_evidence_quality(request, attention, selection, config)
        _append_operation(trace, 6, quality, biological(6))

        # 8. Place all hypotheses on the same centered log-likelihood scale.
        comparison = compare_evidence_symmetrically(request, quality)
        _append_operation(trace, 7, comparison, biological(7))

        # 9. Read out evidence for a judgment, exposing any congruence-dependent gate.
        readout = readout_evidence_for_judgment(
            request, prior, comparison, config,
        )
        focal = _focus_id_for_result(request, prior)
        focal_probability = readout["evidence_posterior"].get(focal, 0.0)
        _append_operation(
            trace, 8, readout,
            biological(8, conflict=abs(focal_probability - prior.get(focal, 0.0))),
        )

        # 10. Update belief and confidence with a bounded update rate.
        update = update_belief_confidence(
            request, prior, readout["evidence_posterior"], current, config,
        )
        updated_beliefs = dict(update["updated_beliefs"])
        _append_operation(
            trace, 9, update,
            biological(9, conflict=abs(update["confidence"] - max(prior.values()))),
        )

        # 11. Retrieve memories using present cues, retaining their original source tags/content.
        memory = retrieve_belief_related_memories(
            request, current, updated_beliefs, config,
        )
        _append_operation(
            trace, 10, memory,
            biological(10, salience=float(request.context.get("emotional_salience", 0.0))),
        )

        # 12. Detect stage-specific feedback rather than applying one global bias label.
        feedback = detect_confirmation_feedback(
            selection, attention, interpretations, readout, memory,
        )
        _append_operation(
            trace, 11, feedback,
            biological(11, conflict=clamp_conflict(feedback["reinforcing_stage_count"])),
        )

        # 13. Generate corrective tests and rank sources by approximate expected information gain.
        corrective = apply_corrective_control(request, updated_beliefs, config)
        _append_operation(
            trace, 12, corrective,
            biological(12, conflict=0.85 if corrective["ranked_discriminating_sources"] else 0.0),
        )

        # 14. Validate each modeled stage independently and preserve the complete evidence audit.
        validation = validate_point21_stages(
            request, selection, attention, interpretations, quality, readout, update, memory,
        )
        audit = build_evidence_audit(
            request, selection, attention, quality, comparison, readout,
        )
        validation["evidence_audit_count"] = len(audit)
        # The validation record describes operation 14 itself, which is appended
        # immediately below. Include that pending final record in the order check.
        completed_and_pending_ids = [
            row["operation_id"] for row in trace
        ] + [OPERATION_REGISTRY[13].operation_id]
        expected_ids = [spec.operation_id for spec in OPERATION_REGISTRY]
        validation["operation_order_verified"] = (
            completed_and_pending_ids == expected_ids
        )
        _append_operation(
            trace, 13, validation,
            biological(13, conflict=min(1.0, len(validation["failure_signatures"]) / 4.0)),
        )

        current.audit_history.append({
            "claim_id": request.claim_id,
            "beliefs": dict(updated_beliefs),
            "confidence": update["confidence"],
            "failure_signatures": list(validation["failure_signatures"]),
            "operation_count": len(trace),
        })
        biological_snapshot = network.snapshot() if network else {
            "model": "disabled",
            "operation_count": 0,
            "scientific_limitations": ["Biological-reference simulation disabled by configuration."],
        }
        result = Point21Result(
            claim_id=request.claim_id,
            beliefs=updated_beliefs,
            confidence=float(update["confidence"]),
            evidence_audit=tuple(audit),
            source_selection=selection,
            attention=attention,
            interpretations=interpretations,
            evidence_quality=quality,
            evidence_readout=readout,
            memory_retrieval=memory,
            feedback_loop=feedback,
            corrective_control=corrective,
            validation=validation,
            biological=biological_snapshot,
            operation_trace=tuple(trace),
            final_state=current.to_dict(),
            metadata={
                "model": "ORE Point 21 confirmation bias and belief-consistent memory",
                "source_sequence_steps": 14,
                "hypothesis_count": len(request.hypotheses),
                "source_count": len(request.sources),
                "evidence_item_count": len(request.evidence),
                "selected_source_count": len(selection["selected_source_ids"]),
                "biological_track_enabled": network is not None,
                "accuracy_first": config.accuracy_first,
                "prior_expectations_separated_from_observations": True,
                "evidence_quality_separated_from_belief_congruence": True,
                "memory_source_provenance_preserved": True,
                "scientific_status": "computational synthesis; not a validated neural simulator",
            },
        )
        return result


def _focus_id_for_result(request: Point21Request, beliefs: Mapping[str, float]) -> str:
    """Return the focal hypothesis ID used for update diagnostics."""
    return request.focus_hypothesis_id or max(beliefs, key=beliefs.get)


def clamp_conflict(stage_count: int) -> float:
    """Convert a count of feedback-loop warning stages to a bounded conflict drive."""
    return max(0.0, min(1.0, float(stage_count) / 4.0))


def register_annotations(
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Register every Point 21 class, function, and method in the shared annotation catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    catalogue = sync_repository(
        LoggerDatabase(target), root,
        include_paths={"ore_confirmation_bias"},
    )
    return {
        **catalogue,
        "database_path": str(target),
        "scope": "ore_confirmation_bias",
        "annotation_table": "annotations",
    }


def record_result_to_sqlite(
    result: Point21Result,
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist the fourteen operation outputs and sync Point 21 symbols into shared SQLite."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    if not isinstance(result, Point21Result):
        raise TypeError("result must be a Point21Result")
    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    database = LoggerDatabase(target)
    catalogue = sync_repository(
        database, root,
        include_paths={"ore_confirmation_bias"},
    )
    run_id = database.create_run(
        script_path="ore_confirmation_bias/point21",
        module_name="ore_confirmation_bias.pipeline",
        entry_kind="confirmation_bias_point21",
        working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id,
                sequence_no=sequence,
                event_kind="operation",
                source_path="ore_confirmation_bias/pipeline.py",
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
            run_id,
            "completed",
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
    except Exception as exc:
        database.finish_run(
            run_id,
            "failed",
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
