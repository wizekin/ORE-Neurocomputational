"""Seventeen-operation Point 22 pipeline and shared ORE SQLite integration."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

from .biology import CognitiveHeuristicsNetwork
from .contracts import JudgmentProblem, Point22Config, Point22Result, Point22State
from .functional import (
    activate_knowledge,
    adjust_anchor,
    availability_heuristic,
    bayesian_base_rate_check,
    build_intuitive_judgment,
    check_correlation_causation,
    detect_reasoning_conflict,
    evaluate_ecological_fit,
    estimate_sample_uncertainty,
    learn_from_feedback,
    make_final_decision,
    monitor_fluency,
    receive_problem,
    representativeness_heuristic,
    select_heuristic_strategy,
    validate_point22,
    verify_judgment,
    weight_salience_and_urgency,
)
from .registry import OPERATION_REGISTRY


_POPULATION_DRIVES: tuple[Mapping[str, float], ...] = (
    {"sensory_association": 0.65, "prefrontal_integration": 0.70},
    {"episodic_memory": 0.85, "semantic_prototypes": 0.65},
    {"prefrontal_integration": 0.70, "vm_pfc_valuation": 0.55},
    {"episodic_memory": 0.85, "sensory_association": 0.60},
    {"semantic_prototypes": 0.90, "sensory_association": 0.65},
    {"prefrontal_integration": 0.85, "lateral_prefrontal_control": 0.60},
    {"parietal_magnitude": 0.90, "prefrontal_integration": 0.50},
    {"metacognitive_confidence": 0.80, "episodic_memory": 0.45},
    {"prefrontal_integration": 0.70, "anterior_cingulate_monitoring": 0.60},
    {"prefrontal_integration": 0.75, "semantic_prototypes": 0.55},
    {"amygdala_insula_salience": 0.85, "vm_pfc_valuation": 0.75},
    {"vm_pfc_valuation": 0.75, "parietal_magnitude": 0.60},
    {"anterior_cingulate_monitoring": 0.90, "metacognitive_confidence": 0.75},
    {"lateral_prefrontal_control": 0.90, "prefrontal_integration": 0.85},
    {"corticostriatal_learning": 0.70, "vm_pfc_valuation": 0.65},
    {"corticostriatal_learning": 0.90, "episodic_memory": 0.75},
    {"metacognitive_confidence": 0.85, "lateral_prefrontal_control": 0.85},
)


def _append_operation(
    trace: list[dict[str, Any]],
    index: int,
    functional_output: Mapping[str, Any],
    biological_output: Mapping[str, Any],
) -> None:
    """Append one completed row, aligned to the seventeen-operation registry."""
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


class Point22CognitiveHeuristicsProcessor:
    """Coordinate rapid heuristics, explicit verification, ecological evaluation, and feedback."""

    def __init__(
        self,
        config: Point22Config | None = None,
        state: Point22State | None = None,
        biological: CognitiveHeuristicsNetwork | None = None,
    ) -> None:
        """Initialize configuration, reusable learning state, and an optional biological track."""
        self.config = config or Point22Config()
        self.state = state or Point22State()
        self.biological = biological or CognitiveHeuristicsNetwork(
            enabled=self.config.simulate_biology
        )
        self.process_count = 0

    def process(
        self,
        problem: JudgmentProblem,
        state: Point22State | None = None,
    ) -> Point22Result:
        """Execute all seventeen source-ordered operations for one judgment problem."""
        if not isinstance(problem, JudgmentProblem):
            raise TypeError("problem must be a JudgmentProblem")
        current = state if state is not None else self.state
        config = self.config
        network = self.biological if config.simulate_biology else None
        current.run_count += 1
        self.process_count += 1
        trace: list[dict[str, Any]] = []

        from .functional import resolve_categories
        categories = resolve_categories(problem)

        def biological(index: int, salience: float = 0.0, conflict: float = 0.0) -> dict[str, Any]:
            """Advance population-rate reference or emit a disabled-track record."""
            if network is None:
                return {
                    "model": "disabled",
                    "operation_id": OPERATION_REGISTRY[index].operation_id,
                    "scope": "Biological-reference track disabled by configuration.",
                }
            return network.advance(
                OPERATION_REGISTRY[index].operation_id,
                _POPULATION_DRIVES[index],
                salience=salience,
                conflict=conflict,
            )

        # 1. Receive the task and establish its output type and checking requirements.
        task = receive_problem(problem, categories)
        _append_operation(trace, 0, task, biological(0))

        # 2. Retrieve memories and knowledge, keeping accessibility separate from evidence.
        knowledge = activate_knowledge(problem)
        _append_operation(trace, 1, knowledge, biological(1))

        # 3. Select candidate shortcuts and expose their environmental assumptions.
        strategy = select_heuristic_strategy(problem, knowledge, config)
        _append_operation(trace, 2, strategy, biological(2))

        # 4. Estimate accessibility-based support from retrieved examples.
        availability = availability_heuristic(problem, knowledge, categories)
        _append_operation(trace, 3, availability, biological(3, salience=availability["availability_score"]))

        # 5. Compute resemblance to category prototypes.
        representativeness = representativeness_heuristic(problem, categories)
        _append_operation(trace, 4, representativeness, biological(4))

        # 6. Apply the denominator check and combine priors with supplied likelihoods.
        bayes = bayesian_base_rate_check(problem, categories)
        _append_operation(trace, 5, bayes, biological(5))

        # 7. Quantify the degree to which an initial numeric anchor remains influential.
        anchor = adjust_anchor(problem, config)
        _append_operation(trace, 6, anchor, biological(6))

        # 8. Separate processing ease from evidence and independent corroboration.
        fluency = monitor_fluency(problem)
        _append_operation(trace, 7, fluency, biological(7, salience=fluency["fluency"]))

        # 9. Calculate uncertainty from sample size and supplied sampling dispersion.
        sampling = estimate_sample_uncertainty(problem, config)
        _append_operation(trace, 8, sampling, biological(8, conflict=0.4 if sampling.get("warnings") else 0.0))

        # 10. Audit association, temporal order, confounding, reverse causality, and selection.
        causal = check_correlation_causation(problem)
        _append_operation(trace, 9, causal, biological(9, conflict=0.7 if causal["warnings"] else 0.0))

        # 11. Model salience, utility, urgency, and the priority of deliberate checking.
        affect = weight_salience_and_urgency(problem, config)
        _append_operation(trace, 10, affect, biological(10, salience=problem.emotional_salience))

        # 12. Create an initial fast estimate using shortcut cues and current learned weights.
        intuitive = build_intuitive_judgment(
            problem, categories, availability, representativeness, bayes, fluency, current, config,
        )
        _append_operation(trace, 11, intuitive, biological(11, salience=problem.fluency))

        # 13. Detect disagreement, overconfidence, uncertainty, and high-consequence conditions.
        conflict = detect_reasoning_conflict(
            problem, intuitive, bayes, sampling, causal, config,
        )
        _append_operation(
            trace, 12, conflict,
            biological(12, conflict=conflict["total_variation_distance"]),
        )

        # 14. Re-evaluate using statistical checks and the explicit likelihood model.
        verified = verify_judgment(problem, categories, bayes, sampling, causal)
        _append_operation(trace, 13, verified, biological(13, conflict=conflict["total_variation_distance"]))

        # 15. Evaluate cue validity and stability in the environment where the rule will be used.
        ecological = evaluate_ecological_fit(problem, strategy)
        _append_operation(trace, 14, ecological, biological(14))

        # 16. Use known outcomes to record prediction error and cautiously update attributable cues.
        feedback = learn_from_feedback(problem, intuitive, verified, current, config)
        _append_operation(trace, 15, feedback, biological(15))

        # 17. Store context, choose by posterior/expected loss, and expose whether to defer.
        decision = make_final_decision(problem, verified, ecological, conflict, feedback, config)
        _append_operation(trace, 16, decision, biological(16, conflict=1.0 if decision["review_required"] else 0.0))

        validation = validate_point22(trace, verified["probabilities"], verified["confidence"], categories)
        trace[-1]["functional_output"]["validation"] = validation
        current.prediction_history = current.prediction_history[-500:]
        biological_snapshot = network.snapshot() if network else {
            "model": "disabled",
            "operation_count": 0,
            "scientific_limitations": ["Biological-reference track disabled by configuration."],
        }
        warnings = list(dict.fromkeys(
            list(strategy["warnings"])
            + list(fluency["warnings"])
            + list(sampling.get("warnings", []))
            + list(causal["warnings"] if problem.target_type == "cause" else [])
            + list(conflict["warnings"])
            + list(verified["warnings"])
        ))
        return Point22Result(
            problem_id=problem.problem_id,
            initial_answer=intuitive["answer"],
            verified_answer=verified["answer"],
            probabilities=dict(verified["probabilities"]),
            confidence=float(verified["confidence"]),
            heuristic_used=tuple(strategy["selected_heuristics"]),
            warnings=tuple(warnings),
            assumptions=tuple(verified["assumptions"]),
            diagnostics={
                "conflict": dict(conflict),
                "availability": dict(availability),
                "representativeness": dict(representativeness),
                "fluency": dict(fluency),
                "affect_and_urgency": dict(affect),
                "validation": validation,
            },
            sampling=dict(sampling),
            causal_check=dict(causal),
            ecological_fit=dict(ecological),
            feedback=dict(feedback),
            decision=dict(decision),
            biological=biological_snapshot,
            operation_trace=tuple(trace),
            final_state=current.to_dict(),
            metadata={
                "model": "ORE Point 22 cognitive heuristics and systematic reasoning errors",
                "source_sequence_steps": 17,
                "category_count": len(categories),
                "example_count": len(problem.examples),
                "biological_track_enabled": network is not None,
                "intuitive_and_verified_answers_separated": True,
                "retrieval_fluency_separated_from_evidence": True,
                "similarity_separated_from_probability": True,
                "correlation_separated_from_causation": True,
                "feedback_updates_require_attribution": True,
                "scientific_status": "computational synthesis; not a validated neural simulator",
            },
        )


def register_annotations(
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Register Point 22 classes, functions, and methods in the shared annotation catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    catalogue = sync_repository(
        LoggerDatabase(target), root,
        include_paths={"ore_cognitive_heuristics"},
    )
    return {
        **catalogue,
        "database_path": str(target),
        "scope": "ore_cognitive_heuristics",
        "annotation_table": "annotations",
    }


def record_result_to_sqlite(
    result: Point22Result,
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist all seventeen operation traces and index every package symbol in SQLite."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    if not isinstance(result, Point22Result):
        raise TypeError("result must be a Point22Result")
    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    database = LoggerDatabase(target)
    catalogue = sync_repository(database, root, include_paths={"ore_cognitive_heuristics"})
    run_id = database.create_run(
        script_path="ore_cognitive_heuristics/point22",
        module_name="ore_cognitive_heuristics.pipeline",
        entry_kind="cognitive_heuristics_point22",
        working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id,
                sequence_no=sequence,
                event_kind="operation",
                source_path="ore_cognitive_heuristics/pipeline.py",
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
