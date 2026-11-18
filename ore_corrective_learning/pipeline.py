"""Twenty-operation Point 24 corrective-learning orchestration and SQLite persistence."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping

from .biology import CorrectiveLearningNetwork
from .contracts import (
    CorrectionRequest, CorrectiveConfig, CorrectiveResult, CorrectiveState,
)
from .functional import (
    assess_evidence, bayesian_update, build_correction, build_explanation,
    choose_information_action, classify_error, consolidate_state, decide_stopping,
    detect_mismatch, detect_regression, generate_predictions,
    prior_distribution, retrieval_feedback, score_retrieval, score_transfer,
    update_components, validate_result, brier_score,
)
from .registry import OPERATION_REGISTRY


_DRIVES = (
    {"sensory_association": 0.45, "frontotemporal_language": 0.30},
    {"sensory_association": 0.35, "semantic_relational": 0.65},
    {"medial_frontal_monitoring": 0.75, "locus_coeruleus_noradrenergic": 0.45},
    {"prefrontal_control": 0.65, "metacognitive_monitoring": 0.45},
    {"prefrontal_control": 0.75, "frontotemporal_language": 0.30},
    {"prefrontal_control": 0.65, "frontotemporal_language": 0.35},
    {"prefrontal_control": 0.65, "metacognitive_monitoring": 0.45},
    {"hippocampal_retrieval": 0.70, "prefrontal_control": 0.55},
    {"semantic_relational": 0.65, "frontotemporal_language": 0.45},
    {"semantic_relational": 0.70, "hippocampal_retrieval": 0.45},
    {"distributed_consolidation": 0.60, "semantic_relational": 0.55},
    {"semantic_relational": 0.75, "frontotemporal_language": 0.45},
    {"hippocampal_retrieval": 0.75, "metacognitive_monitoring": 0.50},
    {"medial_frontal_monitoring": 0.60, "prefrontal_control": 0.60},
    {"hippocampal_retrieval": 0.65, "distributed_consolidation": 0.70},
    {"metacognitive_monitoring": 0.75, "prefrontal_control": 0.60},
    {"distributed_consolidation": 0.80, "hippocampal_retrieval": 0.55},
    {"hippocampal_retrieval": 0.70, "metacognitive_monitoring": 0.55},
    {"prefrontal_control": 0.80, "metacognitive_monitoring": 0.65},
    {"distributed_consolidation": 0.55, "prefrontal_control": 0.45},
)


def _append(trace: list[dict[str, Any]], index: int, output: Mapping[str, Any], biology: Mapping[str, Any]) -> None:
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
        "functional_output": dict(output),
        "biological_output": dict(biology),
    })


class Point24CorrectiveLearningProcessor:
    """Run a feedback loop over caller-supplied hypotheses and auditable evidence.

    This is a configurable research prototype, not an autonomous fact checker. It does
    not browse for evidence, invent candidate explanations, or assume new information
    is correct. Callers must provide evidence likelihoods and source-provenance estimates.
    """

    def __init__(
        self,
        config: CorrectiveConfig | None = None,
        state: CorrectiveState | None = None,
        biological: CorrectiveLearningNetwork | None = None,
    ) -> None:
        self.config = config or CorrectiveConfig()
        self.state = state or CorrectiveState()
        self.biological = biological or CorrectiveLearningNetwork(self.config.simulate_biology)

    def process(self, request: CorrectionRequest, state: CorrectiveState | None = None) -> CorrectiveResult:
        if not isinstance(request, CorrectionRequest):
            raise TypeError("request must be a CorrectionRequest")
        current = state if state is not None else self.state
        current.run_count += 1
        config = self.config
        network = self.biological if config.simulate_biology else None
        trace: list[dict[str, Any]] = []
        biology_rows: list[dict[str, Any]] = []
        mismatch_score = 0.0
        evidence_quality = 0.0

        def bio(index: int, *, mismatch: float = 0.0, feedback: float = 0.0, memory: float = 0.0) -> dict[str, Any]:
            if network is None:
                return {"model": "disabled", "operation_id": OPERATION_REGISTRY[index].operation_id}
            return network.advance(
                OPERATION_REGISTRY[index].operation_id, _DRIVES[index],
                mismatch=mismatch, feedback_quality=feedback, memory_load=memory,
            )

        # 1. Preserve the current knowledge state, source, context, and confidence.
        hypotheses = {item.hypothesis_id: item for item in request.hypotheses}
        priors = prior_distribution(request)
        initial = request.current_hypothesis_id
        prior_record = {
            "hypothesis_id": initial,
            "statement": hypotheses[initial].statement,
            "source": hypotheses[initial].source,
            "prior_probability": priors.get(initial, 0.0),
            "context": request.context,
            "reported_confidence": request.confidence_estimate,
            "evidence_separated_from_confidence": True,
        }
        _append(trace, 0, prior_record, bio(0, memory=0.25))

        # 2. Generate predictions from every candidate, and a prior-weighted prediction.
        prediction = generate_predictions(request, priors)
        _append(trace, 1, prediction, bio(1))

        # 3. Detect mismatch or a gap; no observed outcome means no inferred contradiction.
        mismatch = detect_mismatch(request, prediction, config)
        mismatch_score = float(mismatch["mismatch_score"])
        _append(trace, 2, mismatch, bio(2, mismatch=mismatch_score))

        # 4. Retain a distribution over possible error classes.
        classification = classify_error(request, mismatch)
        _append(trace, 3, {"error_class_probabilities": classification}, bio(3, mismatch=mismatch_score))

        # 5. Select infer, clarify, check, or test based on expected information gain.
        action = choose_information_action(request, priors, mismatch)
        _append(trace, 4, action, bio(4, mismatch=mismatch_score, memory=0.35))

        # 6. Acquire a clarification plan or mark unavailable information unresolved.
        acquisition = {
            "selected_action": action.get("action"),
            "selected_option": action.get("selected_option"),
            "available_evidence_count": len(request.evidence),
            "clarification_response_supplied": bool(request.evidence),
            "unresolved_without_external_response": bool(action.get("unresolved") and not request.evidence),
        }
        _append(trace, 5, acquisition, bio(5, mismatch=mismatch_score))

        # 7. Assess quality and collapse correlated reports to avoid double counting.
        assessment = assess_evidence(request.evidence, list(hypotheses))
        evidence_quality = float(assessment["quality"])
        _append(trace, 6, assessment, bio(6, feedback=evidence_quality))

        # 8. Reactivate the prior interpretation without re-labelling it as current truth.
        reactivation = {
            "retrieved_hypothesis_id": initial,
            "retrieved_statement": hypotheses[initial].statement,
            "source_provenance": hypotheses[initial].source,
            "status": "prior_interpretation",
            "is_current_truth": False,
            "retrieval_is_not_automatically_reinforcement": True,
        }
        _append(trace, 7, reactivation, bio(7, memory=0.65))

        # 9. Candidate generation is intentionally constrained to supplied hypotheses.
        candidate_generation = {
            "candidate_ids": list(hypotheses),
            "candidate_count": len(hypotheses),
            "new_candidates_invented": 0,
            "limitation": "A domain grammar or knowledge base must provide missing alternatives.",
        }
        _append(trace, 8, candidate_generation, bio(8))

        # 10. Compare hypotheses using one weighted contribution per independent evidence group.
        posterior = bayesian_update(priors, assessment)
        _append(trace, 9, {
            "posterior_probabilities": posterior,
            "posterior_entropy_nats": -sum(p * __import__("math").log(max(1e-12, p)) for p in posterior.values()),
            "normalization_verified": abs(sum(posterior.values()) - 1.0) < 1e-8,
        }, bio(9, feedback=evidence_quality))

        # 11. Update only specified components (or components attached to the selected hypothesis).
        component_updates = update_components(request, posterior, config)
        for name, item in component_updates["component_updates"].items():
            current.component_values[name] = item["updated"]
        _append(trace, 10, component_updates, bio(10, feedback=evidence_quality))

        # 12. Construct an explanatory relation and preserve scope/boundary conditions.
        selected_id = max(posterior, key=posterior.get)
        explanation = build_explanation(request, selected_id)
        _append(trace, 11, explanation, bio(11))

        # 13. Test retrieval without the explanation being visible.
        retrieval_target_id = request.verified_hypothesis_id or selected_id
        retrieval = score_retrieval(request, retrieval_target_id, config)
        _append(trace, 12, retrieval, bio(12, memory=0.75 if retrieval["attempted"] else 0.35))

        # 14. Give error-specific corrective feedback, never reinforcing the old answer as truth.
        retrieval_feedback_result = retrieval_feedback(retrieval, retrieval_target_id)
        _append(trace, 13, retrieval_feedback_result, bio(13, mismatch=float(retrieval_feedback_result.get("residual_error", False))))

        # 15. Test transfer to new examples and counterexamples.
        transfer = score_transfer(request, config.transfer_threshold)
        _append(trace, 14, transfer, bio(14, memory=0.55 if transfer["attempted"] else 0.20))

        # 16. Calibrate reported confidence against independently verified correctness when available.
        calibration = {"brier_score": None, "observation_count": len(current.calibration_observations)}
        if request.confidence_estimate is not None and request.verified_hypothesis_id is not None:
            outcome = float(selected_id == request.verified_hypothesis_id)
            current.calibration_observations.append({"confidence": request.confidence_estimate, "outcome": outcome})
            current.calibration_observations = current.calibration_observations[-1000:]
            calibration = {
                "current_confidence": request.confidence_estimate,
                "verified_correctness": outcome,
                "brier_score": brier_score(current.calibration_observations),
                "observation_count": len(current.calibration_observations),
                "confidence_not_used_as_evidence": True,
            }
        _append(trace, 15, calibration, bio(15))

        # 17. Consolidate the correction while retaining context and provenance.
        consolidation = consolidate_state(
            current, selected_id, initial, retrieval, posterior, request, config,
        )
        _append(trace, 16, consolidation, bio(16, memory=0.65))

        # 18. Monitor possible belief regression after delay and renewed activation of the old belief.
        regression = detect_regression(current, initial, selected_id, retrieval)
        _append(trace, 17, regression, bio(17, mismatch=regression["regression_risk"]))

        # 19. Close the loop or identify the earliest stage requiring another pass.
        stopping = decide_stopping(
            request, mismatch, posterior, assessment, retrieval, transfer, config,
        )
        _append(trace, 18, stopping, bio(18, mismatch=regression["regression_risk"], memory=0.65))

        # 20. Persist provenance and validate trace structure, probability mass, and uncertainty.
        unresolved = []
        if not mismatch["observation_available"]:
            unresolved.append("no_observed_outcome_or_numeric_measurement")
        if not request.evidence:
            unresolved.append("no_corrective_evidence_supplied")
        if action.get("unresolved"):
            unresolved.append("clarification_or_independent_check_required")
        if stopping["status"] == "uncertain":
            unresolved.append("correction_not_yet_supported_by_all_required_checks")
        if request.verified_hypothesis_id is None:
            unresolved.append("no_independent_ground_truth_for_accuracy_verification")
        current.correction_history.append({
            "request_id": request.request_id,
            "initial_hypothesis_id": initial,
            "selected_hypothesis_id": selected_id,
            "status": stopping["status"],
            "posterior_probabilities": dict(posterior),
            "evidence_quality": evidence_quality,
            "unresolved_questions": list(unresolved),
        })
        current.correction_history = current.correction_history[-500:]
        validation = validate_result(request, posterior, trace)
        _append(trace, 19, {
            "validation": validation,
            "stored_history_count": len(current.correction_history),
            "unresolved_questions": list(unresolved),
            "provenance_retained": True,
        }, bio(19, feedback=evidence_quality))
        selected_id = max(posterior, key=posterior.get)
        validation = validate_result(request, posterior, trace)
        # Validation is repeated after the final operation so the published trace is fully covered.
        trace[-1]["functional_output"]["validation"] = validation
        biological_snapshot = network.snapshot() if network else {
            "model": "disabled", "operation_count": 0,
            "limitations": ["Biological-reference track disabled by configuration."],
        }
        return CorrectiveResult(
            request_id=request.request_id,
            initial_hypothesis_id=initial,
            selected_hypothesis_id=selected_id,
            posterior_probabilities=dict(posterior),
            prediction=prediction,
            mismatch=mismatch,
            error_classification=classification,
            information_action=action,
            evidence_assessment=assessment,
            prior_interpretation_record=prior_record,
            correction=build_correction(request, priors, posterior),
            component_updates=component_updates,
            explanation_model=explanation,
            retrieval=retrieval,
            retrieval_feedback=retrieval_feedback_result,
            transfer=transfer,
            calibration=calibration,
            consolidation=consolidation,
            regression=regression,
            stopping=stopping,
            unresolved_questions=tuple(unresolved),
            operation_trace=tuple(trace),
            biological=biological_snapshot,
            final_state=current.to_dict(),
            metadata={
                "model": "ORE Point 24 feedback and misconception correction",
                "source_sequence_steps": 20,
                "operation_count": len(trace),
                "biological_track_enabled": network is not None,
                "evidence_is_caller_supplied": True,
                "source_independence_is_caller_supplied": True,
                "candidate_generation_is_bounded": True,
                "confidence_is_not_evidence": True,
                "scientific_status": "computational synthesis; not a validated neural model or autonomous fact checker",
            },
        )


def register_annotations(
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Register package symbols in the shared SQLite annotation catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    catalogue = sync_repository(LoggerDatabase(target), root, include_paths={"ore_corrective_learning"})
    return {
        **catalogue, "database_path": str(target),
        "scope": "ore_corrective_learning", "annotation_table": "annotations",
    }


def record_result_to_sqlite(
    result: CorrectiveResult,
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist all twenty operation outputs as JSON-safe events in the shared logger."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    if not isinstance(result, CorrectiveResult):
        raise TypeError("result must be a CorrectiveResult")
    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    database = LoggerDatabase(target)
    catalogue = sync_repository(database, root, include_paths={"ore_corrective_learning"})
    run_id = database.create_run(
        script_path="ore_corrective_learning/point24",
        module_name="ore_corrective_learning.pipeline",
        entry_kind="corrective_learning_point24",
        working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id,
                sequence_no=sequence,
                event_kind="operation",
                source_path="ore_corrective_learning/pipeline.py",
                elapsed_ms=0.0,
                metadata=json.loads(json.dumps({
                    "operation_id": item["operation_id"],
                    "operation": item["operation"],
                    "status": item["status"],
                    "mathematical_form": item["mathematical_form"],
                    "biological_reference": item["biological_reference"],
                    "evidence_level": item["evidence_level"],
                    "functional_output": item["functional_output"],
                    "biological_output": item["biological_output"],
                }, default=str)),
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
        "run_id": run_id, "annotation_catalogue": catalogue,
        "database_path": str(target), "database_summary": database.summary(),
    }
