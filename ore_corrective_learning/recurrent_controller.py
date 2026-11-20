"""Fourteen-operation recurrent feedback controller for ORE's existing cognitive structures."""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from .contracts import (
    ClarificationOption,
    CorrectionRequest,
    CorrectiveConfig,
    CorrectiveState,
    FeedbackEvidence,
    KnowledgeHypothesis,
)
from .functional import assess_evidence, normalized_entropy
from .pipeline import Point24CorrectiveLearningProcessor
from .recurrent_contracts import (
    ConflictDimension,
    ConflictReport,
    CorrectionRecord,
    FeedbackAction,
    FeedbackEpisode,
    RecurrentFeedbackConfig,
    RecurrentFeedbackRequest,
    RecurrentFeedbackResult,
    _plain,
)
from .feedback_scheduler import FeedbackActionScheduler
from .module_adapters import ModuleAdapterResult, RecurrentModuleAdapters


RECURRENT_FEEDBACK_OPERATIONS: tuple[dict[str, str], ...] = (
    {"operation_id": "RF-001", "name": "preserve_initial_interpretation", "mathematical_form": "E_0=(x,H_0,C_0,q_0,provenance_0)", "purpose": "Freeze the original input and interpretation as an immutable comparison point."},
    {"operation_id": "RF-002", "name": "evaluate_semantic_compatibility", "mathematical_form": "C=sum_j w_j(1-s_j)/sum_j w_j; omit unavailable s_j", "purpose": "Aggregate only measured compatibility dimensions; missing data is not incompatibility."},
    {"operation_id": "RF-003", "name": "classify_conflict_and_route", "mathematical_form": "P(error_class|evidence,context)", "purpose": "Retain a distribution over error causes and identify affected cognitive modules."},
    {"operation_id": "RF-004", "name": "monitor_confirmation_bias", "mathematical_form": "Compare support(H_old) and contradiction(H_old) symmetrically", "purpose": "Keep the original interpretation as a revisable hypothesis."},
    {"operation_id": "RF-005", "name": "schedule_information_action", "mathematical_form": "U(a)=IG(a)-lambda*Cost(a)", "purpose": "Select a useful action within a single shared action and cost budget."},
    {"operation_id": "RF-006", "name": "retrieve_context_and_memory", "mathematical_form": "M=ReadOnlyRetrieve(q); memory is not verification", "purpose": "Request past context and examples while retaining source provenance."},
    {"operation_id": "RF-007", "name": "validate_new_feedback", "mathematical_form": "E_new subset FeedbackEvidence with known hypothesis IDs", "purpose": "Reject malformed, duplicated, or unsupported evidence without fabricating likelihoods."},
    {"operation_id": "RF-008", "name": "revise_hypothesis_distribution", "mathematical_form": "P_t(H_i) proportional to P_(t-1)(H_i)*product_g L(E_g|H_i)^w_g", "purpose": "Apply each new evidence batch once and collapse dependent sources through Point 24."},
    {"operation_id": "RF-009", "name": "reprocess_affected_structures", "mathematical_form": "O_new=ModuleAdapters(affected_modules,H_t,E_new)", "purpose": "Revisit only the structures implicated by the conflict."},
    {"operation_id": "RF-010", "name": "reevaluate_conflict", "mathematical_form": "DeltaC=C_old-C_new", "purpose": "Check whether new context or evidence resolved the original incompatibility."},
    {"operation_id": "RF-011", "name": "recalibrate_uncertainty", "mathematical_form": "H(P)=-sum_i P_i log P_i; Brier=(p-o)^2 when o is verified", "purpose": "Separate posterior concentration, evidence quality, confidence, and independently verified correctness."},
    {"operation_id": "RF-012", "name": "measure_progress_and_convergence", "mathematical_form": "DeltaP=max_i|P_t(H_i)-P_(t-1)(H_i)|; DeltaC=|C_t-C_(t-1)|", "purpose": "Detect repeated no-progress cycles without equating convergence with correctness."},
    {"operation_id": "RF-013", "name": "evaluate_stop_or_continue", "mathematical_form": "Stop only when support, uncertainty, conflict, and task-specific gates pass", "purpose": "Return a supported correction, independent verification, no-conflict result, or explicit uncertainty."},
    {"operation_id": "RF-014", "name": "store_and_validate_episode", "mathematical_form": "E_final=(E_0,conflicts,evidence,actions,H_final,confidence,status)", "purpose": "Preserve the complete correction history and audit trace without changing the active model."},
)


_DIMENSION_MODULES: dict[str, tuple[str, ...]] = {
    "perceptual": ("sensory_perception", "lexical_access"),
    "lexical": ("lexical_processor", "semantic_processor"),
    "semantic": ("semantic_processor",),
    "predicate_argument": ("semantic_processor",),
    "semantic_selection": ("semantic_processor",),
    "syntactic": ("syntax_processor", "semantic_processor"),
    "contextual": ("memory_retrieval", "discourse_processor", "semantic_processor"),
    "knowledge": ("memory_retrieval", "semantic_processor"),
    "reasoning": ("cognitive_heuristics", "cognitive_control"),
    "confidence": ("confirmation_bias", "cognitive_heuristics"),
    "source": ("confirmation_bias", "memory_retrieval"),
}

_ERROR_MODULES: dict[str, tuple[str, ...]] = {
    "perceptual_input": ("sensory_perception", "lexical_access"),
    "lexical_semantic": ("lexical_processor", "semantic_processor"),
    "syntactic_relational": ("syntax_processor", "semantic_processor"),
    "context": ("memory_retrieval", "discourse_processor"),
    "knowledge": ("memory_retrieval", "semantic_processor"),
    "inference": ("cognitive_heuristics", "semantic_processor"),
    "source_monitoring": ("confirmation_bias", "cognitive_control"),
    "confidence_calibration": ("cognitive_heuristics", "confirmation_bias"),
    "procedure": ("feedback_seeking", "cognitive_control"),
}


class RecurrentFeedbackController:
    """Coordinate repeated semantic conflict evaluation around Point 24's existing correction engine.

    This controller updates temporary hypothesis distributions and stores a correction episode.
    It never promotes knowledge, writes model parameters, or makes a retrieved memory true.
    """

    def __init__(
        self,
        config: RecurrentFeedbackConfig | None = None,
        corrective_config: CorrectiveConfig | None = None,
        state: CorrectiveState | None = None,
        corrective_processor: Point24CorrectiveLearningProcessor | None = None,
        module_adapters: RecurrentModuleAdapters | None = None,
        module_handlers: Mapping[str, Callable[[Mapping[str, Any]], Any]] | None = None,
        processors: Mapping[str, Any] | None = None,
        logger_database_path: str | Path | None = None,
        register_logger_annotations: bool = True,
        auto_log: bool = True,
    ) -> None:
        """Initialize the bounded controller and reuse a configured Point 24 processor/state."""
        self.config = config or RecurrentFeedbackConfig()
        self.corrective_config = corrective_config or CorrectiveConfig(simulate_biology=False)
        self.state = state or CorrectiveState()
        self.corrective_processor = corrective_processor or Point24CorrectiveLearningProcessor(
            self.corrective_config, self.state,
        )
        self.module_adapters = module_adapters or RecurrentModuleAdapters(module_handlers, processors)
        self.logger_database_path = logger_database_path
        self.auto_log = bool(auto_log)
        self.annotation_report = (
            register_recurrent_annotations(database_path=logger_database_path)
            if register_logger_annotations else {}
        )

    def _append_operation(
        self,
        trace: list[dict[str, Any]],
        operation_index: int,
        output: Mapping[str, Any],
        *,
        status: str = "completed",
        iteration: int = 0,
    ) -> None:
        """Append an operation row with stable registry identity and method annotation target."""
        specification = RECURRENT_FEEDBACK_OPERATIONS[operation_index]
        method = {
            0: "_snapshot_episode", 1: "_build_conflict_report",
            2: "_classify_and_route", 3: "_run_bias_monitor",
            4: "_schedule_actions", 5: "_run_memory_retrieval",
            6: "_validate_evidence", 7: "_apply_evidence_batch",
            8: "_reprocess_affected_modules", 9: "_reevaluate_conflict",
            10: "_recalibrate_confidence", 11: "_measure_progress",
            12: "_decide_status", 13: "_finalize_episode",
        }[operation_index]
        trace.append({
            "step": operation_index + 1,
            "operation_id": specification["operation_id"],
            "operation": specification["name"],
            "mathematical_form": specification["mathematical_form"],
            "status": status,
            "iteration": iteration,
            "method": f"RecurrentFeedbackController.{method}",
            "source_path": "ore_corrective_learning/recurrent_controller.py",
            "functional_output": _plain(output),
            "model_updated": False,
        })

    def _snapshot_episode(self, request: RecurrentFeedbackRequest) -> FeedbackEpisode:
        """Create the first feedback snapshot and never overwrite the initial interpretation."""
        correction = request.correction_request
        hypotheses = {item.hypothesis_id: item for item in correction.hypotheses}
        initial = hypotheses[correction.current_hypothesis_id]
        initial_probability = float(initial.prior)
        interpretation = dict(request.initial_interpretation) or {
            "interpretation_id": initial.hypothesis_id,
            "statement": initial.statement,
            "components": _plain(initial.components),
            "source": initial.source,
        }
        interpretation.setdefault("interpretation_id", initial.hypothesis_id)
        interpretation.setdefault("hypothesis_id", initial.hypothesis_id)
        interpretation.setdefault("statement", initial.statement)
        input_snapshot = dict(request.input_snapshot) or {
            "request_id": correction.request_id,
            "context": correction.context,
            "explanation": _plain(correction.explanation),
        }
        return FeedbackEpisode(
            episode_id=f"feedback_{uuid4().hex}",
            input_snapshot=input_snapshot,
            initial_interpretation=interpretation,
            current_interpretation=interpretation,
            alternative_hypotheses=tuple({
                "hypothesis_id": item.hypothesis_id,
                "statement": item.statement,
                "prior_probability": item.prior,
                "source": item.source,
            } for item in correction.hypotheses),
            conflict_history=[],
            evidence_history=[asdict(item) for item in correction.evidence],
            action_history=[],
            current_iteration=0,
            maximum_iterations=self.config.max_semantic_feedback_rounds,
            consecutive_no_progress=0,
            initial_confidence=float(correction.confidence_estimate if correction.confidence_estimate is not None else initial_probability),
            current_confidence=initial_probability,
            status="initialized",
            unresolved_questions=[],
        )

    def _dimensions_from_result(self, result: ModuleAdapterResult) -> tuple[ConflictDimension, ...]:
        """Extract only explicitly measured compatibility dimensions from a semantic evaluator."""
        raw = result.output.get("dimensions", result.output.get("compatibility_dimensions", ()))
        if isinstance(raw, ConflictDimension):
            raw = (raw,)
        dimensions = []
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            for item in raw:
                if isinstance(item, ConflictDimension):
                    dimensions.append(item)
                elif isinstance(item, Mapping):
                    try:
                        dimensions.append(ConflictDimension.from_mapping(item))
                    except (TypeError, ValueError, KeyError):
                        continue
        return tuple(dimensions)

    def _collect_dimensions(
        self,
        request: RecurrentFeedbackRequest,
        corrective_result: Any,
        semantic_result: ModuleAdapterResult | None,
    ) -> tuple[ConflictDimension, ...]:
        """Combine supplied signals, explicit semantic feedback, and available Point 24 mismatch."""
        dimensions = list(request.compatibility_dimensions)
        if semantic_result is not None:
            dimensions.extend(self._dimensions_from_result(semantic_result))
        mismatch = getattr(corrective_result, "mismatch", {})
        if isinstance(mismatch, Mapping) and mismatch.get("observation_available"):
            score = _bounded("mismatch_score", float(mismatch.get("mismatch_score", 0.0)))
            dimensions.append(ConflictDimension(
                dimension="observation_mismatch",
                compatibility_score=1.0 - score,
                weight=1.0,
                source="point24_observation_comparison",
                details={"mismatch": _plain(mismatch)},
            ))
        reported_confidence = request.correction_request.confidence_estimate
        if reported_confidence is not None and reported_confidence < self.config.minimum_confidence:
            dimensions.append(ConflictDimension(
                dimension="confidence_uncertainty",
                compatibility_score=reported_confidence,
                weight=0.5,
                source="reported_confidence_review",
                details={"confidence_is_not_evidence": True},
            ))
        # Repeated signals from the same source/dimension can inflate one signal; retain one measured item.
        unique: dict[tuple[str, str], ConflictDimension] = {}
        for item in dimensions:
            if item.compatibility_score is None or item.weight <= 0.0:
                continue
            key = (item.dimension, item.source)
            current = unique.get(key)
            if current is None or item.weight > current.weight:
                unique[key] = item
        return tuple(unique[key] for key in sorted(unique))

    def _build_conflict_report(
        self,
        request: RecurrentFeedbackRequest,
        corrective_result: Any,
        dimensions: Sequence[ConflictDimension],
        previous_score: float | None = None,
    ) -> ConflictReport:
        """Compute a weighted conflict score using only measured compatibility dimensions."""
        denominator = sum(item.weight for item in dimensions)
        score = (
            sum(item.weight * (1.0 - float(item.compatibility_score)) for item in dimensions) / denominator
            if denominator > 0.0 else 0.0
        )
        score = max(0.0, min(1.0, score))
        detected = score >= self.config.conflict_threshold or request.force_review
        primary = max(dimensions, key=lambda item: item.weight * (1.0 - float(item.compatibility_score)), default=None)
        conflict_type = primary.dimension if primary else "confidence_conflict" if request.force_review else "none"
        severity = "high" if score >= 0.70 else "medium" if score >= 0.45 else "low" if detected else "none"
        hypotheses = {item.hypothesis_id: item for item in request.correction_request.hypotheses}
        current = hypotheses[request.correction_request.current_hypothesis_id]
        classification = getattr(corrective_result, "error_classification", {})
        error_classes = sorted(
            classification,
            key=lambda key: float(classification.get(key, 0.0)),
            reverse=True,
        ) if isinstance(classification, Mapping) else []
        affected_modules = []
        for item in dimensions:
            affected_modules.extend(item.affected_modules)
            normalized = item.dimension.casefold()
            for key, values in _DIMENSION_MODULES.items():
                if key in normalized:
                    affected_modules.extend(values)
        for error_class in error_classes[:2]:
            affected_modules.extend(_ERROR_MODULES.get(error_class, ()))
        affected_modules = list(dict.fromkeys(affected_modules))
        affected_nodes = tuple(dict.fromkeys(
            node for item in dimensions for node in item.affected_nodes
        ))
        supporting = tuple(dict.fromkeys(
            evidence for item in dimensions for evidence in item.supporting_evidence
        ))
        contradicting = tuple(dict.fromkeys(
            evidence for item in dimensions for evidence in item.contradicting_evidence
        ))
        missing = tuple(dict.fromkeys(
            evidence for item in dimensions for evidence in item.missing_information
        ))
        actions = tuple(item.action_id for item in request.correction_request.clarification_options)
        return ConflictReport(
            conflict_id=f"conflict_{uuid4().hex}",
            interpretation_id=str(request.initial_interpretation.get("interpretation_id", current.hypothesis_id)),
            conflict_type=conflict_type,
            conflict_score=score,
            severity=severity,
            affected_semantic_nodes=affected_nodes,
            affected_modules=tuple(affected_modules),
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            missing_information=missing,
            suggested_feedback_actions=actions,
            dimensions=tuple(dimensions),
            detected=detected,
            status="requires_investigation" if detected else "no_conflict",
        )

    def _classify_and_route(
        self,
        report: ConflictReport,
        corrective_result: Any,
    ) -> dict[str, Any]:
        """Preserve Point 24's multi-cause error distribution and derive affected module routes."""
        classification = dict(getattr(corrective_result, "error_classification", {}) or {})
        modules = list(report.affected_modules)
        for error_class, probability in classification.items():
            if probability > 0.0:
                modules.extend(_ERROR_MODULES.get(error_class, ()))
        if report.detected:
            modules.extend(("confirmation_bias", "cognitive_heuristics"))
        return {
            "error_class_probabilities": classification,
            "affected_modules": list(dict.fromkeys(modules)),
            "diagnosis_is_probabilistic": True,
            "unsupported_single_cause_not_forced": True,
        }

    def _run_module(
        self,
        module_id: str,
        payload: Mapping[str, Any],
        module_inputs: Mapping[str, Any],
        diagnostics: list[dict[str, Any]],
    ) -> tuple[FeedbackEvidence, ...]:
        """Invoke one structure and retain its explicit availability, output, and evidence result."""
        result = self.module_adapters.invoke(module_id, payload, module_inputs)
        diagnostics.append(result.to_dict())
        return result.feedback_evidence

    def _run_bias_monitor(
        self,
        report: ConflictReport,
        request: RecurrentFeedbackRequest,
        diagnostics: list[dict[str, Any]],
    ) -> tuple[FeedbackEvidence, ...]:
        """Request symmetric evaluation from the existing confirmation-bias component."""
        return self._run_module(
            "confirmation_bias",
            {"conflict_report": report.to_dict(), "original_interpretation_is_revisable": True},
            request.module_inputs,
            diagnostics,
        )

    def _run_memory_retrieval(
        self,
        report: ConflictReport,
        request: RecurrentFeedbackRequest,
        diagnostics: list[dict[str, Any]],
    ) -> tuple[FeedbackEvidence, ...]:
        """Request read-only context/memory retrieval while keeping recalled items distinct from verified facts."""
        if not any(word in report.conflict_type.casefold() for word in ("context", "knowledge", "memory")):
            return ()
        return self._run_module(
            "memory_retrieval",
            {"conflict_report": report.to_dict(), "retrieval_is_not_verification": True},
            request.module_inputs,
            diagnostics,
        )

    def _schedule_actions(
        self,
        request: CorrectionRequest,
        posterior: Mapping[str, float],
        episode: FeedbackEpisode,
        scheduler: FeedbackActionScheduler,
    ) -> FeedbackAction | None:
        """Choose the highest information-gain action still eligible under the episode budget."""
        return scheduler.select(
            request,
            posterior,
            target=str(episode.current_interpretation.get("interpretation_id", request.current_hypothesis_id)),
            context=episode.input_snapshot,
            evidence_ids=tuple(sorted({
                str(item.get("evidence_id"))
                for item in episode.evidence_history
                if isinstance(item, Mapping) and item.get("evidence_id")
            })),
        )

    def _validate_evidence(
        self,
        evidence: Sequence[FeedbackEvidence],
        hypothesis_ids: Sequence[str],
        seen_ids: set[str],
    ) -> tuple[tuple[FeedbackEvidence, ...], list[dict[str, Any]]]:
        """Reject malformed, repeated, and out-of-domain evidence without synthesizing likelihoods."""
        accepted: list[FeedbackEvidence] = []
        rejected: list[dict[str, Any]] = []
        known = set(hypothesis_ids)
        for item in evidence:
            reason = None
            if not isinstance(item, FeedbackEvidence):
                reason = "not_a_Point24_FeedbackEvidence_instance"
            elif not item.likelihoods:
                reason = "missing_explicit_hypothesis_likelihoods"
            elif item.evidence_id in seen_ids:
                reason = "duplicate_evidence_id"
            elif set(item.likelihoods) - known:
                reason = "unknown_hypothesis_identifier"
            if reason:
                rejected.append({"evidence_id": getattr(item, "evidence_id", None), "reason": reason})
            else:
                accepted.append(item)
                seen_ids.add(item.evidence_id)
        return tuple(accepted), rejected

    def _request_with_prior(
        self,
        original: CorrectionRequest,
        posterior: Mapping[str, float],
        current_hypothesis_id: str,
        evidence: Sequence[FeedbackEvidence],
    ) -> CorrectionRequest:
        """Create the next Bayesian correction request with prior equal to the previous posterior."""
        hypotheses = tuple(
            replace(item, prior=float(posterior.get(item.hypothesis_id, 0.0)))
            for item in original.hypotheses
        )
        # Numerical posterior mass is normalized by Point 24; normalize once more for constructor tolerance.
        total = sum(item.prior for item in hypotheses)
        if total <= 0.0:
            raise ValueError("previous posterior has no probability mass")
        hypotheses = tuple(replace(item, prior=item.prior / total) for item in hypotheses)
        return replace(
            original,
            hypotheses=hypotheses,
            current_hypothesis_id=current_hypothesis_id,
            evidence=tuple(evidence),
        )

    def _apply_evidence_batch(
        self,
        original: CorrectionRequest,
        previous: Any,
        new_evidence: Sequence[FeedbackEvidence],
    ) -> Any:
        """Run Point 24 once on only the new evidence, using the previous posterior as the new prior."""
        next_request = self._request_with_prior(
            original,
            previous.posterior_probabilities,
            previous.selected_hypothesis_id,
            new_evidence,
        )
        return self.corrective_processor.process(next_request, self.state)

    def _reprocess_affected_modules(
        self,
        report: ConflictReport,
        interpretation: Mapping[str, Any],
        new_evidence: Sequence[FeedbackEvidence],
        request: RecurrentFeedbackRequest,
        diagnostics: list[dict[str, Any]],
    ) -> tuple[FeedbackEvidence, ...]:
        """Revisit only conflict-affected modules and collect only their explicit new evidence."""
        module_ids = list(report.affected_modules)
        # N9 is an explicit semantic revision option, not an automatic full-brain restart.
        configured_processors = getattr(self.module_adapters, "processors", {})
        if "n9_request" in request.module_inputs and (
            "semantic_reprocessing" in configured_processors or "n9" in configured_processors
        ):
            module_ids.append("semantic_reprocessing")
        payload = {
            "conflict_report": report.to_dict(),
            "revised_interpretation": dict(interpretation),
            "new_evidence_ids": [item.evidence_id for item in new_evidence],
            "affected_components": list(report.affected_semantic_nodes),
            "preserve_original_input": True,
            "model_learning_authorized": False,
        }
        results = self.module_adapters.reprocess(
            tuple(dict.fromkeys(module_ids)),
            payload,
            request.module_inputs,
        )
        evidence = []
        for result in results:
            diagnostics.append(result.to_dict())
            evidence.extend(result.feedback_evidence)
        return tuple(evidence)

    def _reevaluate_conflict(
        self,
        request: RecurrentFeedbackRequest,
        corrective_result: Any,
        prior_dimensions: Sequence[ConflictDimension],
        diagnostics: list[dict[str, Any]],
        previous_score: float,
    ) -> ConflictReport:
        """Ask the semantic evaluator for current compatibility and retain prior measured dimensions only as fallback."""
        result = self.module_adapters.invoke(
            "semantic_conflict",
            {
                "interpretation": {
                    "hypothesis_id": corrective_result.selected_hypothesis_id,
                    "statement": next(
                        item.statement for item in request.correction_request.hypotheses
                        if item.hypothesis_id == corrective_result.selected_hypothesis_id
                    ),
                    "posterior": dict(corrective_result.posterior_probabilities),
                },
                "prior_conflict_score": previous_score,
                "evidence_history": list(request.correction_request.evidence),
            },
            request.module_inputs,
        )
        diagnostics.append(result.to_dict())
        new_dimensions = self._dimensions_from_result(result)
        dimensions = new_dimensions if new_dimensions else tuple(prior_dimensions)
        # Replace the initial Point 24 mismatch indicator with the latest result, avoiding cumulative mismatch rows.
        dimensions = tuple(item for item in dimensions if item.source != "point24_observation_comparison")
        mismatch = corrective_result.mismatch
        if mismatch.get("observation_available"):
            dimensions += (ConflictDimension(
                dimension="observation_mismatch",
                compatibility_score=1.0 - _bounded("mismatch_score", mismatch["mismatch_score"]),
                weight=1.0,
                source="point24_observation_comparison",
            ),)
        return self._build_conflict_report(request, corrective_result, dimensions, previous_score)

    def _recalibrate_confidence(
        self,
        corrective_result: Any,
        evidence: Sequence[FeedbackEvidence],
        verified_hypothesis_id: str | None = None,
    ) -> dict[str, Any]:
        """Separate posterior entropy and source quality from independently verified correctness."""
        posterior = dict(corrective_result.posterior_probabilities)
        selected = corrective_result.selected_hypothesis_id
        assessment = assess_evidence(evidence, tuple(posterior))
        entropy_value = normalized_entropy(posterior)
        verified = verified_hypothesis_id is not None and selected == verified_hypothesis_id
        return {
            "selected_hypothesis_id": selected,
            "posterior_probabilities": posterior,
            "selected_posterior": float(posterior.get(selected, 0.0)),
            "normalized_entropy": entropy_value,
            "evidence_quality": float(assessment["quality"]),
            "independent_evidence_groups": int(assessment["independent_group_count"]),
            "brier_score": corrective_result.calibration.get("brier_score"),
            "verified_hypothesis_id": verified_hypothesis_id,
            "independent_verification_available": bool(verified),
            "confidence_is_not_evidence": True,
            "probability_concentration_is_not_accuracy": True,
        }

    def _measure_progress(
        self,
        previous_posterior: Mapping[str, float],
        current_posterior: Mapping[str, float],
        previous_conflict: float,
        current_conflict: float,
    ) -> dict[str, Any]:
        """Measure posterior movement and conflict movement independently."""
        keys = set(previous_posterior) | set(current_posterior)
        delta_probability = max((
            abs(float(current_posterior.get(key, 0.0)) - float(previous_posterior.get(key, 0.0)))
            for key in keys
        ), default=0.0)
        delta_conflict = abs(float(current_conflict) - float(previous_conflict))
        return {
            "delta_probability": delta_probability,
            "delta_conflict": delta_conflict,
            "progress_detected": max(delta_probability, delta_conflict) > self.config.convergence_threshold,
            "converged": delta_probability <= self.config.convergence_threshold and delta_conflict <= self.config.convergence_threshold,
            "convergence_does_not_imply_correctness": True,
        }

    def _decide_status(
        self,
        initial_id: str,
        corrective_result: Any,
        report: ConflictReport,
        calibration: Mapping[str, Any],
        *,
        no_action_available: bool,
        maximum_iterations_reached: bool,
        verified_hypothesis_id: str | None,
    ) -> str:
        """Accept only evidence-supported and conflict-resolved outcomes; otherwise abstain."""
        selected = corrective_result.selected_hypothesis_id
        quality = float(calibration["evidence_quality"])
        posterior = float(calibration["selected_posterior"])
        entropy_value = float(calibration["normalized_entropy"])
        enough_support = (
            bool(corrective_result.evidence_assessment.get("items"))
            and quality >= self.config.minimum_evidence_quality
            and posterior >= self.config.minimum_confidence
            and entropy_value <= self.corrective_config.uncertainty_threshold
            and report.conflict_score < self.config.conflict_threshold
        )
        externally_verified = verified_hypothesis_id is not None and selected == verified_hypothesis_id
        if enough_support and selected != initial_id and externally_verified:
            return "INDEPENDENTLY_VERIFIED"
        if enough_support and selected != initial_id:
            return "SUPPORTED_CORRECTION"
        if enough_support and selected == initial_id:
            return "NO_CORRECTION_NEEDED"
        if not report.detected and selected == initial_id:
            return "NO_CORRECTION_NEEDED"
        if maximum_iterations_reached:
            return "MAXIMUM_ITERATIONS_REACHED"
        if no_action_available:
            return "UNCERTAIN"
        return "UNCERTAIN"

    def _finalize_episode(
        self,
        episode: FeedbackEpisode,
        corrective_result: Any,
        status: str,
        all_evidence: Sequence[FeedbackEvidence],
        request: RecurrentFeedbackRequest,
        conflict_report: ConflictReport,
    ) -> FeedbackEpisode:
        """Finalize the correction record while preserving the initial reading and leaving models untouched."""
        selected = corrective_result.selected_hypothesis_id
        hypotheses = {item.hypothesis_id: item for item in request.correction_request.hypotheses}
        initial_id = str(episode.initial_interpretation.get("hypothesis_id", episode.initial_interpretation["interpretation_id"]))
        confidence = float(corrective_result.posterior_probabilities.get(selected, 0.0))
        previous_id = (
            initial_id if selected != initial_id
            else str(episode.current_interpretation.get("interpretation_id", initial_id))
        )
        previous_confidence = float(episode.initial_confidence)
        episode.current_interpretation = {
            "interpretation_id": selected,
            "statement": hypotheses[selected].statement,
            "components": _plain(hypotheses[selected].components),
            "posterior_probabilities": dict(corrective_result.posterior_probabilities),
            "source": hypotheses[selected].source,
            "supersedes_in_context": initial_id if selected != initial_id else None,
            "status": "provisional" if status in {"UNCERTAIN", "MAXIMUM_ITERATIONS_REACHED"} else "supported",
        }
        episode.current_confidence = confidence
        episode.status = status
        episode.correction_record = CorrectionRecord(
            episode_id=episode.episode_id,
            previous_interpretation_id=previous_id,
            revised_interpretation_id=selected,
            evidence_ids=tuple(item.evidence_id for item in all_evidence),
            explanation=dict(corrective_result.explanation_model),
            previous_confidence=previous_confidence,
            revised_confidence=confidence,
            correction_status=status,
            verification_status="independently_verified" if status == "INDEPENDENTLY_VERIFIED" else "not_independently_verified",
            affected_context={
                "conflict_id": conflict_report.conflict_id,
                "conflict_score": conflict_report.conflict_score,
                "initial_interpretation_id": initial_id,
                "old_interpretation_preserved": True,
            },
            model_updated=False,
        ).to_dict()
        alternatives = []
        for key, probability in sorted(
            corrective_result.posterior_probabilities.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            alternatives.append({
                "hypothesis_id": key,
                "statement": hypotheses[key].statement,
                "posterior_probability": float(probability),
                "is_selected": key == selected,
            })
        episode.alternative_hypotheses = tuple(alternatives)
        episode.unresolved_questions = list(dict.fromkeys(
            list(episode.unresolved_questions)
            + list(corrective_result.unresolved_questions)
            + list(conflict_report.missing_information)
        ))
        episode.evidence_history = [asdict(item) for item in all_evidence]
        return episode

    def process(self, request: RecurrentFeedbackRequest) -> RecurrentFeedbackResult:
        """Execute a bounded recurrent correction cycle and retain every provisional alternative."""
        if not isinstance(request, RecurrentFeedbackRequest):
            raise TypeError("request must be a RecurrentFeedbackRequest")
        started = time.perf_counter()
        operation_rows: dict[int, list[dict[str, Any]]] = {}
        trace: list[dict[str, Any]] = []

        def record_operation(
            operation_index: int,
            output: Mapping[str, Any],
            *,
            status: str = "completed",
            iteration: int = 0,
        ) -> None:
            """Collect repeated per-round outputs before building the fixed 14-row trace."""
            row_buffer: list[dict[str, Any]] = []
            self._append_operation(row_buffer, operation_index, output, status=status, iteration=iteration)
            operation_rows.setdefault(operation_index, []).append(row_buffer[0])

        module_diagnostics: list[dict[str, Any]] = []
        rejected_evidence: list[dict[str, Any]] = []
        action_count_before = 0
        correction_request = request.correction_request
        episode = self._snapshot_episode(request)
        record_operation(0, {
            "episode_id": episode.episode_id,
            "input_snapshot_preserved": True,
            "initial_interpretation_id": episode.initial_interpretation["interpretation_id"],
            "model_updated": False,
        })

        corrective_result = self.corrective_processor.process(correction_request, self.state)
        semantic_result = self.module_adapters.invoke(
            "semantic_conflict",
            {
                "interpretation": episode.initial_interpretation,
                "input_snapshot": episode.input_snapshot,
                "context": dict(request.context),
                "evaluate_as_potential_conflict": True,
            },
            request.module_inputs,
        )
        module_diagnostics.append(semantic_result.to_dict())
        dimensions = self._collect_dimensions(request, corrective_result, semantic_result)
        conflict_report = self._build_conflict_report(request, corrective_result, dimensions)
        episode.conflict_history.append(conflict_report.to_dict())
        record_operation(1, conflict_report.to_dict())

        classification = self._classify_and_route(conflict_report, corrective_result)
        record_operation(2, classification)
        all_evidence = list(correction_request.evidence)
        seen_ids = {item.evidence_id for item in all_evidence}
        current_posterior = dict(corrective_result.posterior_probabilities)
        current_result = corrective_result
        current_report = conflict_report
        initial_id = correction_request.current_hypothesis_id

        if not conflict_report.detected:
            calibration = self._recalibrate_confidence(current_result, all_evidence, correction_request.verified_hypothesis_id)
            record_operation(3, {"status": "skipped_no_conflict"})
            record_operation(4, {"status": "skipped_no_conflict"})
            record_operation(5, {"status": "skipped_no_conflict"})
            record_operation(6, {"accepted_evidence": 0, "rejected_evidence": []})
            record_operation(7, {"status": "skipped_no_conflict"})
            record_operation(8, {"status": "skipped_no_conflict"})
            record_operation(9, {"status": "skipped_no_conflict"})
            record_operation(10, calibration)
            progress = {"delta_probability": 0.0, "delta_conflict": 0.0, "progress_detected": False, "converged": True, "convergence_does_not_imply_correctness": True}
            record_operation(11, progress)
            status = "NO_CORRECTION_NEEDED"
            record_operation(12, {"status": status})
        else:
            monitor_modules = ["confirmation_bias", "cognitive_heuristics"]
            if any(word in conflict_report.conflict_type.casefold() for word in ("context", "knowledge")):
                monitor_modules.append("memory_retrieval")
            if conflict_report.missing_information:
                monitor_modules.append("curiosity_motivation")
            monitor_evidence = list(semantic_result.feedback_evidence)
            for module_id in dict.fromkeys(monitor_modules):
                monitor_evidence.extend(self._run_module(
                    module_id,
                    {"conflict_report": conflict_report.to_dict(), "error_classification": classification},
                    request.module_inputs,
                    module_diagnostics,
                ))
            record_operation(3, {
                "modules_invoked": monitor_modules,
                "evidence_returned": len(monitor_evidence),
                "evidence_will_be_validated_with_feedback": True,
            })
            record_operation(4, {
                "bias_monitor_invoked": "confirmation_bias" in monitor_modules,
                "old_interpretation_is_not_a_fact": True,
            })

            # Feedback-seeking and curiosity adapters may provide typed ClarificationOption objects.
            option_list = list(correction_request.clarification_options)
            feedback_result = self.module_adapters.invoke(
                "feedback_seeking",
                {"conflict_report": conflict_report.to_dict(), "information_gap": list(conflict_report.missing_information)},
                request.module_inputs,
            )
            module_diagnostics.append(feedback_result.to_dict())
            option_list.extend(
                item for item in feedback_result.clarification_options
                if isinstance(item, ClarificationOption)
            )
            unique_options = {item.action_id: item for item in option_list}
            action_request = replace(correction_request, clarification_options=tuple(unique_options.values()))
            scheduler = FeedbackActionScheduler(self.config)
            scheduler_evidence = list(monitor_evidence) + list(feedback_result.feedback_evidence)
            valid_scheduler_evidence, rejected = self._validate_evidence(
                scheduler_evidence, tuple(current_posterior), seen_ids,
            )
            rejected_evidence.extend(rejected)
            if valid_scheduler_evidence:
                old_posterior = dict(current_posterior)
                current_result = self._apply_evidence_batch(correction_request, current_result, valid_scheduler_evidence)
                current_posterior = dict(current_result.posterior_probabilities)
                all_evidence.extend(valid_scheduler_evidence)
                episode.evidence_history.extend(asdict(item) for item in valid_scheduler_evidence)
                episode.current_interpretation = {
                    "interpretation_id": current_result.selected_hypothesis_id,
                    "hypothesis_id": current_result.selected_hypothesis_id,
                    "statement": next(item.statement for item in correction_request.hypotheses if item.hypothesis_id == current_result.selected_hypothesis_id),
                    "posterior_probabilities": current_posterior,
                }
                episode.current_confidence = float(current_posterior[current_result.selected_hypothesis_id])
                record_operation(7, {
                    "new_evidence_ids": [item.evidence_id for item in valid_scheduler_evidence],
                    "posterior_after": current_posterior,
                    "dependent_sources_collapsed": current_result.evidence_assessment.get("dependent_sources_collapsed", 0),
                })
            record_operation(5, {
                "feedback_seeking_status": feedback_result.status,
                "typed_action_options": [item.action_id for item in unique_options.values()],
                "evidence_available": len(valid_scheduler_evidence),
                "memory_retrieval_status": next((item["status"] for item in module_diagnostics if item.get("module_id") == "memory_retrieval"), "not_requested"),
            })
            record_operation(6, {
                "accepted_evidence": len(valid_scheduler_evidence),
                "rejected_evidence": rejected,
                "source_independence_preserved": True,
            })

            # Evidence is passed into Point 24 exactly once; historical evidence remains in the audit.
            if valid_scheduler_evidence:
                current_report = self._reevaluate_conflict(
                    request, current_result, dimensions, module_diagnostics, conflict_report.conflict_score,
                )
            previous_conflict_score = current_report.conflict_score
            no_progress = 0
            actions_taken = 0
            evidence_after_actions: list[FeedbackEvidence] = []
            maximum_iterations_reached = False
            for iteration in range(1, self.config.max_semantic_feedback_rounds + 1):
                episode.current_iteration = iteration
                action = self._schedule_actions(
                    action_request, current_posterior, episode, scheduler,
                )
                if action is None:
                    break
                actions_taken += 1
                action_result = self.module_adapters.invoke(
                    action.responsible_module,
                    {
                        "action": action.to_dict(),
                        "conflict_report": current_report.to_dict(),
                        "current_interpretation": episode.current_interpretation,
                        "posterior_probabilities": current_posterior,
                        "iteration": iteration,
                        "shared_budget": scheduler.diagnostics(),
                    },
                    request.module_inputs,
                )
                module_diagnostics.append(action_result.to_dict())
                new_valid, rejected = self._validate_evidence(
                    action_result.feedback_evidence, tuple(current_posterior), seen_ids,
                )
                rejected_evidence.extend(rejected)
                result_status = (
                    "completed_with_new_evidence" if new_valid
                    else "completed_no_new_evidence" if action_result.status.startswith("completed")
                    else action_result.status
                )
                recorded_action = scheduler.record_attempt(action, result_status)
                episode.action_history.append(recorded_action.to_dict())
                record_operation(4, {
                    "selected_action": recorded_action.to_dict(),
                    "scheduler": scheduler.diagnostics(),
                }, iteration=iteration)
                record_operation(5, {
                    "module_id": action.responsible_module,
                    "module_status": action_result.status,
                    "feedback_evidence_count": len(new_valid),
                }, status=result_status, iteration=iteration)
                if not new_valid:
                    no_progress += 1
                    record_operation(11, {
                        "delta_probability": 0.0,
                        "delta_conflict": 0.0,
                        "progress_detected": False,
                        "converged": True,
                        "consecutive_no_progress": no_progress,
                        "no_progress_patience": self.config.no_progress_patience,
                        "convergence_does_not_imply_correctness": True,
                    }, iteration=iteration)
                    if no_progress >= self.config.no_progress_patience:
                        break
                    if iteration == self.config.max_semantic_feedback_rounds:
                        maximum_iterations_reached = True
                    continue

                previous_posterior = dict(current_posterior)
                previous_selected = current_result.selected_hypothesis_id
                current_result = self._apply_evidence_batch(correction_request, current_result, new_valid)
                current_posterior = dict(current_result.posterior_probabilities)
                all_evidence.extend(new_valid)
                evidence_after_actions.extend(new_valid)
                episode.evidence_history.extend(asdict(item) for item in new_valid)
                updated_hypothesis = next(
                    item for item in correction_request.hypotheses
                    if item.hypothesis_id == current_result.selected_hypothesis_id
                )
                episode.current_interpretation = {
                    "interpretation_id": current_result.selected_hypothesis_id,
                    "statement": updated_hypothesis.statement,
                    "components": _plain(updated_hypothesis.components),
                    "posterior_probabilities": current_posterior,
                    "source": updated_hypothesis.source,
                }
                episode.current_confidence = float(current_posterior[current_result.selected_hypothesis_id])
                record_operation(7, {
                    "new_evidence_ids": [item.evidence_id for item in new_valid],
                    "posterior_before": previous_posterior,
                    "posterior_after": current_posterior,
                    "dependent_sources_collapsed": current_result.evidence_assessment.get("dependent_sources_collapsed", 0),
                }, iteration=iteration)

                reprocessed_evidence = self._reprocess_affected_modules(
                    current_report,
                    episode.current_interpretation,
                    new_valid,
                    request,
                    module_diagnostics,
                )
                valid_reprocessed, rejected = self._validate_evidence(
                    reprocessed_evidence, tuple(current_posterior), seen_ids,
                )
                rejected_evidence.extend(rejected)
                if valid_reprocessed:
                    current_result = self._apply_evidence_batch(correction_request, current_result, valid_reprocessed)
                    current_posterior = dict(current_result.posterior_probabilities)
                    all_evidence.extend(valid_reprocessed)
                    evidence_after_actions.extend(valid_reprocessed)
                    episode.evidence_history.extend(asdict(item) for item in valid_reprocessed)
                    revised = next(item for item in correction_request.hypotheses if item.hypothesis_id == current_result.selected_hypothesis_id)
                    episode.current_interpretation = {
                        "interpretation_id": current_result.selected_hypothesis_id,
                        "hypothesis_id": current_result.selected_hypothesis_id,
                        "statement": revised.statement,
                        "components": _plain(revised.components),
                        "posterior_probabilities": current_posterior,
                        "source": revised.source,
                    }
                    episode.current_confidence = float(current_posterior[current_result.selected_hypothesis_id])
                record_operation(8, {
                    "affected_modules": list(current_report.affected_modules),
                    "new_evidence_from_reprocessing": len(valid_reprocessed),
                    "original_input_preserved": True,
                    "model_learning_authorized": False,
                }, iteration=iteration)

                next_report = self._reevaluate_conflict(
                    request, current_result, current_report.dimensions,
                    module_diagnostics, current_report.conflict_score,
                )
                episode.conflict_history.append(next_report.to_dict())
                progress = self._measure_progress(
                    previous_posterior,
                    current_posterior,
                    current_report.conflict_score,
                    next_report.conflict_score,
                )
                current_report = next_report
                no_progress = no_progress + 1 if progress["converged"] else 0
                record_operation(9, {
                    "conflict_score_before": previous_conflict_score,
                    "conflict_score_after": current_report.conflict_score,
                    "conflict_delta": previous_conflict_score - current_report.conflict_score,
                    "conflict_resolved": not current_report.detected,
                }, iteration=iteration)
                previous_conflict_score = current_report.conflict_score
                calibration = self._recalibrate_confidence(current_result, all_evidence, correction_request.verified_hypothesis_id)
                record_operation(10, calibration, iteration=iteration)
                record_operation(11, {
                    **progress,
                    "consecutive_no_progress": no_progress,
                    "no_progress_patience": self.config.no_progress_patience,
                }, iteration=iteration)

                status = self._decide_status(
                    initial_id, current_result, current_report, calibration,
                    no_action_available=False, maximum_iterations_reached=False,
                    verified_hypothesis_id=correction_request.verified_hypothesis_id,
                )
                if status in {"SUPPORTED_CORRECTION", "INDEPENDENTLY_VERIFIED", "NO_CORRECTION_NEEDED"}:
                    break
                if no_progress >= self.config.no_progress_patience:
                    break
                if iteration == self.config.max_semantic_feedback_rounds:
                    maximum_iterations_reached = True
                    break

            calibration = self._recalibrate_confidence(current_result, all_evidence, correction_request.verified_hypothesis_id)
            # If no action was available, retain a genuine uncertainty result instead of promoting stability.
            status = self._decide_status(
                initial_id, current_result, current_report, calibration,
                no_action_available=scheduler.total_actions == 0,
                maximum_iterations_reached=maximum_iterations_reached,
                verified_hypothesis_id=correction_request.verified_hypothesis_id,
            )
            record_operation(12, {
                "status": status,
                "iterations": episode.current_iteration,
                "actions_taken": actions_taken,
                "no_progress_rounds": no_progress,
                "scheduler": scheduler.diagnostics(),
            })

        final_status = status if conflict_report.detected else "NO_CORRECTION_NEEDED"
        # Ensure every declared controller operation has exactly one row. Recurrent round
        # records are grouped under their single source-ordered operation entry.
        if 10 not in operation_rows:
            record_operation(10, self._recalibrate_confidence(
                current_result, all_evidence, correction_request.verified_hypothesis_id,
            ), status="completed")
        if 11 not in operation_rows:
            record_operation(11, {
                "progress_measured": False,
                "convergence_does_not_imply_correctness": True,
                "latest_conflict_score": current_report.conflict_score,
            }, status="skipped")
        episode = self._finalize_episode(
            episode, current_result, final_status, all_evidence, request, current_report,
        )
        record_operation(13, {
            "episode_id": episode.episode_id,
            "status": final_status,
            "initial_interpretation_preserved": True,
            "correction_record": episode.correction_record,
            "model_updated": False,
        })
        for operation_index in range(14):
            rows = operation_rows.get(operation_index, [])
            if not rows:
                record_operation(
                    operation_index,
                    {"status": "not_reached", "reason": "no conflict or stage not required"},
                    status="skipped",
                )
                rows = operation_rows[operation_index]
            if len(rows) == 1:
                trace.append(rows[0])
            else:
                combined = dict(rows[-1])
                combined["functional_output"] = {
                    "round_history": [item["functional_output"] for item in rows],
                    "round_count": len(rows),
                }
                combined["iteration"] = max(int(item.get("iteration", 0)) for item in rows)
                trace.append(combined)

        diagnostics = {
            "initial_conflict_score": conflict_report.conflict_score,
            "final_conflict_score": current_report.conflict_score,
            "initial_interpretation_id": initial_id,
            "final_interpretation_id": current_result.selected_hypothesis_id,
            "evidence_count": len(all_evidence),
            "rejected_evidence": rejected_evidence,
            "module_invocations": module_diagnostics,
            "shared_action_budget": scheduler.diagnostics() if conflict_report.detected else {},
            "maximum_iterations": self.config.max_semantic_feedback_rounds,
            "feedback_rounds": episode.current_iteration,
            "model_fingerprint_unchanged_by_inference": True,
            "model_learning_authorized": False,
            "model_updated": False,
            "scientific_scope": "functional integration prototype; not a validated biological feedback circuit",
            "elapsed_seconds": time.perf_counter() - started,
        }
        result = RecurrentFeedbackResult(
            request_id=request.request_id,
            status=final_status,
            episode=episode,
            latest_corrective_result=current_result,
            operation_trace=tuple(trace),
            diagnostics=diagnostics,
        )
        if self.auto_log:
            try:
                diagnostics["sqlite_logging"] = record_recurrent_result_to_sqlite(
                    result, self.logger_database_path,
                )
            except Exception as exc:
                diagnostics["sqlite_logging"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        return result

    def execute_authorized_learning_transaction(
        self,
        parent_model: Any,
        learning_request: Any,
        proposer: Callable[[Any, Any], Any],
        regression_gates: Sequence[tuple[str, Callable[[Any, Any], bool]]],
        metrics_before: Mapping[str, float] | None = None,
        metrics_after: Mapping[str, float] | None = None,
    ) -> Any:
        """Delegate an explicitly authorized detached-child proposal without promoting the returned child."""
        from ore_contracts import LearningRequest, LearningTransactionManager

        if not isinstance(learning_request, LearningRequest):
            raise TypeError("learning_request must be an ore_contracts.LearningRequest")
        manager = LearningTransactionManager()
        return manager.execute(
            parent_model, learning_request, proposer, regression_gates,
            metrics_before=dict(metrics_before or {}),
            metrics_after=dict(metrics_after or {}),
        )


def _bounded(name: str, value: float) -> float:
    """Clamp a finite compatibility score to the unit interval."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return max(0.0, min(1.0, numeric))


def register_recurrent_annotations(
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Register every recurrent-controller and N9-feedback class/function/method in the shared logger."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    configured = database_path or os.environ.get("ORE_LOGGER_DB")
    target = Path(configured).expanduser().resolve() if configured else root / "ore_logs" / "ore_logger.sqlite3"
    catalogue = sync_repository(
        LoggerDatabase(target), root,
        include_paths={
            "ore_corrective_learning", "ore_n9", "ore_feedback_seeking",
            "ore_confirmation_bias", "ore_cognitive_heuristics",
            "ore_curiosity_motivation", "ore_memory_integration",
        },
    )
    return {
        **catalogue,
        "database_path": str(target),
        "scope": (
            "ore_corrective_learning", "ore_n9", "ore_feedback_seeking",
            "ore_confirmation_bias", "ore_cognitive_heuristics",
            "ore_curiosity_motivation", "ore_memory_integration",
        ),
        "annotation_table": "annotations",
    }


def record_recurrent_result_to_sqlite(
    result: RecurrentFeedbackResult,
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist every recurrent operation and link each event to its class and method annotations."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    if not isinstance(result, RecurrentFeedbackResult):
        raise TypeError("result must be a RecurrentFeedbackResult")
    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    configured = database_path or os.environ.get("ORE_LOGGER_DB")
    target = Path(configured).expanduser().resolve() if configured else root / "ore_logs" / "ore_logger.sqlite3"
    database = LoggerDatabase(target)
    catalogue = sync_repository(
        database, root,
        include_paths={
            "ore_corrective_learning", "ore_n9", "ore_feedback_seeking",
            "ore_confirmation_bias", "ore_cognitive_heuristics",
            "ore_curiosity_motivation", "ore_memory_integration",
        },
    )
    run_id = database.create_run(
        script_path="ore_corrective_learning/recurrent_feedback",
        module_name="ore_corrective_learning.recurrent_controller",
        entry_kind="recurrent_feedback_correction",
        working_directory=str(root),
    )
    class_annotation = database.get_annotation(
        "ore_corrective_learning.recurrent_controller",
        "RecurrentFeedbackController", "class",
        "ore_corrective_learning/recurrent_controller.py",
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            qualified = str(item.get("method", ""))
            if "." in qualified:
                qualified = qualified.split(".", 1)[1]
            annotation = None
            if qualified:
                annotation = database.get_annotation(
                    "ore_corrective_learning.recurrent_controller",
                    f"RecurrentFeedbackController.{qualified}",
                    "function",
                    "ore_corrective_learning/recurrent_controller.py",
                )
            database.record_event(
                run_id=run_id,
                sequence_no=sequence,
                event_kind="operation",
                symbol_id=annotation["symbol_id"] if annotation else None,
                class_symbol_id=class_annotation["symbol_id"] if class_annotation else None,
                source_path="ore_corrective_learning/recurrent_controller.py",
                line_number=annotation.get("line_start") if annotation else None,
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                metadata=_plain(dict(item)),
            )
        database.record_event(
            run_id=run_id,
            sequence_no=len(result.operation_trace) + 1,
            event_kind="episode_finalized",
            class_symbol_id=class_annotation["symbol_id"] if class_annotation else None,
            source_path="ore_corrective_learning/recurrent_controller.py",
            metadata={
                "request_id": result.request_id,
                "status": result.status,
                "episode": result.episode.to_dict(),
                "diagnostics": _plain(result.diagnostics),
                "model_updated": False,
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
        "events_written": len(result.operation_trace) + 1,
        "class_annotation_linked": class_annotation is not None,
    }


__all__ = [
    "RECURRENT_FEEDBACK_OPERATIONS",
    "RecurrentFeedbackController",
    "register_recurrent_annotations",
    "record_recurrent_result_to_sqlite",
]
