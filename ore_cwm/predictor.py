"""Predictive transition model: statistical outcome prediction for candidate operations.

The predictor approximates observable consequences with f_theta(state, action, context).
It is not the world itself: every estimate is labeled predictive, never causal, and
simulated states stay provisional planning objects detached from the live workspace.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ore_workspace.models import content_fingerprint

from .baselines import (
    baseline_for,
    evidence_coverage_proxy,
    expected_information_gain,
)
from .contracts import (
    CWM_FEATURE_SCHEMA_VERSION,
    CWM_PREDICTOR_VERSION,
    MISSING,
    ArtifactTypeEffect,
    CognitiveOperation,
    CognitiveStateFeatures,
    CWMConfig,
    FeatureProvenance,
    FeatureValue,
    InformationGainEstimate,
    MissingMeasurement,
    OperationContext,
    OperationType,
    OutcomeClass,
    PredictedOutcome,
    PredictionUncertainty,
    ResourceEnvelope,
    RiskAssessment,
)
from .features import EVIDENCE_FIELDS
from .learning import ExecutionLedger

FAILURE_SPLIT_DEFAULTS = {
    "recoverable_failure": 0.45,
    "no_progress": 0.30,
    "hard_failure": 0.25,
}

CLASS_BY_CAPABILITY_HINT = (
    ("retrieval", "knowledge_retrieval"),
    ("knowledge", "knowledge_retrieval"),
    ("composition", "composition"),
    ("semantic", "composition"),
)


def classify_operation(operation: CognitiveOperation) -> str:
    """Map one candidate onto a coarse operation class used by the estimator bank."""
    if operation.operation_type is OperationType.VERIFY:
        return "verification"
    if operation.operation_type is OperationType.STOP:
        return "stop"
    capability = operation.capability_id.lower()
    for token, operation_class in CLASS_BY_CAPABILITY_HINT:
        if token in capability:
            return operation_class
    return "module_execution"


@dataclass(frozen=True, slots=True)
class PredictedTransition:
    """One application of the transition model f_theta(state, action, context)."""

    operation: CognitiveOperation
    operation_class: str
    predicted_outcome: PredictedOutcome
    next_features: CognitiveStateFeatures
    context: OperationContext


class StatisticalTransitionModel:
    """Estimate the observable consequences of candidate operations from execution history.

    Success uses Beta-Binomial smoothing; duration, CPU, and memory use exponential
    moving estimates with robust dispersion; outcome classes split the failure mass
    using historical failure-class shares with conservative defaults; risks are
    estimated separately; and every prediction carries uncertainty with support counts.
    """

    def __init__(
        self,
        ledger: ExecutionLedger,
        config: CWMConfig | None = None,
        predictor_version: str = CWM_PREDICTOR_VERSION,
    ) -> None:
        """Configure the statistical predictor against one shared execution ledger."""
        self.ledger = ledger
        self.config = config or ledger.config
        self.predictor_version = predictor_version

    # -- public API ---------------------------------------------------------

    def transition(
        self,
        features: CognitiveStateFeatures,
        operation: CognitiveOperation,
        context: OperationContext,
        belief_probabilities: Sequence[float] | None = None,
    ) -> PredictedTransition:
        """Predict the outcome and next compact state for one candidate operation."""
        operation_class = classify_operation(operation)
        bundle = self.ledger.stats_for(operation_class, operation.module_id)
        predicted_outcome = self._predict_outcome(
            features, operation, operation_class, bundle, context, belief_probabilities)
        next_features = self._advance_features(
            features, operation, operation_class, predicted_outcome)
        return PredictedTransition(
            operation=operation,
            operation_class=operation_class,
            predicted_outcome=predicted_outcome,
            next_features=next_features,
            context=context,
        )

    def support_for(self, operation: CognitiveOperation) -> int:
        """Return the historical sample count backing predictions for one candidate."""
        operation_class = classify_operation(operation)
        return self.ledger.stats_for(operation_class, operation.module_id).support()

    # -- prediction internals -------------------------------------------------

    def _predict_outcome(
        self,
        features: CognitiveStateFeatures,
        operation: CognitiveOperation,
        operation_class: str,
        bundle: Any,
        context: OperationContext,
        belief_probabilities: Sequence[float] | None = None,
    ) -> PredictedOutcome:
        """Assemble one predicted outcome from the estimator bundle and context."""
        success_probability, success_interval = self._success_estimate(bundle)
        duration, duration_interval, duration_variance = self._duration_estimate(
            operation, operation_class, bundle, context)
        cpu_cost = self._ema_value(bundle.cpu_cost, baseline_for(operation_class).expected_cpu_cost)
        memory_mb = self._ema_value(
            bundle.memory_mb, baseline_for(operation_class).expected_memory_mb)
        classes = self._outcome_classes(bundle, success_probability)
        goal_progress = self._goal_progress(bundle, operation_class)
        information_gain = self._information_gain(
            features, operation, success_probability, belief_probabilities)
        risks = self._risks(bundle, operation_class, success_probability,
                            duration, operation)
        uncertainty = self._uncertainty(
            bundle, success_interval, duration_interval, duration_variance)
        predicted_artifacts = self._artifact_effects(operation)
        expected_cost = ResourceEnvelope(
            expected_duration_ms=duration,
            expected_cpu_cost=cpu_cost,
            expected_memory_mb=memory_mb,
            expected_io_cost=MISSING,
            sample_count=bundle.support(),
        )
        coverage_delta = self._evidence_delta(features, operation, success_probability)
        explanation = (
            f"predictive estimate for {operation.module_id} under {operation_class}; "
            f"success {success_probability:.2f} within [{success_interval[0]:.2f}, "
            f"{success_interval[1]:.2f}] on {bundle.support()} validated executions; "
            f"expected duration {duration:.0f}ms conditioned on resource load "
            f"{self._load_label(context)}; estimates are predictive, never causal.")
        return PredictedOutcome(
            success_probability=success_probability,
            outcome_class_probabilities=classes,
            expected_cost=expected_cost,
            predicted_feature_delta={
                "evidence_coverage": coverage_delta,
                "goal_completion": goal_progress,
            },
            goal_progress_estimate=goal_progress,
            information_gain=information_gain,
            risks=risks,
            uncertainty=uncertainty,
            predicted_artifacts=predicted_artifacts,
            explanation=explanation,
        )

    def _success_estimate(self, bundle: Any) -> tuple[float, tuple[float, float]]:
        """Return the smoothed success probability with its credible interval."""
        probability = bundle.success.posterior_mean()
        interval = bundle.success.credible_interval(0.90)
        drift_multiplier = self.ledger.drift.status().confidence_multiplier
        if drift_multiplier < 1.0:
            probability = 0.5 + (probability - 0.5) * drift_multiplier
        return max(0.0, min(1.0, probability)), interval

    def _duration_estimate(
        self,
        operation: CognitiveOperation,
        operation_class: str,
        bundle: Any,
        context: OperationContext,
    ) -> tuple[float, tuple[float, float], float | MissingMeasurement]:
        """Return expected duration with dispersion interval and moving variance."""
        baseline = baseline_for(operation_class)
        mean = bundle.duration.mean()
        robust = bundle.robust_duration.estimate()
        duration = (
            robust if robust is not None
            else (mean if mean is not None else baseline.expected_duration_ms)
        )
        load_multiplier = 1.0
        if context.resource_load is not MISSING:
            load_multiplier = 1.0 + 0.25 * float(context.resource_load)
        depth_penalty = 50.0 * context.dependency_depth
        conditioned = duration * load_multiplier + depth_penalty
        dispersion = bundle.robust_duration.dispersion()
        if dispersion is None:
            dispersion = max(0.25 * conditioned, 1.0)
        interval = (
            max(0.0, conditioned - 1.645 * dispersion),
            conditioned + 1.645 * dispersion,
        )
        variance = (
            bundle.duration.variance()
            if bundle.duration.sample_count() > 1 else MISSING
        )
        return conditioned, interval, variance

    def _ema_value(self, estimator: Any, fallback: float) -> float:
        """Return an EMA estimate or the conservative fallback."""
        mean = estimator.mean()
        return mean if mean is not None else fallback

    def _outcome_classes(self, bundle: Any, success_probability: float) -> dict[str, float]:
        """Split success and failure mass into the documented outcome classes."""
        samples = bundle.success.sample_count()
        share_partial = 0.15
        if samples > 0:
            accepted_share = bundle.success.successes / samples
            share_partial = max(0.0, 1.0 - accepted_share)
        full_success = success_probability * (1.0 - share_partial)
        partial = success_probability * share_partial
        failure_mass = 1.0 - success_probability
        failure_classes = dict(FAILURE_SPLIT_DEFAULTS)
        total_known = sum(bundle.failure_classes.values())
        if total_known > 0:
            failure_classes = {
                name: count / total_known for name, count in bundle.failure_classes.items()
            }
        classes = {
            OutcomeClass.SUCCESS.value: full_success,
            OutcomeClass.PARTIAL_SUCCESS.value: partial,
        }
        for name, share in failure_classes.items():
            classes[name] = failure_mass * share
        total = sum(classes.values())
        if total <= 0.0:
            return {OutcomeClass.NO_PROGRESS.value: 1.0}
        return {name: value / total for name, value in classes.items()}

    def _goal_progress(self, bundle: Any, operation_class: str) -> float:
        """Estimate goal progress from the moving average of observed deltas."""
        baseline = baseline_for(operation_class)
        mean = bundle.goal_progress.mean()
        return mean if mean is not None else baseline.expected_goal_progress

    def _information_gain(
        self,
        features: CognitiveStateFeatures,
        operation: CognitiveOperation,
        success_probability: float,
        belief_probabilities: Sequence[float] | None,
    ) -> InformationGainEstimate:
        """Estimate information gain with entropy when beliefs exist, else a labeled proxy."""
        if belief_probabilities:
            prior = list(belief_probabilities)
            posterior_candidates = {
                "success": _sharpened(prior, success_probability),
                "failure": _flattened(prior),
            }
            value = expected_information_gain(
                prior, posterior_candidates,
                {"success": success_probability, "failure": 1.0 - success_probability},
            )
            return InformationGainEstimate(
                max(0.0, value), "entropy",
                "prior hypothesis entropy minus expected posterior entropy")
        coverage_now = features.value("evidence_coverage")
        if coverage_now is MISSING:
            coverage_now = 0.0
        produced = {name for name in EVIDENCE_FIELDS if name in operation.produced_fields}
        gain = 0.0
        if produced:
            gain = success_probability * (len(produced) / len(EVIDENCE_FIELDS))
        return InformationGainEstimate(
            gain, "proxy",
            "expected accepted-evidence coverage increase (proxy; no defensible "
            "hypothesis distribution was supplied)")

    def _risks(
        self,
        bundle: Any,
        operation_class: str,
        success_probability: float,
        duration: float,
        operation: CognitiveOperation,
    ) -> RiskAssessment:
        """Estimate the five risk dimensions separately with conservative defaults."""
        executed = max(bundle.executed_count, 1)
        timeout_risk = baseline_for(operation_class).timeout_risk
        if operation.timeout_ms is not MISSING:
            timeout = float(operation.timeout_ms)
            if timeout > 0:
                fraction = min(1.0, max(0.0, (duration - timeout) / max(duration, 1.0)))
                timeout_risk = max(timeout_risk, fraction)
        return RiskAssessment(
            failure_risk=max(0.0, min(1.0, 1.0 - success_probability)),
            conflict_creation_risk=_share(bundle.risk_counters.get("conflict_created", 0), executed),
            artifact_invalidation_risk=_share(
                bundle.risk_counters.get("artifact_invalidated", 0), executed),
            timeout_risk=timeout_risk,
            policy_violation_risk=_share(
                bundle.risk_counters.get("policy_violation", 0), executed),
            severe_tail_notes=(
                f"hard-failure tail preserved at "
                f"{bundle.failure_classes.get('hard_failure', 0) / executed:.3f}"
                if bundle.executed_count > 0 else
                "rare severe risks are preserved through separate risk dimensions"),
        )

    def _uncertainty(
        self,
        bundle: Any,
        success_interval: tuple[float, float],
        duration_interval: tuple[float, float],
        duration_variance: Any,
    ) -> PredictionUncertainty:
        """Quantify uncertainty with support counts, calibration, and OOD indicators."""
        calibration_status = self.ledger.calibration.status()
        if self.ledger.drift.status().drifted:
            calibration_status = "drifted"
        return PredictionUncertainty(
            success_credible_interval=success_interval,
            duration_interval_ms=duration_interval,
            duration_variance_ms2=(
                duration_variance if duration_variance is not MISSING else MISSING),
            sample_count=bundle.support(),
            calibration_status=calibration_status,
            out_of_distribution=False,
            confidence_multiplier=self.ledger.drift.status().confidence_multiplier,
        )

    def _artifact_effects(self, operation: CognitiveOperation) -> tuple[Any, ...]:
        """Predict typed artifact categories and quantities, never semantic content."""
        return tuple(
            _artifact_effect(contract_id) for contract_id in operation.output_schema
        )

    def _evidence_delta(
        self, features: CognitiveStateFeatures, operation: CognitiveOperation,
        success_probability: float,
    ) -> float:
        """Return the predicted evidence-coverage delta for this candidate."""
        coverage_now = features.value("evidence_coverage")
        if coverage_now is MISSING:
            coverage_now = 0.0
        produced = [name for name in EVIDENCE_FIELDS if name in operation.produced_fields]
        if not produced:
            return 0.0
        predicted_coverage = min(1.0, float(coverage_now) + len(produced) / len(EVIDENCE_FIELDS))
        return evidence_coverage_proxy(float(coverage_now), predicted_coverage) * (
            1.0 if success_probability > 0.5 else 0.5)

    def _advance_features(
        self,
        features: CognitiveStateFeatures,
        operation: CognitiveOperation,
        operation_class: str,
        predicted: PredictedOutcome,
    ) -> CognitiveStateFeatures:
        """Build the provisional next feature state; imagined states are not observations."""
        values: dict[str, FeatureValue] = {}
        deltas = predicted.predicted_feature_delta
        for name, item in features.values.items():
            value = item.value
            changed = name in deltas and value is not MISSING
            if changed:
                value = float(value) + float(deltas[name])
            values[name] = FeatureValue(
                name=name,
                value=value,
                provenance=(
                    FeatureProvenance.PREDICTED if changed else item.provenance),
                definition=(
                    f"imagined next value after {operation.operation_id}" if changed
                    else f"carried over unchanged from revision {features.workspace_revision}"),
            )
        coverage = values.get("evidence_coverage")
        if coverage is not None and coverage.value is not MISSING:
            clipped = min(1.0, max(0.0, float(coverage.value)))
            if clipped != float(coverage.value):
                values["evidence_coverage"] = FeatureValue(
                    "evidence_coverage", clipped, coverage.provenance,
                    "imagined evidence coverage clipped to [0, 1]")
        feature_snapshot_hash = content_fingerprint({
            "schema_version": CWM_FEATURE_SCHEMA_VERSION,
            "workspace_id": features.workspace_id,
            "workspace_revision": features.workspace_revision,
            "imagined_after": operation.operation_id,
            "values": {
                name: (None if item.value is MISSING else float(item.value))
                for name, item in values.items()
            },
        })
        return CognitiveStateFeatures(
            schema_version=CWM_FEATURE_SCHEMA_VERSION,
            workspace_id=features.workspace_id,
            workspace_revision=features.workspace_revision,
            goal=features.goal,
            feature_snapshot_hash=feature_snapshot_hash,
            values=values,
        )

    def _load_label(self, context: OperationContext) -> str:
        """Return a human-readable resource-load label for explanations."""
        if context.resource_load is MISSING:
            return "unavailable"
        return f"{float(context.resource_load):.2f}"


def _artifact_effect(contract_id: str) -> ArtifactTypeEffect:
    """Build one typed artifact-effect prediction for a declared output contract."""
    return ArtifactTypeEffect(
        contract_id=contract_id,
        category="predicted_type_only",
        expected_quantity=1,
    )


def _share(count: int, total: int) -> float:
    """Return a bounded empirical share with a conservative floor of zero."""
    if total <= 0:
        return 0.0
    return min(1.0, count / total)


def _sharpened(prior: Sequence[float], success_probability: float) -> list[float]:
    """Return a success-conditioned posterior that concentrates the leading hypothesis."""
    weights = [max(value, 1e-6) for value in prior]
    sharpened = [
        value * (success_probability if index == 0 else 1.0 - success_probability)
        for index, value in enumerate(weights)
    ]
    total = sum(sharpened)
    return [value / total for value in sharpened]


def _flattened(prior: Sequence[float]) -> list[float]:
    """Return a failure-conditioned posterior that flattens the hypothesis distribution."""
    if not prior:
        return [1.0]
    flattened = [max(value, 1e-6) for value in prior]
    total = sum(flattened)
    return [value / total for value in flattened]


__all__ = [
    "FAILURE_SPLIT_DEFAULTS",
    "PredictedTransition",
    "StatisticalTransitionModel",
    "classify_operation",
]
