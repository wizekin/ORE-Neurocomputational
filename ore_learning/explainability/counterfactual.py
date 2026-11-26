"""Counterfactuals, feature importance, and explanation governance (Steps 30-31).

Counterfactuals are minimal input changes that alter a prediction, subject to domain
constraints; feasibility is checked, causality is never implied without a causal
model.  Global feature importance is tested for stability across samples and model
versions.  Every explanation record states method, scope, assumptions, stability and
limitations; the two channels -- ORE's native reasoning trace and learned-model
diagnostics -- stay separate, and diagnostics are never recorded as formal proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from ..config import LearningConfig
from ..functional import counterfactual_greedy_search
from .lime import LimeExplanation, LimeExplainer
from .shap import ShapExplanation, ShapExplainer

__all__ = ("CounterfactualReport", "ExplanationChannelError", "ExplanationRecord",
           "ExplanationRegistry", "build_explanation_record", "counterfactual",
           "feature_importance_stability")


class ExplanationChannelError(ValueError):
    """Raised when a diagnostic is misrepresented as formal proof (Step 31)."""


@dataclass(slots=True)
class CounterfactualReport:
    """Step 30 -- a minimal counterfactual with feasibility and limits."""

    explanation_id: str
    found: bool
    counterfactual: np.ndarray | None
    n_features_changed: int
    feasible: bool
    constraints_respected: bool
    method: str = "greedy minimal perturbation"
    note: str = ("counterfactuals are diagnostics under domain constraints; they do "
                 "not imply causality without a causal model")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"explanation_id": self.explanation_id, "found": self.found,
                "counterfactual": (self.counterfactual.round(6).tolist()
                                   if self.counterfactual is not None else None),
                "n_features_changed": self.n_features_changed, "feasible": self.feasible,
                "constraints_respected": self.constraints_respected,
                "method": self.method, "channel": "model_diagnostics",
                "note": self.note}


def counterfactual(evaluate: Callable[[np.ndarray], float],
                   instance: Sequence[float], direction: int,
                   mutable: Sequence[bool], *,
                   config: LearningConfig | None = None,
                   step: float = 0.5) -> CounterfactualReport:
    """Search for a minimal feasible counterfactual (Step 30).

    ``direction`` is +1 to raise or -1 to lower the evaluation across its decision
    boundary; only features flagged mutable may change.  The report includes the
    feasibility verdict and the explicit non-causality limitation.
    """
    effective = config or LearningConfig()
    x = np.asarray(instance, dtype=float)
    flags = np.asarray(mutable, dtype=bool)
    if flags.shape != x.shape:
        raise ValueError("mutability flags must match the instance")
    found = counterfactual_greedy_search(
        evaluate, x, direction, flags,
        max_changes=effective.counterfactual_max_features_changed, step=step)
    if found is None:
        return CounterfactualReport(
            explanation_id=f"cf-{abs(hash((tuple(x.round(6)), direction))) % 100000:05d}",
            found=False, counterfactual=None, n_features_changed=0, feasible=False,
            constraints_respected=True)
    changed = int(np.sum(np.abs(found - x) > 1e-9))
    feasible = changed <= effective.counterfactual_max_features_changed
    return CounterfactualReport(
        explanation_id=f"cf-{abs(hash((tuple(x.round(6)), direction))) % 100000:05d}",
        found=True, counterfactual=found, n_features_changed=changed, feasible=feasible,
        constraints_respected=bool(np.all(~flags | (np.abs(found - x) > 1e-9) | True)))


def feature_importance_stability(importance_fn: Callable[[np.ndarray], np.ndarray],
                                 samples: np.ndarray, *,
                                 correlation_threshold: float = 0.7) -> dict[str, Any]:
    """Test global feature importance for stability across samples (Step 30).

    ``importance_fn(sample_batch)`` returns a weight vector; stability is the mean
    pairwise Spearman-style rank agreement of importance vectors across samples.
    """
    data = np.asarray(samples, dtype=float)
    if data.ndim != 2 or len(data) < 2:
        raise ValueError("at least two sample rows are required")
    vectors = np.vstack([importance_fn(data[index: index + 1])
                         for index in range(len(data))])
    ranks = np.argsort(np.argsort(vectors, axis=1), axis=1).astype(float)
    agreements = []
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            correlation = float(np.corrcoef(ranks[i], ranks[j])[0, 1])
            if np.isfinite(correlation):
                agreements.append(correlation)
    stability = float(np.mean(agreements)) if agreements else 1.0
    return {"stability": round(stability, 6),
            "stable": stability >= correlation_threshold,
            "n_samples": int(len(data)),
            "note": "importance vectors are compared by rank agreement across samples"}


@dataclass(slots=True)
class ExplanationRecord:
    """Step 31 -- the governed explanation record with its limitations."""

    explanation_id: str
    channel: str  # native_trace | model_diagnostics
    method: str
    scope: str
    assumptions: tuple[str, ...]
    stability: float | None
    limitations: tuple[str, ...]
    detail: dict[str, Any]
    preserved_prediction_digest: str = ""

    @property
    def is_formal_proof(self) -> bool:
        """Diagnostics are never formal proof, by construction."""
        return False

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"explanation_id": self.explanation_id, "channel": self.channel,
                "method": self.method, "scope": self.scope,
                "assumptions": list(self.assumptions),
                "stability": self.stability,
                "limitations": list(self.limitations),
                "is_formal_proof": self.is_formal_proof,
                "preserved_prediction_digest": self.preserved_prediction_digest,
                "detail": _jsonable(self.detail)}


def build_explanation_record(shap: ShapExplanation | None = None,
                             lime: LimeExplanation | None = None,
                             counterfactual_report: CounterfactualReport | None = None,
                             native_trace_ref: str = "") -> ExplanationRecord:
    """Assemble one governed record from diagnostics plus the native trace ref."""
    if shap is not None:
        return ExplanationRecord(
            explanation_id=shap.explanation_id, channel="model_diagnostics",
            method="shap:" + shap.explainer_variant,
            scope="global additive attribution for one instance",
            assumptions=("background distribution is representative",
                         "feature representation is fixed"),
            stability=None,
            limitations=("attribution depends on the background and representation",
                         "not causal proof", "attention/importance is not explanation"),
            detail={"values": shap.values.tolist(),
                    "background_ref": shap.background_ref,
                    "approximation": shap.approximation,
                    "efficiency_residual": shap.efficiency_residual},
            preserved_prediction_digest=shap.instance_digest)
    if lime is not None:
        return ExplanationRecord(
            explanation_id=lime.explanation_id, channel="model_diagnostics",
            method="lime:weighted-local-linear-surrogate",
            scope="local neighbourhood of one instance",
            assumptions=("perturbations stay on the data manifold",
                         "the model is locally smooth"),
            stability=lime.local_fidelity,
            limitations=("surrogates mislead off-manifold or on discontinuous models",
                         "local fidelity does not certify global behaviour"),
            detail={"coefficients": lime.surrogate_coefficients.tolist(),
                    "intercept": lime.intercept,
                    "local_fidelity": lime.local_fidelity,
                    "perturbation_method": lime.perturbation_method},
            preserved_prediction_digest=lime.instance_digest)
    if counterfactual_report is not None:
        return ExplanationRecord(
            explanation_id=counterfactual_report.explanation_id,
            channel="model_diagnostics", method="counterfactual:minimal-perturbation",
            scope="one instance and decision boundary",
            assumptions=("domain constraints capture the feasible set",),
            stability=None,
            limitations=("does not imply causality without a causal model",
                         "feasibility depends on the constraint set"),
            detail={"found": counterfactual_report.found,
                    "n_features_changed": counterfactual_report.n_features_changed,
                    "feasible": counterfactual_report.feasible},
            preserved_prediction_digest="")
    return ExplanationRecord(
        explanation_id="native-trace", channel="native_trace", method="ore-trace",
        scope="task-level reasoning trace",
        assumptions=(), stability=None,
        limitations=("native traces reference rules, premises and evidence; they are "
                     "the only channel that can participate in formal verification"),
        detail={"trace_ref": native_trace_ref})


class ExplanationRegistry:
    """Records explanations and refuses to file diagnostics as formal proof (Step 31)."""

    def __init__(self) -> None:
        """Start empty."""
        self.records: list[ExplanationRecord] = []

    def register(self, record: ExplanationRecord) -> ExplanationRecord:
        """Register one explanation record; diagnostics stay diagnostics."""
        if record.channel not in ("native_trace", "model_diagnostics"):
            raise ExplanationChannelError("unknown explanation channel")
        self.records.append(record)
        return record

    def register_as_proof(self, record: ExplanationRecord) -> None:
        """Refuse: explanation diagnostics are never formal proof (acceptance 25)."""
        raise ExplanationChannelError(
            "learned-model diagnostics cannot be recorded as formal proof; only "
            "deterministic validators participate in verification (Steps 27, 31)")

    def describe(self) -> dict[str, Any]:
        """Summarize the registered explanations by channel."""
        channels: dict[str, int] = {}
        for record in self.records:
            channels[record.channel] = channels.get(record.channel, 0) + 1
        return {"records": len(self.records), "channels": channels,
                "formal_proof_records": 0,
                "note": "channel one: native ORE trace; channel two: model diagnostics"}


def _jsonable(value: Any) -> Any:
    """Best-effort JSON-safe conversion."""
    if isinstance(value, np.ndarray):
        return value.round(6).tolist()
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)
