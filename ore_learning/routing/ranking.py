"""Utility ranking of eligible candidates (Point 14, Step 8).

U(m,t) = w_q Q + w_c C + w_r R - w_l L_hat - w_k K_hat - w_u U_hat with metrics
normalized to [0, 1] and weights set by task risk.  The score ranks candidates but
can never override hard eligibility; ties break deterministically on identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..config import LearningConfig
from ..contracts import CapabilityRequest, ModelManifest, RiskClass
from .eligibility import ModelCandidate, best_task_quality

__all__ = ("RankingRecord", "UtilityWeights", "rank_candidates", "weights_for_risk")


@dataclass(frozen=True, slots=True)
class UtilityWeights:
    """The six utility weights in force for one ranking run (Step 8)."""

    weight_quality: float
    weight_calibration: float
    weight_robustness: float
    weight_latency: float
    weight_resource: float
    weight_uncertainty: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"w_q": self.weight_quality, "w_c": self.weight_calibration,
                "w_r": self.weight_robustness, "w_l": self.weight_latency,
                "w_k": self.weight_resource, "w_u": self.weight_uncertainty}


@dataclass(frozen=True, slots=True)
class RankingRecord:
    """One ranked candidate with its utility decomposition."""

    identity: str
    utility: float
    features: dict[str, float]
    rank: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"identity": self.identity, "utility": round(self.utility, 6),
                "features": {key: round(value, 6) for key, value in self.features.items()},
                "rank": self.rank}


def weights_for_risk(config: LearningConfig,
                     risk_class: RiskClass) -> UtilityWeights:
    """Risk-adjusted utility weights: uncertainty and calibration weigh more at risk.

    LOW/MEDIUM tasks use the configured weights directly; HIGH/CRITICAL tasks double
    the uncertainty penalty weight and add half again to the calibration weight so a
    badly calibrated or highly uncertain model cannot win on raw quality alone.
    """
    if risk_class in (RiskClass.HIGH, RiskClass.CRITICAL):
        return UtilityWeights(
            weight_quality=config.utility_weight_quality,
            weight_calibration=config.utility_weight_calibration * 1.5,
            weight_robustness=config.utility_weight_robustness,
            weight_latency=config.utility_weight_latency_penalty,
            weight_resource=config.utility_weight_resource_penalty,
            weight_uncertainty=config.utility_weight_uncertainty_penalty * 2.0)
    return UtilityWeights(
        weight_quality=config.utility_weight_quality,
        weight_calibration=config.utility_weight_calibration,
        weight_robustness=config.utility_weight_robustness,
        weight_latency=config.utility_weight_latency_penalty,
        weight_resource=config.utility_weight_resource_penalty,
        weight_uncertainty=config.utility_weight_uncertainty_penalty)


def utility_features(manifest: ModelManifest, request: CapabilityRequest,
                     config: LearningConfig | None = None) -> dict[str, float]:
    """Compute the six normalized features of the utility formula for one candidate.

    ``quality`` is the best held-out measurement relevant to the task (default 0.0
    when no benchmark exists -- such candidates already fail eligibility when the
    policy requires evidence).  ``calibration`` is 1 - ECE, ``robustness`` comes from
    a robustness benchmark when present, and the three penalties (latency, resource,
    uncertainty) are clipped to [0, 1].  All definitions are documented so rankings
    are reproducible.
    """
    quality = best_task_quality(manifest, request)
    quality_value = float(quality) if quality is not None else 0.0
    calibration_value = 0.5
    brier = 0.5
    if manifest.calibration_report is not None:
        calibration_value = max(0.0, 1.0 - manifest.calibration_report.expected_calibration_error)
        brier = min(1.0, manifest.calibration_report.brier_score)
    robustness_value = 0.5
    for benchmark in manifest.benchmark_results:
        if benchmark.metric == "robustness" and not benchmark.lower_is_better:
            robustness_value = min(1.0, max(0.0, float(benchmark.value)))
            break
    latency_ratio = (manifest.resource_profile.estimated_latency_ms
                     / max(request.latency_target_ms, 1e-9))
    latency_penalty = min(1.0, max(0.0, latency_ratio - 1.0))
    resource_penalty = min(1.0, (manifest.resource_profile.cpu_cores / 8.0
                                 + manifest.resource_profile.memory_mb / 8192.0) / 2.0)
    return {
        "quality": round(quality_value, 6),
        "calibration": round(calibration_value, 6),
        "robustness": round(robustness_value, 6),
        "latency_penalty": round(latency_penalty, 6),
        "resource_penalty": round(resource_penalty, 6),
        "uncertainty_penalty": round(brier, 6),
    }


def rank_candidates(candidates: Sequence[ModelCandidate], request: CapabilityRequest,
                    config: LearningConfig | None = None) -> list[ModelCandidate]:
    """Assign utilities and deterministic ranks to eligible candidates (Step 8).

    Ineligible candidates are left untouched at rank 0 with utility None.  Ranking is
    a pure function of (manifests, request, config): identical state and inputs
    produce identical orderings, with ties broken by ``model_id`` then ``version``.
    """
    effective_config = config or LearningConfig()
    weights = weights_for_risk(effective_config, request.risk_class)
    from ..functional import utility_score

    scored: list[tuple[float, str, str, ModelCandidate]] = []
    for candidate in candidates:
        if not candidate.eligible:
            candidate.rank = 0
            candidate.utility = None
            continue
        features = utility_features(candidate.manifest, request, effective_config)
        utility = utility_score(
            features["quality"], features["calibration"], features["robustness"],
            features["latency_penalty"], features["resource_penalty"],
            features["uncertainty_penalty"],
            weight_quality=weights.weight_quality,
            weight_calibration=weights.weight_calibration,
            weight_robustness=weights.weight_robustness,
            weight_latency=weights.weight_latency,
            weight_resource=weights.weight_resource,
            weight_uncertainty=weights.weight_uncertainty)
        candidate.utility = round(utility, 9)
        candidate.utility_features = features
        scored.append((-utility, candidate.manifest.model_id,
                       candidate.manifest.version, candidate))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    for rank_index, item in enumerate(scored, start=1):
        item[3].rank = rank_index
    return list(candidates)


def ranking_table(candidates: Sequence[ModelCandidate]) -> list[RankingRecord]:
    """Build the ranking rationale table for a selection record."""
    table = []
    for candidate in sorted((c for c in candidates if c.eligible and c.utility is not None),
                            key=lambda c: c.rank):
        table.append(RankingRecord(identity=candidate.identity, utility=candidate.utility,
                                   features=dict(candidate.utility_features),
                                   rank=candidate.rank))
    return table
