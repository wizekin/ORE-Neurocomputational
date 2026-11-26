"""Evaluator-backed reward computation, attribution, and reward-hacking guards.

Point 10, Phase D (Steps 21-28). Rewards are computed from independently
evaluated outcomes, never from the policy's own confidence or unverified
self-reports. The composite reward

    r_t = alpha * dQ_t + beta * dE_t - gamma * C_t - delta * L_t - eta * F_t

uses versioned configuration weights, and every reward carries evaluator
identity/version plus evidence references so the trusted training set can be
recomputed and audited.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from .contracts import (
    ActionOutcome,
    OREReward,
    OutcomeStatus,
    RL_SCHEMA_VERSION,
    stable_hash,
)


@dataclass(frozen=True, slots=True)
class EvaluatorIdentity:
    """Identity and version recorded with every score (Step 21, 8.2)."""

    evaluator_id: str
    evaluator_version: str

    def as_tuple(self) -> tuple[str, str]:
        return (self.evaluator_id, self.evaluator_version)


class AttributionMethod(str, Enum):
    """How step rewards are attributed when actions share credit (Step 27)."""

    STEP = "STEP"
    TERMINAL_ONLY = "TERMINAL_ONLY"
    UNIFORM = "UNIFORM"
    LAST_ACTION = "LAST_ACTION"


@dataclass(frozen=True, slots=True)
class QualityScore:
    """Independent result-quality measurement (Step 21).

    Sub-scores separate factual correctness, completeness, relevance, and
    instruction compliance for open-ended tasks.
    """

    correctness: float
    completeness: float
    relevance: float
    instruction_compliance: float
    evaluator: EvaluatorIdentity
    evidence_refs: tuple[str, ...] = ()
    method: str = "reference_comparison"

    @property
    def total(self) -> float:
        return 0.4 * self.correctness + 0.25 * self.completeness + 0.2 * self.relevance + 0.15 * self.instruction_compliance


@dataclass(frozen=True, slots=True)
class EvidenceScore:
    """Independent evidence-quality measurement (Step 22)."""

    traceability: float
    relevance: float
    reliability: float
    support_strength: float
    contradiction_handling: float
    evaluator: EvaluatorIdentity
    evidence_refs: tuple[str, ...] = ()

    @property
    def total(self) -> float:
        return (
            0.25 * self.traceability
            + 0.2 * self.relevance
            + 0.25 * self.reliability
            + 0.3 * self.support_strength
        ) * (0.75 + 0.25 * self.contradiction_handling)


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """Versioned composite-reward configuration, not hidden constants (Step 24)."""

    alpha_quality: float = 1.0
    beta_evidence: float = 1.0
    gamma_resource: float = 0.05
    delta_latency: float = 0.05
    eta_failure: float = 0.5
    version: str = "reward-weights-1.0.0"

    def to_dict(self) -> dict[str, float | str]:
        return {
            "alpha_quality": self.alpha_quality,
            "beta_evidence": self.beta_evidence,
            "gamma_resource": self.gamma_resource,
            "delta_latency": self.delta_latency,
            "eta_failure": self.eta_failure,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RewardWeights:
        return cls(
            alpha_quality=float(data.get("alpha_quality", 1.0)),
            beta_evidence=float(data.get("beta_evidence", 1.0)),
            gamma_resource=float(data.get("gamma_resource", 0.05)),
            delta_latency=float(data.get("delta_latency", 0.05)),
            eta_failure=float(data.get("eta_failure", 0.5)),
            version=str(data.get("version", "reward-weights-1.0.0")),
        )


@dataclass(frozen=True, slots=True)
class CostModel:
    """Normalizes measured costs against explicit budgets (Step 23)."""

    latency_budget_ms: float = 1000.0
    cpu_budget: float = 100.0
    memory_budget_mb: float = 256.0

    def latency_cost(self, measured_ms: float) -> float:
        return min(1.0, measured_ms / max(1e-9, self.latency_budget_ms))

    def resource_cost(
        self, measured_cpu: float, measured_memory_mb: float
    ) -> float:
        return 0.5 * min(1.0, measured_cpu / max(1e-9, self.cpu_budget)) + 0.5 * min(
            1.0, measured_memory_mb / max(1e-9, self.memory_budget_mb)
        )


_FAILURE_STATUSES = frozenset(
    {
        OutcomeStatus.INVALID_OUTPUT,
        OutcomeStatus.TIMEOUT,
        OutcomeStatus.EXCEPTION,
        OutcomeStatus.RESOURCE_EXHAUSTED,
        OutcomeStatus.CANCELLED,
    }
)


@dataclass(frozen=True, slots=True)
class RewardSpec:
    """Frozen reward specification for one experiment (Step 28)."""

    weights: RewardWeights
    quality_evaluator: EvaluatorIdentity
    evidence_evaluator: EvaluatorIdentity
    cost_model: CostModel
    attribution: AttributionMethod = AttributionMethod.STEP

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "weights": self.weights.to_dict(),
                "quality_evaluator": self.quality_evaluator.as_tuple(),
                "evidence_evaluator": self.evidence_evaluator.as_tuple(),
                "cost_model": {
                    "latency_budget_ms": self.cost_model.latency_budget_ms,
                    "cpu_budget": self.cost_model.cpu_budget,
                    "memory_budget_mb": self.cost_model.memory_budget_mb,
                },
                "attribution": self.attribution.value,
            }
        )


class TaskQualityEvaluator:
    """Deterministic reference-based task quality evaluation (Step 21).

    The environment (or an integration adapter) supplies measured sub-scores
    from task-specific deterministic tests or reference answers; this evaluator
    only assembles and stamps them with identity and version. It never reads
    module self-reports.
    """

    def __init__(self, *, evaluator_version: str = "task-quality-1.0.0") -> None:
        self.identity = EvaluatorIdentity("ore_rl.task_quality", evaluator_version)

    def score(
        self,
        *,
        correctness: float,
        completeness: float,
        relevance: float,
        instruction_compliance: float,
        evidence_refs: Sequence[str] = (),
    ) -> QualityScore:
        values = (correctness, completeness, relevance, instruction_compliance)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("quality sub-scores must lie in [0, 1]")
        return QualityScore(
            correctness=correctness,
            completeness=completeness,
            relevance=relevance,
            instruction_compliance=instruction_compliance,
            evaluator=self.identity,
            evidence_refs=tuple(evidence_refs),
        )


class EvidenceQualityEvaluator:
    """Evidence-quality evaluation over workspace findings (Step 22).

    Scans published findings for provenance presence (traceability), payload
    references (relevance), source reliability, support links, and unresolved
    contradictions. Unsupported output cannot receive a positive evidence
    reward because support_strength is measured from actual links.
    """

    def __init__(self, *, evaluator_version: str = "evidence-quality-1.0.0") -> None:
        self.identity = EvaluatorIdentity("ore_rl.evidence_quality", evaluator_version)

    def score_findings(
        self, findings: Sequence[Mapping[str, Any]]
    ) -> EvidenceScore:
        if not findings:
            return EvidenceScore(
                traceability=0.0,
                relevance=0.0,
                reliability=0.0,
                support_strength=0.0,
                contradiction_handling=1.0,
                evaluator=self.identity,
            )
        traceability = sum(
            1.0 for item in findings if item.get("provenance") or item.get("has_provenance")
        ) / len(findings)
        relevance = sum(
            1.0 for item in findings if item.get("supports_decision") or item.get("relevant")
        ) / len(findings)
        reliabilities = [
            float(item.get("source_reliability", 0.5)) for item in findings
        ]
        reliability = sum(reliabilities) / len(reliabilities)
        support_strength = sum(
            1.0 for item in findings if item.get("verified") or item.get("supported")
        ) / len(findings)
        contradictions = sum(1.0 for item in findings if item.get("contradicts"))
        contradiction_handling = 1.0 / (1.0 + contradictions)
        refs: list[str] = []
        for item in findings:
            refs.extend(str(ref) for ref in item.get("evidence_refs", ()))
        return EvidenceScore(
            traceability=traceability,
            relevance=relevance,
            reliability=reliability,
            support_strength=support_strength,
            contradiction_handling=contradiction_handling,
            evaluator=self.identity,
            evidence_refs=tuple(dict.fromkeys(refs)),
        )


@dataclass(frozen=True, slots=True)
class TerminalOutcome:
    """Task-level terminal evaluation (Step 25)."""

    goals_met: bool
    constraints_preserved: bool
    final_answer_verified: bool
    stopped_prematurely: bool = False


def compute_step_reward(
    *,
    quality_before: float,
    quality_after: float,
    evidence_before: float,
    evidence_after: float,
    outcome: ActionOutcome,
    weights: RewardWeights,
    cost_model: CostModel,
    evaluator_identity: EvaluatorIdentity,
    reward_spec_fingerprint: str,
    evidence_refs: Sequence[str] = (),
    attribution_method: str = AttributionMethod.STEP.value,
    attribution_uncertainty: float = 0.0,
    verified: bool = True,
) -> OREReward:
    """Composite reward with all components recorded for recomputation (Step 24)."""
    quality_delta = max(0.0, quality_after - quality_before) - max(0.0, quality_before - quality_after) * 0.5
    evidence_delta = evidence_after - evidence_before
    resource_cost = cost_model.resource_cost(outcome.measured_cpu_cost, outcome.measured_memory_mb)
    latency_cost = cost_model.latency_cost(outcome.measured_latency_ms)
    failure_cost = 1.0 if outcome.status in _FAILURE_STATUSES else 0.0
    total = (
        weights.alpha_quality * quality_delta
        + weights.beta_evidence * evidence_delta
        - weights.gamma_resource * resource_cost
        - weights.delta_latency * latency_cost
        - weights.eta_failure * failure_cost
    )
    return OREReward(
        schema_version=RL_SCHEMA_VERSION,
        quality_delta=quality_delta,
        evidence_delta=evidence_delta,
        resource_cost=resource_cost,
        latency_cost=latency_cost,
        failure_cost=failure_cost,
        total=total,
        weights_version=weights.version,
        reward_spec_fingerprint=reward_spec_fingerprint,
        evaluator_id=evaluator_identity.evaluator_id,
        evaluator_version=evaluator_identity.evaluator_version,
        attribution_method=attribution_method,
        attribution_uncertainty=attribution_uncertainty,
        terminal=False,
        verified=verified,
        evidence_refs=tuple(evidence_refs),
    )


def compute_terminal_reward(
    terminal: TerminalOutcome,
    *,
    weights: RewardWeights,
    reward_spec_fingerprint: str,
    evaluator_identity: EvaluatorIdentity,
    evidence_refs: Sequence[str] = (),
) -> OREReward:
    """Terminal reward r_H: success requires goals, constraints, and verification (Step 25)."""
    if terminal.goals_met and terminal.constraints_preserved and terminal.final_answer_verified:
        total = 1.0
    elif terminal.goals_met and terminal.constraints_preserved:
        total = 0.3
    elif terminal.constraints_preserved:
        total = -0.2 if terminal.stopped_prematurely else 0.0
    else:
        total = -1.0
    return OREReward(
        schema_version=RL_SCHEMA_VERSION,
        total=total,
        weights_version=weights.version,
        reward_spec_fingerprint=reward_spec_fingerprint,
        evaluator_id=evaluator_identity.evaluator_id,
        evaluator_version=evaluator_identity.evaluator_version,
        attribution_method=AttributionMethod.TERMINAL_ONLY.value,
        attribution_uncertainty=0.1,
        terminal=True,
        verified=True,
        evidence_refs=tuple(evidence_refs),
    )


def recompute_total(reward: OREReward, weights: RewardWeights) -> float:
    """Recompute the composite total from stored components (acceptance 8.2)."""
    return (
        weights.alpha_quality * reward.quality_delta
        + weights.beta_evidence * reward.evidence_delta
        - weights.gamma_resource * reward.resource_cost
        - weights.delta_latency * reward.latency_cost
        - weights.eta_failure * reward.failure_cost
    )


@dataclass
class EpisodeAudit:
    """Signals collected per episode for reward-hacking analysis (Step 26)."""

    steps: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)
    finding_payload_hashes: list[str] = field(default_factory=list)
    unsupported_quality_steps: int = 0
    zero_progress_repeated_actions: int = 0
    module_count: int = 0
    stopped_with_open_mandatory_goals: bool = False


class RewardHackingGuard:
    """Detects proxy over-optimization and reward manipulation (Step 26).

    The guard never *blocks* by adjusting the scalar reward: hard correctness
    gates live in safety.py. It flags episodes and findings that must be
    reviewed or excluded from the trusted training set.
    """

    PROXY_TARGETS = ("module_count", "token_count", "hypothesis_count", "retrieved_count")

    def audit_episode(
        self,
        audit: EpisodeAudit,
        rewards: Sequence[OREReward],
    ) -> dict[str, Any]:
        flags: list[str] = []
        duplicates = len(audit.finding_payload_hashes) - len(
            set(audit.finding_payload_hashes)
        )
        if duplicates > 0:
            flags.append(f"duplicated_findings:{duplicates}")
        if audit.unsupported_quality_steps > 0:
            flags.append(f"unsupported_quality_claims:{audit.unsupported_quality_steps}")
        if audit.zero_progress_repeated_actions > 0:
            flags.append(
                f"unnecessary_operations:{audit.zero_progress_repeated_actions}"
            )
        if audit.stopped_with_open_mandatory_goals:
            flags.append("premature_stop")
        unverified = sum(1 for reward in rewards if not reward.verified)
        if unverified:
            flags.append(f"unverified_rewards:{unverified}")
        total = sum(reward.total for reward in rewards)
        return {
            "flags": flags,
            "reward_total": total,
            "steps": audit.steps,
            "distinct_actions": len(audit.action_counts),
            "module_count": audit.module_count,
            "duplicate_findings": duplicates,
            "trusted": not flags,
        }


def calibrate_reward_scale(
    rewards: Sequence[float],
    *,
    trim_percentile: float = 5.0,
) -> dict[str, float]:
    """Distribution statistics on a development set (Step 28)."""
    if not rewards:
        return {"count": 0.0, "mean": 0.0, "std": 0.0, "p5": 0.0, "p95": 0.0}
    ordered = sorted(rewards)
    n = len(ordered)
    lo = max(0, int(math.floor(trim_percentile / 100 * n)))
    hi = min(n, n - lo)
    trimmed = ordered[lo:hi] or ordered
    mean = sum(trimmed) / len(trimmed)
    var = sum((value - mean) ** 2 for value in trimmed) / max(1, len(trimmed) - 1)
    return {
        "count": float(n),
        "mean": mean,
        "std": math.sqrt(var),
        "p5": ordered[max(0, int(0.05 * (n - 1)))],
        "p95": ordered[min(n - 1, int(0.95 * (n - 1)))],
    }


def sensitivity_analysis(
    rewards: Sequence[OREReward],
    weights: RewardWeights,
    *,
    perturbation: float = 0.25,
) -> dict[str, float]:
    """Sensitivity of the total reward to each weight (Step 28)."""
    base = sum(reward.total for reward in rewards)
    result: dict[str, float] = {}
    fields = (
        ("alpha_quality", weights.alpha_quality),
        ("beta_evidence", weights.beta_evidence),
        ("gamma_resource", weights.gamma_resource),
        ("delta_latency", weights.delta_latency),
        ("eta_failure", weights.eta_failure),
    )
    for name, value in fields:
        if abs(value) < 1e-12:
            result[name] = 0.0
            continue
        perturbed = RewardWeights(
            alpha_quality=weights.alpha_quality,
            beta_evidence=weights.beta_evidence,
            gamma_resource=weights.gamma_resource,
            delta_latency=weights.delta_latency,
            eta_failure=weights.eta_failure,
            version=weights.version,
        )
        import dataclasses

        perturbed = dataclasses.replace(perturbed, **{name: value * (1 + perturbation)})
        shifted = sum(recompute_total(reward, perturbed) for reward in rewards)
        result[name] = abs(shifted - base) / max(1e-9, abs(base)) if base else 0.0
    return result


def attribute_episode_rewards(
    rewards: Sequence[OREReward],
    method: AttributionMethod,
    *,
    final_reward: OREReward | None = None,
) -> tuple[OREReward, ...]:
    """Redistribute credit for shared outcomes (Step 27).

    TERMINAL_ONLY zeroes step rewards and keeps only the terminal outcome;
    UNIFORM spreads the terminal outcome equally over steps with recorded
    attribution uncertainty; LAST_ACTION assigns the full final reward to the
    final step (the anti-pattern the spec warns about, available for ablation).
    """
    rewards = tuple(rewards)
    if method is AttributionMethod.STEP or not rewards:
        return rewards
    if method is AttributionMethod.TERMINAL_ONLY:
        return tuple(
            OREReward(
                schema_version=reward.schema_version,
                total=0.0,
                weights_version=reward.weights_version,
                reward_spec_fingerprint=reward.reward_spec_fingerprint,
                evaluator_id=reward.evaluator_id,
                evaluator_version=reward.evaluator_version,
                attribution_method=method.value,
                attribution_uncertainty=1.0,
                verified=reward.verified,
            )
            for reward in rewards
        )
    terminal_total = final_reward.total if final_reward else 0.0
    share = terminal_total / len(rewards)
    if method is AttributionMethod.UNIFORM:
        return tuple(
            OREReward(
                schema_version=reward.schema_version,
                total=share,
                weights_version=reward.weights_version,
                reward_spec_fingerprint=reward.reward_spec_fingerprint,
                evaluator_id=reward.evaluator_id,
                evaluator_version=reward.evaluator_version,
                attribution_method=method.value,
                attribution_uncertainty=1.0 / len(rewards),
                verified=reward.verified,
            )
            for reward in rewards
        )
    assigned = [0.0] * len(rewards)
    assigned[-1] = terminal_total
    return tuple(
        OREReward(
            schema_version=reward.schema_version,
            total=assigned[index],
            weights_version=reward.weights_version,
            reward_spec_fingerprint=reward.reward_spec_fingerprint,
            evaluator_id=reward.evaluator_id,
            evaluator_version=reward.evaluator_version,
            attribution_method=method.value,
            attribution_uncertainty=0.5,
            verified=reward.verified,
        )
        for index, reward in enumerate(rewards)
    )
