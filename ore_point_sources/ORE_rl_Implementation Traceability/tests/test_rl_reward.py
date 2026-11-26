"""Evaluator-backed rewards, attribution, calibration, and hacking guards (Steps 21-28)."""
import math

from ore_rl.contracts import ActionOutcome, OutcomeStatus, OREReward, RL_SCHEMA_VERSION
from ore_rl.reward import (
    AttributionMethod,
    CostModel,
    EpisodeAudit,
    EvidenceQualityEvaluator,
    EvaluatorIdentity,
    RewardHackingGuard,
    RewardSpec,
    RewardWeights,
    TaskQualityEvaluator,
    TerminalOutcome,
    attribute_episode_rewards,
    calibrate_reward_scale,
    compute_step_reward,
    compute_terminal_reward,
    recompute_total,
    sensitivity_analysis,
)


def _identity() -> EvaluatorIdentity:
    return EvaluatorIdentity("test.evaluator", "1.2.3")


def test_composite_reward_matches_formula_and_recomputation():
    """Step 24 + acceptance 8.2: totals are the weighted composite of stored components."""
    weights = RewardWeights(alpha_quality=2.0, beta_evidence=1.0, gamma_resource=0.1,
                            delta_latency=0.05, eta_failure=0.5)
    reward = compute_step_reward(
        quality_before=0.2, quality_after=0.6,
        evidence_before=0.1, evidence_after=0.5,
        outcome=ActionOutcome(status=OutcomeStatus.SUCCESS, measured_latency_ms=100.0,
                              measured_cpu_cost=10.0, measured_memory_mb=10.0),
        weights=weights,
        cost_model=CostModel(latency_budget_ms=1000.0, cpu_budget=100.0, memory_budget_mb=100.0),
        evaluator_identity=_identity(),
        reward_spec_fingerprint="spec-1",
    )
    expected = (
        weights.alpha_quality * reward.quality_delta
        + weights.beta_evidence * reward.evidence_delta
        - weights.gamma_resource * reward.resource_cost
        - weights.delta_latency * reward.latency_cost
        - weights.eta_failure * reward.failure_cost
    )
    assert math.isclose(reward.total, expected, abs_tol=1e-12)
    assert math.isclose(recompute_total(reward, weights), reward.total, abs_tol=1e-12)
    assert reward.verified and reward.evaluator_id == "test.evaluator"
    assert reward.evaluator_version == "1.2.3"


def test_failure_statuses_incur_failure_cost():
    """Invalid, timed-out, and crashed work is penalized as a distinct component."""
    weights = RewardWeights()
    for status in (OutcomeStatus.EXCEPTION, OutcomeStatus.TIMEOUT, OutcomeStatus.INVALID_OUTPUT,
                   OutcomeStatus.RESOURCE_EXHAUSTED):
        reward = compute_step_reward(
            quality_before=0.5, quality_after=0.5, evidence_before=0.3, evidence_after=0.3,
            outcome=ActionOutcome(status=status),
            weights=weights, cost_model=CostModel(),
            evaluator_identity=_identity(), reward_spec_fingerprint="f",
        )
        assert reward.failure_cost == 1.0
    ok = compute_step_reward(
        quality_before=0.5, quality_after=0.5, evidence_before=0.3, evidence_after=0.3,
        outcome=ActionOutcome(status=OutcomeStatus.SUCCESS),
        weights=weights, cost_model=CostModel(),
        evaluator_identity=_identity(), reward_spec_fingerprint="f",
    )
    assert ok.failure_cost == 0.0


def test_terminal_rewards_distinguish_success_and_insufficient_evidence():
    """Step 25: success needs goals, constraints, and verification; premature stop is penalized."""
    weights = RewardWeights()
    success = compute_terminal_reward(
        TerminalOutcome(goals_met=True, constraints_preserved=True, final_answer_verified=True),
        weights=weights, reward_spec_fingerprint="f", evaluator_identity=_identity(),
    )
    unverified = compute_terminal_reward(
        TerminalOutcome(goals_met=True, constraints_preserved=True, final_answer_verified=False),
        weights=weights, reward_spec_fingerprint="f", evaluator_identity=_identity(),
    )
    premature = compute_terminal_reward(
        TerminalOutcome(goals_met=False, constraints_preserved=True, final_answer_verified=False,
                        stopped_prematurely=True),
        weights=weights, reward_spec_fingerprint="f", evaluator_identity=_identity(),
    )
    violated = compute_terminal_reward(
        TerminalOutcome(goals_met=False, constraints_preserved=False, final_answer_verified=False),
        weights=weights, reward_spec_fingerprint="f", evaluator_identity=_identity(),
    )
    assert success.total > unverified.total > premature.total > violated.total
    assert success.total == 1.0 and violated.total == -1.0


def test_reward_weights_are_versioned_not_hidden():
    """Step 24: weights carry a version and round-trip through configuration."""
    weights = RewardWeights(alpha_quality=1.5, version="experiment-42")
    data = weights.to_dict()
    assert data["version"] == "experiment-42"
    restored = RewardWeights.from_dict(data)
    assert restored == weights
    spec = RewardSpec(
        weights=weights, quality_evaluator=_identity(), evidence_evaluator=_identity(),
        cost_model=CostModel(),
    )
    assert spec.fingerprint
    assert spec.fingerprint == RewardSpec(
        weights=weights, quality_evaluator=_identity(), evidence_evaluator=_identity(),
        cost_model=CostModel(),
    ).fingerprint


def test_quality_scores_require_in_range_subscores():
    """Step 21: evaluator sub-scores are validated, separating correctness and compliance."""
    evaluator = TaskQualityEvaluator(evaluator_version="q-2")
    score = evaluator.score(correctness=1.0, completeness=0.8, relevance=0.9,
                            instruction_compliance=1.0)
    assert score.evaluator.evaluator_version == "q-2"
    assert 0.0 <= score.total <= 1.0
    try:
        evaluator.score(correctness=1.5, completeness=0.8, relevance=0.9, instruction_compliance=1.0)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_evidence_quality_scores_support_and_traceability():
    """Step 22: unsupported output cannot earn a positive evidence reward."""
    evaluator = EvidenceQualityEvaluator()
    supported = evaluator.score_findings([
        {"has_provenance": True, "relevant": True, "source_reliability": 0.9,
         "supported": True, "contradicts": False},
    ])
    unsupported = evaluator.score_findings([
        {"has_provenance": False, "relevant": False, "source_reliability": 0.1,
         "supported": False, "contradicts": True},
    ])
    assert supported.total > 0.5 > unsupported.total
    empty = evaluator.score_findings([])
    assert empty.support_strength == 0.0


def test_reward_hacking_guard_flags_known_proxies():
    """Step 26: duplicated findings, unsupported claims, and premature stops are flagged."""
    guard = RewardHackingGuard()
    audit = EpisodeAudit(
        steps=6,
        finding_payload_hashes=["h1", "h1", "h2"],
        unsupported_quality_steps=1,
        zero_progress_repeated_actions=2,
        module_count=12,
        stopped_with_open_mandatory_goals=True,
    )
    rewards = [OREReward(schema_version=RL_SCHEMA_VERSION, total=0.1, verified=True) for _ in range(6)]
    report = guard.audit_episode(audit, rewards)
    assert not report["trusted"]
    assert any(flag.startswith("duplicated_findings") for flag in report["flags"])
    assert any(flag.startswith("unnecessary_operations") for flag in report["flags"])
    assert "premature_stop" in report["flags"]
    clean = guard.audit_episode(EpisodeAudit(steps=2, module_count=2), rewards[:2])
    assert clean["trusted"]


def test_reward_never_counts_module_or_token_totals():
    """Step 26: the composite uses deltas and costs only; counts are audit fields."""
    weights = RewardWeights()
    assert not any(
        field in weights.to_dict()
        for field in RewardHackingGuard.PROXY_TARGETS
    )


def test_attribution_methods_redistribute_credit():
    """Step 27: attribution method and uncertainty are recorded with every reward."""
    rewards = [
        OREReward(schema_version=RL_SCHEMA_VERSION, total=0.1, verified=True) for _ in range(4)
    ]
    terminal = OREReward(schema_version=RL_SCHEMA_VERSION, total=2.0, verified=True)
    uniform = attribute_episode_rewards(rewards, AttributionMethod.UNIFORM, final_reward=terminal)
    assert all(math.isclose(item.total, 0.5) for item in uniform)
    assert all(item.attribution_method == "UNIFORM" for item in uniform)
    last = attribute_episode_rewards(rewards, AttributionMethod.LAST_ACTION, final_reward=terminal)
    assert last[-1].total == 2.0 and sum(item.total for item in last[:-1]) == 0.0
    terminal_only = attribute_episode_rewards(rewards, AttributionMethod.TERMINAL_ONLY)
    assert all(item.total == 0.0 for item in terminal_only)


def test_calibration_and_sensitivity_use_development_statistics():
    """Step 28: reward scale calibration and weight sensitivity on dev data."""
    rewards = [0.1 * i for i in range(20)]
    stats = calibrate_reward_scale(rewards)
    assert stats["count"] == 20
    assert stats["mean"] > 0
    weights = RewardWeights()
    rewards_objs = [
        OREReward(schema_version=RL_SCHEMA_VERSION, quality_delta=0.1, evidence_delta=0.05,
                  resource_cost=0.01, latency_cost=0.01, failure_cost=0.0,
                  total=0.14, verified=True)
        for _ in range(5)
    ]
    sensitivity = sensitivity_analysis(rewards_objs, weights)
    assert set(sensitivity) == {
        "alpha_quality", "beta_evidence", "gamma_resource", "delta_latency", "eta_failure"
    }
    assert sensitivity["alpha_quality"] > sensitivity["eta_failure"]
