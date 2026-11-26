"""Contextual bandits: UCB, LinUCB, Thompson Sampling (Steps 34-38, 8.3)."""
import math

import numpy as np

from ore_rl.contracts import OREState
from ore_rl.policies.base import ProgressBoundaryPolicy, RandomEligiblePolicy
from ore_rl.policies.bandit import (
    EmpiricalBandit,
    LinUCBPolicy,
    ThompsonSamplingPolicy,
    UCBPolicy,
)


def _state(features: dict | None = None) -> OREState:
    values = features or {"f0": 0.5, "f1": 0.2}
    return OREState(
        state_id="s", task_id="t", task_family="f", encoder_version="1.0.0",
        feature_names=tuple(values), features=values,
    )


def test_ucb_matches_the_specified_formula():
    """Step 35: UCB(a) = mu_a + c*sqrt(ln(1+t)/(1+n_a)) over eligible actions only."""
    bandit = UCBPolicy(exploration=0.5)
    bandit.decisions = 10
    bandit.update(_state(), "a", 0.6)
    bandit.update(_state(), "a", 0.8)
    n = 2
    expected = 0.7 + 0.5 * math.sqrt(math.log(11) / (1 + n))
    assert math.isclose(bandit._ucb("a"), expected, abs_tol=1e-12)
    unseen = 0.5 * math.sqrt(math.log(11) / 1.0)
    assert math.isclose(bandit._ucb("b"), unseen, abs_tol=1e-12)


def test_ucb_never_proposes_ineligible_actions():
    """Eligibility masks are respected by construction (Step 38)."""
    bandit = UCBPolicy(exploration=0.9)
    for step in range(5):
        decision = bandit.propose_actions(_state(), ["eligible_a", "eligible_b"])
        bandit.update(_state(), decision.chosen_action_id, 0.1)
    decision = bandit.propose_actions(_state(), ["only_eligible"])
    assert decision.chosen_action_id == "only_eligible"
    assert all(item.action_id == "only_eligible" for item in decision.ranked)


def test_linucb_update_uses_only_validated_feedback():
    """Steps 36/38: A and b update only when the reward is verified."""
    policy = LinUCBPolicy(alpha=0.3, ridge=1.0)
    state = _state()
    policy.update(state, "a", 1.0, verified=False)
    assert policy.A.get("a") is None
    policy.update(state, "a", 1.0, verified=True)
    assert policy.A["a"].shape[0] == len(state.feature_names)
    raw = np.array([0.5, 0.2])
    x = raw / np.linalg.norm(raw)  # LinUCB contexts are L2-normalized
    assert np.allclose(policy.A["a"], np.eye(2) * 1.0 + np.outer(x, x))
    assert np.allclose(policy.b["a"], x)


def test_linucb_score_matches_the_confidence_bonus_formula():
    """Step 36: p(a) = x^T A^-1 b + alpha*sqrt(x^T A^-1 x)."""
    policy = LinUCBPolicy(alpha=0.7, ridge=2.0)
    state = _state({"f0": 1.0, "f1": 0.0})
    policy.update(state, "a", 2.0)
    x = np.array([1.0, 0.0])
    A = np.eye(2) * 2.0 + np.outer(x, x)
    b = x * 2.0
    A_inv = np.linalg.inv(A)
    expected = x @ (A_inv @ b) + 0.7 * math.sqrt(x @ A_inv @ x)
    decision = policy.propose_actions(_state({"f0": 1.0, "f1": 0.0}), ["a", "b"])
    scores = {item.action_id: item.score for item in decision.ranked}
    assert math.isclose(scores["a"], expected, abs_tol=1e-9)
    assert decision.chosen_action_id == "a"


def test_thompson_sampling_is_seed_reproducible():
    """Step 37: posterior draws use recorded seeds; identical seeds agree."""
    first = ThompsonSamplingPolicy(seed=42)
    second = ThompsonSamplingPolicy(seed=42)
    state = _state({"f0": 0.7, "f1": 0.1})
    for _ in range(3):
        d1 = first.propose_actions(state, ["a", "b"])
        d2 = second.propose_actions(state, ["a", "b"])
        assert d1.chosen_action_id == d2.chosen_action_id
    other = ThompsonSamplingPolicy(seed=43)
    choices_a = {first.propose_actions(state, ["a", "b"]).chosen_action_id for _ in range(5)}
    choices_b = {other.propose_actions(state, ["a", "b"]).chosen_action_id for _ in range(5)}
    assert choices_a <= {"a", "b"} and choices_b <= {"a", "b"}


def test_empirical_bandit_learns_the_best_arm():
    """Step 34: the simple empirical baseline concentrates on the better action."""
    bandit = EmpiricalBandit(seed=0, epsilon=0.0)
    for _ in range(30):
        decision = bandit.propose_actions(_state(), ["good", "bad"])
        bandit.update(_state(), decision.chosen_action_id, 1.0 if decision.chosen_action_id == "good" else 0.0)
    assert bandit.propose_actions(_state(), ["good", "bad"]).chosen_action_id == "good"


def test_bandit_beats_random_selection_on_a_simulated_problem():
    """Acceptance 8.3: bandit performance is compared against random eligible selection."""
    rng = np.random.default_rng(7)
    policy = LinUCBPolicy(seed=7, alpha=0.25)
    random_policy = RandomEligiblePolicy(seed=8)
    context_pool = [
        {"f0": float(rng.random()), "f1": float(rng.random())} for _ in range(40)
    ]

    def true_reward(context: dict, action: str) -> float:
        return 1.0 if (action == "a") == (context["f0"] > 0.5) else 0.0

    for step in range(200):
        context = context_pool[step % len(context_pool)]
        state = _state(context)
        decision = policy.propose_actions(state, ["a", "b"])
        policy.update(state, decision.chosen_action_id, true_reward(context, decision.chosen_action_id))

    def cumulative(policy) -> float:
        total = 0.0
        for context in context_pool:
            state = _state(context)
            total += true_reward(context, policy.propose_actions(state, ["a", "b"]).chosen_action_id)
        return total

    assert cumulative(policy) > cumulative(random_policy) + 5.0


def test_progress_boundary_wrapper_keeps_stop_decisions_deterministic():
    """Step 2: the boundary wrapper stops on the coordinator rule, delegates progress."""
    inner = EmpiricalBandit(seed=1, epsilon=0.0)
    wrapper = ProgressBoundaryPolicy(inner)
    decision = wrapper.propose_actions(_state(), ["meta.stop_success", "run.x@depth1"])
    assert decision.chosen_action_id == "meta.stop_success"
    decision = wrapper.propose_actions(_state(), ["run.x@depth1", "run.y@depth2"])
    assert decision.chosen_action_id in ("run.x@depth1", "run.y@depth2")
    decision = wrapper.propose_actions(_state(), ["meta.stop_insufficient_evidence"])
    assert decision.chosen_action_id == "meta.stop_insufficient_evidence"
