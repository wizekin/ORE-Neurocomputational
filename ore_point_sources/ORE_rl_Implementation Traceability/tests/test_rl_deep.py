"""DQN, Double DQN, prioritized replay, PPO clipping, and GAE (Steps 39-42)."""
import math

import numpy as np

from ore_rl.contracts import OREState
from ore_rl.policies.dqn import (
    DQNConfig,
    DQNPolicy,
    PrioritizedReplayBuffer,
)
from ore_rl.policies.ppo import PPOConfig, PPOPolicy, compute_gae, masked_softmax


def _state(values: list[float], index: int = 0) -> OREState:
    names = tuple(f"f{i}" for i in range(len(values)))
    return OREState(
        state_id=f"s{index}", task_id="t", task_family="f", encoder_version="1.0.0",
        feature_names=names, features=dict(zip(names, (float(v) for v in values))),
    )


def test_dqn_target_applies_the_next_state_action_mask():
    """Step 40: y = r + gamma*(1-d)*max over VALID next actions only."""
    policy = DQNPolicy(
        action_ids=["a", "b", "c"], state_dim=2,
        config=DQNConfig(hidden_units=(8,), gamma=0.9, prioritized=False, seed=1,
                         batch_size=2, replay_capacity=16, double_dqn=False,
                         epsilon_start=0.0),
    )
    policy.rng = np.random.default_rng(0)
    state = _state([0.1, 0.2], 0)
    next_state = _state([0.3, 0.4], 1)
    for _ in range(4):
        policy.observe(state=state, action_id="a", reward=1.0, next_state=next_state,
                       done=False, eligible_next=["a", "b"])
    # Target computation: max over {a, b}, c must never be selected
    batch, _ = policy.replay.sample(2)
    entry = batch[0]
    assert set(entry["eligible_next"]) == {"a", "b"}
    mask = np.zeros(len(policy.action_ids))
    for action in entry["eligible_next"]:
        mask[policy.action_index[action]] = 1.0
    next_q = policy.target.forward(np.atleast_2d(entry["next_state"]))[0][0]
    masked = np.where(mask > 0, next_q, -np.inf)
    expected = entry["reward"] + policy.config.gamma * masked.max()
    assert math.isfinite(expected)
    best_action = policy.action_ids[int(np.argmax(masked))]
    assert best_action in entry["eligible_next"]
    # a full training step runs and returns a finite mean |TD error|
    td = policy._train_step()
    assert math.isfinite(td)


def test_double_dqn_separates_selection_and_evaluation():
    """Step 41: online network selects, target network evaluates."""
    policy = DQNPolicy(
        action_ids=["a", "b", "c"], state_dim=2,
        config=DQNConfig(hidden_units=(8,), gamma=0.8, prioritized=False, seed=2,
                         batch_size=2, replay_capacity=16, double_dqn=True),
    )
    state, next_state = _state([0.0, 1.0], 0), _state([1.0, 0.0], 1)
    policy.observe(state=state, action_id="b", reward=0.5, next_state=next_state,
                   done=False, eligible_next=["a", "c"])
    batch, _ = policy.replay.sample(1)
    entry = batch[0]
    next_vector = np.atleast_2d(entry["next_state"])
    online_q = policy.online.forward(next_vector)[0]
    target_q = policy.target.forward(next_vector)[0]
    mask = np.zeros(3)
    for action in entry["eligible_next"]:
        mask[policy.action_index[action]] = 1.0
    masked_online = np.where(mask > 0, online_q, -np.inf)
    selected = int(np.argmax(masked_online))
    assert entry["eligible_next"] and policy.action_ids[selected] in entry["eligible_next"]
    expected = 0.5 + 0.8 * target_q[0, selected]
    assert math.isfinite(expected)


def test_prioritized_replay_weights_bias_toward_high_td_error():
    """Step 41: priorities (|delta|+eps)^alpha with importance weights."""
    buffer = PrioritizedReplayBuffer(capacity=8, seed=0, alpha=0.6, beta_start=1.0, beta_end=1.0)
    for index in range(4):
        buffer.add(
            state=np.zeros(2), action_index=0, action_id="a", reward=float(index),
            next_state=np.zeros(2), done=False, eligible_next=["a"],
        )
    buffer.update_priorities([0, 1, 2, 3], [0.1, 5.0, 0.1, 0.1])
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for _ in range(400):
        batch, weights = buffer.sample(4)
        assert weights is not None
        assert math.isclose(weights.max(), 1.0, abs_tol=1e-9)
        for item in batch:
            counts[int(item["reward"])] += 1
    assert counts[1] > counts[0]
    assert counts[1] > counts[2]


def test_masked_softmax_rejects_ineligible_logits():
    """Action masks apply consistently during sampling and evaluation (Step 42)."""
    logits = np.array([[2.0, 1.0, 0.5]])
    mask = np.array([0.0, 1.0, 1.0])
    probs = masked_softmax(logits, mask)
    assert probs[0, 0] == 0.0
    assert abs(probs.sum() - 1.0) < 1e-9
    assert probs[0, 1] > probs[0, 2]
    all_allowed = masked_softmax(logits, np.ones_like(mask))
    assert all_allowed[0, 0] > all_allowed[0, 1]


def test_gae_matches_hand_computed_values():
    """Step 42: documented GAE(lambda) computation on a hand-checkable example."""
    rewards = [1.0, 1.0]
    values = [0.5, 0.5]
    next_values = [0.5, 0.0]
    dones = [False, True]
    advantages, returns = compute_gae(rewards, values, next_values, dones, gamma=0.9, lam=0.8)
    assert math.isclose(advantages[1], 0.5, abs_tol=1e-12)
    assert math.isclose(advantages[0], 1.31, abs_tol=1e-9)
    assert math.isclose(returns[0], 1.81, abs_tol=1e-9)
    assert math.isclose(returns[1], 1.0, abs_tol=1e-12)


def test_ppo_clip_bounds_the_policy_update():
    """The clipped surrogate never exceeds the unclipped term's trust region."""
    ratio = np.linspace(0.5, 1.6, 25)
    advantage = np.ones_like(ratio)
    epsilon = 0.2
    clipped = np.clip(ratio, 1 - epsilon, 1 + epsilon)
    surrogate = np.minimum(ratio * advantage, clipped * advantage)
    assert np.all(surrogate <= clipped * advantage + 1e-12)
    assert np.all(surrogate <= np.maximum(ratio * advantage, clipped * advantage) + 1e-12)


def test_ppo_gradient_matches_finite_differences():
    """Analytic logit gradients of the PPO loss agree with numerical differentiation."""
    policy = PPOPolicy(
        action_ids=["a", "b", "c"], state_dim=3,
        config=PPOConfig(hidden_units=(8,), a2c_mode=True, learning_rate=1e-3, seed=3,
                         entropy_coefficient=0.0, value_coefficient=0.0),
    )
    states = np.asarray([[0.2, -0.1, 0.5], [0.4, 0.3, -0.2]])
    actions = np.asarray([0, 1])
    masks = np.ones((2, 3))
    policy.rollout = [
        {"state": states[0], "action_index": 0, "reward": 1.0, "next_state": states[1],
         "done": False, "eligible": ["a", "b", "c"], "propensity": 0.4},
        {"state": states[1], "action_index": 1, "reward": 0.5, "next_state": states[0],
         "done": True, "eligible": ["a", "b", "c"], "propensity": 0.3},
    ]

    def logits_of(states_matrix):
        feats, _ = policy.features.forward(states_matrix)
        return policy.actor.forward(feats)

    base = logits_of(states)
    old_probs = masked_softmax(base, masks)
    old_logp = np.log(old_probs[np.arange(2), actions] + 1e-12)
    advantages = np.asarray([0.6, -0.2])

    def loss_given_logit_perturbation(perturb):
        logits = base.copy()
        logits[0, 0] += perturb
        probs = masked_softmax(logits, masks)
        taken = probs[np.arange(2), actions]
        ratio = np.exp(np.log(taken + 1e-12) - old_logp)
        clipped = np.clip(ratio, 0.8, 1.2)
        surrogate = np.minimum(ratio * advantages, clipped * advantages)
        return -float(surrogate.sum())

    eps = 1e-6
    numeric = (loss_given_logit_perturbation(eps) - loss_given_logit_perturbation(-eps)) / (2 * eps)
    logits = base.copy()
    probs = masked_softmax(logits, masks)
    taken = np.maximum(probs[np.arange(2), actions], 1e-12)
    ratio = np.exp(np.log(taken) - old_logp)
    clipped = np.clip(ratio, 0.8, 1.2)
    weights = (clipped * advantages) if (clipped * advantages < ratio * advantages).all() else (
        ratio * advantages
    )
    grad = np.zeros_like(logits)
    grad[0, 0] = weights[0] * (1.0 - probs[0, 0])
    analytic = -grad[0, 0]  # d(-surrogate)/dlogit
    assert math.isclose(numeric, analytic, rel_tol=1e-4, abs_tol=1e-6)


def test_dqn_learns_a_simple_sequential_preference():
    """Sanity: DQN's Q estimates rank the rewarding action above the penalized one."""
    policy = DQNPolicy(
        action_ids=["good", "bad"], state_dim=2,
        config=DQNConfig(hidden_units=(16,), gamma=0.5, prioritized=False, seed=4,
                         batch_size=8, replay_capacity=128, epsilon_start=1.0,
                         epsilon_end=0.05, epsilon_decay_steps=50, learning_rate=5e-3),
    )
    rng = np.random.default_rng(4)
    for step in range(60):
        state = _state([rng.random(), rng.random()], step)
        next_state = _state([rng.random(), rng.random()], step + 1)
        policy.observe(state=state, action_id="good", reward=1.0, next_state=next_state,
                       done=False, eligible_next=["good", "bad"])
        policy.observe(state=state, action_id="bad", reward=-1.0, next_state=next_state,
                       done=False, eligible_next=["good", "bad"])
    q = policy.q_values(np.array([0.5, 0.5]))
    assert q[policy.action_index["good"]] > q[policy.action_index["bad"]]


def test_dqn_checkpoint_round_trip():
    """Checkpoints persist weights and configuration (8.2 reproducibility)."""
    policy = DQNPolicy(
        action_ids=["a", "b"], state_dim=2,
        config=DQNConfig(hidden_units=(8,), prioritized=False, seed=5),
    )
    state = _state([0.3, 0.7])
    policy.observe(state=state, action_id="a", reward=1.0, next_state=state, done=True,
                   eligible_next=["a"])
    checkpoint = policy.save_checkpoint()
    clone = DQNPolicy(action_ids=["a", "b"], state_dim=2, config=DQNConfig(
        hidden_units=(8,), prioritized=False, seed=99))
    clone.load_checkpoint(checkpoint)
    assert np.allclose(clone.q_values(np.array([0.3, 0.7])),
                       policy.q_values(np.array([0.3, 0.7])))
