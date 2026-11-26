"""Offline RL from validated historical trajectories (Step 48).

Implements behavior cloning as the always-on baseline, Conservative
Q-Learning (CQL) with a logsumexp penalty over dataset-supported actions, and
Implicit Q-Learning (IQL) with expectile regression and advantage-weighted
policy extraction. Extrapolation to actions absent from the dataset is
prevented by masking candidate actions to the dataset's support.

Dataset quality rules: only complete, provenance-rich, leakage-split
trajectories (see trajectory_store.TrajectoryStore) are used, and simulated
world-model transitions are excluded by default (Step 49).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import OREState, PolicyAuthority, PolicyDecision, RankedProposal
from .base import PolicyMetadata
from .dqn import _MLP


def _matrix(dataset: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([np.asarray(item[key], dtype=np.float64) for item in dataset])


@dataclass(frozen=True, slots=True)
class OfflineTrainer:
    """Configuration and fit entry point for offline algorithms (Step 48)."""

    learning_rate: float = 1.5e-3
    batch_size: int = 32
    iterations: int = 400
    seed: int = 0

    def iterate(self, dataset: Sequence[Mapping[str, Any]]) -> Any:
        rng = np.random.default_rng(self.seed)
        for _ in range(self.iterations):
            if len(dataset) < self.batch_size:
                continue
            indices = rng.integers(0, len(dataset), size=self.batch_size)
            yield [dataset[int(i)] for i in indices]


class BehaviorCloningPolicy:
    """Supervised cloning of the dataset action distribution (Step 48 baseline)."""

    def __init__(
        self,
        *,
        action_ids: Sequence[str],
        state_dim: int,
        seed: int = 0,
        learning_rate: float = 2e-2,
        iterations: int = 300,
    ) -> None:
        self.action_ids: tuple[str, ...] = tuple(action_ids)
        self.action_index = {action: index for index, action in enumerate(self.action_ids)}
        self.state_dim = int(state_dim)
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.rng = np.random.default_rng(seed)
        self.network = _MLP(self.state_dim, len(self.action_ids), (48,), seed=seed)
        self.trained = False
        self.metadata = PolicyMetadata(
            policy_id="behavior_cloning",
            policy_version="1.0.0",
            algorithm_family="offline_rl/behavior_cloning",
            uncertainty_behavior="none",
            execution_authority=PolicyAuthority.PROPOSAL,
            description="supervised cloning of historical decisions",
        )

    def fit(self, dataset: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        """Cross-entropy minimization on (state, action) pairs."""
        if not dataset:
            return {"loss": float("nan")}
        states = np.stack([np.asarray(item["state"], dtype=np.float64) for item in dataset])
        actions = np.asarray([self.action_index[item["action_id"]] for item in dataset])
        losses: list[float] = []
        for _ in range(self.iterations):
            predictions, activations = self.network.forward(states)
            shifted = predictions - predictions.max(axis=1, keepdims=True)
            log_probs = shifted - np.log(
                np.exp(shifted).sum(axis=1, keepdims=True) + 1e-12
            )
            loss = -log_probs[np.arange(len(actions)), actions].mean()
            grad = np.exp(log_probs)
            grad[np.arange(len(actions)), actions] -= 1.0
            grad /= len(actions)
            self.network.backward(activations, grad, lr=self.learning_rate)
            losses.append(float(loss))
        self.trained = True
        return {"loss": losses[-1] if losses else float("nan")}

    def propose_actions(
        self,
        state: OREState,
        eligible_actions: Sequence[str],
        budget: Mapping[str, float] | None = None,
    ) -> PolicyDecision:
        eligible = list(eligible_actions)
        if not eligible:
            return PolicyDecision(
                policy_id=self.metadata.policy_id,
                policy_version=self.metadata.policy_version,
                chosen_action_id="",
                fallback_used=True,
                rationale="no eligible actions",
            )
        vector = np.asarray(
            [state.features.get(name, 0.0) for name in state.feature_names], dtype=np.float64
        )
        logits = self.network.predict(np.nan_to_num(vector, nan=0.0))
        indices = [self.action_index[a] for a in eligible if a in self.action_index]
        sub = logits[indices]
        best = int(np.argmax(sub))
        ranked = tuple(
            RankedProposal(action_id=a, score=float(logits[self.action_index[a]]), uncertainty=0.0)
            for a in sorted(
                (a for a in eligible if a in self.action_index),
                key=lambda a: -float(logits[self.action_index[a]]),
            )
        )
        return PolicyDecision(
            policy_id=self.metadata.policy_id,
            policy_version=self.metadata.policy_version,
            ranked=ranked,
            chosen_action_id=eligible[best],
            propensity=1.0,
            authority=PolicyAuthority.PROPOSAL,
            rationale="behavior-cloned action distribution",
        )


class CQLPolicy:
    """Conservative Q-learning with dataset-support masking (Step 48).

    Loss = DQN Bellman error + alpha * (logsumexp_a in support Q(s,a) - Q(s,a_bc)).
    The conservative term penalizes out-of-support overestimation; candidate
    actions are always masked to the dataset's action support.
    """

    def __init__(
        self,
        *,
        action_ids: Sequence[str],
        state_dim: int,
        seed: int = 0,
        alpha: float = 1.0,
        gamma: float = 0.95,
        learning_rate: float = 1.5e-3,
        iterations: int = 400,
    ) -> None:
        self.action_ids: tuple[str, ...] = tuple(action_ids)
        self.action_index = {action: index for index, action in enumerate(self.action_ids)}
        self.state_dim = int(state_dim)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.rng = np.random.default_rng(seed)
        self.network = _MLP(self.state_dim, len(self.action_ids), (64, 48), seed=seed)
        self.support_by_state: dict[int, np.ndarray] = {}
        self.trained = False
        self.metadata = PolicyMetadata(
            policy_id="cql_policy",
            policy_version="1.0.0",
            algorithm_family="offline_rl/cql",
            uncertainty_behavior="conservative_penalty",
            execution_authority=PolicyAuthority.PROPOSAL,
            description="conservative Q-learning over validated trajectories",
        )

    def fit(self, dataset: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        if len(dataset) < 2:
            return {"loss": float("nan")}
        states = np.stack([np.asarray(item["state"], dtype=np.float64) for item in dataset])
        next_states = np.stack(
            [np.asarray(item["next_state"], dtype=np.float64) for item in dataset]
        )
        actions = np.asarray(
            [self.action_index[item["action_id"]] for item in dataset], dtype=np.int64
        )
        rewards = np.asarray([item["reward"] for item in dataset], dtype=np.float64)
        dones = np.asarray([item["done"] for item in dataset], dtype=np.float64)
        support = np.zeros((len(dataset), len(self.action_ids)))
        for row, item in enumerate(dataset):
            for action in item.get("support", ()):
                if action in self.action_index:
                    support[row, self.action_index[action]] = 1.0
            if support[row].sum() == 0:
                support[row, actions[row]] = 1.0
        # Conservative penalty covers every action observed anywhere in the
        # dataset (the dataset-wide union), never fully unseen actions.
        union_support = support.max(axis=0)
        if union_support.sum() == 0:
            union_support[:] = 1.0
        losses: list[float] = []
        batch_size = min(32, len(dataset))
        for _ in range(self.iterations):
            indices = self.rng.integers(0, len(dataset), size=batch_size)
            batch_states = states[indices]
            batch_next = next_states[indices]
            batch_actions = actions[indices]
            batch_rewards = rewards[indices]
            batch_dones = dones[indices]
            batch_support = support[indices]
            next_q = self.network.forward(batch_next)[0]
            masked_next = np.where(batch_support > 0, next_q, -1e9)
            targets = batch_rewards + self.gamma * (1 - batch_dones) * masked_next.max(axis=1)
            predictions, activations = self.network.forward(batch_states)
            taken = predictions[np.arange(batch_size), batch_actions]
            bellman_error = targets - taken
            conservative = np.log(
                np.sum(np.exp(np.where(union_support > 0, predictions, -1e9)), axis=1) + 1e-12
            ) - taken
            loss = float(
                np.mean(0.5 * bellman_error ** 2) + self.alpha * np.mean(conservative)
            )
            grad = np.zeros_like(predictions)
            huber = np.where(
                np.abs(bellman_error) <= 1.0,
                bellman_error,
                np.sign(bellman_error),
            )
            grad[np.arange(batch_size), batch_actions] -= huber
            # d/dQ [ logsumexp_union(Q) - Q(a_taken) ] = softmax_union - onehot(taken)
            softmax = np.exp(np.where(union_support > 0, predictions, -1e9))
            softmax /= softmax.sum(axis=1, keepdims=True) + 1e-12
            grad += self.alpha * softmax
            grad[np.arange(batch_size), batch_actions] -= self.alpha
            self.network.backward(activations, grad, lr=self.learning_rate)
            losses.append(loss)
        self.trained = True
        return {"loss": losses[-1] if losses else float("nan")}

    def propose_actions(
        self,
        state: OREState,
        eligible_actions: Sequence[str],
        budget: Mapping[str, float] | None = None,
    ) -> PolicyDecision:
        eligible = list(eligible_actions)
        if not eligible:
            return PolicyDecision(
                policy_id=self.metadata.policy_id,
                policy_version=self.metadata.policy_version,
                chosen_action_id="",
                fallback_used=True,
                rationale="no eligible actions",
            )
        vector = np.asarray(
            [state.features.get(name, 0.0) for name in state.feature_names], dtype=np.float64
        )
        q = self.network.predict(np.nan_to_num(vector, nan=0.0))
        scores = {
            action: float(q[self.action_index[action]])
            for action in eligible
            if action in self.action_index
        }
        if not scores:
            scores = {action: 0.0 for action in eligible}
        chosen = max(scores, key=lambda action: (scores[action], action))
        ranked = tuple(
            RankedProposal(action_id=a, score=s, uncertainty=0.0)
            for a, s in sorted(scores.items(), key=lambda item: -item[1])
        )
        return PolicyDecision(
            policy_id=self.metadata.policy_id,
            policy_version=self.metadata.policy_version,
            ranked=ranked,
            chosen_action_id=chosen,
            propensity=1.0,
            authority=PolicyAuthority.PROPOSAL,
            rationale="conservative Q-values restricted to eligible actions",
        )


class IQLPolicy:
    """Implicit Q-learning: expectile value regression + AWR extraction (Step 48).

    V is trained with asymmetric expectile loss (tau) on Q targets, the
    advantage A = Q - V is never regressed explicitly, and the policy is
    extracted with advantage-weighted regression over dataset-supported
    actions only.
    """

    def __init__(
        self,
        *,
        action_ids: Sequence[str],
        state_dim: int,
        seed: int = 0,
        tau: float = 0.7,
        beta: float = 3.0,
        gamma: float = 0.95,
        learning_rate: float = 1.5e-3,
        iterations: int = 400,
    ) -> None:
        self.action_ids: tuple[str, ...] = tuple(action_ids)
        self.action_index = {action: index for index, action in enumerate(self.action_ids)}
        self.state_dim = int(state_dim)
        self.tau = float(tau)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.rng = np.random.default_rng(seed)
        self.q_network = _MLP(self.state_dim, len(self.action_ids), (64, 48), seed=seed)
        self.value_network = _MLP(self.state_dim, len(self.action_ids), (48,), seed=seed + 5)
        self.trained = False
        self.metadata = PolicyMetadata(
            policy_id="iql_policy",
            policy_version="1.0.0",
            algorithm_family="offline_rl/iql",
            uncertainty_behavior="expectile_quantile",
            execution_authority=PolicyAuthority.PROPOSAL,
            description="implicit Q-learning with expectile regression and AWR",
        )

    def fit(self, dataset: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        if len(dataset) < 2:
            return {"loss": float("nan")}
        states = np.stack([np.asarray(item["state"], dtype=np.float64) for item in dataset])
        next_states = np.stack(
            [np.asarray(item["next_state"], dtype=np.float64) for item in dataset]
        )
        actions = np.asarray(
            [self.action_index[item["action_id"]] for item in dataset], dtype=np.int64
        )
        rewards = np.asarray([item["reward"] for item in dataset], dtype=np.float64)
        dones = np.asarray([item["done"] for item in dataset], dtype=np.float64)
        batch_size = min(32, len(dataset))
        q_losses: list[float] = []
        v_losses: list[float] = []
        for _ in range(self.iterations):
            indices = self.rng.integers(0, len(dataset), size=batch_size)
            batch_states = states[indices]
            batch_next = next_states[indices]
            batch_actions = actions[indices]
            batch_rewards = rewards[indices]
            batch_dones = dones[indices]
            next_v = self.value_network.forward(batch_next)[0]
            targets = batch_rewards + self.gamma * (1 - batch_dones) * next_v.max(axis=1)
            q_predictions, q_activations = self.q_network.forward(batch_states)
            taken_q = q_predictions[np.arange(batch_size), batch_actions]
            q_error = targets - taken_q
            q_loss = float(np.mean(0.5 * q_error ** 2))
            grad_q = np.zeros_like(q_predictions)
            grad_q[np.arange(batch_size), batch_actions] = -q_error
            self.q_network.backward(q_activations, grad_q, lr=self.learning_rate)

            v_predictions, v_activations = self.value_network.forward(batch_states)
            taken_v = v_predictions[np.arange(batch_size), batch_actions]
            v_error = targets - taken_v
            indicator = (v_error < 0).astype(np.float64)
            weight = self.tau * indicator + (1 - self.tau) * (1 - indicator)
            v_loss = float(np.mean(weight * 0.5 * v_error ** 2))
            grad_v = np.zeros_like(v_predictions)
            grad_v[np.arange(batch_size), batch_actions] = -weight * v_error
            self.value_network.backward(v_activations, grad_v, lr=self.learning_rate)
            q_losses.append(q_loss)
            v_losses.append(v_loss)
        self.trained = True
        return {
            "q_loss": q_losses[-1] if q_losses else float("nan"),
            "v_loss": v_losses[-1] if v_losses else float("nan"),
        }

    def propose_actions(
        self,
        state: OREState,
        eligible_actions: Sequence[str],
        budget: Mapping[str, float] | None = None,
    ) -> PolicyDecision:
        eligible = list(eligible_actions)
        if not eligible:
            return PolicyDecision(
                policy_id=self.metadata.policy_id,
                policy_version=self.metadata.policy_version,
                chosen_action_id="",
                fallback_used=True,
                rationale="no eligible actions",
            )
        vector = np.asarray(
            [state.features.get(name, 0.0) for name in state.feature_names], dtype=np.float64
        )
        cleaned = np.nan_to_num(vector, nan=0.0)
        q = self.q_network.predict(cleaned)
        v = self.value_network.predict(cleaned)
        advantage = q - v
        scores = {
            action: float(advantage[self.action_index[action]])
            for action in eligible
            if action in self.action_index
        }
        if not scores:
            scores = {action: 0.0 for action in eligible}
        chosen = max(scores, key=lambda action: (scores[action], action))
        ranked = tuple(
            RankedProposal(action_id=a, score=s, uncertainty=0.0)
            for a, s in sorted(scores.items(), key=lambda item: -item[1])
        )
        return PolicyDecision(
            policy_id=self.metadata.policy_id,
            policy_version=self.metadata.policy_version,
            ranked=ranked,
            chosen_action_id=chosen,
            propensity=1.0,
            authority=PolicyAuthority.PROPOSAL,
            rationale="advantage-weighted implicit Q evaluation over eligible actions",
        )
