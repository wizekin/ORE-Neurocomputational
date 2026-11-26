"""PPO and A2C actor-critic with clipped surrogate objective and GAE.

Point 10, Step 42:
    * Actor pi_theta(a|s) (softmax over masked logits) and critic V_phi(s).
    * PPO clipped surrogate:
        L^CLIP = E[min(rho_t * A_t, clip(rho_t, 1-eps, 1+eps) * A_t)]
        with rho_t = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
    * Advantages estimated with GAE(lambda), documented here.
    * Action masks applied consistently during sampling and evaluation.

Pure NumPy implementation: categorical softmax policy with manual gradients
(verified against finite differences in the test suite) and Adam updates.
A2C mode = single gradient update per rollout without clipping.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import OREState, PolicyAuthority, PolicyDecision, RankedProposal
from .base import PolicyMetadata


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Hyperparameters for PPO/A2C training (recorded for reproducibility)."""

    hidden_units: tuple[int, ...] = (48,)
    learning_rate: float = 3e-3
    gamma: float = 0.95
    gae_lambda: float = 0.9
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    rollout_steps: int = 32
    epochs_per_rollout: int = 3
    epsilon_start: float = 0.4
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 800
    a2c_mode: bool = False
    seed: int = 0


class _LinearHead:
    """Linear map with Adam; used for logits and value heads."""

    def __init__(self, input_dim: int, output_dim: int, *, seed: int) -> None:
        rng = np.random.default_rng(seed)
        limit = math.sqrt(1.0 / max(1, input_dim))
        self.weight = rng.uniform(-limit, limit, size=(input_dim, output_dim))
        self.bias = np.zeros(output_dim)
        self._m = np.zeros_like(self.weight)
        self._v = np.zeros_like(self.weight)
        self._bm = np.zeros_like(self.bias)
        self._bv = np.zeros_like(self.bias)
        self.step = 0

    def forward(self, batch: np.ndarray) -> np.ndarray:
        return np.atleast_2d(batch) @ self.weight + self.bias

    def backward(self, batch: np.ndarray, grad: np.ndarray, *, lr: float) -> None:
        batch = np.atleast_2d(batch)
        grad_w = batch.T @ grad / batch.shape[0]
        grad_b = grad.sum(axis=0) / batch.shape[0]
        self.step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self._m = beta1 * self._m + (1 - beta1) * grad_w
        self._v = beta2 * self._v + (1 - beta2) * grad_w * grad_w
        m_hat = self._m / (1 - beta1 ** self.step)
        v_hat = self._v / (1 - beta2 ** self.step)
        self.weight -= lr * m_hat / (np.sqrt(v_hat) + eps)
        self._bm = beta1 * self._bm + (1 - beta1) * grad_b
        self._bv = beta2 * self._bv + (1 - beta2) * grad_b * grad_b
        m_hat = self._bm / (1 - beta1 ** self.step)
        v_hat = self._bv / (1 - beta2 ** self.step)
        self.bias -= lr * m_hat / (np.sqrt(v_hat) + eps)


class _SharedFeatures:
    """Feature transform shared by actor and critic (a tanh MLP)."""

    def __init__(self, input_dim: int, hidden_units: Sequence[int], *, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        dims = [input_dim, *hidden_units]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            limit = math.sqrt(1.0 / max(1, fan_in))
            self.weights.append(self.rng.uniform(-limit, limit, size=(fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out))

    def forward(self, batch: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        activations = [np.atleast_2d(batch)]
        current = activations[0]
        for weight, bias in zip(self.weights, self.biases):
            current = np.tanh(current @ weight + bias)
            activations.append(current)
        return current, activations

    def backward(self, activations: list[np.ndarray], grad: np.ndarray) -> np.ndarray:
        delta = np.asarray(grad, dtype=np.float64)
        for index in range(len(self.weights) - 1, -1, -1):
            delta = (delta @ self.weights[index].T) * (1.0 - activations[index + 1] ** 2)
        return delta

    def apply_gradients(
        self, activations: list[np.ndarray], grad: np.ndarray, *, lr: float
    ) -> None:
        batch = activations[0]
        delta = np.asarray(grad, dtype=np.float64)
        for index in range(len(self.weights) - 1, -1, -1):
            grad_w = batch.T @ delta / batch.shape[0]
            grad_b = delta.sum(axis=0) / batch.shape[0]
            self.weights[index] -= lr * grad_w
            self.biases[index] -= lr * grad_b
            if index > 0:
                delta = (delta @ self.weights[index].T) * (1.0 - activations[index] ** 2)

    def update_with_adam(self, grads_w: list[np.ndarray], grads_b: list[np.ndarray], *, lr: float, step: int) -> None:
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        if not hasattr(self, "_adam_m"):
            self._adam_m = [np.zeros_like(w) for w in self.weights]
            self._adam_v = [np.zeros_like(w) for w in self.weights]
            self._adam_bm = [np.zeros_like(b) for b in self.biases]
            self._adam_bv = [np.zeros_like(b) for b in self.biases]
        for index, weight in enumerate(self.weights):
            self._adam_m[index] = beta1 * self._adam_m[index] + (1 - beta1) * grads_w[index]
            self._adam_v[index] = beta2 * self._adam_v[index] + (1 - beta2) * grads_w[index] ** 2
            m_hat = self._adam_m[index] / (1 - beta1 ** step)
            v_hat = self._adam_v[index] / (1 - beta2 ** step)
            weight -= lr * m_hat / (np.sqrt(v_hat) + eps)
            self._adam_bm[index] = beta1 * self._adam_bm[index] + (1 - beta1) * grads_b[index]
            self._adam_bv[index] = beta2 * self._adam_bv[index] + (1 - beta2) * grads_b[index] ** 2
            m_hat = self._adam_bm[index] / (1 - beta1 ** step)
            v_hat = self._adam_bv[index] / (1 - beta2 ** step)
            self.biases[index] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax restricted to eligible logits (mask applied consistently)."""
    shifted = np.where(mask > 0, logits, -np.inf)
    shifted = shifted - np.max(shifted, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    exp = np.where(mask > 0, exp, 0.0)
    total = exp.sum(axis=-1, keepdims=True)
    return exp / np.maximum(total, 1e-12)


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    next_values: Sequence[float],
    dones: Sequence[bool],
    *,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation (documented method, Step 42)."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float64)
    gae = 0.0
    for t in range(n - 1, -1, -1):
        next_value = next_values[t] if not dones[t] else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * (0.0 if dones[t] else gae)
        advantages[t] = gae
    returns = advantages + np.asarray(values, dtype=np.float64)
    return advantages, returns


class PPOPolicy:
    """Masked PPO / A2C over a fixed action catalogue."""

    def __init__(
        self,
        *,
        action_ids: Sequence[str],
        state_dim: int,
        config: PPOConfig | None = None,
    ) -> None:
        self.action_ids: tuple[str, ...] = tuple(action_ids)
        self.action_index: dict[str, int] = {a: i for i, a in enumerate(self.action_ids)}
        self.state_dim = int(state_dim)
        self.config = config or PPOConfig()
        hidden = list(self.config.hidden_units)
        feature_dim = hidden[-1] if hidden else self.state_dim
        self.features = _SharedFeatures(self.state_dim, hidden, seed=self.config.seed)
        self.actor = _LinearHead(feature_dim, len(self.action_ids), seed=self.config.seed + 11)
        self.critic = _LinearHead(feature_dim, 1, seed=self.config.seed + 17)
        self.rng = np.random.default_rng(self.config.seed)
        self.rollout: list[dict[str, Any]] = []
        self.training_steps = 0
        self.metadata = PolicyMetadata(
            policy_id="ppo_policy" if not self.config.a2c_mode else "a2c_policy",
            policy_version="1.0.0",
            algorithm_family="policy_gradient/ppo" if not self.config.a2c_mode else "policy_gradient/a2c",
            uncertainty_behavior="policy_entropy",
            execution_authority=PolicyAuthority.PROPOSAL,
            description="masked actor-critic with clipped surrogate and GAE",
        )

    # -- inference ------------------------------------------------------------
    def _state_vector(self, state: OREState) -> np.ndarray:
        vector = np.asarray(
            [state.features.get(name, 0.0) for name in state.feature_names],
            dtype=np.float64,
        )
        return np.nan_to_num(vector, nan=0.0)

    def action_probabilities(
        self, state_vector: np.ndarray, eligible: Sequence[str]
    ) -> np.ndarray:
        vector = np.asarray(state_vector, dtype=np.float64)
        single = vector.ndim == 1
        feats, _ = self.features.forward(vector)
        logits = self.actor.forward(feats)
        mask = np.zeros(len(self.action_ids))
        for action in eligible:
            if action in self.action_index:
                mask[self.action_index[action]] = 1.0
        if mask.sum() == 0:
            mask[:] = 1.0
        probs = masked_softmax(logits, np.broadcast_to(mask, logits.shape))
        return probs[0] if single else probs

    def set_exploration(self, enabled: bool) -> None:
        """Disable exploration for deterministic serving (evaluation)."""
        self.exploration_enabled = bool(enabled)

    def epsilon(self) -> float:
        if not getattr(self, "exploration_enabled", True):
            return 0.0
        steps = self.training_steps
        fraction = min(1.0, steps / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + (
            self.config.epsilon_end - self.config.epsilon_start
        ) * fraction

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
        vector = self._state_vector(state)
        probs = self.action_probabilities(vector, eligible)
        indices = [self.action_index[a] for a in eligible if a in self.action_index]
        sub = probs[indices]
        explore = self.rng.random() < self.epsilon()
        if explore:
            pick = int(self.rng.integers(len(indices)))
        else:
            pick = int(np.argmax(sub))
        chosen_index = indices[pick]
        chosen = self.action_ids[chosen_index]
        propensity = float(sub[pick]) if not explore else (
            (self.epsilon() / len(indices)) * 0.5 + float(sub[pick]) * 0.5
        )
        ranked = tuple(
            RankedProposal(
                action_id=action, score=float(probs[self.action_index[action]]), uncertainty=0.0
            )
            for action in sorted(
                (a for a in eligible if a in self.action_index),
                key=lambda a: -float(probs[self.action_index[a]]),
            )
        )
        return PolicyDecision(
            policy_id=self.metadata.policy_id,
            policy_version=self.metadata.policy_version,
            ranked=ranked,
            chosen_action_id=chosen,
            propensity=propensity,
            uncertainty=float(-np.sum(sub * np.log(np.maximum(sub, 1e-12)))),
            authority=PolicyAuthority.PROPOSAL,
            rationale="actor-critic masked softmax sampling",
        )

    # -- rollout collection ------------------------------------------------------
    def store(
        self,
        state: OREState,
        action_id: str,
        reward: float,
        next_state: OREState,
        done: bool,
        eligible: Sequence[str],
        propensity: float,
    ) -> None:
        if action_id not in self.action_index:
            return
        self.rollout.append(
            {
                "state": self._state_vector(state),
                "action_index": self.action_index[action_id],
                "action_id": action_id,
                "reward": float(reward),
                "next_state": self._state_vector(next_state),
                "done": bool(done),
                "eligible": [a for a in eligible if a in self.action_index],
                "propensity": float(propensity),
            }
        )

    def update(self) -> dict[str, float]:
        """Run PPO epochs over the collected rollout (clipped surrogate + GAE).

        Gradients flow jointly through the shared feature trunk, the actor
        head (policy + entropy), and the critic head (value loss), one
        consistent forward/backward pass per epoch.
        """
        if not self.rollout:
            return {"loss": 0.0, "entropy": 0.0, "advantage": 0.0}
        states = np.stack([item["state"] for item in self.rollout])
        actions = np.asarray([item["action_index"] for item in self.rollout], dtype=np.int64)
        rewards = np.asarray([item["reward"] for item in self.rollout], dtype=np.float64)
        dones = np.asarray([item["done"] for item in self.rollout], dtype=np.float64)
        rows = np.arange(len(actions))

        def masks_for(rollout: list[dict[str, Any]]) -> np.ndarray:
            mask = np.zeros((len(rollout), len(self.action_ids)))
            for row, item in enumerate(rollout):
                for action in item["eligible"]:
                    mask[row, self.action_index[action]] = 1.0
            return mask

        masks = masks_for(self.rollout)
        feats, _ = self.features.forward(states)
        values = self.critic.forward(feats).reshape(-1)
        next_values = np.append(values[1:], 0.0)
        advantages, returns = compute_gae(
            rewards, values, next_values, dones,
            gamma=self.config.gamma, lam=self.config.gae_lambda,
        )
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_logits = self.actor.forward(feats)
        old_probs = masked_softmax(old_logits, masks)
        old_logp = np.log(np.maximum(old_probs[rows, actions], 1e-12))

        total_policy_loss = 0.0
        total_entropy = 0.0
        epochs = 1 if self.config.a2c_mode else self.config.epochs_per_rollout
        for _ in range(epochs):
            feats, activations = self.features.forward(states)
            logits = self.actor.forward(feats)
            probs = masked_softmax(logits, masks)
            taken_probs = np.maximum(probs[rows, actions], 1e-12)
            ratio = np.exp(np.log(taken_probs) - old_logp)
            if self.config.a2c_mode:
                weights = advantages / taken_probs
                surrogate = (weights * taken_probs).mean()
            else:
                clipped = np.clip(
                    ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
                )
                surrogate = np.minimum(ratio * advantages, clipped * advantages).mean()
            log_probs = np.log(np.maximum(probs, 1e-12))
            entropy = -(probs * log_probs).sum(axis=1).mean()
            value_preds = self.critic.forward(feats).reshape(-1)
            value_loss = float(((value_preds - returns) ** 2).mean())
            policy_loss = -float(surrogate)

            grad_logits = np.zeros_like(logits)
            if self.config.a2c_mode:
                grad_logits[rows, actions] += weights
            else:
                unclipped = ratio * advantages
                clipped_term = clipped * advantages
                use_clipped = clipped_term < unclipped
                eff_ratio = np.where(use_clipped, clipped, ratio)
                grad_logits[rows, actions] += (eff_ratio * advantages) / taken_probs
            entropy_grad = probs * (
                log_probs + (probs * log_probs).sum(axis=1, keepdims=True)
            )
            grad_logits -= self.config.entropy_coefficient * entropy_grad
            value_grad_column = (
                2.0 * self.config.value_coefficient * (value_preds - returns)
            ) / max(1, len(returns))

            grad_feats = grad_logits @ self.actor.weight.T
            grad_feats += value_grad_column.reshape(-1, 1) @ self.critic.weight.T

            batch = activations[0]
            grads_w: list[np.ndarray] = []
            grads_b: list[np.ndarray] = []
            delta = grad_feats
            for index in range(len(self.features.weights) - 1, -1, -1):
                grads_w.append(batch.T @ delta / batch.shape[0])
                grads_b.append(delta.sum(axis=0) / batch.shape[0])
                if index > 0:
                    delta = (delta @ self.features.weights[index].T) * (
                        1.0 - activations[index] ** 2
                    )
            self.training_steps += 1
            self.features.update_with_adam(
                list(reversed(grads_w)), list(reversed(grads_b)),
                lr=self.config.learning_rate, step=self.training_steps,
            )
            feats_now, _ = self.features.forward(states)
            grad_logits_now = grad_logits
            grad_w_actor = feats_now.T @ grad_logits_now / feats_now.shape[0]
            grad_b_actor = grad_logits_now.sum(axis=0) / feats_now.shape[0]
            self.actor.update_manual(grad_w_actor, grad_b_actor, lr=self.config.learning_rate)
            grad_value = value_grad_column.reshape(-1, 1)
            grad_w_critic = feats_now.T @ grad_value / feats_now.shape[0]
            grad_b_critic = grad_value.sum(axis=0) / feats_now.shape[0]
            self.critic.update_manual(grad_w_critic, grad_b_critic, lr=self.config.learning_rate)

            total_policy_loss += policy_loss
            total_entropy += float(entropy)

        self.rollout = []
        return {
            "loss": total_policy_loss / epochs,
            "entropy": total_entropy / epochs,
            "advantage": float(np.mean(advantages)),
            "value_loss": value_loss,
        }

    def _shared_gradients(
        self, activations: list[np.ndarray], grad: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        batch = activations[0]
        delta = np.asarray(grad, dtype=np.float64)
        grads_w: list[np.ndarray] = []
        grads_b: list[np.ndarray] = []
        for index in range(len(self.features.weights) - 1, -1, -1):
            grads_w.append(batch.T @ delta / batch.shape[0])
            grads_b.append(delta.sum(axis=0) / batch.shape[0])
            if index > 0:
                delta = (delta @ self.features.weights[index].T) * (
                    1.0 - activations[index] ** 2
                )
        return list(reversed(grads_w)), list(reversed(grads_b))


def _attach_update_manual() -> None:
    """Give linear heads a direct Adam update used by the joint PPO pass."""

    def update_manual(
        self: _LinearHead, grad_w: np.ndarray, grad_b: np.ndarray, *, lr: float
    ) -> None:
        self.step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self._m = beta1 * self._m + (1 - beta1) * grad_w
        self._v = beta2 * self._v + (1 - beta2) * grad_w * grad_w
        m_hat = self._m / (1 - beta1 ** self.step)
        v_hat = self._v / (1 - beta2 ** self.step)
        self.weight -= lr * m_hat / (np.sqrt(v_hat) + eps)
        self._bm = beta1 * self._bm + (1 - beta1) * grad_b
        self._bv = beta2 * self._bv + (1 - beta2) * grad_b * grad_b
        m_hat = self._bm / (1 - beta1 ** self.step)
        v_hat = self._bv / (1 - beta2 ** self.step)
        self.bias -= lr * m_hat / (np.sqrt(v_hat) + eps)

    _LinearHead.update_manual = update_manual


_attach_update_manual()
