"""DQN, Double DQN, replay buffers, and prioritized replay in pure NumPy.

Point 10, Steps 39-41:
    * Q-network over a finite discrete action space with eligibility masks.
    * Uniform and proportional prioritized replay ((|delta|+eps)^alpha with
      importance-sampling weights).
    * Target network and Huber loss:
        y = r + gamma * (1 - d) * max_{a' in A_valid(s')} Q_theta-(s', a')
    * Double DQN separates selection and evaluation:
        y = r + gamma * (1 - d) * Q_theta-(s', argmax_{a'} Q_theta(s', a'))
    * The next-state action mask is applied inside the target computation.

The network is a small tanh MLP trained with Adam; no torch dependency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import OREState, PolicyAuthority, PolicyDecision, RankedProposal
from .base import PolicyMetadata


@dataclass(frozen=True, slots=True)
class DQNConfig:
    """Hyperparameters, recorded with checkpoints for reproducibility (8.2)."""

    hidden_units: tuple[int, ...] = (64, 48)
    learning_rate: float = 2.5e-3
    gamma: float = 0.95
    target_sync_steps: int = 100
    batch_size: int = 32
    replay_capacity: int = 4096
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 1200
    huber_delta: float = 1.0
    double_dqn: bool = True
    prioritized: bool = True
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_end: float = 1.0
    priority_epsilon: float = 1e-6
    seed: int = 0


class ReplayBuffer:
    """Uniform experience replay with terminal flags and eligibility masks."""

    def __init__(self, capacity: int = 4096, *, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self._storage: list[dict[str, Any]] = []
        self._position = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(
        self,
        *,
        state: np.ndarray,
        action_index: int,
        action_id: str,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        eligible_next: Sequence[str],
    ) -> None:
        entry = {
            "state": np.asarray(state, dtype=np.float64),
            "action_index": int(action_index),
            "action_id": action_id,
            "reward": float(reward),
            "next_state": np.asarray(next_state, dtype=np.float64),
            "done": bool(done),
            "eligible_next": list(eligible_next),
        }
        if len(self._storage) < self.capacity:
            self._storage.append(entry)
        else:
            self._storage[self._position] = entry
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[list[dict[str, Any]], np.ndarray | None]:
        if not self._storage:
            return [], None
        indices = self.rng.integers(0, len(self._storage), size=min(batch_size, len(self._storage)))
        return [self._storage[int(i)] for i in indices], None

    def update_priorities(self, indices: Sequence[int], td_errors: Sequence[float]) -> None:
        return


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritization |delta|^alpha with importance weights (Step 41)."""

    def __init__(
        self,
        capacity: int = 4096,
        *,
        seed: int = 0,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        epsilon: float = 1e-6,
        anneal_steps: int = 2000,
    ) -> None:
        super().__init__(capacity, seed=seed)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.epsilon = float(epsilon)
        self.anneal_steps = max(1, int(anneal_steps))
        self.priorities: list[float] = []
        self._samples_drawn = 0

    def add(self, **kwargs: Any) -> None:
        super().add(**kwargs)
        priority = max(self.priorities) if self.priorities else 1.0
        if len(self.priorities) < self.capacity:
            self.priorities.append(priority)
        else:
            self.priorities[self._position - 1] = priority

    def sample(self, batch_size: int) -> tuple[list[dict[str, Any]], np.ndarray | None]:
        if not self._storage:
            return [], None
        self._samples_drawn += 1
        beta = min(
            self.beta_end,
            self.beta_start
            + (self.beta_end - self.beta_start) * self._samples_drawn / self.anneal_steps,
        )
        priorities = np.asarray(self.priorities[: len(self._storage)], dtype=np.float64)
        probabilities = priorities ** self.alpha
        total = probabilities.sum()
        if total <= 0:
            probabilities = np.ones_like(probabilities)
            total = probabilities.sum()
        probabilities /= total
        indices = self.rng.choice(
            len(self._storage), size=min(batch_size, len(self._storage)), p=probabilities
        )
        weights = (len(self._storage) * probabilities[indices]) ** (-beta)
        weights = weights / max(weights.max(), 1e-12)
        return [self._storage[int(i)] for i in indices], weights

    def update_priorities(self, indices: Sequence[int], td_errors: Sequence[float]) -> None:
        for index, error in zip(indices, td_errors):
            if 0 <= index < len(self.priorities):
                self.priorities[index] = float(abs(error)) + self.epsilon


class _MLP:
    """Two-layer-cap tanh network with Adam and manual backpropagation."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_units: Sequence[int], *, seed: int = 0
    ) -> None:
        self.rng = np.random.default_rng(seed)
        dims = [input_dim, *hidden_units, output_dim]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            limit = math.sqrt(1.0 / max(1, fan_in))
            self.weights.append(self.rng.uniform(-limit, limit, size=(fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out))
        self._m = [np.zeros_like(w) for w in self.weights]
        self._v = [np.zeros_like(w) for w in self.weights]
        self._bm = [np.zeros_like(b) for b in self.biases]
        self._bv = [np.zeros_like(b) for b in self.biases]
        self.step = 0

    def forward(self, batch: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        activations = [np.atleast_2d(batch)]
        current = activations[0]
        last = len(self.weights) - 1
        for index, weight in enumerate(self.weights):
            affine = current @ weight + self.biases[index]
            if index < last:
                current = np.tanh(affine)
                activations.append(current)
            else:
                activations.append(affine)
        return activations[-1], activations

    def predict(self, vector: np.ndarray) -> np.ndarray:
        output, _ = self.forward(np.asarray(vector, dtype=np.float64))
        return output[0]

    def backward(
        self, activations: list[np.ndarray], grad_output: np.ndarray, *, lr: float
    ) -> None:
        batch_size = activations[0].shape[0]
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]
        delta = np.asarray(grad_output, dtype=np.float64)
        for index in range(len(self.weights) - 1, -1, -1):
            grads_w[index] = activations[index].T @ delta / batch_size
            grads_b[index] = delta.sum(axis=0) / batch_size
            if index > 0:
                delta = (delta @ self.weights[index].T) * (
                    1.0 - activations[index] ** 2
                )
        self.step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for index, (weight, grad) in enumerate(zip(self.weights, grads_w)):
            self._m[index] = beta1 * self._m[index] + (1 - beta1) * grad
            self._v[index] = beta2 * self._v[index] + (1 - beta2) * grad * grad
            m_hat = self._m[index] / (1 - beta1 ** self.step)
            v_hat = self._v[index] / (1 - beta2 ** self.step)
            self.weights[index] = weight - lr * m_hat / (np.sqrt(v_hat) + eps)
        for index, (bias, grad) in enumerate(zip(self.biases, grads_b)):
            self._bm[index] = beta1 * self._bm[index] + (1 - beta1) * grad
            self._bv[index] = beta2 * self._bv[index] + (1 - beta2) * grad * grad
            m_hat = self._bm[index] / (1 - beta1 ** self.step)
            v_hat = self._bv[index] / (1 - beta2 ** self.step)
            self.biases[index] = bias - lr * m_hat / (np.sqrt(v_hat) + eps)

    def copy_from(self, other: _MLP) -> None:
        self.weights = [matrix.copy() for matrix in other.weights]
        self.biases = [vector.copy() for vector in other.biases]

    def parameters(self) -> list[np.ndarray]:
        return [*self.weights, *self.biases]


class DQNPolicy:
    """Masked DQN / Double DQN over a fixed action catalogue."""

    def __init__(
        self,
        *,
        action_ids: Sequence[str],
        state_dim: int,
        config: DQNConfig | None = None,
    ) -> None:
        self.action_ids: tuple[str, ...] = tuple(action_ids)
        self.action_index: dict[str, int] = {a: i for i, a in enumerate(self.action_ids)}
        self.state_dim = int(state_dim)
        self.config = config or DQNConfig()
        self.online = _MLP(
            self.state_dim,
            len(self.action_ids),
            self.config.hidden_units,
            seed=self.config.seed,
        )
        self.target = _MLP(
            self.state_dim,
            len(self.action_ids),
            self.config.hidden_units,
            seed=self.config.seed + 1,
        )
        self.target.copy_from(self.online)
        buffer_class = (
            PrioritizedReplayBuffer if self.config.prioritized else ReplayBuffer
        )
        self.replay = buffer_class(
            self.config.replay_capacity,
            seed=self.config.seed,
            **(
                {
                    "alpha": self.config.priority_alpha,
                    "beta_start": self.config.priority_beta_start,
                    "beta_end": self.config.priority_beta_end,
                    "epsilon": self.config.priority_epsilon,
                }
                if self.config.prioritized
                else {}
            ),
        )
        self.training_steps = 0
        self.rng = np.random.default_rng(self.config.seed)
        self.metadata = PolicyMetadata(
            policy_id="dqn_policy",
            policy_version="1.0.0",
            algorithm_family="value_based/dqn" + ("/double" if self.config.double_dqn else ""),
            uncertainty_behavior="td_error_magnitude",
            execution_authority=PolicyAuthority.PROPOSAL,
            description="masked DQN with target network and Huber loss",
        )

    # -- inference ---------------------------------------------------------
    def q_values(self, state_vector: np.ndarray) -> np.ndarray:
        return self.online.predict(np.nan_to_num(state_vector, nan=0.0))

    def set_exploration(self, enabled: bool) -> None:
        """Disable exploration for deterministic serving (evaluation)."""
        self.exploration_enabled = bool(enabled)

    def epsilon(self) -> float:
        if not getattr(self, "exploration_enabled", True):
            return 0.0
        steps = self.training_steps
        fraction = min(1.0, steps / max(1, self.config.epsilon_decay_steps))
        return (
            self.config.epsilon_start
            + (self.config.epsilon_end - self.config.epsilon_start) * fraction
        )

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
            [state.features.get(name, 0.0) for name in state.feature_names],
            dtype=np.float64,
        )
        q = self.q_values(vector)
        eligible_indices = [
            self.action_index[a] for a in eligible if a in self.action_index
        ]
        if not eligible_indices:
            return PolicyDecision(
                policy_id=self.metadata.policy_id,
                policy_version=self.metadata.policy_version,
                ranked=tuple(RankedProposal(a, 0.0, 0.0) for a in eligible),
                chosen_action_id=eligible[0],
                propensity=1.0 / len(eligible),
                rationale="unseen actions; first eligible chosen",
            )
        masked = q[eligible_indices]
        greedy_index = int(np.argmax(masked))
        explore = self.rng.random() < self.epsilon()
        if explore:
            pick = int(self.rng.integers(len(eligible_indices)))
        else:
            pick = greedy_index
        chosen_index = eligible_indices[pick]
        chosen = next(a for a, i in self.action_index.items() if i == chosen_index)
        propensity = self.epsilon() / len(eligible_indices) + (
            (1 - self.epsilon()) if not explore else 0.0
        )
        scores = {a: float(q[i]) for a, i in self.action_index.items() if a in eligible}
        ranked = tuple(
            RankedProposal(action_id=a, score=scores[a], uncertainty=0.0)
            for a in sorted(eligible, key=lambda x: -scores.get(x, float("-inf")))
        )
        return PolicyDecision(
            policy_id=self.metadata.policy_id,
            policy_version=self.metadata.policy_version,
            ranked=ranked,
            chosen_action_id=chosen,
            propensity=propensity,
            uncertainty=float(abs(masked.max() - np.median(masked))),
            authority=PolicyAuthority.PROPOSAL,
            rationale="epsilon-greedy over masked Q values",
        )

    # -- training ------------------------------------------------------------
    def observe(
        self,
        *,
        state: OREState,
        action_id: str,
        reward: float,
        next_state: OREState | None,
        done: bool,
        eligible_next: Sequence[str],
        state_vectors: Mapping[str, np.ndarray] | None = None,
    ) -> float:
        """Store one transition and run a training step (target masks applied)."""
        if action_id not in self.action_index:
            return 0.0
        vector = (
            state_vectors[state.state_id]
            if state_vectors and state.state_id in state_vectors
            else np.asarray(
                [state.features.get(name, 0.0) for name in state.feature_names],
                dtype=np.float64,
            )
        )
        if next_state is not None:
            next_vector = (
                state_vectors[next_state.state_id]
                if state_vectors and next_state.state_id in state_vectors
                else np.asarray(
                    [next_state.features.get(name, 0.0) for name in next_state.feature_names],
                    dtype=np.float64,
                )
            )
        else:
            next_vector = np.zeros_like(vector)
        self.replay.add(
            state=np.nan_to_num(vector, nan=0.0),
            action_index=self.action_index[action_id],
            action_id=action_id,
            reward=float(reward),
            next_state=np.nan_to_num(next_vector, nan=0.0),
            done=bool(done),
            eligible_next=[a for a in eligible_next if a in self.action_index],
        )
        if len(self.replay) < self.config.batch_size:
            return 0.0
        return self._train_step()

    def _train_step(self) -> float:
        batch, weights = self.replay.sample(self.config.batch_size)
        if not batch:
            return 0.0
        states = np.stack([item["state"] for item in batch])
        next_states = np.stack([item["next_state"] for item in batch])
        actions = np.asarray([item["action_index"] for item in batch], dtype=np.int64)
        rewards = np.asarray([item["reward"] for item in batch], dtype=np.float64)
        dones = np.asarray([item["done"] for item in batch], dtype=np.float64)

        next_q_online = self.online.forward(next_states)[0]
        next_q_target = self.target.forward(next_states)[0]
        next_valid_mask = np.zeros_like(next_q_online)
        for row, item in enumerate(batch):
            eligible = item["eligible_next"]
            for action_id in eligible:
                next_valid_mask[row, self.action_index[action_id]] = 1.0
        masked_online = np.where(next_valid_mask > 0, next_q_online, -np.inf)
        if self.config.double_dqn:
            selected = np.argmax(masked_online, axis=1)
            evaluated = next_q_target[np.arange(len(batch)), selected]
        else:
            masked_target = np.where(next_valid_mask > 0, next_q_target, -np.inf)
            evaluated = np.max(masked_target, axis=1)
        evaluated = np.where(dones > 0, 0.0, evaluated)
        targets = rewards + self.config.gamma * evaluated

        predictions_all, activations = self.online.forward(states)
        predictions = predictions_all[np.arange(len(batch)), actions]
        td_errors = targets - predictions

        gradient = np.zeros_like(predictions_all)
        errors = td_errors
        delta = self.config.huber_delta
        abs_errors = np.abs(errors)
        huber_grad = np.where(abs_errors <= delta, errors, delta * np.sign(errors))
        # d Huber(target - prediction) / d prediction = -clip(target - prediction)
        gradient[np.arange(len(batch)), actions] = -huber_grad
        if weights is not None:
            gradient *= np.asarray(weights, dtype=np.float64)[:, None]
        self.online.backward(activations, gradient, lr=self.config.learning_rate)

        if isinstance(self.replay, PrioritizedReplayBuffer):
            self.replay.update_priorities(
                range(len(batch)), [float(abs(error)) for error in td_errors]
            )
        self.training_steps += 1
        if self.training_steps % self.config.target_sync_steps == 0:
            self.target.copy_from(self.online)
        return float(np.mean(np.abs(td_errors)))

    # -- persistence -----------------------------------------------------------
    def save_checkpoint(self) -> dict[str, Any]:
        return {
            "config": {
                "hidden_units": list(self.config.hidden_units),
                "learning_rate": self.config.learning_rate,
                "gamma": self.config.gamma,
                "double_dqn": self.config.double_dqn,
                "prioritized": self.config.prioritized,
            },
            "action_ids": list(self.action_ids),
            "state_dim": self.state_dim,
            "weights": [w.tolist() for w in self.online.weights],
            "biases": [b.tolist() for b in self.online.biases],
            "training_steps": self.training_steps,
        }

    def load_checkpoint(self, data: Mapping[str, Any]) -> None:
        self.online.weights = [np.asarray(w) for w in data["weights"]]
        self.online.biases = [np.asarray(b) for b in data["biases"]]
        self.target.copy_from(self.online)
        self.training_steps = int(data.get("training_steps", 0))
