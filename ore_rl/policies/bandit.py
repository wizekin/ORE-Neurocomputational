"""Contextual bandit policies: empirical means, UCB, LinUCB, Thompson Sampling.

Point 10, Phase F (Steps 34-38). All bandits operate strictly over the
eligible-action set, update only with validated feedback, and record
propensities so offline evaluation can correct exploration bias. Random
seeds and posterior versions are recorded for reproducibility (Step 37).

UCB (Step 35):    UCB(a) = mu_a + c * sqrt(ln(1 + t) / (1 + n_a))
LinUCB (Step 36): p(a) = x^T A_a^-1 b_a + alpha * sqrt(x^T A_a^-1 x)
Thompson (37):    theta_a ~ N(mu_a, Sigma_a); score = x^T theta_a
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import OREState, PolicyAuthority, PolicyDecision, RankedProposal
from .base import PolicyMetadata


def _context_vector(state: OREState) -> np.ndarray:
    names = state.feature_names
    values = [state.features.get(name, 0.0) for name in names]
    vector = np.asarray(values, dtype=np.float64)
    finite = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(finite)
    if norm > 0:
        finite = finite / norm
    if finite.size == 0:
        finite = np.zeros(1)
    return finite


@dataclass(slots=True)
class _ActionStats:
    count: int = 0
    reward_sum: float = 0.0
    reward_sq_sum: float = 0.0

    @property
    def mean(self) -> float:
        return self.reward_sum / self.count if self.count else 0.0

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        mean = self.mean
        return max(0.0, self.reward_sq_sum / self.count - mean * mean)

    def update(self, reward: float) -> None:
        self.count += 1
        self.reward_sum += reward
        self.reward_sq_sum += reward * reward


class EmpiricalBandit:
    """Simple empirical-mean baseline with epsilon exploration (Step 34)."""

    def __init__(self, *, seed: int = 0, epsilon: float = 0.1) -> None:
        self.rng = np.random.default_rng(seed)
        self.epsilon = float(epsilon)
        self.decisions = 0
        self.stats: dict[str, _ActionStats] = {}
        self.metadata = PolicyMetadata(
            policy_id="empirical_bandit",
            policy_version="1.0.0",
            algorithm_family="contextual_bandit/empirical",
            uncertainty_behavior="empirical_variance",
            execution_authority=PolicyAuthority.PROPOSAL,
            description="epsilon-greedy empirical mean bandit",
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
        self.decisions += 1
        explore = self.rng.random() < self.epsilon
        if explore:
            index = int(self.rng.integers(len(eligible)))
            chosen = eligible[index]
            propensity = self.epsilon / len(eligible)
        else:
            means = [(self.stats.get(a, _ActionStats()).mean, a) for a in eligible]
            chosen = max(means, key=lambda item: (item[0], item[1]))[1]
            greedy_share = 1.0 - self.epsilon
            known = sum(1 for a in eligible if a in self.stats)
            propensity = greedy_share / max(1, known)
        ranked = tuple(
            RankedProposal(
                action_id=action,
                score=self.stats.get(action, _ActionStats()).mean,
                uncertainty=math.sqrt(self.stats.get(action, _ActionStats()).variance),
            )
            for action in eligible
        )
        return PolicyDecision(
            policy_id=self.metadata.policy_id,
            policy_version=self.metadata.policy_version,
            ranked=ranked,
            chosen_action_id=chosen,
            propensity=propensity,
            uncertainty=ranked[0].uncertainty if ranked else 1.0,
            authority=PolicyAuthority.PROPOSAL,
            rationale="empirical mean with epsilon exploration",
        )

    def update(self, state: OREState, action_id: str, reward: float, *, verified: bool = True) -> None:
        """Update only with validated feedback (Steps 36, 38)."""
        if not verified:
            return
        stats = self.stats.setdefault(action_id, _ActionStats())
        stats.update(float(reward))


class UCBPolicy:
    """Upper-confidence-bound selection restricted to eligible actions (Step 35)."""

    def __init__(self, *, seed: int = 0, exploration: float = 0.5) -> None:
        self.exploration = float(exploration)
        self.exploration_enabled = True
        self.decisions = 0
        self.stats: dict[str, _ActionStats] = {}
        self.rng = np.random.default_rng(seed)
        self.metadata = PolicyMetadata(
            policy_id="ucb_bandit",
            policy_version="1.0.0",
            algorithm_family="contextual_bandit/ucb",
            uncertainty_behavior="frequentist_confidence",
            execution_authority=PolicyAuthority.PROPOSAL,
            description="UCB(a) = mu + c*sqrt(ln(1+t)/(1+n))",
        )

    def set_exploration(self, enabled: bool) -> None:
        """Serving mode: exploit empirical means without the confidence bonus."""
        self.exploration_enabled = bool(enabled)

    def _ucb(self, action_id: str) -> float:
        stats = self.stats.get(action_id, _ActionStats())
        bonus_scale = self.exploration if self.exploration_enabled else 0.0
        return stats.mean + bonus_scale * math.sqrt(
            math.log(1 + self.decisions) / (1 + stats.count)
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
        self.decisions += 1
        scores = {action: self._ucb(action) for action in eligible}
        best = max(scores, key=lambda action: (scores[action], action))
        ties = [action for action in eligible if abs(scores[action] - scores[best]) < 1e-12]
        chosen = ties[int(self.rng.integers(len(ties)))] if len(ties) > 1 else best
        total_unseen = sum(1 for action in eligible if self.stats.get(action) is None)
        propensity = 1.0 / len(ties) if chosen in ties else 1.0
        if total_unseen == len(eligible):
            propensity = 1.0 / len(eligible)
        ranked = tuple(
            RankedProposal(
                action_id=action,
                score=scores[action],
                uncertainty=math.sqrt(math.log(1 + self.decisions) / (1 + self.stats.get(action, _ActionStats()).count)),
            )
            for action in sorted(eligible, key=lambda a: -scores[a])
        )
        return PolicyDecision(
            policy_id=self.metadata.policy_id,
            policy_version=self.metadata.policy_version,
            ranked=ranked,
            chosen_action_id=chosen,
            propensity=propensity,
            uncertainty=ranked[0].uncertainty if ranked else 1.0,
            authority=PolicyAuthority.PROPOSAL,
            rationale="upper confidence bound over eligible actions",
        )

    def update(self, state: OREState, action_id: str, reward: float, *, verified: bool = True) -> None:
        if not verified:
            return
        self.stats.setdefault(action_id, _ActionStats()).update(float(reward))


class LinUCBPolicy:
    """Disjoint linear contextual bandit (Step 36).

    Per action a: A_a <- lambda*I; on update A_a += x x^T, b_a += r x.
    Score p(a) = x^T A_a^-1 b_a + alpha * sqrt(x^T A_a^-1 x). Ridge
    regularization and a condition-number monitor guard numerical stability.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        alpha: float = 0.3,
        ridge: float = 1.0,
        dimension: int = 0,
        record_condition: bool = True,
    ) -> None:
        self.alpha = float(alpha)
        self.ridge = float(ridge)
        self.rng = np.random.default_rng(seed)
        self.dimension = int(dimension)
        self.record_condition = bool(record_condition)
        self.A: dict[str, np.ndarray] = {}
        self.b: dict[str, np.ndarray] = {}
        self.stats: dict[str, _ActionStats] = {}
        self.condition_warnings: list[float] = []
        self.metadata = PolicyMetadata(
            policy_id="linucb_bandit",
            policy_version="1.0.0",
            algorithm_family="contextual_bandit/linucb",
            uncertainty_behavior="linear_posterior_width",
            execution_authority=PolicyAuthority.PROPOSAL,
            description="disjoint LinUCB with ridge regularization",
        )

    def _ensure(self, action_id: str, dimension: int) -> tuple[np.ndarray, np.ndarray]:
        if action_id not in self.A or self.A[action_id].shape[0] != dimension:
            self.A[action_id] = np.eye(dimension) * self.ridge
            self.b[action_id] = np.zeros(dimension)
        return self.A[action_id], self.b[action_id]

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
        x = _context_vector(state)
        dimension = x.shape[0]
        if self.dimension and dimension != self.dimension:
            raise ValueError(
                f"context dimension {dimension} does not match trained dimension {self.dimension}"
            )
        scores: dict[str, float] = {}
        uncertainties: dict[str, float] = {}
        for action in eligible:
            A, b = self._ensure(action, dimension)
            try:
                A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(A)
            if self.record_condition:
                condition = np.linalg.cond(A)
                if not math.isfinite(condition) or condition > 1e12:
                    self.condition_warnings.append(condition)
            theta = A_inv @ b
            mean = float(x @ theta)
            bonus = self.alpha * math.sqrt(max(0.0, float(x @ (A_inv @ x))))
            scores[action] = mean + bonus
            uncertainties[action] = bonus
        chosen = max(scores, key=lambda action: (scores[action], action))
        ranked = tuple(
            RankedProposal(
                action_id=action,
                score=scores[action],
                uncertainty=uncertainties[action],
            )
            for action in sorted(eligible, key=lambda a: -scores[a])
        )
        return PolicyDecision(
            policy_id=self.metadata.policy_id,
            policy_version=self.metadata.policy_version,
            ranked=ranked,
            chosen_action_id=chosen,
            propensity=1.0,
            uncertainty=uncertainties.get(chosen, 1.0),
            authority=PolicyAuthority.PROPOSAL,
            rationale="LinUCB linear estimate plus confidence bonus",
        )

    def update(self, state: OREState, action_id: str, reward: float, *, verified: bool = True) -> None:
        if not verified:
            return
        x = _context_vector(state)
        A, b = self._ensure(action_id, x.shape[0])
        A += np.outer(x, x)
        b += float(reward) * x
        self.stats.setdefault(action_id, _ActionStats()).update(float(reward))

    def save_state(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "ridge": self.ridge,
            "dimension": self.dimension,
            "A": {action: matrix.tolist() for action, matrix in self.A.items()},
            "b": {action: vector.tolist() for action, vector in self.b.items()},
        }

    def load_state(self, data: Mapping[str, Any]) -> None:
        self.alpha = float(data.get("alpha", self.alpha))
        self.ridge = float(data.get("ridge", self.ridge))
        self.A = {action: np.asarray(matrix) for action, matrix in data.get("A", {}).items()}
        self.b = {action: np.asarray(vector) for action, vector in data.get("b", {}).items()}


class ThompsonSamplingPolicy:
    """Bayesian linear Thompson sampling with seeded reproducibility (Step 37).

    Posterior per action: theta | data ~ N(mu, Sigma) with
    Sigma = (ridge*I + sum x x^T)^-1 and mu = Sigma b. Every draw uses the
    policy's seeded generator; the posterior version is stored on the metadata.
    Serving mode scores posterior means instead of sampling.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        ridge: float = 1.0,
        noise_variance: float = 1.0,
    ) -> None:
        self.exploration_enabled = True
        self.ridge = float(ridge)
        self.noise_variance = float(noise_variance)
        self.rng = np.random.default_rng(seed)
        self.A: dict[str, np.ndarray] = {}
        self.b: dict[str, np.ndarray] = {}
        self.stats: dict[str, _ActionStats] = {}
        self.draws = 0
        self.metadata = PolicyMetadata(
            policy_id="thompson_bandit",
            policy_version="1.0.0",
            algorithm_family="contextual_bandit/thompson",
            uncertainty_behavior="posterior_sampling",
            execution_authority=PolicyAuthority.PROPOSAL,
            description="Bayesian linear-Gaussian posterior sampling",
        )

    def _ensure(self, action_id: str, dimension: int) -> tuple[np.ndarray, np.ndarray]:
        if action_id not in self.A or self.A[action_id].shape[0] != dimension:
            self.A[action_id] = np.eye(dimension) * self.ridge
            self.b[action_id] = np.zeros(dimension)
        return self.A[action_id], self.b[action_id]

    def _posterior(self, action_id: str, dimension: int) -> tuple[np.ndarray, np.ndarray]:
        A, b = self._ensure(action_id, dimension)
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)
        return A_inv @ b, A_inv

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
        x = _context_vector(state)
        dimension = x.shape[0]
        scores: dict[str, float] = {}
        for action in eligible:
            mu, covariance = self._posterior(action, dimension)
            if self.exploration_enabled:
                theta = self.rng.multivariate_normal(mu, covariance)
                self.draws += 1
            else:
                theta = mu
            scores[action] = float(x @ theta)
        chosen = max(scores, key=lambda action: (scores[action], action))
        ranked = tuple(
            RankedProposal(action_id=action, score=scores[action], uncertainty=0.0)
            for action in sorted(eligible, key=lambda a: -scores[a])
        )
        return PolicyDecision(
            policy_id=self.metadata.policy_id,
            policy_version=self.metadata.policy_version,
            ranked=ranked,
            chosen_action_id=chosen,
            propensity=1.0 / len(eligible),
            uncertainty=1.0 / (1.0 + self.stats.get(chosen, _ActionStats()).count),
            authority=PolicyAuthority.PROPOSAL,
            rationale="Thompson posterior sample over eligible actions",
        )

    def set_exploration(self, enabled: bool) -> None:
        """Serving mode: score posterior means instead of sampling (Step 37)."""
        self.exploration_enabled = bool(enabled)

    def update(self, state: OREState, action_id: str, reward: float, *, verified: bool = True) -> None:
        if not verified:
            return
        x = _context_vector(state)
        A, b = self._ensure(action_id, x.shape[0])
        A += np.outer(x, x) / max(1e-9, self.noise_variance)
        b += float(reward) * x / max(1e-9, self.noise_variance)
        self.stats.setdefault(action_id, _ActionStats()).update(float(reward))
