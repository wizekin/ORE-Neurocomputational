"""Constrained learning layer (ORE Point 23, Steps 50-51).

A constrained contextual bandit selects only from an approved safe action
set, logs context, action, propensity, outcome and constraints, and
supports offline replay. Full constrained RL and MPC remain research-path
interfaces: they are prerequisites-gated stubs, not trained policies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Mapping, Sequence

from ..config import AdaptationLimits


@dataclass(frozen=True)
class BanditLogRecord:
    """One bandit decision with full counterfactual logging (Step 50)."""

    timestamp: float
    action_id: str
    context: Mapping[str, float]
    propensity: float
    reward: float | None
    constraints_satisfied: bool
    shadow: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the record."""
        return {
            "timestamp": self.timestamp, "action_id": self.action_id,
            "context": dict(self.context), "propensity": self.propensity,
            "reward": self.reward, "constraints_satisfied": self.constraints_satisfied,
            "shadow": self.shadow,
        }


class ConstrainedContextualBandit:
    """Epsilon-greedy contextual bandit over an approved action set (Step 50).

    The bandit scores each safe action with a linear value estimate plus a
    count-based exploration bonus, then picks greedily or explores with
    probability ``epsilon``. It can never return an action outside the
    approved safe set: ``propose`` intersects the approved set with the
    currently-safe subset supplied by the caller.
    """

    def __init__(
        self,
        *,
        approved_actions: Sequence[str],
        feature_names: Sequence[str],
        epsilon: float = 0.10,
        learning_rate: float = 0.05,
        exploration_bonus: float = 0.5,
        seed: int = 7,
        shadow: bool = True,
    ) -> None:
        """Configure the action set, features and exploration policy."""
        if not approved_actions:
            raise ValueError("approved_actions must be non-empty")
        if len(set(approved_actions)) != len(approved_actions):
            raise ValueError("approved action ids must be unique")
        if not feature_names:
            raise ValueError("feature_names must be non-empty")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be within [0, 1]")
        if learning_rate <= 0.0 or exploration_bonus < 0.0:
            raise ValueError("learning_rate must be positive and bonus non-negative")
        self.approved_actions: tuple[str, ...] = tuple(approved_actions)
        self.feature_names: tuple[str, ...] = tuple(feature_names)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.exploration_bonus = exploration_bonus
        self._rng = random.Random(seed)
        self.shadow = shadow
        self._weights: dict[str, dict[str, float]] = {
            action: {name: 0.0 for name in self.feature_names}
            for action in self.approved_actions
        }
        self._counts: dict[str, int] = {action: 0 for action in self.approved_actions}
        self._log: list[BanditLogRecord] = []

    def _feature_vector(self, context: Mapping[str, float]) -> dict[str, float]:
        """Project a context mapping onto the declared feature names."""
        vector: dict[str, float] = {}
        for name in self.feature_names:
            value = context.get(name, 0.0)
            value = float(value)
            if not math.isfinite(value):
                value = 0.0
            vector[name] = value
        return vector

    def _score(self, action_id: str, features: Mapping[str, float]) -> float:
        """Return the value estimate plus exploration bonus for one action."""
        weights = self._weights[action_id]
        value = sum(weights[name] * features[name] for name in self.feature_names)
        bonus = self.exploration_bonus / math.sqrt(1.0 + self._counts[action_id])
        return value + bonus

    def safe_actions(self, currently_safe: Sequence[str]) -> tuple[str, ...]:
        """Intersect the approved set with the currently-safe subset."""
        allowed = set(currently_safe)
        return tuple(
            action for action in self.approved_actions if action in allowed
        )

    def propose(
        self, context: Mapping[str, float], *,
        currently_safe: Sequence[str] | None = None,
        timestamp: float = 0.0,
    ) -> tuple[str, float]:
        """Return (action_id, propensity) chosen only from the safe set.

        In shadow mode the proposal is recorded but nothing is executed;
        the caller remains responsible for applying baseline decisions.
        """
        safe = self.safe_actions(
            currently_safe if currently_safe is not None else self.approved_actions
        )
        if not safe:
            raise ValueError("no safe action available for the bandit to propose")
        features = self._feature_vector(context)
        scores = {action: self._score(action, features) for action in safe}
        ranked = sorted(safe, key=lambda action: (-scores[action], action))
        greedy = ranked[0]
        if self._rng.random() < self.epsilon and len(safe) > 1:
            chosen = self._rng.choice(sorted(safe))
            propensity = self.epsilon / len(safe)
        else:
            chosen = greedy
            propensity = 1.0 - self.epsilon + self.epsilon / len(safe)
        self._log.append(BanditLogRecord(
            timestamp=float(timestamp), action_id=chosen, context=dict(features),
            propensity=propensity, reward=None,
            constraints_satisfied=True, shadow=self.shadow,
        ))
        return chosen, propensity

    def update(
        self, action_id: str, reward: float, *,
        context: Mapping[str, float], constraints_satisfied: bool = True,
    ) -> None:
        """Record the outcome and update the value estimate for one action."""
        if action_id not in self._weights:
            raise ValueError(f"action {action_id!r} is not in the approved set")
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        if not constraints_satisfied:
            # Constraint-violating outcomes never reinforce the policy.
            self._log.append(BanditLogRecord(
                timestamp=0.0, action_id=action_id,
                context=dict(self._feature_vector(context)), propensity=0.0,
                reward=reward, constraints_satisfied=False, shadow=self.shadow,
            ))
            return
        features = self._feature_vector(context)
        weights = self._weights[action_id]
        estimate = sum(weights[name] * features[name] for name in self.feature_names)
        error = reward - estimate
        for name in self.feature_names:
            weights[name] += self.learning_rate * error * features[name]
        self._counts[action_id] += 1
        self._log.append(BanditLogRecord(
            timestamp=0.0, action_id=action_id, context=dict(features),
            propensity=1.0, reward=reward, constraints_satisfied=True,
            shadow=self.shadow,
        ))

    def log_records(self) -> tuple[BanditLogRecord, ...]:
        """Return the full decision log for offline evaluation (Step 50)."""
        return tuple(self._log)

    def counts(self) -> dict[str, int]:
        """Return the selection counts per approved action."""
        return dict(self._counts)


def replay_offline(
    records: Sequence[BanditLogRecord],
    *,
    target_action_id: str,
) -> dict[str, float]:
    """Inverse-propensity-weighted offline evaluation of one fixed policy.

    Evaluates the counterfactual "always choose ``target_action_id``"
    policy from logged bandit data. Records with zero propensity are
    skipped (they contribute no unbiased estimate).
    """
    if not records:
        return {"estimate": 0.0, "effective_samples": 0.0}
    weighted_reward = 0.0
    weight_total = 0.0
    effective = 0.0
    for record in records:
        if record.reward is None:
            continue
        if not record.constraints_satisfied:
            continue
        if record.action_id != target_action_id:
            continue
        if record.propensity <= 0.0:
            continue
        weight = 1.0 / record.propensity
        weighted_reward += weight * record.reward
        weight_total += weight
        effective += min(1.0, weight)
    estimate = weighted_reward / weight_total if weight_total > 0 else 0.0
    return {"estimate": estimate, "effective_samples": effective}


class ConstrainedRlResearchPath:
    """Prerequisites-gated research path toward RL and MPC (Step 51).

    Constrained RL or model-predictive control may only be evaluated after
    reliable telemetry, simulation, stable reward definitions and safety
    enforcement exist. This stub refuses to train until every prerequisite
    is explicitly attested, keeping the late-stage option honest.
    """

    PREREQUISITES: tuple[str, ...] = (
        "reliable_telemetry",
        "calibrated_simulation",
        "stable_reward_definition",
        "safety_enforcement",
        "shadow_evaluation_history",
    )

    def __init__(self, attestations: Mapping[str, bool] | None = None) -> None:
        """Record which prerequisites the deployment can attest."""
        self.attestations = dict(attestations or {})

    def missing_prerequisites(self) -> tuple[str, ...]:
        """Return the prerequisites that are not yet attested."""
        return tuple(
            name for name in self.PREREQUISITES if not self.attestations.get(name)
        )

    def train(self) -> None:
        """Refuse training while prerequisites are missing (Step 51)."""
        missing = self.missing_prerequisites()
        if missing:
            raise NotImplementedError(
                "constrained RL is a late-stage option; missing prerequisites: "
                f"{list(missing)}"
            )
        raise NotImplementedError(
            "prerequisites attested but no RL trainer is implemented in this version; "
            "extend this research path deliberately with safety constraints first"
        )


__all__ = [
    "ConstrainedContextualBandit", "BanditLogRecord", "replay_offline",
    "ConstrainedRlResearchPath",
]
