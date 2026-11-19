"""Goal-sensitive working-memory priority scoring, independent of shared evidence storage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class RetentionWeights:
    """Configure the weighted contribution of goal relevance, recency, context and open tasks."""

    goal: float = 0.35
    recency: float = 0.25
    context: float = 0.25
    unresolved: float = 0.15

    def __post_init__(self) -> None:
        """Require non-negative weights with a positive total for meaningful normalization."""
        values = (self.goal, self.recency, self.context, self.unresolved)
        if any(value < 0 for value in values) or sum(values) <= 0:
            raise ValueError("retention weights must be non-negative with positive total")


class WorkingMemoryPolicy:
    """Compute retention priorities without deleting findings from the shared workspace."""

    def __init__(self, weights: RetentionWeights | None = None, capacity: int | None = None) -> None:
        """Set scoring coefficients and an optional bound on the number of active memory items."""
        if capacity is not None and capacity <= 0:
            raise ValueError("capacity must be positive")
        self.weights = weights or RetentionWeights()
        self.capacity = capacity

    def retention_score(self, factors: Mapping[str, float]) -> float:
        """Calculate P_i = alpha*G_i + beta*R_i + gamma*C_i + delta*U_i on normalized inputs."""
        names = {"goal": self.weights.goal, "recency": self.weights.recency,
                 "context": self.weights.context, "unresolved": self.weights.unresolved}
        for name in names:
            value = float(factors.get(name, 0.0))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} factor must be in [0, 1]")
        total_weight = sum(names.values())
        return sum(weight * float(factors.get(name, 0.0)) for name, weight in names.items()) / total_weight

    def prioritize(self, items: Mapping[str, Mapping[str, object]]) -> tuple[tuple[str, float], ...]:
        """Return active items ordered by descending retention score and stable item identifier."""
        scores = []
        for item_id, item in items.items():
            if not bool(item.get("is_active", True)):
                continue
            factors = item.get("retention_factors", {})
            if not isinstance(factors, Mapping):
                raise TypeError("retention_factors must be a mapping")
            scores.append((item_id, self.retention_score(factors)))
        return tuple(sorted(scores, key=lambda pair: (-pair[1], pair[0])))

    def select(self, items: Mapping[str, Mapping[str, object]]) -> tuple[str, ...]:
        """Select the highest-priority active items up to capacity without deleting other evidence."""
        ordered = self.prioritize(items)
        limit = self.capacity if self.capacity is not None else len(ordered)
        return tuple(item_id for item_id, _ in ordered[:limit])