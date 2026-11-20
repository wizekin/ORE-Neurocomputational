"""Selective field subscriptions and weighted module relevance calculations."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .models import WORKSPACE_FIELDS


DEFAULT_SUBSCRIPTIONS: dict[str, tuple[str, ...]] = {
    "lexical_processor": ("current_input", "current_goal"),
    "syntax_processor": ("current_input", "lexical_hypotheses"),
    "semantic_processor": ("lexical_hypotheses", "syntactic_hypotheses", "retrieved_memories", "working_memory", "current_goal"),
    "working_memory": ("lexical_hypotheses", "syntactic_hypotheses", "semantic_hypotheses", "retrieved_memories", "current_goal"),
    "memory_retrieval": ("lexical_hypotheses", "semantic_hypotheses", "discourse_context", "working_memory", "current_goal"),
    "schema_formation": ("retrieved_memories", "working_memory", "current_goal"),
    "schema_guided_inference": ("retrieved_memories", "semantic_hypotheses", "working_memory", "current_goal"),
    "analogical_transfer": ("retrieved_memories", "working_memory", "current_goal"),
    "reconstructive_memory": ("retrieved_memories", "working_memory", "unresolved_questions", "current_goal"),
    "working_memory_overload": ("retrieved_memories", "working_memory", "unresolved_questions", "current_goal"),
    "cognitive_control": ("semantic_hypotheses", "retrieved_memories", "working_memory", "unresolved_questions", "module_status"),
    "discourse_processor": ("semantic_hypotheses", "discourse_context", "retrieved_memories", "working_memory", "current_input"),
    "goal_control": ("module_status", "unresolved_questions", "semantic_hypotheses", "retrieved_memories", "working_memory"),
}


@dataclass(frozen=True, slots=True)
class Subscription:
    """Declare the fields a module observes and its change-sensitivity threshold."""

    module_id: str
    fields: frozenset[str]
    sensitivity: Mapping[str, float]
    threshold: float = 0.0


class SubscriptionRegistry:
    """Maintain configurable subscriptions and evaluate matrix-based relevance scores."""

    def __init__(self, defaults: Mapping[str, Iterable[str]] | None = None) -> None:
        """Initialize known subscriptions using the shared workspace's default topology."""
        self._subscriptions: dict[str, Subscription] = {}
        for module_id, fields in (defaults if defaults is not None else DEFAULT_SUBSCRIPTIONS).items():
            self.register(module_id, fields)

    def register(
        self,
        module_id: str,
        fields: Iterable[str],
        *,
        sensitivity: Mapping[str, float] | None = None,
        threshold: float = 0.0,
    ) -> Subscription:
        """Register or replace a module subscription after validating fields and weights."""
        if not module_id.strip():
            raise ValueError("module_id must be non-empty")
        selected = frozenset(fields)
        unknown = selected.difference(WORKSPACE_FIELDS)
        if unknown:
            raise ValueError(f"unknown subscribed fields: {sorted(unknown)}")
        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        weights = dict(sensitivity or {})
        if any(field not in selected for field in weights):
            raise ValueError("sensitivity may only be supplied for subscribed fields")
        if any(value < 0 for value in weights.values()):
            raise ValueError("sensitivity weights cannot be negative")
        normalized = {field: float(weights.get(field, 1.0)) for field in selected}
        subscription = Subscription(module_id, selected, normalized, float(threshold))
        self._subscriptions[module_id] = subscription
        return subscription

    def unregister(self, module_id: str) -> bool:
        """Remove a subscription and report whether it existed."""
        return self._subscriptions.pop(module_id, None) is not None

    def get(self, module_id: str) -> Subscription | None:
        """Return one subscription declaration, if the module is registered."""
        return self._subscriptions.get(module_id)

    def subscribers_for(self, changed_fields: Iterable[str]) -> tuple[str, ...]:
        """Return only modules whose Boolean matrix row intersects the changed fields."""
        changed = set(changed_fields)
        return tuple(sorted(
            module_id for module_id, item in self._subscriptions.items()
            if item.fields.intersection(changed)
        ))

    def matrix(
        self,
        modules: Iterable[str] | None = None,
        fields: Iterable[str] = WORKSPACE_FIELDS,
    ) -> tuple[tuple[int, ...], ...]:
        """Return the binary adjacency matrix A_ij for the selected module and field order."""
        field_order = tuple(fields)
        module_order = tuple(modules) if modules is not None else tuple(sorted(self._subscriptions))
        return tuple(
            tuple(int(field in self._subscriptions.get(module, Subscription(module, frozenset(), {}, 0)).fields)
                  for field in field_order)
            for module in module_order
        )

    def relevance_score(self, module_id: str, deltas: Mapping[str, float]) -> float:
        """Compute R_m = sum_j a_mj * delta_j for subscribed, non-negative field changes."""
        subscription = self._subscriptions.get(module_id)
        if subscription is None:
            return 0.0
        score = 0.0
        for field, delta in deltas.items():
            if delta < 0:
                raise ValueError("change magnitudes cannot be negative")
            if field in subscription.fields:
                score += subscription.sensitivity.get(field, 1.0) * float(delta)
        return score

    def eligible(self, deltas: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
        """Return subscribed modules whose relevance scores meet their own thresholds."""
        results = []
        for module_id, subscription in self._subscriptions.items():
            score = self.relevance_score(module_id, deltas)
            if score >= subscription.threshold and any(
                field in subscription.fields and magnitude > 0 for field, magnitude in deltas.items()
            ):
                results.append((module_id, score))
        return tuple(sorted(results))

    def as_dict(self) -> dict[str, list[str]]:
        """Expose an ordered, JSON-friendly map of module IDs to subscribed fields."""
        return {key: sorted(value.fields) for key, value in sorted(self._subscriptions.items())}