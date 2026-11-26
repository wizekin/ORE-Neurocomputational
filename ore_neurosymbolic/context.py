"""Bounded reasoning context assembly (Step A7).

The bounded context carries exactly what one inference run needs: the bound
workspace snapshot reference, the selected facts and rules, the goal literal,
declared constraints and budgets. It reuses the existing shared-workspace
snapshot instead of duplicating state, and it enforces size bounds so a large
USG import cannot expand uncontrolled into the reasoning context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import (
    InferenceMode,
    Literal,
    RuleDefinition,
    SemanticFact,
    sha256_json,
)


@dataclass(slots=True)
class BoundedContext:
    """One immutable-view reasoning context (Step A7)."""

    snapshot_revision: int
    goal: Literal
    facts: tuple[SemanticFact, ...]
    rules: tuple[RuleDefinition, ...]
    constraints: tuple[Mapping[str, Any], ...] = ()
    mode: InferenceMode = InferenceMode.HYBRID
    budgets: Mapping[str, Any] = field(default_factory=dict)
    max_facts: int = 512
    max_rules: int = 256

    def __post_init__(self) -> None:
        """Enforce the declared size bounds (bounded context, Section 2)."""
        if len(self.facts) > self.max_facts:
            raise ValueError(
                f"context facts ({len(self.facts)}) exceed the bounded maximum "
                f"{self.max_facts}",
            )
        if len(self.rules) > self.max_rules:
            raise ValueError(
                f"context rules ({len(self.rules)}) exceed the bounded maximum "
                f"{self.max_rules}",
            )

    @property
    def fact_digest(self) -> str:
        """Return the content digest over the selected facts."""
        return sha256_json([fact.to_dict() for fact in self.facts])

    @property
    def ruleset_digest(self) -> str:
        """Return the content digest over the selected rules."""
        return sha256_json([rule.to_dict() for rule in self.rules])

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record without duplicating full payloads."""
        return {
            "snapshot_revision": self.snapshot_revision,
            "goal": self.goal.to_dict(),
            "fact_count": len(self.facts),
            "rule_count": len(self.rules),
            "fact_digest": self.fact_digest,
            "ruleset_digest": self.ruleset_digest,
            "constraints": [dict(item) for item in self.constraints],
            "mode": self.mode.value,
            "budgets": dict(self.budgets),
        }


def build_bounded_context(
    *,
    snapshot_revision: int,
    goal: Literal,
    facts: tuple[SemanticFact, ...],
    rules: tuple[RuleDefinition, ...],
    mode: InferenceMode,
    budgets: Mapping[str, Any],
    constraints: tuple[Mapping[str, Any], ...] = (),
    max_facts: int = 512,
    max_rules: int = 256,
) -> BoundedContext:
    """Assemble the bounded context, truncating deterministically at the bounds (A7)."""
    if len(facts) > max_facts:
        facts = tuple(sorted(facts, key=lambda item: (item.fact_id,)))[:max_facts]
    if len(rules) > max_rules:
        rules = tuple(sorted(rules, key=lambda item: (item.rule_id, item.version)))[:max_rules]
    return BoundedContext(
        snapshot_revision=snapshot_revision,
        goal=goal,
        facts=facts,
        rules=rules,
        constraints=constraints,
        mode=mode,
        budgets=dict(budgets),
        max_facts=max_facts,
        max_rules=max_rules,
    )


__all__ = ["BoundedContext", "build_bounded_context"]
