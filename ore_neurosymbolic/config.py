"""Bounded NSRL policy: budgets, weights, gates and ablation switches.

Every budget is a hard cap (Section 2). The default policy keeps the
deterministic core usable with no learned components and no GPU: learned
assistance, probabilistic combination and optional consistency signals are all
switchable and default to enabled-but-optional with graceful degradation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from .ambiguity_resolver import RankingWeights
from .resource_policy import ResourceBudgets
from .rule_validation import RuleProfile
from .workspace_publisher import PublicationGates


@dataclass(frozen=True, slots=True)
class NSRLAblationSwitches:
    """Integration mechanisms evaluation harnesses may disable one at a time."""

    learned_assistance: bool = True
    probabilistic_combination: bool = True
    independent_verification: bool = True
    constraint_filtering: bool = True
    ambiguity_preservation: bool = True
    defeasible_resolution: bool = True
    usg_retrieval: bool = True

    def disabled_names(self) -> tuple[str, ...]:
        """Return the names of the mechanisms currently switched off."""
        names = (
            "learned_assistance", "probabilistic_combination",
            "independent_verification", "constraint_filtering",
            "ambiguity_preservation", "defeasible_resolution", "usg_retrieval",
        )
        return tuple(name for name in names if not getattr(self, name))


@dataclass(frozen=True, slots=True)
class NSRLPolicy:
    """Complete policy for one neuro-symbolic reasoning run."""

    budgets: ResourceBudgets = field(default_factory=ResourceBudgets)
    rule_profile: RuleProfile = field(default_factory=RuleProfile)
    ranking_weights: RankingWeights = field(default_factory=RankingWeights)
    publication_gates: PublicationGates = field(default_factory=PublicationGates)
    ablations: NSRLAblationSwitches = field(default_factory=NSRLAblationSwitches)
    continuation_threshold: float = 1.0
    ambiguity_epsilon: float = 0.05
    max_queue_size: int = 4096
    max_candidates: int = 256
    min_evidence_reliability: float = 0.05
    verification_level_default: str = "basic"
    deterministic_seed: int = 0
    enable_logging: bool = False
    logger_database_path: str | None = None

    def __post_init__(self) -> None:
        """Validate thresholds, epsilon and seed bounds."""
        if not math.isfinite(self.continuation_threshold) or self.continuation_threshold < 0:
            raise ValueError("continuation_threshold must be finite and non-negative")
        if not math.isfinite(self.ambiguity_epsilon) or self.ambiguity_epsilon <= 0:
            raise ValueError("ambiguity_epsilon must be finite and positive")
        if self.max_queue_size <= 0 or self.max_candidates <= 0:
            raise ValueError("queue and candidate bounds must be positive")
        if not 0.0 <= self.min_evidence_reliability <= 1.0:
            raise ValueError("min_evidence_reliability must be in [0, 1]")
        if self.deterministic_seed < 0:
            raise ValueError("deterministic_seed cannot be negative")
        if self.verification_level_default not in {"none", "basic", "strict"}:
            raise ValueError("verification_level_default must be none, basic or strict")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible policy record."""
        return {
            "budgets": self.budgets.to_dict(),
            "rule_profile": self.rule_profile.to_dict(),
            "ranking_weights": self.ranking_weights.to_dict(),
            "publication_gates": self.publication_gates.to_dict(),
            "ablations": {
                name: getattr(self.ablations, name)
                for name in (
                    "learned_assistance", "probabilistic_combination",
                    "independent_verification", "constraint_filtering",
                    "ambiguity_preservation", "defeasible_resolution", "usg_retrieval",
                )
            },
            "continuation_threshold": self.continuation_threshold,
            "ambiguity_epsilon": self.ambiguity_epsilon,
            "max_queue_size": self.max_queue_size,
            "max_candidates": self.max_candidates,
            "min_evidence_reliability": self.min_evidence_reliability,
            "verification_level_default": self.verification_level_default,
            "deterministic_seed": self.deterministic_seed,
        }


def default_policy() -> NSRLPolicy:
    """Return the default NSRL policy instance."""
    return NSRLPolicy()


__all__ = ["NSRLAblationSwitches", "NSRLPolicy", "default_policy"]
