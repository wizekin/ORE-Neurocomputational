"""Policy service interface, metadata, and the deterministic baseline policy.

Point 10, Step 1: RL is a replaceable, subordinate decision service behind the
narrow interface ``propose_actions(state, eligible_actions, budget)``. It
returns ranked proposals with confidence/uncertainty metadata and a
policy/model version; it never mutates workspace state or executes tools.

Step 5: every policy carries a registration record (identity, version,
algorithm family, schema versions, training dataset fingerprint, evaluation
results, resource requirements, uncertainty behavior, execution authority).

Step 6: ``DeterministicBaselinePolicy`` captures the coordinator-style
heuristic ordering and is the comparison target for all learned policies.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from ..contracts import (
    OREState,
    PolicyAuthority,
    PolicyDecision,
    RankedProposal,
)


class PolicyServiceError(RuntimeError):
    """Raised when a policy crashes or exceeds its decision time budget (8.1)."""


@dataclass(frozen=True, slots=True)
class PolicyMetadata:
    """Registration record required by the policy registry (Step 5)."""

    policy_id: str
    policy_version: str
    algorithm_family: str
    observation_schema_version: str = "1.0.0"
    action_schema_version: str = "1.0.0"
    training_dataset_fingerprint: str = ""
    evaluation_fingerprint: str = ""
    resource_requirements: Mapping[str, float] = field(default_factory=dict)
    uncertainty_behavior: str = "deterministic"
    execution_authority: PolicyAuthority = PolicyAuthority.PROPOSAL
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "algorithm_family": self.algorithm_family,
            "observation_schema_version": self.observation_schema_version,
            "action_schema_version": self.action_schema_version,
            "training_dataset_fingerprint": self.training_dataset_fingerprint,
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "resource_requirements": dict(self.resource_requirements),
            "uncertainty_behavior": self.uncertainty_behavior,
            "execution_authority": self.execution_authority.value,
            "description": self.description,
        }


class PolicyService(Protocol):
    """The narrow interface every policy must expose (Step 1)."""

    metadata: PolicyMetadata

    def propose_actions(
        self,
        state: OREState,
        eligible_actions: Sequence[str],
        budget: Mapping[str, float] | None = None,
    ) -> PolicyDecision:
        """Return ranked proposals over ``eligible_actions`` only."""
        ...


class DeterministicBaselinePolicy:
    """The coordinator-style heuristic baseline (Step 6).

    Mirrors the greedy information-gain-over-cost priority used by the ORE
    resource planner: prefer primitive module actions that make progress, run
    them at a fixed default depth, and stop as soon as a verified conclusion
    is available. This deterministic ordering is the target every learned
    policy is compared against.
    """

    _MODULE_ORDER = (
        "ore.rl.input_analyzer",
        "ore.rl.knowledge_retriever",
        "ore.rl.hypothesis_generator",
        "ore.rl.verifier",
        "ore.rl.causal_analyzer",
    )

    def __init__(self, *, default_depth: int = 3) -> None:
        self.default_depth = int(default_depth)
        self.metadata = PolicyMetadata(
            policy_id="deterministic_baseline",
            policy_version="1.1.0",
            algorithm_family="deterministic_heuristic",
            uncertainty_behavior="none",
            execution_authority=PolicyAuthority.PROPOSAL,
            description="coordinator-style greedy chain order with difficulty-scaled depth",
        )

    def target_depth(self, state: OREState) -> int:
        """Coordinator heuristic: harder tasks get deeper processing.

        Task difficulty drives the number of modifier tokens preceding the
        subject, so the baseline escalates retrieval depth accordingly:
        depth 3 / 4 / 5 for low / medium / high difficulty.
        """
        difficulty = state.features.get("task.difficulty", 0.5)
        if difficulty is None or difficulty != difficulty:
            difficulty = 0.5
        return max(1, min(5, 3 + int(float(difficulty) >= 0.34) + int(float(difficulty) >= 0.67)))

    def _module_rank(self, action_id: str) -> int:
        for index, module_id in enumerate(self._MODULE_ORDER):
            if module_id in action_id:
                return index
        return len(self._MODULE_ORDER)

    def _depth_of(self, action_id: str) -> int:
        if "@depth" in action_id:
            try:
                return int(action_id.split("@depth")[-1])
            except ValueError:
                return self.default_depth
        return self.default_depth

    def propose_actions(
        self,
        state: OREState,
        eligible_actions: Sequence[str],
        budget: Mapping[str, float] | None = None,
    ) -> PolicyDecision:
        eligible = list(eligible_actions)
        target = self.target_depth(state)

        def score(action_id: str) -> tuple[int, int, int]:
            if action_id == "meta.stop_success":
                return (-1, 0, 0)
            if action_id == "meta.stop_insufficient_evidence":
                return (5, 0, 0)
            if action_id.startswith("run."):
                return (
                    self._module_rank(action_id),
                    abs(self._depth_of(action_id) - target),
                    0,
                )
            return (6, 0, 0)

        ranked_ids = sorted(eligible, key=score)
        proposals = tuple(
            RankedProposal(action_id=action_id, score=float(-index), uncertainty=0.0)
            for index, action_id in enumerate(ranked_ids)
        )
        chosen = ranked_ids[0] if ranked_ids else ""
        return PolicyDecision(
            policy_id=self.metadata.policy_id,
            policy_version=self.metadata.policy_version,
            ranked=proposals,
            chosen_action_id=chosen,
            propensity=1.0,
            uncertainty=0.0,
            authority=PolicyAuthority.PROPOSAL,
            rationale="deterministic chain order with difficulty-scaled depth",
        )

    def act(
        self, state: OREState, eligible_actions: Sequence[str]
    ) -> tuple[str, dict[str, Any]]:
        """Convenience wrapper for direct environment interaction."""
        decision = self.propose_actions(state, eligible_actions)
        return decision.chosen_action_id, {}


class RandomEligiblePolicy:
    """Uniform random selection among eligible actions (evaluation floor, 8.3)."""

    def __init__(self, *, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.metadata = PolicyMetadata(
            policy_id="random_eligible",
            policy_version="1.0.0",
            algorithm_family="random_baseline",
            uncertainty_behavior="uniform",
            execution_authority=PolicyAuthority.SHADOW,
            description="uniform random eligible-action selection",
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
                ranked=(),
                chosen_action_id="",
                rationale="no eligible actions",
                fallback_used=True,
            )
        shuffled = list(eligible)
        self.rng.shuffle(shuffled)
        proposals = tuple(
            RankedProposal(action_id=action_id, score=0.0, uncertainty=1.0)
            for action_id in shuffled
        )
        return PolicyDecision(
            policy_id=self.metadata.policy_id,
            policy_version=self.metadata.policy_version,
            ranked=proposals,
            chosen_action_id=shuffled[0],
            propensity=1.0 / len(shuffled),
            uncertainty=1.0,
            authority=PolicyAuthority.SHADOW,
            rationale="uniform random selection",
        )


class ProgressBoundaryPolicy:
    """Serve a learned policy inside the coordinator's decision boundary (Step 2).

    The wrapped policy is consulted only for progress decisions (which
    module, at what depth, whether to batch); STOP_SUCCESS and
    STOP_INSUFFICIENT_EVIDENCE remain deterministic coordinator rules, and
    exploration is disabled at serving time. Training APIs (observe, store,
    update) are forwarded to the inner policy so the full MDP can still be
    learned.
    """

    _PROGRESS_PREFIXES = ("run.",)
    _PROGRESS_EXACT: frozenset[str] = frozenset()

    def __init__(self, inner: PolicyService) -> None:
        self.inner = inner
        self.metadata = type(inner.metadata)(
            policy_id=f"boundary::{inner.metadata.policy_id}",
            policy_version=inner.metadata.policy_version,
            algorithm_family=inner.metadata.algorithm_family,
            observation_schema_version=inner.metadata.observation_schema_version,
            action_schema_version=inner.metadata.action_schema_version,
            training_dataset_fingerprint=inner.metadata.training_dataset_fingerprint,
            evaluation_fingerprint=inner.metadata.evaluation_fingerprint,
            resource_requirements=dict(inner.metadata.resource_requirements),
            uncertainty_behavior=inner.metadata.uncertainty_behavior,
            execution_authority=inner.metadata.execution_authority,
            description=f"progress-boundary serving of {inner.metadata.policy_id}",
        )
        self.set_exploration(False)

    def set_exploration(self, enabled: bool) -> None:
        self.exploration_enabled = bool(enabled)
        setter = getattr(self.inner, "set_exploration", None)
        if setter is not None:
            setter(enabled)

    def _progress_actions(self, eligible: Sequence[str]) -> list[str]:
        return [
            action
            for action in eligible
            if action.startswith(self._PROGRESS_PREFIXES) or action in self._PROGRESS_EXACT
        ]

    def _deterministic_stop(self, eligible: Sequence[str]) -> str:
        if "meta.stop_success" in eligible:
            return "meta.stop_success"
        if "meta.stop_insufficient_evidence" in eligible:
            return "meta.stop_insufficient_evidence"
        if "meta.defer_resource_limit" in eligible:
            return "meta.defer_resource_limit"
        return eligible[0] if eligible else ""

    def propose_actions(
        self,
        state: OREState,
        eligible_actions: Sequence[str],
        budget: Mapping[str, float] | None = None,
    ) -> PolicyDecision:
        eligible = list(eligible_actions)
        verified = "meta.stop_success" in eligible
        progress = self._progress_actions(eligible)
        if verified or not progress:
            chosen = self._deterministic_stop(eligible)
            return PolicyDecision(
                policy_id=self.metadata.policy_id,
                policy_version=self.metadata.policy_version,
                chosen_action_id=chosen,
                propensity=1.0,
                rationale="deterministic coordinator stop rule",
            )
        decision = self.inner.propose_actions(state, progress, budget)
        return PolicyDecision(
            policy_id=self.metadata.policy_id,
            policy_version=self.metadata.policy_version,
            ranked=decision.ranked,
            chosen_action_id=decision.chosen_action_id,
            propensity=decision.propensity,
            uncertainty=decision.uncertainty,
            authority=decision.authority,
            fallback_used=decision.fallback_used,
            rationale=f"progress decision: {decision.rationale}",
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def timed_proposal(
    policy: PolicyService,
    state: OREState,
    eligible_actions: Sequence[str],
    *,
    max_ms: float = 50.0,
) -> tuple[PolicyDecision, float]:
    """Propose with a wall-clock guard; raise PolicyServiceError on overrun."""
    started = time.perf_counter()
    decision = policy.propose_actions(state, eligible_actions)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if elapsed_ms > max_ms:
        raise PolicyServiceError(
            f"policy {policy.metadata.policy_id} exceeded decision budget "
            f"({elapsed_ms:.1f} ms > {max_ms:.1f} ms)"
        )
    return decision, elapsed_ms
