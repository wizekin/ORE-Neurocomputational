"""Read-only proposal integration with the ORE coordinators (Steps 1, 2, 19, 51).

* ``RLPolicyService`` wraps any PolicyService as a subordinate decision
  service: ranked proposals only, never workspace mutation, with a crash and
  timeout guard that falls back to the deterministic coordinator (8.1).
* ``DecisionBoundaryPolicy`` defines exactly when the policy is consulted
  (Step 2): after task decomposition, after a module completes, when
  evidence is insufficient, when a dependency becomes ready, and at
  stop/continue decisions.
* ``ShadowModeController`` records proposed versus selected actions beside
  the coordinator without execution authority (Step 51).
* ``LimitedAuthorityController`` adopts proposals only for low-risk,
  feasible decisions and rolls back through the registry.
* ``ModulePreferenceBridge`` converts ranked proposals into the real
  ``resource_module_preferences`` task metadata consumed by
  ``CognitiveTaskCoordinator`` / ``AdaptiveResourceCoordinator`` — the
  production integration hook.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .contracts import (
    OREState,
    PolicyAuthority,
    PolicyDecision,
    RiskLevel,
)
from .policies.base import (
    DeterministicBaselinePolicy,
    PolicyMetadata,
    PolicyService,
    PolicyServiceError,
)
from .safety import AuthorityPolicy, DecisionValidation, FeasibleActionSet, HardConstraintGate


class DecisionBoundary(str, Enum):
    """The exact points where ORE asks the policy for a decision (Step 2)."""

    AFTER_TASK_DECOMPOSITION = "AFTER_TASK_DECOMPOSITION"
    AFTER_MODULE_COMPLETION = "AFTER_MODULE_COMPLETION"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    DEPENDENCY_READY = "DEPENDENCY_READY"
    STOP_CONTINUE = "STOP_CONTINUE"


@dataclass(frozen=True, slots=True)
class DecisionEvent:
    """A coordinator event that may trigger a policy consultation."""

    boundary: DecisionBoundary
    task_id: str
    module_id: str | None = None
    evidence_sufficient: bool = True
    dependency_ready: bool = False


class DecisionBoundaryPolicy:
    """Rules deciding when consultation happens (Step 2).

    The policy is never invoked at arbitrary internal points where the state
    is incomplete; only at the declared boundaries.
    """

    def __init__(self, *, enabled: Sequence[DecisionBoundary] | None = None) -> None:
        self.enabled: frozenset[DecisionBoundary] = frozenset(
            enabled or tuple(DecisionBoundary)
        )

    def should_consult(self, event: DecisionEvent) -> bool:
        if event.boundary not in self.enabled:
            return False
        if event.boundary is DecisionBoundary.EVIDENCE_INSUFFICIENT and event.evidence_sufficient:
            return False
        if event.boundary is DecisionBoundary.DEPENDENCY_READY and not event.dependency_ready:
            return False
        return True


class RLPolicyService:
    """The subordinate, replaceable decision service (Step 1).

    Guarantees:
      * The wrapped policy never receives workspace objects, only typed
        OREState observations — it cannot mutate workspace state.
      * A crash or timeout raises a single ``PolicyServiceError``-free
        fallback decision produced by the deterministic coordinator.
      * Every decision carries policy id/version and uncertainty metadata.
    """

    def __init__(
        self,
        inner: PolicyService,
        *,
        max_decision_ms: float = 100.0,
        authority: PolicyAuthority = PolicyAuthority.PROPOSAL,
    ) -> None:
        self.inner = inner
        self.max_decision_ms = float(max_decision_ms)
        self.authority = authority
        self.fallback_policy = DeterministicBaselinePolicy()
        self.fallback_count = 0
        self.proposal_count = 0

    @property
    def metadata(self) -> PolicyMetadata:
        return self.inner.metadata

    def propose_actions(
        self,
        state: OREState,
        eligible_actions: Sequence[str],
        budget: Mapping[str, float] | None = None,
    ) -> PolicyDecision:
        """Ranked proposals over eligible actions with a deterministic fallback."""
        started = time.perf_counter()
        try:
            decision = self.inner.propose_actions(state, eligible_actions, budget)
            elapsed = (time.perf_counter() - started) * 1000.0
            if elapsed > self.max_decision_ms:
                raise PolicyServiceError(
                    f"decision exceeded {self.max_decision_ms:.0f} ms budget"
                )
            self.proposal_count += 1
            return PolicyDecision(
                policy_id=decision.policy_id,
                policy_version=decision.policy_version,
                ranked=decision.ranked,
                chosen_action_id=decision.chosen_action_id,
                propensity=decision.propensity,
                uncertainty=decision.uncertainty,
                authority=self.authority,
                wall_time_ms=elapsed,
                fallback_used=False,
                rationale=decision.rationale,
            )
        except (PolicyServiceError, Exception):
            self.fallback_count += 1
            fallback = self.fallback_policy.propose_actions(state, eligible_actions, budget)
            return PolicyDecision(
                policy_id=f"{fallback.policy_id}::fallback",
                policy_version=fallback.policy_version,
                ranked=fallback.ranked,
                chosen_action_id=fallback.chosen_action_id,
                propensity=1.0,
                uncertainty=0.0,
                authority=PolicyAuthority.PROPOSAL,
                wall_time_ms=(time.perf_counter() - started) * 1000.0,
                fallback_used=True,
                rationale="policy crashed or timed out; deterministic coordinator fallback",
            )

    def stats(self) -> dict[str, int]:
        return {
            "proposals": self.proposal_count,
            "fallbacks": self.fallback_count,
        }


@dataclass(slots=True)
class ShadowDecisionRecord:
    """One shadow-mode observation: proposal versus coordinator choice (Step 51)."""

    boundary: DecisionBoundary
    task_id: str
    state_id: str
    proposed_action: str
    coordinator_action: str
    adopted: bool
    rationale: str = ""


class ShadowModeController:
    """Run the policy beside the coordinator without execution authority (Step 51)."""

    def __init__(
        self,
        service: RLPolicyService,
        *,
        boundary_policy: DecisionBoundaryPolicy | None = None,
    ) -> None:
        self.service = service
        self.boundary_policy = boundary_policy or DecisionBoundaryPolicy()
        self.records: list[ShadowDecisionRecord] = []
        self.authority = PolicyAuthority.SHADOW

    def consider(
        self,
        event: DecisionEvent,
        state: OREState,
        eligible_actions: Sequence[str],
        coordinator_action: str,
    ) -> ShadowDecisionRecord:
        """Consult the policy, record the comparison, never adopt (Step 51)."""
        if not self.boundary_policy.should_consult(event):
            record = ShadowDecisionRecord(
                boundary=event.boundary,
                task_id=event.task_id,
                state_id=state.state_id,
                proposed_action="",
                coordinator_action=coordinator_action,
                adopted=False,
                rationale="boundary not enabled",
            )
            self.records.append(record)
            return record
        decision = self.service.propose_actions(state, eligible_actions)
        record = ShadowDecisionRecord(
            boundary=event.boundary,
            task_id=event.task_id,
            state_id=state.state_id,
            proposed_action=decision.chosen_action_id,
            coordinator_action=coordinator_action,
            adopted=False,
            rationale=decision.rationale,
        )
        self.records.append(record)
        return record

    def agreement_rate(self) -> float:
        compared = [
            record
            for record in self.records
            if record.proposed_action and record.coordinator_action
        ]
        if not compared:
            return 0.0
        agreements = sum(
            1 for record in compared if record.proposed_action == record.coordinator_action
        )
        return agreements / len(compared)

    def summary(self) -> dict[str, Any]:
        return {
            "records": len(self.records),
            "agreement_rate": self.agreement_rate(),
            "service": self.service.stats(),
            "authority": self.authority.value,
        }


class LimitedAuthorityController:
    """Adopt policy proposals only under strict authority limits (Step 51).

    Adoption requires: the boundary is enabled, the action passes the hard
    constraint gate, the action risk is within the authority policy, and the
    policy's confidence clears the threshold. Every rejection falls back to
    the deterministic coordinator; rollback is delegated to the registry.
    """

    def __init__(
        self,
        service: RLPolicyService,
        gate: HardConstraintGate,
        *,
        boundary_policy: DecisionBoundaryPolicy | None = None,
        authority_policy: AuthorityPolicy | None = None,
        actions_by_id: Mapping[str, Any] | None = None,
    ) -> None:
        self.service = service
        self.gate = gate
        self.boundary_policy = boundary_policy or DecisionBoundaryPolicy()
        self.authority_policy = authority_policy or AuthorityPolicy(
            allow_limited_control=True,
            limited_control_max_risk=RiskLevel.LOW,
            limited_control_min_confidence=0.0,
        )
        self.actions_by_id = dict(actions_by_id or {})
        self.adopted_count = 0
        self.rejected_count = 0
        self.records: list[ShadowDecisionRecord] = []

    def decide(
        self,
        event: DecisionEvent,
        state: OREState,
        eligible_actions: Sequence[str],
        feasible_set: FeasibleActionSet,
        coordinator_action: str,
    ) -> tuple[str, PolicyDecision, DecisionValidation | None]:
        """Return (adopted_action_id, decision, validation)."""
        decision = self.service.propose_actions(state, eligible_actions)
        chosen = decision.chosen_action_id
        if not self.boundary_policy.should_consult(event):
            adopted = coordinator_action
        else:
            validation = self.gate.validate_decision(
                decision, feasible_set, actions_by_id=self.actions_by_id
            )
            action = self.actions_by_id.get(chosen)
            if (
                validation.valid
                and action is not None
                and self.authority_policy.permits_adoption(action, decision)
            ):
                adopted = chosen
                self.adopted_count += 1
            else:
                adopted = coordinator_action
                self.rejected_count += 1
                self.records.append(
                    ShadowDecisionRecord(
                        boundary=event.boundary,
                        task_id=event.task_id,
                        state_id=state.state_id,
                        proposed_action=chosen,
                        coordinator_action=coordinator_action,
                        adopted=False,
                        rationale="rejected: infeasible, risky, or low confidence",
                    )
                )
                return adopted, decision, validation
        self.records.append(
            ShadowDecisionRecord(
                boundary=event.boundary,
                task_id=event.task_id,
                state_id=state.state_id,
                proposed_action=chosen,
                coordinator_action=coordinator_action,
                adopted=adopted == chosen,
                rationale=decision.rationale,
            )
        )
        return adopted, decision, None

    def summary(self) -> dict[str, Any]:
        return {
            "adopted": self.adopted_count,
            "rejected": self.rejected_count,
            "service": self.service.stats(),
        }


class ModulePreferenceBridge:
    """Translate ranked proposals into coordinator module preferences.

    ``CognitiveTaskCoordinator`` and ``AdaptiveResourceCoordinator`` consume
    ``task.metadata["resource_module_preferences"]`` (a capability -> module
    mapping). This bridge converts a policy's ranked RUN_MODULE proposals
    into that mapping, so a learned policy can bias real coordinator
    planning without bypassing it (integration mapping, section 6).
    """

    def __init__(self, capability_of_action: Mapping[str, str]) -> None:
        self.capability_of_action = dict(capability_of_action)
        self.translations = 0

    def to_preferences(self, decision: PolicyDecision) -> dict[str, str]:
        """Map ranked proposals to {capability: module_id} preferences."""
        preferences: dict[str, str] = {}
        for proposal in decision.ranked:
            action_id = proposal.action_id
            capability = self.capability_of_action.get(action_id)
            if capability is None or not action_id.startswith("run."):
                continue
            module_id = action_id[len("run."):].split("@", 1)[0]
            if capability not in preferences:
                preferences[capability] = module_id
        self.translations += 1
        return preferences

    @staticmethod
    def from_environment(environment: Any) -> ModulePreferenceBridge:
        """Build the bridge from an OREEnvironment action catalogue."""
        capability_of_action = {
            entry.spec.action_id: entry.spec.capability
            for entry in environment.catalog_actions.list_actions()
            if entry.spec.capability
        }
        return ModulePreferenceBridge(capability_of_action)

    def attach_to_task(self, task: Any, decision: PolicyDecision) -> Any:
        """Attach preferences to a real CognitiveTask's metadata (Step 19).

        The task is executed by the real coordinator; the policy only biased
        provider selection through the documented metadata hook.
        """
        from dataclasses import replace

        preferences = self.to_preferences(decision)
        metadata = dict(task.metadata)
        metadata["resource_module_preferences"] = preferences
        return replace(task, metadata=metadata)


class StopDecisionBridge:
    """Propose continue/stop at the coordinator's stop boundary (Step 2).

    The adaptive resource coordinator stops when support thresholds are met;
    this bridge lets a learned policy propose stopping earlier (saving
    budget) or continuing (improving evidence) while the coordinator retains
    the final call.
    """

    def __init__(self, *, evidence_threshold: float = 0.55) -> None:
        self.evidence_threshold = float(evidence_threshold)
        self.proposals: list[dict[str, Any]] = []

    def propose_stop(
        self, state: OREState, decision: PolicyDecision
    ) -> dict[str, Any]:
        evidence = state.features.get("workspace.evidence_quality", 0.0)
        verified = state.features.get("goal.completed_count", 0.0) >= 4.0
        proposal = {
            "state_id": state.state_id,
            "policy_action": decision.chosen_action_id,
            "evidence_quality": evidence,
            "verified_pipeline_complete": verified,
            "recommendation": (
                "STOP"
                if decision.chosen_action_id.startswith("meta.stop")
                else "CONTINUE"
            ),
            "coordinator_rule": (
                "stop when support thresholds are met"
            ),
            "coordinator_retains_authority": True,
        }
        self.proposals.append(proposal)
        return proposal
