"""Hard constraints, feasible action sets, and policy authority boundaries.

Point 10, Step 4: mandatory rules are deterministic gates, not reward penalties.
Step 17: A_valid(s) = {a in A : m_a(s) = 1 and g_k(s, a) <= 0 for all k}.
The learned objective only ever operates within the feasible action set; if the
set is empty a deterministic recovery/escalation path is invoked and the policy
is never asked to invent an invalid action.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .action_catalog import ActionCatalog, CatalogAction
from .contracts import (
    ActionSpecRecord,
    ActionType,
    OREAction,
    OREState,
    PolicyDecision,
    RiskLevel,
)


class ConstraintKind(str, Enum):
    """Categories of hard constraints enforced outside the learned objective."""

    CAPABILITY_ELIGIBILITY = "CAPABILITY_ELIGIBILITY"
    DEPENDENCY = "DEPENDENCY"
    RESOURCE_BUDGET = "RESOURCE_BUDGET"
    AUTHORIZATION = "AUTHORIZATION"
    VERIFICATION = "VERIFICATION"
    USG_WRITE_PROTECTION = "USG_WRITE_PROTECTION"


@dataclass(frozen=True, slots=True)
class SafetyViolation:
    """One violated constraint with a deterministic, human-readable reason."""

    constraint: ConstraintKind
    action_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class ResourceBudgetState:
    """Remaining hard resource budget (CROS/coordinator is the authority, Step 12)."""

    remaining_latency_ms: float
    remaining_cpu_cost: float
    remaining_memory_mb: float
    remaining_module_calls: int
    reserved_latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class FeasibleActionSet:
    """Result of Step 17: the eligible actions and per-action rejection reasons."""

    state_id: str
    feasible: tuple[CatalogAction, ...]
    mask: Mapping[str, bool]
    rejections: Mapping[str, tuple[SafetyViolation, ...]]
    escalation_used: bool = False

    @property
    def feasible_ids(self) -> tuple[str, ...]:
        return tuple(item.action_id for item in self.feasible)

    def is_feasible(self, action_id: str) -> bool:
        return bool(self.mask.get(action_id, False))


@dataclass(frozen=True, slots=True)
class DecisionValidation:
    """Outcome of validate_decision (interface contract, section 5.1)."""

    valid: bool
    decision: PolicyDecision
    chosen_action: OREAction | None = None
    violations: tuple[SafetyViolation, ...] = ()

    @property
    def reason(self) -> str:
        if self.valid:
            return "accepted"
        return "; ".join(item.reason for item in self.violations) or "no feasible action"


def _already_executed_at_depth(action_id: str, executed: Iterable[str]) -> bool:
    """True when the same module variant already ran at an equal-or-greater depth.

    Parameterized action ids follow the convention ``run.<module>@depth<k>``.
    """
    if "@depth" not in action_id or not action_id.startswith("run."):
        return False
    module_part, depth_part = action_id.split("@depth", 1)
    try:
        depth = int(depth_part)
    except ValueError:
        return False
    for other in executed:
        if not other.startswith(module_part + "@depth"):
            continue
        try:
            other_depth = int(other.split("@depth", 1)[1])
        except ValueError:
            continue
        if other_depth >= depth:
            return True
    return False


_MODULE_ACTION_TYPES = frozenset(
    {
        ActionType.RUN_MODULE,
        ActionType.RETRIEVE_KNOWLEDGE,
        ActionType.GENERATE_HYPOTHESES,
        ActionType.CAUSAL_ANALYSIS,
        ActionType.VERIFY,
    }
)

_ESCALATION_ACTION_IDS = (
    "meta.defer_resource_limit",
    "meta.request_human_review",
    "meta.stop_insufficient_evidence",
)

_ESCALATION_TYPES = {
    "meta.defer_resource_limit": ActionType.DEFER_RESOURCE_LIMIT,
    "meta.request_human_review": ActionType.REQUEST_HUMAN_REVIEW,
    "meta.stop_insufficient_evidence": ActionType.STOP_INSUFFICIENT_EVIDENCE,
}


class HardConstraintGate:
    """Deterministic safety gate applied before and after any policy proposal.

    The gate implements, per action:
      * m_a(s): dependency mask over prerequisite capabilities (Step 11)
      * g_1: module must be present, enabled, and backend-bound (authorization)
      * g_2: estimated cost must fit the remaining hard resource budget
      * g_3: USG targets are read-only; mutating parameters are rejected
      * g_4: STOP_SUCCESS requires a verified conclusion (verification gate)
    """

    def __init__(
        self,
        catalog: ActionCatalog,
        *,
        authorized_modules: Sequence[str] | None = None,
        max_concurrent_batch: int = 3,
    ) -> None:
        self.catalog = catalog
        self.authorized_modules: frozenset[str] | None = (
            frozenset(authorized_modules) if authorized_modules is not None else None
        )
        self.max_concurrent_batch = max_concurrent_batch
        self.rejected_proposals = 0

    # -- feasibility -------------------------------------------------------
    def compute_feasible_actions(
        self,
        state: OREState,
        *,
        completed_capabilities: Iterable[str],
        budget: ResourceBudgetState,
        module_runnable: Mapping[str, bool] | None = None,
        verified_conclusion_available: bool = False,
        current_depth: int = 1,
        executed_action_ids: Sequence[str] | None = None,
    ) -> FeasibleActionSet:
        """Compute A_valid(s) with rejection reasons for every excluded action."""
        completed = set(completed_capabilities)
        runnable = module_runnable or {}
        mask: dict[str, bool] = {}
        rejections: dict[str, list[SafetyViolation]] = {}
        feasible: list[CatalogAction] = []

        for entry in self.catalog.list_actions():
            spec = entry.spec
            violations: list[SafetyViolation] = []

            unsatisfied = [cap for cap in spec.prerequisite_capabilities if cap not in completed]
            if unsatisfied:
                violations.append(
                    SafetyViolation(
                        ConstraintKind.DEPENDENCY,
                        spec.action_id,
                        f"unmet prerequisites: {unsatisfied}",
                    )
                )

            target = spec.target_module
            if target is not None:
                if self.authorized_modules is not None and target not in self.authorized_modules:
                    violations.append(
                        SafetyViolation(
                            ConstraintKind.AUTHORIZATION,
                            spec.action_id,
                            f"module {target} is not authorized for this policy",
                        )
                    )
                if target in runnable and not runnable[target]:
                    violations.append(
                        SafetyViolation(
                            ConstraintKind.CAPABILITY_ELIGIBILITY,
                            spec.action_id,
                            f"module {target} is not runnable (disabled, unbound, or unhealthy)",
                        )
                    )

            estimate = spec.default_resource_estimate
            if estimate.latency_ms > budget.remaining_latency_ms + 1e-9:
                violations.append(
                    SafetyViolation(
                        ConstraintKind.RESOURCE_BUDGET,
                        spec.action_id,
                        f"latency estimate {estimate.latency_ms:.1f} ms exceeds remaining "
                        f"{budget.remaining_latency_ms:.1f} ms",
                    )
                )
            if estimate.cpu_cost > budget.remaining_cpu_cost + 1e-9:
                violations.append(
                    SafetyViolation(
                        ConstraintKind.RESOURCE_BUDGET,
                        spec.action_id,
                        f"cpu estimate {estimate.cpu_cost:.2f} exceeds remaining "
                        f"{budget.remaining_cpu_cost:.2f}",
                    )
                )
            if estimate.memory_mb > budget.remaining_memory_mb + 1e-9:
                violations.append(
                    SafetyViolation(
                        ConstraintKind.RESOURCE_BUDGET,
                        spec.action_id,
                        f"memory estimate {estimate.memory_mb:.2f} MB exceeds remaining "
                        f"{budget.remaining_memory_mb:.2f} MB",
                    )
                )

            if executed_action_ids:
                if spec.action_id in executed_action_ids:
                    violations.append(
                        SafetyViolation(
                            ConstraintKind.CAPABILITY_ELIGIBILITY,
                            spec.action_id,
                            "this parameterized action was already executed in this episode; "
                            "deterministic re-execution adds no information",
                        )
                    )
                elif _already_executed_at_depth(spec.action_id, executed_action_ids):
                    violations.append(
                        SafetyViolation(
                            ConstraintKind.CAPABILITY_ELIGIBILITY,
                            spec.action_id,
                            "the same module was already executed at an equal or greater "
                            "depth; a shallower re-run adds no information",
                        )
                    )

            if self._violates_usg_protection(spec):
                violations.append(
                    SafetyViolation(
                        ConstraintKind.USG_WRITE_PROTECTION,
                        spec.action_id,
                        "actions may consult USG knowledge read-only; mutation is rejected",
                    )
                )

            if spec.action_type is ActionType.STOP_SUCCESS and not verified_conclusion_available:
                violations.append(
                    SafetyViolation(
                        ConstraintKind.VERIFICATION,
                        spec.action_id,
                        "STOP_SUCCESS requires an independently verified conclusion",
                    )
                )

            if spec.action_type is ActionType.ADJUST_DEPTH:
                delta = spec.depth_change
                if current_depth + delta < 1 or current_depth + delta > 5:
                    violations.append(
                        SafetyViolation(
                            ConstraintKind.RESOURCE_BUDGET,
                            spec.action_id,
                            f"depth change {delta:+d} would leave the 1..5 range",
                        )
                    )

            mask[spec.action_id] = not violations
            rejections[spec.action_id] = violations
            if not violations:
                feasible.append(entry)

        escalation_used = False
        if not feasible:
            escalation_used = True
            feasible = []
            for action_id in _ESCALATION_ACTION_IDS:
                entry = self.catalog.get(action_id)
                if entry is None:
                    entry = CatalogAction(
                        ActionSpecRecord(
                            action_id=action_id,
                            action_type=_ESCALATION_TYPES[action_id],
                            description="deterministic escalation fallback",
                        )
                    )
                    mask[action_id] = True
                    rejections[action_id] = []
                feasible.append(entry)
            for entry in feasible:
                mask[entry.action_id] = True
                rejections[entry.action_id] = []

        return FeasibleActionSet(
            state_id=state.state_id,
            feasible=tuple(feasible),
            mask=mask,
            rejections={
                action_id: tuple(items) for action_id, items in rejections.items()
            },
            escalation_used=escalation_used,
        )

    def _violates_usg_protection(self, spec: ActionSpecRecordLike) -> bool:
        """USG targets accept read-only consults only (acceptance 8.1).

        Any action targeting a USG module must be a RETRIEVE_KNOWLEDGE
        consult; other action types over USG targets and any parameter that
        names a mutation verb are rejected deterministically.
        """
        target = spec.target_module or ""
        if "usg" in target.lower() and spec.action_type is not ActionType.RETRIEVE_KNOWLEDGE:
            return True
        mutation_verbs = ("write", "mutate", "update", "delete", "overwrite", "replace")
        for parameter in getattr(spec, "parameters", ()):
            if parameter.name.lower() in mutation_verbs:
                return True
        return False

    # -- decision validation -------------------------------------------------
    def validate_decision(
        self,
        decision: PolicyDecision,
        feasible_set: FeasibleActionSet,
        *,
        actions_by_id: Mapping[str, OREAction] | None = None,
    ) -> DecisionValidation:
        """A policy proposal is only valid if its chosen action is feasible (8.1)."""
        chosen_id = decision.chosen_action_id
        if chosen_id not in feasible_set.mask:
            self.rejected_proposals += 1
            return DecisionValidation(
                valid=False,
                decision=decision,
                violations=(
                    SafetyViolation(
                        ConstraintKind.CAPABILITY_ELIGIBILITY,
                        chosen_id,
                        "proposed action is not part of the catalogue",
                    ),
                ),
            )
        if not feasible_set.is_feasible(chosen_id):
            self.rejected_proposals += 1
            violations = feasible_set.rejections.get(chosen_id, ())
            return DecisionValidation(
                valid=False,
                decision=decision,
                violations=tuple(
                    SafetyViolation(
                        item.constraint,
                        item.action_id,
                        f"ineligible action proposed: {item.reason}",
                    )
                    for item in violations
                ) or (
                    SafetyViolation(
                        ConstraintKind.CAPABILITY_ELIGIBILITY,
                        chosen_id,
                        "action is masked as ineligible",
                    ),
                ),
            )
        chosen = (actions_by_id or {}).get(chosen_id)
        return DecisionValidation(valid=True, decision=decision, chosen_action=chosen)


ActionSpecRecordLike = Any  # structural: objects with .target_module and .action_type


@dataclass(frozen=True, slots=True)
class AuthorityPolicy:
    """Limits of what a policy may do at runtime (Step 1, Step 51)."""

    allow_shadow: bool = True
    allow_proposal: bool = True
    allow_limited_control: bool = False
    limited_control_max_risk: RiskLevel = RiskLevel.LOW
    limited_control_min_confidence: float = 0.0

    def permits_adoption(self, action: OREAction, decision: PolicyDecision) -> bool:
        if not self.allow_limited_control:
            return False
        risk_order = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
        if risk_order[action.risk] > risk_order[self.limited_control_max_risk]:
            return False
        if decision.uncertainty > 0 and decision.propensity < self.limited_control_min_confidence:
            return False
        return True


@dataclass
class SafetyLedger:
    """Audit trail of rejected proposals and gate outcomes (Step 26, 8.4 metrics)."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, kind: str, **payload: Any) -> None:
        self.entries.append({"kind": kind, **payload})

    @property
    def invalid_action_rejections(self) -> int:
        return sum(1 for entry in self.entries if entry["kind"] == "rejected_proposal")

    def summary(self) -> dict[str, int]:
        kinds: dict[str, int] = {}
        for entry in self.entries:
            kinds[entry["kind"]] = kinds.get(entry["kind"], 0) + 1
        return kinds
