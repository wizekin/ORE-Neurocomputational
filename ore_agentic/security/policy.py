"""Authorization policy engine and risk classification (Steps 3, 12, 30, 34).

Authorization comes from policy, not from the wording of a request.  The
engine is default-deny, separates read and write scopes, maps the risk score
to automatic / approval / prohibited decisions, and records every decision
into an audit log.  Authorization failures are policy failures, never
transient network failures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from ..contracts import (
    ExecutionDecision,
    RiskAssessment,
    SideEffectClass,
    utc_now,
)
from ..config import AgenticConfig, RiskWeights
from ..functional import risk_score

__all__ = (
    "Action",
    "AuthorizationDecision",
    "PolicyEngine",
    "PolicyRule",
    "Principal",
    "ScopeDeniedError",
    "classify_execution",
    "assess_risk",
)


class ScopeDeniedError(PermissionError):
    """Raised when a principal lacks a required scope (Step 34)."""


@dataclass(frozen=True, slots=True)
class Principal:
    """A requesting identity with roles and granted scopes (least privilege)."""

    principal_id: str
    roles: tuple[str, ...] = ()
    scopes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Validate identity."""
        if not self.principal_id or not self.principal_id.strip():
            raise ValueError("principal_id must be a non-empty string")

    def has_scope(self, scope: str) -> bool:
        """Return whether the principal holds exactly this scope."""
        return scope in self.scopes

    def to_dict(self) -> dict[str, Any]:
        """Serialize the principal (scopes only, never credentials)."""
        return {"principal_id": self.principal_id, "roles": list(self.roles),
                "scopes": sorted(self.scopes)}


@dataclass(frozen=True, slots=True)
class Action:
    """One action under authorization: resource plus effect class and scopes."""

    action_id: str
    resource: str
    effect_class: SideEffectClass = SideEffectClass.READ_ONLY
    required_scopes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the action."""
        return {"action_id": self.action_id, "resource": self.resource,
                "effect_class": self.effect_class.value,
                "required_scopes": list(self.required_scopes)}


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    """The outcome of one authorization check (never a silent default allow)."""

    allowed: bool
    reason: str
    scopes_granted: tuple[str, ...] = ()
    decided_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the decision for the audit log."""
        return {"allowed": self.allowed, "reason": self.reason,
                "scopes_granted": list(self.scopes_granted),
                "decided_at": self.decided_at}


@dataclass(frozen=True, slots=True)
class PolicyRule:
    """One explicit allow/deny rule with priority (higher wins)."""

    principal_id: str
    action_id: str
    effect: str  # "allow" | "deny"
    priority: int = 0
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the rule."""
        if self.effect not in ("allow", "deny"):
            raise ValueError("effect must be 'allow' or 'deny'")


class PolicyEngine:
    """Default-deny authorization with scope checks and audit (Steps 3, 34)."""

    def __init__(self, rules: Iterable[PolicyRule] = (), *,
                 default_deny: bool = True,
                 audit_sink: Callable[[Mapping[str, Any]], None] | None = None,
                 config: AgenticConfig | None = None) -> None:
        """Compose the engine from explicit rules; absence of rules denies."""
        self._rules = tuple(sorted(rules, key=lambda rule: -rule.priority))
        self._default_deny = default_deny
        self._audit_sink = audit_sink
        self._config = config or AgenticConfig()
        self.decisions: list[AuthorizationDecision] = []

    def authorize(self, principal: Principal, action: Action) -> AuthorizationDecision:
        """Authorize one action for one principal (default deny)."""
        missing = [scope for scope in action.required_scopes
                   if not principal.has_scope(scope)]
        matched = [rule for rule in self._rules
                   if rule.principal_id in (principal.principal_id, "*")
                   and rule.action_id in (action.action_id, "*")]
        if missing:
            decision = AuthorizationDecision(
                False, f"missing required scopes: {missing}")
        elif any(rule.effect == "deny" for rule in matched):
            deny = next(rule for rule in matched if rule.effect == "deny")
            decision = AuthorizationDecision(False, f"denied by policy rule: {deny.description or deny.principal_id}")
        elif any(rule.effect == "allow" for rule in matched):
            decision = AuthorizationDecision(
                True, "allowed by policy rule",
                scopes_granted=tuple(action.required_scopes))
        elif not self._default_deny:
            decision = AuthorizationDecision(True, "allowed by default-open policy",
                                             scopes_granted=tuple(action.required_scopes))
        else:
            decision = AuthorizationDecision(
                False, "no policy rule allows this action (default deny)")
        self.decisions.append(decision)
        self._emit_audit(principal, action, decision)
        return decision

    def authorize_scopes(self, principal: Principal, required: Iterable[str]) -> None:
        """Require every scope or raise ScopeDeniedError (used by tool calls)."""
        missing = [scope for scope in required if not principal.has_scope(scope)]
        if missing:
            raise ScopeDeniedError(
                f"{principal.principal_id} lacks scopes {missing}")

    def _emit_audit(self, principal: Principal, action: Action,
                    decision: AuthorizationDecision) -> None:
        """Record one authorization decision (Step 43 audit)."""
        if self._audit_sink is not None:
            self._audit_sink({
                "kind": "authorization",
                "principal": principal.to_dict(),
                "action": action.to_dict(),
                "decision": decision.to_dict(),
            })


def assess_risk(side_effect_class: SideEffectClass,
                sensitivity: float = 0.5,
                uncertainty: float = 0.3,
                external_impact: float | None = None,
                irreversibility: float | None = None,
                weights: RiskWeights | None = None,
                config: AgenticConfig | None = None) -> RiskAssessment:
    """Compute R = sum(w_i * r_i) and the execution decision (Step 3).

    Factor defaults derive from the side-effect class floor; callers can
    override any factor with measured values.  Thresholds come from the
    governed configuration, and high risk triggers review, not permission.
    """
    cfg = config or AgenticConfig()
    chosen = weights or cfg.risk_weights
    if irreversibility is None:
        irreversibility = side_effect_class.risk_floor
    if external_impact is None:
        external_impact = (0.0 if side_effect_class is SideEffectClass.READ_ONLY
                           else side_effect_class.risk_floor)
    factors = {
        "sensitivity": max(0.0, min(1.0, sensitivity)),
        "irreversibility": max(0.0, min(1.0, irreversibility)),
        "uncertainty": max(0.0, min(1.0, uncertainty)),
        "external_impact": max(0.0, min(1.0, external_impact)),
    }
    weight_map = {"sensitivity": chosen.sensitivity,
                  "irreversibility": chosen.irreversibility,
                  "uncertainty": chosen.uncertainty,
                  "external_impact": chosen.external_impact}
    score = risk_score(factors, weight_map)
    return RiskAssessment(
        score=score, factors=factors, weights=weight_map,
        side_effect_class=side_effect_class,
        decision=classify_execution(score, cfg),
        reason=(
            f"side-effect class {side_effect_class.value} with score "
            f"{score:.3f} against automatic<{cfg.automatic_risk_threshold:.2f} "
            f"and approval<{cfg.approval_risk_threshold:.2f}"))


def classify_execution(risk_score_value: float,
                       config: AgenticConfig | None = None) -> ExecutionDecision:
    """Map a risk score to automatic, approval-gated or prohibited (Step 3)."""
    cfg = config or AgenticConfig()
    if risk_score_value >= cfg.approval_risk_threshold:
        return ExecutionDecision.REQUIRES_APPROVAL
    return ExecutionDecision.AUTOMATIC
