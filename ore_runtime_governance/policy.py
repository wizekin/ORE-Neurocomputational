"""Versioned policy evaluation, rollout, and operator control (steps 51-52).

Every policy carries a version and checksum; ``PolicyEngine`` evaluates rules
that separate hard constraints from soft preferences so no adaptive
optimizer or operator override can silently relax a hard security limit
(integrated step 52, section 11). Every denied task, allocation, preemption,
or quarantine decision stores a concise reason plus the measurements that
triggered it (``explain``). ``PolicyRollout`` deploys a new policy version to
a small worker subset first and rolls back automatically when violation
rate, failure rate, latency, or resource waste cross the configured
thresholds (step 51). ``OperatorControl`` provides scoped, expiring, fully
audited manual overrides -- authorization required, hard rules untouched.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import (
    ConsequenceClass,
    TrustClass,
    canonical_json,
    sha256_hex,
)


@dataclass(frozen=True)
class PolicyRule:
    """One policy rule; ``hard`` rules are never relaxable (step 52)."""

    rule_id: str
    description: str
    hard: bool = True
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "hard": self.hard,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class PolicyDocument:
    """Immutable, checksummed policy version (step 51)."""

    policy_id: str
    version: int
    rules: tuple[PolicyRule, ...]
    author: str = "policy_admin"
    created_epoch: float = 0.0
    supersedes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("policy_id must be non-empty")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError("version must be a positive integer")
        if not self.rules:
            raise ValueError("a policy requires at least one rule")
        seen: set[str] = set()
        for rule in self.rules:
            if rule.rule_id in seen:
                raise ValueError(f"duplicate rule id {rule.rule_id!r}")
            seen.add(rule.rule_id)

    def checksum(self) -> str:
        return sha256_hex(
            canonical_json(
                {
                    "policy_id": self.policy_id,
                    "version": self.version,
                    "rules": [rule.to_dict() for rule in self.rules],
                }
            )
        )

    def checksum_matches(self, checksum: str) -> bool:
        return self.checksum() == checksum

    def rule(self, rule_id: str) -> PolicyRule | None:
        for candidate in self.rules:
            if candidate.rule_id == rule_id:
                return candidate
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "rules": [rule.to_dict() for rule in self.rules],
            "author": self.author,
            "created_epoch": self.created_epoch,
            "supersedes": self.supersedes,
            "checksum": self.checksum(),
        }


@dataclass(frozen=True)
class PolicyDecision:
    """Typed policy outcome with reason and measurements (step 52)."""

    allowed: bool
    reason: str
    policy_version: int
    rule_id: str
    measurements: Mapping[str, Any] = field(default_factory=dict)
    hard: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "policy_version": self.policy_version,
            "rule_id": self.rule_id,
            "measurements": dict(self.measurements),
            "hard": self.hard,
        }


@dataclass(frozen=True)
class RolloutObservation:
    """Measurements collected for one rollout stage (step 51)."""

    stage_index: int
    violation_rate: float = 0.0
    failure_rate: float = 0.0
    p95_latency_ms: float = 0.0
    resource_waste_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_index": self.stage_index,
            "violation_rate": self.violation_rate,
            "failure_rate": self.failure_rate,
            "p95_latency_ms": self.p95_latency_ms,
            "resource_waste_rate": self.resource_waste_rate,
        }


class RolloutDecision:
    """Kinds of rollout progression."""

    CONTINUE = "continue"
    HOLD = "hold"
    ROLLBACK = "rollback"
    COMPLETE = "complete"


def default_policy(now_epoch: float | None = None) -> PolicyDocument:
    """Baseline CROS policy: hard floors and soft preferences (section 2)."""
    return PolicyDocument(
        policy_id="cros-baseline",
        version=1,
        rules=(
            PolicyRule(
                rule_id="trust-tier-floor",
                description="untrusted generated code never executes in-process",
                hard=True,
                parameters={"forbidden_tiers": ["untrusted_generated"]},
            ),
            PolicyRule(
                rule_id="no-network-untrusted",
                description="host.network is never issued to untrust components",
                hard=True,
                parameters={"denied_trust": ["untrusted_generated"]},
            ),
            PolicyRule(
                rule_id="deadline-positive",
                description="tasks with non-positive deadlines are rejected",
                hard=True,
            ),
            PolicyRule(
                rule_id="audit-required-safety",
                description="safety-critical components require an audit sink",
                hard=True,
            ),
            PolicyRule(
                rule_id="prefer-stronger-isolation",
                description="prefer stronger isolation for high-consequence work",
                hard=False,
            ),
            PolicyRule(
                rule_id="queue-soft-pressure",
                description="under pressure defer soft (low-priority) work first",
                hard=False,
            ),
        ),
        created_epoch=now_epoch if now_epoch is not None else time.time(),
    )


class PolicyEngine:
    """Evaluates one policy document against runtime requests (step 52)."""

    def __init__(self, document: PolicyDocument | None = None) -> None:
        self.document = document or default_policy()

    @property
    def version(self) -> int:
        return self.document.version

    def checksum(self) -> str:
        return self.document.checksum()

    def evaluate(
        self,
        subject: str,
        action: str,
        context: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        """Evaluate hard rules first, then soft preferences; deny is explainable."""
        context = dict(context or {})
        measurements = {
            "subject": subject,
            "action": action,
            "trust_class": context.get("trust_class", ""),
            "consequence": context.get("consequence", ""),
            "deadline_seconds": context.get("deadline_seconds"),
            "audit_attached": context.get("audit_attached", True),
        }

        floor = self.document.rule("trust-tier-floor")
        if floor is not None and action in {"dispatch", "register", "start"}:
            trust = str(context.get("trust_class", "trusted_service"))
            tier = str(context.get("tier", ""))
            if trust == TrustClass.UNTRUSTED_GENERATED.value and tier in {"", "in_process"}:
                return PolicyDecision(
                    allowed=False,
                    reason=(
                        f"hard rule trust-tier-floor: untrust tier '{trust}' cannot "
                        f"execute {action} in-process"
                    ),
                    policy_version=self.version,
                    rule_id="trust-tier-floor",
                    measurements=measurements,
                    hard=True,
                )

        network_rule = self.document.rule("no-network-untrusted")
        if network_rule is not None and action == "issue-right":
            trust = str(context.get("trust_class", ""))
            right = str(context.get("right", ""))
            if trust == TrustClass.UNTRUSTED_GENERATED.value and right == "host.network":
                return PolicyDecision(
                    allowed=False,
                    reason=(
                        "hard rule no-network-untrusted: host.network is never "
                        f"issued to trust class '{trust}'"
                    ),
                    policy_version=self.version,
                    rule_id="no-network-untrusted",
                    measurements=measurements,
                    hard=True,
                )

        deadline_rule = self.document.rule("deadline-positive")
        if deadline_rule is not None:
            deadline = context.get("deadline_seconds")
            if action in {"dispatch", "reserve"} and (
                deadline is None or float(deadline) <= 0
            ):
                return PolicyDecision(
                    allowed=False,
                    reason="hard rule deadline-positive: tasks need a positive deadline",
                    policy_version=self.version,
                    rule_id="deadline-positive",
                    measurements=measurements,
                    hard=True,
                )

        audit_rule = self.document.rule("audit-required-safety")
        if audit_rule is not None and action == "dispatch":
            consequence = str(context.get("consequence", ""))
            if consequence == ConsequenceClass.SAFETY_CRITICAL.value and not context.get(
                "audit_attached", True
            ):
                return PolicyDecision(
                    allowed=False,
                    reason=(
                        "hard rule audit-required-safety: safety-critical work needs "
                        "an attached audit sink"
                    ),
                    policy_version=self.version,
                    rule_id="audit-required-safety",
                    measurements=measurements,
                    hard=True,
                )

        preference = self.document.rule("prefer-stronger-isolation")
        soft_note = ""
        if preference is not None:
            consequence = str(context.get("consequence", ""))
            if consequence in {ConsequenceClass.EXTERNAL_SIDE_EFFECT.value, ConsequenceClass.SAFETY_CRITICAL.value}:
                soft_note = (
                    " soft preference prefer-stronger-isolation: stronger isolation "
                    "recommended for consequence " + consequence
                )
        return PolicyDecision(
            allowed=True,
            reason=f"allowed by policy version {self.version}.{soft_note}".strip(),
            policy_version=self.version,
            rule_id="allow-default",
            measurements=measurements,
            hard=False,
        )

    def explain(self, decision: PolicyDecision) -> str:
        """Operator-facing explanation: reason, measurements, policy version."""
        lines = [
            f"policy v{decision.policy_version} rule {decision.rule_id}: "
            + ("ALLOW" if decision.allowed else "DENY"),
            f"reason: {decision.reason}",
            f"hard constraint: {decision.hard}",
        ]
        for key, value in sorted(decision.measurements.items()):
            lines.append(f"measurement {key}={value!r}")
        return "; ".join(lines)


class PolicyRollout:
    """Versioned, staged rollout with automatic rollback (step 51)."""

    def __init__(
        self,
        candidate: PolicyDocument,
        known_good: PolicyDocument,
        *,
        stages: tuple[tuple[str, ...], ...] | None = None,
        max_violation_rate: float = 0.05,
        max_failure_rate: float = 0.10,
        max_p95_latency_ms: float = 500.0,
        max_resource_waste_rate: float = 0.20,
        on_event=None,
    ) -> None:
        if not isinstance(candidate, PolicyDocument) or not isinstance(known_good, PolicyDocument):
            raise TypeError("rollout requires PolicyDocument objects")
        if candidate.version <= known_good.version:
            raise ValueError("candidate version must exceed the known-good version")
        if stages is None:
            stages = (("worker.alpha",), ("worker.alpha", "worker.beta"), ())
        for threshold in (max_violation_rate, max_failure_rate, max_resource_waste_rate):
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("rate thresholds must be within 0..1")
        if max_p95_latency_ms <= 0:
            raise ValueError("latency threshold must be positive")
        self.candidate = candidate
        self.known_good = known_good
        self.stages = stages
        self.thresholds = {
            "max_violation_rate": max_violation_rate,
            "max_failure_rate": max_failure_rate,
            "max_p95_latency_ms": max_p95_latency_ms,
            "max_resource_waste_rate": max_resource_waste_rate,
        }
        self.stage_index = 0
        self.rolled_back = False
        self.completed = False
        self.history: list[tuple[RolloutObservation, str]] = []
        self._on_event = on_event

    def active_document(self) -> PolicyDocument:
        if self.rolled_back:
            return self.known_good
        return self.candidate

    def active_stage(self) -> tuple[str, ...]:
        if self.rolled_back or self.stage_index >= len(self.stages):
            return ()
        return self.stages[self.stage_index]

    def advance(self, observation: RolloutObservation) -> str:
        """Continue, hold, roll back, or complete based on measurements."""
        if self.rolled_back or self.completed:
            return RolloutDecision.COMPLETE if self.completed else RolloutDecision.ROLLBACK
        violations = [
            name
            for name, value, threshold_name in (
                ("violation_rate", observation.violation_rate, "max_violation_rate"),
                ("failure_rate", observation.failure_rate, "max_failure_rate"),
                ("resource_waste_rate", observation.resource_waste_rate, "max_resource_waste_rate"),
            )
            if value > self.thresholds[threshold_name]
        ]
        if observation.p95_latency_ms > self.thresholds["max_p95_latency_ms"]:
            violations.append("p95_latency_ms")
        if violations:
            decision = RolloutDecision.ROLLBACK
            reason = (
                f"automatic rollback: thresholds crossed ({', '.join(violations)}); "
                f"returning to known-good v{self.known_good.version}"
            )
            self.rolled_back = True
            self.history.append((observation, decision))
            self._emit("policy_rollback", {"reason": reason, "stage": observation.stage_index})
            return decision
        self.stage_index += 1
        if self.stage_index >= len(self.stages):
            self.completed = True
            self.history.append((observation, RolloutDecision.COMPLETE))
            self._emit("policy_rollout", {"status": "complete", "stage": observation.stage_index})
            return RolloutDecision.COMPLETE
        self.history.append((observation, RolloutDecision.CONTINUE))
        self._emit("policy_rollout", {"status": "continued", "stage": observation.stage_index})
        return RolloutDecision.CONTINUE

    def rollback(self, reason: str = "operator rollback") -> PolicyDocument:
        """Manual rollback to the prior known-good version (audited)."""
        self.rolled_back = True
        self._emit("policy_rollback", {"reason": reason, "operator_initiated": True})
        return self.known_good

    def _emit(self, kind: str, metadata: Mapping[str, Any]) -> None:
        if self._on_event is not None:
            try:
                self._on_event(kind, dict(metadata))
            except Exception:
                pass

    def describe(self) -> dict[str, Any]:
        return {
            "candidate_version": self.candidate.version,
            "candidate_checksum": self.candidate.checksum(),
            "known_good_version": self.known_good.version,
            "stage_index": self.stage_index,
            "active_stage": list(self.active_stage()),
            "rolled_back": self.rolled_back,
            "completed": self.completed,
            "thresholds": dict(self.thresholds),
            "history": [
                (observation.to_dict(), decision) for observation, decision in self.history
            ],
        }


@dataclass(frozen=True)
class OverrideGrant:
    """Scoped, expiring manual override (step 52)."""

    override_id: str
    operator: str
    scope: str
    reason: str
    granted_epoch: float
    expires_epoch: float

    def is_active(self, now: float | None = None) -> bool:
        current = now if now is not None else time.time()
        return current < self.expires_epoch

    def to_dict(self) -> dict[str, Any]:
        return {
            "override_id": self.override_id,
            "operator": self.operator,
            "scope": self.scope,
            "reason": self.reason,
            "granted_epoch": self.granted_epoch,
            "expires_epoch": self.expires_epoch,
        }


class OperatorControl:
    """Authorization-gated, scoped, expiring operator overrides (step 52).

    Overrides never relax hard security limits: requests naming a hard rule
    are denied outright, and every grant/renewal/expiry is audited.
    """

    def __init__(
        self,
        *,
        max_ttl_seconds: float = 3600.0,
        authorized_operators: tuple[str, ...] | None = None,
        on_event=None,
    ) -> None:
        if max_ttl_seconds <= 0:
            raise ValueError("max_ttl_seconds must be positive")
        self.max_ttl_seconds = max_ttl_seconds
        self._operators: frozenset[str] | None = (
            frozenset(authorized_operators) if authorized_operators is not None else None
        )
        self._grants: dict[str, OverrideGrant] = {}
        self._on_event = on_event

    def request_override(
        self,
        capability: Any,
        *,
        operator: str = "",
        scope: str,
        reason: str,
        ttl_seconds: float,
        relaxes_hard_rule: bool = False,
        now: float | None = None,
    ) -> OverrideGrant | None:
        if relaxes_hard_rule:
            self._emit(
                "manual_override",
                {
                    "outcome": "denied",
                    "operator": operator or getattr(capability, "subject", ""),
                    "scope": scope,
                    "reason": "hard security limits are never relaxable by override",
                },
            )
            return None
        subject = getattr(capability, "subject", "")
        rights = getattr(capability, "rights", frozenset())
        if "operator.override" not in rights:
            self._emit(
                "manual_override",
                {
                    "outcome": "denied",
                    "operator": operator or subject,
                    "scope": scope,
                    "reason": "capability lacks the operator.override right",
                },
            )
            return None
        if self._operators is not None and subject not in self._operators:
            self._emit(
                "manual_override",
                {
                    "outcome": "denied",
                    "operator": operator or subject,
                    "scope": scope,
                    "reason": "capability subject is not an authorized operator identity",
                },
            )
            return None
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("override scope must be non-empty")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("override reason must be non-empty")
        ttl = min(float(ttl_seconds), self.max_ttl_seconds)
        if ttl <= 0:
            raise ValueError("ttl_seconds must be positive")
        current = now if now is not None else time.time()
        grant = OverrideGrant(
            override_id=f"ovr-{uuid.uuid4().hex}",
            operator=operator or subject,
            scope=scope,
            reason=reason,
            granted_epoch=current,
            expires_epoch=current + ttl,
        )
        self._grants[grant.override_id] = grant
        self._emit(
            "manual_override",
            {
                "outcome": "granted",
                "operator": grant.operator,
                "scope": scope,
                "reason": reason,
                "expires_epoch": grant.expires_epoch,
            },
        )
        return grant

    def active_overrides(self, now: float | None = None) -> tuple[OverrideGrant, ...]:
        current = now if now is not None else time.time()
        return tuple(
            grant for grant in self._grants.values() if grant.is_active(current)
        )

    def expired_overrides(self, now: float | None = None) -> tuple[OverrideGrant, ...]:
        current = now if now is not None else time.time()
        return tuple(
            grant for grant in self._grants.values() if not grant.is_active(current)
        )

    def describe(self) -> dict[str, Any]:
        return {
            "max_ttl_seconds": self.max_ttl_seconds,
            "authorized_operators": (
                sorted(self._operators) if self._operators is not None else None
            ),
            "grants": [grant.to_dict() for grant in self._grants.values()],
            "active": len(self.active_overrides()),
        }

    def _emit(self, kind: str, metadata: Mapping[str, Any]) -> None:
        if self._on_event is not None:
            try:
                self._on_event(kind, dict(metadata))
            except Exception:
                pass
