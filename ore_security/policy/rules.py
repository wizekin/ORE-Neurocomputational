"""Default-deny policy engine implementing A(i,o,r,c) (steps 3, 5, 8, 10).

The decision predicate grants an operation only when authentication, policy, data
classification and context constraints all pass:

    A(i, o, r, c) = I(i) and P(i, o, r, c) and D(r) and T(c)

Missing policy data means deny. Internal engine failure means deny (fail closed).
Coordinator task priority is deliberately absent from the predicate: raising priority
cannot override security policy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..contracts import AuthorizationRequest, PolicyDecision, Principal
from ..exceptions import PolicyEngineFailure
from .classification import DataClassification, ResourceDescriptor, ResourceRegistry

Clock = Callable[[], float]
POLICY_VERSION = "ore-security-policy-1.0.0"


@dataclass(frozen=True, slots=True)
class PolicyRule:
    """One allow rule: subject, action, resource prefix, classification ceiling, context."""

    rule_id: str
    subject: str  # exact principal id or "role:<ROLE>"
    actions: frozenset[str]
    resource_prefixes: tuple[str, ...]
    max_classification: DataClassification
    required_purpose: str | None = None
    required_scopes: frozenset[str] = field(default_factory=frozenset)
    context_constraints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rule_id.strip() or not self.subject.strip():
            raise ValueError("rule_id and subject must be non-empty")
        if not self.actions:
            raise ValueError("a rule must allow at least one action")
        if not self.resource_prefixes:
            raise ValueError("a rule must cover at least one resource prefix")
        if isinstance(self.max_classification, str):
            raise TypeError("max_classification must be a DataClassification")

    def subject_matches(self, principal: Principal) -> bool:
        """Match by exact principal id or by role selector."""
        if self.subject.startswith("role:"):
            return self.subject[len("role:"):] in principal.roles
        return self.subject == principal.principal_id


class PolicyEngine:
    """Evaluate authorization requests with default-deny semantics."""

    def __init__(self, *, resource_registry: ResourceRegistry | None = None,
                 rules: tuple[PolicyRule, ...] = (),
                 clock: Clock = time.time) -> None:
        self.resource_registry = resource_registry or ResourceRegistry()
        self._rules: list[PolicyRule] = list(rules)
        self.clock = clock
        self._simulate_failure = False
        self._decision_count = 0

    def add_rule(self, rule: PolicyRule) -> None:
        """Add one allow rule; ordering does not matter, all matching rules are checked."""
        if not isinstance(rule, PolicyRule):
            raise TypeError("rule must be a PolicyRule")
        self._rules.append(rule)

    def simulate_engine_failure(self, enabled: bool = True) -> None:
        """Simulate an authorization-service outage (recovery drill support)."""
        self._simulate_failure = bool(enabled)

    def decide(self, request: AuthorizationRequest, principal: Principal,
               resource: ResourceDescriptor | None = None,
               *, now: float | None = None) -> PolicyDecision:
        """Return the allow/deny decision with the full predicate breakdown."""
        current_time = self.clock() if now is None else float(now)
        self._decision_count += 1

        def deny(reason: str, *, identity_ok: bool = False, policy_ok: bool = False,
                 classification_ok: bool = False, context_ok: bool = False,
                 matched: tuple[str, ...] = ()) -> PolicyDecision:
            return PolicyDecision(
                allowed=False, policy_version=POLICY_VERSION, reason_code=reason,
                decision_expires_at=current_time, identity_ok=identity_ok,
                policy_ok=policy_ok, classification_ok=classification_ok,
                context_ok=context_ok, matched_rule_ids=matched,
            )

        # I(i): authentication
        if not isinstance(principal, Principal) or not principal.authenticated:
            return deny("UNAUTHENTICATED")
        if principal.expires_at and current_time >= principal.expires_at:
            return deny("PRINCIPAL_CREDENTIAL_EXPIRED", identity_ok=True)

        # resource resolution and D(r): classification data must exist
        resolved = resource if resource is not None else self.resource_registry.resolve_or_default(request.resource)
        if resolved is None:
            return deny("RESOURCE_NOT_CLASSIFIED", identity_ok=True)
        try:
            request_classification = DataClassification.parse(request.data_classification)
        except ValueError:
            return deny("REQUEST_CLASSIFICATION_UNKNOWN", identity_ok=True)

        if self._simulate_failure:
            # fail closed: engine outage must not silently allow protected access
            raise PolicyEngineFailure(
                "authorization engine unavailable; failing closed",
                correlation_id=None,
            )

        matched: list[str] = []
        best_partial: tuple[str, ...] = ()
        for rule in self._rules:
            if not rule.subject_matches(principal):
                continue
            if request.action not in rule.actions:
                continue
            if not any(request.resource.startswith(prefix) for prefix in rule.resource_prefixes):
                continue
            matched.append(rule.rule_id)

        if not matched:
            return deny("NO_MATCHING_POLICY", identity_ok=True)

        for rule in self._rules:
            if rule.rule_id not in matched:
                continue
            # scope requirements
            if any(not principal.has_scope(scope) for scope in rule.required_scopes):
                best_partial = (rule.rule_id, "MISSING_REQUIRED_SCOPE")
                continue
            # D(r): classification ceiling
            if request_classification > rule.max_classification or resolved.classification > rule.max_classification:
                best_partial = (rule.rule_id, "CLASSIFICATION_EXCEEDS_RULE")
                continue
            # purpose binding
            if rule.required_purpose is not None:
                if request.purpose != rule.required_purpose and request.purpose not in resolved.allowed_purposes:
                    best_partial = (rule.rule_id, "PURPOSE_NOT_PERMITTED")
                    continue
            # T(c): context constraints
            constraint_result = self._check_context_constraints(rule, request.context, principal)
            if constraint_result is not None:
                best_partial = (rule.rule_id, constraint_result)
                continue
            return PolicyDecision(
                allowed=True, policy_version=POLICY_VERSION, reason_code="ALLOWED_BY_RULE",
                decision_expires_at=current_time, identity_ok=True, policy_ok=True,
                classification_ok=True, context_ok=True,
                matched_rule_ids=(rule.rule_id,),
            )

        reason = best_partial[1] if best_partial else "NO_PASSING_RULE"
        matched_tuple = tuple(item for item in matched)
        classification_ok = best_partial[1] != "CLASSIFICATION_EXCEEDS_RULE" if best_partial else False
        context_ok = best_partial[1] == "PURPOSE_NOT_PERMITTED" if best_partial else False
        return deny(reason, identity_ok=True, policy_ok=True,
                    classification_ok=classification_ok, context_ok=context_ok,
                    matched=matched_tuple)

    @staticmethod
    def _check_context_constraints(rule: PolicyRule, context: Mapping[str, Any],
                                   principal: Principal) -> str | None:
        """Evaluate T(c); return a denial reason or None when constraints pass."""
        constraints = rule.context_constraints
        if not constraints:
            return None
        if "required_scopes" in constraints:
            needed = set(constraints["required_scopes"])
            if not needed.issubset(principal.scopes):
                return "CONTEXT_SCOPE_CONSTRAINT_FAILED"
        if "allowed_hours_utc" in constraints:
            hour = context.get("hour_utc")
            if hour is None or int(hour) not in constraints["allowed_hours_utc"]:
                return "CONTEXT_TIME_WINDOW_FAILED"
        if "require_context_flag" in constraints:
            flag = constraints["require_context_flag"]
            if not context.get(flag):
                return f"CONTEXT_FLAG_{flag.upper()}_REQUIRED"
        # NOTE: task priority is intentionally never consulted as a constraint;
        # the ORE coordinator cannot override policy by raising priority.
        return None

    def decision_statistics(self) -> dict[str, Any]:
        """Return counters for operations dashboards."""
        return {"decisions_evaluated": self._decision_count, "rules_loaded": len(self._rules)}


def threat_model_catalog() -> list[dict[str, str]]:
    """Return the adversary and threat scenario catalog consulted by policy (step 2)."""
    return [
        {"scenario": "credential_theft", "capability": "attacker holds a stolen credential token",
         "controls": "short ttl, audience binding, revocation, replay nonces"},
        {"scenario": "malicious_insider", "capability": "authenticated principal exceeds role",
         "controls": "least-privilege rules, per-resource checks, audit chain"},
        {"scenario": "replay_and_substitution", "capability": "resends accepted request or swaps body",
         "controls": "nonce cache, body-digest-bound signatures, clock skew window"},
        {"scenario": "message_tampering", "capability": "modifies data in transit",
         "controls": "authenticated encryption, sequence numbers"},
        {"scenario": "forged_artifacts", "capability": "publishes unsigned or modified artifacts",
         "controls": "manifest digests, signature verification, signer authorization"},
        {"scenario": "malicious_participants", "capability": "colluding aggregation members",
         "controls": "documented collusion thresholds, validity checks"},
        {"scenario": "poisoned_telemetry", "capability": "contributes malformed measurements",
         "controls": "VDAF validity proofs, cohort minimums, duplicate rejection"},
        {"scenario": "future_cryptanalysis", "capability": "harvest-now-decrypt-later adversary",
         "controls": "PQC inventory, hybrid envelopes, migration priority"},
        {"scenario": "compromised_endpoints", "capability": "controls a participant machine",
         "controls": "key compromise drills, revocation, checkpoint anchoring"},
        {"scenario": "denial_of_service", "capability": "exhausts security services",
         "controls": "fail-closed design with documented availability exceptions"},
    ]
