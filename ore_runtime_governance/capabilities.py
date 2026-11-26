"""Capability-based authority plane (integrated steps 15-22).

A capability is a sealed reference granting a narrow operation on a specific
resource: C = (subject, object, rights, constraints, expiry, issuer,
generation). ``CapabilityAuthority`` implements the validation formula of
section 5.3 -- authenticated subject matches, object matches, right granted,
constraints hold, t < expiry, generation current -- plus attenuation (child
authority never exceeds parent rights, scope, or expiry), short-lived
task-scoped grants, and a revocation registry with per-subject generation
counters so a worker restart or trust downgrade invalidates old grants
(integrated steps 17 and 22). Identity and authority stay separate: a
component ID identifies a component, the capability grants permission
(step 19).
"""
from __future__ import annotations

import hmac
import secrets as _secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .contracts import (
    CapabilityDeniedError,
    canonical_json,
    validate_component_id,
    validate_right,
)

#: Default resource scope granted per right when a manifest is registered.
DEFAULT_RIGHT_SCOPES: dict[str, str] = {
    "ipc.send": "ipc",
    "task.execute": "task",
    "workspace.read_snapshot": "workspace",
    "workspace.propose_fact": "workspace",
    "workspace.append_observation": "workspace",
    "workspace.request_merge": "workspace",
    "usg.read_snapshot": "usg",
    "usg.propose_update": "usg",
    "artifact.read": "artifacts",
    "artifact.publish": "artifacts",
    "sandbox.execute": "sandbox",
    "host.network": "network",
    "host.filesystem.write": "files",
    "remote.execute": "remote",
    "operator.override": "operator",
}

#: Constraint vocabulary understood by ``CapabilityAuthority.authorize``.
CONSTRAINT_KEYS: frozenset[str] = frozenset(
    {"fields", "max_payload_bytes", "task_id", "path_prefixes", "hosts"}
)

_CONSTRAINT_REQUEST_KEYS = {
    "fields": "field",
    "max_payload_bytes": "payload_bytes",
    "task_id": "task_id",
    "path_prefixes": "path",
    "hosts": "host",
}


@dataclass(frozen=True)
class AccessRequest:
    """The operation a subject wants to perform on a resource (step 16)."""

    subject: str
    resource: str
    right: str
    constraints: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.subject, str) or not self.subject.strip():
            raise ValueError("request subject must be a non-empty component id")
        if not isinstance(self.resource, str) or not self.resource.strip():
            raise ValueError("request resource must be a non-empty scope string")
        validate_right(self.right, "right")
        if not isinstance(self.constraints, Mapping):
            raise TypeError("request constraints must be a mapping")


@dataclass(frozen=True)
class AuthorizationDecision:
    """Typed authorization outcome -- never an ambiguous bool (section 6)."""

    allowed: bool
    reason: str
    subject: str
    resource: str
    right: str
    generation: int
    policy_version: int = 1
    checked_at_epoch: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "subject": self.subject,
            "resource": self.resource,
            "right": self.right,
            "generation": self.generation,
            "policy_version": self.policy_version,
            "checked_at_epoch": self.checked_at_epoch,
        }


def scope_matches(granted_scope: str, requested_resource: str) -> bool:
    """True when ``requested_resource`` lies inside ``granted_scope``.

    Scope matching is prefix-based without wildcards: "workspace" covers
    "workspace/lexical_hypotheses" but not "workspace-secret".
    """
    if not granted_scope or not requested_resource:
        return False
    if granted_scope == requested_resource:
        return True
    prefix = granted_scope if granted_scope.endswith("/") else granted_scope + "/"
    return requested_resource.startswith(prefix)


@dataclass(frozen=True)
class Capability:
    """Sealed grant of narrow rights on a scoped resource (step 15)."""

    capability_id: str
    subject: str
    resource: str
    rights: frozenset[str]
    constraints: Mapping[str, Any] = field(default_factory=dict)
    issued_at_epoch: float = 0.0
    expires_at_epoch: float = 0.0
    issuer: str = "capability_authority"
    generation: int = 0
    task_id: str | None = None
    mac: str = ""

    def __post_init__(self) -> None:
        validate_component_id(self.subject, "capability subject")
        if not isinstance(self.resource, str) or not self.resource.strip():
            raise ValueError("capability resource scope must be non-empty")
        if isinstance(self.rights, str):
            object.__setattr__(self, "rights", frozenset({self.rights}))
        if not isinstance(self.rights, frozenset):
            object.__setattr__(self, "rights", frozenset(self.rights))
        if not self.rights:
            raise ValueError("capability must carry at least one right")
        for right in self.rights:
            validate_right(right, "capability rights")
        if not isinstance(self.constraints, Mapping):
            raise TypeError("capability constraints must be a mapping")
        if self.expires_at_epoch <= self.issued_at_epoch:
            raise ValueError("capability expiry must be after issuance")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int):
            raise TypeError("generation must be an int")
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if not isinstance(self.capability_id, str) or not self.capability_id.strip():
            raise ValueError("capability_id must be non-empty")

    def covers(self, requested_resource: str) -> bool:
        return scope_matches(self.resource, requested_resource)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "subject": self.subject,
            "resource": self.resource,
            "rights": sorted(self.rights),
            "constraints": dict(self.constraints),
            "issued_at_epoch": self.issued_at_epoch,
            "expires_at_epoch": self.expires_at_epoch,
            "issuer": self.issuer,
            "generation": self.generation,
            "task_id": self.task_id,
        }


class RevocationRegistry:
    """Revocation records and per-subject generation counters (step 22)."""

    def __init__(self) -> None:
        self._revoked_ids: set[str] = set()
        self._generations: dict[str, int] = {}

    def generation_of(self, subject: str) -> int:
        return self._generations.get(subject, 0)

    def bump_generation(self, subject: str) -> int:
        """Invalidate every previously issued grant for the subject."""
        validate_component_id(subject, "revocation subject")
        self._generations[subject] = self._generations.get(subject, 0) + 1
        return self._generations[subject]

    def revoke_capability(self, capability_id: str) -> None:
        if not isinstance(capability_id, str) or not capability_id.strip():
            raise ValueError("capability_id must be non-empty")
        self._revoked_ids.add(capability_id)

    def is_revoked(self, capability: Capability) -> bool:
        return (
            capability.capability_id in self._revoked_ids
            or capability.generation < self.generation_of(capability.subject)
        )

    def revoked_capability_ids(self) -> frozenset[str]:
        return frozenset(self._revoked_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "revoked_capability_ids": sorted(self._revoked_ids),
            "generations": dict(sorted(self._generations.items())),
        }


class CapabilityAuthority:
    """Issues, verifies, attenuates, and revokes capabilities (steps 15-22)."""

    def __init__(self, secret: str | bytes | None = None, clock=None) -> None:
        raw = secret if secret is not None else _secrets.token_hex(32)
        self._secret = raw.encode("utf-8") if isinstance(raw, str) else raw
        self._clock = clock or time.time
        self.registry = RevocationRegistry()

    # -- sealing -----------------------------------------------------------

    def _seal(self, capability_id: str, payload: str) -> str:
        message = f"{capability_id}|{payload}".encode("utf-8")
        return hmac.new(self._secret, message, "sha256").hexdigest()

    def _verify_seal(self, capability: Capability) -> bool:
        expected = self._seal(capability.capability_id, self._sealable_payload(capability))
        return hmac.compare_digest(expected, capability.mac)

    def _sealable_payload(self, capability: Capability) -> str:
        return canonical_json(
            {
                "subject": capability.subject,
                "resource": capability.resource,
                "rights": sorted(capability.rights),
                "constraints": dict(capability.constraints),
                "issued_at_epoch": capability.issued_at_epoch,
                "expires_at_epoch": capability.expires_at_epoch,
                "issuer": capability.issuer,
                "generation": capability.generation,
                "task_id": capability.task_id,
            }
        )

    def _validate_constraints(self, constraints: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(constraints, Mapping):
            raise TypeError("constraints must be a mapping")
        unknown = set(constraints).difference(CONSTRAINT_KEYS)
        if unknown:
            raise ValueError(
                f"unknown constraint keys {sorted(unknown)}; supported: {sorted(CONSTRAINT_KEYS)}"
            )
        validated: dict[str, Any] = {}
        for key, value in constraints.items():
            if key in {"fields", "path_prefixes", "hosts"}:
                if isinstance(value, str):
                    value = [value]
                items = tuple(value)
                if not items or not all(isinstance(item, str) and item for item in items):
                    raise ValueError(f"constraint {key} must be a non-empty sequence of strings")
                validated[key] = list(items)
            elif key == "max_payload_bytes":
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError("constraint max_payload_bytes must be a positive int")
                validated[key] = value
            elif key == "task_id":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("constraint task_id must be a non-empty string")
                validated[key] = value
        return validated

    # -- issuing -----------------------------------------------------------

    def issue(
        self,
        subject: str,
        resource: str,
        rights: str | Iterable[str],
        *,
        constraints: Mapping[str, Any] | None = None,
        ttl_seconds: float = 3600.0,
        issuer: str = "capability_authority",
        task_id: str | None = None,
        now: float | None = None,
    ) -> Capability:
        """Issue a scoped, short-lived grant (steps 15 and 17)."""
        validate_component_id(subject, "capability subject")
        if not isinstance(resource, str) or not resource.strip():
            raise ValueError("resource scope must be non-empty")
        if isinstance(rights, str):
            rights = [rights]
        right_set = frozenset(rights)
        if not right_set:
            raise ValueError("at least one right is required")
        for right in right_set:
            validate_right(right, "rights")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive (short-lived grants only)")
        validated_constraints = self._validate_constraints(constraints or {})
        current = now if now is not None else self._clock()
        capability_id = f"cap-{uuid.uuid4().hex}"
        capability = Capability(
            capability_id=capability_id,
            subject=subject,
            resource=resource,
            rights=right_set,
            constraints=validated_constraints,
            issued_at_epoch=current,
            expires_at_epoch=current + ttl_seconds,
            issuer=issuer,
            generation=self.registry.generation_of(subject),
            task_id=task_id,
            mac="",
        )
        mac = self._seal(capability_id, self._sealable_payload(capability))
        return Capability(**{**capability.to_dict(), "rights": right_set, "mac": mac})

    def issue_component_grants(
        self,
        manifest,
        *,
        ttl_seconds: float = 3600.0,
        constraints: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Capability]:
        """Issue the baseline capability set declared by a manifest (step 16)."""
        constraints = constraints or {}
        grants: dict[str, Capability] = {}
        for right in manifest.capabilities:
            scope = DEFAULT_RIGHT_SCOPES.get(right, "runtime")
            if right == "task.execute":
                scope = f"task/{manifest.component_id}"
            elif right == "artifact.publish":
                scope = f"artifacts/{manifest.component_id}"
            elif right == "host.filesystem.write":
                scope = f"files/{manifest.component_id}"
            grants[right] = self.issue(
                manifest.component_id,
                scope,
                right,
                constraints=constraints.get(right),
                ttl_seconds=ttl_seconds,
                issuer="component_registration",
            )
        return grants

    # -- authorization (section 5.3) ---------------------------------------

    def authorize(
        self,
        capability: Capability,
        request: AccessRequest,
        now: float | None = None,
    ) -> AuthorizationDecision:
        """Evaluate valid(C, request, t) with typed, explainable denials."""
        current = now if now is not None else self._clock()

        def decision(allowed: bool, reason: str) -> AuthorizationDecision:
            return AuthorizationDecision(
                allowed=allowed,
                reason=reason,
                subject=request.subject,
                resource=request.resource,
                right=request.right,
                generation=(
                    capability.generation
                    if isinstance(capability, Capability)
                    else self.registry.generation_of(request.subject)
                ),
                checked_at_epoch=current,
            )

        if not isinstance(capability, Capability):
            return decision(False, "authorization requires a Capability object")
        if not self._verify_seal(capability):
            return decision(False, "invalid capability seal: grant is forged or tampered")
        if not isinstance(request, AccessRequest):
            return decision(False, "authorization requires an AccessRequest object")
        if self.registry.is_revoked(capability):
            return decision(
                False,
                "capability revoked or generation superseded "
                "(subject restarted, downgraded, or revoked)",
            )
        if current >= capability.expires_at_epoch:
            return decision(
                False,
                f"capability expired at {capability.expires_at_epoch} (now {current})",
            )
        if request.subject != capability.subject:
            return decision(
                False,
                f"authenticated subject {request.subject!r} does not match the "
                f"capability subject {capability.subject!r} (impersonation denied)",
            )
        if not capability.covers(request.resource):
            return decision(
                False,
                f"resource {request.resource!r} lies outside granted scope "
                f"{capability.resource!r}",
            )
        if request.right not in capability.rights:
            return decision(
                False,
                f"right {request.right!r} not granted; grants are {sorted(capability.rights)}",
            )
        constraint_violation = self._constraint_violation(capability, request)
        if constraint_violation:
            return decision(False, constraint_violation)
        return decision(True, "capability valid for the requested operation")

    def _constraint_violation(self, capability: Capability, request: AccessRequest) -> str | None:
        provided = request.constraints
        for key, bound in capability.constraints.items():
            request_key = _CONSTRAINT_REQUEST_KEYS[key]
            value = provided.get(request_key)
            if value is None:
                return (
                    f"constraint {key} cannot be evaluated: request did not declare "
                    f"{request_key}"
                )
            if key == "fields" and value not in set(bound):
                return f"constraint fields denies {value!r} (allowed: {sorted(bound)})"
            if key == "max_payload_bytes":
                if not isinstance(value, (int, float)) or value < 0:
                    return "constraint max_payload_bytes needs a numeric payload_bytes value"
                if value > bound:
                    return (
                        f"constraint max_payload_bytes denies {value} bytes "
                        f"(limit {bound})"
                    )
            if key == "task_id" and value != bound:
                return f"constraint task_id binds the grant to {bound!r} (got {value!r})"
            if key == "path_prefixes":
                if not isinstance(value, str) or not any(
                    value == prefix or value.startswith(prefix.rstrip("/") + "/")
                    for prefix in bound
                ):
                    return f"constraint path_prefixes denies {value!r} (allowed: {sorted(bound)})"
            if key == "hosts" and value not in set(bound):
                return f"constraint hosts denies {value!r} (allowed: {sorted(bound)})"
        return None

    def require(self, capability: Capability, request: AccessRequest, now: float | None = None) -> None:
        """Authorize or raise a typed :class:`CapabilityDeniedError` (step 16)."""
        decision = self.authorize(capability, request, now=now)
        if not decision.allowed:
            raise CapabilityDeniedError(
                decision.reason, subject=decision.subject, right=decision.right
            )

    # -- attenuation (section 5.3) ------------------------------------------

    def attenuate(
        self,
        parent: Capability,
        *,
        rights: str | Iterable[str] | None = None,
        resource: str | None = None,
        ttl_seconds: float | None = None,
        task_id: str | None = None,
        now: float | None = None,
    ) -> Capability:
        """Derive a child grant that only reduces authority (step 18 / 5.3).

        rights(child) is a subset of rights(parent); scope(child) lies within
        scope(parent); expiry(child) never extends past expiry(parent).
        """
        current = now if now is not None else self._clock()
        if not isinstance(parent, Capability) or not self._verify_seal(parent):
            raise CapabilityDeniedError("attenuation requires a validly sealed parent capability")
        if self.registry.is_revoked(parent):
            raise CapabilityDeniedError("cannot attenuate a revoked capability")
        if parent.expires_at_epoch <= current:
            raise CapabilityDeniedError("cannot attenuate an expired capability")

        child_rights = parent.rights if rights is None else (
            frozenset({rights}) if isinstance(rights, str) else frozenset(rights)
        )
        if not child_rights or not child_rights.issubset(parent.rights):
            raise CapabilityDeniedError(
                f"attenuation must only reduce authority: requested rights "
                f"{sorted(child_rights)} exceed parent {sorted(parent.rights)}"
            )
        child_scope = parent.resource if resource is None else resource
        if not scope_matches(parent.resource, child_scope):
            raise CapabilityDeniedError(
                f"attenuation scope {child_scope!r} escapes the parent scope "
                f"{parent.resource!r}"
            )
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("attenuation ttl_seconds must be positive")
        child_expiry = (
            min(parent.expires_at_epoch, current + ttl_seconds)
            if ttl_seconds is not None
            else parent.expires_at_epoch
        )
        if child_expiry > parent.expires_at_epoch or child_expiry <= current:
            raise CapabilityDeniedError(
                "attenuation may never extend the parent expiry "
                f"(child {child_expiry} vs parent {parent.expires_at_epoch})"
            )
        child_task = task_id if task_id is not None else parent.task_id
        capability_id = f"cap-{uuid.uuid4().hex}"
        child = Capability(
            capability_id=capability_id,
            subject=parent.subject,
            resource=child_scope,
            rights=child_rights,
            constraints=dict(parent.constraints),
            issued_at_epoch=current,
            expires_at_epoch=child_expiry,
            issuer=f"attenuated:{parent.issuer}",
            generation=parent.generation,
            task_id=child_task,
            mac="",
        )
        mac = self._seal(capability_id, self._sealable_payload(child))
        return Capability(
            capability_id=capability_id,
            subject=parent.subject,
            resource=child_scope,
            rights=child_rights,
            constraints=dict(parent.constraints),
            issued_at_epoch=current,
            expires_at_epoch=child_expiry,
            issuer=f"attenuated:{parent.issuer}",
            generation=parent.generation,
            task_id=child_task,
            mac=mac,
        )

    # -- revocation (step 22) -----------------------------------------------

    def revoke(self, capability: Capability) -> None:
        if not isinstance(capability, Capability):
            raise TypeError("revoke requires a Capability")
        self.registry.revoke_capability(capability.capability_id)

    def revoke_subject(self, subject: str) -> int:
        """Revoke every outstanding grant for a subject by generation bump."""
        return self.registry.bump_generation(subject)

    def generation_of(self, subject: str) -> int:
        return self.registry.generation_of(subject)
