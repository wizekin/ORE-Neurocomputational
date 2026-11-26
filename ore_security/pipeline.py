"""Security decision pipeline and ORE integration (spec sections 2.2 and 6).

``SecurityDecisionPipeline`` implements the ten-step protected-operation sequence:
authenticate the caller, resolve roles, classify the resource, check policy
(default-deny, fail-closed), establish an authenticated channel for remote calls,
verify artifact signatures and integrity, execute the operation, validate the result
and any required proof, record an auditable event, and surface credential
revocation needs. ``CoordinatorGuard`` demonstrates that the ORE Main Brain
Coordinator cannot override security policy by raising task priority.
``SecuredRecordStore`` gates USG / shared-workspace updates on artifact
verification, authorization and semantic validation, because cryptographic validity
alone does not make a record semantically true.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .audit.events import SecurityEvent
from .audit.service import AuditService
from .canonical import digest_structure
from .contracts import AuthorizationRequest, PolicyDecision, Principal, SignedArtifactManifest
from .exceptions import AuthenticationError, AuthorizationError, SecurityError
from .identity.identities import IdentityService, RequestAuthenticator
from .policy.classification import DataClassification, ResourceRegistry
from .policy.rules import PolicyEngine
from .proofs.service import ProofService
from .signing.artifacts import ArtifactSigningService

Clock = Callable[[], float]

SemanticValidator = Callable[[Any], bool]


@dataclass
class ProtectedOperationResult:
    """Outcome of one protected operation through the ten-step pipeline."""

    action: str
    resource: str
    correlation_id: str
    allowed: bool
    executed: bool
    decision: PolicyDecision | None
    principal_id: str
    artifact_verification: dict[str, Any] | None = None
    proof_verification: dict[str, Any] | None = None
    semantic_valid: bool | None = None
    result: Any = None
    verified: bool = False
    error_category: str = ""
    failure_reason: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "resource": self.resource,
            "correlation_id": self.correlation_id,
            "allowed": self.allowed,
            "executed": self.executed,
            "decision": self.decision.to_dict() if self.decision else None,
            "principal_id": self.principal_id,
            "artifact_verification": self.artifact_verification,
            "proof_verification": self.proof_verification,
            "semantic_valid": self.semantic_valid,
            "verified": self.verified,
            "error_category": self.error_category,
            "failure_reason": self.failure_reason,
            "duration_ms": self.duration_ms,
        }


class SecurityDecisionPipeline:
    """Execute protected operations through the ten-step security sequence."""

    def __init__(self, identity_service: IdentityService,
                 policy_engine: PolicyEngine, audit_service: AuditService, *,
                 signing_service: ArtifactSigningService | None = None,
                 proof_service: ProofService | None = None,
                 resource_registry: ResourceRegistry | None = None,
                 request_authenticator: RequestAuthenticator | None = None,
                 clock: Clock = time.time) -> None:
        self.identity_service = identity_service
        self.policy_engine = policy_engine
        self.audit_service = audit_service
        self.signing_service = signing_service
        self.proof_service = proof_service
        self.resource_registry = resource_registry or policy_engine.resource_registry
        self.request_authenticator = (request_authenticator
                                      or RequestAuthenticator(identity_service, clock=clock))
        self.clock = clock

    def execute(self, *, action: str, resource: str, purpose: str,
                credential_token: str | None = None,
                request_envelope: Mapping[str, str] | None = None,
                body: bytes | str = b"", context: Mapping[str, Any] | None = None,
                expected_audience: str = "ore-service",
                classification: str = "INTERNAL",
                operation: Callable[[], Any] | None = None,
                artifact: tuple[SignedArtifactManifest, bytes] | None = None,
                proof: tuple[dict[str, Any], Any] | None = None,
                semantic_validator: SemanticValidator | None = None,
                expected_artifact_use: str | None = None,
                authorized_signers: frozenset[str] = frozenset(),
                correlation_id: str | None = None,
                remote_peer_establishment: Callable[[], dict[str, Any]] | None = None,
                ) -> ProtectedOperationResult:
        """Run one protected operation; every failure fails closed with an audit event."""
        correlation = correlation_id or ("corr-" + uuid.uuid4().hex[:12])
        started = self.clock()
        request_context = dict(context or {})
        result = ProtectedOperationResult(
            action=action, resource=resource, correlation_id=correlation,
            allowed=False, executed=False, decision=None, principal_id="",
        )

        def finish(error: SecurityError | None = None) -> ProtectedOperationResult:
            result.duration_ms = (self.clock() - started) * 1000.0
            result.error_category = error.error_category if error else ""
            if error is not None:
                result.failure_reason = error.message
            self._audit(result, error)
            return result

        try:
            # Step 1-2: authenticate the caller and resolve roles.
            if request_envelope is not None:
                principal = self.request_authenticator.verify_request(
                    request_envelope, expected_audience=expected_audience, body=body,
                )
            elif credential_token is not None:
                principal = self.identity_service.authenticate_token(
                    credential_token, expected_audience=expected_audience,
                )
            else:
                raise AuthenticationError("no credential presented: unauthenticated request rejected")
            result.principal_id = principal.principal_id

            # Step 3: classify the requested resource.
            descriptor = self.resource_registry.resolve_or_default(resource)
            if descriptor is not None:
                classification = descriptor.classification.name

            # Step 4: check purpose, policy, time, context (default-deny, fail-closed).
            request = AuthorizationRequest(
                principal_id=principal.principal_id, action=action, resource=resource,
                purpose=purpose, context=request_context, data_classification=classification,
            )
            decision = self.policy_engine.decide(request, principal, descriptor)
            result.decision = decision
            if not decision.allowed:
                # task priority in context can never grant access (spec section 2.2)
                if "task_priority" in request_context:
                    result.failure_reason = (
                        "coordinator task priority does not participate in the "
                        "authorization predicate"
                    )
                return finish()
            result.allowed = True

            # Step 5: establish an authenticated secure channel for remote calls.
            if remote_peer_establishment is not None:
                remote_peer_establishment()

            # Step 6: verify artifact signatures and integrity.
            if artifact is not None:
                manifest, artifact_bytes = artifact
                if self.signing_service is None:
                    raise AuthorizationError("no signing service configured for artifact verification")
                report = self.signing_service.verify_artifact(
                    manifest, artifact_bytes, expected_use=expected_artifact_use,
                    authorized_signers=authorized_signers,
                )
                result.artifact_verification = report

            # Step 7: execute the permitted operation.
            if operation is not None:
                result.result = operation()
                result.executed = True

            # Step 8: validate the result and any required proof.
            if proof is not None:
                if self.proof_service is None:
                    raise AuthorizationError("no proof service configured for proof verification")
                proof_record, statement = proof
                verification = self.proof_service.verify(proof_record, statement)
                result.proof_verification = verification.to_dict()
                if not verification.verified:
                    result.verified = False
                    result.failure_reason = (
                        "proof verification failed: operation is not marked verified"
                    )
                    return finish()
            result.verified = True

            # Integrity is separate from semantic validity (spec step 25).
            if semantic_validator is not None:
                payload = result.result if result.result is not None else (
                    artifact[1] if artifact is not None else None
                )
                result.semantic_valid = bool(semantic_validator(payload))
                if not result.semantic_valid:
                    result.failure_reason = (
                        "cryptographic checks passed but semantic validation failed; "
                        "record is not treated as true"
                    )
                    return finish()
            return finish()
        except SecurityError as error:
            error.correlation_id = correlation
            return finish(error)

    def _audit(self, result: ProtectedOperationResult,
               error: SecurityError | None) -> None:
        """Step 9: record an auditable event without protected content."""
        event = SecurityEvent(
            action=result.action,
            result=("allowed" if result.allowed else "denied") if error is None else "error",
            correlation_id=result.correlation_id,
            timestamp=self.clock(),
            principal_id=result.principal_id,
            resource_ref=result.resource,
            policy_decision=(result.decision.reason_code if result.decision else ""),
            decision_reason=result.failure_reason or (
                result.decision.reason_code if result.decision else ""),
            algorithm="Ed25519" if result.artifact_verification else "",
            error_category=result.error_category,
            duration_ms=result.duration_ms,
            details={
                "executed": result.executed,
                "verified": result.verified,
                "semantic_valid": result.semantic_valid,
                "artifact_verified": bool(result.artifact_verification and
                                          result.artifact_verification.get("verified")),
            },
        )
        self.audit_service.record(event)


class CoordinatorGuard:
    """Wrap pipeline access for the ORE Main Brain Coordinator.

    The coordinator may request priority for a task, but priority is not part of
    the authorization predicate: A(i,o,r,c) never reads it. A priority request on
    a denied operation is recorded as a rejected override attempt.
    """

    def __init__(self, pipeline: SecurityDecisionPipeline) -> None:
        self.pipeline = pipeline

    def request_with_priority(self, *, task_priority: int, **kwargs: Any) -> ProtectedOperationResult:
        """Execute with an explicit priority marker that policy deliberately ignores."""
        context = dict(kwargs.pop("context", {}) or {})
        context["task_priority"] = int(task_priority)
        result = self.pipeline.execute(context=context, **kwargs)
        if not result.allowed:
            self.pipeline.audit_service.record_decision(
                action="coordinator.priority_override_attempt",
                principal_id=result.principal_id or "coordinator",
                resource_ref=kwargs.get("resource", ""),
                policy_decision="DENIED",
                decision_reason="task priority cannot override security policy",
                result="rejected",
                correlation_id=result.correlation_id,
                error_category="PRIORITY_OVERRIDE_REJECTED",
            )
        return result


class SecuredRecordStore:
    """Gate USG / shared-workspace record ingestion on security and semantics.

    A record enters the store only when its artifact verifies (digest, signature,
    signer authorization, intended use), the writer is authorized, and an
    independent semantic validator accepts the content. Cryptographic validity
    alone never makes a record semantically true.
    """

    def __init__(self, pipeline: SecurityDecisionPipeline,
                 sink: MutableMappingLike | None = None) -> None:
        self.pipeline = pipeline
        self.sink = sink if sink is not None else {}

    def ingest(self, record_id: str, record_bytes: bytes,
               manifest: SignedArtifactManifest, credential_token: str, *,
               purpose: str, resource_uri: str,
               semantic_validator: SemanticValidator,
               expected_use: str,
               authorized_signers: frozenset[str] = frozenset(),
               correlation_id: str | None = None) -> dict[str, Any]:
        """Attempt to store one record; refusal reasons are explicit."""
        result = self.pipeline.execute(
            action="usg.record.ingest",
            resource=resource_uri,
            purpose=purpose,
            credential_token=credential_token,
            context={"record_id": record_id},
            artifact=(manifest, record_bytes),
            expected_artifact_use=expected_use,
            authorized_signers=authorized_signers,
            semantic_validator=lambda payload: semantic_validator(payload if payload is not None else record_bytes),
            correlation_id=correlation_id,
        )
        if not result.allowed:
            return {"stored": False, "reason": result.failure_reason or "denied",
                    "correlation_id": result.correlation_id}
        if result.artifact_verification is None or not result.artifact_verification.get("verified"):
            return {"stored": False, "reason": "artifact verification did not pass",
                    "correlation_id": result.correlation_id}
        if not result.semantic_valid:
            return {"stored": False,
                    "reason": "semantic validation failed: intact records can still be false",
                    "correlation_id": result.correlation_id}
        self.sink[record_id] = record_bytes
        return {"stored": True, "record_id": record_id,
                "correlation_id": result.correlation_id,
                "verified": result.verified,
                "semantic_valid": True}


# Minimal typing shim to avoid importing typing.Protocol variations across versions.
MutableMappingLike = Any


def build_default_trust_stack(*, clock: Clock = time.time,
                              sqlite_path: Any | None = None,
                              ) -> dict[str, Any]:
    """Assemble a complete CTPL stack wired for local operation (integration stage 1-2).

    Returns the identity service, policy engine, KMS, audit services, signing and
    proof services, and a ready ``SecurityDecisionPipeline``.
    """
    from .audit.events import RedactionEngine
    from .audit.ledger import AuditLedger
    from .audit.service import AuditService
    from .keys.kms import LocalKMS
    from .integrity.checkpoints import CheckpointService, ExternalAnchorVault
    from .signing.artifacts import ArtifactSigningService

    kms = LocalKMS(clock=clock)
    checkpoint_key = kms.generate_key("SIGNING", "Ed25519", key_id="key-audit-checkpoint")
    anchor_vault = ExternalAnchorVault(clock=clock)
    checkpoint_service = CheckpointService(kms, checkpoint_key.key_id, clock=clock)
    ledger = AuditLedger(
        redaction_engine=RedactionEngine(kms.canary_registry),
        checkpoint_service=checkpoint_service, anchor_vault=anchor_vault,
        sqlite_path=sqlite_path, clock=clock,
    )
    audit = AuditService(ledger, clock=clock)
    identity_service = IdentityService(clock=clock)
    policy_engine = PolicyEngine(clock=clock)
    signing = ArtifactSigningService(kms, clock=clock)
    proof_service = ProofService(clock=clock)
    pipeline = SecurityDecisionPipeline(
        identity_service, policy_engine, audit,
        signing_service=signing, proof_service=proof_service,
        clock=clock,
    )
    return {
        "kms": kms,
        "identity_service": identity_service,
        "policy_engine": policy_engine,
        "resource_registry": policy_engine.resource_registry,
        "audit": audit,
        "ledger": ledger,
        "checkpoint_service": checkpoint_service,
        "anchor_vault": anchor_vault,
        "signing": signing,
        "proof_service": proof_service,
        "pipeline": pipeline,
    }
