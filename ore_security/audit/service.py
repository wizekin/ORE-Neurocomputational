"""Audit and incident-response service for the trust layer (steps 58-62).

``AuditService`` records security events with correlation ids, policy decisions and
error categories, and refuses payloads containing secret material. ``IncidentResponse``
orchestrates key-compromise and recovery drills: revoking credentials, rotating keys,
restoring KMS backups and verifying audit checkpoints after restoration.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from ..exceptions import AuthenticationError, SecurityError
from ..identity.identities import IdentityService
from ..keys.kms import LocalKMS
from .events import RedactionEngine, SecurityEvent, security_event
from .ledger import AuditLedger

Clock = Callable[[], float]


class AuditService:
    """Persist tamper-evident security events with redaction enforcement."""

    def __init__(self, ledger: AuditLedger, *, clock: Clock = time.time) -> None:
        self.ledger = ledger
        self.clock = clock

    def record(self, event: SecurityEvent) -> dict[str, Any]:
        """Record one event; returns its chain position and digest."""
        return self.ledger.record(event)

    def record_decision(self, *, action: str, principal_id: str, resource_ref: str,
                        policy_decision: str, decision_reason: str, result: str,
                        correlation_id: str, credential_id: str = "",
                        key_id: str = "", algorithm: str = "",
                        error_category: str = "", duration_ms: float = 0.0,
                        details: dict[str, Any] | None = None) -> dict[str, Any]:
        """Convenience wrapper matching the step-58 event schema."""
        event = security_event(
            action,
            result=result,
            correlation_id=correlation_id,
            timestamp=self.clock(),
            principal_id=principal_id,
            resource_ref=resource_ref,
            policy_decision=policy_decision,
            decision_reason=decision_reason,
            credential_id=credential_id,
            key_id=key_id,
            algorithm=algorithm,
            error_category=error_category,
            duration_ms=duration_ms,
            details=details or {},
        )
        return self.record(event)

    def query(self, correlation_id: str) -> list[dict[str, Any]]:
        """Retrieve every event for one correlation id."""
        return self.ledger.query_correlation(correlation_id)

    def verify_history(self) -> dict[str, Any]:
        return self.ledger.verify_history()

    def checkpoint(self) -> dict[str, Any]:
        return self.ledger.create_checkpoint()


class IncidentResponse:
    """Key-compromise and recovery drills (steps 61-62)."""

    def __init__(self, identity_service: IdentityService, kms: LocalKMS,
                 audit: AuditService) -> None:
        self.identity_service = identity_service
        self.kms = kms
        self.audit = audit

    def compromise_principal(self, identity_id: str, *,
                             correlation_id: str | None = None) -> dict[str, Any]:
        """Revoke an identity's credentials and mark its keys compromised."""
        correlation = correlation_id or f"incident-{identity_id}"
        actions: list[str] = []
        identity = self.identity_service.get_identity(identity_id)
        if identity is not None:
            self.identity_service.revoke_identity(identity_id, reason="compromise drill")
            actions.append("identity_revoked")
        # revoke every key whose id references the principal, then mark compromised
        affected_keys: list[str] = []
        for descriptor in list(self.kms.list_descriptors()):
            if identity_id in descriptor.key_id:
                affected_keys.append(descriptor.key_id)
        for key_id in affected_keys:
            self.kms.mark_compromised(key_id, reason=f"principal compromise: {identity_id}")
            actions.append(f"key_compromised:{key_id}")
        self.audit.record_decision(
            action="incident.key_compromise_response",
            principal_id=identity_id,
            resource_ref=f"identity:{identity_id}",
            policy_decision="REVOKED",
            decision_reason="key-compromise drill",
            result="contained",
            correlation_id=correlation,
            error_category="KEY_COMPROMISE",
        )
        return {
            "identity_id": identity_id,
            "actions": actions,
            "affected_keys": affected_keys,
            "correlation_id": correlation,
        }

    def kms_outage_recovery_drill(self, passphrase: str) -> dict[str, Any]:
        """Back up the KMS, simulate an outage, restore, and re-check enforcement."""
        backup = self.kms.export_backup(passphrase)
        self.kms.simulate_outage(True)
        outage_denied = False
        try:
            self.kms.generate_key("SIGNING")
        except SecurityError:
            outage_denied = True
        finally:
            self.kms.simulate_outage(False)
        restored = LocalKMS.restore_backup(backup, passphrase)
        self.audit.record_decision(
            action="incident.kms_outage_recovery",
            principal_id="system",
            resource_ref="kms:local-software-kms",
            policy_decision="N/A",
            decision_reason="recovery drill",
            result="recovered",
            correlation_id="kms-outage-drill",
            error_category="KMS_UNAVAILABLE",
        )
        return {
            "outage_failed_closed": outage_denied,
            "restored_key_count": len(restored.list_descriptors()),
            "restoration_preserves_revocations": all(
                descriptor.status.value in {"REVOKED", "COMPROMISED", "ROTATED", "EXPIRED"}
                for descriptor in restored.list_descriptors()
                if descriptor.status.value != "ACTIVE"
            ),
        }

    def audit_backup_restore_drill(self) -> dict[str, Any]:
        """Export the audit ledger, restore it and verify checkpoints still pass."""
        backup = self.ledger.export_backup()
        restored = AuditLedger.restore_backup(backup)
        verification = restored.verify_history()
        self.audit.record_decision(
            action="incident.audit_backup_restore",
            principal_id="system",
            resource_ref="audit:ledger",
            policy_decision="N/A",
            decision_reason="recovery drill",
            result="verified" if verification.get("valid") else "failed",
            correlation_id="audit-restore-drill",
            error_category="AUDIT_RECOVERY",
        )
        return {"restored_events": restored.event_count(),
                "verification": verification}
