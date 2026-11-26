"""Security event schema and secret redaction (step 58, step 60).

Events record identity, action, resource reference, policy decision, credential and
key identifiers, algorithm, result, correlation id and error category. They never
carry secrets: a redaction engine scans serialized events against registered canary
values and forbidden patterns, and the audit service refuses to persist leaking
events instead of silently storing them.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..canonical import canonical_json
from ..contracts import SCHEMA_VERSION, utc_timestamp
from ..exceptions import RedactionError
from ..keys.kms import SecretCanaryRegistry

Clock = Callable[[], float]

FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\bprivate[_ ]?key\b\s*[:=]"),
    re.compile(r"(?i)\bpassphrase\b\s*[:=]"),
    re.compile(r"(?i)\bsecret[_ ]?material\b\s*[:=]"),
)
_HEX_KEY_LENGTHS = (64, 96, 128)


class RedactionEngine:
    """Scan event payloads for secret material before persistence."""

    def __init__(self, canary_registry: SecretCanaryRegistry | None = None) -> None:
        self.canary_registry = canary_registry or SecretCanaryRegistry()

    def scan_text(self, text: str) -> str | None:
        """Return a human-readable finding for the first leak detected, or None."""
        canary = self.canary_registry.find_in_text(text)
        if canary is not None:
            return "registered secret canary value present in payload"
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                return f"forbidden pattern {pattern.pattern!r} present in payload"
        return None

    def sanitize(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Return a copy with sensitive-looking fields dropped, or raise when a canary leaks."""
        serialized = canonical_json(dict(payload))
        finding = self.scan_text(serialized)
        if finding is not None and "canary" in finding:
            raise RedactionError(f"security event refused: {finding}")
        if finding is not None:
            raise RedactionError(f"security event refused: {finding}")
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            if any(marker in str(key).lower() for marker in
                   ("private", "passphrase", "secret", "password", "material")):
                cleaned[str(key)] = "[REDACTED]"
            else:
                cleaned[str(key)] = value
        return cleaned


@dataclass(frozen=True, slots=True)
class SecurityEvent:
    """One auditable security decision or verification outcome (step 58)."""

    action: str
    result: str
    correlation_id: str
    timestamp: float
    principal_id: str = ""
    resource_ref: str = ""
    policy_decision: str = ""
    decision_reason: str = ""
    credential_id: str = ""
    key_id: str = ""
    algorithm: str = ""
    error_category: str = ""
    duration_ms: float = 0.0
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("action", "result", "correlation_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.timestamp <= 0:
            raise ValueError("timestamp must be a positive POSIX time")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "event_id": "sev-" + uuid.uuid5(uuid.NAMESPACE_URL, canonical_json({
                "action": self.action, "correlation_id": self.correlation_id,
                "timestamp": self.timestamp, "details": dict(self.details),
            })).hex[:16],
            "timestamp_utc": utc_timestamp(self.timestamp),
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "principal_id": self.principal_id,
            "action": self.action,
            "resource_ref": self.resource_ref,
            "policy_decision": self.policy_decision,
            "decision_reason": self.decision_reason,
            "credential_id": self.credential_id,
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "result": self.result,
            "error_category": self.error_category,
            "duration_ms": self.duration_ms,
            "details": dict(self.details),
        }


def security_event(action: str, *, result: str, correlation_id: str | None = None,
                   timestamp: float = 0.0, **fields: Any) -> SecurityEvent:
    """Build a SecurityEvent with an auto-generated correlation id when absent."""
    return SecurityEvent(
        action=action,
        result=result,
        correlation_id=correlation_id or ("corr-" + uuid.uuid4().hex[:12]),
        timestamp=timestamp or _default_now(),
        **fields,
    )


def _default_now() -> float:
    import time
    return time.time()
