"""Audit event schema for the RLMS (Point 14, Step 54).

Lifecycle transitions, selections, executions, validations, drift alerts,
quarantines, and rollbacks are recorded as append-only events.  Corrections become
new linked events (``correction_of``), never in-place rewrites.  The schema
whitelists metadata keys and refuses raw sensitive payloads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping

__all__ = ("AuditEvent", "AUDIT_ALLOWED_METADATA_KEYS", "sanitize_audit_metadata",
           "AUDIT_FORBIDDEN_KEYS")


#: Metadata keys that may appear in an audit record.  Everything else is dropped.
AUDIT_ALLOWED_METADATA_KEYS: frozenset[str] = frozenset({
    "operation", "phase", "step", "status", "detail", "model_id", "model_version",
    "artifact_digest", "task_id", "selection_id", "candidate_ids", "utility",
    "utility_features", "fallback_reason", "reason", "validation_state",
    "validator", "validator_version", "outcome", "policy_reference", "explanation_id",
    "explanation_method", "confidence", "brier_score", "ece", "psi", "drift_metric",
    "resource_units", "duration_ms", "latency_ms", "memory_mb", "failure_category",
    "reviewer", "approval_id", "scope", "decision", "evidence_refs", "lineage_ref",
    "suite", "metric_name", "metric_value", "canary_fraction", "exposure_count",
    "rollforward_note", "correction_of", "sandbox_violation", "from_state", "to_state",
})

#: Raw payload / secret markers that must never be persisted (acceptance test).
AUDIT_FORBIDDEN_KEYS: frozenset[str] = frozenset({
    "raw_payload", "raw_input", "raw_output", "payload", "inputs", "outputs",
    "dataset_rows", "secret", "secrets", "password", "token", "api_key",
    "credentials", "authorization", "private_key", "generated_code", "code_text",
})


def sanitize_audit_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep only allowlisted metadata keys; forbidden keys are dropped entirely.

    This is the enforcement point for the acceptance criterion "Audit records do not
    leak raw sensitive payloads": raw inputs, outputs, code, and secrets are removed
    while references, hashes, and approved summaries survive.
    """
    kept: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if str(key) in AUDIT_FORBIDDEN_KEYS:
            continue
        if str(key) not in AUDIT_ALLOWED_METADATA_KEYS:
            continue
        kept[str(key)] = value
    return kept


@dataclass(slots=True)
class AuditEvent:
    """One append-only audit record (Step 54)."""

    event_id: str
    component: str
    operation: str
    event_kind: str  # lifecycle | selection | execution | validation | drift | approval
    status: str = "completed"
    task_id: str = ""
    model_id: str = ""
    model_version: str = ""
    duration_ms: float = 0.0
    exception_category: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    sequence: int = 0
    recorded_at: float = field(default_factory=time.time)
    correction_of: str = ""

    def __post_init__(self) -> None:
        """Sanitize metadata at construction time."""
        self.metadata = sanitize_audit_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"event_id": self.event_id, "component": self.component,
                "operation": self.operation, "event_kind": self.event_kind,
                "status": self.status, "task_id": self.task_id,
                "model_id": self.model_id, "model_version": self.model_version,
                "duration_ms": round(float(self.duration_ms), 3),
                "exception_category": self.exception_category,
                "metadata": dict(self.metadata), "sequence": self.sequence,
                "recorded_at": self.recorded_at, "correction_of": self.correction_of}
