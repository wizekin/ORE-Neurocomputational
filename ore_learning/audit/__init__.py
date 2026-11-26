"""Audit event schemas and SQLite logger integration (Point 14, Section 5)."""
from .events import (  # noqa: F401
    AUDIT_ALLOWED_METADATA_KEYS,
    AUDIT_FORBIDDEN_KEYS,
    AuditEvent,
    sanitize_audit_metadata,
)
from .sqlite import AuditStore, connect_audit_store  # noqa: F401

__all__ = (
    "AUDIT_ALLOWED_METADATA_KEYS",
    "AUDIT_FORBIDDEN_KEYS",
    "AuditEvent",
    "AuditStore",
    "connect_audit_store",
    "sanitize_audit_metadata",
)
