"""Field-level ownership rules for modules publishing into the shared workspace."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from .models import WORKSPACE_FIELDS


DEFAULT_WRITE_PERMISSIONS: dict[str, frozenset[str]] = {
    "lexical_processor": frozenset({"lexical_hypotheses"}),
    "syntax_processor": frozenset({"syntactic_hypotheses"}),
    "semantic_processor": frozenset({"semantic_hypotheses"}),
    "working_memory": frozenset({"working_memory"}),
    "memory_retrieval": frozenset({"retrieved_memories"}),
    "cognitive_control": frozenset({"unresolved_questions", "module_status"}),
    "discourse_processor": frozenset({"discourse_context"}),
    "goal_control": frozenset({"current_goal", "module_status"}),
}


class PermissionPolicy:
    """Authorize writes by the identity of the producer and the target workspace field."""

    def __init__(self, permissions: Mapping[str, Iterable[str]] | None = None) -> None:
        """Build a policy from explicit permissions or the documented ORE defaults."""
        source = permissions if permissions is not None else DEFAULT_WRITE_PERMISSIONS
        normalized: dict[str, frozenset[str]] = {}
        for module_id, fields in source.items():
            if not module_id or not module_id.strip():
                raise ValueError("module IDs must be non-empty")
            field_set = frozenset(fields)
            unknown = field_set.difference(WORKSPACE_FIELDS)
            if unknown:
                raise ValueError(f"unknown workspace fields for {module_id}: {sorted(unknown)}")
            normalized[module_id] = field_set
        self._permissions = normalized

    def can_write(self, module_id: str, field: str) -> bool:
        """Return whether a producer owns the requested field without wildcard escalation."""
        return field in self._permissions.get(module_id, frozenset())

    def allowed_fields(self, module_id: str) -> frozenset[str]:
        """Return the immutable field set currently granted to a producer."""
        return self._permissions.get(module_id, frozenset())

    def grant(self, module_id: str, fields: Iterable[str]) -> None:
        """Add explicitly validated field permissions for a registered module identity."""
        requested = frozenset(fields)
        unknown = requested.difference(WORKSPACE_FIELDS)
        if unknown:
            raise ValueError(f"unknown workspace fields: {sorted(unknown)}")
        self._permissions[module_id] = self._permissions.get(module_id, frozenset()) | requested

    def revoke(self, module_id: str, fields: Iterable[str]) -> None:
        """Remove only the selected permissions while preserving other module grants."""
        current = self._permissions.get(module_id, frozenset())
        self._permissions[module_id] = current.difference(fields)

    def as_dict(self) -> dict[str, list[str]]:
        """Expose a detached, deterministically ordered policy representation."""
        return {key: sorted(value) for key, value in sorted(self._permissions.items())}