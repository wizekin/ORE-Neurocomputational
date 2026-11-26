"""Field-specific change magnitudes and subscription relevance (Steps 11 and 12).

Step 11 requires a non-negative change magnitude ``Delta_f`` per changed field,
computed with a field-appropriate metric that is recorded with the event. Step 12
requires the subscription relevance ``R_m = sum_f A_mf * w_mf * Delta_f`` and
deterministic ordering by score with a stable tie-breaker. Both reuse the
existing ``ore_workspace.SubscriptionRegistry`` machinery; nothing here
introduces a separate universal ranking system.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ore_workspace.models import WORKSPACE_FIELDS


_COLLECTION_FIELDS = (
    "lexical_hypotheses", "syntactic_hypotheses", "semantic_hypotheses",
    "retrieved_memories", "unresolved_questions", "artifact_references",
)
_KEYED_COLLECTION_FIELDS = ("discourse_context",)
_BOOLEAN_FIELDS = ("current_input", "current_goal")
_KEYED_RECORD_FIELDS = ("working_memory", "module_status")


def _collection_items(value: Any) -> tuple[str, ...]:
    """Extract stable record identities from a list or keyed-list workspace field."""
    if isinstance(value, Mapping):
        items: list[str] = []
        for key in sorted(value):
            nested = value[key]
            if isinstance(nested, list):
                items.extend(f"{key}:{item.get('finding_id', index)}" for index, item in enumerate(nested))
            else:
                items.append(str(key))
        return tuple(items)
    if isinstance(value, list):
        return tuple(
            str(item.get("finding_id", index)) if isinstance(item, Mapping) else str(index)
            for index, item in enumerate(value)
        )
    return ()


def _collection_delta(old: Any, new: Any) -> float:
    """Return ``|A triangle B| / max(1, |A union B|)`` for collection fields (Step 11)."""
    old_items, new_items = set(_collection_items(old)), set(_collection_items(new))
    union = old_items | new_items
    symmetric = old_items ^ new_items
    return len(symmetric) / max(1, len(union))


def _keyed_record_delta(old: Mapping[str, Any], new: Mapping[str, Any]) -> float:
    """Return the changed-key ratio for keyed record fields such as working memory."""
    changed = {
        key for key in set(old) | set(new)
        if old.get(key) != new.get(key)
    }
    union = set(old) | set(new)
    return len(changed) / max(1, len(union))


@dataclass(frozen=True, slots=True)
class FieldChange:
    """Record one field's measured change magnitude and the metric used (Step 11)."""

    field: str
    delta: float
    metric: str


def measure_workspace_changes(
    old_state: Mapping[str, Any],
    new_state: Mapping[str, Any],
) -> tuple[FieldChange, ...]:
    """Measure non-negative change magnitudes between two workspace states (Step 11).

    Boolean fields use 0/1 changes, collections use the symmetric-difference
    ratio, and keyed record fields use the changed-key ratio. The metric chosen
    for each field is stable for that field and returned with the measurement.
    """
    changes: list[FieldChange] = []
    for field in WORKSPACE_FIELDS:
        old_value, new_value = old_state.get(field), new_state.get(field)
        if old_value == new_value:
            continue
        if field in _BOOLEAN_FIELDS:
            delta = 1.0 if old_value != new_value else 0.0
            metric = "boolean_indicator"
        elif field in _COLLECTION_FIELDS:
            delta = _collection_delta(old_value, new_value)
            metric = "collection_symmetric_difference_ratio"
        elif field in _KEYED_COLLECTION_FIELDS:
            delta = _collection_delta(old_value, new_value)
            metric = "keyed_collection_symmetric_difference_ratio"
        elif field in _KEYED_RECORD_FIELDS:
            delta = _keyed_record_delta(old_value or {}, new_value or {})
            metric = "keyed_record_changed_ratio"
        else:  # pragma: no cover - WORKSPACE_FIELDS fully partitioned above
            delta = 1.0
            metric = "boolean_indicator"
        if delta > 0:
            changes.append(FieldChange(field, min(1.0, max(0.0, delta)), metric))
    return tuple(changes)


def relevance_report(workspace, deltas: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    """Return eligible ``(module_id, R_m)`` pairs sorted by score then module id (Step 12).

    The relevance score is computed by the existing subscription registry as
    ``R_m = sum_f A_mf * w_mf * Delta_f``; this function only orders the result
    deterministically by descending score with a stable module-ID tie-breaker.
    """
    eligible = workspace.subscriptions.eligible(deltas)
    return tuple(sorted(eligible, key=lambda item: (-item[1], item[0])))


def deltas_mapping(changes: Mapping[str, float]) -> dict[str, float]:
    """Normalize a change mapping into non-negative deltas for the registry call."""
    normalized = {field: float(value) for field, value in changes.items()}
    if any(value < 0 for value in normalized.values()):
        raise ValueError("change magnitudes cannot be negative")
    return normalized


__all__ = [
    "FieldChange",
    "measure_workspace_changes",
    "deltas_mapping",
    "relevance_report",
]
