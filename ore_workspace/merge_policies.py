"""Field-specific integration policies for heterogeneous cognitive representations."""
from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

from .models import Finding, WorkspaceState, content_fingerprint


def _finding_record(finding: Finding) -> dict[str, Any]:
    """Add publication identity and provenance to the detached payload for shared storage."""
    return {
        **copy.deepcopy(dict(finding.payload)),
        "finding_id": finding.finding_id,
        "producer_module": finding.producer_module,
        "finding_type": finding.finding_type,
        "confidence": finding.confidence,
        "provenance": finding.provenance.to_dict(),
        "dependencies": list(finding.dependencies),
        "timestamp": finding.timestamp,
    }


def _append_unique(items: list[dict[str, Any]], record: dict[str, Any], identity: str = "finding_id") -> bool:
    """Append a detached record only when its stable ID or content fingerprint is new."""
    token = record.get(identity) or content_fingerprint(record)
    if any((item.get(identity) or content_fingerprint(item)) == token for item in items):
        return False
    items.append(copy.deepcopy(record))
    return True


class MergePolicyRegistry:
    """Dispatch findings to field-specific policies instead of overwriting generic dictionaries."""

    def __init__(self) -> None:
        """Register built-in policies for all writable cognitive workspace fields."""
        self._policies: dict[str, Callable[[WorkspaceState, Finding], bool]] = {
            "lexical_hypotheses": self._merge_lexical,
            "syntactic_hypotheses": self._merge_list,
            "semantic_hypotheses": self._merge_list,
            "discourse_context": self._merge_discourse,
            "working_memory": self._merge_working_memory,
            "retrieved_memories": self._merge_memories,
            "unresolved_questions": self._merge_questions,
            "current_goal": self._merge_goal,
            "current_input": self._merge_input,
            "module_status": self._merge_module_status,
        }

    def register(self, field: str, policy: Callable[[WorkspaceState, Finding], bool]) -> None:
        """Replace a field's integration strategy with a caller-supplied merge function."""
        if not callable(policy):
            raise TypeError("merge policy must be callable")
        self._policies[field] = policy

    def apply(self, state: WorkspaceState, finding: Finding) -> bool:
        """Apply one finding through the policy assigned to its target field."""
        policy = self._policies.get(finding.target.field)
        if policy is None:
            raise ValueError(f"no merge policy registered for {finding.target.field}")
        return bool(policy(state, finding))

    def _merge_lexical(self, state: WorkspaceState, finding: Finding) -> bool:
        """Maintain independent candidate/evidence records for the target lexical item."""
        key = finding.target.key or str(finding.payload.get("word", ""))
        if not key:
            raise ValueError("lexical hypothesis requires target.key or payload.word")
        candidates = state.lexical_hypotheses.setdefault(key, [])
        return _append_unique(candidates, _finding_record(finding))

    def _merge_list(self, state: WorkspaceState, finding: Finding) -> bool:
        """Preserve multiple syntactic or semantic interpretations as separate alternatives."""
        field = finding.target.field
        items = getattr(state, field)
        return _append_unique(items, _finding_record(finding))

    def _merge_discourse(self, state: WorkspaceState, finding: Finding) -> bool:
        """Append discourse contributions to a keyed history without erasing prior context."""
        key = finding.target.key or "default"
        history = state.discourse_context.setdefault(key, [])
        return _append_unique(history, _finding_record(finding))

    def _merge_working_memory(self, state: WorkspaceState, finding: Finding) -> bool:
        """Update active memory items while representing removal as deactivation, not deletion."""
        key = finding.target.key or str(finding.payload.get("item_id", ""))
        if not key:
            raise ValueError("working-memory finding requires target.key or payload.item_id")
        current = state.working_memory.get(key, {})
        updated = {**copy.deepcopy(current), **copy.deepcopy(dict(finding.payload))}
        action = str(updated.get("action", "activate")).casefold()
        if action in {"remove", "deactivate", "mark_inactive"}:
            updated["is_active"] = False
        else:
            updated["is_active"] = bool(updated.get("is_active", True))
        updated.setdefault("item_id", key)
        updated["finding_ids"] = list(dict.fromkeys(
            [*current.get("finding_ids", []), finding.finding_id]
        ))
        state.working_memory[key] = updated
        return updated != current

    def _merge_memories(self, state: WorkspaceState, finding: Finding) -> bool:
        """Deduplicate memories by ID while retaining the separate evidence contributed by each finding."""
        record = _finding_record(finding)
        memory_id = record.get("memory_id")
        if memory_id is not None:
            for existing in state.retrieved_memories:
                if existing.get("memory_id") == memory_id:
                    evidence_ids = existing.setdefault("evidence_finding_ids", [])
                    if finding.finding_id not in evidence_ids:
                        evidence_ids.append(finding.finding_id)
                    old_score = float(existing.get("relevance_score", 0.0))
                    new_score = float(record.get("relevance_score", 0.0))
                    existing["relevance_score"] = max(old_score, new_score)
                    return True
        record["evidence_finding_ids"] = [finding.finding_id]
        return _append_unique(state.retrieved_memories, record, identity="memory_id")

    def _merge_questions(self, state: WorkspaceState, finding: Finding) -> bool:
        """Append question state transitions while retaining earlier unresolved-question evidence."""
        record = _finding_record(finding)
        question_id = finding.target.key or str(record.get("question_id", ""))
        if question_id:
            record["question_id"] = question_id
        return _append_unique(state.unresolved_questions, record)

    def _merge_goal(self, state: WorkspaceState, finding: Finding) -> bool:
        """Apply an explicitly authorized goal transition without changing other workspace fields."""
        value = finding.payload.get("goal", finding.payload.get("value"))
        if not isinstance(value, str) or not value.strip():
            raise ValueError("current_goal update requires a non-empty payload.goal or payload.value")
        changed = state.current_goal != value
        state.current_goal = value
        return changed

    def _merge_input(self, state: WorkspaceState, finding: Finding) -> bool:
        """Apply an authorized replacement input and advance its independent input revision."""
        value = finding.payload.get("input", finding.payload.get("value"))
        if not isinstance(value, str):
            raise ValueError("current_input update requires a string payload.input or payload.value")
        changed = state.current_input != value
        if changed:
            state.current_input = value
            state.input_revision += 1
        return changed

    def _merge_module_status(self, state: WorkspaceState, finding: Finding) -> bool:
        """Record the publishing module's lifecycle status in the status map."""
        module_id = finding.target.key or finding.producer_module
        status = finding.payload.get("status", finding.payload.get("value"))
        if not isinstance(status, str) or not status.strip():
            raise ValueError("module_status update requires a non-empty status string")
        changed = state.module_status.get(module_id) != status
        state.module_status[module_id] = status
        return changed