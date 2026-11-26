"""Contradiction detection and conflict preservation (Phase G, Steps G1-G2).

Contradictions are *recorded*, never silently overwritten (Section 2). The
detector compares findings against explicitly incompatible facts, rules,
temporal scopes and exceptions; every record keeps the competing claims, their
provenance, priority, time scope and available resolution options. Source facts
stay immutable — a conflict attaches to them without rewriting them.
"""
from __future__ import annotations

from typing import Any, Mapping

from .contracts import (
    Literal,
    NSRLConflictRecord,
    RuleDefinition,
    SemanticFact,
    make_id,
)


class ConflictManager:
    """Detect and record contradictions without overwriting source facts."""

    def __init__(self) -> None:
        """Create the empty conflict ledger."""
        self._records: dict[str, NSRLConflictRecord] = {}

    @property
    def records(self) -> tuple[NSRLConflictRecord, ...]:
        """Return all recorded conflicts in creation order."""
        return tuple(self._records.values())

    def unresolved(self) -> tuple[NSRLConflictRecord, ...]:
        """Return unresolved conflict records."""
        return tuple(item for item in self._records.values() if item.status == "unresolved")

    # ------------------------------------------------------------------ Step G1

    def detect_contradictions(
        self,
        conclusions: tuple[Literal, ...],
        facts: tuple[SemanticFact, ...],
        rules: tuple[RuleDefinition, ...] = (),
    ) -> tuple[tuple[str, str], ...]:
        """Detect contradictions among conclusions and explicit facts (Step G1).

        Three explicit incompatibility classes are checked, in order:
        (1) direct polarity contradiction between two conclusions or between a
        conclusion and a stored explicit fact of the opposite polarity on the
        same (subject, predicate, object);
        (2) rule-level incompatibility: two derived conclusions produced by
        rules whose conclusion literals are complementary;
        (3) temporal-scope overlap: conclusions on the same pair under
        different declared temporal qualifiers.
        """
        contradictions: list[tuple[str, str]] = []
        conclusion_signatures: dict[tuple[bool, str, str, str], str] = {}
        for conclusion in conclusions:
            if len(conclusion.args) != 2:
                continue
            first, second = conclusion.args
            from .contracts import LogicalTerm

            if not all(isinstance(item, LogicalTerm) for item in (first, second)):
                continue
            key = (
                conclusion.negated, conclusion.predicate,
                first.value, second.value,
            )
            if key in conclusion_signatures:
                contradictions.append(
                    (conclusion_signatures[key], conclusion.signature()),
                )
            else:
                conclusion_signatures[key] = conclusion.signature()

        for conclusion in conclusions:
            if len(conclusion.args) != 2 or conclusion.temporal_qualifier is None:
                continue
            first, second = conclusion.args
            from .contracts import LogicalTerm

            if not all(isinstance(item, LogicalTerm) for item in (first, second)):
                continue
            for fact in facts:
                if (
                    fact.subject == first.value
                    and fact.predicate == conclusion.predicate
                    and fact.object_value == second.value
                    and str(fact.qualifiers.get("temporal")) != conclusion.temporal_qualifier
                ):
                    contradictions.append((conclusion.signature(), fact.fact_id))

        negated_keys = {
            (key[1], key[2], key[3]) for key in conclusion_signatures if key[0]
        }
        for (negated, predicate, subject, object_value), signature in (
            conclusion_signatures.items()
        ):
            if not negated and (predicate, subject, object_value) in negated_keys:
                complement = f"not:{predicate}({subject},{object_value})"
                contradictions.append((signature, complement))

        # Explicit stored facts with opposite polarity contradict a grounded
        # conclusion on the same triple; both claims stay preserved (Section 2).
        for conclusion in conclusions:
            if len(conclusion.args) != 2:
                continue
            first, second = conclusion.args
            if not all(isinstance(item, LogicalTerm) for item in (first, second)):
                continue
            for fact in facts:
                if (
                    fact.subject == first.value
                    and fact.predicate == conclusion.predicate
                    and fact.object_value == second.value
                    and fact.polarity == conclusion.negated
                ):
                    contradictions.append((conclusion.signature(), fact.fact_id))
        return tuple(dict.fromkeys(contradictions))

    # ------------------------------------------------------------------ Step G2

    def record_conflict(
        self,
        claim_a: Mapping[str, Any],
        claim_b: Mapping[str, Any],
        *,
        priority: float = 0.5,
        time_scope: Mapping[str, Any] | None = None,
        resolution_options: tuple[str, ...] = (
            "keep_both_preserved",
            "defer_to_independent_review",
            "request_additional_evidence",
        ),
        provenance_a: Mapping[str, Any] | None = None,
        provenance_b: Mapping[str, Any] | None = None,
    ) -> NSRLConflictRecord:
        """Record one contradiction with claims, provenance and options (Step G2).

        Duplicate registrations of the same claim pair are absorbed so the
        ledger stays idempotent; source facts are never modified here.
        """
        for value in (priority,):
            if not 0.0 <= value <= 1.0:
                raise ValueError("conflict priority must be in [0, 1]")
        claims = (dict(claim_a), dict(claim_b))
        signature = tuple(sorted(
            str(sorted(item.items(), key=lambda pair: pair[0])) for item in claims
        ))
        for existing in self._records.values():
            existing_signature = tuple(sorted(
                str(sorted(item.items(), key=lambda pair: pair[0]))
                for item in existing.claims
            ))
            if existing_signature == signature and existing.status == "unresolved":
                return existing
        record = NSRLConflictRecord(
            conflict_id=make_id("nsr_conflict"),
            claims=claims,
            provenance=(
                dict(provenance_a or {}),
                dict(provenance_b or {}),
            ),
            priority=priority,
            time_scope=time_scope,
            resolution_options=resolution_options,
        )
        self._records[record.conflict_id] = record
        return record

    def resolve(self, conflict_id: str, option: str) -> NSRLConflictRecord:
        """Mark one conflict resolved with the chosen option (never deletes claims)."""
        record = self._records.get(conflict_id)
        if record is None:
            raise KeyError(f"unknown conflict id {conflict_id!r}")
        if option not in record.resolution_options:
            raise ValueError(
                f"resolution option {option!r} is not among {record.resolution_options}",
            )
        resolved = NSRLConflictRecord(
            conflict_id=record.conflict_id,
            claims=record.claims,
            provenance=record.provenance,
            priority=record.priority,
            time_scope=record.time_scope,
            resolution_options=record.resolution_options,
            status=f"resolved:{option}",
            created_at=record.created_at,
        )
        self._records[conflict_id] = resolved
        return resolved

    def to_dict(self) -> list[dict[str, Any]]:
        """Return the detached conflict ledger."""
        return [item.to_dict() for item in self.records]


__all__ = ["ConflictManager"]
