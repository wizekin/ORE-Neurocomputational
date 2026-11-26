"""Conflict detection and non-destructive contradiction policy (Steps 21-22).

The consistency layer checks explicit positive and negative assertions,
mutually exclusive classifications, cardinality constraints, and incompatible
values. Each conflict is classified as a logical contradiction, source
disagreement, temporal mismatch, scope mismatch, or unresolved semantic
ambiguity. The policy never deletes or overwrites facts: both sides are
preserved, the conflict is marked, and resolution is requested from the owning
subsystem with the minimal relevant evidence set attached where feasible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .contracts import EvidenceTag, ResultStatus
from .datalog import DatalogFact


class ConflictKind:
    """Conflict classification vocabulary (Step 21)."""

    LOGICAL_CONTRADICTION = "logical_contradiction"
    SOURCE_DISAGREEMENT = "source_disagreement"
    TEMPORAL_MISMATCH = "temporal_mismatch"
    SCOPE_MISMATCH = "scope_mismatch"
    UNRESOLVED_SEMANTIC_AMBIGUITY = "unresolved_semantic_ambiguity"

    ALL = (
        LOGICAL_CONTRADICTION,
        SOURCE_DISAGREEMENT,
        TEMPORAL_MISMATCH,
        SCOPE_MISMATCH,
        UNRESOLVED_SEMANTIC_AMBIGUITY,
    )


@dataclass(frozen=True, slots=True)
class ExclusivityGroup:
    """A declared set of mutually exclusive classifications (Step 21)."""

    group_id: str
    predicate: str
    exclusive_values: tuple[str, ...]
    position: int = 0

    def __post_init__(self) -> None:
        if not self.group_id.strip():
            raise ValueError("ExclusivityGroup.group_id must be non-empty")
        if len(self.exclusive_values) < 2:
            raise ValueError("an exclusivity group needs at least two values")


@dataclass(frozen=True, slots=True)
class FunctionalConstraint:
    """A declared functional (single-valued) property (Step 21)."""

    constraint_id: str
    predicate: str
    position: int = 1

    def __post_init__(self) -> None:
        if not self.constraint_id.strip():
            raise ValueError("FunctionalConstraint.constraint_id must be non-empty")


@dataclass(frozen=True, slots=True)
class CardinalityConstraint:
    """An exactly-one / at-most-one cardinality constraint (Step 21)."""

    constraint_id: str
    predicate: str
    max_count: int
    key_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.constraint_id.strip():
            raise ValueError("CardinalityConstraint.constraint_id must be non-empty")
        if self.max_count < 1:
            raise ValueError("cardinality max_count must be >= 1")


@dataclass(frozen=True, slots=True)
class ConflictRecord:
    """One detected conflict with both sides preserved (Step 22)."""

    conflict_id: str
    kind: str
    description: str
    side_a: tuple[str, ...]                 # fact IDs on one side
    side_b: tuple[str, ...]                 # fact IDs on the other side
    minimal_evidence: tuple[str, ...] = ()  # minimal relevant evidence set
    resolution_request: str = ""            # requested owning subsystem action
    fact_keys: tuple[tuple[Any, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ConflictKind.ALL:
            raise ValueError(f"unknown conflict kind: {self.kind!r}")
        if not self.side_a and not self.side_b:
            raise ValueError("a conflict record needs at least one evidenced side")


@dataclass(frozen=True, slots=True)
class ConsistencyReport:
    """The outcome of the consistency pass (Steps 21-22)."""

    checked_facts: int
    conflicts: tuple[ConflictRecord, ...]
    preserved_sides: int
    destructive_repairs: int = 0            # always zero by construction
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "conflicts", tuple(self.conflicts))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def by_kind(self, kind: str) -> tuple[ConflictRecord, ...]:
        """Return conflicts of one kind."""
        return tuple(c for c in self.conflicts if c.kind == kind)


def _classify_pair_conflict(
    fact_a: DatalogFact,
    fact_b: DatalogFact,
) -> str:
    """Classify the conflict between two opposing assertions (Step 21)."""
    scopes_differ = fact_a.scope != fact_b.scope
    if scopes_differ:
        return ConflictKind.SCOPE_MISMATCH
    intervals = (fact_a.interval, fact_b.interval)
    if intervals[0] is not None and intervals[1] is not None:
        if not fact_a.interval_overlaps(fact_b.interval):
            return ConflictKind.TEMPORAL_MISMATCH
    # Source identity falls back to the fact id when no statement ids are kept.
    sources_a = set(fact_a.source_statement_ids) or {fact_a.fact_id}
    sources_b = set(fact_b.source_statement_ids) or {fact_b.fact_id}
    if fact_a.tag == EvidenceTag.OBSERVED_FACT and fact_b.tag == EvidenceTag.OBSERVED_FACT:
        if sources_a and sources_b and not (sources_a & sources_b):
            return ConflictKind.SOURCE_DISAGREEMENT
    return ConflictKind.LOGICAL_CONTRADICTION


class ConsistencyChecker:
    """Detect conflicting conclusions without destructive repair (Steps 21-22)."""

    def __init__(
        self,
        *,
        exclusivity_groups: Iterable[ExclusivityGroup] = (),
        functional_constraints: Iterable[FunctionalConstraint] = (),
        cardinality_constraints: Iterable[CardinalityConstraint] = (),
    ) -> None:
        self._exclusivity = tuple(exclusivity_groups)
        self._functional = tuple(functional_constraints)
        self._cardinality = tuple(cardinality_constraints)

    def check(
        self,
        facts: Sequence[DatalogFact],
        *,
        ambiguous_references: Mapping[str, int] | None = None,
    ) -> ConsistencyReport:
        """Run every conflict check over the fact set (Step 21)."""
        conflicts: list[ConflictRecord] = []
        checked = len(facts)
        counter = 0

        positives = [f for f in facts if not f.negated]
        negatives = [f for f in facts if f.negated]

        # 1. explicit positive vs explicit negative assertions on the same atom.
        for pos in positives:
            for neg in negatives:
                if (pos.predicate, pos.args) == (neg.predicate, neg.args):
                    kind = _classify_pair_conflict(pos, neg)
                    counter += 1
                    conflicts.append(ConflictRecord(
                        conflict_id=f"conflict-{counter:04d}",
                        kind=kind,
                        description=(
                            f"explicit positive and negative assertions of "
                            f"{pos.predicate}({', '.join(pos.args)}); both sides preserved"
                        ),
                        side_a=(pos.fact_id,),
                        side_b=(neg.fact_id,),
                        minimal_evidence=tuple(sorted({pos.fact_id, neg.fact_id})),
                        resolution_request=(
                            "request authority/temporal resolution from the owning subsystem"
                        ),
                        fact_keys=(pos.key, neg.key),
                    ))

        # 2. mutually exclusive classifications.
        for group in self._exclusivity:
            by_key: dict[tuple[str, ...], list[DatalogFact]] = {}
            for fact in positives:
                if fact.predicate != group.predicate:
                    continue
                if group.position >= len(fact.args):
                    continue
                subject = tuple(fact.args[:group.position] + fact.args[group.position + 1:])
                by_key.setdefault(subject, []).append(fact)
            for subject, group_facts in sorted(by_key.items()):
                values = [f.args[group.position] for f in group_facts]
                exclusive_present = [
                    v for v in values if v in group.exclusive_values
                ]
                if len(set(exclusive_present)) > 1:
                    counter += 1
                    conflicts.append(ConflictRecord(
                        conflict_id=f"conflict-{counter:04d}",
                        kind=ConflictKind.LOGICAL_CONTRADICTION,
                        description=(
                            f"mutually exclusive classifications "
                            f"{sorted(set(exclusive_present))} both asserted for "
                            f"{group.predicate}{subject}"
                        ),
                        side_a=(group_facts[0].fact_id,),
                        side_b=tuple(f.fact_id for f in group_facts[1:]),
                        minimal_evidence=tuple(
                            sorted(f.fact_id for f in group_facts
                                   if f.args[group.position] in set(exclusive_present))[:2]
                        ),
                        resolution_request="classify the object under one exclusive value",
                        fact_keys=tuple(f.key for f in group_facts),
                    ))

        # 3. functional (single-valued) properties with incompatible values.
        for constraint in self._functional:
            by_key: dict[tuple[str, ...], list[DatalogFact]] = {}
            for fact in positives:
                if fact.predicate != constraint.predicate:
                    continue
                if constraint.position >= len(fact.args):
                    continue
                subject = tuple(
                    fact.args[:constraint.position] + fact.args[constraint.position + 1:]
                )
                by_key.setdefault(subject, []).append(fact)
            for subject, group_facts in sorted(by_key.items()):
                values = {f.args[constraint.position] for f in group_facts}
                if len(values) > 1:
                    counter += 1
                    conflicts.append(ConflictRecord(
                        conflict_id=f"conflict-{counter:04d}",
                        kind=ConflictKind.SOURCE_DISAGREEMENT,
                        description=(
                            f"functional property {constraint.predicate}{subject} asserted "
                            f"with incompatible values {sorted(values)}"
                        ),
                        side_a=(group_facts[0].fact_id,),
                        side_b=(group_facts[1].fact_id,),
                        minimal_evidence=tuple(sorted(
                            (group_facts[0].fact_id, group_facts[1].fact_id)
                        )),
                        resolution_request=(
                            "resolve the functional value via authority or temporal precedence"
                        ),
                        fact_keys=tuple(f.key for f in group_facts),
                    ))

        # 4. cardinality constraints (exactly-one / at-most-one).
        for constraint in self._cardinality:
            by_key: dict[tuple[str, ...], list[DatalogFact]] = {}
            for fact in positives:
                if fact.predicate != constraint.predicate:
                    continue
                key = tuple(
                    fact.args[position] for position in constraint.key_positions
                    if position < len(fact.args)
                )
                by_key.setdefault(key, []).append(fact)
            for key, group_facts in sorted(by_key.items()):
                if len(group_facts) > constraint.max_count:
                    counter += 1
                    conflicts.append(ConflictRecord(
                        conflict_id=f"conflict-{counter:04d}",
                        kind=ConflictKind.LOGICAL_CONTRADICTION,
                        description=(
                            f"cardinality violated: {constraint.predicate} has "
                            f"{len(group_facts)} facts for key {key} but allows "
                            f"{constraint.max_count}"
                        ),
                        side_a=(group_facts[0].fact_id,),
                        side_b=tuple(f.fact_id for f in group_facts[1:constraint.max_count + 1]),
                        minimal_evidence=tuple(
                            sorted(f.fact_id for f in group_facts)[:constraint.max_count + 1]
                        ),
                        resolution_request="reduce to the permitted cardinality",
                        fact_keys=tuple(f.key for f in group_facts),
                    ))

        # 5. unresolved semantic ambiguity from entity resolution (Step 3).
        if ambiguous_references:
            for reference, candidate_count in sorted(ambiguous_references.items()):
                if candidate_count > 1:
                    counter += 1
                    conflicts.append(ConflictRecord(
                        conflict_id=f"conflict-{counter:04d}",
                        kind=ConflictKind.UNRESOLVED_SEMANTIC_AMBIGUITY,
                        description=(
                            f"reference {reference!r} resolved to {candidate_count} "
                            "USG candidates without a discriminator"
                        ),
                        side_a=(f"reference:{reference}",),
                        side_b=(),
                        resolution_request=(
                            "supply a record-id discriminator or clarification"
                        ),
                    ))

        return ConsistencyReport(
            checked_facts=checked,
            conflicts=tuple(conflicts),
            preserved_sides=sum(len(c.side_a) + len(c.side_b) for c in conflicts),
        )

    def check_temporal_non_conflict(
        self, fact_a: DatalogFact, fact_b: DatalogFact,
    ) -> bool:
        """Return True when two opposing facts are merely time-separated (Step 6).

        Opposing assertions with non-overlapping validity intervals are not
        simultaneous contradictions; the checker classifies them as temporal
        mismatches instead and never merges the facts across time.
        """
        if (fact_a.predicate, fact_a.args) != (fact_b.predicate, fact_b.args):
            return False
        return not fact_a.interval_overlaps(fact_b.interval)


def result_from_report(report: ConsistencyReport) -> Any:
    """Map a consistency report into a typed FRCE result status.

    Conflicts are reported, never resolved silently: the status is
    ``satisfiable`` when no conflicts exist and ``unknown`` otherwise, because
    a detected conflict does not by itself establish falsity of either side.
    """
    if not report.conflicts:
        return ResultStatus.SATISFIABLE
    return ResultStatus.UNKNOWN


__all__ = [
    "ConflictKind",
    "ExclusivityGroup",
    "FunctionalConstraint",
    "CardinalityConstraint",
    "ConflictRecord",
    "ConsistencyReport",
    "ConsistencyChecker",
    "result_from_report",
]
