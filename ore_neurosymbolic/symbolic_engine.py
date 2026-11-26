"""Deterministic symbolic engine: forward/backward chaining with defeasible semantics.

Implements Phase B (Steps B2-B9) and the mathematical operations 6.1 and 6.2:

- strict rule firing is a logical gate: ``Fire(r, θ) = 1 iff every P_iθ is
  supported and type-valid`` (6.1) — never a probability;
- forward inference iterates the monotone fixed point ``F_{k+1} = F_k ∪ T_R(F_k)``
  until no change, goal completion, or budget exhaustion (6.2);
- defeasible rules additionally require valid scope and no overriding exception;
  explicit specificity/priority policy resolves competing defeasible rules and
  incomparable conflicts are preserved as conflict records, never silently won;
- backward chaining expands rules capable of producing the goal, memoizes
  subgoals and shares the same proof semantics;
- cycle detection tracks active goals/substitutions and returns explicit
  partial status instead of unsafe recursion.

Facts are immutable: the engine never overwrites a source fact; derived facts
carry new identities with lineage back to their premises.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import (
    EpistemicStatus,
    Literal,
    LogicalTerm,
    LogicalVariable,
    RuleClass,
    RuleDefinition,
    SemanticFact,
    make_id,
    term_key,
)
from .inference_queue import BoundedInferenceQueue, QueueEntry
from .rule_validation import select_relevant_rules
from .unification import Substitution, TypeTable, UnificationDiagnostic, unify_literals


@dataclass(frozen=True, slots=True)
class Derivation:
    """One rule application candidate produced by the engine (Step B5)."""

    conclusion_literal: Literal
    rule_ref: tuple[str, str]
    rule_class: RuleClass
    substitution: tuple[tuple[str, str], ...]
    premise_literals: tuple[Literal, ...]
    source_fact_ids: tuple[str, ...]
    defeated_by: tuple[str, ...] = ()
    conflict_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "conclusion": self.conclusion_literal.to_dict(),
            "rule_ref": [self.rule_ref[0], self.rule_ref[1]],
            "rule_class": self.rule_class.value,
            "substitution": dict(self.substitution),
            "premises": [item.to_dict() for item in self.premise_literals],
            "source_fact_ids": list(self.source_fact_ids),
            "defeated_by": list(self.defeated_by),
            "conflict_key": self.conflict_key,
        }


@dataclass(frozen=True, slots=True)
class CycleEvent:
    """One detected unsafe recursion attempt (Step B9)."""

    goal_signature: str
    depth: int
    rule_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "goal_signature": self.goal_signature,
            "depth": self.depth,
            "rule_id": self.rule_id,
        }


@dataclass(slots=True)
class EngineBudgets:
    """Hard engine bounds; every chain stops when one is exhausted (Section 2)."""

    max_rule_fires: int = 256
    max_depth: int = 8
    max_iterations: int = 32
    max_candidates: int = 4096
    wall_clock_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Reject non-positive bounds."""
        positive = (
            self.max_rule_fires, self.max_depth, self.max_iterations,
            self.max_candidates,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("engine budgets must be positive")
        if self.wall_clock_seconds <= 0:
            raise ValueError("wall_clock_seconds must be positive")

    def to_dict(self) -> dict[str, float]:
        """Return the JSON-compatible record."""
        return {
            "max_rule_fires": self.max_rule_fires,
            "max_depth": self.max_depth,
            "max_iterations": self.max_iterations,
            "max_candidates": self.max_candidates,
            "wall_clock_seconds": self.wall_clock_seconds,
        }


@dataclass(slots=True)
class EngineReport:
    """Mutable outcome record of one symbolic inference pass."""

    derived: list[Derivation] = field(default_factory=list)
    defeated: list[Derivation] = field(default_factory=list)
    conflicts: list[tuple[Derivation, Derivation]] = field(default_factory=list)
    cycle_events: list[CycleEvent] = field(default_factory=list)
    iterations: int = 0
    rule_fires: int = 0
    stop_reason: str = "not_started"
    fixed_point_reached: bool = False
    goal_satisfied: bool = False
    selected_rules: tuple[str, ...] = ()
    selection_rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "derived_count": len(self.derived),
            "defeated_count": len(self.defeated),
            "conflict_count": len(self.conflicts),
            "cycle_events": [item.to_dict() for item in self.cycle_events],
            "iterations": self.iterations,
            "rule_fires": self.rule_fires,
            "stop_reason": self.stop_reason,
            "fixed_point_reached": self.fixed_point_reached,
            "goal_satisfied": self.goal_satisfied,
            "selected_rules": list(self.selected_rules),
            "selection_rationale": self.selection_rationale,
        }


class FactTable:
    """Indexed, append-only fact store with exact keys (Steps A4, B7).

    Indexes: by subject, by predicate, by (subject, predicate). Explicitly
    negated facts are stored separately from positive facts so absence is never
    conflated with negation (Section 2: unknown is not false).
    """

    def __init__(self) -> None:
        """Create the empty append-only table and its indexes."""
        self._facts: dict[str, SemanticFact] = {}
        self._by_content: dict[str, str] = {}
        self._by_subject: dict[str, list[str]] = {}
        self._by_predicate: dict[str, list[str]] = {}
        self._by_subject_predicate: dict[tuple[str, str], list[str]] = {}

    def add(self, fact: SemanticFact) -> bool:
        """Add one fact; return False when the identical content already exists."""
        content_key = fact.fingerprint()
        if content_key in self._by_content:
            return False
        self._facts[fact.fact_id] = fact
        self._by_content[content_key] = fact.fact_id
        self._by_subject.setdefault(fact.subject, []).append(fact.fact_id)
        self._by_predicate.setdefault(fact.predicate, []).append(fact.fact_id)
        self._by_subject_predicate.setdefault(
            (fact.subject, fact.predicate), [],
        ).append(fact.fact_id)
        return True

    def get(self, fact_id: str) -> SemanticFact | None:
        """Return one fact by identity."""
        return self._facts.get(fact_id)

    def all_facts(self) -> tuple[SemanticFact, ...]:
        """Return every stored fact in insertion order."""
        return tuple(self._facts.values())

    def by_predicate(self, predicate: str) -> tuple[SemanticFact, ...]:
        """Return facts whose predicate matches (exact index first, Step A4)."""
        return tuple(self._facts.get(item) for item in self._by_predicate.get(predicate, ()))

    def by_subject(self, subject: str) -> tuple[SemanticFact, ...]:
        """Return facts whose subject matches."""
        return tuple(self._facts.get(item) for item in self._by_subject.get(subject, ()))

    def has_positive(self, literal: Literal) -> SemanticFact | None:
        """Return the supported positive fact matching a grounded literal, or None.

        A literal is *supported* only by an explicitly stored positive fact;
        the absence of a fact returns None (unknown), never negation.
        """
        if literal.negated or len(literal.args) != 2:
            return None
        first, second = literal.args
        if any(isinstance(item, LogicalVariable) for item in (first, second)):
            return None
        subject_value = first.value if isinstance(first, LogicalTerm) else ""
        matches = self._by_subject_predicate.get((subject_value, literal.predicate), ())
        for fact_id in matches:
            fact = self._facts[fact_id]
            if fact.object_value == (second.value if isinstance(second, LogicalTerm) else ""):
                if fact.polarity:
                    return fact
        return None

    def has_negated(self, literal: Literal) -> SemanticFact | None:
        """Return the stored explicitly-negated fact matching a grounded literal."""
        if len(literal.args) != 2:
            return None
        first, second = literal.args
        if any(isinstance(item, LogicalVariable) for item in (first, second)):
            return None
        subject_value = first.value if isinstance(first, LogicalTerm) else ""
        matches = self._by_subject_predicate.get((subject_value, literal.predicate), ())
        for fact_id in matches:
            fact = self._facts[fact_id]
            if fact.object_value == (second.value if isinstance(second, LogicalTerm) else ""):
                if not fact.polarity:
                    return fact
        return None

    def remove(self, fact_id: str) -> bool:
        """Remove one derived fact from the scratch table (defeat bookkeeping).

        Only in-engine derived facts are removed, never source facts; source
        evidence stays immutable for the whole run (Section 2).
        """
        fact = self._facts.get(fact_id)
        if fact is None:
            return False
        del self._facts[fact_id]
        content_key = fact.fingerprint()
        if self._by_content.get(content_key) == fact_id:
            del self._by_content[content_key]
        if fact_id in self._by_subject.get(fact.subject, []):
            self._by_subject[fact.subject].remove(fact_id)
        if fact_id in self._by_predicate.get(fact.predicate, []):
            self._by_predicate[fact.predicate].remove(fact_id)
        pair_key = (fact.subject, fact.predicate)
        if fact_id in self._by_subject_predicate.get(pair_key, []):
            self._by_subject_predicate[pair_key].remove(fact_id)
        return True

    def __len__(self) -> int:
        """Return the number of stored facts."""
        return len(self._facts)


def _ground_literal(literal: Literal, substitution: Substitution) -> Literal | None:
    """Resolve one literal under a substitution; None when variables remain."""
    resolved = substitution.resolve_literal_args(literal.args)
    if any(isinstance(item, LogicalVariable) for item in resolved):
        return None
    return Literal(
        predicate=literal.predicate,
        args=resolved,
        negated=literal.negated,
        temporal_qualifier=literal.temporal_qualifier,
    )


def _matches_predicate(literal: Literal, fact: SemanticFact) -> bool:
    """Check predicate, arity, polarity and temporal scope against one fact."""
    if literal.predicate != fact.predicate or literal.arity != 2:
        return False
    if literal.negated == fact.polarity:
        return False
    fact_temporal = fact.qualifiers.get("temporal")
    if literal.temporal_qualifier is not None and fact_temporal is not None:
        if str(fact_temporal) != literal.temporal_qualifier:
            return False
    return True


class SymbolicEngine:
    """Deterministic forward/backward chaining over typed facts and rules."""

    def __init__(
        self,
        rules: tuple[RuleDefinition, ...],
        *,
        types: TypeTable | None = None,
        budgets: EngineBudgets | None = None,
        scope_context: Mapping[str, Any] | None = None,
        max_queue_size: int = 4096,
    ) -> None:
        """Bind the validated ruleset, type table, budgets and scope context."""
        for rule in rules:
            for literal in (*rule.premises, rule.conclusion):
                if literal.arity != 2:
                    raise ValueError(
                        f"rule {rule.rule_id!r} literal {literal.predicate!r} has arity "
                        f"{literal.arity}; the triple-based fact store supports "
                        "binary literals only",
                    )
        self.rules = rules
        self.types = types or TypeTable()
        self.budgets = budgets or EngineBudgets()
        self.scope_context = dict(scope_context or {})
        self.queue = BoundedInferenceQueue(max_size=max_queue_size)
        self.facts = FactTable()
        self._rule_by_id: dict[tuple[str, str], RuleDefinition] = {
            rule.rule_ref: rule for rule in rules
        }
        self._derivation_facts: dict[tuple[str, str], str] = {}
        self._defeated_keys: set[tuple[str, str]] = set()

    # ---------------------------------------------------------------- support

    def add_fact(self, fact: SemanticFact) -> bool:
        """Insert one base fact into the table (never overwrites, Step A4)."""
        return self.facts.add(fact)

    def _literal_supported(self, literal: Literal, substitution: Substitution) -> bool:
        """Evaluate one grounded premise literal against the fact table.

        Positive premises need an explicitly supported positive fact (6.1).
        Negated premises need an explicitly stored negated fact: missing
        evidence is unknown, never negative evidence (Section 2).
        """
        grounded = _ground_literal(literal, substitution)
        if grounded is None:
            return False
        if grounded.negated:
            return self.facts.has_negated(grounded) is not None
        return self.facts.has_positive(grounded) is not None

    def _source_ids_for(self, grounded: Literal) -> tuple[str, ...]:
        """Return the fact IDs supporting one grounded literal."""
        if grounded.negated:
            fact = self.facts.has_negated(grounded)
        else:
            fact = self.facts.has_positive(grounded)
        return (fact.fact_id,) if fact is not None else ()

    def _scope_valid(self, rule: RuleDefinition) -> bool:
        """Check a defeasible rule's declared scope against the run context."""
        rule_scope = rule.scope
        if rule.rule_class is RuleClass.STRICT:
            return True
        domain = rule_scope.get("domain")
        if domain is not None and str(domain) != str(
            self.scope_context.get("domain", domain),
        ):
            return False
        return True

    def _exception_overrides(self, rule: RuleDefinition, substitution: Substitution) -> tuple[str, ...]:
        """Return exception literals that currently hold against this rule."""
        active: list[str] = []
        for exception in rule.exceptions:
            grounded = _ground_literal(exception, substitution)
            if grounded is None:
                # Exceptions with unbound variables: test existential support.
                candidate = self._exception_supported_existentially(exception)
                if candidate is not None:
                    active.append(candidate.signature())
                continue
            if grounded.negated:
                if self.facts.has_negated(grounded) is not None:
                    active.append(grounded.signature())
            elif self.facts.has_positive(grounded) is not None:
                active.append(grounded.signature())
        return tuple(active)

    def _exception_supported_existentially(self, exception: Literal) -> Literal | None:
        """Match an exception pattern with variables against known facts."""
        for fact in self.facts.by_predicate(exception.predicate):
            if len(fact.literal().args) != exception.arity:
                continue
            substitution = Substitution()
            ok, _diagnostic = unify_literals(
                exception.args, fact.literal().args, substitution, self.types,
            )
            if ok and exception.negated != fact.polarity:
                grounded = _ground_literal(exception, substitution)
                if grounded is not None:
                    return grounded
        return None

    def _evaluate_conjunction(
        self,
        rule: RuleDefinition,
        substitution: Substitution,
    ) -> tuple[bool, tuple[str, ...], UnificationDiagnostic | None]:
        """Evaluate all premises under one substitution (Step B4).

        Strict rules require every premise explicitly supported and type-valid.
        Defeasible rules additionally require valid scope and no overriding
        exception. Returns (fired, source_fact_ids, first_diagnostic).
        """
        source_ids: list[str] = []
        working = substitution.copy()
        for premise in rule.premises:
            grounded = _ground_literal(premise, working)
            if grounded is None:
                return False, (), UnificationDiagnostic(
                    code="unresolved_premise",
                    message=f"premise {premise.signature()} has unbound variables",
                    context={"rule": rule.rule_id},
                )
            if not self._literal_supported(premise, working):
                return False, (), UnificationDiagnostic(
                    code="premise_unsupported",
                    message=f"premise {grounded.signature()} is not explicitly supported",
                    context={"rule": rule.rule_id},
                )
            source_ids.extend(self._source_ids_for(grounded))
        if rule.rule_class is RuleClass.DEFEASIBLE:
            if not self._scope_valid(rule):
                return False, (), UnificationDiagnostic(
                    code="scope_invalid",
                    message=f"rule {rule.rule_id!r} scope does not cover the current context",
                    context={"rule": rule.rule_id, "scope": dict(rule.scope)},
                )
            exceptions = self._exception_overrides(rule, working)
            if exceptions:
                return False, (), UnificationDiagnostic(
                    code="exception_overrides",
                    message=f"rule {rule.rule_id!r} is defeated by exceptions {exceptions}",
                    context={"rule": rule.rule_id, "exceptions": list(exceptions)},
                )
        return True, tuple(source_ids), None

    def _candidate_conclusions(
        self,
        rule: RuleDefinition,
    ) -> list[Substitution]:
        """Enumerate complete typed joins across all premises (Steps B3-B4).

        Each premise is joined against the fact index in turn, extending the
        substitution; premise type mismatches and conflicting bindings are
        rejected during the join so only fully-grounded, type-correct
        substitutions survive (bounded diagnostics preserved on rejection).
        """
        candidates: list[Substitution] = []

        def extend(index: int, substitution: Substitution) -> None:
            if index == len(rule.premises):
                candidates.append(substitution)
                return
            premise = rule.premises[index]
            for fact in self.facts.by_predicate(premise.predicate):
                if not _matches_predicate(premise, fact):
                    continue
                trial = substitution.copy()
                ok, _diagnostic = unify_literals(
                    premise.args, fact.literal().args, trial, self.types,
                )
                if not ok:
                    continue
                extend(index + 1, trial)

        extend(0, Substitution())
        return candidates

    # ------------------------------------------------------------ forward chain

    def forward_chain(
        self,
        goal: Literal | None = None,
        report: EngineReport | None = None,
    ) -> EngineReport:
        """Run bounded forward chaining to fixed point (Steps B7, 6.2)."""
        report = report or EngineReport()
        started = time.perf_counter()
        goal_predicates = (goal.predicate,) if goal is not None else tuple(
            {rule.conclusion.predicate for rule in self.rules},
        )
        selected, rationale = select_relevant_rules(
            self.rules, goal_predicates, max_rules=len(self.rules),
        )
        report.selected_rules = tuple(rule.rule_ref[0] for rule in selected)
        report.selection_rationale = rationale

        for rule in selected:
            for substitution in self._candidate_conclusions(rule):
                self._try_fire(rule, substitution, report, depth=0, goal=goal)

        for iteration in range(self.budgets.max_iterations):
            report.iterations = iteration + 1
            if report.goal_satisfied and goal is not None:
                report.stop_reason = "goal_completion"
                break
            if not len(self.queue):
                report.fixed_point_reached = True
                report.stop_reason = "fixed_point"
                break
            if report.rule_fires >= self.budgets.max_rule_fires:
                report.stop_reason = "rule_fire_budget"
                break
            if time.perf_counter() - started >= self.budgets.wall_clock_seconds:
                report.stop_reason = "wall_clock_budget"
                break
            entry = self.queue.pop()
            if entry is None:
                report.fixed_point_reached = True
                report.stop_reason = "fixed_point"
                break
            if entry.depth + 1 > self.budgets.max_depth:
                report.cycle_events.append(CycleEvent(
                    goal_signature=entry.fact.literal().signature(),
                    depth=entry.depth + 1,
                    rule_id=entry.origin_rule_ref[0] if entry.origin_rule_ref else "*",
                ))
                continue
            for rule in selected:
                for substitution in self._candidate_conclusions(rule):
                    self._try_fire(
                        rule, substitution, report,
                        depth=entry.depth + 1, goal=goal,
                    )
        else:
            report.stop_reason = "iteration_budget"
        if report.stop_reason == "not_started":
            report.stop_reason = "no_change"
        return report

    def _try_fire(
        self,
        rule: RuleDefinition,
        substitution: Substitution,
        report: EngineReport,
        *,
        depth: int,
        goal: Literal | None,
    ) -> None:
        """Apply the firing gate 6.1 for one rule/substitution pair (Steps B4-B6).

        Order of checks: budget guards, the strict/defeasible gate, defeasible
        competition resolution (specificity/priority; Step B6), the
        known-conclusion dedup guard, then derivation recording (Step B5).
        """
        if report.rule_fires >= self.budgets.max_rule_fires:
            report.stop_reason = "rule_fire_budget"
            return
        if len(report.derived) >= self.budgets.max_candidates:
            report.stop_reason = "candidate_budget"
            return
        fired, source_ids, _diagnostic = self._evaluate_conjunction(rule, substitution)
        if not fired:
            return
        grounded = _ground_literal(rule.conclusion, substitution)
        if grounded is None:
            return
        if goal is not None and grounded.signature() == goal.signature():
            report.goal_satisfied = True

        derivation = Derivation(
            conclusion_literal=grounded,
            rule_ref=rule.rule_ref,
            rule_class=rule.rule_class,
            substitution=tuple(sorted(substitution.as_dict().items())),
            premise_literals=tuple(
                resolved
                for resolved in (
                    _ground_literal(premise, substitution) for premise in rule.premises
                )
                if resolved is not None
            ),
            source_fact_ids=tuple(dict.fromkeys(source_ids)),
        )

        if rule.rule_class is RuleClass.DEFEASIBLE:
            if (rule.rule_id, grounded.signature()) in self._defeated_keys:
                return
            outcome = self._resolve_defeasible_competition(rule, derivation, report)
            if outcome in {"loses", "conflict_duplicate"}:
                self._defeated_keys.add((rule.rule_id, grounded.signature()))
                return

        if self._conclusion_known(grounded):
            return
        report.rule_fires += 1
        report.derived.append(derivation)
        fact = self._derivation_to_fact(derivation, depth)
        # The fact table learns the conclusion immediately so later joins and
        # the known-conclusion guard see it within the same pass (Step B7);
        # the queue is the propagation frontier for the fixed-point loop.
        self.facts.add(fact)
        self._derivation_facts[(rule.rule_id, grounded.signature())] = fact.fact_id
        self.queue.push(QueueEntry(fact=fact, origin_rule_ref=rule.rule_ref, depth=depth))

    def _conclusion_known(self, grounded: Literal) -> bool:
        """Return whether the grounded conclusion is already explicitly known."""
        if grounded.negated:
            return self.facts.has_negated(grounded) is not None
        return self.facts.has_positive(grounded) is not None

    def _derivation_to_fact(self, derivation: Derivation, depth: int) -> SemanticFact:
        """Materialize one derivation as a derived SemanticFact (Step B5)."""
        args = derivation.conclusion_literal.args
        subject = args[0].value if isinstance(args[0], LogicalTerm) else ""
        object_value = args[1].value if len(args) > 1 and isinstance(args[1], LogicalTerm) else ""
        return SemanticFact(
            fact_id=make_id("nsr_derived"),
            subject=subject,
            predicate=derivation.conclusion_literal.predicate,
            object_value=object_value,
            qualifiers={
                "temporal": derivation.conclusion_literal.temporal_qualifier,
                "rule_ref": list(derivation.rule_ref),
                "depth": depth,
            },
            polarity=not derivation.conclusion_literal.negated,
            epistemic_status=EpistemicStatus.DERIVED,
            source_refs=tuple(derivation.source_fact_ids),
            usg_version=None,
        )

    def _resolve_defeasible_competition(
        self,
        rule: RuleDefinition,
        derivation: Derivation,
        report: EngineReport,
    ) -> str:
        """Resolve competing defeasible derivations over the same conclusion (Step B6).

        A more specific rule (declared ``specificity`` in scope, or more
        premises) or a higher-priority rule wins; general rules never defeat
        valid specific exceptions. Priorities/specificities that do not order
        the pair are incomparable: the competition produces a conflict record
        and both claims stay preserved instead of a forced winner.

        Returns ``"wins_through"`` when this derivation may proceed (an
        incumbent was defeated), ``"loses"`` when this derivation is defeated,
        and ``"conflict_duplicate"`` when the conclusion is already preserved
        under an incomparable competitor.
        """
        signature = derivation.conclusion_literal.signature()
        incumbents = [
            item for item in report.derived
            if item.rule_class is RuleClass.DEFEASIBLE
            and item.conclusion_literal.signature() == signature
            and item.rule_ref != derivation.rule_ref
        ]
        outcome = "wins_through"
        for incumbent in incumbents:
            incumbent_rule = self._rule_by_id.get(incumbent.rule_ref)
            if incumbent_rule is None:
                continue
            relation = self._priority_relation(rule, incumbent_rule)
            if relation == "wins":
                self._defeat_incumbent(incumbent, by_rule=rule.rule_id, report=report)
            elif relation == "loses":
                self._defeated_keys.add((rule.rule_id, signature))
                defeated = Derivation(
                    conclusion_literal=derivation.conclusion_literal,
                    rule_ref=derivation.rule_ref,
                    rule_class=derivation.rule_class,
                    substitution=derivation.substitution,
                    premise_literals=derivation.premise_literals,
                    source_fact_ids=derivation.source_fact_ids,
                    defeated_by=(incumbent_rule.rule_id,),
                )
                report.defeated.append(defeated)
                outcome = "loses"
            else:
                report.conflicts.append((derivation, incumbent))
                outcome = "conflict_duplicate"
        return outcome

    def _defeat_incumbent(
        self,
        incumbent: Derivation,
        *,
        by_rule: str,
        report: EngineReport,
    ) -> None:
        """Move one incumbent derivation to the defeated ledger and scratch table."""
        signature = incumbent.conclusion_literal.signature()
        self._defeated_keys.add((incumbent.rule_ref[0], signature))
        defeated = Derivation(
            conclusion_literal=incumbent.conclusion_literal,
            rule_ref=incumbent.rule_ref,
            rule_class=incumbent.rule_class,
            substitution=incumbent.substitution,
            premise_literals=incumbent.premise_literals,
            source_fact_ids=incumbent.source_fact_ids,
            defeated_by=(by_rule,),
        )
        report.derived = [
            item for item in report.derived if item.rule_ref != incumbent.rule_ref
        ]
        if not any(
            item.rule_ref == defeated.rule_ref
            and item.conclusion_literal.signature() == signature
            for item in report.defeated
        ):
            report.defeated.append(defeated)
        fact_id = self._derivation_facts.get((incumbent.rule_ref[0], signature))
        if fact_id is not None:
            self.facts.remove(fact_id)

    def _priority_relation(self, candidate: RuleDefinition, incumbent: RuleDefinition) -> str:
        """Compare two competing defeasible rules: wins / loses / incomparable."""
        candidate_specificity = float(candidate.scope.get("specificity", len(candidate.premises)))
        incumbent_specificity = float(incumbent.scope.get("specificity", len(incumbent.premises)))
        if candidate_specificity > incumbent_specificity:
            return "wins"
        if candidate_specificity < incumbent_specificity:
            return "loses"
        if candidate.priority > incumbent.priority:
            return "wins"
        if candidate.priority < incumbent.priority:
            return "loses"
        return "incomparable"

    # ----------------------------------------------------------- backward chain

    def backward_chain(
        self,
        goal: Literal,
        report: EngineReport | None = None,
    ) -> EngineReport:
        """Run SLD-style goal-directed backward chaining with memoization (Step B8).

        Premises resolve against facts first, then against rule conclusions,
        sharing the strict firing gate (6.1) and defeasible semantics with
        forward chaining. Grounded subgoal results are memoized; repeated
        goal/substitution states on the active resolution path are recorded as
        cycle events and the branch stops with an explicit partial outcome
        (Step B9). Negated subgoals are supported only by explicit negated
        facts: absence is never treated as failure-proves-not.
        """
        report = report or EngineReport()
        started = time.perf_counter()
        selected, rationale = select_relevant_rules(
            self.rules, (goal.predicate,), max_rules=len(self.rules),
        )
        report.selected_rules = tuple(rule.rule_ref[0] for rule in selected)
        report.selection_rationale = rationale
        memo: dict[str, bool] = {}
        active_states: list[str] = []
        proven, _sources = self._prove_conjunction(
            (goal,), Substitution(), selected, report, memo, active_states,
            depth=0, started=started,
        )
        if report.stop_reason == "not_started":
            report.stop_reason = "goal_proven" if proven else "goal_unproven"
        if proven or self._goal_known(goal):
            report.goal_satisfied = True
        return report

    def _goal_known(self, goal: Literal) -> bool:
        """Return whether the fully-grounded goal literal is explicitly supported."""
        if any(isinstance(item, LogicalVariable) for item in goal.args):
            return False
        if goal.negated:
            return self.facts.has_negated(goal) is not None
        return self.facts.has_positive(goal) is not None

    def _prove_conjunction(
        self,
        goals: tuple[Literal, ...],
        substitution: Substitution,
        rules: tuple[RuleDefinition, ...],
        report: EngineReport,
        memo: dict[str, bool],
        active_states: list[str],
        *,
        depth: int,
        started: float,
    ) -> tuple[bool, tuple[str, ...]]:
        """Resolve one goal conjunction against facts then rules (Steps B8-B9).

        Bindings are made in place on ``substitution`` and restored on branch
        failure, so a successful resolution leaves the caller's substitution
        fully extended. Grounded subgoal outcomes are memoized; repeated
        goal/substitution states on the active path stop as explicit cycles.
        """
        if not goals:
            return True, ()
        if report.rule_fires >= self.budgets.max_rule_fires:
            report.stop_reason = "rule_fire_budget"
            return False, ()
        if time.perf_counter() - started >= self.budgets.wall_clock_seconds:
            report.stop_reason = "wall_clock_budget"
            return False, ()
        if depth > self.budgets.max_depth:
            report.cycle_events.append(CycleEvent(
                goal_signature=goals[0].signature(substitution.as_dict()),
                depth=depth, rule_id="*",
            ))
            return False, ()

        premise = goals[0]
        state_key = premise.signature(substitution.as_dict())
        if state_key in active_states:
            report.cycle_events.append(CycleEvent(
                goal_signature=state_key, depth=depth, rule_id="*",
            ))
            return False, ()
        if len(substitution) == 0 and state_key in memo:
            return memo[state_key], ()

        active_states.append(state_key)
        saved_bindings = {
            name: item for name, item in substitution._bindings.items()  # noqa: SLF001
        }

        def restore() -> None:
            substitution._bindings.clear()  # noqa: SLF001
            substitution._bindings.update(saved_bindings)  # noqa: SLF001

        try:
            # 1) Direct support from explicit facts (strict gate for both polarities).
            for fact in self.facts.by_predicate(premise.predicate):
                if not _matches_predicate(premise, fact):
                    continue
                ok, _diagnostic = unify_literals(
                    premise.args, fact.literal().args, substitution, self.types,
                )
                if not ok:
                    restore()
                    continue
                grounded = self._ground(premise, substitution)
                if grounded is None:
                    restore()
                    continue
                if grounded.negated:
                    if self.facts.has_negated(grounded) is None:
                        restore()
                        continue
                elif self.facts.has_positive(grounded) is None:
                    restore()
                    continue
                proven, sources = self._prove_conjunction(
                    goals[1:], substitution, rules, report, memo, active_states,
                    depth=depth + 1, started=started,
                )
                if proven:
                    if len(saved_bindings) == 0:
                        memo[state_key] = True
                    return True, (fact.fact_id, *sources)
                restore()

            # 2) Rule expansion for positive premises only (unknown != false).
            if premise.negated:
                if len(saved_bindings) == 0:
                    memo[state_key] = False
                return False, ()
            for rule in rules:
                if rule.conclusion.predicate != premise.predicate:
                    continue
                if rule.conclusion.arity != premise.arity:
                    continue
                if rule.conclusion.negated != premise.negated:
                    continue
                if report.rule_fires >= self.budgets.max_rule_fires:
                    report.stop_reason = "rule_fire_budget"
                    break
                if time.perf_counter() - started >= self.budgets.wall_clock_seconds:
                    report.stop_reason = "wall_clock_budget"
                    break
                ok, _diagnostic = unify_literals(
                    rule.conclusion.args, premise.args, substitution, self.types,
                )
                if not ok:
                    restore()
                    continue
                proven, sources = self._prove_conjunction(
                    (*rule.premises, *goals[1:]), substitution, rules, report, memo,
                    active_states, depth=depth + 1, started=started,
                )
                if not proven:
                    restore()
                    continue
                fired, rule_source_ids, _diagnostic = self._evaluate_conjunction(rule, substitution)
                if not fired:
                    restore()
                    continue
                grounded_conclusion = self._ground(rule.conclusion, substitution)
                if grounded_conclusion is None:
                    restore()
                    continue
                if not self._conclusion_known(grounded_conclusion):
                    report.rule_fires += 1
                    derivation = Derivation(
                        conclusion_literal=grounded_conclusion,
                        rule_ref=rule.rule_ref,
                        rule_class=rule.rule_class,
                        substitution=tuple(sorted(substitution.as_dict().items())),
                        premise_literals=tuple(
                            resolved
                            for resolved in (
                                self._ground(item, substitution) for item in rule.premises
                            )
                            if resolved is not None
                        ),
                        source_fact_ids=tuple(dict.fromkeys(rule_source_ids)),
                    )
                    report.derived.append(derivation)
                    self.facts.add(self._derivation_to_fact(derivation, depth))
                if len(saved_bindings) == 0:
                    memo[state_key] = True
                return True, sources
            if len(saved_bindings) == 0:
                memo[state_key] = False
            return False, ()
        finally:
            active_states.pop()

    def _ground(self, literal: Literal, substitution: Substitution) -> Literal | None:
        """Resolve one literal under a substitution; None when variables remain."""
        resolved = substitution.resolve_literal_args(literal.args)
        if any(isinstance(item, LogicalVariable) for item in resolved):
            return None
        return Literal(
            predicate=literal.predicate,
            args=resolved,
            negated=literal.negated,
            temporal_qualifier=literal.temporal_qualifier,
        )


__all__ = [
    "CycleEvent",
    "Derivation",
    "EngineBudgets",
    "EngineReport",
    "FactTable",
    "SymbolicEngine",
]
