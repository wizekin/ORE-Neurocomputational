"""Datalog rule evaluation with derivation provenance (Phase C, Steps 16-20).

The engine evaluates finite, range-restricted, stratified Datalog programs over
typed facts extracted from the normalized IR. It implements:

- a dependency graph with strongly connected components and stratification
  checks (Step 17): negation through recursion is rejected;
- indexed, semi-naive fixed-point evaluation with delta iteration (Step 19),
  plus a naive reference evaluator used to cross-check logical equivalence;
- an explanation graph that records, for every derived fact, the exact rule,
  substitution, and parent facts, preserving multiple distinct derivations
  (Step 20);
- temporal validity intervals and context scopes carried through inference so
  conflicting time-scoped facts are never treated as simultaneous (Step 6);
- explicit iteration and fact limits for unsafe extensions, and a content-
  addressed cache invalidated by any change to rules, facts, or engine version.

Negation in rule bodies is stratified negation-as-absence over lower strata:
``not p(X, Y)`` is satisfied when no positive ``p(X, Y)`` fact exists in the
current evaluation state. Explicitly asserted negative facts are a separate
concept handled by the consistency layer, never silently reused as negation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .contracts import EvidenceTag, sha256_json
from .exceptions import NonStratifiableProgramError, UnsafeRuleError
from .ir import (
    Atom,
    BodyLiteral,
    BoolLit,
    Const,
    FormalIR,
    NumLit,
    RuleDecl,
    StatementKind,
    StrLit,
    Var,
)

DATALOG_ENGINE_VERSION = "frce-datalog-1.0.0"

_UNBOUNDED_LOW = -(2**62)
_UNBOUNDED_HIGH = 2**62


# ---------------------------------------------------------------------------
# Facts, derivations, results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatalogFact:
    """One typed fact with scope, validity interval, and provenance (Steps 5, 6)."""

    predicate: str
    args: tuple[str, ...]
    fact_id: str
    negated: bool = False
    scope: str = "default"
    interval: tuple[int | None, int | None] | None = None
    source_statement_ids: tuple[str, ...] = ()
    tag: EvidenceTag = EvidenceTag.OBSERVED_FACT

    def __post_init__(self) -> None:
        if not self.predicate.strip():
            raise ValueError("DatalogFact.predicate must be non-empty")
        if not self.fact_id.strip():
            raise ValueError("DatalogFact.fact_id must be non-empty")
        if self.interval is not None:
            low, high = self.interval
            if low is not None and high is not None and low > high:
                raise ValueError("fact validity interval requires from <= until")

    @property
    def key(self) -> tuple[Any, ...]:
        """Return the logical key used for joins and contradiction checks."""
        return (self.predicate, self.args, self.negated, self.scope)

    def interval_overlaps(self, other: tuple[int | None, int | None] | None) -> bool:
        """Return whether this fact's validity interval overlaps another's."""
        if self.interval is None or other is None:
            return True
        low = max(
            self.interval[0] if self.interval[0] is not None else _UNBOUNDED_LOW,
            other[0] if other[0] is not None else _UNBOUNDED_LOW,
        )
        high = min(
            self.interval[1] if self.interval[1] is not None else _UNBOUNDED_HIGH,
            other[1] if other[1] is not None else _UNBOUNDED_HIGH,
        )
        return low <= high

    def intersect_interval(
        self, other: tuple[int | None, int | None] | None,
    ) -> tuple[int | None, int | None] | None:
        """Return the intersection interval, or None when it is empty."""
        if self.interval is None:
            return other
        if other is None:
            return self.interval
        low = max(
            self.interval[0] if self.interval[0] is not None else _UNBOUNDED_LOW,
            other[0] if other[0] is not None else _UNBOUNDED_LOW,
        )
        high = min(
            self.interval[1] if self.interval[1] is not None else _UNBOUNDED_HIGH,
            other[1] if other[1] is not None else _UNBOUNDED_HIGH,
        )
        if low > high:
            return None
        return (
            None if low == _UNBOUNDED_LOW else low,
            None if high == _UNBOUNDED_HIGH else high,
        )


@dataclass(frozen=True, slots=True)
class Derivation:
    """One derivation path of a derived fact with its exact provenance (Step 20)."""

    rule_id: str
    substitution: Mapping[str, str]
    parent_fact_ids: tuple[str, ...]
    scope: str = "default"
    interval: tuple[int | None, int | None] | None = None


@dataclass(frozen=True, slots=True)
class DerivedFact:
    """A derived fact with every distinct derivation preserved (Step 20)."""

    predicate: str
    args: tuple[str, ...]
    scope: str
    interval: tuple[int | None, int | None] | None
    derivations: tuple[Derivation, ...]
    fact_id: str

    def __post_init__(self) -> None:
        if not self.derivations:
            raise ValueError("a derived fact requires at least one derivation")

    @property
    def key(self) -> tuple[Any, ...]:
        """Return the logical key."""
        return (self.predicate, self.args, False, self.scope)


@dataclass(frozen=True, slots=True)
class DatalogLimits:
    """Explicit limits bounding rule evaluation (Step 19)."""

    max_iterations: int = 1000
    max_derived_facts: int = 20000


@dataclass(frozen=True, slots=True)
class DatalogResult:
    """The fixed-point result with full provenance and operational metadata."""

    input_facts: tuple[DatalogFact, ...]
    derived_facts: tuple[DerivedFact, ...]
    iterations: int
    fixed_point_reached: bool
    strata: tuple[tuple[str, ...], ...]
    engine_version: str = DATALOG_ENGINE_VERSION
    program_digest: str = ""
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_facts", tuple(self.input_facts))
        object.__setattr__(self, "derived_facts", tuple(self.derived_facts))
        object.__setattr__(self, "strata", tuple(self.strata))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def derived_by_predicate(self, predicate: str) -> tuple[DerivedFact, ...]:
        """Return derived facts for one predicate."""
        return tuple(f for f in self.derived_facts if f.predicate == predicate)

    def all_positive_keys(self) -> set[tuple[Any, ...]]:
        """Return all positive fact keys (input plus derived) across scopes."""
        keys = {
            (f.predicate, f.args, False, f.scope)
            for f in self.input_facts if not f.negated
        }
        keys |= {f.key for f in self.derived_facts}
        return keys


# ---------------------------------------------------------------------------
# Compilation from the IR
# ---------------------------------------------------------------------------


def _term_to_symbol(term: Any) -> str:
    """Canonical symbolic form of a ground term or bound variable name."""
    if isinstance(term, Const):
        return term.name
    if isinstance(term, Var):
        return term.name
    if isinstance(term, NumLit):
        return str(term.value)
    if isinstance(term, StrLit):
        return term.value
    if isinstance(term, BoolLit):
        return "true" if term.value else "false"
    raise UnsafeRuleError(
        f"unbounded rule construct in Datalog term: {type(term).__name__} "
        "(function symbols and arithmetic are outside the Datalog subset)"
    )


def facts_from_ir(ir: FormalIR) -> tuple[DatalogFact, ...]:
    """Extract typed facts from IR statements (Step 8 mapping into the engine)."""
    facts: list[DatalogFact] = []
    for statement in ir.statements:
        if statement.kind != StatementKind.FACT:
            continue
        formula = statement.formula
        negated = False
        if formula.__class__.__name__ == "NotF":
            negated = True
            formula = formula.body
        if formula.__class__.__name__ != "Atom":
            continue
        assert isinstance(formula, Atom)
        args = tuple(_term_to_symbol(t) for t in formula.args)
        interval = (
            statement.temporal_scope.as_tuple() if statement.temporal_scope else None
        )
        facts.append(DatalogFact(
            predicate=formula.predicate,
            args=args,
            fact_id=statement.statement_id,
            negated=negated,
            scope=statement.context_scope,
            interval=interval,
            source_statement_ids=(statement.statement_id,),
            tag=statement.tag,
        ))
    return tuple(facts)


def rules_from_ir(ir: FormalIR) -> tuple[RuleDecl, ...]:
    """Return the IR rules usable by the Datalog engine, rejecting unsafe forms."""
    usable: list[RuleDecl] = []
    for rule in ir.rules:
        for literal in (*rule.body, BodyLiteral(rule.head)):
            for term in literal.atom.args:
                _term_to_symbol(term)  # rejects function symbols / arithmetic
        positive_vars: set[str] = set()
        for literal in rule.body:
            if not literal.negated:
                for term in literal.atom.args:
                    if isinstance(term, Var):
                        positive_vars.add(term.name)
        for literal in rule.body:
            if literal.negated:
                missing = [
                    term.name for term in literal.atom.args
                    if isinstance(term, Var) and term.name not in positive_vars
                ]
                if missing:
                    raise UnsafeRuleError(
                        f"rule {rule.rule_id!r}: negated literal binds variables "
                        f"{missing} that never occur positively (unsafe negation)"
                    )
        usable.append(rule)
    return tuple(usable)


# ---------------------------------------------------------------------------
# Dependency graph and stratification (Step 17)
# ---------------------------------------------------------------------------


def build_dependency_graph(
    rules: Sequence[RuleDecl],
) -> tuple[dict[str, set[tuple[str, bool]]], list[tuple[str, ...]]]:
    """Build the predicate dependency graph and compute strata (Step 17).

    Returns an adjacency mapping ``predicate -> {(dependency, negated)}`` and
    the computed strata (predicate groups in evaluation order). A negative
    dependency inside one strongly connected component makes the program
    non-stratifiable and raises ``NonStratifiableProgramError``.
    """
    adjacency: dict[str, set[tuple[str, bool]]] = {}
    for rule in rules:
        head = rule.head.predicate
        for literal in rule.body:
            dependency = literal.atom.predicate
            adjacency.setdefault(dependency, set()).add((head, literal.negated))
            adjacency.setdefault(head, set())

    # Tarjan strongly connected components over the predicate graph.
    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    sccs: list[list[str]] = []

    def strongconnect(node: str) -> None:
        index[node] = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack.add(node)
        for neighbor, _negated in sorted(adjacency.get(node, set())):
            if neighbor not in index:
                strongconnect(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node] = min(lowlink[node], index[neighbor])
        if lowlink[node] == index[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == node:
                    break
            sccs.append(component)

    for node in sorted(adjacency):
        if node not in index:
            strongconnect(node)

    comp_id: dict[str, int] = {}
    for position, component in enumerate(sccs):
        for member in component:
            comp_id[member] = position
    comp_edges: dict[int, set[tuple[int, bool]]] = {
        position: set() for position in range(len(sccs))
    }
    for source, edges in adjacency.items():
        for target, negated in edges:
            a, b = comp_id[source], comp_id[target]
            if a != b:
                comp_edges[a].add((b, negated))

    indegree = {position: 0 for position in comp_edges}
    for _source, edges in comp_edges.items():
        for target, _negated in edges:
            indegree[target] += 1
    order: list[int] = []
    ready = sorted(p for p, d in indegree.items() if d == 0)
    while ready:
        position = ready.pop(0)
        order.append(position)
        for target, _negated in sorted(comp_edges[position]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
        ready.sort()
    if len(order) != len(sccs):
        raise NonStratifiableProgramError(
            "rule dependency graph contains a cycle that prevents stratification"
        )

    for rule in rules:
        head_component = comp_id[rule.head.predicate]
        for literal in rule.body:
            if literal.negated and comp_id[literal.atom.predicate] == head_component:
                raise NonStratifiableProgramError(
                    f"negation through recursion: rule {rule.rule_id!r} negates "
                    f"{literal.atom.predicate!r} inside its own stratum"
                )

    strata = tuple(tuple(sorted(sccs[position])) for position in order)
    return adjacency, strata


# ---------------------------------------------------------------------------
# Indexed fact store
# ---------------------------------------------------------------------------


class _FactIndex:
    """Indexed store of known facts keyed by predicate for efficient joins."""

    def __init__(self) -> None:
        self._by_predicate: dict[str, dict[tuple[Any, ...], list[DatalogFact]]] = {}

    def add(self, fact: DatalogFact) -> bool:
        """Add one fact; returns False when an identical fact already exists."""
        bucket = self._by_predicate.setdefault(fact.predicate, {})
        key = (fact.args, fact.negated, fact.scope)
        existing = bucket.get(key)
        if existing is None:
            bucket[key] = [fact]
            return True
        for candidate in existing:
            if candidate.interval == fact.interval:
                return False
        existing.append(fact)
        return True

    def matches(
        self,
        predicate: str,
        args: tuple[str, ...],
        *,
        negated: bool = False,
        scope: str | None = None,
    ) -> list[DatalogFact]:
        """Return facts matching predicate and constant args (scope-filtered).

        Pattern positions beginning with ``?`` bind to any constant.
        """
        bucket = self._by_predicate.get(predicate, {})
        results: list[DatalogFact] = []
        for key, facts in bucket.items():
            fact_args, fact_negated, fact_scope = key
            if fact_negated != negated:
                continue
            if scope is not None and fact_scope != scope:
                continue
            if len(args) == len(fact_args) and all(
                a == f or a.startswith("?") for a, f in zip(args, fact_args)
            ):
                results.extend(facts)
        return results

    def has_positive(
        self,
        predicate: str,
        args: tuple[str, ...],
        scope: str | None,
        interval: tuple[int | None, int | None] | None,
    ) -> bool:
        """Stratified absence check: any overlapping positive fact blocks negation."""
        for fact in self.matches(predicate, args, negated=False, scope=scope):
            if fact.interval_overlaps(interval):
                return True
        return False

    def all_facts(self) -> list[DatalogFact]:
        """Return every stored fact."""
        out: list[DatalogFact] = []
        for bucket in self._by_predicate.values():
            for facts in bucket.values():
                out.extend(facts)
        return out


def _substitution_from_match(
    rule_args: Sequence[Any], fact_args: Sequence[str],
) -> dict[str, str] | None:
    """Compute the substitution binding rule variables to fact constants."""
    binding: dict[str, str] = {}
    for pattern, value in zip(rule_args, fact_args):
        if isinstance(pattern, Var):
            if pattern.name in binding and binding[pattern.name] != value:
                return None
            binding[pattern.name] = value
        else:
            if _term_to_symbol(pattern) != value:
                return None
    return binding


def _substitute(atom: Atom, binding: Mapping[str, str]) -> tuple[str, ...]:
    """Apply a substitution to an atom's arguments."""
    out: list[str] = []
    for term in atom.args:
        if isinstance(term, Var):
            value = binding.get(term.name)
            if value is None:
                raise UnsafeRuleError(
                    f"negated literal argument {term.name!r} is unbound under the "
                    "current substitution; unsafe negation is rejected"
                )
            out.append(value)
        else:
            out.append(_term_to_symbol(term))
    return tuple(out)


# ---------------------------------------------------------------------------
# The engine (Steps 16, 18, 19, 20)
# ---------------------------------------------------------------------------


class DatalogEngine:
    """Evaluate stratified Datalog programs with full provenance (Steps 16-20)."""

    def __init__(
        self,
        facts: Iterable[DatalogFact],
        rules: Iterable[RuleDecl],
        *,
        limits: DatalogLimits | None = None,
    ) -> None:
        self._input_facts = tuple(facts)
        self._rules = tuple(rules)
        self._limits = limits or DatalogLimits()
        self._adjacency, self._strata = build_dependency_graph(self._rules)
        self._derived_seen: dict[tuple[Any, ...], DerivedFact] = {}
        self.program_digest = sha256_json({
            "engine": DATALOG_ENGINE_VERSION,
            "facts": [
                [f.predicate, list(f.args), f.negated, f.scope,
                 list(f.interval) if f.interval else None, f.fact_id]
                for f in sorted(self._input_facts, key=lambda x: x.fact_id)
            ],
            "rules": [
                {
                    "rule_id": r.rule_id, "version": r.version,
                    "head": [r.head.predicate, [_term_to_symbol(t) for t in r.head.args]],
                    "body": [
                        [b.atom.predicate, [_term_to_symbol(t) for t in b.atom.args], b.negated]
                        for b in r.body
                    ],
                }
                for r in sorted(self._rules, key=lambda x: x.rule_id)
            ],
        })

    @property
    def strata(self) -> tuple[tuple[str, ...], ...]:
        """Return the computed stratification order."""
        return self._strata

    def _rules_by_head(self) -> dict[str, list[RuleDecl]]:
        grouped: dict[str, list[RuleDecl]] = {}
        for rule in self._rules:
            grouped.setdefault(rule.head.predicate, []).append(rule)
        return grouped

    def _match_positive(
        self,
        literal: BodyLiteral,
        index: _FactIndex,
        scope: str | None,
        interval: tuple[int | None, int | None] | None,
    ) -> list[tuple[DatalogFact, dict[str, str]]]:
        """Return (fact, binding) matches for one positive literal."""
        args_pattern = tuple(
            "?" + term.name if isinstance(term, Var) else _term_to_symbol(term)
            for term in literal.atom.args
        )
        matches: list[tuple[DatalogFact, dict[str, str]]] = []
        for fact in index.matches(
            literal.atom.predicate, args_pattern, negated=False, scope=scope,
        ):
            binding = _substitution_from_match(literal.atom.args, fact.args)
            if binding is None:
                continue
            if not fact.interval_overlaps(interval):
                continue
            matches.append((fact, binding))
        return matches

    def _emit(
        self,
        rule: RuleDecl,
        facts_used: list[DatalogFact],
        binding: Mapping[str, str],
        accept: Any,
    ) -> bool:
        """Assemble and accept one derived fact from a completed join."""
        scopes = {f.scope for f in facts_used}
        if len(scopes) != 1:
            return False  # never join across context scopes
        fact_scope = scopes.pop()
        combined_interval: tuple[int | None, int | None] | None = None
        empty = False
        for f in facts_used:
            if f.interval is None:
                continue
            if combined_interval is None:
                combined_interval = f.interval
                continue
            low = max(
                combined_interval[0] if combined_interval[0] is not None else _UNBOUNDED_LOW,
                f.interval[0] if f.interval[0] is not None else _UNBOUNDED_LOW,
            )
            high = min(
                combined_interval[1] if combined_interval[1] is not None else _UNBOUNDED_HIGH,
                f.interval[1] if f.interval[1] is not None else _UNBOUNDED_HIGH,
            )
            if low > high:
                empty = True
                break
            combined_interval = (
                None if low == _UNBOUNDED_LOW else low,
                None if high == _UNBOUNDED_HIGH else high,
            )
        if empty:
            return False  # premise validity times never co-occur
        head_args = _substitute(rule.head, binding)
        new_fact = DerivedFact(
            predicate=rule.head.predicate,
            args=head_args,
            scope=fact_scope,
            interval=combined_interval,
            derivations=(Derivation(
                rule_id=rule.rule_id,
                substitution=dict(sorted(binding.items())),
                parent_fact_ids=tuple(f.fact_id for f in facts_used),
                scope=fact_scope,
                interval=combined_interval,
            ),),
            fact_id=(
                f"{rule.rule_id}:{rule.head.predicate}({','.join(head_args)})@{fact_scope}"
            ),
        )
        return accept(new_fact)

    def _join_and_check_negation(
        self,
        rule: RuleDecl,
        groups: list[list[tuple[DatalogFact, dict[str, str]]]],
        index: _FactIndex,
        accept: Any,
    ) -> int:
        """Join positive literal matches and enforce negated literals (Step 18)."""
        derived = 0
        combos: list[tuple[list[DatalogFact], dict[str, str]]] = [
            ([fact], dict(binding)) for fact, binding in groups[0]
        ]
        for group in groups[1:]:
            nxt: list[tuple[list[DatalogFact], dict[str, str]]] = []
            for facts_so_far, binding in combos:
                for fact, new_binding in group:
                    merged = dict(binding)
                    conflict = False
                    for var, value in new_binding.items():
                        if var in merged and merged[var] != value:
                            conflict = True
                            break
                        merged[var] = value
                    if conflict:
                        continue
                    nxt.append((facts_so_far + [fact], merged))
            combos = nxt
            if not combos:
                return 0
        for facts_used, binding in combos:
            interval: tuple[int | None, int | None] | None = None
            for f in facts_used:
                interval = f.intersect_interval(interval)
            scope = facts_used[0].scope if facts_used else None
            blocked = False
            for literal in rule.body:
                if not literal.negated:
                    continue
                neg_args = _substitute(literal.atom, binding)
                if index.has_positive(literal.atom.predicate, neg_args, scope, interval):
                    blocked = True
                    break
            if blocked:
                continue
            if self._emit(rule, facts_used, binding, accept):
                derived += 1
        return derived

    def _fire_rule(
        self,
        rule: RuleDecl,
        index: _FactIndex,
        accept: Any,
    ) -> int:
        """Fire one rule: evaluate all positive literals, then check negation."""
        groups: list[list[tuple[DatalogFact, dict[str, str]]]] = []
        interval: tuple[int | None, int | None] | None = None
        for literal in rule.body:
            if literal.negated:
                continue
            matches = self._match_positive(literal, index, None, interval)
            if not matches:
                return 0
            groups.append(matches)
        if not groups:
            return 0
        return self._join_and_check_negation(rule, groups, index, accept)

    def _accept_factory(self, newly_derived: list[DerivedFact]) -> Any:
        """Create an accept callback merging provenance for duplicate facts."""
        def accept(fact: DerivedFact) -> bool:
            key = (fact.predicate, fact.args, fact.scope, fact.interval)
            existing = self._derived_seen.get(key)
            if existing is not None:
                merged = DerivedFact(
                    predicate=existing.predicate, args=existing.args,
                    scope=existing.scope, interval=existing.interval,
                    derivations=existing.derivations + fact.derivations,
                    fact_id=existing.fact_id,
                )
                self._derived_seen[key] = merged
                return False
            self._derived_seen[key] = fact
            newly_derived.append(fact)
            return True
        return accept

    def _check_limits(self) -> None:
        if len(self._derived_seen) > self._limits.max_derived_facts:
            raise UnsafeRuleError(
                "derived-fact limit exceeded; refusing unbounded extension"
            )

    def _index_fact_for_delta(self, fact: DerivedFact) -> DatalogFact:
        return DatalogFact(
            predicate=fact.predicate, args=fact.args,
            fact_id=fact.fact_id, negated=False, scope=fact.scope,
            interval=fact.interval,
            source_statement_ids=(fact.fact_id,),
            tag=EvidenceTag.INFERRED_RESULT,
        )

    def evaluate_semi_naive(self) -> DatalogResult:
        """Semi-naive fixed-point evaluation with delta iteration (Step 19).

        Within each stratum, only rule firings that involve at least one newly
        derived (delta) fact are re-evaluated; unchanged tuples are never
        rejoined. The fixed point is reached when a delta round derives nothing.
        """
        index = _FactIndex()
        for fact in self._input_facts:
            index.add(fact)
        self._derived_seen = {}
        rules_by_head = self._rules_by_head()
        iterations = 0

        for stratum_predicates in self._strata:
            stratum_rules = [
                rule for predicate in stratum_predicates
                for rule in rules_by_head.get(predicate, [])
            ]
            if not stratum_rules:
                continue
            # First round of a stratum re-evaluates against every known fact
            # (lower strata plus input facts); later rounds use only deltas.
            delta_facts: list[DatalogFact] = list(index.all_facts())
            while True:
                iterations += 1
                if iterations > self._limits.max_iterations:
                    raise UnsafeRuleError(
                        "iteration limit exceeded; refusing unbounded recursion"
                    )
                newly_derived: list[DerivedFact] = []
                accept = self._accept_factory(newly_derived)
                for rule in stratum_rules:
                    positive_predicates = {
                        lit.atom.predicate for lit in rule.body if not lit.negated
                    }
                    for delta_fact in delta_facts:
                        if delta_fact.predicate not in positive_predicates:
                            continue
                        binding = _substitution_from_match(
                            next(
                                lit.atom.args for lit in rule.body
                                if not lit.negated
                                and lit.atom.predicate == delta_fact.predicate
                            ),
                            delta_fact.args,
                        )
                        if binding is None:
                            continue
                        self._fire_rule_through_binding(
                            rule, index, delta_fact.predicate, binding, accept,
                        )
                self._check_limits()
                if not newly_derived:
                    break
                for fact in newly_derived:
                    index.add(self._index_fact_for_delta(fact))
                delta_facts = [self._index_fact_for_delta(fact) for fact in newly_derived]

        derived = tuple(
            self._derived_seen[key] for key in sorted(self._derived_seen, key=repr)
        )
        return DatalogResult(
            input_facts=self._input_facts,
            derived_facts=derived,
            iterations=iterations,
            fixed_point_reached=True,
            strata=self._strata,
            program_digest=self.program_digest,
        )

    def _fire_rule_through_binding(
        self,
        rule: RuleDecl,
        index: _FactIndex,
        delta_predicate: str,
        delta_binding: Mapping[str, str],
        accept: Any,
    ) -> None:
        """Fire one rule with the join focused through one delta-literal binding."""
        groups: list[list[tuple[DatalogFact, dict[str, str]]]] = []
        interval: tuple[int | None, int | None] | None = None
        for literal in rule.body:
            if literal.negated:
                continue
            matches = self._match_positive(literal, index, None, interval)
            if not matches:
                return
            if literal.atom.predicate == delta_predicate:
                filtered: list[tuple[DatalogFact, dict[str, str]]] = []
                for fact, binding in matches:
                    merged = dict(binding)
                    conflict = False
                    for var, value in delta_binding.items():
                        if var in merged and merged[var] != value:
                            conflict = True
                            break
                        merged[var] = value
                    if conflict:
                        continue
                    filtered.append((fact, merged))
                if not filtered:
                    return
                groups.append(filtered)
            else:
                groups.append(matches)
        if not groups:
            return
        self._join_and_check_negation(rule, groups, index, accept)

    def evaluate_naive(self) -> DatalogResult:
        """Reference naive fixed-point evaluator (equivalence checking, Step 19)."""
        index = _FactIndex()
        for fact in self._input_facts:
            index.add(fact)
        self._derived_seen = {}
        rules_by_head = self._rules_by_head()
        iterations = 0
        for stratum_predicates in self._strata:
            stratum_rules = [
                rule for predicate in stratum_predicates
                for rule in rules_by_head.get(predicate, [])
            ]
            if not stratum_rules:
                continue
            while True:
                iterations += 1
                if iterations > self._limits.max_iterations:
                    raise UnsafeRuleError("iteration limit exceeded in naive evaluation")
                newly_derived: list[DerivedFact] = []
                accept = self._accept_factory(newly_derived)
                for rule in stratum_rules:
                    self._fire_rule(rule, index, accept)
                self._check_limits()
                if not newly_derived:
                    break
                for fact in newly_derived:
                    index.add(self._index_fact_for_delta(fact))
        derived = tuple(
            self._derived_seen[key] for key in sorted(self._derived_seen, key=repr)
        )
        return DatalogResult(
            input_facts=self._input_facts,
            derived_facts=derived,
            iterations=iterations,
            fixed_point_reached=True,
            strata=self._strata,
            program_digest=self.program_digest,
        )

    def explain(
        self,
        result: DatalogResult,
        predicate: str,
        args: tuple[str, ...],
        scope: str = "default",
    ) -> tuple[Derivation, ...]:
        """Return every derivation for one derived fact (Step 20)."""
        for fact in result.derived_facts:
            if fact.predicate == predicate and fact.args == args and fact.scope == scope:
                return fact.derivations
        return ()


class DatalogCache:
    """Content-addressed cache for Datalog results (Section 11: stale caches).

    Keys include the engine version, the program digest (facts + rules), and
    domain profiles, so any rule or fact change yields a different key.
    Targeted invalidation by derived predicate is also provided.
    """

    def __init__(self, max_entries: int = 128) -> None:
        self._entries: dict[str, DatalogResult] = {}
        self._max_entries = max(1, max_entries)

    def key(self, engine: DatalogEngine, domain_profiles: Iterable[str] = ()) -> str:
        """Compute the cache key for one program."""
        return sha256_json({
            "engine_version": DATALOG_ENGINE_VERSION,
            "program_digest": engine.program_digest,
            "domains": sorted(set(domain_profiles)),
        })

    def get(self, key: str) -> DatalogResult | None:
        """Return a cached result or None."""
        return self._entries.get(key)

    def put(self, key: str, result: DatalogResult) -> None:
        """Store one result with bounded size."""
        if len(self._entries) >= self._max_entries:
            oldest = next(iter(self._entries))
            del self._entries[oldest]
        self._entries[key] = result

    def invalidate(self, predicates: Iterable[str]) -> int:
        """Drop cached results that derive any of the given predicates."""
        predicates = set(predicates)
        doomed = [
            key for key, result in self._entries.items()
            if any(fact.predicate in predicates for fact in result.derived_facts)
        ]
        for key in doomed:
            del self._entries[key]
        return len(doomed)

    def clear(self) -> None:
        """Drop every cached entry."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


__all__ = [
    "DATALOG_ENGINE_VERSION",
    "DatalogFact",
    "Derivation",
    "DerivedFact",
    "DatalogLimits",
    "DatalogResult",
    "DatalogEngine",
    "DatalogCache",
    "build_dependency_graph",
    "facts_from_ir",
    "rules_from_ir",
]
